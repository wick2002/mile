"""JEPA-style latent world model for MILE.

Predicts the VLM feature-space state after executing a candidate action
chunk, plus a normalized steps-to-completion value. Uncertainty is
estimated by an ensemble of prediction heads.

Design (matches tech design doc §3.0):
  - input : 256 VLM patch tokens (2560d -> 1024d projection)
          + 8 action tokens (50-step chunk compressed by Conv1d k=7 s=6)
          + 4 text tokens (frozen stage-descriptor embedding)
          + 1 value-query token
  - trunk : 4-layer pre-LN Transformer, d=1024, 16 heads, FFN 4096
  - heads : E=3 ensemble latent-prediction heads (linear 1024 -> 2560)
          + 1 value head (MLP -> scalar in [0, 1])
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WMOutput:
    pred_latents: torch.Tensor   # [E, B, P, vlm_dim] ensemble predictions of z_{t+H}
    value: torch.Tensor          # [B] normalized steps-to-completion in [0, 1]
    uncertainty: torch.Tensor    # [B, P] ensemble disagreement (variance)


class LatentWorldModel(nn.Module):
    def __init__(
        self,
        num_patches: int = 256,
        vlm_dim: int = 2560,
        d_model: int = 1024,
        n_layers: int = 4,
        n_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        action_dim: int = 14,
        chunk_len: int = 50,
        action_tokens: int = 8,
        text_dim: int = 2560,
        text_tokens: int = 4,
        n_ensemble: int = 3,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.vlm_dim = vlm_dim
        self.action_tokens = action_tokens
        self.text_tokens = text_tokens
        self.n_ensemble = n_ensemble

        # ---- input projections ----
        self.patch_proj = nn.Linear(vlm_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        # compress a [B, chunk_len, action_dim] chunk into action_tokens tokens
        k = chunk_len - (action_tokens - 1) * 6  # stride 6 -> kernel 7 for 50->8
        self.action_compress = nn.Conv1d(action_dim, d_model, kernel_size=k, stride=6)
        self.value_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        seq_len = num_patches + action_tokens + text_tokens + 1
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ---- trunk ----
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # ---- heads ----
        self.pred_heads = nn.ModuleList(
            [nn.Linear(d_model, vlm_dim) for _ in range(n_ensemble)]
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self._patch_slice = slice(0, num_patches)

    def forward(
        self,
        patch_tokens: torch.Tensor,     # [B, P, vlm_dim]
        action_chunk: torch.Tensor,     # [B, chunk_len, action_dim]
        text_emb: torch.Tensor,         # [B, text_tokens, text_dim]
        patch_mask: torch.Tensor | None = None,  # [B, P] bool, True = mask out
    ) -> WMOutput:
        B = patch_tokens.shape[0]

        x_patch = self.patch_proj(patch_tokens)
        if patch_mask is not None:
            m = patch_mask.unsqueeze(-1).to(x_patch.dtype)
            x_patch = x_patch * (1.0 - m) + self.mask_token * m

        a = self.action_compress(action_chunk.transpose(1, 2)).transpose(1, 2)  # [B, A, d]
        t = self.text_proj(text_emb)                                             # [B, T, d]
        q = self.value_query.expand(B, -1, -1)                                   # [B, 1, d]

        x = torch.cat([x_patch, a, t, q], dim=1) + self.pos_embed
        h = self.final_norm(self.trunk(x))

        h_patch = h[:, self._patch_slice, :]                      # [B, P, d]
        preds = torch.stack([head(h_patch) for head in self.pred_heads], dim=0)  # [E, B, P, vlm]
        value = torch.sigmoid(self.value_head(h[:, -1, :])).squeeze(-1)          # [B]
        uncertainty = preds.var(dim=0)                            # [B, P, vlm]

        return WMOutput(pred_latents=preds, value=value, uncertainty=uncertainty)

    @torch.no_grad()
    def forward_value(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Value-only path for recovery-data distillation (paper §4.6).

        Runs the trunk with zeroed action/text conditioning so the value
        head can be queried on raw latents without a candidate chunk.
        Callers that need gradients for the value head should use
        `value_path` instead.
        """
        return self.value_path(patch_tokens)

    def value_path(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Differentiable value-only forward (used by recovery distillation)."""
        B = patch_tokens.shape[0]
        d = self.value_query.shape[-1]
        x = self.patch_proj(patch_tokens)
        zeros_a = torch.zeros(B, self.action_tokens, d, device=x.device, dtype=x.dtype)
        zeros_t = torch.zeros(B, self.text_tokens, d, device=x.device, dtype=x.dtype)
        q = self.value_query.expand(B, -1, -1)
        seq = torch.cat([x, zeros_a, zeros_t, q], dim=1) + self.pos_embed
        h = self.final_norm(self.trunk(seq))
        return torch.sigmoid(self.value_head(h[:, -1, :])).squeeze(-1)


def wm_loss(
    out: WMOutput,
    target_latents: torch.Tensor,   # [B, P, vlm_dim] ground-truth z_{t+H}
    target_value: torch.Tensor,     # [B]
    w_latent: float = 1.0,
    w_cosine: float = 0.5,
    w_value: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint prediction + value loss (design doc §3/§4)."""
    pred_mean = out.pred_latents.mean(dim=0)                      # [B, P, vlm]
    l_smooth = F.smooth_l1_loss(pred_mean, target_latents)
    l_cos = 1.0 - F.cosine_similarity(
        pred_mean.flatten(1), target_latents.flatten(1), dim=-1
    ).mean()
    l_value = F.huber_loss(out.value, target_value)
    total = w_latent * l_smooth + w_cosine * l_cos + w_value * l_value
    return total, {
        "smooth_l1": l_smooth.item(),
        "cosine": l_cos.item(),
        "value": l_value.item(),
        "total": total.item(),
    }


def sample_patch_mask(batch: int, num_patches: int, ratio: float, device) -> torch.Tensor:
    """Random 20% patch masking (anti-hallucination, cf. GigaBrain)."""
    mask = torch.rand(batch, num_patches, device=device) < ratio
    return mask
