"""MILE execution controller (Algorithm 1 in the paper).

Wraps a frozen VLA base policy with:
  candidate sampling -> imagination -> verification scoring ->
  execute-or-reject -> milestone verification -> rollback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..extractor.extractor import MilestoneBank
from .scoring import milestone_reached, pick_best, score_candidates
from .stack import VerifiedEntry, VerifiedStack


@dataclass
class ControllerConfig:
    num_candidates: int = 4
    lam: float = 1.0
    eta: float = 0.5
    tau_J: float = 0.65
    radius: float = 0.35
    patience: int = 2
    max_pop: int = 2
    device: str = "cuda"


class MILEController:
    """Verify-then-act controller with safe-anchor rollback."""

    def __init__(
        self,
        base_client,                 # mile.base.dm05_client.BasePolicyClient
        world_model,                 # LatentWorldModel (eval mode)
        bank: MilestoneBank,         # milestone bank for the current task
        cfg: ControllerConfig,
    ) -> None:
        self.base = base_client
        self.wm = world_model
        self.bank = bank
        self.cfg = cfg
        self.stack = VerifiedStack(max_pop=cfg.max_pop)
        self.deviation_streak = 0
        self.one_way_ranks: set[int] = set()   # irreversible stages: verify-only

    # ----------------------------------------------------------- episode
    def reset(self) -> None:
        self.stack.reset()
        self.deviation_streak = 0

    # ------------------------------------------------------------- step
    @torch.no_grad()
    def act(self, obs: dict) -> tuple[np.ndarray, dict]:
        """One decision point. Returns (action_chunk, info).

        obs must contain whatever base_client needs (images, state, instruction)
        plus precomputed 'patch_tokens' [P, D] and 'text_emb' for the next stage.
        """
        cfg = self.cfg
        device = cfg.device

        k_next = self.stack.next_rank
        if k_next >= self.bank.k:
            # all milestones verified; run base policy open-loop to the end
            chunk = self.base.sample(obs, seed=None)
            return chunk, {"mode": "tail", "verified": len(self.stack)}

        target = torch.from_numpy(self.bank.latents[k_next]).float().to(device)

        # 1. candidate generation: C chunks with different noise seeds
        candidates = self.base.sample_candidates(obs, n=cfg.num_candidates)

        # 2. imagination: batch WM rollout for all candidates
        patch = torch.from_numpy(obs["patch_tokens"]).float().unsqueeze(0).to(device)
        text = torch.from_numpy(obs["text_emb"]).float().unsqueeze(0).to(device)
        acts = torch.from_numpy(np.stack(candidates)).float().to(device)
        B = acts.shape[0]
        out = self.wm(
            patch.expand(B, -1, -1),
            acts,
            text.expand(B, -1, -1),
        )
        pred_mean = out.pred_latents.mean(dim=0)                    # [B, P, D]
        unc = out.uncertainty.mean(dim=(1, 2))                      # [B]

        # 3. verification scoring
        scores = score_candidates(pred_mean, out.value, unc, target,
                                  lam=cfg.lam, eta=cfg.eta)

        # 4. accept or reject
        best, accepted = pick_best(scores, cfg.tau_J)
        info = {
            "mode": "execute" if accepted else "reject",
            "scores": scores.detach().cpu().numpy().tolist(),
            "k_next": k_next,
        }
        if not accepted:
            info["action"] = "rollback"
            return np.zeros_like(candidates[0]), info
        return candidates[best], info

    # ------------------------------------------------- post-execution
    def verify(self, realized_latent: np.ndarray, env_snapshot: dict) -> dict:
        """Call after the executed chunk finishes. Returns an event dict."""
        k_next = self.stack.next_rank
        if k_next >= self.bank.k:
            return {"event": "done"}

        target = self.bank.latents[k_next]
        reached, dist = milestone_reached(realized_latent, target, self.cfg.radius)

        if reached:
            self.stack.push(VerifiedEntry(
                rank=k_next, latent=realized_latent, env_snapshot=env_snapshot,
            ))
            self.deviation_streak = 0
            return {"event": "verified", "rank": k_next, "dist": dist}

        self.deviation_streak += 1
        if self.deviation_streak >= self.cfg.patience:
            return {"event": "rollback", "dist": dist, **self.rollback()}
        return {"event": "deviated", "dist": dist, "streak": self.deviation_streak}

    # --------------------------------------------------------- rollback
    def rollback(self) -> dict:
        """Retreat to the most recent verified milestone (stack top)."""
        if k_irreversible := self._current_stage_is_one_way():
            return {"event": "abort", "reason": "one-way stage"}

        entry = self.stack.rollback_target()
        if entry is None:
            return {"event": "abort", "reason": "no verified anchor"}
        self.deviation_streak = 0
        return {
            "event": "rollback_to",
            "rank": entry.rank,
            "env_snapshot": entry.env_snapshot,
        }

    def _current_stage_is_one_way(self) -> bool:
        return self.stack.next_rank in self.one_way_ranks
