"""Verification scoring for candidate action chunks (design doc §5).

J(c) = cos(pred_c, milestone) + lambda * value(pred_c) - eta * uncertainty(pred_c)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_to_milestone(pred_latents: torch.Tensor, milestone_latent: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between predicted latents and target milestone latent.

    Args:
        pred_latents: [B, P, D] predicted z_{t+H} (ensemble-mean).
        milestone_latent: [P, D] target milestone latent.
    Returns:
        [B] cosine similarities.
    """
    a = pred_latents.flatten(1)
    b = milestone_latent.flatten().unsqueeze(0).expand_as(a)
    return F.cosine_similarity(a, b, dim=-1)


def score_candidates(
    pred_latents: torch.Tensor,   # [B, P, D] ensemble-mean predictions
    values: torch.Tensor,         # [B] predicted progress values
    uncertainties: torch.Tensor,  # [B] mean ensemble disagreement
    milestone_latent: torch.Tensor,  # [P, D]
    lam: float = 1.0,
    eta: float = 0.5,
) -> torch.Tensor:
    """Compute J(c) for each candidate. Higher is better."""
    align = cosine_to_milestone(pred_latents, milestone_latent)
    return align + lam * values - eta * uncertainties


def pick_best(scores: torch.Tensor, tau_J: float) -> tuple[int, bool]:
    """Return (best_index, accepted). Reject all if best < tau_J."""
    best = int(torch.argmax(scores).item())
    return best, bool(scores[best].item() > tau_J)


def milestone_reached(realized_latent: np.ndarray, milestone_latent: np.ndarray,
                      radius: float) -> tuple[bool, float]:
    """Check whether the realized observation is within radius r of the milestone."""
    d = float(np.linalg.norm(realized_latent.flatten() - milestone_latent.flatten()))
    return d < radius, d
