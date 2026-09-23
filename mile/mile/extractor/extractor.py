"""Milestone extraction: top-K selection, DTW alignment across demos,
and milestone-bank construction (design doc §2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Milestone:
    index: int                     # frame index in the source trajectory
    latent: np.ndarray             # [P, D] patch features at that frame
    state_snapshot: dict = field(default_factory=dict)  # env state for rollback
    stage_text: str = ""           # e.g. "after picking the red cup"


@dataclass
class MilestoneBank:
    """Per-task milestone sequence with reference latents and snapshots."""
    task_id: str
    milestones: list[Milestone]

    @property
    def latents(self) -> np.ndarray:       # [K, P, D]
        return np.stack([m.latent for m in self.milestones])

    @property
    def k(self) -> int:
        return len(self.milestones)


def select_topk(scores: np.ndarray, k: int, min_separation: int) -> list[int]:
    """Greedy top-K with minimum temporal separation."""
    order = np.argsort(-scores)
    chosen: list[int] = []
    for idx in order:
        if scores[idx] <= 0:
            break
        if all(abs(int(idx) - c) >= min_separation for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= k:
            break
    return sorted(chosen)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard DTW on two latent sequences [Ta, D] x [Tb, D]."""
    Ta, Tb = len(a), len(b)
    cost = np.full((Ta + 1, Tb + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            d = np.linalg.norm(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[Ta, Tb])


def align_milestones(demo_milestones: list[list[int]]) -> list[int]:
    """Pick a representative milestone index set across demos (median per rank).

    Simple robust strategy: sort each demo's milestones, take the element-wise
    median across demos (requires equal K; use the modal K).
    """
    from collections import Counter

    ks = Counter(len(m) for m in demo_milestones)
    k_modal = ks.most_common(1)[0][0]
    subset = [sorted(m) for m in demo_milestones if len(m) == k_modal]
    arr = np.array(subset)                    # [N, K]
    return [int(x) for x in np.median(arr, axis=0)]


def build_bank(task_id: str,
               demo_latents: list[np.ndarray],
               demo_milestones: list[list[int]],
               stage_texts: list[str],
               demo_snapshots: list[list[dict]] | None = None) -> MilestoneBank:
    """Build a per-task milestone bank from aligned demonstrations.

    For each aligned milestone rank, the representative latent is the median
    (per-patch median) across demos.
    """
    aligned = align_milestones(demo_milestones)
    milestones: list[Milestone] = []
    for rank, _ in enumerate(aligned):
        lats = []
        snap: dict = {}
        for di, mlist in enumerate(demo_milestones):
            if len(mlist) != len(aligned):
                continue
            frame = sorted(mlist)[rank]
            lats.append(demo_latents[di][frame])
            if demo_snapshots is not None and not snap:
                snap = demo_snapshots[di][frame]
        rep_latent = np.median(np.stack(lats), axis=0)
        milestones.append(Milestone(
            index=aligned[rank],
            latent=rep_latent,
            state_snapshot=snap,
            stage_text=stage_texts[rank] if rank < len(stage_texts) else f"stage {rank + 1}",
        ))
    return MilestoneBank(task_id=task_id, milestones=milestones)
