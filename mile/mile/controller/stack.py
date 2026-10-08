"""Verified-milestone stack with environment snapshots (design doc §7).

Each entry is a *verified* milestone: the policy provably reached it, and
the full environment state at that moment is stored for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VerifiedEntry:
    rank: int                       # milestone rank in the bank (0-based)
    latent: np.ndarray              # realized latent at verification time
    env_snapshot: dict[str, Any] = field(default_factory=dict)  # sim state


class VerifiedStack:
    def __init__(self, max_pop: int = 2) -> None:
        self._entries: list[VerifiedEntry] = []
        self.max_pop = max_pop
        self._pops_this_episode = 0

    # ---- basic stack ops ----
    def push(self, entry: VerifiedEntry) -> None:
        self._entries.append(entry)

    def top(self) -> VerifiedEntry | None:
        return self._entries[-1] if self._entries else None

    @property
    def next_rank(self) -> int:
        """Rank of the milestone the controller should currently target."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ---- rollback ----
    def rollback_target(self) -> VerifiedEntry | None:
        """Entry to restore on rollback. Returns None if the episode should abort.

        NOTE: the caller (``MILEController.rollback``) MUST call ``pop()``
        after consuming this entry. Otherwise ``_pops_this_episode`` never
        advances, ``exhausted`` stays False forever, and every rollback in an
        episode returns the same anchor — the episode then spins until
        ``max_decisions`` with zero progress and a corrupted rollback count.
        """
        if self._pops_this_episode >= self.max_pop or not self._entries:
            return None
        # retreat to stack top; on repeated failures the caller pops first
        return self._entries[-1]

    def pop(self) -> VerifiedEntry | None:
        if not self._entries:
            return None
        self._pops_this_episode += 1
        return self._entries.pop()

    @property
    def pops_this_episode(self) -> int:
        return self._pops_this_episode

    def reset_pops(self) -> None:
        """Clear the *consecutive* roll-back budget (called after progress).

        ``max_pop`` bounds CONSECUTIVE retreats, not the per-episode total:
        once a milestone is verified again the retreat budget is restored.
        """
        self._pops_this_episode = 0

    @property
    def exhausted(self) -> bool:
        return self._pops_this_episode >= self.max_pop

    # ---- per-episode lifecycle ----
    def reset(self) -> None:
        self._entries.clear()
        self._pops_this_episode = 0
