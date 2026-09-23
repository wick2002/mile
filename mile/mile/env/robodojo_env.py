"""RoboDojo (Isaac Lab) environment wrapper.

Provides the two capabilities MILE needs beyond plain stepping:
  save_state()    -> full physics snapshot at a verified milestone
  restore_state() -> rollback target restoration (sim-side rollback is free)

The wrapper is deliberately thin: adapt `make_env` to the official
RoboDojo/XPolicyLab task registry on your server.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class RoboDojoEnv:
    def __init__(self, task_id: str, render: bool = False) -> None:
        self.task_id = task_id
        self.env = self._make_env(task_id, render)

    @staticmethod
    def _make_env(task_id: str, render: bool):
        """Instantiate the official RoboDojo task env (server-side)."""
        # from robodojo.tasks import make_task_env  # official registry
        # return make_task_env(task_id, headless=not render)
        raise NotImplementedError(
            "instantiate the official RoboDojo/Isaac Lab env here; "
            "see https://github.com/RoboDojo-Benchmark/RoboDojo"
        )

    # ---------------------------------------------------------- gym API
    def reset(self, seed: int | None = None) -> dict:
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray):
        return self.env.step(action)

    # ------------------------------------------------- snapshot support
    def save_state(self) -> dict[str, Any]:
        """Full physics state: joint positions/velocities + object poses."""
        state = {
            "qpos": np.asarray(self.env.get_joint_positions()).copy(),
            "qvel": np.asarray(self.env.get_joint_velocities()).copy(),
            "objects": self.env.get_object_poses(),   # {name: (pos, quat)}
        }
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Rollback: restore a saved physics state (sim-only, zero cost)."""
        self.env.set_joint_positions(state["qpos"])
        self.env.set_joint_velocities(state["qvel"])
        self.env.set_object_poses(state["objects"])
        # let physics settle for a few steps so contacts are consistent
        for _ in range(5):
            self.env.step(np.zeros_like(state["qpos"]))

    # ------------------------------------------------------ convenience
    def current_observation(self) -> dict:
        return self.env.get_observation()

    def contact_flags(self) -> np.ndarray:
        """Per-finger contact flags from privileged sim info (training-time)."""
        return np.asarray(self.env.get_contact_forces() > 1e-3, dtype=np.float32)
