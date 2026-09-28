"""Evaluation loop: run MILE (or the bare base policy) on RoboDojo tasks.

Protocol (matches the official leaderboard):
  - 50 episodes per task: 25 standard + 25 randomized scenes
  - report sub-step-sensitive Score and Success Rate per dimension
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..controller.controller import MILEController


def run_episode(env, controller: MILEController | None, base_client,
                instruction: str, episode_seed: int,
                max_decisions: int = 40) -> dict:
    """One episode. controller=None means bare base policy (open-loop)."""
    obs = env.reset(seed=episode_seed)
    controller and controller.reset()

    events: list[dict] = []
    for _ in range(max_decisions):
        obs["instruction"] = instruction
        obs["episode_seed"] = episode_seed

        if controller is None:
            chunk = base_client.sample(obs, seed=None)
            info = {"mode": "open_loop"}
        else:
            chunk, info = controller.act(obs)

        if info.get("action") == "rollback":
            rb = controller.rollback()
            events.append(rb)
            if rb["event"] == "abort":
                break
            env.restore_state(rb["env_snapshot"])
            continue

        # execute the chunk step by step (record steps for recovery data)
        for a in chunk:
            obs = env.step(a)
            if controller is not None:
                controller.record_step(obs["patch_tokens"], a)

        if controller is not None:
            realized = obs["patch_tokens"]  # refresh from new obs in practice
            ev = controller.verify(realized, env.save_state())
            events.append(ev)
            if ev["event"] == "rollback":
                if "env_snapshot" in ev:
                    env.restore_state(ev["env_snapshot"])
                else:
                    break
            if ev["event"] == "done":
                break

    result = env.get_episode_result()   # official sub-step scoring
    result["events"] = events
    return result


def evaluate(env_factory, task_ids: list[str], controller_factory,
             base_client, episodes_per_task: int, out_path: str) -> dict:
    """Full benchmark sweep. Writes per-task results to out_path (jsonl)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    with out.open("w") as fout:
        for task_id in task_ids:
            env = env_factory(task_id)
            controller = controller_factory(task_id) if controller_factory else None
            scores, successes = [], []
            for ep in range(episodes_per_task):
                randomized = ep >= episodes_per_task // 2
                env.set_randomized(randomized)
                r = run_episode(env, controller, base_client,
                                instruction=env.task_instruction(),
                                episode_seed=hash((task_id, ep)) % (2**31))
                scores.append(r["score"])
                successes.append(float(r["success"]))
                fout.write(json.dumps({"task": task_id, "ep": ep,
                                       "randomized": randomized,
                                       "score": r["score"],
                                       "success": r["success"]}) + "\n")
            summary[task_id] = {
                "score": float(np.mean(scores)),
                "success": float(np.mean(successes)),
                "score_ci95": float(1.96 * np.std(scores) / np.sqrt(len(scores))),
            }
            print(f"[eval] {task_id}: score={summary[task_id]['score']:.2f} "
                  f"sr={summary[task_id]['success']:.3f}")
    return summary
