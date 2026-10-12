"""Evaluation loop: run MILE (or the bare base policy) on RoboDojo tasks.

Protocol (matches the official leaderboard):
  - 50 episodes per task: 25 standard + 25 randomized scenes
  - report sub-step-sensitive Score and Success Rate per dimension
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from ..controller.controller import MILEController


def run_episode(env, controller: MILEController | None, base_client,
                instruction: str, episode_seed: int,
                max_decisions: int = 40,
                exec_horizon: int | None = None,
                randomized: bool = False,
                record_stride: int = 1) -> dict:
    """One episode. controller=None means bare base policy (open-loop).

    exec_horizon: execute only the first N steps of each predicted chunk
    (RoboDojo official adapter uses 25 of 50). None = execute full chunk.

    randomized: randomized scene (official protocol = 25 standard + 25
    randomized per task). Applied HERE rather than by the caller because
    ``set_randomized`` must be called BEFORE ``reset`` — the task wrapper
    draws the scene layout inside reset(), so setting it afterwards would
    silently leave the current episode on the previous episode's setting.
    Keeping it inside run_episode makes that ordering impossible to get
    wrong for any new caller (e.g. the deployment episodes in run_rounds).
    """
    env.set_randomized(randomized)      # must precede reset(): scene drawn there
    obs = env.reset(seed=episode_seed)
    controller and controller.reset()
    base_client.reset_history()   # memory-enabled base: clear per-episode buffer

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

        # execute the chunk step by step (record steps for recovery data);
        # official protocol executes only the first exec_horizon steps.
        # record_stride: encoding a step costs a full VLM forward, so the
        # recovery buffer is subsampled (it only feeds value-head distillation).
        for i, a in enumerate(chunk[:exec_horizon]):
            obs = env.step(a)
            if controller is not None and i % max(1, record_stride) == 0:
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
    result["randomized"] = randomized
    return result


def stable_seed(task_id: str, ep: int) -> int:
    """Deterministic episode seed (stable across processes and runs).

    Do NOT use the builtin ``hash()`` here: ``str.__hash__`` is salted by
    PYTHONHASHSEED, so every process would draw different episode seeds and
    the 50-episode sweep (and every ablation built on it) would not be
    reproducible.
    """
    return zlib.crc32(f"{task_id}:{ep}".encode("utf-8")) % (2**31)


def evaluate(env_factory, task_ids: list[str], controller_factory,
             base_client, episodes_per_task: int, out_path: str,
             exec_horizon: int | None = None,
             record_stride: int = 1) -> dict:
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
                r = run_episode(env, controller, base_client,
                                instruction=env.task_instruction(),
                                episode_seed=stable_seed(task_id, ep),
                                exec_horizon=exec_horizon,
                                randomized=randomized,
                                record_stride=record_stride)
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
