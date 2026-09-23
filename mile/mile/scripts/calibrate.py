"""Threshold calibration on the 8-task calibration set (stage S3).

Grid-searches (radius r, tau_J, lam, eta) to maximize the official Score
on calibration tasks ONLY. The other 34 tasks are never touched here —
this is the anti-overfitting protocol written into the paper.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import yaml


def grid_search(cfg: dict, calibration_tasks: list[str], eval_fn) -> dict:
    grid = {
        "radius": [0.2, 0.35, 0.5],
        "tau_J": [0.5, 0.65, 0.8],
        "lam": [0.5, 1.0, 2.0],
        "eta": [0.0, 0.5, 1.0],
    }
    best = {"mean_score": -np.inf}
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        scores = [eval_fn(task, params) for task in calibration_tasks]
        mean = float(np.mean(scores))
        print(f"[calib] {params} -> mean score {mean:.2f}")
        if mean > best["mean_score"]:
            best = {**params, "mean_score": mean}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="the 8 calibration task ids")
    ap.add_argument("--out", default="configs/calibrated.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    def eval_fn(task_id: str, params: dict) -> float:
        # TODO(server): build env + controller with `params`, run 25 episodes,
        # return mean official score. Reuse mile.controller.evaluate.run_episode.
        raise NotImplementedError("wire to the RoboDojo env on the server")

    best = grid_search(cfg, args.tasks, eval_fn)
    with open(args.out, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[calib] best params -> {args.out}: {best}")


if __name__ == "__main__":
    main()
