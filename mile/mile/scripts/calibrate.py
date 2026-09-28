"""Conformal calibration of the acceptance threshold (stage S3, upgraded 2026-09-28).

Replaces the heuristic grid search over tau_J with split-conformal calibration:
  1. run the BARE BASE POLICY on the 8 calibration tasks, recording (J, d)
     pairs at every decision point;
  2. choose tau_hat so accepted segments terminate within radius r with
     probability >= 1 - alpha (finite-sample corrected);
  3. the 34 held-out tasks are used ONLY for coverage validation, never for
     fitting — the anti-overfitting protocol written into the paper.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from mile.controller.conformal import calibrate_tau, empirical_coverage


def collect_calibration_pairs(cfg: dict, tasks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run the bare base policy on calibration tasks; record (J, d) per chunk.

    For each executed chunk we need the verification score J (computed from
    the frozen WM + value head, which are already trained at this stage) and
    the realized deviation d to the target milestone.
    """
    # TODO(server): wire to RoboDojo env; reuse controller.act() with
    # tau_J = -inf (accept everything) so that d is observed for all chunks.
    raise NotImplementedError("collect (J, d) pairs via bare-base rollouts on the server")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="the 8 calibration task ids")
    ap.add_argument("--radius", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--heldout-tasks", nargs="*", default=[],
                    help="held-out tasks for coverage validation only")
    ap.add_argument("--out", default="configs/calibrated.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    J_cal, d_cal = collect_calibration_pairs(cfg, args.tasks)
    result = calibrate_tau(J_cal, d_cal, radius=args.radius, alpha=args.alpha)
    print(f"[calib] tau_hat={result.tau_hat:.4f} (n={result.n_calib}, "
          f"guarantee>={result.guarantee:.4f}, rejection={result.rejection_rate:.2%})")

    report = {
        "tau_J": result.tau_hat,
        "radius": result.radius,
        "alpha": result.alpha,
        "n_calib": result.n_calib,
        "guarantee": result.guarantee,
        "rejection_rate": result.rejection_rate,
    }

    # coverage validation on held-out tasks (NEVER used for fitting tau)
    if args.heldout_tasks:
        J_val, d_val = collect_calibration_pairs(cfg, args.heldout_tasks)
        cov = empirical_coverage(J_val, d_val, result.tau_hat, result.radius)
        report["heldout_coverage"] = cov
        print(f"[calib] held-out empirical coverage = {cov:.4f} "
              f"(target >= {1.0 - args.alpha:.2f})")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[calib] saved -> {args.out}")


if __name__ == "__main__":
    main()
