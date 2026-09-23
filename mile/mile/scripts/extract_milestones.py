"""Extract milestone banks for all tasks (stage S1).

v1: use official RoboDojo sub-step boundaries (--use-official).
v2: automatic extraction from cues + calibration against official boundaries.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

from mile.extractor.cues import consensus, contact_cue, kinematic_cue, visual_cue
from mile.extractor.extractor import build_bank, select_topk


def extract_auto(traj: dict, cfg: dict, k: int) -> list[int]:
    m = cfg["milestones"]
    kin = kinematic_cue(traj["gripper"], traj["ee_vel"])
    con = contact_cue(traj["contact"])
    vis = visual_cue(traj["latents"], traj.get("attn_mask"))
    s = consensus(kin, con, vis, m["alpha"], m["beta"], m["gamma"])
    return select_topk(s, k=k, min_separation=m["min_separation"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--demo-dir", required=True, help="dir with per-task demo .npz files")
    ap.add_argument("--out", default="data/milestone_banks.pkl")
    ap.add_argument("--use-official", action="store_true",
                    help="use official sub-step boundaries instead of auto extraction")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    demo_dir = Path(args.demo_dir)
    banks = {}

    for task_dir in sorted(demo_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        files = sorted(task_dir.glob("*.npz"))
        latents, milestone_lists, snapshots = [], [], []
        for f in files:
            with np.load(f) as z:
                latents.append(z["latents"])
                if args.use_official:
                    milestone_lists.append(z["official_substeps"].tolist())
                else:
                    traj = {k2: z[k2] for k2 in z.files}
                    k = len(z["official_substeps"]) if "official_substeps" in z.files else 5
                    milestone_lists.append(extract_auto(traj, cfg, k))
                snapshots.append(z["snapshots"].item() if "snapshots" in z.files else None)

        stage_texts = [f"stage {i + 1}" for i in range(max(map(len, milestone_lists)))]
        banks[task_dir.name] = build_bank(
            task_dir.name, latents, milestone_lists, stage_texts,
            demo_snapshots=snapshots if any(snapshots) else None,
        )
        print(f"[extract] {task_dir.name}: {banks[task_dir.name].k} milestones")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(banks, f)
    print(f"[extract] saved {len(banks)} banks -> {out}")


if __name__ == "__main__":
    main()
