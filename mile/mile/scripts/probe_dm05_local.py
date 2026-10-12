"""Smoke test for in-process DM0.5 (run this FIRST on the server).

Run inside the OpenDM conda env, from the OpenDM repo root (so that the
``robodojo`` dataset registration in opendm/dataset/robodojo.py is importable):

    conda activate opendm
    python -m mile.scripts.probe_dm05_local --config configs/default.yaml

It answers, in order, the five things that decide whether you can start:
  1. does the checkpoint load and produce an action chunk of the right shape?
  2. does FeatureHook fire, and is the captured sequence >= num_patches?
  3. **candidate diversity** (W1 item 2) — the single highest-risk item: if the
     C chunks from different seeds are nearly identical, MILE's
     imagine-then-select mechanism has nothing to select;
  4. does a 20-frame history get accepted (OpenDM's default max is 5)?
  5. what is the real per-call latency (feeds the compute-scaling curve)?
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml


def fake_obs(cfg: dict, h: int = 224, w: int = 224) -> dict:
    b = cfg["base"]
    return {
        "images": {k: np.zeros((h, w, 3), dtype=np.uint8) for k in b["image_keys"]},
        "state": np.zeros(int(b["action_dim"]), dtype=np.float32),
        "instruction": "pick up the red block",
        "episode_seed": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--history-frames", type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    b = cfg["base"]

    from mile.base.dm05_local import build_local_policy

    t0 = time.perf_counter()
    policy = build_local_policy(cfg)
    print(f"[1] loaded in {time.perf_counter() - t0:.1f}s "
          f"| device={getattr(policy.ic, 'device', '?')} "
          f"| robot_type={policy.ic.default_robot_type!r} "
          f"| speed={policy.ic.default_speed!r} "
          f"| is_history={policy.ic.is_history} "
          f"| max_history_images={policy.ic.max_history_images}")

    obs = fake_obs(cfg)
    policy.reset_history()

    t0 = time.perf_counter()
    chunk = policy.sample(obs, seed=0)
    dt = time.perf_counter() - t0
    print(f"[2] sample -> shape={chunk.shape} dtype={chunk.dtype} "
          f"wall={dt * 1000:.0f}ms model_latency={policy.last_latency_ms}")
    expected = (int(b["chunk_len"]), int(b["action_dim"]))
    if tuple(chunk.shape) != expected:
        print(f"    !! expected {expected} — check base.chunk_len / "
              f"the checkpoint's real chunk_size (RoboDojo may be 25, not 50)")
    print(f"    action range: min={chunk.min():.4f} max={chunk.max():.4f} "
          f"| all-zero={bool(np.all(chunk == 0))}")

    hidden = policy.current_patch_tokens()
    P = int(b["num_patches"])
    print(f"[3] FeatureHook -> shape={hidden.shape} "
          f"| num_patches={P} | ok={hidden.shape[0] >= P}")
    if hidden.shape[0] < P:
        print("    !! captured sequence shorter than num_patches; check "
              "base.feature_layer / base.num_patches")

    cands = policy.sample_candidates(obs, n=args.candidates)
    stack = np.stack(cands)
    flat = stack.reshape(len(cands), -1)
    dists = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            dists.append(float(np.linalg.norm(flat[i] - flat[j])))
    d = np.asarray(dists)
    print(f"[4] candidate diversity (C={len(cands)}): "
          f"pairwise L2 mean={d.mean():.4f} min={d.min():.4f} max={d.max():.4f} "
          f"| std-within-candidate={stack.std():.4f}")
    rel = d.mean() / max(1e-8, float(np.abs(flat).mean()))
    print(f"    relative spread = {rel:.4f}")
    if rel < 1e-3:
        print("    !! candidates are essentially identical: flow-matching seed "
              "diversity is flat. Fall back to perturbation-based candidates "
              "(history conditioning / action-space noise / prompt paraphrase), "
              "otherwise the whole verification stage is vacuous.")

    for _ in range(args.history_frames):
        policy.sample(obs, seed=1)
    print(f"[5] history accepted: buffer={len(policy._history)} "
          f"/ max_history_images={policy.ic.max_history_images}")

    lat = []
    for s in range(5):
        t0 = time.perf_counter()
        policy.sample(obs, seed=s)
        lat.append((time.perf_counter() - t0) * 1000)
    print(f"[6] latency over 5 calls: mean={np.mean(lat):.0f}ms "
          f"min={np.min(lat):.0f}ms max={np.max(lat):.0f}ms "
          f"(C=4 => ~{np.mean(lat) * 4 / 1000:.2f}s per decision)")


if __name__ == "__main__":
    main()
