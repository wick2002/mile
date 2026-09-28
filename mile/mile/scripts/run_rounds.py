"""Self-improvement rounds (paper §4.6 / Figure 3, stage W10).

Loop over rounds R0..R{n}: deploy the current system to collect recovery
segments, distill the buffer into the value head (optionally + LoRA on the
base's action expert), then re-evaluate. Produces the round-success curve.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import yaml

from mile.controller.recovery import RecoveryBuffer, distill_value_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out-dir", default="runs/self_improve")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    si = cfg["self_improve"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load world model ----
    from mile.wm.model import LatentWorldModel
    wm = LatentWorldModel(
        num_patches=cfg["base"]["num_patches"], vlm_dim=cfg["base"]["vlm_dim"],
        d_model=cfg["world_model"]["d_model"],
        n_layers=cfg["world_model"]["n_layers"],
        n_heads=cfg["world_model"]["n_heads"],
        d_ff=cfg["world_model"]["d_ff"],
        action_dim=cfg["base"]["action_dim"],
        chunk_len=cfg["base"]["chunk_len"],
        action_tokens=cfg["world_model"]["action_tokens"],
        text_tokens=cfg["world_model"]["text_tokens"],
        n_ensemble=cfg["world_model"]["n_ensemble"],
    )
    wm.load_state_dict(torch.load(args.wm_ckpt, map_location="cpu")["model"])
    wm.cuda().eval()

    results = []
    for rnd in range(args.rounds + 1):
        # ---- evaluate current system (R0 = before any distillation) ----
        # TODO(server): call run_eval on held-out tasks, parse mean score
        score = None  # = run_eval_and_get_score(wm, ...)
        results.append({"round": rnd, "score": score})
        print(f"[rounds] R{rnd}: score={score}")

        if rnd == args.rounds:
            break

        # ---- deploy & collect recovery segments ----
        buffer = RecoveryBuffer()
        # TODO(server): run si["episodes_per_round"] deployment episodes with
        # controller(recovery_buffer=buffer); reuse scripts/run_eval.py with
        # --recovery-buffer runs/self_improve/buffer_r{rnd}.pkl

        # ---- distill into value head (base and WM trunk frozen) ----
        stats = distill_value_head(
            wm, buffer,
            steps=si["distill_steps"],
            batch_size=si["distill_batch"],
            lr=si["distill_lr"],
        )
        print(f"[rounds] R{rnd} distilled: {stats}")

        # optional: LoRA adapter on the base's action expert (variant switch)
        if si["use_lora"]:
            raise NotImplementedError("LoRA distillation variant (server-side)")

        ckpt = out_dir / f"wm_round{rnd + 1}.pt"
        torch.save({"model": wm.state_dict(), "round": rnd + 1}, ckpt)

    with (out_dir / "rounds.json").open("w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"[rounds] done -> {out_dir}")


if __name__ == "__main__":
    main()
