"""Training loop for the latent world model + progress value head.

Single-GPU reference implementation (1 x RTX 4090, ~1-2 days).
Run via: python -m mile.scripts.train_wm --config configs/default.yaml
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .data import make_dataloader
from .model import LatentWorldModel, sample_patch_mask, wm_loss


def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def train_wm(cfg: dict) -> Path:
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wm_cfg = cfg["world_model"]
    tr_cfg = cfg["train"]
    base_cfg = cfg["base"]

    model = LatentWorldModel(
        num_patches=base_cfg["num_patches"],
        vlm_dim=base_cfg["vlm_dim"],
        d_model=wm_cfg["d_model"],
        n_layers=wm_cfg["n_layers"],
        n_heads=wm_cfg["n_heads"],
        d_ff=wm_cfg["d_ff"],
        dropout=wm_cfg["dropout"],
        action_dim=base_cfg["action_dim"],
        chunk_len=base_cfg["chunk_len"],
        action_tokens=wm_cfg["action_tokens"],
        text_tokens=wm_cfg["text_tokens"],
        n_ensemble=wm_cfg["n_ensemble"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train_wm] params: {n_params:.1f}M")

    loader = make_dataloader(
        tr_cfg["data_dir"], chunk_len=base_cfg["chunk_len"],
        batch_size=tr_cfg["batch_size"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    out_dir = Path(tr_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    w = wm_cfg["loss_weights"]
    use_bf16 = tr_cfg["bf16"] and device == "cuda"

    step = 0
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            lr = cosine_lr(step, tr_cfg["lr"], tr_cfg["warmup_steps"], tr_cfg["max_steps"])
            for g in opt.param_groups:
                g["lr"] = lr

            patch = batch["patch_tokens"].to(device, non_blocking=True)
            act = batch["action_chunk"].to(device, non_blocking=True)
            txt = batch["text_emb"].to(device, non_blocking=True)
            y = batch["target_latents"].to(device, non_blocking=True)
            v = batch["target_value"].to(device, non_blocking=True)
            mask = sample_patch_mask(patch.shape[0], patch.shape[1],
                                     wm_cfg["mask_ratio"], device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(patch, act, txt, patch_mask=mask)
                loss, parts = wm_loss(
                    out, y, v,
                    w_latent=w["latent"], w_cosine=w["cosine"], w_value=w["value"],
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % tr_cfg["log_every"] == 0:
                fps = (step + 1) / (time.time() - t0)
                print(f"[step {step}] lr={lr:.2e} loss={parts['total']:.4f} "
                      f"(l1={parts['smooth_l1']:.4f} cos={parts['cosine']:.4f} "
                      f"val={parts['value']:.4f}) {fps:.1f} it/s")

            if step > 0 and step % tr_cfg["ckpt_every"] == 0:
                ckpt = out_dir / f"wm_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, ckpt)
                print(f"[train_wm] saved {ckpt}")

            step += 1
            if step >= tr_cfg["max_steps"]:
                done = True
                break

    final = out_dir / "wm_final.pt"
    torch.save({"model": model.state_dict(), "step": step, "cfg": cfg}, final)
    print(f"[train_wm] done -> {final}")
    return final
