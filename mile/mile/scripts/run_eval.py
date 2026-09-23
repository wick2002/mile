"""Entry point: full benchmark evaluation (stage S4).

Runs MILE (or the bare base with --open-loop) on the held-out 34 tasks
with calibrated parameters. Calibration tasks are reported separately.
"""
from __future__ import annotations

import argparse
import json
import pickle

import torch
import yaml

from mile.base.dm05_client import BasePolicyClient, DM05Config
from mile.controller.controller import ControllerConfig, MILEController
from mile.controller.evaluate import evaluate
from mile.wm.model import LatentWorldModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--banks", default="data/milestone_banks.pkl")
    ap.add_argument("--calibrated", default="configs/calibrated.json")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--open-loop", action="store_true", help="bare base policy")
    ap.add_argument("--out", default="results/eval.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_client = BasePolicyClient(DM05Config(server_url=cfg["base"]["server_url"]))

    controller_factory = None
    if not args.open_loop:
        with open(args.banks, "rb") as f:
            banks = pickle.load(f)
        with open(args.calibrated) as f:
            best = json.load(f)

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
        wm.eval().cuda()

        ccfg = ControllerConfig(
            num_candidates=cfg["controller"]["num_candidates"],
            lam=best.get("lam", cfg["controller"]["lam"]),
            eta=best.get("eta", cfg["controller"]["eta"]),
            tau_J=best.get("tau_J", cfg["controller"]["tau_J"]),
            radius=best.get("radius", cfg["controller"]["radius"]),
            patience=cfg["controller"]["patience"],
            max_pop=cfg["controller"]["max_pop"],
        )

        def controller_factory(task_id: str) -> MILEController:
            return MILEController(base_client, wm, banks[task_id], ccfg)

    def env_factory(task_id: str):
        # TODO(server): return mile.env.robodojo_env.RoboDojoEnv(task_id)
        raise NotImplementedError("wire to RoboDojo task registry")

    evaluate(env_factory, args.tasks, controller_factory, base_client,
             episodes_per_task=cfg["eval"]["episodes_per_task"], out_path=args.out)


if __name__ == "__main__":
    main()
