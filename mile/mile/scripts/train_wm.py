"""Entry point: train the latent world model (stage S2)."""
from __future__ import annotations

import argparse

import yaml

from mile.wm.train import train_wm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train_wm(cfg)


if __name__ == "__main__":
    main()
