<div align="center">

# MILE: Milestone-Grounded Closed-Loop Imagination for Long-Horizon VLA Policies

**Think, Verify, Retreat — turning any chunked VLA into a closed-loop long-horizon agent**

[![Paper](https://img.shields.io/badge/Paper-ICML%202027%20(submitted)-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-RoboDojo-green)](https://robodojo-benchmark.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()

*Anonymous submission — code released for review. Identifying links intentionally omitted.*

</div>

---

## Overview

Modern Vision-Language-Action (VLA) policies execute **open-loop**: they map observations to action chunks reactively, with no mechanism to anticipate consequences, detect drift, or recover from their own mistakes. On the [RoboDojo](https://robodojo-benchmark.com/) benchmark this shows up as a characteristic failure pattern — policies earn substantial *partial* credit on long-horizon tasks but rarely complete them end-to-end, because small errors compound across steps.

**MILE** (Milestone-grounded closed-**L**oop imagination and **E**xecution) is a framework for **spending test-time compute in embodied policies** — a lightweight, **base-agnostic wrapper** that closes the execution loop at the granularity of physical milestones:

1. **Milestone extraction** — long-horizon tasks are segmented into short stages by sparse, physically grounded milestones (gripper transitions, contact onsets, visual-change events), discovered automatically from demonstrations *without sub-task annotation*;
2. **Imagination** — a ~30M-parameter latent world model predicts, in the frozen VLA's own feature space, the outcome of each candidate action chunk, scored by a jointly trained steps-to-completion value head;
3. **Conformal verification & rollback** — candidates are executed only if they pass a **conformally calibrated** acceptance test (accepted segments terminate within the verification radius with probability ≥ 1−α, distribution-free, finite-sample); a verified-milestone stack defines safe anchors, deviation triggers retreat-and-replan, and every retreat becomes hindsight-labeled **recovery data** for reward-free self-improvement.

The base policy stays **completely frozen**. MILE trains only ~35M parameters.

```
   candidates ──► imagination ──► J(c) = alignment + λ·progress − η·uncertainty
        ▲                              │
        │                    accept? ──┴── reject ──► rollback to
   frozen VLA (DM0.5)                  │              last verified milestone
                                 execute chunk
                                       │
                        verified? push milestone + snapshot
```

## Key Results (RoboDojo)

> ⚠️ Numbers below are placeholders pending final evaluation. The leaderboard
> baselines (2026-09-17 snapshot) are public; our rows will be updated before release.

| Method | Long-Horizon (score/SR%) | Memory (score/SR%) | Avg (score/SR%) |
|---|---|---|---|
| Liber-0 Preview (closed) | 45.98 / 33.17 | 37.77 / 37.33 | 30.74 / 25.52 |
| GPT-6-Astra (board baseline) | 21.45 / 8.25 | 43.04 / 38.67 | 28.97 / 22.48 |
| GalaxeaVLA (G0.5) | 44.12 / 32.25 | 8.61 / 7.33 | 20.23 / 14.88 |
| DM0.5 (our base) | 33.70 / 19.50 | 47.74 / — | 24.98 / 19.34 |
| **DM0.5 + MILE (ours)** | **TBD** | **TBD** | **TBD** |
| π0 + MILE (ours) | TBD | TBD | TBD |

## Installation

MILE is developed and evaluated on Ubuntu 22.04 + NVIDIA RTX 4090/A100.
We recommend running inside the official Docker images of our dependencies.

```bash
# 1. Base policy: DM0.5 (OpenDM)
git clone https://github.com/dexmal/opendm.git
docker run -it --rm --gpus all --network host --shm-size=16g \
  -v $PWD:/app/opendm -w /app/opendm dexmal/opendm:latest /bin/bash
modelscope download --model Dexmal/DM05-robotwin2-bf16 \
  --local_dir ./checkpoints/DM05-robotwin2-bf16

# 2. Benchmark: RoboDojo (Isaac Sim / Isaac Lab)
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
# follow the official setup; XPolicyLab is used as the policy interface

# 3. MILE itself
git clone <this-repo> && cd mile
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA build first
pip install -r requirements.txt   # see the file header for external (non-pip) components
pip install -e .
```

Launch the frozen base policy inference service:

```bash
script/dm05_launcher.sh --task inference --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 --inference-config.port 7891
```

## Quickstart

All hyper-parameters live in [`configs/default.yaml`](configs/default.yaml).

```bash
# S1 — milestone banks (v1: official sub-step boundaries; v2: automatic cues)
python -m mile.scripts.extract_milestones --use-official --demo-dir /path/to/demos

# S2 — train the latent world model + value head (1×4090, ~1–2 days)
python -m mile.scripts.train_wm --config configs/default.yaml

# S3 — calibrate thresholds on the 8 calibration tasks ONLY
python -m mile.scripts.calibrate --tasks <task1> ... <task8>

# S4 — zero-tuning evaluation on the 34 held-out tasks
python -m mile.scripts.run_eval --wm-ckpt runs/wm_v1/wm_final.pt --tasks <task ids...>

# Ablations
python -m mile.scripts.run_eval --open-loop ...          # bare base policy
# set controller.num_candidates: 1 in the config         # w/o verification
# set controller.radius: inf                              # w/o rollback
```

**Anti-overfitting protocol.** All thresholds are tuned on 8 calibration tasks
only; the remaining 34 tasks are evaluated with zero parameter tuning.
Each task runs 50 episodes (25 standard + 25 randomized scenes), reporting the
official sub-step-sensitive Score and Success Rate with 95% confidence intervals.

## Repository Structure

```
mile/
├── configs/default.yaml        # every knob in one place
├── mile/
│   ├── wm/                     # latent world model
│   │   ├── model.py            #   JEPA-style predictor + value head + 3-head ensemble
│   │   ├── data.py             #   sliding-window dataset + progress labels
│   │   └── train.py            #   training loop (AdamW, cosine, bf16)
│   ├── extractor/              # milestone extraction
│   │   ├── cues.py             #   kinematic / contact / visual-change cues
│   │   └── extractor.py        #   top-K selection, DTW alignment, bank builder
│   ├── controller/             # closed-loop execution
│   │   ├── scoring.py          #   J(c) verification scoring
│   │   ├── stack.py            #   verified-milestone stack with snapshots
│   │   ├── controller.py       #   MILE controller (Algorithm 1)
│   │   └── evaluate.py         #   RoboDojo evaluation protocol
│   ├── base/dm05_client.py     # frozen DM0.5 client (HTTP) + feature hook
│   ├── env/robodojo_env.py     # Isaac Lab wrapper: state save/restore
│   └── scripts/                # S1–S4 entry points
└── docs/SETUP_zh.md            # 中文部署与接口说明
    docs/CODE_GUIDE_zh.md       # 逐文件代码详解（中文）
```

> 📖 Per-file code walkthrough (Chinese): [docs/CODE_GUIDE_zh.md](docs/CODE_GUIDE_zh.md)

## Method in 60 Seconds

At every action-chunk decision point (~5 s), the controller:

1. samples **C=4** candidate chunks from the frozen flow-matching policy (free multimodality via different noise seeds);
2. **imagines** each outcome with the latent world model;
3. **scores** candidates by milestone alignment + imagined progress − ensemble uncertainty, and executes the best — *only if it clears a calibrated threshold* (otherwise: rollback);
4. after execution, **verifies** the realized observation against the target milestone: on success it pushes the milestone (with a full physics snapshot) onto the verified stack; on repeated deviation it **retreats** to the most recent verified anchor and replans.

Irreversible stages (pouring, cutting) are flagged one-way: verified but never rolled back. A proposition in the paper shows this re-anchoring converts the multiplicative compounding-error bound of behavior cloning into an additive one over verified segments.

## Roadmap

- [x] Latent world model + value head (reference implementation)
- [x] Milestone extractor (rule-based v1 + cue-based v2)
- [x] Controller with verification & rollback
- [ ] RoboDojo full-suite numbers (in progress)
- [ ] Automatic-extraction ablation vs. official sub-step boundaries
- [ ] RoboDojo-RealEval remote real-robot results
- [ ] π0 as second base policy

## Citation

```bibtex
@inproceedings{mile2027,
  title     = {Think, Verify, Retreat: Milestone-Grounded Closed-Loop
               Imagination for Long-Horizon Vision-Language-Action Policies},
  author    = {Anonymous},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2027},
  note      = {Submitted}
}
```

## Acknowledgements

- [DM0.5 / OpenDM](https://github.com/dexmal/opendm) (Dexmal) — frozen base policy
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) — benchmark & evaluation protocol
- AHEAD, StructVLA, World-in-Loop, WoVR, GigaBrain — conceptual baselines discussed in the paper

## License

Apache 2.0. See [LICENSE](LICENSE). Dependencies retain their own licenses.
