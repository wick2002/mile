# MILE — Milestone-Grounded Closed-Loop Imagination and Execution

给冻结的 VLA（DM0.5 / π0）外挂"预判—验证—回退"闭环的代码工程。
对应文档：`../ICML2027_VLA世界模型研究计划.md`、`../MILE_技术设计文档.md`、`../main.tex`。

> 本代码在开发机（Windows）上只做语法检查；**实际运行请在 GPU 服务器（Ubuntu 22.04 + RTX 4090 ×8 + Docker）上进行**。

## 目录结构

```
mile/
├── configs/default.yaml          # 全部超参数（唯一改配置的地方）
├── mile/
│   ├── wm/
│   │   ├── model.py              # JEPA 式 latent 世界模型 + 进度价值头 + 3头集成
│   │   ├── data.py               # 滑窗数据集（40万三元组）
│   │   └── train.py              # 训练循环（AdamW + cosine + bf16）
│   ├── extractor/
│   │   ├── cues.py               # 三路里程碑线索（运动学/接触/视觉）
│   │   └── extractor.py          # top-K 选择、DTW 对齐、里程碑库构建
│   ├── controller/
│   │   ├── scoring.py            # J(c) 打分、过线判定、里程碑到达判定
│   │   ├── stack.py              # 已验证里程碑栈（含物理快照）
│   │   ├── controller.py         # MILE 控制器（论文 Algorithm 1）
│   │   └── evaluate.py           # RoboDojo 评测循环（50局/任务，标准+随机）
│   ├── base/
│   │   └── dm05_client.py        # DM0.5 HTTP 客户端 + 特征 hook（本地模式）
│   ├── env/
│   │   └── robodojo_env.py       # Isaac Lab 环境封装（快照存档/读档）
│   └── scripts/
│       ├── extract_milestones.py # S1: 里程碑库（v1 用官方子步骤边界）
│       ├── train_wm.py           # S2: 训练世界模型
│       ├── calibrate.py          # S3: 8 任务标定（r, τ_J, λ, η）
│       └── run_eval.py           # S4: 34 任务零调参评测
└── README.md
```

## 服务器部署顺序（对应研究计划 W1-W8）

```bash
# W1: 环境（在服务器上）
git clone https://github.com/dexmal/opendm.git && cd opendm
docker run -it --rm --gpus all --network host --shm-size=16g \
  -v $PWD:/app/opendm -w /app/opendm dexmal/opendm:latest /bin/bash
# 另开容器部署 RoboDojo (Isaac Sim/Lab)：
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git

# W1 检查点：启动 DM05-robotwin2 推理服务并复现榜单数字
modelscope download --model Dexmal/DM05-robotwin2-bf16 --local_dir ./checkpoints/DM05-robotwin2-bf16
script/dm05_launcher.sh --task inference --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 --inference-config.port 7891

# S1 (W3): 里程碑库 —— v1 直接用官方子步骤边界
python -m mile.scripts.extract_milestones --use-official --demo-dir /path/to/demos

# S2 (W4): 训练世界模型（先用 prepare_latents 生成滑窗数据，见 wm/data.py 注释）
python -m mile.scripts.train_wm --config configs/default.yaml

# S3 (W5): 8 个标定任务上调参
python -m mile.scripts.calibrate --tasks <task1> ... <task8>

# S4 (W6-8): 34 个任务零调参评测
python -m mile.scripts.run_eval --wm-ckpt runs/wm_v1/wm_final.pt --tasks <34 task ids>
# 消融：--open-loop 跑裸基座；改 configs 里的 num_candidates=1 去验证；等
```

## 待你在服务器上补的接口（标了 TODO 的位置）

1. `mile/env/robodojo_env.py::_make_env` —— 接 RoboDojo 官方任务注册表（XPolicyLab）；
2. `mile/scripts/calibrate.py::eval_fn` —— 用 `run_episode` 实现标定评测；
3. `mile/scripts/run_eval.py::env_factory` —— 同上；
4. 一个 `prepare_latents.py` 预处理脚本：用 `FeatureHook` 批量把 3,500 条示范
   转成 `wm/data.py` 期望的 npz 格式（latents/actions/milestones/text_emb/snapshots）。

## 关键设计约束（改代码前必读）

- **基座全程冻结**：只训练 wm/ 下的约 35M 参数；
- 标定只许动 8 个 calibration 任务，34 个 held-out 任务禁调参；
- 仿真回退必须用 `env.save_state()/restore_state()`，不要重新 reset 整局；
- 不可逆阶段（倒水/切割）在 `controller.one_way_ranks` 登记，禁止回退。
