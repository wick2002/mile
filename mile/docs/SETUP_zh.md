# MILE 部署与运行指南（中文）

> **MILE** = Milestone-grounded closed-Loop imagination and Execution
> 给冻结的 VLA（DM0.5 / π0）外挂"想象—验证—回退—自改进"闭环。
> 配套文档：`ICML2027_VLA世界模型研究计划.md`、`MILE_技术设计文档.md`、`main.tex`、
> `MILE_创新性与可行性评估.md`（投稿前评估）、`docs/CODE_GUIDE_zh.md`（逐文件说明）、
> **`docs/DEPLOY_SERVER_zh.md`（从裸机到跑通的完整服务器部署手册，第一步看这个）**、
> **`docs/DEPLOY_DM05_LOCAL_zh.md`（进程内不启服务部署 DM0.5，推荐）**。
>
> 本仓库在开发机（Windows）上只做语法检查；**实际运行请在 GPU 服务器**（Ubuntu 22.04 + RTX 4090/A100 + Docker + NVIDIA Container Toolkit）。
> 最后更新：2026-10-08（Plan A+，接口已切到官方 `/v1/infer`）

---

## 1. 安装

```bash
git clone <this-repo> && cd mile
conda create -n mile python=3.10 -y && conda activate mile
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 版优先
pip install -r requirements.txt   # torch / numpy<2 / pyyaml / gymnasium / requests / Pillow
pip install -e .                  # pyproject.toml（README 里的安装命令依赖它）
```

> Pillow 是必需的：`/v1/infer` 的图像走 base64 JPEG 编码。

**不走 pip 的外部组件**（各自按官方文档装，顺序即依赖顺序）：

| 组件 | 用途 | 说明 |
|---|---|---|
| Isaac Sim 4.x + Isaac Lab 2.x | RoboDojo 仿真环境 | 需 RTX GPU，官方安装器 |
| RoboDojo 基准 + XPolicyLab | 42 任务 + 3,500 条示范 | 官方仓库，XPolicyLab 作策略接口 |
| OpenDM（dexmal/opendm） | DM0.5 推理服务 | Docker 镜像，暴露 `POST /v1/infer` |
| RealEval 账号（可选） | W10 远程真机评测 | RoboDojo 官网注册排队 |

---

## 2. DM0.5 接入方式（二选一）

### 2a. `base.mode: local` —— 进程内加载，**推荐**

不启服务，直接在 Python 进程里加载权重。除了省一个进程，更重要的是
**一次前向同时拿到动作和 VLM patch 特征**（`/v1/infer` 只返回动作，而 MILE
每个验证点都要 `obs["patch_tokens"]`）。完整步骤见 **`docs/DEPLOY_DM05_LOCAL_zh.md`**。

### 2b. `base.mode: http` —— OpenDM 的 Flask 服务

按下面启服务，然后客户端走 `POST /v1/infer`。注意此模式下你必须**另想办法**
提供 `patch_tokens`（额外加载一份 VLM 抽特征），否则评测循环跑不起来。

```bash
git clone https://github.com/dexmal/opendm.git && cd opendm
docker run -it --rm --gpus all --network host --shm-size=16g \
  -v $PWD:/app/opendm -w /app/opendm dexmal/opendm:latest /bin/bash
conda activate opendm && pip install -e .

# 官方 RoboDojo 微调权重（即榜单 24.90 / 19.34% 的同款 checkpoint）
pip install -U "huggingface_hub[cli]"
hf download Dexmal/DM05-MEM-Robodojo-Sim --local-dir ./checkpoints/DM05-MEM-Robodojo-Sim
# ModelScope 镜像：modelscope download --model Dexmal/DM05-MEM-Robodojo-Sim --local_dir ...
# 目录内必须含 norm_stats.json（缺失会回退到 ./norm_stats/，数值可能漂移）

script/dm05_launcher.sh --task inference \
  --exp <robodojo 入口，见官方 docs/en/dm05_robodojo.md> \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM-Robodojo-Sim \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.max-history-images 20 \
  --data-config.is-history \
  --inference-config.port 7891
```

### `/v1/infer` 接口契约（客户端已按此实现）

```jsonc
// 请求
{
  "observation": {
    "prompt": "任务指令",
    "state": [14 个 float],                       // 必须匹配 checkpoint 的 norm stats
    "images": {"1": "b64头", "2": "b64左腕", "3": "b64右腕"},   // 1-based 槽位，按序对应 image-prompts
    "history_images": ["b64最旧", ..., "b64最新"], // 需服务带 --data-config.is-history
    "robot_type": "…", "control_mode": "joint", "speed": "0.1"   // 可选，缺省用数据集默认
  },
  "sampling": {"num_steps": 10, "seed": 123}      // seed → 确定性采样（候选多样性的来源）
}
// 响应
{"actions": [[...] × 50], "metadata": {"latency_ms": 123.4}}
```

- 旧的 `/process_frame`（multipart）仍可用但将淘汰；客户端 `legacy_api=True` 可切回。
- `num_steps` 必须等于服务端固定扩散步数，不确定就在配置里留空（客户端会自动省略该字段）。
- `metadata.latency_ms` 会被客户端记录到 `last_latency_ms`，**算力—成功率缩放曲线直接用它**。

### 官方协议要点（已固化在 `configs/default.yaml`，改动会让数字不可比）

| 项 | 值 | 说明 |
|---|---|---|
| 相机顺序 | 头 / 左腕 / 右腕 | 不可换序 |
| 动作 | 14 维绝对关节位置 | ARX X5 双臂 |
| 块长 / 执行 | 50 / **25** | `exec_horizon: 25`，WM 想象窗口同步为 25 |
| 历史帧 | 20 帧 @1FPS | episode 开头左补齐；客户端每局 `reset_history()` |
| 策略接入 | XPolicyLab | PR #101（已合并则用 main 分支） |
| `robot_type` | 待确认 | TODO：按官方 RoboDojo 指南 / checkpoint 的 norm_stats 填 ARX X5 对应字符串 |

---

## 3. W1 冒烟测试清单（决定后续能否开工）

按顺序验，任何一项不过就先解决：

1. **服务健康**：发一个假 obs，返回 `actions` 形状 `[50, 14]`；
2. **⭐ 候选多样性**（最高风险项）：同一 obs 发 8 个不同 `seed`，量测候选间两两动作距离。
   若候选几乎相同 → 整个"免费多模态 + 选择"机制失效，需改用扰动式候选生成（历史条件扰动 / 动作空间噪声 / 指令 paraphrase）；
3. **协议复现**：跑官方评估流程，确认裸基座复现榜单 24.90/19.34%（±1 分）；
4. **特征通道**：进容器用 `FeatureHook` 抓一次，输出 `[seq, 2560]`，取前 256 个位置为 patch 特征；
5. **环境回退**：`save_state()` → 走 25 步 → `restore_state()`，确认物体位姿与接触状态还原。

---

## 4. 实验流水线（S1–S5）

```bash
# S1 (W3) 里程碑库：v1 直接用官方子步骤边界，v2 用三线索自动提取
python -m mile.scripts.extract_milestones --use-official --demo-dir /path/to/demos

# 数据准备（S2 前置，**必须在 OpenDM 容器内跑**）：
#   /v1/infer 只返回动作，取不到 patch 特征，所以用 FeatureHook 在进程内抓
python -m mile.scripts.prepare_latents --demo-dir /path/to/demos \
       --out-dir data/robodojo_demos_latents \
       --model-path ./checkpoints/DM05-MEM-Robodojo-Sim --batch-size 32
#   先跑 --estimate-only 看磁盘开销（默认 T≈300、P=256、D=2560、fp16 约 375
#   MiB/条，3,500 条约 1.3 TiB）；别用抽帧缩小——WM 窗口必须等于 exec_horizon
#   个真实控制步，抽帧会把它悄悄变成 25×stride 步。

# S2 (W4) 训练世界模型 + 价值头（1×4090，约 1–2 天）
python -m mile.scripts.train_wm --config configs/default.yaml

# S3 (W5) split-conformal 标定：只在 8 个标定任务上拟合 τ̂
python -m mile.scripts.calibrate --tasks <task1> ... <task8> \
       --heldout-tasks <34 个裸考任务>      # 只做覆盖率验证，绝不参与拟合
# 产出 configs/calibrated.json（τ̂ + 理论保证下界 + 拒绝率）

# S4 (W6-8) 34 个裸考任务零调参评测 —— 论文主表数字
python -m mile.scripts.run_eval --wm-ckpt runs/wm_v1/wm_final.pt --tasks <34 ids>
#   消融开关：
#     --open-loop                 裸基座（无 MILE）
#     --candidates 1              去掉验证打分（单候选）
#     controller.radius = inf     去掉回退（永不判"未到达"）
#     --candidates {1,2,4,8}      算力—成功率缩放曲线（W9）
#     --recovery-buffer <path>    收集恢复片段（自改进用）

# S5 (W10) 自改进轮次：R0 基线 → 部署收片段 → 蒸馏价值头 → 复评
python -m mile.scripts.run_rounds --wm-ckpt runs/wm_v1/wm_final.pt --rounds 3
```

---

## 5. 待你在服务器上补的接口（`TODO(server)`）

| # | 位置 | 要做什么 |
|---|---|---|
| 1 | `mile/env/robodojo_env.py::_make_env` | 接 RoboDojo 官方任务注册表（XPolicyLab） |
| 2 | `mile/scripts/calibrate.py::collect_calibration_pairs` | 裸基座 rollout 收集 (J, d) 对（用 `τ_J=-inf` 全接受以便观察真实偏差） |
| 3 | `mile/scripts/run_eval.py::env_factory` | 同上，接 RoboDojo 环境 |
| 4 | `mile/scripts/prepare_latents.py` | 已补脚本；仍需按你的 RoboDojo 示范实际字段适配 `load_demo()`，并确认容器里 VLM backbone 的属性路径（不匹配时用 `--adapter`） |
| 5 | `configs/default.yaml::base.robot_type` | 确认 ARX X5 在 norm_stats 中的机型字符串 |

**另外一项未实现的高优先级功能**：`rollback()` 的 **re-approach（无状态读档）变体**——
这是应对"仿真回退 = 真机没有的读档特权"这一审稿攻击的唯一解药，建议 W6 一并实现并纳入消融。

---

## 6. 改代码前必读的约束

- **基座全程冻结**：只训练 `wm/` 下约 30M 参数；自改进时只微调 value head；
- **防过拟合纪律**：标定只许动 8 个 calibration 任务，34 个 held-out 任务零调参、只做验证；
- **想象与执行对齐**：WM 窗口必须等于 `exec_horizon`（25），否则想象终点与真实观测不可比；
- **回退语义**：用 `env.save_state()/restore_state()`（读档后空转 5 步稳定接触），不要整局 reset；
- **不可逆阶段**：倒水/切割等登记进 `controller.one_way_ranks`，只允许 verify 或 abort。

---

## 7. 目录结构（当前）

```
mile/
├── configs/default.yaml        # 全部超参数（唯一需要改配置的地方）
├── mile/
│   ├── wm/{model,data,train}.py        # JEPA latent 世界模型 + 价值头 + 滑窗训练
│   ├── extractor/{cues,extractor}.py   # 三路里程碑线索 + top-K + DTW 对齐 + 里程碑库
│   ├── controller/
│   │   ├── scoring.py                  # J(c) 打分、过线判定、到达判定
│   │   ├── stack.py                    # 已验证里程碑栈（含物理快照）
│   │   ├── controller.py               # MILE 控制器（Algorithm 1）+ 恢复录制钩子
│   │   ├── conformal.py                # split-conformal 阈值标定
│   │   ├── recovery.py                 # 恢复缓冲 + 价值头蒸馏
│   │   └── evaluate.py                 # 评测循环（50 局/任务，标准 + 随机）
│   ├── base/dm05_client.py             # /v1/infer 客户端 + 历史帧缓冲 + FeatureHook
│   ├── base/dm05_local.py              # 进程内 DM0.5（不启服务，动作+特征一次前向）
│   ├── env/robodojo_env.py             # Isaac Lab 封装（存档/读档）
│   └── scripts/
│       ├── check_env.py                # 上服务器第一跑：环境/路径/显存/磁盘自检
│       ├── probe_dm05_local.py         # DM0.5 进程内冒烟（含候选多样性判定）
│       ├── prepare_latents.py          # 示范 → npz（latents/actions/milestones/text_emb）
│       └── {extract_milestones,train_wm,calibrate,run_eval,run_rounds}.py
├── docs/{SETUP_zh.md, CODE_GUIDE_zh.md, DEPLOY_DM05_LOCAL_zh.md, DEPLOY_SERVER_zh.md}
├── requirements.txt / pyproject.toml / LICENSE / .gitignore
└── README.md                            # GitHub 英文自述
```
