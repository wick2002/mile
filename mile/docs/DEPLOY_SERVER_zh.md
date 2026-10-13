# MILE 服务器部署手册（从裸机到跑通）

> 目标：在 GPU 服务器上把 MILE 完整跑起来 —— DM0.5 进程内加载（不启 HTTP 服务）
> + RoboDojo 环境 + S1→S5 流水线。
> 最后更新：2026-10-12
> 配套：`docs/DEPLOY_DM05_LOCAL_zh.md`（DM0.5 进程内接入细节）、`docs/SETUP_zh.md`（流水线）、
> `docs/CODE_GUIDE_zh.md`（逐文件说明）。

---

## 0. 先读这几条（能省你半天）

1. **不需要 Docker**。我们走 `base.mode: local`，DM0.5 直接在 Python 进程里加载。
   只有当你想对齐 RoboDojo 官方榜单流程（XPolicyLab 协议）时才需要 OpenDM 的服务镜像。
2. **一个 conda 环境就够**。MILE 的依赖（numpy / pyyaml / Pillow / torch）是 OpenDM 环境的
   子集，所以 `pip install -e . --no-deps` 装进同一个环境，不要拆两个（拆了就退化成跨进程 IPC）。
3. **第一天的唯一目标是让 `probe_dm05_local` 的 `[1]–[4]` 全绿**，尤其是 `[4]` 候选多样性。
   这一项不过，后面所有实验都没有意义（详见 §6）。
4. 手册里所有路径用 `~` 展开。全文约定：

   | 路径 | 内容 |
   |---|---|
   | `~/opendm` | OpenDM 仓库根目录 |
   | `~/opendm/checkpoints/DM05-MEM-Robodojo-Sim` | 官方 RoboDojo 权重（23.35 GB） |
   | `~/mile` | 本仓库 |
   | `~/data/robodojo/demos` | RoboDojo 官方示范（3,500 条） |
   | `~/data/robodojo/latents` | `prepare_latents.py` 产物 |
   | `~/runs/wm_v1` | 世界模型 checkpoint |

---

## Step 1 — 机器自检

```bash
nvidia-smi                       # 驱动 + CUDA 版本，记下来
nvidia-smi --query-gpu=memory.total --format=csv
df -h ~                          # 权重 24 GB + latents（见 §Step 9，量很大）
python -V                        # 要求 >= 3.10
```

**门槛**：单卡 ≥ 40 GiB 显存（4B VLM bf16 ≈ 8–10 GiB + 激活 + WM）；磁盘 ≥ 2 TB 才敢跑全量 latents。

---

## Step 2 — 把代码弄到服务器

GitHub `wick2002/mile` 目前推不上去（404/403），所以直接从开发机拷：

```bash
# Windows 开发机（Git Bash / PowerShell）
scp -r /c/Users/zdr2002/WorkBuddy/2026-09-23-11-04-16/mile user@server:~/mile

# 或者先建 git 仓库，方便服务器上回滚
cd mile && git init && git add -A && git commit -m "MILE v0.1"   # 本地先版本化
```

到服务器上立刻做一次版本化（后面改配置随时能 diff）：

```bash
cd ~/mile && git init && git add -A && git commit -m "baseline from dev machine"
```

---

## Step 3 — conda 环境（唯一一个）

```bash
conda create -n opendm python=3.10 -y
conda activate opendm

# CUDA 版 torch 必须先装，否则后面的依赖会把你降级成 CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

> `cu128` 是 OpenDM 官方钉的版本。如果你的驱动更老，退到 `cu121`，但 torch 必须带 CUDA。

---

## Step 4 — 装 OpenDM

```bash
cd ~ && git clone https://github.com/dexmal/opendm.git && cd opendm
pip install -e .
python -c "import opendm, torch; print('opendm ok', opendm.__file__)"
```

**记下 commit hash**，后面升级 OpenDM 时要复查 `dm05_local.py` 用到的私有方法：

```bash
git -C ~/opendm rev-parse HEAD | tee ~/mile/.opendm_commit
```

---

## Step 5 — 装 MILE（同一环境，不装依赖）

```bash
conda activate opendm
cd ~/mile
pip install -e . --no-deps        # 关键：--no-deps，别碰 OpenDM 钉死的版本
python -c "import mile, yaml, PIL, numpy; print('mile ok', mile.__file__)"
```

---

## Step 6 — 下 DM0.5 权重

```bash
conda activate opendm
pip install -U "huggingface_hub[cli]"

# 国内服务器 HF 不通的话先设镜像
export HF_ENDPOINT=https://hf-mirror.com

cd ~/opendm
hf download Dexmal/DM05-MEM-Robodojo-Sim \
    --local-dir ./checkpoints/DM05-MEM-Robodojo-Sim

# ModelScope 备选
# pip install modelscope
# modelscope download --model Dexmal/DM05-MEM-Robodojo-Sim \
#     --local_dir ./checkpoints/DM05-MEM-Robodojo-Sim
```

**必须确认的三件事**：

```bash
ls -la ~/opendm/checkpoints/DM05-MEM-Robodojo-Sim | head -30   # 总体积 ~23 GB
ls ~/opendm/checkpoints/DM05-MEM-Robodojo-Sim/norm_stats.json  # 必须有！
du -sh ~/opendm/checkpoints/DM05-MEM-Robodojo-Sim
```

> `norm_stats.json` 缺失会让动作反归一化回退到 `./norm_stats/`，数值漂移且**与榜单不可比**。
> 缺了就单独补：`hf download Dexmal/DM05-MEM-Robodojo-Sim norm_stats.json --local-dir ./norm_stats/robodojo_hf`

---

## Step 7 — 改配置

**不要直接改 `configs/default.yaml`**（那份是带完整注释的设计基线）。复制一份服务器专用配置：

```bash
cd ~/mile
cp configs/default.yaml configs/server.yaml
```

然后只改这几行（用 sed，保留注释）：

```bash
cd ~/mile
sed -i "s|^  opendm_root:.*|  opendm_root: \"$HOME/opendm\"|"          configs/server.yaml
sed -i "s|^  checkpoint:.*|  checkpoint: \"$HOME/opendm/checkpoints/DM05-MEM-Robodojo-Sim\"|" configs/server.yaml
sed -i "s|^  dataset_name:.*|  dataset_name: robodojo_sim_cover_blocks|" configs/server.yaml
sed -i "s|^  data_dir:.*|  data_dir: \"$HOME/data/robodojo/latents\"|"  configs/server.yaml
```

**确认 exp_entry 真的存在**（名字可能随 OpenDM 版本变）：

```bash
ls ~/opendm/playground/ | grep -i robodojo
grep -rl "DM05Exp" ~/opendm/playground/ | head
```

看到的 RoboDojo **MEM** 入口文件名填进 `base.exp_entry`（默认
`playground/dm05_mem_sft_robodojo_cover_blocks.py`）。

核对改动：

```bash
grep -nE "^(  (mode|checkpoint|opendm_root|exp_entry|dataset_name|robot_type|speed|max_history_images|backend|model_max_length|chunk_len|exec_horizon|enable_features):)" configs/server.yaml
```

**这些值来自 OpenDM 源码，别"想当然"改**：

| 项 | 值 | 为什么 |
|---|---|---|
| `mode` | `local` | 只有进程内能一次前向同时拿动作 + VLM patch tokens |
| `robot_type` | `null` | 留空走注册项（解析为 **Aloha**，14 维双臂）。**别填 "ARX X5"** |
| `speed` | `"0.5"` | 注册项/deploy.yml 是 0.5（原来是 0.1，错的） |
| `max_history_images` | `20` | OpenDM 默认 **5**，不改直接报错 |
| `model_max_length` | `1536` | deploy.yml 值（OpenDM 默认 1024） |
| `backend` | `default` | `fast` 走 TensorRT+CUDA Graph，**会让 FeatureHook 抓不到特征** |
| `chunk_len` / `exec_horizon` | `50` / `25` | 已在 HF model card 核对：出 50 步块、执行前 25 步 |

最后跑环境自检：

```bash
conda activate opendm
cd ~/opendm                                  # RoboDojo 注册项要在根目录下才 import 得到
python -m mile.scripts.check_env --config ~/mile/configs/server.yaml
```

全 `ok` 才继续；有 `FAIL` 就按提示修。

---

## Step 8 — 冒烟测试（**第一天的生死线**）

必须在 **OpenDM 仓库根目录**下跑：

```bash
conda activate opendm
cd ~/opendm
python -m mile.scripts.probe_dm05_local --config ~/mile/configs/server.yaml
```

六项输出与判定：

| # | 看什么 | 通过标准 | 不通过怎么办 |
|---|---|---|---|
| `[1]` | 加载耗时 / `robot_type` / `speed` / `max_history_images` | `robot_type='Aloha'`、`speed='0.5'`、`max_history_images=20` | 注册项不对 → 检查 `dataset_name` |
| `[2]` | 动作块形状 | `(50, 14)`，非全零、数值量级合理 | 若吐 `(25,14)`，把 `chunk_len` 改 25（**同时要复查 WM 的 `window`**） |
| `[3]` | FeatureHook 形状 | `[>=256, 2560]` | 换 `feature_layer`，或 `enable_features: false` 走离线 `prepare_latents` 路径 |
| `[4]` | **候选多样性** | `relative spread > 1e-3` | ⚠️ 见下 |
| `[5]` | 历史帧 | `buffer=20 / 20` | `max_history_images` 没生效 |
| `[6]` | 单步延迟 | 记下来，`C=4` 时每决策点 ≈ 4× 该值 | 太慢就调 `eval.record_stride`、num_diffusion_steps |

> **`[4]` 是最高风险项**。如果不同 seed 出来的 C 个候选几乎一样，
> "想象—验证—选择" 就退化成空转，论文的核心机制没了。
> 备选方案（代码里已留口子）：历史条件扰动 / 动作空间噪声 / 指令 paraphrase 生成候选。
> 一旦要走这条路，**立刻告诉我**，这会改 `ControllerConfig` 与实验设计。

---

## Step 9 — 接 RoboDojo 环境（MILE 侧唯一必须你补的代码）

三处 `TODO(server)`，按官方仓库 + XPolicyLab 接口填：

| # | 位置 | 要做什么 |
|---|---|---|
| 1 | `mile/env/robodojo_env.py::_make_env` | 接 RoboDojo 任务注册表，返回遵循 gymnasium 接口的环境 |
| 2 | `mile/scripts/run_eval.py::env_factory` | 同上的工厂函数 |
| 3 | `mile/scripts/calibrate.py::collect_calibration_pairs` | 裸基座 rollout 收集 `(J, d)` 对（用 `tau_J=-inf` 全接受，观察真实偏差） |

`RoboDojoEnv` 必须提供 `save_state()` / `restore_state()`（回退用，**不是整局 reset**）
和 `set_randomized(bool)`（评测里 25 标准 + 25 随机）。

先装 Isaac Sim 4.x + Isaac Lab 2.x + RoboDojo 基准（各自官方文档），然后单独验：

```bash
python - <<'PY'
from mile.env.robodojo_env import RoboDojoEnv
e = RoboDojoEnv("<some task id>")
e.set_randomized(False); e.reset()
s = e.save_state()
for _ in range(25): e.step(e.action_space.sample() * 0)
e.restore_state(s)
print("save/restore ok")
PY
```

> §5 里剩下的一项**未实现**功能也在这里一并做：**re-approach 回退变体**（不靠状态读档，
> 靠"回到锚点附近重新接近"）。这是应对审稿人"仿真回退 = 真机没有的读档特权"攻击的唯一解药，
> 建议 W6 前实现并进消融。

---

## Step 10 — S1：里程碑库

```bash
conda activate opendm && cd ~/mile
python -m mile.scripts.extract_milestones \
    --config configs/server.yaml \
    --use-official \
    --demo-dir ~/data/robodojo/demos \
    --out data/milestone_banks.pkl
```

v1 直接用 RoboDojo 官方子步骤边界；v2 再切三线索自动提取。

---

## Step 11 — S2 前置：抽 latents（**先看磁盘**）

```bash
# ① 先估算，别盲跑
python -m mile.scripts.prepare_latents \
    --config configs/server.yaml \
    --demo-dir ~/data/robodojo/demos \
    --out-dir ~/data/robodojo/latents \
    --model-path ~/opendm/checkpoints/DM05-MEM-Robodojo-Sim \
    --estimate-only

# ② 小批量试跑（每个任务 20 条），确认 npz 字段与形状
python -m mile.scripts.prepare_latents ... --limit-per-task 20 --batch-size 16

# ③ 全量
python -m mile.scripts.prepare_latents ... --dtype fp16 --batch-size 32
```

**三条硬提醒**：

- **不能抽帧缩小**。`WM 窗口 == exec_horizon(25)` 个真实控制步，抽帧会把它悄悄变成
  `25 × stride` 步，想象终点和真实观测就不再可比（这是项目的硬约束 #1）。
- fp16 + T≈300 ≈ **375 MiB/条**，3,500 条 ≈ **1.3 TiB**。要么先跑子集，要么
  `--limit-per-task` 分批。
- 线上/离线一致性：阶段文本模板 `world_model.stage_text_template` 两端必须相同
  （默认 `"{task}: stage {i}"`），否则 `text_emb` 语义错位。

---

## Step 12 — S2：训练世界模型

```bash
python -m mile.scripts.train_wm --config configs/server.yaml
# 产出 ~/runs/wm_v1/wm_final.pt
```

单张 4090/A100 约 1–2 天。只训 `wm/` 下 ~30M 参数，**基座全程冻结**。

---

## Step 13 — S3：保形标定（只许动 8 个任务）

```bash
python -m mile.scripts.calibrate \
    --config configs/server.yaml \
    --tasks <8 个标定任务 id> \
    --heldout-tasks <34 个裸考任务 id> \
    --alpha 0.05 --radius 0.35 \
    --out configs/calibrated.json
```

产出 `τ̂`。**防过拟合纪律**：34 个 held-out 任务零调参，只做覆盖率验证。

---

## Step 14 — S4：34 个裸考任务评测（论文主表）

```bash
python -m mile.scripts.run_eval \
    --config configs/server.yaml \
    --wm-ckpt ~/runs/wm_v1/wm_final.pt \
    --banks data/milestone_banks.pkl \
    --calibrated configs/calibrated.json \
    --tasks <34 个任务 id> \
    --out results/eval.jsonl
```

消融开关：

```bash
--open-loop                 # 裸基座基线
--candidates 1              # 去掉验证打分
--candidates {1,2,4,8}      # 算力—成功率 scaling 曲线
--recovery-buffer rb.pkl    # 收集恢复片段（S5 自改进用）
```
`controller.radius = inf` → 去掉回退。

---

## Step 15 — S5：自改进轮次

```bash
python -m mile.scripts.run_rounds \
    --config configs/server.yaml \
    --wm-ckpt ~/runs/wm_v1/wm_final.pt \
    --rounds 3 --out-dir ~/runs/rounds
```

只微调 value head；`self_improve.use_lora: true` 是 LoRA 变体开关。

---

## 故障排查速查表

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: opendm` | 没在 opendm 仓库根目录跑，或 `pip install -e .` 没做 |
| `exp entry not found` | `base.exp_entry` 文件名不对，`ls ~/opendm/playground/ \| grep robodojo` |
| `max_history_images` 报历史帧太长 | 配置没生效，确认是 20 不是 5 |
| FeatureHook 抓不到 / 长度不足 | `backend` 误设成 `fast`；或换 `feature_layer` |
| 动作数值全零 / 量级离谱 | `norm_stats.json` 缺失 → 反归一化用了错的统计 |
| 候选几乎一样 | §Step 8 `[4]`，走扰动式候选生成 |
| 卡在 `NotImplementedError: wire to RoboDojo task registry` | §Step 9 的三处 TODO 没补 |
| `numpy>=2` 相关报错 | Isaac Sim 生态不兼容 numpy 2.x，降到 1.26 |
| CUDA OOM | 关掉占卡进程；`eval.record_stride` 调大（默认 5） |

---

## 时间/算力预算（参考）

| 阶段 | 单卡耗时 | 备注 |
|---|---|---|
| Step 1–8 环境 + 冒烟 | 0.5–1 天 | **必须当天完成** |
| Step 9 RoboDojo 接入 | 2–4 天 | 主要是 Isaac Sim 安装 + 三处 TODO |
| Step 10 里程碑 | 数小时 | 官方子步骤，快 |
| Step 11 latents | 1–3 天 | 磁盘是瓶颈，不是算力 |
| Step 12 训练 WM | 1–2 天 | |
| Step 13 标定 | 0.5 天 | 8 个任务 |
| Step 14 评测 | 2–4 天 | 34 任务 × 50 局 × 4 消融 |
| Step 15 自改进 | 2–3 天 | R0→R3 |

---

## 卡住时给我什么

把这三样贴给我，我能直接定位：

1. `python -m mile.scripts.check_env --config configs/server.yaml` 的完整输出；
2. `python -m mile.scripts.probe_dm05_local --config configs/server.yaml` 的 `[1]–[6]`；
3. 报错的完整 traceback + `git -C ~/opendm rev-parse HEAD`。
