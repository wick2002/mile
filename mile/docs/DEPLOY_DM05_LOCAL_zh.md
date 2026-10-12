# DM0.5 进程内部署手册（服务器，不走 HTTP 服务）

> 目标：在服务器上用**独立 conda 环境**把 DM0.5 直接加载进 Python 进程，
> 不启 Flask/OpenDM HTTP 服务，MILE 与基座同进程调用。
> 最后更新：2026-10-10

---

## 0. 为什么必须走进程内（不只是"省一个服务"）

MILE 的控制器在每个验证点都需要 **`obs["patch_tokens"]`**（VLM patch 特征）：

```python
# controller.py
patch = torch.from_numpy(obs["patch_tokens"]).float()...
```

而 `/v1/infer` 只返回 `actions`，**拿不到特征**。所以：

- HTTP 模式下你必须额外再加载一份 VLM 专门抽特征 → 两份权重、两次前向、还可能出现
  训练(`prepare_latents`)与推理两条不同特征通道不一致的问题；
- 进程内模式**一次前向同时给出动作和特征**，特征通道天然一致。

这是选 `mode: local` 的根本理由。

---

## 1. 环境（独立 conda 环境）

OpenDM 的依赖很重且版本钉死（torch cu128 / flash-attn / triton / 可能的 tensorrt），
**不要**和 MILE 的环境混装之外再叠加。官方本地安装步骤：

```bash
conda create -n opendm python=3.10 -y
conda activate opendm

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation

git clone https://github.com/dexmal/opendm.git && cd opendm
pip install -e .
```

**关于"两个环境"**：MILE 自己的依赖只有 `numpy / pyyaml / Pillow / torch`，
**全部已经在 opendm 环境里了**。所以最省事的做法是：

```bash
conda activate opendm
cd /path/to/mile
pip install -e . --no-deps      # 只装 MILE 包本身，不重复装依赖
```

> 如果你坚持 MILE 与 OpenDM 严格分环境，那就必须给两者之间加一层 IPC
> （ZeroMQ / 共享内存），但那样又退化成"跨进程服务"，而且特征要跨进程序列化
> （`[256, 2560]` float，量不小）。**除非 MILE 的依赖真的和 OpenDM 冲突，否则不要这么干。**

验证环境可用：

```bash
python -c "import torch, opendm, numpy, yaml, PIL; print('ok')"
```

---

## 2. 权重

```bash
cd opendm
pip install -U "huggingface_hub[cli]"

# 官方 RoboDojo 通用权重（榜单 24.90 / 19.34% 的同款）
hf download Dexmal/DM05-MEM-Robodojo-Sim --local-dir ./checkpoints/DM05-MEM-Robodojo-Sim

# 目录内必须含 norm_stats.json；缺失会回退到 ./norm_stats/ 导致数值漂移。
# 官方建议直接用权重自带的那份：
hf download Dexmal/DM05-MEM-Robodojo-Sim norm_stats.json \
  --local-dir ./norm_stats/robodojo_hf
```

MILE 侧指向它：

```yaml
# configs/default.yaml
base:
  mode: local
  checkpoint: "./checkpoints/DM05-MEM-Robodojo-Sim"
```

> `base.checkpoint` 之前只是记录、没有任何代码读它；现在 `dm05_local.py` 会真正消费它。

---

## 3. 关键配置（值均取自 OpenDM 源码，不是猜的）

来源：`opendm/dataset/robodojo.py` 注册项 + XPolicyLab `policy/OpenDM/deploy.yml`。

| 项 | 值 | 来源 / 坑 |
|---|---|---|
| `robot_type` | `null` → 实测解析为 **Aloha** | 注册项写的是 `RobotType.ALOHA`（14 维双臂）。传 `null` 让它用注册项默认值，**别自己填 "ARX X5"** |
| `speed` | **`"0.5"`** | 注册项与 deploy.yml 都是 0.5。项目里原来是 `"0.1"`，**已修正** |
| `control_mode` | `null` | 注册项里没有这个字段，不要凭空编一个 |
| `max_history_images` | **20** | ⚠️ OpenDM 默认是 **5**，不改成 20 发历史帧会直接报错 |
| `model_max_length` | **1536** | deploy.yml 给的值（OpenDM 默认 1024） |
| `is_history` | `true` | MEM 检查点需要历史帧 |
| `backend` | **`default`** | ⚠️ 只能是 default。`fast` 走 TensorRT + CUDA Graph，会绕过 eager 前向，**让 FeatureHook 抓不到特征** |
| `dataset_name` | 必须是 OpenDM 里注册过的名字 | 见下节 |

### ✅ chunk_size 已确认 = 50（不是 25）

`DM05-MEM-Robodojo-Sim` 的 HF model card 原文：

> "The model generates 14-dimensional absolute joint-position action chunks of
> length **50**, and the RoboDojo evaluation adapter executes the first **25**
> actions from each predicted chunk."

所以 `chunk_len: 50` / `exec_horizon: 25` 是对的，与 README 默认 launcher 一致。
那些 `action_chunk_size: 25` / `action_steps: 25`（块长 25 且全部执行）来自
**cover_blocks 单任务 SFT entry** 的 deploy.yml，**不适用于**本通用权重——
两者别混。

探测脚本的 `[2]` 仍会打出真实形状，用它做最终确认（若模型吐 25 行，
形状校验会直接报错，把 `chunk_len` 改成 25 即可）。

---

## 4. dataset_name 怎么办

`_initialize_inference_runtime()` 会从**数据集注册项**里取
`robot_type / state_desc / speed`，注册项由 `opendm/dataset/robodojo.py` 提供。
当前官方只注册了：

```python
register_dataset({"sim_cover_blocks": {...}}, prefix="robodojo")
# -> 可用名字 "robodojo_sim_cover_blocks"
```

所以：

- 如果你用官方 `DM05-MEM-Robodojo-Sim` 跑 `cover_blocks`，直接填
  `robodojo_sim_cover_blocks`；
- 如果你要跑 RoboDojo 的**全部 42 个任务**，官方注册项不够用——需要在
  `opendm/dataset/register.py` 里按同样格式注册你的数据集（或让 checkpoint
  目录自带 `norm_stats.json` 并把 `dataset_name` 指向一个已注册项）。

> 这是目前**唯一需要你在服务器上补的适配点**，其余都已就绪。

---

## 5. 冒烟验证（上服务器第一件事）

必须在 **OpenDM 仓库根目录**下跑，否则 RoboDojo 注册项导不进来：

```bash
conda activate opendm
cd /path/to/opendm
PYTHONPATH=/path/to/opendm:/path/to/mile python -m mile.scripts.probe_dm05_local \
    --config /path/to/mile/configs/default.yaml
```

它依次回答 5 件事：

1. `[1]` 权重能否加载、实际解析出的 `robot_type / speed / max_history_images`；
2. `[2]` 动作块形状 —— **直接暴露 chunk 是 25 还是 50**；
3. `[3]` `FeatureHook` 是否真的抓到东西、长度是否 ≥ `num_patches`；
4. `[4]` **候选多样性**（W1 最高风险项）：8 个 seed 的两两动作距离。
   若相对离散度 `< 1e-3`，说明 flow-matching 的 seed 多样性是平的，
   "想象—选择"机制退化为空转，必须换扰动式候选；
5. `[6]` 单次前向延迟 → 决定 C=4 时每个决策点的耗时（算力—成功率曲线要用它）。

---

## 6. 跑评测

```bash
python -m mile.scripts.run_eval \
    --config configs/default.yaml \
    --wm-ckpt runs/wm_v1/wm_final.pt \
    --tasks <task ids...>
```

`base.mode: local` 时 `run_eval.py` 会自动走 `build_local_policy()`，不再连 HTTP。

**注意 `record_stride`**：恢复片段录制需要对执行步抽 VLM 特征，而
`encode()` 是一次完整模型前向。逐帧录（stride=1）意味着每个决策点多 25 次前向。
默认已设为 `5`（每 5 步抽一次）。恢复片段只用于价值头蒸馏，抽帧不损害语义。

---

## 7. 已知限制

- `mode: local` 下**不能**用 `fast` 后端（会打断特征 hook），推理会比 TensorRT 慢；
  若后续要拼吞吐量，需要另想办法在 fast 路径上取特征（OpenDM 未暴露）。
- RoboDojo 官方评测走的是 **XPolicyLab 协议**，不是 OpenDM 的 `/v1/infer`
  （`docs/en/dm05_robodojo.md` 明确写了）。本手册的进程内方式用于 MILE 自己的
  评测循环；若要对齐官方榜单流程，仍需走 XPolicyLab。
- 一个 policy 进程实例目前只服务一个评测客户端。
