# MILE 代码文件详解
## 逐文件说明：每个文件做什么、对外接口是什么、和谁协作

> 本文档对应 **Plan A+** 版本（2026-09-28）：在原始闭环方案之上新增三个升级件——
> ① split-conformal 阈值标定（替代手工网格搜索，带有限样本覆盖率保证）；
> ② 测试时计算量缩放实验（C=1/2/4/8 scaling 曲线）；
> ③ 自改进回路（回退时录制恢复片段 → 蒸馏价值头）。
> 新增文件：`controller/conformal.py`、`controller/recovery.py`、`scripts/run_rounds.py`；
> 重写文件：`scripts/calibrate.py`；改动文件：`configs/default.yaml`、`wm/model.py`、`controller/controller.py`、`controller/evaluate.py`、`scripts/run_eval.py`。
>
> 阅读顺序建议：`configs/default.yaml` → `wm/model.py` → `controller/controller.py` → `controller/conformal.py` → `controller/recovery.py` → 其余按需查阅。
> 约定：张量形状用 `[B, P, D]` 记法——B=批大小，P=画面块数（256），D=特征维数（2560）。

---

## 0. 全局配置

### `configs/default.yaml`
**作用**：整个工程唯一需要改的配置文件，所有"旋钮"集中在这里。
**分节**：
| 节 | 内容 | 关键参数 |
|---|---|---|
| `base` | 冻结基座 DM0.5 的接口信息 | 推理服务地址、特征层（-2=倒数第二层）、动作维数 14、chunk 50 步、3 路相机键名 |
| `world_model` | 世界模型架构 | d_model=1024、4 层、16 头、动作 token 8 个、集成头 3 个、掩码率 20%、三项损失权重 |
| `train` | 训练超参 | lr 3e-4 余弦、batch 256、6 万步、bf16 |
| `milestones` | 里程碑提取 | 三线索权重 α/β/γ、最小间隔 100 步、K∈[3,10]、`use_official_substeps`（v1 开关） |
| `controller` | 执行控制器 | C=4 候选、λ=1.0、η=0.5、r=0.35、n=2、最多弹栈 2 层（τ_J 不再手填，由 conformal 标定给出） |
| `conformal` | 阈值标定（新增） | α=0.05（目标误接受率）、半径 r=0.35 固定不搜、只用 8 个标定任务、至少 200 个标定点 |
| `self_improve` | 自改进（新增） | 每轮部署 500 局、价值头蒸馏 500 步（batch 64、lr 1e-4）、`use_lora` 变体开关 |
| `eval` | 评测协议 | 每任务 50 局、8 标定任务 / 34 裸考任务、`candidate_sweep: [1,2,4,8]`（计算量缩放曲线） |

---

## 1. 世界模型（`mile/wm/`）

### `mile/wm/model.py` —— 核心中的核心
**作用**：JEPA 式 latent 世界模型，回答"执行这段动作后场景特征会变成什么样"，并附带进度评估。

**类 `LatentWorldModel(nn.Module)`**
- 输入投影：
  - `patch_proj`：画面 patch 特征 2560 维 → 1024 维；
  - `action_compress`：Conv1d（k=7, s=6）把 [B, 50, 14] 的动作块压成 [B, 8, 1024] 的 8 个动作 token；
  - `text_proj`：阶段描述文本特征 2560 维 → 1024 维；
  - `value_query`：一个可学习的查询 token，它的输出位置用来读进度值；
  - `mask_token`：训练时替换被遮掉的 patch（20% 掩码）。
- 主干：`trunk` = 4 层 pre-LN TransformerEncoder（d=1024, 16 头, FFN 4096）。输入序列 = 256 patch + 8 动作 + 4 文本 + 1 查询 = 269 个 token。
- 输出头：
  - `pred_heads`：3 个独立线性层（1024 → 2560），**集成预测**——三个头预测越不一致，说明模型越没把握（不确定性的来源）；
  - `value_head`：2 层 MLP + sigmoid，输出 [0,1] 进度值。
- `forward(patch_tokens, action_chunk, text_emb, patch_mask=None)` → 返回 `WMOutput(pred_latents [3,B,256,2560], value [B], uncertainty [B,256])`。
- `forward_value(patch_tokens, text_emb)` → 推理时只读进度值的快捷方式（`@torch.no_grad`）。
- `value_path(patch_tokens)` → **新增**：可微分的"只走价值头"前向（动作/文本条件置零）。自改进蒸馏专用——恢复片段只有场景特征、没有对应动作，用这条路算 value 并反传。

**函数 `wm_loss(out, target_latents, target_value, ...)`**
三项损失加权和：预测 Smooth-L1 + 预测余弦距离 + 进度 Huber。返回总损失和各项数值（供日志）。

**函数 `sample_patch_mask(batch, num_patches, ratio, device)`**
生成 20% 随机掩码，训练时防"策略钻想象漏洞"。

### `mile/wm/data.py` —— 训练数据
**类 `SlidingWindowDataset(Dataset)`**
- 读取预处理好的 npz 轨迹（每条含 `latents/actions/milestones/text_emb`）；
- 初始化时扫描所有文件，用宽度 50 步、步长 10 的滑动窗口建索引（约 40 万样本）；
- `__getitem__` 返回一个训练样本：当前帧 patch 特征、未来 50 步动作、50 步后的目标特征、下一阶段文本、进度标签。
- **静态方法 `progress_label(t, milestones, T)`**：分段线性进度标签——第 i 个里程碑处进度 = i/(K+1)，中间线性插值。无需人工标注。

**函数 `make_dataloader(...)`**：标准 DataLoader 封装（batch 256、8 workers、pin_memory）。

### `mile/wm/train.py` —— 训练循环
**函数 `train_wm(cfg)`**：完整训练流程——建模型（打印参数量）、AdamW + 余弦学习率（`cosine_lr` 函数，含 warmup）、bf16 autocast、梯度裁剪 1.0、每 100 步打日志、每 5000 步存档。跑完输出 `wm_final.pt`。单卡 4090 参考实现。

---

## 2. 里程碑提取（`mile/extractor/`）

### `mile/extractor/cues.py` —— 三路信号
**`kinematic_cue(gripper, ee_vel)` → [T]**
运动学线索：夹爪开合度突变超过 30% 量程记 1 分（抓/放瞬间）；末端速度低于峰值 5% 或主速度方向反转记 0.5 分（动作阶段结束）。

**`contact_cue(contact)` → [T]**
接触线索：任意手指"没碰→碰"记 1 分，"碰→没碰"记 0.5 分。训练时用仿真器接触力真值，部署时换成探针预测值。

**`visual_cue(latents, attn_mask=None)` → [T]**
视觉线索：相邻帧特征距离的局部极大值。可选 `attn_mask`（指令注意力图）做加权——只看指令相关区域，屏蔽背景干扰。

**`consensus(kin, con, vis, α, β, γ)` → [T]**
三路加权投票，得到每帧的最终里程碑得分。

### `mile/extractor/extractor.py` —— 筛选与建库
**数据类 `Milestone`**：一个里程碑 = 帧号 + 该帧特征 + 物理快照 + 阶段文字。
**数据类 `MilestoneBank`**：一个任务的完整里程碑序列（提供 `.latents` 属性直接取 [K,P,D]）。

**`select_topk(scores, k, min_separation)`**
贪心选 top-K：按得分从高到低挑，跳过与已选点距离小于 100 步的候选。

**`dtw_distance(a, b)`**：标准 DTW 距离，用于衡量两条示范的特征序列相似度。

**`align_milestones(demo_milestones)`**
50 条示范的里程碑对齐：取出现次数最多的 K（众数），再对同一序位的里程碑帧号取中位数。

**`build_bank(task_id, demo_latents, demo_milestones, stage_texts, demo_snapshots)`**
把多条对齐后的示范合并成里程碑库：代表特征取各示范的逐 patch 中位数，快照取第一条可用的。

---

## 3. 执行控制器（`mile/controller/`）

### `mile/controller/scoring.py` —— 打分
**`cosine_to_milestone(pred_latents, milestone_latent)` → [B]**
预测结局与目标里程碑的余弦相似度（打分第一项"方向分"）。

**`score_candidates(pred_latents, values, uncertainties, milestone_latent, lam, eta)` → J[B]**
完整打分公式 `J(c) = 方向分 + λ×进度分 − η×不确定分`。

**`pick_best(scores, tau_J)` → (best_idx, accepted)**
选最高分；但**最高分低于阈值 τ_J 时全员拒绝**（返回 accepted=False，控制器将触发回退）。τ_J 由 conformal 标定给出（见下），不再是手调超参。

**`milestone_reached(realized_latent, milestone_latent, radius)` → (bool, dist)**
执行后核对：真实特征与里程碑的欧氏距离是否小于半径 r。

### `mile/controller/conformal.py` —— 阈值标定（新增，论文 §4.4）
**问题**：接受阈值 τ_J 手调既没理论保证又容易过拟合。
**做法**：split-conformal 标定——用**裸基座**在 8 个标定任务上跑，记录每个决策点的 (J, d) 对（打分 + 实际偏差），选一个**最小的** τ̂，使"被接受样本中偏差的 (1−α) 分位数 ≤ r"。
**保证**：有限样本边缘覆盖率 P(被接受 且 偏差 ≤ r) ≥ 1 − α − 1/(n+1)，分布无关。

- `conformal_quantile(values, alpha)`：带 ⌈(1−α)(n+1)⌉ 有限样本修正的经验分位数；
- `calibrate_tau(J, d, radius, alpha)` → `ConformalResult`：扫候选阈值取满足覆盖率的最小者；若任何阈值都不满足，返回"全拒绝"阈值（安全退化）。结果里带 n_calib、理论保证下界、拒绝率；
- `empirical_coverage(J, d, tau_hat, radius)`：在 **34 个裸考任务**上验证实际覆盖率是否达标（只验证，不参与拟合——防过拟合纪律的代码化）。

### `mile/controller/recovery.py` —— 恢复缓冲与蒸馏（新增，论文 §4.6）
**核心思想**：每次回退后若能成功抵达下一个里程碑，这段"从失败边缘爬回来"的轨迹就是免费的恢复示范。

- 数据类 `RecoverySegment`：一段恢复 = 任务号 + 回退锚点序位 + 实际走过的特征/动作序列 + 最终验证通过的序位；
- 类 `RecoveryBuffer`：片段容器。`value_distill_batch(batch_size)` 采样蒸馏批——**片段开头附近正是基座进度估计最不靠谱的"失败边界"状态**，标签 = 在成功恢复中的归一化位置（0=在锚点，1=到达里程碑）；
- `distill_value_head(world_model, buffer, ...)`：冻结 WM 其余全部参数，**只微调 value_head**（Huber 损失，默认 500 步）。通过 `world_model.value_path(z)` 做可微前向。基座和 WM 主干全程不动——这是"安全自改进"的关键。

### `mile/controller/stack.py` —— 已验证里程碑栈
**数据类 `VerifiedEntry`**：一条入栈记录 = 里程碑序号 + 当时真实特征 + **环境物理快照**（回退的读档点）。
**类 `VerifiedStack`**
- `push/top/next_rank`：常规栈操作；`next_rank` 直接给出"下一个目标里程碑是第几个"；
- `rollback_target()`：回退目标 = 栈顶（最近一个验证通过的里程碑）；弹栈次数超限返回 None（放弃本局）；
- `pop()`：连续失败时弹出栈顶，允许退到更早的里程碑；
- `reset()`：每局开始时清空。
- `max_pop=2`：最多连退 2 层，防止在错误里无限打转。

### `mile/controller/controller.py` —— 论文 Algorithm 1
**数据类 `ControllerConfig`**：C、λ、η、τ_J、r、n、max_pop 的容器。
**类 `MILEController`**：把基座客户端、世界模型、里程碑库、验证栈组装起来。

- **`act(obs)` → (action_chunk, info)**：一次决策的完整流程——
  1. 取下一个目标里程碑（全部完成则进入 `tail` 模式直接执行基座）；
  2. 向基座要 C=4 个候选动作块（不同噪声种子）；
  3. 世界模型批量想象 4 个结局；
  4. 打分选优；全员不过线则返回 `action: "rollback"`；
  5. 返回最优候选动作块。
- **`rollback()`**：退到栈顶快照；**同时 `_begin_recording()` 开始录制恢复片段**；当前阶段在 `one_way_ranks`（不可逆操作）里则直接 `abort`（录制丢弃）。
- **`record_step(latent, action)`**：执行每一帧时调用——若正在录制，把实际特征和动作追加进当前片段。
- **`verify(realized_latent, env_snapshot)` → event**：执行完 50 步后调用——
  - 到达里程碑：`verified`，入栈并存快照；**若在录制中则 `_commit_recording()`——片段入 RecoveryBuffer**（回退→成功恢复 = 一条免费恢复示范）；
  - 轻微偏离：`deviated`，连续计数 +1；
  - 连续 2 次偏离：触发 `rollback`。
- **`reset()`**：每局开始调用。
- 构造参数新增 `recovery_buffer=None`：不传则退化为纯执行模式（W6-8 首轮评选用），传了则启用自改进数据收集（W10 轮次实验用）。

### `mile/controller/evaluate.py` —— 评测循环
**`run_episode(env, controller, base_client, instruction, episode_seed)`**
跑一局：reset → 循环（决策 → 逐帧执行动作块，**每帧 `controller.record_step(...)` 供恢复录制** → 验证/回退）→ 取官方子步骤评分。`controller=None` 时就是裸基座对照组（开环）。
**`evaluate(env_factory, task_ids, controller_factory, base_client, episodes_per_task, out_path)`**
全量评测：每任务 50 局（前 25 标准场景、后 25 随机场景），逐局写 jsonl，最后打印每任务的 Score/SR 和 95% 置信区间。

---

## 4. 基座接口（`mile/base/`）

### `mile/base/dm05_client.py`
**数据类 `DM05Config`**：服务地址、动作维数、chunk 长度、相机键名。
**类 `BasePolicyClient`**：HTTP 客户端，对接 OpenDM 推理服务（`/process_frame`）。
- `sample(obs, seed)`：要一个动作块；
- `sample_candidates(obs, n)`：用 n 个不同随机种子要 n 个候选——**flow matching 的随机性免费提供了候选多样性**；
- `_request`：把指令、状态、多路图像打包成 multipart 发 POST，解析返回的动作块。
**类 `FeatureHook`**：本地模式专用——在 OpenDM 容器内注册 forward hook 到 VLM 倒数第二层，抓取 patch 特征（世界模型的输入 z 就从这里来）。

---

## 5. 仿真环境（`mile/env/`）

### `mile/env/robodojo_env.py`
**类 `RoboDojoEnv`**：Isaac Lab 环境的薄封装。
- `_make_env`：**待接线**（接 RoboDojo 官方任务注册表 XPolicyLab）；
- `save_state()` → dict：存档——关节角、关节速度、所有物体位姿；
- `restore_state(state)`：读档——写回状态后再空转 5 步让接触物理稳定（**仿真回退的实现，零成本**）；
- `contact_flags()`：训练期取接触力真值（给里程碑接触线索用）；
- 其余 `reset/step` 为标准 gym 接口。

---

## 6. 命令行脚本（`mile/scripts/`）

### `mile/scripts/extract_milestones.py`（S1，W3）
遍历每个任务的示范目录，`--use-official` 时直接用官方子步骤边界（v1 保底），否则跑三线索自动提取；输出 `milestone_banks.pkl`（42 个任务的里程碑库）。

### `mile/scripts/train_wm.py`（S2，W4）
一行入口：读 YAML 配置 → 调 `wm/train.py::train_wm`。

### `mile/scripts/calibrate.py`（S3，W5，已重写）
**split-conformal 标定**（替代原网格搜索）：
1. `collect_calibration_pairs(cfg, tasks)`：裸基座在 8 个标定任务上跑，每个决策点记录 (J, d)（**待接线**到 RoboDojo；实现时用 τ_J=−inf 全接受以便观察所有 chunk 的实际偏差）；
2. `calibrate_tau(...)` 给出满足覆盖率的最小 τ̂，写 `configs/calibrated.json`（含理论保证下界和拒绝率）；
3. `--heldout-tasks` 传入 34 个裸考任务时，只做覆盖率验证打印，**绝不参与拟合**——防过拟合纪律。

### `mile/scripts/run_eval.py`（S4，W6-8）
加载里程碑库 + 标定参数 + WM 权重，组装控制器工厂，对任务列表跑全量评测；`--open-loop` 切换为裸基座对照组；`env_factory` 待接线到 RoboDojo。
**新增参数**：
- `--candidates N`：覆盖候选数 C——配合 `candidate_sweep: [1,2,4,8]` 跑 W9 计算量-成功率 scaling 曲线；
- `--recovery-buffer PATH`：把本批评测产生的恢复片段存成 pkl——W10 自改进轮次的数据收集入口。

### `mile/scripts/run_rounds.py`（S5，W10，新增）
自改进轮次实验（论文 Figure 3）：R0 先评一次（蒸馏前基线）→ 每轮部署 `episodes_per_round` 局收集恢复片段 → `distill_value_head` 蒸馏 → 存 `wm_round{r}.pt` → 再评。输出 `rounds.json`（R0→R3 成功率曲线数据）。评测/部署两处 TODO(server) 接 run_eval；`use_lora` 变体暂抛 NotImplementedError（服务器端实现）。

---

## 7. 包结构文件

| 文件 | 作用 |
|---|---|
| `mile/__init__.py` | 包说明 + 版本号（0.1.0） |
| `mile/{wm,extractor,controller,base,env,scripts}/__init__.py` | 空文件，使各目录成为 Python 包 |

## 8. 文档

| 文件 | 作用 |
|---|---|
| `mile/README.md` | GitHub 英文自述（项目门面） |
| `mile/docs/SETUP_zh.md` | 中文部署指引（服务器操作顺序、4 处待接线接口清单） |
| `mile/docs/CODE_GUIDE_zh.md` | 本文档 |

---

## 调用关系总图

```
extract_milestones.py ──► extractor/{cues, extractor}.py ──► milestone_banks.pkl
                                                              │
prepare_latents (待写) ──► base/dm05_client.FeatureHook ──► npz 轨迹数据
                                                              │
train_wm.py ──► wm/{data, model, train}.py ◄──────────────────┘
                     │
                     ▼
                wm_final.pt ──┐
milestone_banks.pkl ──────────┼──► run_eval.py ──► controller/{controller, scoring, stack}.py
configs/calibrated.json ──────┘    │    ▲                  │
      ▲                            │    │                  ├──► base/dm05_client（要候选动作）
      │                            │    │                  └──► env/robodojo_env（执行+存档/读档）
      │                            ▼    │                  │
      │                     controller/evaluate.py ──► results/eval.jsonl
      │                                   │ record_step 录制
      │                                   ▼
      │                     controller/recovery.py ──► buffer pkl ──┐
      │                                                            ▼
      └──── calibrate.py（split-conformal，8 标定任务）      run_rounds.py
            └──► controller/conformal.py ──► τ̂             （部署→蒸馏 value head→复评，
                  └── 裸考任务只验证覆盖率                      输出 R0→R3 曲线 rounds.json）

计算量缩放实验（W9）：run_eval.py --candidates {1,2,4,8} ──► scaling 曲线数据
```
