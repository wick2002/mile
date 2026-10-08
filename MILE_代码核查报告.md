# MILE 代码核查报告

> 核查时间：2026-09-23（会话续接）；**§2-A/B/C/D 已于 2026-10-08 修复并验证**
> 范围：`mile/` 全部 24 个 Python 文件 + `configs/default.yaml` + 3 份文档
> 结论：本轮计划内的修改全部落地并通过编译；§2-A–D 四个缺陷已修复并通过零依赖行为测试；剩余项见 §3。

---

## 0. 总体判定

| 项目 | 状态 |
|---|---|
| 语法编译（`compileall mile`） | ✅ exit 0，全部通过 |
| `/v1/infer` 迁移一致性 | ✅ 6 处全部同步，无残留旧主路径 |
| RoboDojo 25/50 执行协议 | ✅ 全链路接线一致 |
| MEM 20 帧历史 @1FPS | ✅ 已接线 |
| Plan A+ 三杠杆（保形 / 计算量 scaling / 自改进） | ✅ 代码齐备 |
| 部署文件（pyproject / requirements / LICENSE / .gitignore） | ✅ 齐备 |
| **逻辑正确性** | ✅ §2-A / §2-B 已修复（2026-10-08） |
| 真机 re-approach 回退变体 | ❌ 未实现 |
| Mondrian 分组保形 | ❌ 未实现 |
| 服务器端桩函数 | ⚠️ 7 处 TODO / 4 处 NotImplementedError |

---

## 1. 已完成的修改（逐文件核对）

| 文件 | 关键内容 | 状态 |
|---|---|---|
| `configs/default.yaml` | `api_path: /v1/infer`、`chunk_len: 50` / `exec_horizon: 25`、`history_frames: 20`、`robot_type` 占位、`conformal:` 段、`candidate_sweep: [1,2,4,8]`、`self_improve:` 段 | ✅ |
| `mile/base/dm05_client.py` | JSON 请求体、1-based 图像槽位 `{"1","2","3"}`、`history_images`(oldest→newest)、`sampling.seed`、`num_steps` 可选、`last_latency_ms` 抓取 `[50,14]` 形状校验、`reset_history()`、`FeatureHook` | ✅ |
| `mile/controller/controller.py` | Algorithm 1；`exec_horizon` 只想象被执行前缀 `[:, :25, :]`；拒绝→回退；one-way 阶段；恢复段录制钩子 | ✅ |
| `mile/controller/evaluate.py` | `run_episode(..., exec_horizon)` 只执行前 25 步；`record_step`；`reset_history()`；95% CI | ✅ |
| `mile/controller/conformal.py` | `conformal_quantile`（有限样本修正 `⌈(1−α)(n+1)⌉`）、`calibrate_tau`、`empirical_coverage`、拒绝一切时的降级 | ✅ |
| `mile/controller/recovery.py` | `RecoverySegment`、`value_distill_batch`、`distill_value_head`（只解冻 `value_head`，走 `value_path`） | ✅ |
| `mile/controller/stack.py` | `VerifiedEntry` + 物理快照、`max_pop`、`exhausted` | ⚠️ 见 §2-A |
| `mile/controller/scoring.py` | `J = cos + λ·v − η·u`、`pick_best(τ)`、`milestone_reached(r)` | ✅ |
| `mile/scripts/calibrate.py` | 已改为 split-conformal（取代网格搜索），held-out 只做覆盖验证 | ✅（数据采集为桩） |
| `mile/scripts/run_eval.py` | `--candidates`（scaling 扫描）、`--recovery-buffer`、`window` 构建 WM、`exec_horizon` 透传 | ✅（env_factory 为桩） |
| `mile/scripts/run_rounds.py` | R0→R3 自改进轮次、`window` 构建 | ✅（部署/评测为桩） |
| `mile/wm/model.py` | 4 层 pre-LN Transformer、3 头集成 `uncertainty`、`value_head`、`value_path`、`wm_loss`、`sample_patch_mask` | ✅ |
| `mile/wm/train.py` | `window` 读取逻辑、cosine+warmup（`train.py:18-21`）、bf16 | ✅ |
| `mile/wm/data.py` | 滑窗数据集、`make_dataloader(chunk_len=window)` | ✅ |
| `mile/env/robodojo_env.py` | `save_state` / `restore_state`（含 5 步 settle） | ⚠️ `_make_env` 为桩 |
| `docs/SETUP_zh.md` | 2026-10-08 重写版：`/v1/infer` 契约表、官方协议表、W1 五项冒烟、S1–S5、TODO 表 | ✅ |
| `docs/CODE_GUIDE_zh.md` | 已同步 Plan A+ 与 `/v1/infer` | ✅ |
| `README.md` / `requirements.txt` / `pyproject.toml` / `LICENSE` / `.gitignore` | 齐备 | ✅ |

### 关键一致性验证

```
configs: chunk_len=50, exec_horizon=25, world_model.window=25
controller.py:122  np.stack(candidates)[:, : cfg.exec_horizon, :]
evaluate.py:51     for a in chunk[:exec_horizon]
train.py:34        window = wm_cfg.get("window", base.get("exec_horizon", chunk_len))
run_eval.py:91     exec_horizon=cfg["base"].get("exec_horizon", chunk_len)
```
→ 想象窗口 == 实际执行步数，`ẑ_{t+25}` 与执行后真实 `z_{t+25}` 严格可比。✅

---

## 2. 发现的缺陷（按严重度）

### A. 【严重·已修复 2026-10-08】`max_pop` 完全失效 → 回退可能空转整局

`stack.py` 的 `_pops_this_episode` **只**在 `pop()` 里递增，但 `controller.rollback()`（第 179 行）只调用 `stack.rollback_target()`，**从不调用 `pop()`**：

```python
# controller.py:179
entry = self.stack.rollback_target()      # 永远返回同一个 _entries[-1]
```
```python
# stack.py:43-48
def rollback_target(self):
    if self._pops_this_episode >= self.max_pop or not self._entries:
        return None                        # 永远为 0 → 此分支永不触发
    return self._entries[-1]
```

**后果**：一局内第 3、4…N 次回退仍返回同一锚点，`exhausted` 永远 False，episode 会一路空转到 `max_decisions=40` 才结束。既烧评测预算，又会让论文里"平均回退次数 / abort 率"统计失真——这是审稿人一定会看的数。

**修法（已实施）**：
1. `controller.py::rollback()` 取到锚点后调用 `self.stack.pop()`，并区分三种 abort 原因（one-way / no verified anchor / max_pop exceeded），返回新增字段 `depth`（本次连续回退的第几层，供论文统计）。
2. `stack.py` 新增 `pops_this_episode` 属性与 `reset_pops()`；`controller.verify()` 在里程碑验证成功时调用 `reset_pops()` —— 让 `max_pop` 严格对应配置注释里的"**连续**回退最多弹栈层数"：有进展就恢复预算，不会误杀正在推进的 episode。
3. `stack.py::rollback_target()` 加显式注释，警告调用方必须 `pop()`。

**验证**（零依赖桩测试，未安装任何包）：
```
start: stack_len=3 next_rank=3
rollback#1: event=rollback_to  rank=2 depth=1 stack_len=2 pops=1
rollback#2: event=rollback_to  rank=1 depth=2 stack_len=1 pops=2
rollback#3: event=abort  reason=max_pop exceeded   ← 不再空转
after progress: exhausted=False pops=0 → 下一次 rollback 恢复为 depth=1
empty stack → abort(no verified anchor)；one-way 阶段仍优先 abort
```

### B. 【中等·已修复 2026-10-08】`hash()` 做 episode seed → 评测不可复现

`evaluate.py:91`：
```python
episode_seed=hash((task_id, ep)) % (2**31)
```
Python 的 `str.__hash__` 默认受 `PYTHONHASHSEED` 随机化影响，**同一条命令两次运行种子不同**。50 局 × 42 任务的评测结果会飘，直接违背论文可复现性声明，也让 ablation 之间不可比。

**修法（已实施）**：`evaluate.py` 新增 `stable_seed(task_id, ep)`（基于 `zlib.crc32`，跨进程稳定），替换原 `hash((task_id, ep))`。

**验证**：`PYTHONHASHSEED` 取 0 / 1 / 42 / 12345 / random 五种取值，`stable_seed('task_01', 3)` 恒为 `487359246`；同一任务 50 个 episode 种子互不重复。

### C. 【轻微·已处理 2026-10-08】`set_randomized` 的调用顺序是隐式契约

> **更正**：初版报告此处诊断方向有误。实际代码是 `set_randomized` **在** `reset()` **之前**（`evaluate.py:100` → `run_episode` 内 `reset`），对"场景在 reset 内生成"的语义而言**顺序本就是正确的**，不存在被覆盖的问题。真正的问题在于这是一个**隐式契约**：约定由调用方负责，而 `run_episode` 自己不设——`run_rounds.py` 的部署局（TODO）若直接调用它，就会沿用上一局遗留的 randomized 状态，静默污染 25 标准 / 25 随机的划分统计。

**修法（已实施）**：把 `randomized` 提升为 `run_episode` 的显式参数，在函数内部于 `reset()` **之前**调用 `env.set_randomized(...)`，并在 docstring 里写明"必须在 reset 之前"的原因；返回值增加 `result["randomized"]` 便于逐局核对。`evaluate()` 不再自己设。`CODE_GUIDE_zh.md:166` 的签名说明已同步。

**验证**：
```
randomized=False -> ['set_randomized(False)', 'reset(seed=801765195)']  scene=standard
randomized=True  -> ['set_randomized(True)',  'reset(seed=801765195)']  scene=randomized
```

### D. 【轻微·已修复 2026-10-08】`uncertainty` 形状注释错误

`model.py:29` 注释写 `[B, P]`，实际 `preds.var(dim=0)`（`preds` 为 `[E, B, P, vlm]`）得 `[B, P, vlm_dim]`；`controller.py` 的 `.mean(dim=(1, 2))` 是按 `[B, P, vlm]` 写的，**代码对、注释错**。

已改为 `[B, P, vlm_dim]`，并加一行警告：这是逐 patch 逐维度的集成方差，**不要**被"修正"成 `[B, P]`——调用方需自行归约（`controller.py` 用 `.mean(dim=(1,2)) -> [B]`）。

### E. 【健壮性】`restore_state` 用 qpos 当零动作

`robodojo_env.py:56` `self.env.step(np.zeros_like(state["qpos"]))` 恰好在 ARX X5（14 维）下成立，换个 embodiment 就会维度错。建议显式 `np.zeros(self.action_dim)`。

---

## 3. 未实现项清单

### 3.1 设计文档有、代码无（论文风险项）

| 项 | 位置 | 影响 |
|---|---|---|
| **re-approach 回退变体**（真机无 state restore） | `controller.py` 无 | 审稿人第一攻击点："回退特权"只在仿真成立。必须有"回退 = 重新接近锚点姿态"的对照实现，否则真机可扩展性存疑 |
| **Mondrian / 分组条件保形** | `conformal.py` 只有 marginal | 跨任务/跨阶段异质性下 marginal 覆盖会失衡；外部评审已列为准入项 |
| **修正后的成功率下界**（正文 `M_kδ_k`、附录 `δ_k^{M_k}`） | 仅文档，无代码 | 论文公式与代码需对应 |

### 3.2 服务器端桩（必须在服务器上补）

| # | 位置 | 内容 |
|---|---|---|
| 1 | `env/robodojo_env.py:27` | `_make_env` → 接官方 RoboDojo/Isaac Lab registry |
| 2 | `scripts/calibrate.py:29` | `collect_calibration_pairs` → 裸基座 rollout 收集 (J, d) |
| 3 | `scripts/run_eval.py:99` | `env_factory` → 任务注册表 |
| 4 | `base/dm05_client.py:55` | `robot_type` 精确字符串（查 `norm_stats.json`） |
| 5 | `base/dm05_client.py:105` | 历史帧 1FPS 降采样（若决策频率 > 1Hz） |
| 6 | `scripts/run_rounds.py:54` | 每轮 held-out 评测解析 |
| 7 | `scripts/run_rounds.py:64` | 每轮 500 局部署收集 |
| 8 | `scripts/run_rounds.py:79` | LoRA 蒸馏变体（可选 ablation） |

### 3.3 缺失文件

- **`mile/scripts/prepare_latents.py`**：`SETUP_zh.md` 提到需要离线抽取 VLM patch tokens 落盘供 WM 训练，但目录里**没有这个脚本**。这是 S2 的前置，必须补。

### 3.4 工程状态

- `mile/` **不是 git 仓库**（`git rev-parse` 失败）→ GitHub 推送仍阻塞。需要先 `git init` + 建远端 `wick2002/mile`。

---

## 4. 修复进度

- [x] §2-A `max_pop` 失效（2026-10-08，已验证）
- [x] §2-B episode seed 不可复现（2026-10-08，已验证）
- [x] §2-C `randomized` 顺序改为显式契约（2026-10-08，已验证；原诊断方向有误，见 §2-C 更正）
- [x] §2-D `uncertainty` 注释形状（2026-10-08，已验证）
- [ ] §2-E `restore_state` 零动作维度
- [ ] 补 `mile/scripts/prepare_latents.py`（S2 前置）
- [ ] Mondrian / 分组条件保形
- [ ] re-approach 回退变体
