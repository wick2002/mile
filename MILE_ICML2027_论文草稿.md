# Think, Verify, Retreat: Milestone-Grounded Closed-Loop Imagination for Long-Horizon Vision-Language-Action Policies

> **草稿说明（投稿前删除此段）**：本文档为基于研究计划方案 A 的 ICML 2027 论文骨架草稿。所有标为 `⟨TODO⟩` 的内容需要在实验完成后替换为真实数字；§5 的表格结构已按 RoboDojo 官方协议排好，只需填数。写作约定：method 章节以"基座无关的 wrapper"叙事展开，主基座写 DM0.5，第二基座写 π0。

---

**Anonymous Author(s)**
**Affiliation, Address**
**email@domain**

---

## Abstract

Long-horizon manipulation remains the weakest capability of modern Vision-Language-Action (VLA) policies: on the recent RoboDojo benchmark, the strongest generalist policy completes only ⟨TODO⟩% of long-horizon tasks, and detailed scoring reveals that partial sub-steps are frequently achieved while full tasks are not — evidence that errors compound across steps rather than arise from a single missing skill. We argue that the root cause is *open-loop execution*: VLA policies map observations to action chunks reactively, without predicting, verifying, or recovering from the consequences of their own actions. We propose **MILE** (**MI**lestone-grounded closed-**L**oop imagination and **E**xecution), a model-based wrapper that turns any chunked VLA policy into a closed-loop agent. MILE equips a frozen VLA with three lightweight components: (i) a *milestone extractor* that represents a long-horizon task as a sparse sequence of physically grounded milestone latents, discovered automatically from kinematic and visual cues without sub-task annotation; (ii) a *predictive world model* operating in the VLA's feature space, which verifies candidate action chunks by imagined short-horizon rollouts scored by a learned steps-to-completion value head; and (iii) a *safe-anchor rollback* mechanism that maintains a stack of verified milestones and retreats to the most recent anchor whenever executed trajectories deviate from imagination. We prove that milestone-level re-anchoring converts the multiplicative compounding-error bound of behavior cloning into an additive one over verified segments. On RoboDojo, MILE improves long-horizon ⟨TODO metric⟩ from ⟨TODO⟩ to ⟨TODO⟩ (⟨TODO⟩% relative) over the DM0.5 base policy, with consistent gains on memory tasks and on randomized scenes, while adding only ⟨TODO⟩M parameters and ⟨TODO⟩ms of latency per decision. Code, models, and evaluation scripts will be released.

---

## 1. Introduction

Vision-Language-Action (VLA) models have scaled rapidly in the past two years, and on narrowly scoped benchmarks such as LIBERO the strongest policies now exceed 98% success (⟨cite⟩). Recent comprehensive evaluations, however, tell a different story. On RoboDojo (⟨cite⟩), which measures 30 policies across 42 simulation tasks in five capability dimensions, the best policy achieves an average success rate of only ⟨TODO⟩% — versus 76.03% for human experts — and the gap is largest precisely in *long-horizon execution*. More telling than the aggregate numbers is the pattern: on long-horizon tasks, policies score substantially higher on intermediate sub-step credit than on full-task success. Robots can pick the cup, and they can stack the cup, but they cannot reliably *pick and then stack*.

This pattern is the signature of **compounding error**: each step of a behavior-cloned policy introduces a small deviation, and deviations multiply over a horizon of dozens to hundreds of steps. Existing remedies attack individual links of this chain — better backbones (⟨cite⟩), better action decoders (⟨cite⟩), more data (⟨cite⟩) — but leave the execution protocol untouched: the policy still acts open-loop, committing to each action chunk with no mechanism to anticipate consequences, detect drift, or recover.

We argue for a complementary view: **long-horizon failure is a systems problem, not a capacity problem**. A robust long-horizon agent must (a) *imagine* what its action will change, (b) *verify* the imagination against task progress before committing, and (c) *retreat* to a known-good state when execution deviates from plan. None of the four current world-model paradigms for VLA implements this loop completely: world-model post-training uses dynamics only at training time; world-model-as-planner predicts but never verifies; world-model-as-simulator optimizes in imagination but does not correct execution; event-triggered correctors intervene only at isolated decision points such as pre-grasp.

We introduce **MILE** (Milestone-grounded closed-Loop imagination and Execution), a wrapper that closes this loop at the granularity of **physical milestones** — sparse, automatically discovered events (gripper transitions, contact onsets, kinematic turning points) that segment a long-horizon task into short verified stages. Milestones are the right unit for closing the loop: they are frequent enough to truncate error accumulation, sparse enough to keep verification cheap, and physically grounded enough to be predicted by a lightweight world model operating in the VLA's own feature space.

**Contributions.**
1. **Milestone representation without annotation.** We present an automatic extractor that discovers sparse, physically meaningful milestones from demonstration trajectories using kinematic and visual-change cues, and show that milestone latents provide a compact, execution-aligned representation of long-horizon tasks.
2. **Verified execution.** A lightweight world model (⟨TODO⟩M parameters) predicts milestone-conditioned futures in the frozen VLA's feature space; candidate action chunks are ranked by imagined progress (a steps-to-completion value head) before commitment.
3. **Safe-anchor rollback.** A stack of verified milestones defines *heal states* explicitly; deviation beyond a calibrated threshold triggers retreat-and-replan, converting catastrophic task failure into bounded local re-attempts. We show theoretically that this converts the multiplicative compounding-error bound of behavior cloning into an additive bound over verified segments.
4. **Comprehensive evaluation.** On RoboDojo (long-horizon and memory dimensions), MILE improves ⟨TODO⟩ by ⟨TODO⟩% relative over DM0.5 and transfers to π0 unchanged, with ablations isolating the contribution of verification and rollback, and a failure-taxonomy analysis showing which long-horizon failure modes are repaired.

---

## 2. Related Work

**Vision-Language-Action models.** VLA policies extend vision-language models with action tokens or diffusion/flow-matching action decoders (⟨cite: RT-2, OpenVLA, π0, UniVLA, GR00T, DM0.5⟩). Despite architectural diversity, the execution protocol is shared: reactive, open-loop action chunking conditioned on current observation and instruction. Our work is orthogonal — MILE is a base-agnostic wrapper, and we demonstrate it on two bases (DM0.5, π0).

**World models as training objectives.** UniVLA (⟨cite⟩) and CauVA (⟨cite⟩) show that causal-dynamics modeling is a strong post-training objective for VLAs, especially for long-horizon tasks. These methods internalize dynamics into the policy but do not *use* prediction at execution time; MILE is complementary and can sit on top of such backbones.

**World models as planners.** StructVLA (⟨cite⟩) replaces dense rollouts with sparse structured frames, and Act2Goal (⟨cite⟩) generates intermediate visual states for goal-conditioned control; AHEAD (⟨cite⟩) predicts future latents for dynamic grasping with a frozen VLA. All predict, none verify: an erroneous imagination steers execution astray. MILE's milestone latents borrow StructVLA's insight (sparsity with kinematic grounding) but serve a closed-loop verification role rather than a one-shot planning role.

**World models as simulators.** WoVR (⟨cite⟩), RISE (⟨cite⟩), and GigaBrain (⟨cite⟩) optimize policies inside learned imagination, with WoVR's keyframe-initialized rollouts mitigating hallucination. Our value head follows GigaBrain's finding that jointly predicting future states and steps-to-completion outperforms either objective alone; unlike these methods, MILE performs RL-free, inference-time verification and needs no on-policy imagined training.

**Execution-time correction.** World-in-Loop (⟨cite⟩) triggers a corrector at isolated events (e.g., pre-grasp) and rolls back to a heuristic heal state; RB-VLA (⟨cite⟩) maintains a persistent belief state. MILE generalizes both: verification is triggered at every milestone boundary (a full coverage of task stages), and rollback targets are defined *structurally* by the verified-milestone stack rather than heuristically.

**Long-horizon evaluation.** RoboDojo (⟨cite⟩) provides the first unified sim-and-real protocol with a public leaderboard, five capability dimensions, and per-sub-step scoring; its finding that long-horizon *scores* exceed long-horizon *success rates* is direct evidence of compounding error and motivates our design. ⟨cite: LIBERO-X, CALVIN⟩ serve as supplementary benchmarks.

---

## 3. Preliminaries and Problem Statement

**Setting.** A language-conditioned manipulation task is a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, \rho, l)$ with instruction $l$. A chunked VLA policy $\pi_\theta$ maps an observation $o_t$ (multi-view RGB, proprioception) and $l$ to an action chunk $a_{t:t+H} = \pi_\theta(o_t, l)$, executed for $H$ steps before re-querying. Long-horizon tasks decompose into $K$ semantically ordered stages; success requires all stages completed in order.

**Compounding error.** Let $\epsilon$ be the per-step distribution-shift error of behavior cloning. Standard analysis (⟨cite⟩) bounds the total-variation divergence between $\pi_\theta$ and expert state distributions by $O(H_{\mathrm{task}} \cdot \epsilon)$ (linear in the task horizon), with *success* (a conjunction over all stages) decaying multiplicatively: $P(\mathrm{success}) \le \prod_{k=1}^{K}(1 - h_k \epsilon) \approx 1 - \epsilon \sum_k h_k$. For long horizons, $h_k \epsilon \ll 1$ fails and success collapses. Section 4.4 shows how verified re-anchoring truncates each $h_k$ to a bounded, verified segment length.

---

## 4. MILE: Milestone-Grounded Closed-Loop Imagination

MILE wraps a frozen VLA $\pi_\theta$ with three components: a milestone extractor $\mathcal{M}$, a milestone-conditioned world model $p_\phi$, and an execution controller implementing verify-then-act with safe-anchor rollback. Figure ⟨TODO 1⟩ gives an overview.

### 4.1 Automatic milestone extraction

Given a demonstration trajectory $\tau = (o_0, a_0, \dots, o_T)$, we extract a sparse milestone index set $\mathcal{M}(\tau) = \{t_1, \dots, t_K\}$ from three complementary cues:

1. **Kinematic events** $c^{\mathrm{kin}}_t$: gripper aperture transitions (open↔close crossings), joint-velocity turning points (sign changes in the dominant principal component of end-effector velocity).
2. **Contact/progress events** $c^{\mathrm{con}}_t$: sharp changes in predicted contact probability (from a small probe trained on proprioception, or force readings where available).
3. **Visual-change events** $c^{\mathrm{vis}}_t$: local maxima of frame-to-frame latent distance $\|f(o_{t}) - f(o_{t-1})\|$ in the frozen VLA's visual encoder $f$, restricted to instruction-relevant regions via the VLA's cross-attention map.

Candidate events are pooled with a minimum separation $\Delta$ and ranked by a consensus score $s_t = \alpha c^{\mathrm{kin}}_t + \beta c^{\mathrm{con}}_t + \gamma c^{\mathrm{vis}}_t$; the top-$K$ indices (with $K$ estimated per task family from demonstrations) define milestones. Each milestone $m_k$ is represented as a latent $\mathbf{z}_k = f(o_{t_k})$ plus its stage descriptor $\ell_k$ (from the instruction template, e.g., "*after picking X*"). No human sub-task annotation is required; when task intermediate-step signals are available (as in RoboDojo's sub-step scoring), they are used only as weak supervision to calibrate thresholds $(\alpha, \beta, \gamma, \Delta)$.

### 4.2 Milestone-conditioned world model and progress value

The world model $p_\phi$ predicts future feature-space states conditioned on the current observation embedding and a candidate action chunk:
$$\hat{\mathbf{z}}_{t+k} \sim p_\phi(\cdot \mid \mathbf{z}_t, a_{t:t+k}, \ell_{k+1}).$$
We implement $p_\phi$ as a ⟨TODO: 4-layer / d=1024⟩ Transformer with per-token motion conditioning (⟨AHEAD-style⟩), trained with masked latent prediction on demonstration data plus ⟨TODO: Open X-Embodiment subset⟩. Following ⟨GigaBrain⟩, $p_\phi$ shares a backbone with a **progress value head** $v_\psi(\mathbf{z}, \ell_K)$ predicting normalized steps-to-completion; joint training of prediction and value is essential (ablation in §5.3).

### 4.3 Verified execution with safe-anchor rollback

At each decision point, the controller operates as follows (Algorithm ⟨TODO 1⟩):

1. **Milestone tracking.** Maintain a stack $\mathcal{V}$ of *verified* milestones. The next target is $\mathbf{z}_{k+1}$, where $m_k = \mathrm{top}(\mathcal{V})$.
2. **Candidate generation.** Sample $C$ candidate chunks from the (stochastic) base policy: $\{a^{(c)}_{t:t+H}\}_{c=1}^{C} \sim \pi_\theta(o_t, l)$.
3. **Imagination.** Roll out $p_\phi$ for each candidate to obtain $\hat{\mathbf{z}}^{(c)}_{t+H}$.
4. **Verification scoring.** Rank candidates by
$$J(c) = \underbrace{\cos\big(\hat{\mathbf{z}}^{(c)}_{t+H},\, \mathbf{z}_{k+1}\big)}_{\text{milestone alignment}} + \lambda \underbrace{\big[v_\psi(\mathbf{z}_t) - v_\psi(\hat{\mathbf{z}}^{(c)}_{t+H})\big]}_{\text{progress}} - \eta \underbrace{u_\phi(\hat{\mathbf{z}}^{(c)}_{t+H})}_{\text{uncertainty}},$$
where $u_\phi$ is an ensemble/disagreement-based uncertainty estimate. Execute $\arg\max_c J(c)$ only if $J(c^\ast) > \tau_J$ (threshold calibrated on validation rollouts).
5. **Advance or rollback.** After execution, if the realized latent is within radius $r$ of a verified milestone, push it onto $\mathcal{V}$. If the realized latent deviates from the predicted next milestone beyond radius $r$ for $n$ consecutive decisions, **retreat**: reset (physically or via the best re-approach policy) to the state of $\mathrm{top}(\mathcal{V})$, clear unverified suffix, and re-plan from that anchor.

The full controller adds no retraining of $\pi_\theta$: only $p_\phi$, $v_\psi$, and the extractor thresholds are learned (⟨TODO⟩M parameters total).

### 4.4 Theory: verified re-anchoring truncates error compounding

**Proposition 4.1 (informal).** Let the task be segmented into $K$ stages by milestones, with per-step shift error $\epsilon$ and stage length $h$. Suppose the verifier accepts an executed segment only if its terminal-state deviation from the target milestone is below radius $r$, and rollback restores a state within the $r$-neighborhood of the last verified milestone with probability at least $1-\delta$. Then task success is bounded below by
$$P(\mathrm{success}) \;\ge\; \prod_{k=1}^{K} \Big[\, (1-\delta)\, P\big(\text{segment } k \text{ accepted}\big) \Big], \quad \text{with} \quad P(\text{accepted}) \ge 1 - \frac{h \epsilon}{r}\ \text{(first-order)},$$
so that per-stage error enters **additively relative to the acceptance radius** rather than compounding multiplicatively across the full horizon $Kh$. A naive open-loop policy has $r = \infty$ (no truncation), recovering the standard bound.

*Proof sketch.* Induction over stages with a union bound over verification failures; deviation within an accepted segment is a martingale with bounded increments, so terminal deviation concentrates around $h\epsilon$; acceptance at radius $r$ truncates the tail. Full proof in Appendix ⟨TODO⟩.

**Remark.** The proposition makes the design trade-offs explicit: smaller $r$ (stricter verification) tightens the bound but increases rollback frequency and latency; milestone sparsity controls $h$. Section 5.3 ablates both.

---

## 5. Experiments

### 5.1 Setup

**Benchmark.** RoboDojo simulation suite: 42 tasks in five dimensions (generalization 12, memory 6, precision 8, long-horizon 8, open-semantic 8), 50 episodes per task (25 standard + 25 randomized scenes), official sub-step–sensitive score and success rate, co-train (clean2random) protocol with 50 clean demonstrations per task. Training data: official 3,500 demonstration trajectories (1.86M frames).

**Base policies.** **DM0.5** (primary; leaderboard top, ⟨TODO params⟩) and **π0** (secondary; cross-base validation). For each base we compare: (i) base policy as-is; (ii) base + MILE; (iii) base + MILE − verification; (iv) base + MILE − rollback.

**Baselines.** Public leaderboard numbers for DM0.5, GalaxeaVLA, Hy-Embodied-0.5-VLA, π0.5, X-VLA; plus ⟨TODO: World-in-Loop / StructVLA adapted as wrapper baselines, if reproducible⟩.

**MILE configuration.** $C = ⟨TODO⟩$ candidates, $\lambda = ⟨TODO⟩$, $\eta = ⟨TODO⟩$, $r = ⟨TODO⟩$, $\tau_J = ⟨TODO⟩$, $n = ⟨TODO⟩$. $p_\phi$/$v_\psi$: ⟨TODO⟩M params, trained ⟨TODO⟩ steps on ⟨TODO⟩ GPUs.

### 5.2 Main results

**Table 1. RoboDojo long-horizon and memory dimensions (score / success %).** ⟨TODO: fill⟩

| Method | Long-horizon (std) | Long-horizon (rand) | Memory (std) | Memory (rand) |
|---|---|---|---|---|
| DM0.5 | ⟨33.70/19.50⟩ | ⟨TODO⟩ | ⟨47.74⟩ | ⟨TODO⟩ |
| DM0.5 + MILE | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| π0 | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| π0 + MILE | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| GalaxeaVLA (leaderboard) | 44.12/32.25 | — | ⟨TODO⟩ | — |
| π0.5 (leaderboard) | 23.54/14.67 | — | ⟨TODO⟩ | — |

**Table 2. All five dimensions, average score / success % (DM0.5 base).** ⟨TODO: fill⟩

**Key findings to report:**
- MILE's improvement concentrates on long-horizon and memory dimensions, with no degradation on precision/generalization — evidence that gains come from the execution loop, not from better perception. ⟨TODO⟩
- Gains are *larger on randomized scenes* than standard ones, consistent with the hypothesis that re-anchoring mitigates distribution-shift accumulation. ⟨TODO⟩
- The wrapper transfers from DM0.5 to π0 with only extractor-threshold recalibration. ⟨TODO⟩

### 5.3 Ablations

**Table 3. Ablations (long-horizon dimension, DM0.5 base).** ⟨TODO: fill⟩

| Variant | Score | Success | Latency/decision |
|---|---|---|---|
| Full MILE | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| − verification ($C{=}1$, no scoring) | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| − rollback | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| − milestones (dense rollout horizon $H'$) | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| − joint value training (separate heads) | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Rule-only extractor vs. learned | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Acceptance radius $r \in \{r/2, r, 2r\}$ | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |
| Milestone stack depth | ⟨TODO⟩ | ⟨TODO⟩ | ⟨TODO⟩ |

**Figure ⟨TODO 2⟩.** Latency–success trade-off: MILE vs. World-in-Loop (event-triggered) vs. WoVR-style full rollouts. MILE's milestone-level triggering achieves ⟨TODO⟩% of the rollback benefit at ⟨TODO⟩× lower latency.

### 5.4 Failure taxonomy analysis

Following ⟨RoboDojo's analysis protocol⟩, we annotate ⟨TODO: 300⟩ failure episodes into four classes: *perceptual aliasing* (progress loss), *execution drift*, *stage-transition failure*, *recovery absence*. Table ⟨TODO 4⟩ reports per-class failure rates for DM0.5 vs. DM0.5+MILE: MILE reduces recovery-absence failures by ⟨TODO⟩% and stage-transition failures by ⟨TODO⟩%, but leaves ⟨TODO⟩ largely unchanged — pointing to perception-side limits of a frozen base.

### 5.5 Real-world evaluation (optional, rebuttal-ready)

Via RoboDojo-RealEval remote evaluation: 18 tasks on ARX X5 / Piper / Piper X, 10 trials per task, blind scoring. ⟨TODO: fill — submitted W10, expected back by ⟨date⟩⟩.

---

## 6. Limitations

(i) The world model inherits the base's perception; if $f$ fails to distinguish two states (strong perceptual aliasing), verification cannot rescue execution (§5.4). (ii) Rollback assumes stages are physically reversible or re-approachable; irreversible operations (cutting, pouring) need one-way milestone flags — we currently exclude ⟨TODO⟩ such tasks. (iii) Verification latency, though small (⟨TODO⟩ms), may matter in highly dynamic scenes; adaptive candidate budgets (⟨our Plan B variant⟩) are future work. (iv) Absolute success rates on RoboDojo remain far below human level; MILE narrows but does not close the gap.

---

## 7. Conclusion

We presented MILE, a model-based wrapper that turns open-loop VLA policies into closed-loop long-horizon agents by grounding imagination in automatically discovered physical milestones, verifying action chunks before commitment, and retreating to safe anchors on deviation. Both theory (additive vs. multiplicative error bounds) and experiments (RoboDojo long-horizon/memory gains on two bases) support the central claim: *long-horizon failure is largely a systems problem, and closing the loop at the right granularity is a cheap and general way to fix it.*

---

## References ⟨TODO: BibTeX⟩

- RoboDojo (arXiv 2607.04434) — benchmark & six findings
- DM0.5 technical report ⟨TODO link⟩ — primary base
- UniVLA (ICLR 2026), CauVA/LingBot-VA (RSS 2026), CronusVLA (AAAI 2026) — route ①
- StructVLA (arXiv 2603.12553), Act2Goal (RSS 2026), AHEAD (arXiv 2606.02486) — route ②
- WoVR (arXiv 2602.13977), RISE (RSS 2026), GigaBrain-0.5M (arXiv 2026) — route ③
- World-in-Loop (ECCV 2026), RB-VLA (arXiv 2026) — route ④
- RT-2, OpenVLA, π0/π0.5, GR00T, X-VLA, GalaxeaVLA, Hy-Embodied, InternVLA — bases/baselines
- LIBERO-X, CALVIN — supplementary benchmarks
- Ross & Bagnell (DAgger), compounding error analysis

---

## Appendix ⟨TODO⟩

- A. Full proof of Proposition 4.1
- B. Milestone extractor details and extraction-quality statistics on RoboDojo demonstrations (precision/recall vs. intermediate-step signals)
- C. World model architecture, training curves, prediction visualizations
- D. Threshold calibration protocol ($r, \tau_J, \lambda, \eta, n, C$)
- E. Full 42-task per-task results
- F. Evaluation noise analysis (per-task 50-episode confidence intervals)
- G. Reproducibility statement and code release

---

### 使用与填数指南（投稿前删除本节）

| 待办 | 依赖 | 预计完成 |
|---|---|---|
| 替换所有 `⟨TODO⟩` 实验数字 | W6–W10 实验 | 2026-12 |
| Algorithm 1 伪代码框 | 控制器实现定稿（W7） | 2026-11 |
| Figure 1 系统总览图 | 方法定稿 | 2026-11 |
| Figure 2 latency–success 曲线 | W9 消融 | 2026-11 |
| Proposition 4.1 完整证明（Appendix A） | W11 理论小节 | 2026-12 |
| RealEval 真机结果（§5.5） | W10 提交排队 | 视排队情况 |
| BibTeX 参考文献 | 写作阶段 | 2026-12 |
| 匿名化 + 复现性附录 | W15 | 2027-01 |
