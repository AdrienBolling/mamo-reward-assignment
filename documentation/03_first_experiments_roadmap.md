# First Experiments for Factorized Credit Assignment in Craftax / MA-Craftax

**Experimental roadmap — v0.1**  
**Goal:** test the mechanism before building the ambitious architecture.

---

## 1. Experimental principle

The first question is **not**:

> Does a three-head architecture score higher?

The first question is:

> Is there measurable evidence that dense, sparse, and terminal reward channels induce different and interfering learning signals?

If the answer is no, the architectural premise should be revised.

The experiment sequence therefore proceeds from **diagnosis → minimal intervention → architecture → temporal/spatial attribution**.

---

# 2. Phase 0 — define the reward channels

Before training anything, implement an explicit reward decomposition interface:

```text
reward_components = {
    "dense": r_dense,
    "sparse": r_sparse,
    "final": r_final,
}
```

Require:

\[
r_t^{\mathrm{env}}
=
r_t^d+r_t^s+r_t^f
\]

where possible, or record clearly if the research reward decomposition differs from the original environment reward.

## 2.1 Initial operational definitions

The exact Craftax mechanics must be inspected before finalizing these.

### Dense
Candidates:
- health change;
- hunger/thirst maintenance signals if present;
- other frequent per-step survival/resource signals.

### Sparse
Candidates:
- first-time achievements;
- crafting milestones;
- resource/biome/progression events.

### Final
Candidates:
- explicit episode-end success signal if available;
- a designated high-level objective or terminal score;
- alternatively, a constructed evaluation-only terminal return equal to an end-of-episode summary of selected achievements.

### Important warning
Craftax may not naturally expose a clean “terminal reward” component matching the conceptual category. Do not force the environment into an artificial taxonomy without documenting it.

A better first implementation may define three channels by empirical temporal statistics:

- frequent;
- event-sparse;
- long-delay/end-of-episode.

---

# 3. Phase 1 — gradient forensics with an unchanged baseline

## Objective
Measure channel-wise gradients without changing the baseline action policy.

Start with PPO in Craftax, then IPPO/MAPPO in MA-Craftax.

For each minibatch, compute policy losses separately:

\[
L_\pi^d,\quad
L_\pi^s,\quad
L_\pi^f.
\]

Obtain

\[
g_d = \nabla_\theta L_\pi^d,\qquad
g_s = \nabla_\theta L_\pi^s,\qquad
g_f = \nabla_\theta L_\pi^f.
\]

Do **not** step the optimizer three times. These are diagnostics.

## 3.1 Log gradient norms

\[
n_k = \|g_k\|_2.
\]

Log:
- per minibatch;
- EMA over updates;
- per layer/module;
- normalized by number of nonzero reward events;
- normalized by advantage standard deviation.

Key test:

\[
\frac{\|g_d\|}
{\|g_s\|+\epsilon},
\qquad
\frac{\|g_d\|}
{\|g_f\|+\epsilon}.
\]

### Hypothesis
Dense rewards may dominate because they are frequent, not necessarily because their directions conflict.

---

## 3.2 Log pairwise cosine similarities

\[
c_{ij}
=
\frac{g_i^\top g_j}
{\|g_i\|\|g_j\|+\epsilon}.
\]

Track:

\[
c_{ds},\quad c_{df},\quad c_{sf}.
\]

Report:
- mean;
- median;
- fraction negative;
- 10/90% quantiles;
- distribution over training;
- layer-wise distributions.

### Hypothesis
The dense-vs-final pair will exhibit stronger or more frequent conflict than dense-vs-sparse.

This is a hypothesis, not an assumption.

---

## 3.3 Log the Gram matrix

\[
K=G^\top G,
\qquad
G=[g_d,g_s,g_f].
\]

Store the normalized \(3\times3\) matrix periodically.

Useful visualizations:
- heatmap over training;
- trajectories of eigenvalues;
- condition number;
- principal direction explained variance.

Question:

> Is the credit-gradient system effectively one-dimensional, or do multiple independent directions persist?

If \(G\) is nearly rank 1, the “higher-dimensional credit space” may not add much.

---

# 4. Phase 1b — temporal localization of conflict

The same global minibatch gradient can hide when conflict occurs.

Partition trajectories into temporal windows:

- early episode;
- middle episode;
- late episode;

or by progress events:

- pre-first-tool;
- post-tool;
- pre-combat;
- post-key-achievement;
- pre-terminal phase.

Compute \(G\) and \(K\) within each bucket.

### Goal
Test whether credit geometry is state/phase dependent:

\[
K_t \neq K_{t'}.
\]

If yes, this motivates a state/history-dependent interaction operator \(C_\psi\).

---

# 5. Phase 1c — event-conditioned gradient analysis

Condition sparse/final gradients on events:

- achievement reached;
- death;
- starvation/dehydration;
- tool crafted;
- resource transfer;
- role-specific cooperation event.

Measure whether the gradient relations around these events differ from background play.

This can reveal whether a terminal signal only becomes informative in a narrow context.

---

# 6. Phase 2 — critic decomposition only

## Model
Keep one actor policy but use:

\[
V_d,\quad V_s,\quad V_f.
\]

Compute separate advantages:

\[
A_d,\quad A_s,\quad A_f.
\]

Then aggregate at the actor loss level:

\[
A^{\mathrm{agg}}
=
w_d A_d+w_s A_s+w_f A_f.
\]

## Why
This tests whether the main bottleneck is simply that one scalar critic struggles to model returns with very different temporal statistics.

## Compare
- scalar critic;
- vector critic with shared trunk;
- fully separate critics.

## Metrics
- value loss per channel;
- explained variance per channel;
- advantage variance;
- total return;
- sparse achievement coverage;
- terminal success.

### Interpretation
If this alone gives most of the gain, the actor-gradient-routing story is weaker.

---

# 7. Phase 2b — channel-specific temporal parameters

Test separate GAE/discount settings:

\[
(\gamma_d,\lambda_d),
\quad
(\gamma_s,\lambda_s),
\quad
(\gamma_f,\lambda_f).
\]

Expected qualitative direction:
- dense channel can tolerate shorter horizon;
- final channel needs a much longer effective horizon.

Do not aggressively tune all six parameters at first.

A compact ablation:
1. all channels share baseline \((\gamma,\lambda)\);
2. dense uses shorter horizon;
3. final uses longer horizon;
4. learned/meta-gradient variants only later.

---

# 8. Phase 3 — gradient aggregation baselines

Before introducing new actor architecture, compare established methods on the same channel gradients.

## 8.1 Fixed scalarization

\[
g=w_dg_d+w_sg_s+w_fg_f.
\]

## 8.2 Normalized sum

\[
g=
\sum_k
\frac{g_k}{\|g_k\|+\epsilon}.
\]

## 8.3 GradNorm-style balancing

Dynamically adjust channel weights to balance learning rates / gradient norms.

## 8.4 PCGrad

Project only when channel gradients conflict.

## 8.5 CAGrad or MGDA

Use multi-objective aggregation to find a compromise direction.

### Purpose
These baselines answer:

> Is an architectural split necessary, or is gradient aggregation sufficient?

---

# 9. Phase 4 — three actor heads, simplest version

Use a shared encoder:

\[
h=f_\phi(o).
\]

Three logit heads:

\[
z_d,\quad z_s,\quad z_f.
\]

Final logits:

\[
z=z_d+z_s+z_f.
\]

Each policy loss updates only its own actor head:

```text
L_dense  -> dense_head
L_sparse -> sparse_head
L_final  -> final_head
```

For the shared encoder, initially use summed gradients.

### Why start here
This isolates architectural specialization without adding gates.

### Important implementation check
Because all three heads affect the final policy, PPO ratios/log-probs must be computed from the **composed policy**, not from individual heads as if each were independently executed.

The channel-specific loss therefore needs careful derivation. A naive “three PPO losses with the same sampled action” may produce unintended coupling.

One safe starting point is:
- generate one composed policy;
- compute per-channel advantages;
- use separate stop-gradient routing through additive logits.

This should be verified with finite-difference/autodiff tests.

---

# 10. Phase 4b — residual architecture

Preferred architecture:

\[
z=
z_d+\alpha_s\Delta z_s+\alpha_f\Delta z_f.
\]

Initially use fixed:

\[
\alpha_s=\alpha_f=1.
\]

Then learn gates.

## Diagnostic quantities
Log:
- \(\|\Delta z_s\|\);
- \(\|\Delta z_f\|\);
- KL between baseline dense policy and composed policy;
- action rank changes induced by each residual;
- gate activations once learned.

### Key qualitative question
Does the final head intervene at strategically meaningful states rather than constantly perturbing behavior?

---

# 11. Phase 5 — learned gates

Let

\[
\alpha_s=\sigma(q_s(h)),
\quad
\alpha_f=\sigma(q_f(h)).
\]

Variants:

1. scalar gate per channel;
2. vector gate per action logit;
3. gate conditioned on recurrent history;
4. gate conditioned on credit-geometry statistics.

### Regularization
Prevent trivial always-on gates:

\[
L_{\mathrm{gate}}
=
\beta_s |\alpha_s|
+
\beta_f |\alpha_f|
\]

or constrain expected activation.

### Analysis
Correlate gates with:
- achievement proximity;
- episode phase;
- inventory state;
- agent specialization;
- food/water crisis;
- trading events.

---

# 12. Phase 6 — shared encoder treatments

Once specialist heads work, test where interaction should happen.

## Variant A — all channel gradients update encoder

Baseline.

## Variant B — PCGrad on encoder only

Keep specialist heads separate; resolve only shared-representation conflict.

## Variant C — gradient normalization on encoder

Balance channel magnitudes.

## Variant D — separate channel adapters

\[
h_k=f_\phi(o)+a_k(f_\phi(o)).
\]

Only adapter \(a_k\) receives channel \(k\)'s specialist gradient.

## Variant E — fully separate recurrent cores

Expensive but useful upper-bound ablation.

### Expected outcome
A partial-separation architecture may dominate full separation by preserving beneficial shared representations.

---

# 13. Phase 7 — explicit credit interaction matrix

Only after simpler experiments show persistent multi-dimensional structure.

Let

\[
G=[g_d,g_s,g_f].
\]

Introduce

\[
\tilde G=GC.
\]

Start with a globally learned matrix:

\[
C\in\mathbb R^{3\times3}.
\]

Then state/history dependent:

\[
C_t=C_\psi(h_t).
\]

## Constraints worth testing
- identity initialization;
- diagonal dominance;
- row/column stochasticity;
- triangular/hierarchical structure;
- low-rank \(C\);
- residual parameterization
  \[
  C=I+\Delta C.
  \]

### First objective
Do not use meta-gradients yet. Train \(C\) end-to-end through the ordinary policy objective if mathematically well-defined.

### Later
Use a genuine outer objective and meta-gradient.

---

# 14. Phase 8 — temporal attribution for sparse/final channels

If the terminal gradient remains too weak/noisy, add return decomposition.

## Variant A — RUDDER-like redistribution

Redistribute final return to earlier events.

Feed redistributed signal only to:

\[
V_f,\quad A_f,\quad \text{final actor module}.
\]

## Variant B — trajectory attention

Train a sequence model to predict final return and produce timestep contribution scores.

## Variant C — event boundary attribution

Assign final credit to high-level events/achievements rather than raw timesteps.

### Main experiment
Cross:

| Gradient factorization | Temporal redistribution |
|---|---|
| No | No |
| Yes | No |
| No | Yes |
| Yes | Yes |

This determines whether the two mechanisms are complementary.

---

# 15. Phase 9 — MARL agent credit

Move to Craftax-Coop after single-agent/MA-Craftax instrumentation is stable.

Define:

\[
A_t^{i,k}.
\]

Possible agent-credit methods:

1. individual rewards where available — diagnostic upper bound;
2. centralized critic;
3. COMA-style counterfactual baseline;
4. STAS-like spatio-temporal decomposition;
5. role-aware attribution.

### Core factorial experiment

Two binary axes:

- channel factorization: off/on;
- agent attribution: off/on.

This yields four conditions and directly tests interaction.

---

# 16. Recommended first benchmark ladder

## Ladder A — synthetic toy environment

Create a tiny MDP with:
- dense local reward that encourages a tempting behavior;
- sparse milestone reward;
- delayed terminal reward that requires temporarily sacrificing dense reward.

Purpose:
- validate code;
- make gradient conflict controllable;
- obtain ground-truth causal structure.

## Ladder B — single-agent Craftax

Use symbolic observations and a reduced compute budget for fast iteration.

## Ladder C — Craftax-MA

Start with 2 agents before scaling.

## Ladder D — Craftax-Coop

Use the full heterogeneous 3-agent setting after the method is stable.

---

# 17. A useful synthetic task

Construct a corridor/tree MDP:

At each state, agent chooses:
- **harvest**: +dense reward, little progress;
- **invest**: zero or negative dense reward, advances toward milestone;
- **commit**: only useful after milestone, eventually produces terminal reward.

Example:

\[
r_d(\text{harvest})=+0.1,
\]

milestone after \(N\) consecutive invest actions:

\[
r_s=+1,
\]

terminal success after another delayed sequence:

\[
r_f=+10.
\]

This environment should produce an interpretable conflict:

\[
g_d^\top g_f < 0
\]

early in learning.

Then verify:
- diagnostics detect it;
- PCGrad helps;
- specialist residual policy helps;
- learned gate shifts from dense behavior to long-horizon strategy.

This is much cheaper than debugging the concept first in Craftax.

---

# 18. Evaluation metrics

## Performance
- total scalarized return;
- vector return by channel;
- Craftax achievement score;
- unique achievements;
- late-game achievement rate;
- survival duration;
- terminal success probability.

## Optimization
- gradient norm per channel;
- pairwise cosine;
- negative-cosine frequency;
- Gram eigenvalues;
- gradient rank/effective rank;
- update norm per module;
- gradient variance.

## Value learning
- critic MSE;
- explained variance;
- target variance;
- TD error distribution;
- advantage variance.

## Policy interaction
- KL divergence between dense-only and composed policy;
- residual logit norm;
- action rank flips;
- gate entropy;
- gate activation by episode phase.

## MARL
- per-agent contribution metrics;
- trading count;
- role-specific achievement rate;
- team survival;
- reward under shared vs individual diagnostic setting.

---

# 19. Statistical protocol

For early debugging:
- 1–3 seeds are acceptable.

For meaningful conclusions:
- target at least 5–10 seeds depending on variance and compute.

Report:
- mean;
- bootstrap confidence intervals or standard error;
- learning curves;
- final performance;
- area under learning curve;
- seed-level scatter.

Avoid choosing one “best” training checkpoint without a predeclared rule.

---

# 20. Compute strategy

Exploit JAX vectorization aggressively.

Recommended progression:
1. symbolic observations;
2. small number of agents;
3. shortened budget for diagnostics;
4. full-scale runs only after gradient instrumentation is stable.

The MA-Craftax paper demonstrates that hundreds of millions of environment interactions are practical on a single modern GPU, but gradient-forensics instrumentation may significantly reduce throughput.

Therefore:
- log expensive full gradients periodically, not every update;
- compute per-layer summary statistics;
- consider random projections/sketches of gradients if memory becomes prohibitive.

---

# 21. Efficient gradient diagnostics in JAX

Avoid materializing gigantic flattened gradients whenever possible.

For each pytree gradient \(g_i\):

\[
g_i^\top g_j
=
\sum_\ell
\langle
g_i^{(\ell)},
g_j^{(\ell)}
\rangle.
\]

Compute tree-wise:
- squared norm;
- dot product;
- cosine.

Store only scalar summaries and selected layer-level values.

For three channels, the Gram matrix requires only six unique dot products.

---

# 22. Unit tests before training

## Gradient routing tests
For a dummy batch:

- dense loss changes dense-head parameters;
- dense loss does **not** change sparse/final specialist parameters;
- sparse loss behaves analogously;
- shared encoder receives the intended combination.

## Composition tests
Check:
- logits have correct shape;
- distribution normalizes;
- gates remain in bounds;
- residual removal recovers dense-only policy.

## PPO correctness
Numerically verify:
- old/new log-prob ratio comes from the actual composed policy;
- stop-gradient/routing choices do not accidentally detach all specialist influence.

## Reward decomposition
Assert:

\[
r_{\mathrm{env}}
\approx
r_d+r_s+r_f
\]

where this identity is intended.

---

# 23. First concrete experiment matrix

Start with the synthetic task and then Craftax.

| ID | Critics | Actor | Encoder gradient | Temporal attribution |
|---|---|---|---|---|
| E0 | scalar | shared | summed | none |
| E1 | 3-channel | shared | summed | none |
| E2 | 3-channel | 3 additive heads | summed | none |
| E3 | 3-channel | residual heads | summed | none |
| E4 | 3-channel | residual heads | normalized | none |
| E5 | 3-channel | residual heads | PCGrad | none |
| E6 | 3-channel | residual + learned gates | PCGrad | none |
| E7 | 3-channel | residual + learned gates | PCGrad | final redistribution |

Do not initially run a huge Cartesian hyperparameter sweep.

---

# 24. Decision rules after the first results

## Case A — gradients are mostly aligned
If

\[
\cos(g_i,g_j) \gg 0
\]

most of the time, gradient conflict is not the main mechanism.

Shift focus toward:
- temporal attribution;
- exploration;
- critic horizon;
- representation capacity.

## Case B — dense gradients dominate in norm but align directionally
Prioritize:
- GradNorm;
- reward/advantage normalization;
- per-channel learning rates.

Architecture may be unnecessary.

## Case C — strong directional conflict
Proceed with:
- PCGrad/CAGrad baselines;
- actor specialization;
- learned interaction.

## Case D — conflict only in shared encoder
Use:
- shared trunk + adapters;
- encoder-specific gradient surgery.

## Case E — sparse/final gradients are nearly zero
The problem is attribution/exploration, not interference.
Prioritize:
- RUDDER-like redistribution;
- longer horizon;
- exploration/subgoal methods.

---

# 25. What a convincing first workshop-level result might look like

1. Define a reproducible dense/sparse/final decomposition for Craftax.
2. Show gradient norms/cosines throughout PPO training.
3. Identify a systematic pathology, e.g. dense gradient domination or conflict with late-return channels.
4. Introduce a minimal factorized actor/critic intervention.
5. Show improvement specifically on late/sparse achievements.
6. Demonstrate the effect is not reproduced by simple reward reweighting.
7. Add MA-Craftax/Coop evidence if compute permits.

This is stronger than starting with a complicated meta-learned metric.

---

# 26. Coding-agent task list

## Task 1 — repository reconnaissance
Inspect:
- MA-Craftax reward code;
- Craftax reward/achievement code;
- PPO/MAPPO/IPPO training loops;
- Flax/Equinox network modules;
- optimizer update flow.

Output:
- file map;
- tensor shapes;
- current reward structure;
- minimal patch points.

## Task 2 — reward decomposition API
Add channel rewards without changing the baseline scalar reward.

## Task 3 — vector critic
Implement three critic heads and channel-wise GAE.

## Task 4 — gradient diagnostics
Implement pytree norms/dot products/cosines.

## Task 5 — logging
Add WandB/TensorBoard metrics for channel statistics.

## Task 6 — specialist actor
Implement additive/residual logit heads with tested routing.

## Task 7 — aggregation baselines
Implement normalized sum, PCGrad, optional GradNorm/CAGrad.

## Task 8 — experiment configs
Create reproducible YAML configs for E0–E7.

## Task 9 — analysis notebook/script
Generate:
- learning curves;
- gradient cosine plots;
- Gram heatmaps;
- gate traces;
- per-achievement comparisons.

---

# 27. Reasoning-agent task list

1. Formalize the objective and gradient-routing derivation under PPO clipping.
2. Determine whether per-channel PPO losses share the same ratio or require a different surrogate construction.
3. Review whether additive logit specialists introduce identifiability problems.
4. Formalize the direct-sum credit-space interpretation.
5. Determine conditions under which strict parameter separation is harmful.
6. Derive a constrained interaction matrix \(C\) that preserves stable policy updates.
7. Investigate natural-gradient/Fisher geometry as an alternative to arbitrary learned metric \(M\).
8. Perform a dedicated novelty search around:
   - gradient routing across reward components;
   - reward-timescale-specific actor modules;
   - learned cross-objective gradient interaction in RL.

---

# 28. Stop conditions / failure criteria

Pause architectural escalation if:

- channel gradients cannot be reliably estimated;
- reward partition is arbitrary or unstable;
- three critics do not predict sparse/final returns better than one;
- specialist heads collapse to identical policies;
- gates saturate trivially;
- simple gradient normalization matches all gains;
- performance improvements come only from increased parameter count.

Every architectural result must include a parameter-count-matched control.

---

# 29. Immediate recommended order

1. Build synthetic conflicting-timescale MDP.
2. Instrument scalar PPO with channel-wise gradients.
3. Instrument Craftax PPO.
4. Plot norm/cosine/Gram statistics.
5. Add three critics.
6. Add residual actor heads.
7. Compare normalization vs PCGrad vs architectural routing.
8. Only then move to learned gates.
9. Move to MA-Craftax.
10. Add agent-credit decomposition.
11. Explore learned interaction matrix.
12. Explore meta-gradients / learned metric last.

---

# 30. Reference links used by this roadmap

- Craftax: https://proceedings.mlr.press/v235/matthews24a.html
- Multi-Agent Craftax: https://arxiv.org/abs/2511.04904
- MA-Craftax code: https://github.com/BaselOmari/MA-Craftax
- Hybrid Reward Architecture: https://arxiv.org/abs/1706.04208
- MO-MPO: https://proceedings.mlr.press/v119/abdolmaleki20a.html
- GradNorm: https://proceedings.mlr.press/v80/chen18a.html
- PCGrad: https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- CAGrad: https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- RUDDER: https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html
- STAS: https://ojs.aaai.org/index.php/AAAI/article/view/29681
- COMA: https://ojs.aaai.org/index.php/AAAI/article/view/11794
- QMIX: https://proceedings.mlr.press/v80/rashid18a.html
- Meta-Gradient RL: https://papers.neurips.cc/paper_files/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html
