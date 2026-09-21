# Factorized Credit Spaces for Long-Horizon Multi-Objective Multi-Agent Reinforcement Learning

**Working research memo — v0.1**  
**Primary target:** Craftax-MA / Craftax-Coop  
**Secondary target:** single-agent Craftax  
**Status:** exploratory research hypothesis, not a claim of established novelty

---

## 1. One-sentence idea

Instead of immediately scalarizing dense, sparse, and terminal reward signals into a single policy-gradient update, preserve them as distinct **credit channels**, route them through partially separated critic/actor modules, and explicitly control or learn how those channels interact before producing the final policy update or action distribution.

---

## 2. Motivation

Long-horizon reinforcement learning often combines reward signals with very different temporal statistics:

- **Dense rewards:** frequent, low-delay feedback tied to local competence or survival.
- **Sparse rewards:** infrequent event/achievement feedback associated with subgoals.
- **Terminal/final rewards:** very delayed feedback representing long-horizon task success.

These labels should be treated as **temporal credit channels**, not as a universal taxonomy of multi-objective RL objectives. A semantic objective may itself contain dense and sparse components, and a sparse reward may occur before the end of an episode.

A standard scalarized actor-critic pipeline tends to collapse these signals early:

\[
r_t^{\mathrm{scalar}}
=
w_d r_t^d + w_s r_t^s + w_f r_t^f
\]

followed by one return estimate, one advantage estimate, and one policy-gradient direction.

This can destroy information about:

1. **which reward channel produced a gradient;**
2. **the relative frequency and scale of the channels;**
3. **whether their gradients agree or conflict;**
4. **whether a channel should modify a shared skill representation or only modulate behavior;**
5. **which temporal and agent-level events deserve credit for delayed team rewards.**

Craftax-MA / Craftax-Coop is a particularly suitable test bed because it combines long horizons, exploration, shared rewards, heterogeneous roles, trading/cooperation, and explicit evidence that standard MARL methods suffer from credit-assignment failures.

---

## 3. Core research hypothesis

Let the policy parameters be \(\theta\), with channel-specific objectives

\[
J_d(\theta), \qquad J_s(\theta), \qquad J_f(\theta).
\]

Their ordinary policy gradients are

\[
g_d = \nabla_\theta J_d,\qquad
g_s = \nabla_\theta J_s,\qquad
g_f = \nabla_\theta J_f.
\]

### Important non-claim

The raw gradients are **not assumed to be intrinsically orthogonal** in the ordinary parameter space. In general,

\[
g_d^\top g_s \neq 0,\qquad
g_d^\top g_f \neq 0,\qquad
g_s^\top g_f \neq 0.
\]

Positive inner products may encode useful transfer; negative inner products may encode conflict.

### Modeling proposal

Treat the three signals as distinct elements of an objective-indexed product/direct-sum space:

\[
\mathcal H_{\mathrm{credit}}
=
T_\theta \Theta_d
\oplus
T_\theta \Theta_s
\oplus
T_\theta \Theta_f,
\]

and retain

\[
\Gamma_t
=
g_t^d \oplus g_t^s \oplus g_t^f
\]

as a structured object rather than immediately projecting it to one vector.

The useful notion is therefore:

> **orthogonal identity, not necessarily orthogonal direction.**

The channels remain separately identifiable even when the directions they recommend are aligned.

---

## 4. Why this is different from ordinary scalarization

Standard scalarization can be viewed as a projection

\[
\mathcal P:
\mathcal H_{\mathrm{credit}}
\rightarrow
T_\theta\Theta
\]

such as

\[
\mathcal P(\Gamma)
=
w_d g_d + w_s g_s + w_f g_f.
\]

This is a dimensional collapse from a structured multi-gradient object to a single direction.

The proposed approach delays this projection. It first computes or approximates relations between the channels and then applies a **credit interaction operator**

\[
\mathcal I_\psi:
\mathcal H_{\mathrm{credit}}
\rightarrow
\mathcal H_{\mathrm{credit}}.
\]

Only after this interaction step does the algorithm:

- update shared/specialist parameters;
- construct a policy-gradient direction;
- or combine channel-specific action preferences.

---

## 5. The gradient matrix and credit geometry

Let

\[
G_t =
\begin{bmatrix}
| & | & |\\
g_t^d & g_t^s & g_t^f\\
| & | & |
\end{bmatrix}
\in \mathbb R^{P\times 3}.
\]

A basic diagnostic object is the Gram matrix

\[
K_t = G_t^\top G_t,
\]

with entries

\[
K_{ij}=g_i^\top g_j.
\]

The normalized version gives pairwise cosine similarities.

This exposes:

- channel gradient magnitudes;
- agreement;
- conflict;
- domination by frequent/dense channels;
- changes in gradient geometry over training or over episode phase.

A learnable interaction matrix could be

\[
C_t \in \mathbb R^{3\times 3},
\]

with

\[
\tilde G_t = G_t C_t.
\]

The entries of \(C_t\) can represent how one credit channel is allowed to modify or condition another.

Possible parameterization:

\[
C_t
=
C_\psi(
h_t,\;
G_t^\top G_t,\;
\tau_t,\;
m_t
),
\]

where:

- \(h_t\): state/history representation;
- \(\tau_t\): episode progress or phase;
- \(m_t\): optional task/agent metadata.

---

## 6. Architecture family A: separated critics + residual actor experts

This is the most practical first architecture.

### Shared representation

For agent \(i\):

\[
h_t^i = f_\phi(o_t^i, h_{t-1}^i).
\]

A recurrent encoder is likely useful in Craftax due to partial observability and long-term dependencies.

### Channel-specific critics

Use separate value functions:

\[
V_d,\qquad V_s,\qquad V_f.
\]

Compute channel-specific return/advantage estimates:

\[
A_t^d,\qquad A_t^s,\qquad A_t^f.
\]

Allow different discount/trace parameters if justified:

\[
(\gamma_d,\lambda_d),\quad
(\gamma_s,\lambda_s),\quad
(\gamma_f,\lambda_f).
\]

This is important because the channels have different temporal characteristics.

### Actor modules

A natural discrete-action formulation for Craftax is:

\[
z_d = g_d(h),
\]

\[
\Delta z_s = g_s(h),
\]

\[
\Delta z_f = g_f(h).
\]

The dense module produces a baseline policy, while sparse and terminal modules produce logit corrections.

Use gates

\[
\alpha_s=\sigma(q_s(h)),\qquad
\alpha_f=\sigma(q_f(h)).
\]

Final logits:

\[
z =
z_d
+
\alpha_s \Delta z_s
+
\alpha_f \Delta z_f.
\]

Then

\[
\pi(a|o)=\mathrm{softmax}(z).
\]

This is preferable to multiplying sampled actions, especially because Craftax has a discrete action space.

### Interpretation

- **Dense module:** local competence / default policy.
- **Sparse module:** achievement- or subgoal-oriented correction.
- **Terminal module:** long-horizon strategic correction.
- **Gates:** decide when a specialist should influence action selection.

---

## 7. Architecture family B: independent policy experts + composition

An alternative is to let each channel produce its own distribution:

\[
\pi_d(a|s),\quad
\pi_s(a|s),\quad
\pi_f(a|s).
\]

### Product-of-experts composition

\[
\pi(a|s)
\propto
\pi_d(a|s)^{\alpha_d}
\pi_s(a|s)^{\alpha_s}
\pi_f(a|s)^{\alpha_f}.
\]

In logit space this becomes approximately

\[
z
=
\alpha_d z_d
+
\alpha_s z_s
+
\alpha_f z_f.
\]

### Mixture composition

\[
\pi(a|s)
=
\alpha_d \pi_d(a|s)
+
\alpha_s \pi_s(a|s)
+
\alpha_f \pi_f(a|s).
\]

This architecture is closer to existing multi-objective distribution-composition methods, so it may have less novelty than gradient-routing variants. It is still a useful ablation.

---

## 8. Architecture family C: specialist parameter spaces

Suppose

\[
\theta =
(\theta_d,\theta_s,\theta_f).
\]

Enforce channel-specific updates:

\[
J_d \rightarrow \theta_d,\qquad
J_s \rightarrow \theta_s,\qquad
J_f \rightarrow \theta_f.
\]

Then the lifted gradients can be written

\[
\bar g_d =
\begin{pmatrix}
\nabla_{\theta_d} J_d\\
0\\
0
\end{pmatrix},
\quad
\bar g_s =
\begin{pmatrix}
0\\
\nabla_{\theta_s} J_s\\
0
\end{pmatrix},
\quad
\bar g_f =
\begin{pmatrix}
0\\
0\\
\nabla_{\theta_f} J_f
\end{pmatrix}.
\]

These are orthogonal by construction in the direct-sum parameterization.

This is a strong inductive bias:

> Reward channels do not directly overwrite each other's specialist parameters; interaction occurs through explicit forward composition, shared representations, adapters, or learned transfer operators.

### Risk

Strict separation may eliminate beneficial transfer. Therefore this should be compared with:

- fully shared parameters;
- shared trunk + separate heads;
- partial adapters;
- soft routing;
- gradient projection;
- learned interaction.

---

## 9. Shared representation is a central research question

Even with separated action heads, a shared encoder \(\phi\) can still be dominated by dense-reward gradients.

For the encoder, log

\[
g_{\phi,d},\quad
g_{\phi,s},\quad
g_{\phi,f}.
\]

Potential treatments:

1. **Naive sum**
   \[
   g_\phi = g_{\phi,d}+g_{\phi,s}+g_{\phi,f}.
   \]

2. **Gradient normalization**
   Balance channel magnitudes before aggregation.

3. **Conflict-aware projection**
   Apply PCGrad/CAGrad-like operations.

4. **Separate adapters**
   Shared trunk plus channel-specific LoRA-like/adaptor blocks.

5. **Mixture-of-experts representation**
   Route state features through channel specialists.

6. **Learned credit interaction operator**
   Use \(C_\psi\) to transform channel gradients.

A likely early contribution is empirical evidence about *where* interference occurs: encoder, recurrent core, critic, or actor head.

---

## 10. Learned metric / non-Euclidean credit geometry

Ordinary gradient compatibility uses Euclidean inner products:

\[
\langle g_i,g_j\rangle = g_i^\top g_j.
\]

A more ambitious direction is to define

\[
\langle g_i,g_j\rangle_M
=
g_i^\top M g_j
\]

for a positive-semidefinite metric \(M\).

Then the notion of compatibility is learned or adapted to the problem.

Possible forms:

\[
M = L^\top L,
\]

with low-rank \(L\), or state-dependent

\[
M_t = M_\psi(h_t).
\]

This could define a task-relevant credit geometry rather than assuming Euclidean parameter geometry is meaningful.

### Caution

This is substantially more complex and should not be the first implementation. It is a later research branch after establishing that channel identity and interference matter empirically.

---

## 11. Genuine hypergradient extension

The phrase *hypergradient* should be reserved for a meta-optimization layer.

Suppose an interaction operator \(C_\psi\) determines the inner policy update:

\[
\theta'
=
\theta
+
\alpha \mathcal U(G,C_\psi).
\]

Evaluate an outer objective:

\[
J_{\mathrm{meta}}(\theta').
\]

Then optimize

\[
\nabla_\psi J_{\mathrm{meta}}(\theta').
\]

This learns **how credit channels should interact** based on downstream performance.

Possible outer objectives:

- total environment return;
- terminal success rate;
- hypervolume in objective space;
- fairness across objectives;
- team return in MARL;
- held-out long-horizon achievement score.

This provides a clean two-level interpretation:

1. specialist channels learn what locally improves each type of return;
2. a meta-learner learns how these credit channels should influence one another.

---

## 12. Temporal credit assignment is not solved by separation alone

A major conceptual caveat:

> Splitting gradients by reward channel reduces gradient/representation interference, but it does not automatically tell a sparse or terminal reward which earlier actions caused it.

For terminal rewards, \(A_t^f\) may still be high variance and weakly informative over thousands of steps.

Therefore the full research program may require two distinct mechanisms:

### A. Credit channel factorization

Preserve dense/sparse/final identity and control cross-channel interference.

### B. Within-channel temporal attribution

For sparse/final channels, improve attribution over time using methods such as:

- return decomposition;
- reward redistribution;
- sequence models;
- attention over trajectory history;
- eligibility traces;
- event-conditioned critics;
- learned subgoal boundaries.

The architecture should make these mechanisms composable rather than conflate them.

---

## 13. Multi-agent extension: credit as a tensor

Craftax-Coop adds a second credit axis: **which agent caused the shared outcome?**

A useful conceptual object is

\[
A_t^{i,k}
\]

where:

- \(i\): agent;
- \(k \in \{d,s,f\}\): reward/credit channel;
- \(t\): timestep.

This gives an explicit factorization:

\[
\boxed{
\text{time}
\times
\text{agent}
\times
\text{reward channel}
}
\]

instead of a single team advantage.

A policy-gradient contribution becomes

\[
g^{i,k}
=
\sum_t
A_t^{i,k}
\nabla_\theta
\log \pi_i(a_t^i|o_t^i).
\]

This suggests a broader research direction:

> **Spatio-Temporal-Objective Credit Assignment (STOCA)** or **Factorized Credit Assignment** for open-ended MARL.

The name is provisional.

---

## 14. Why Craftax-MA / Craftax-Coop

The official Multi-Agent Craftax work describes:

- a JAX implementation suitable for very high interaction budgets;
- 53 discrete Craftax actions;
- shared rewards across agents;
- heterogeneous specialization and trading in Craftax-Coop;
- degradation under shared rewards attributed to noisy credit assignment;
- explicit long-horizon temporal credit failures;
- an ablation where adding immediate food/water incentives improves performance.

This makes the environment valuable because it contains both:

1. **temporal credit assignment** problems;
2. **inter-agent/spatial credit assignment** problems.

The baseline repository also includes IPPO, MAPPO, and PQN implementations, which should make controlled algorithmic modifications feasible.

---

## 15. Falsifiable hypotheses

### H1 — Dense-gradient domination

Dense channels have substantially larger cumulative gradient norm than sparse/final channels:

\[
\mathbb E \|g_d\|
\gg
\mathbb E \|g_s\|,
\mathbb E \|g_f\|.
\]

### H2 — Gradient conflict

Channel gradients frequently disagree:

\[
\Pr[
\cos(g_d,g_f)<0
]
\]

and/or

\[
\Pr[
\cos(g_d,g_s)<0
]
\]

is non-trivial and changes over training.

### H3 — Interference is layer-dependent

Conflict is concentrated in specific parts of the network, likely shared representations/recurrent state rather than every layer.

### H4 — Preserving channel identity improves rare-objective learning

Separate critics/advantages and partially separated actor updates improve sparse/terminal achievement acquisition without catastrophically degrading dense competence.

### H5 — Dynamic interaction beats strict isolation

Learned/gated interaction outperforms both:

- naive scalarized sharing;
- fully independent specialists.

### H6 — MARL benefits from factorizing both agent and reward-channel credit

A model that decomposes team credit across both agents and reward channels improves Craftax-Coop cooperation more than either decomposition alone.

---

## 16. Minimum viable algorithm

A first implementation should avoid the learned metric and meta-gradient machinery.

### Baseline

MAPPO or IPPO in Craftax-MA / Craftax-Coop, PPO in Craftax.

### Reward partition

Produce three streams:

\[
r_t^d,\quad r_t^s,\quad r_t^f.
\]

The exact partition should be explicit and version-controlled.

### Critics

Three value heads:

\[
V_d,\quad V_s,\quad V_f.
\]

### Advantages

Three independent GAE estimates:

\[
A_d,\quad A_s,\quad A_f.
\]

### Actor

Shared encoder + three residual logit heads:

\[
z=z_d+\alpha_s\Delta z_s+\alpha_f\Delta z_f.
\]

### Gradient routing

At minimum:

- dense policy loss updates dense head;
- sparse policy loss updates sparse head;
- final policy loss updates final head;
- all three may update the shared encoder in the baseline variant.

Then compare encoder treatments.

### Required diagnostics

Log per channel and per relevant layer:

- gradient norm;
- cosine similarity;
- update norm;
- explained variance of critic;
- advantage mean/std;
- reward frequency;
- gate values;
- logit residual norms;
- objective-specific returns;
- achievement rates.

---

## 17. Essential ablations

1. **Scalarized PPO/MAPPO**
2. **Three critics only**
3. **Three critics + three actor heads, summed logits**
4. **Residual heads + fixed gates**
5. **Residual heads + learned gates**
6. **Separate heads + PCGrad on shared encoder**
7. **Separate heads + GradNorm**
8. **Fully separate specialist networks**
9. **Sparse/final temporal redistribution**
10. **Agent-credit decomposition in Craftax-Coop**

The purpose is to distinguish gains from:

- better value estimation;
- gradient scale balancing;
- conflict avoidance;
- architectural specialization;
- temporal attribution;
- multi-agent attribution.

---

## 18. What would count as a meaningful result

A good paper does not require the most complex version.

A strong result could be:

1. dense/sparse/final gradient interference is measured and shown to be systematic;
2. the interference correlates with failure to learn long-horizon achievements;
3. preserving reward-channel gradient identity reduces the failure;
4. explicit interaction/routing outperforms naive scalarization;
5. the effect transfers from Craftax to Craftax-MA/Coop.

A negative result is also scientifically valuable if it shows that the channels are mostly aligned and the bottleneck is instead temporal attribution or exploration.

---

## 19. Main novelty risks

### Risk A — “reward decomposition already exists”

True. Hybrid Reward Architecture and related methods decompose rewards/value functions.

**Response:** novelty must not be claimed at the level of reward decomposition alone.

### Risk B — “multi-objective policies already combine objective-specific distributions”

True. MO-MPO and related work do this.

**Response:** focus on persistent reward-timescale credit channels and their optimization/gradient routing.

### Risk C — “gradient conflict methods already manipulate multiple gradients”

True. MGDA, GradNorm, PCGrad, CAGrad, and MORL-specific gradient aggregation all exist.

**Response:** investigate the specific semantics of temporal reward channels, architecture-level specialist routing, and interaction with delayed/spatial credit.

### Risk D — “three-way partition is arbitrary”

Possible.

**Response:** treat dense/sparse/final as an initial operational partition. Later test learned clustering by reward delay/frequency/variance or event type.

### Risk E — “strict orthogonality harms positive transfer”

Likely in some states.

**Response:** preserve channel identity but do not force all raw directions to be perpendicular.

---

## 20. Terminology

Recommended provisional terms:

- **credit channel** — one reward/advantage/gradient stream;
- **factorized credit space** — structured product/direct-sum representation of channel gradients;
- **credit geometry** — relations among credit-channel gradients;
- **credit interaction operator** — transformation controlling cross-channel influence;
- **credit routing** — mapping each channel to parameters/modules;
- **residual policy expert** — channel-specific logit correction;
- **spatio-temporal-objective credit** — factorization over agent, time, and reward channel.

Avoid using **hypergradient** unless there is an actual outer differentiation through an inner update.

---

## 21. Immediate research questions

1. How should Craftax rewards be partitioned into dense/sparse/final channels without introducing arbitrary semantics?
2. Are the channel gradients measurably conflicting?
3. Does dense reward dominate by norm/frequency?
4. Is conflict primarily in the shared encoder?
5. Do separate critics already solve most of the problem?
6. Does actor specialization add value beyond critic decomposition?
7. Should the terminal channel act directly on logits or only gate other modules?
8. Should channels use different discount factors or GAE parameters?
9. Does reward redistribution help the terminal channel more than architectural separation?
10. In Craftax-Coop, how should agent credit decomposition interact with reward-channel decomposition?
11. Can the interaction matrix \(C_t\) be learned stably?
12. Is a learned gradient metric useful after simpler methods are exhausted?

---

## 22. Recommended project progression

### Stage 0 — instrumentation
Do not change the policy architecture. Measure per-channel gradient geometry.

### Stage 1 — critic decomposition
Three critics, scalarized/shared actor.

### Stage 2 — actor gradient routing
Three residual actor heads with explicit channel-specific losses.

### Stage 3 — interaction
Learned state/history-dependent gates or a small \(3\times3\) interaction operator.

### Stage 4 — temporal attribution
Add RUDDER/STAS-inspired attribution for sparse/final channels.

### Stage 5 — multi-agent tensor credit
Factorize across agent × time × reward channel.

### Stage 6 — meta-learning
Learn interaction rules using genuine hypergradients.

---

## 23. Reference anchors

These references are expanded in the companion bibliography artifact.

- Craftax: https://proceedings.mlr.press/v235/matthews24a.html
- Multi-Agent Craftax: https://arxiv.org/abs/2511.04904
- MA-Craftax code: https://github.com/BaselOmari/MA-Craftax
- Hybrid Reward Architecture: https://arxiv.org/abs/1706.04208
- MO-MPO: https://proceedings.mlr.press/v119/abdolmaleki20a.html
- MGDA / Multi-task as Multi-objective Optimization: https://arxiv.org/abs/1810.04650
- GradNorm: https://proceedings.mlr.press/v80/chen18a.html
- PCGrad: https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- CAGrad: https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- RUDDER: https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html
- STAS: https://ojs.aaai.org/index.php/AAAI/article/view/29681
- COMA: https://ojs.aaai.org/index.php/AAAI/article/view/11794
- QMIX: https://proceedings.mlr.press/v80/rashid18a.html
- Meta-Gradient RL: https://papers.neurips.cc/paper_files/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html
