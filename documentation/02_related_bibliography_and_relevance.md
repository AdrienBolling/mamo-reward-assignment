# Bibliography Map for Factorized Credit Assignment in Craftax / Multi-Agent Craftax

**Working bibliography — v0.1**  
**Purpose:** provide reasoning and coding agents with a prioritized map of related work, what to extract from each paper, and how it constrains novelty.

---

## 1. Reading strategy

The literature falls into five main clusters:

1. **Craftax / Multi-Agent Craftax** — benchmark motivation and implementation.
2. **Reward/value decomposition** — precedent for maintaining reward-component structure.
3. **Multi-objective policy optimization** — precedent for objective-specific distributions and non-scalarized optimization.
4. **Gradient interaction / multi-task optimization** — tools for measuring and manipulating multiple gradients.
5. **Temporal and multi-agent credit assignment** — methods addressing delayed and shared rewards.
6. **Meta-gradients** — relevant only for the later idea of learning the credit-interaction rule itself.

The most important novelty constraint is that **none of reward decomposition, multiple critics, multiple policy distributions, or gradient manipulation is novel by itself**. The research opportunity is in their combination around **reward-timescale-specific credit channels**, especially in long-horizon MOMARL.

---

# A. Benchmark and environment

## A1. Matthews et al. (2024) — Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning

**Venue:** ICML 2024  
**Link:** https://proceedings.mlr.press/v235/matthews24a.html  
**Code:** https://github.com/MichaelTMatthews/Craftax

### Core idea
Craftax is a high-throughput JAX benchmark designed to require exploration, long-term planning, memory, and adaptation while remaining computationally tractable.

### Why relevant
This is the single-agent environment from which the research should first inherit:

- reward mechanics;
- achievement hierarchy;
- action space;
- PPO baselines;
- long-horizon structure;
- efficient JAX experimentation.

### What to extract
- Exact reward definition and achievement list.
- Existing PPO/PPO-RNN/GTrXL baselines.
- Episode length / termination semantics.
- Symbolic observation representation.
- Which achievements empirically remain unsolved.
- Whether health changes or other continuous signals already behave like dense reward.

### Novelty implication
Craftax is a good smaller test bed for temporal credit-channel separation before introducing inter-agent credit.

---

## A2. Al Omari et al. (2025) — Multi-Agent Craftax: Benchmarking Open-Ended Multi-Agent Reinforcement Learning at the Hyperscale

**Link:** https://arxiv.org/abs/2511.04904  
**Code:** https://github.com/BaselOmari/MA-Craftax

### Core idea
Extends Craftax to multi-agent settings and introduces Craftax-Coop with heterogeneous roles, trading, and cooperation.

### Key details for this project
The paper reports that:

- rewards are shared among agents regardless of which agent completes an achievement;
- Craftax-Coop contains long-horizon dependencies such as food/water acquisition whose health consequences appear much later;
- MAPPO performance degrades under shared rewards, which the authors associate with noisy credit assignment;
- adding an immediate food/water incentive improves performance, providing direct evidence that temporal credit is a bottleneck;
- standard baselines include MAPPO, IPPO, and PQN;
- the environments retain Craftax's large discrete action space.

### Why relevant
This benchmark gives two independent credit axes:

\[
\text{temporal credit}
\times
\text{agent credit}.
\]

That makes it ideal for the later factorized object

\[
A_t^{i,k}.
\]

### What to extract
- Exact code path where rewards are constructed.
- Whether rewards can be separated into components without modifying environment semantics.
- Baseline MAPPO/IPPO implementation details.
- Achievement ownership vs globally shared reward.
- Role-specific achievements in Craftax-Coop.
- Food/water shaping ablation.
- Instrumentation hooks for per-agent action/log-prob/value losses.

### Novelty implication
The environment paper itself explicitly motivates long-horizon and shared-reward credit assignment, strengthening the case for the proposed research.

---

# B. Reward and value decomposition

## B1. van Seijen et al. (2017) — Hybrid Reward Architecture for Reinforcement Learning

**Venue:** NeurIPS 2017  
**Link:** https://arxiv.org/abs/1706.04208  
**Proceedings:** https://papers.nips.cc/paper_files/paper/2017/hash/1264a061d82a2edae1574b07249800d6-Abstract.html

### Core idea
Decompose a reward into components and learn a separate value function for each component, then combine the component values for action selection.

### Why relevant
This is one of the closest historical precedents for the idea that distinct reward components should remain structurally separated rather than immediately collapsed.

### Difference from proposed project
HRA mainly decomposes **value estimation**. The proposed project is interested in:

- reward-timescale semantics;
- separate advantage streams;
- actor-gradient routing;
- residual actor specialists;
- learned interaction among gradient channels;
- multi-agent × temporal decomposition.

### What to reuse
- Reward-component interface.
- Separate value-head intuition.
- Ablation logic comparing scalarized vs decomposed values.

### Novelty constraint
Do not claim that decomposing rewards or learning separate value heads is new.

---

# C. Multi-objective policy optimization

## C1. Abdolmaleki et al. (2020) — A Distributional View on Multi-Objective Policy Optimization (MO-MPO)

**Venue:** ICML 2020  
**Link:** https://proceedings.mlr.press/v119/abdolmaleki20a.html  
**arXiv:** https://arxiv.org/abs/2005.07513

### Core idea
Learn an action distribution for each objective and combine objective-specific distributions into a single parametric policy, avoiding direct reward-space scalarization.

### Why relevant
This is a very close precedent for the original proposal:

> each reward/objective-specific subnetwork produces a distribution/logit preference, then these are combined.

### Difference from proposed project
MO-MPO is not specifically about:

- dense vs sparse vs terminal temporal credit regimes;
- separate actor parameter spaces receiving channel-specific policy gradients;
- delayed-reward attribution;
- factorized agent × time × reward-channel credit;
- learned gradient-space interaction.

### What to extract
- Mathematical treatment of objective-specific action distributions.
- How objective preferences are encoded.
- Distribution-combination mechanism.
- Scale invariance arguments.

### Novelty constraint
A paper cannot claim novelty merely for “one policy/action distribution per objective followed by aggregation.”

---

## C2. Abdolmaleki et al. (2021) — On Multi-objective Policy Optimization as a Tool for Reinforcement Learning: Case Studies in Offline RL and Finetuning

**arXiv:** https://arxiv.org/abs/2106.08199

### Core idea
Frames auxiliary RL objectives as multi-objective optimization and introduces Distillation of a Mixture of Experts (DiME).

### Why relevant
It broadens the interpretation of “objective” beyond semantic reward goals. This supports thinking of optimization signals as separately handled experts.

### What to extract
- Expert distillation mechanism.
- How conflicting auxiliary objectives are combined.
- Whether specialist behaviors remain separately parameterized or are distilled into one policy.

### Novelty implication
Useful precedent for **forward policy composition**, but less directly about temporal credit assignment.

---

## C3. Kim et al. (2025) — Conflict-Averse Gradient Aggregation for Constrained Multi-Objective Reinforcement Learning (CoMOGA)

**Venue:** ICLR 2025  
**Link:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/59ddfff7979b43f54690fa986c0e5138-Abstract-Conference.html  
**arXiv:** https://arxiv.org/abs/2403.00282

### Core idea
Treat multi-objective policy improvement as a constrained optimization problem and construct updates that avoid harmful objective-gradient conflicts.

### Why relevant
Direct evidence that gradient conflict is important in MORL, not only supervised multi-task learning.

### What to extract
- Objective-gradient definition in actor-critic RL.
- Conflict criteria.
- Optimization program used to aggregate directions.
- Computational overhead.

### Novelty implication
The proposed work should distinguish **persistent channel factorization and routing** from standard conflict-aware gradient aggregation.

---

## C4. Wang et al. (2025) — Theoretical Study of Conflict-Avoidant Multi-Objective Reinforcement Learning

**Journal:** IEEE Transactions on Information Theory, 2025  
**DOI:** https://doi.org/10.1109/TIT.2025.3581454  
**arXiv version:** https://arxiv.org/abs/2405.16077

### Core idea
Develops conflict-avoidant multi-objective actor-critic variants and analyzes convergence/sample complexity.

### Why relevant
Provides formal language and theory around conflicting objective gradients in RL.

### What to extract
- Definition of conflict-avoidant direction/distance.
- Dynamic weighting.
- Actor-critic gradient structure.
- Theoretical assumptions that fail or hold in deep MARL.

### Novelty implication
Useful baseline/analysis reference if claiming that dense gradients dominate or conflict with sparse/final gradients.

---

## C5. D3PO — Preference-Conditioned Multi-Objective RL: Decomposed, Diversity-Driven Policy Optimization

**Status:** ICLR 2026 review-era preprint; verify final publication status before citation in a paper.  
**OpenReview PDF surfaced:** https://openreview.net/pdf/e9440539e2e1d0f0cb171ebca9a1274f88b399b1.pdf

### Core idea
A decomposed preference-conditioned MORL optimization process designed to mitigate destructive gradient interference and policy diversity collapse.

### Why relevant
Potentially close recent work around maintaining objective-wise learning signals.

### What to extract
- Whether critics and advantages remain objective-wise.
- Exact point at which gradients are combined.
- Architecture of actor decomposition.
- Whether reward components are semantically or temporally defined.

### Novelty warning
Because this is recent and potentially very close, re-check the final version and citation before writing novelty claims.

---

# D. Gradient interaction and multi-task optimization

## D1. Sener & Koltun (2018) — Multi-Task Learning as Multi-Objective Optimization

**Venue:** NeurIPS 2018  
**Link:** https://arxiv.org/abs/1810.04650  
**Proceedings:** https://papers.nips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html

### Core idea
Treats multi-task learning explicitly as a multi-objective optimization problem and uses gradient-based Pareto optimization rather than a fixed weighted sum.

### Why relevant
Provides a formal precedent for treating the **matrix/set of gradients** as the primitive object instead of a pre-summed loss gradient.

### What to extract
- Minimum-norm multi-gradient solution.
- Pareto stationarity conditions.
- Practical approximations.
- Gradient normalization choices.

### Relation to factorized credit space
The proposed gradient matrix

\[
G=[g_d,g_s,g_f]
\]

is naturally connected to this literature.

---

## D2. Chen et al. (2018) — GradNorm

**Venue:** ICML 2018  
**Link:** https://proceedings.mlr.press/v80/chen18a.html

### Core idea
Adaptively balance multi-task losses by controlling gradient magnitudes.

### Why relevant
A dense reward channel may dominate simply because it produces larger/more frequent gradients, even without directional conflict.

### What to extract
- Gradient norm measurement.
- Dynamic weighting rule.
- Training-rate balancing.

### Essential baseline
If channel separation helps, compare against GradNorm to determine whether the gain is merely magnitude balancing.

---

## D3. Yu et al. (2020) — Gradient Surgery for Multi-Task Learning (PCGrad)

**Venue:** NeurIPS 2020  
**Link:** https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

### Core idea
When two task gradients conflict, project one gradient onto the normal plane of the other before aggregation.

### Why relevant
It provides:

- a standard operational definition of gradient conflict;
- a simple baseline;
- a layer/parameter-level diagnostic language.

### Relation to proposed idea
PCGrad edits conflicting gradients in a common parameter space. The proposed factorized-credit view instead asks whether reward channels should retain persistent identity and interact through explicit modules/operators.

### Essential baseline
Apply PCGrad only to the shared encoder and compare with explicit specialist heads.

---

## D4. Liu et al. (2021) — Conflict-Averse Gradient Descent (CAGrad)

**Venue:** NeurIPS 2021  
**Link:** https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

### Core idea
Constructs a gradient direction that balances the average objective while improving the worst-aligned task direction.

### Why relevant
A stronger gradient-aggregation baseline than simple projection.

### What to extract
- Optimization objective.
- Relationship to MGDA.
- Practical computational cost.
- RL experiments, if applicable.

---

# E. Temporal credit assignment

## E1. Arjona-Medina et al. (2019) — RUDDER: Return Decomposition for Delayed Rewards

**Venue:** NeurIPS 2019  
**Link:** https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html

### Core idea
Redistributes delayed rewards toward earlier state-action events that contributed to the return, aiming to reduce expected future reward and improve learning under long delays.

### Why relevant
This addresses the main limitation of pure gradient-channel separation:

> a separated terminal-reward head still needs meaningful temporal attribution.

### Possible integration
Use RUDDER-like redistribution **only for sparse/final channels**, then feed the redistributed rewards into the corresponding critics/actor heads.

### Experimental question
Does reward redistribution and gradient factorization provide complementary gains?

---

## E2. Chen et al. (2024) — STAS: Spatial-Temporal Return Decomposition for Solving Sparse Rewards Problems in Multi-Agent Reinforcement Learning

**Venue:** AAAI 2024  
**Link:** https://ojs.aaai.org/index.php/AAAI/article/view/29681  
**DOI:** https://doi.org/10.1609/aaai.v38i16.29681

### Core idea
First decomposes a delayed global return across time, then redistributes payoff across agents using Shapley-inspired attribution.

### Why relevant
This is arguably the most important prior work for the **agent × time** part of the proposed research.

### Difference from proposed project
STAS does not center on dense/sparse/final reward-channel gradient factorization or specialist actor routing.

### Possible integration
Generalize the attribution object from

\[
A_t^i
\]

to

\[
A_t^{i,k}.
\]

### Novelty constraint
Do not claim that jointly addressing spatial and temporal MARL credit is itself new.

---

# F. Multi-agent credit and value factorization

## F1. Foerster et al. (2018) — Counterfactual Multi-Agent Policy Gradients (COMA)

**Venue:** AAAI 2018  
**Link:** https://ojs.aaai.org/index.php/AAAI/article/view/11794

### Core idea
Uses a centralized critic and a counterfactual baseline that marginalizes one agent's action while holding other agents fixed.

### Why relevant
Classic policy-gradient solution to **which agent deserves credit?**

### Possible role
COMA-style counterfactual advantages could be computed per reward channel:

\[
A_t^{i,d},\quad
A_t^{i,s},\quad
A_t^{i,f}.
\]

### Caveat
Computational scaling and compatibility with the chosen MA-Craftax baseline need evaluation.

---

## F2. Sunehag et al. (2017) — Value-Decomposition Networks (VDN)

**Link:** https://arxiv.org/abs/1706.05296

### Core idea
Decomposes a team value into per-agent value components.

### Why relevant
Provides a structural analogy:

- VDN factorizes **team value across agents**;
- the proposed approach factorizes **credit across reward channels**, potentially jointly across agents.

### Novelty implication
Factorized representations are deeply established in MARL; novelty must lie in the axes and interaction mechanism.

---

## F3. Rashid et al. (2018) — QMIX

**Venue:** ICML 2018  
**Link:** https://proceedings.mlr.press/v80/rashid18a.html

### Core idea
Combines per-agent action values through a monotonic mixing network so decentralized action selection remains tractable.

### Why relevant
QMIX is a useful architectural analogy for a learned **credit mixing operator**.

### Possible inspiration
Instead of arbitrary \(C_t\), impose structure/constraints on how channel experts are combined to preserve desirable properties.

---

## F4. Lowe et al. (2017) — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments

**Venue:** NeurIPS 2017  
**Link:** https://proceedings.neurips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html

### Core idea
Centralized critics condition on other agents while decentralized actors execute locally.

### Why relevant
General CTDE background and reminder that policy-gradient variance/non-stationarity already worsen with multiple agents.

### Priority
Lower than COMA/STAS/QMIX for this project, but useful framing.

---

# G. Meta-gradient learning

## G1. Xu, van Hasselt & Silver (2018) — Meta-Gradient Reinforcement Learning

**Venue:** NeurIPS 2018  
**Link:** https://papers.neurips.cc/paper_files/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html

### Core idea
Differentiate an outer objective through an inner RL update to adapt components of the return/learning process.

### Why relevant
This is the right conceptual precedent for the later proposal:

\[
\theta'
=
\theta+\alpha \mathcal U(G,C_\psi)
\]

followed by

\[
\nabla_\psi J_{\mathrm{outer}}(\theta').
\]

### Terminology implication
Use **hypergradient/meta-gradient** only once this bilevel differentiation exists. The mere matrix of reward-channel gradients is better called a factorized or lifted credit-gradient space.

---

# H. Recommended reading priority

## Tier 0 — read before implementation

1. Multi-Agent Craftax (Al Omari et al., 2025)
2. Craftax (Matthews et al., 2024)
3. HRA (van Seijen et al., 2017)
4. MO-MPO (Abdolmaleki et al., 2020)
5. PCGrad (Yu et al., 2020)
6. GradNorm (Chen et al., 2018)
7. RUDDER (Arjona-Medina et al., 2019)
8. STAS (Chen et al., 2024)

## Tier 1 — important for method design

9. MGDA / Multi-task as Multi-objective Optimization
10. CAGrad
11. CoMOGA
12. Conflict-Avoidant MORL theory
13. COMA
14. QMIX / VDN
15. DiME

## Tier 2 — later-stage extensions

16. Meta-Gradient RL
17. D3PO final/publication version, once verified
18. Natural-gradient / learned-metric literature if pursuing non-Euclidean credit geometry

---

# I. Novelty map

| Component | Prior art strength | Potential novelty |
|---|---:|---|
| Reward decomposition | Very strong | Low |
| Separate value heads per reward component | Very strong | Low |
| Objective-specific action distributions | Strong | Low–moderate |
| Multiple objective gradients kept separately | Strong | Low |
| Gradient conflict measurement/manipulation | Very strong | Low |
| Dense/sparse/terminal **temporal-credit semantics** | Less direct | Moderate |
| Reward-timescale-specific actor gradient routing | Less direct | Moderate |
| Persistent specialist modules + explicit interaction operator | Some analogues | Moderate |
| Learned interaction among credit channels | Less direct | Moderate–high, needs search |
| Agent × time × reward-channel credit tensor | STAS covers agent × time | Potentially high |
| Learned metric over credit channels | General related math exists | Potentially high but risky |
| Hypergradient learning of credit interactions | Meta-gradient precedents | Potentially moderate–high |

This table is provisional. A formal novelty review should be repeated after the first implementation and again before paper submission.

---

# J. Search terms for further literature mining

Use combinations of:

- "reward decomposition reinforcement learning policy gradient"
- "objective-specific actor heads reinforcement learning"
- "multi-objective policy gradient decomposition"
- "gradient routing reinforcement learning multiple rewards"
- "multi-objective reinforcement learning gradient interference"
- "delayed reward multi-objective reinforcement learning"
- "temporal credit assignment multi-objective RL"
- "reward timescale reinforcement learning"
- "multi-timescale reinforcement learning reward"
- "sparse dense reward gradient interference"
- "multi-agent multi-objective credit assignment"
- "spatio temporal objective credit assignment"
- "multi-agent reward decomposition actor critic"
- "meta-gradient reward weighting reinforcement learning"
- "learned gradient aggregation multi-objective RL"
- "gradient interaction matrix multi-task learning"
- "Riemannian multi-objective optimization gradient metric"
- "learned metric gradient conflict multi-task"

---

# K. Implementation repositories to inspect

## MA-Craftax
https://github.com/BaselOmari/MA-Craftax

Priority files/directories:
- reward construction;
- baseline MAPPO/IPPO/PQN;
- configuration files;
- logging/evaluation;
- achievement definitions.

## Craftax
https://github.com/MichaelTMatthews/Craftax

Priority:
- PPO baseline;
- symbolic environment;
- achievements/rewards;
- recurrent baseline implementations if available.

## MGDA reference implementation
https://github.com/isl-org/MultiObjectiveOptimization

Useful primarily for minimum-norm multi-gradient solvers and normalization ideas.

---

# L. Bibliographic hygiene for agents

When generating a paper draft:

1. Verify final publication metadata for all 2025–2026 works.
2. Prefer proceedings/DOI pages over secondary aggregators.
3. Do not describe D3PO as accepted/published unless verified.
4. Do not say the proposed method is the “first” without a dedicated novelty search.
5. Separate claims about:
   - reward decomposition;
   - objective decomposition;
   - gradient decomposition;
   - temporal attribution;
   - inter-agent attribution.
6. Use "dense/sparse/terminal reward channels" as an operational design choice, not an established universal MORL taxonomy.
