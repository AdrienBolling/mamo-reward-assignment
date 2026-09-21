# mamo-reward-assignment

Factorized credit assignment for multi-objective, multi-agent reinforcement learning.

The project splits the reward into credit channels by granularity: dense, sparse
and final. It keeps the identity of each channel through the value and the
gradient computation. It then applies explicit aggregation and metric operators
before the policy update. The question is whether this improves credit
assignment on long-horizon, multi-objective tasks.

`documentation/` holds the research memo, the bibliography map and the
experiment roadmap.

## Layout

| Path | Content |
|---|---|
| `src/mamora/operators/` | Aggregation and metric operators on channel gradients. This is the contribution. |
| `src/mamora/envs/` | Environments written for this project. |
| `src/mamora/connectors/` | Outside environments adapted to the reward-channel contract. |
| `src/mamora/agents/` | Learners that keep the channel identity of the reward. |
| `src/mamora/runner/` | One experiment: configuration, training loop, run record. |
| `src/mamora/analysis/` | Statistics and figures computed from run records. |
| `documentation/` | Research documents. |
| `paper/` | The article repository, as a submodule. Figures and tables go there. |

Each layer states its import rule in its own docstring.

## Setup

```bash
git clone --recurse-submodules git@github.com:AdrienBolling/mamo-reward-assignment.git
cd mamo-reward-assignment
uv sync                     # Python 3.12 venv, dev + gpu groups
uv run pre-commit install   # pre-commit, commit-msg and pre-push hooks
dvc pull                    # the PDF under documentation/
uv run pytest               # CPU tests
```

## Status

The project restarted on 2026-09-21. This branch holds the layout, the
documentation and the tool chain. The implementation lands step by step:

1. the reward-channel contract, with the first toy environment;
2. the metric operators, and the diagnostics they feed;
3. the agent, with channel critics and channel losses;
4. the runner, the run record and the experiment configs;
5. the aggregation operators, and the analysis scripts;
6. the first connector, for MA-Craftax.

The first implementation is on the `legacy` branch. It is not merged here. It
holds a working timescale corridor, gradient forensics, a PPO with channel
losses, and the premise-1 result in `reports/premise1_corridor.md`.
