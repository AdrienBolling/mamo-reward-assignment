# mamo-reward-assignment

Reward assignment for open-ended multi-agent multi-objective reinforcement learning,
studied on [MA-Craftax](https://github.com/BaselOmari/MA-Craftax) (Craftax-MA / Craftax-Coop, JAX).

## Setup

```bash
git clone --recurse-submodules git@github.com:AdrienBolling/mamo-reward-assignment.git
cd mamo-reward-assignment
uv sync                     # Python 3.12 venv, dev + gpu groups
uv run pre-commit install   # pre-commit, commit-msg and pre-push hooks
```

## Run

```bash
uv run mamora-run dry_run=true                                   # compose + print the config
uv run mamora-run                                                # random-policy rollout, Craftax-MA
uv run mamora-run env=craftax_coop_symbolic rollout.steps=32     # override group / values
uv run pytest                                                    # CPU tests
```

Runs are configured with Hydra (`conf/`), outputs land under `outputs/<date>/<time>/`.
Development rules live in `CLAUDE.md`.
