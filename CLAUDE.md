# mamo-reward-assignment (`mamora`)

Reward assignment for open-ended multi-agent multi-objective RL (MOMARL), studied on
MA-Craftax (Craftax-MA / Craftax-Coop, JAX, JaxMARL interface). uv-managed Python 3.12
research codebase: `mamora` holds our algorithms and the experiment CLI, the environment
lives in a pinned fork checked out as a git submodule, every run is fully described by
Hydra config under `conf/`.

## Workflow rules (PROTECTED)

| Rule | Detail |
|---|---|
| One concern per branch | Branch `<type>/<scope>-<short-desc>` (e.g. `feat/rewards-shapley`) off `main`. No repo-wide sweeps, no drive-by refactors, no unrelated fixes in the same branch. |
| Commits | Conventional Commits with a mandatory scope: `type(scope): subject`. Types: `feat fix docs refactor test chore ci build perf`. Several scoped commits per branch are fine. Enforced by the commit-msg hook. |
| Pull requests | `git push -u origin <branch>` then `gh pr create` using `.github/PULL_REQUEST_TEMPLATE.md`. The agent NEVER merges a PR, NEVER pushes to `main`, NEVER force-pushes. The user reviews and merges manually. |
| Hooks | Never `--no-verify`, never skip or disable a hook, never edit `.git/hooks`. Before opening a PR: `uv run pre-commit run --all-files && uv run pytest`. |
| Suppressions | `# noqa: RULE  # reason: ...` and `# ty: ignore[rule]  # reason: ...` ONLY with explicit user authorization in the current conversation. Same for any change to `[tool.ruff]` / `[tool.ty]` rules. Bare `# noqa` / `# ty: ignore` / `# type: ignore` are rejected by the hooks. |
| uv only | `uv add` / `uv add --group dev` / `uv remove`; never pip. Commit `uv.lock` with the change (the `uv-lock` hook checks it). `uv sync` installs the `dev` and `gpu` groups by default. |
| Hydra | Every run parameter lives in `conf/` (root `config.yaml` + groups). No hard-coded hyperparameters, seeds or paths in code. A new knob is a new config key; a new choice is a new group file. |
| Submodule | Env changes go to the fork `AdrienBolling/MA-Craftax` on their own branch + PR there; bumping the pointer in `third_party/MA-Craftax` is a separate `build(env): ...` PR here. Never commit inside the submodule from this repo's branches. Clone with `--recurse-submodules`. |
| Imports | MA-Craftax internals via the top-level `craftax_ma`, `craftax_coop`, `environment_base` names; only `craftax.craftax_env` from the `craftax.` namespace (the two namespaces give distinct classes). Never install PyPI `craftax`. |
| GPU | Tests always run on CPU (`tests/conftest.py` sets `JAX_PLATFORMS=cpu`). When sharing the card with other work: `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Single RTX 4080 SUPER, 16 GB. |
| Artefacts | Keeper checkpoints: `dvc add` into `~/model-store` (remote `store`), metrics as `metrics/*.json`. `outputs/`, `multirun/`, `wandb/`, `checkpoints/`, `runs/` are git-ignored. |
| Tool pins | `ruff` / `ty` versions in `[dependency-groups].dev`, the hook revs in `.pre-commit-config.yaml` and the uv version in `ci.yml` move together, in a `chore(tooling)` PR. |

## Paths (PROTECTED)

| Path | Role |
|---|---|
| `src/mamora/cli.py` | Hydra entrypoint `mamora-run` (random-policy rollout for now) |
| `src/mamora/envs/factory.py` | `make_env(name)` over MA-Craftax; documents the import rule |
| `src/mamora/rollout.py` | jitted vmapped random rollout, `RolloutStats` |
| `src/mamora/paths.py` | `REPO_ROOT`, `CONF_DIR` (absolute Hydra config path) |
| `conf/` | Hydra root `config.yaml` + group `env/` |
| `tests/` | pytest, CPU only; `conftest.py` forces `JAX_PLATFORMS=cpu` |
| `third_party/MA-Craftax` | submodule → `AdrienBolling/MA-Craftax` (fork of BaselOmari/MA-Craftax, MIT); uv workspace member, editable dep `ma-craftax` |
| `.pre-commit-config.yaml` | hygiene, uv-lock, ruff, ty (`uv check --frozen`), conventional commits, pytest on pre-push |
| `.github/workflows/ci.yml` | job `ci`: ruff check/format, ty, pytest (CPU, `UV_NO_GROUP=gpu`); required by branch protection |
| `.dvc/` | DVC config; default remote `store` (ssh), local override in untracked `config.local` |
| `outputs/`, `multirun/` | Hydra run dirs (ignored); `checkpoint_dir` = `${hydra:runtime.output_dir}/checkpoints` |

## Status

| Area | State |
|---|---|
| Bootstrap | `chore/bootstrap` PR: standards, env, hydra runner, CI |
| Env | MA-Craftax fork packaged (hatchling `dev-mode-dirs`), MA + Coop run on GPU, smoke test on CPU (~20 s) |
| Algorithms | none yet |
| Branch protection | enabled once the `ci` check exists on the bootstrap PR |

## Key decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-07 | Python 3.12 (`requires-python >=3.12,<3.14`) | upstream MA-Craftax developed on 3.12; gymnax caps `<3.14` |
| 2026-09-07 | jax 0.11.1 stack (flax 0.12.9, chex 0.1.92, gymnax 0.0.9) | resolved naturally by uv; MA-Craftax verified to reset/step on it; no cap needed |
| 2026-09-07 | jaxmarl from git, pinned rev `b0c4d77b` | PyPI jaxmarl 0.1.0 pins `jax<=0.4.38` |
| 2026-09-07 | Fork + submodule + `dev-mode-dirs = [".", "craftax"]` | upstream has no packaging and imports both `craftax.*` and top-level `craftax_ma.*` |
| 2026-09-07 | Submodule as uv workspace member (editable) | uv's default for an in-repo editable; root `tool.uv.sources` then also govern the fork's deps |
| 2026-09-07 | `gpu` dependency group, default-on | plain `uv sync` keeps CUDA locally; CI sets `UV_NO_GROUP=gpu` |
| 2026-09-07 | ty via `uv check --frozen` hook | whole-project check, no lock mutation, no env rebuild |
| 2026-09-07 | Absolute Hydra `config_path` from `paths.CONF_DIR` | relative paths break under the console script (`__module__ != "__main__"`) |
| 2026-09-07 | Linear history, rebase/squash merges only | keeps the scoped commits visible on `main` |

## Blockers / Warnings

- `import jaxmarl` prints two lines to stdout and imports every JaxMARL env (slow import); never parse CLI stdout.
- The upstream env ctor's `num_agents` is dead: agent count comes from `StaticEnvParams.player_count` (MA 2, Coop 3). Configurable player count is a future PR.
- Craftax-Coop returns the same reward to every agent (team reward); Craftax-MA rewards are per agent.
- Pre-push runs the full test suite (~20 s on CPU); CI runs on 2-core runners, so expect a few minutes.

## Next steps

1. User merges `chore/bootstrap`.
2. `feat(rewards)`: multi-objective (per-achievement vector) reward wrapper around the env step.
3. `feat(algos)`: IPPO baseline port from `third_party/MA-Craftax/baselines/` behind Hydra config.
4. `feat(logging)`: wandb toggle in `conf/`.
