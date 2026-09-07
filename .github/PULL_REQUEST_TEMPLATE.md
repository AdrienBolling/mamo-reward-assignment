## What

<!-- one paragraph: the single concern this PR addresses -->

## Why

<!-- motivation, link to issue / experiment note if any -->

## Checks

- [ ] `uv run pre-commit run --all-files` clean
- [ ] `uv run pytest` green (CPU)
- [ ] run parameters live in `conf/` (no hard-coded hyperparameters, seeds or paths)
- [ ] no new `# noqa` / `# ty: ignore` / rule changes, or each one authorized and justified
