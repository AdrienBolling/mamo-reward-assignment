#!/usr/bin/env bash
# pre-push guard: refuse any push that targets main on the remote.
#
# GitHub's branch protection accepts a direct push of a PR's head commit when that
# PR already satisfies every rule (green ci, no approvals required) and marks the
# PR as merged, so the server cannot enforce "main changes only through a merge
# done by the user". This hook does, for every clone that ran `pre-commit install`.
# pre-commit exports PRE_COMMIT_REMOTE_BRANCH as the remote ref being pushed to.
set -euo pipefail

if [[ "${PRE_COMMIT_REMOTE_BRANCH:-}" == "refs/heads/main" ]]; then
  echo "error: pushing to 'main' is forbidden; push a branch and open a pull request." >&2
  exit 1
fi
