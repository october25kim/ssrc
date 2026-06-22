#!/usr/bin/env bash
set -euo pipefail

echo "[git] start-of-day sync"
git rev-parse --is-inside-work-tree >/dev/null
CURRENT_BRANCH=$(git branch --show-current)
echo "[git] branch: ${CURRENT_BRANCH}"
git fetch --all --prune
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  git pull --ff-only
else
  echo "[git] no upstream configured; skipping pull."
fi
git status --short
