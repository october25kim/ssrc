#!/usr/bin/env bash
set -euo pipefail

MSG=${1:-"daily srcc experiment update"}

echo "[git] running preflight"
bash scripts/preflight_repo.sh

echo "[git] running smoke verification"
if command -v docker >/dev/null 2>&1; then
  bash scripts/docker_smoke.sh
else
  PYTHONPATH=. python -m pytest -q
  python scripts/smoke_certification_with_fake_logits.py
fi

echo "[git] staging source/config/script/doc changes"
git add -A

echo "[git] status after staging"
git status --short

if git diff --cached --quiet; then
  echo "[git] no staged changes; nothing to commit."
else
  git commit -m "$MSG"
fi

if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  git pull --rebase
  git push
  echo "[git] pushed commit: $(git rev-parse --short HEAD)"
else
  echo "[git] no upstream configured; commit exists locally: $(git rev-parse --short HEAD)"
  echo "[git] set upstream manually, e.g.: git push -u origin $(git branch --show-current)"
fi
