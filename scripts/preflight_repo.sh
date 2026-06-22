#!/usr/bin/env bash
set -euo pipefail

echo "[preflight] git status"
git status --short || true

echo "[preflight] checking for large tracked experiment artifacts"
if git ls-files | grep -E '^(data|outputs|runs|logs|checkpoints|wandb)/|\.(pt|pth|ckpt|npy|npz)$' >/tmp/srcc_tracked_artifacts.txt; then
  cat /tmp/srcc_tracked_artifacts.txt
  echo "ERROR: experiment artifacts are tracked. Remove them from git." >&2
  exit 1
fi

echo "[preflight] checking required instruction files"
test -f CLAUDE.md
test -f AGENTS.md

echo "[preflight] checking required scripts"
for f in scripts/docker_build.sh scripts/docker_test.sh scripts/docker_smoke.sh scripts/git_start_day.sh scripts/git_end_day.sh; do
  test -x "$f"
done

echo "[preflight] OK"
