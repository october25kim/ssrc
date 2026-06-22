#!/usr/bin/env bash
set -euo pipefail

echo "[host] git"
git --version || true

echo "[host] docker"
docker --version || true

echo "[host] docker compose"
docker compose version || true

echo "[host] nvidia-smi"
nvidia-smi || true

echo "[host] docker info GPU hint"
docker info 2>/dev/null | grep -i "runtime\|nvidia" || true
