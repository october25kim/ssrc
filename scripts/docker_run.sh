#!/usr/bin/env bash
set -euo pipefail

IMAGE=${SRCC_DOCKER_IMAGE:-srcc:latest}
CMD=${*:-bash}

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia\|nvidia"; then
    GPU_ARGS=(--gpus all)
  fi
fi

mkdir -p data outputs runs logs checkpoints

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --shm-size=16g \
  -e PYTHONPATH=/workspace \
  -e WANDB_DISABLED=true \
  -v "$PWD":/workspace \
  -v "$PWD/data":/data \
  -v "$PWD/outputs":/outputs \
  -v "$PWD/runs":/workspace/runs \
  -w /workspace \
  "$IMAGE" \
  bash -lc "$CMD"
