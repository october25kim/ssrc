#!/usr/bin/env bash
set -euo pipefail
IMAGE=${SRCC_DOCKER_IMAGE:-srcc:latest}
docker build -t "$IMAGE" .
