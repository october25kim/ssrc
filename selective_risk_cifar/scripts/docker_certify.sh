#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-runs/cifar10_sym35_seed0}
ALPHA=${ALPHA:-"0.05 0.10"}
DELTA=${DELTA:-0.05}
GAMMAS=${GAMMAS:-"0.5 0.7 1.0"}
SCORES=${SCORES:-"msp entropy margin energy"}
NUM_THRESHOLDS=${NUM_THRESHOLDS:-200}

bash scripts/docker_run.sh "python -m srcc.certify_run \
  --run-dir ${RUN_DIR} \
  --alpha ${ALPHA} \
  --delta ${DELTA} \
  --gammas ${GAMMAS} \
  --scores ${SCORES} \
  --num-thresholds ${NUM_THRESHOLDS}"
