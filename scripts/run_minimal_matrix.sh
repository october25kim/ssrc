#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data}
RUN_ROOT=${RUN_ROOT:-./runs}
SEEDS=${SEEDS:-"0 1 2"}

for SEED in ${SEEDS}; do
  python -m srcc.train \
    --dataset cifar10 \
    --data-root "${DATA_ROOT}" \
    --out-dir "${RUN_ROOT}/cifar10_sym35_seed${SEED}" \
    --noise-type symmetric \
    --noise-rate 0.35 \
    --model resnet18 \
    --epochs 120 \
    --batch-size 128 \
    --lr 0.1 \
    --trusted-prop-size 2500 \
    --trusted-cert-size 2500 \
    --seed "${SEED}" \
    --amp

  python -m srcc.certify_run \
    --run-dir "${RUN_ROOT}/cifar10_sym35_seed${SEED}" \
    --alpha 0.05 0.10 \
    --delta 0.05 \
    --gammas 0.5 0.7 1.0 \
    --scores msp entropy margin energy \
    --num-thresholds 200

  python -m srcc.train \
    --dataset cifar10 \
    --data-root "${DATA_ROOT}" \
    --out-dir "${RUN_ROOT}/cifar10_asym20_seed${SEED}" \
    --noise-type asymmetric \
    --noise-rate 0.20 \
    --model resnet18 \
    --epochs 120 \
    --batch-size 128 \
    --lr 0.1 \
    --trusted-prop-size 2500 \
    --trusted-cert-size 2500 \
    --seed "${SEED}" \
    --amp

  python -m srcc.certify_run \
    --run-dir "${RUN_ROOT}/cifar10_asym20_seed${SEED}" \
    --alpha 0.05 0.10 \
    --delta 0.05 \
    --gammas 0.5 0.7 1.0 \
    --scores msp entropy margin energy \
    --num-thresholds 200

done
