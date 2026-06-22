#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-runs/cifar10_sym35_seed0}
DATASET=${DATASET:-cifar10}
NOISE_TYPE=${NOISE_TYPE:-symmetric}
NOISE_RATE=${NOISE_RATE:-0.35}
SEED=${SEED:-0}
MODEL=${MODEL:-resnet18}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.1}
TRUSTED_PROP_SIZE=${TRUSTED_PROP_SIZE:-2500}
TRUSTED_CERT_SIZE=${TRUSTED_CERT_SIZE:-2500}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP_FLAG=${AMP_FLAG:---amp}

bash scripts/docker_run.sh "python -m srcc.train \
  --dataset ${DATASET} \
  --data-root /data \
  --out-dir ${RUN_DIR} \
  --noise-type ${NOISE_TYPE} \
  --noise-rate ${NOISE_RATE} \
  --model ${MODEL} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --trusted-prop-size ${TRUSTED_PROP_SIZE} \
  --trusted-cert-size ${TRUSTED_CERT_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --seed ${SEED} \
  ${AMP_FLAG}"
