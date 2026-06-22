#!/usr/bin/env bash
set -euo pipefail

bash scripts/docker_run.sh "python -m srcc.train \
  --dataset cifar10 \
  --data-root /data \
  --out-dir runs/smoke_cifar \
  --noise-type symmetric \
  --noise-rate 0.35 \
  --model small_cnn \
  --epochs 1 \
  --batch-size 128 \
  --trusted-prop-size 500 \
  --trusted-cert-size 500 \
  --max-train-samples 2000 \
  --num-workers 0 \
  --seed 0 && \
python -m srcc.certify_run \
  --run-dir runs/smoke_cifar \
  --alpha 0.05 0.10 \
  --delta 0.05 \
  --gammas 0.5 0.7 1.0 \
  --scores msp entropy margin energy \
  --num-thresholds 50"
