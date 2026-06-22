# Selective Risk Control after Corrupted Training — CIFAR Harness

This repository is a minimal CIFAR-level experimental harness for **Selective Risk Control after Corrupted Training**.

It supports:

1. Training a CIFAR classifier on corrupted labels.
2. Holding out a small trusted clean calibration set.
3. Risk-buffered proposal selection.
4. Independent Clopper-Pearson accepted-risk certification.
5. Certified coverage reporting at target risk levels.

The core experimental question is:

> At a target accepted selective risk `alpha`, can we certify non-trivial coverage after corrupted training?

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version first.

---

## Quick unit test

```bash
python -m pytest tests/test_certify.py
```

or without pytest:

```bash
python tests/test_certify.py
```

---

## 1. Train a corrupted CIFAR classifier

Example: CIFAR-10 with 35% symmetric noise.

```bash
python -m srcc.train \
  --dataset cifar10 \
  --data-root ./data \
  --out-dir runs/cifar10_sym35_seed0 \
  --noise-type symmetric \
  --noise-rate 0.35 \
  --model resnet18 \
  --epochs 120 \
  --batch-size 128 \
  --lr 0.1 \
  --trusted-prop-size 2500 \
  --trusted-cert-size 2500 \
  --seed 0
```

For a fast smoke run on CPU:

```bash
python -m srcc.train \
  --dataset cifar10 \
  --data-root ./data \
  --out-dir runs/smoke \
  --noise-type symmetric \
  --noise-rate 0.35 \
  --model small_cnn \
  --epochs 1 \
  --batch-size 128 \
  --trusted-prop-size 500 \
  --trusted-cert-size 500 \
  --max-train-samples 2000 \
  --seed 0
```

Training saves logits for proposal, certification, and test splits:

```text
runs/<run>/logits_prop.npy
runs/<run>/labels_prop.npy
runs/<run>/logits_cert.npy
runs/<run>/labels_cert.npy
runs/<run>/logits_test.npy
runs/<run>/labels_test.npy
runs/<run>/metadata.json
```

---

## 2. Run accepted-risk certification

```bash
python -m srcc.certify_run \
  --run-dir runs/cifar10_sym35_seed0 \
  --alpha 0.05 \
  --delta 0.05 \
  --gammas 0.5 0.7 1.0 \
  --scores msp entropy margin energy \
  --num-thresholds 200
```

The script prints and saves a certification report.

Main metrics:

- `certified`: whether CP UCB <= alpha.
- `cert_risk_ucb`: accepted-risk upper confidence bound.
- `cert_coverage_lcb`: lower confidence bound for deployment coverage.
- `test_coverage`: empirical clean test coverage.
- `test_risk`: empirical accepted selective risk on clean test data.
- `gamma`: proposal risk buffer multiplier, using proposal constraint `risk <= gamma * alpha`.

---

## 3. Run alpha sweep

```bash
python -m srcc.certify_run \
  --run-dir runs/cifar10_sym35_seed0 \
  --alpha 0.05 0.10 \
  --delta 0.05 \
  --gammas 0.5 0.7 1.0 \
  --scores msp entropy margin energy \
  --num-thresholds 200
```

---

## 4. Suggested AAAI PoC matrix

Minimum matrix:

| Dataset | Noise | Rate | Seeds |
|---|---|---:|---:|
| CIFAR-10 | symmetric | 0.35 | 3 |
| CIFAR-10 | asymmetric | 0.20 | 3 |
| CIFAR-100 | symmetric | 0.35 | 3 |

Targets:

```text
alpha in {0.05, 0.10}
gamma in {0.5, 0.7, 1.0}
```

Pass/fail heuristic:

| Result | Interpretation |
|---|---|
| certified coverage LCB > 30% at alpha=5% | strong AAAI signal |
| 15–30% at alpha=5% | usable with honest framing |
| only alpha=10% works | pragmatic safety framing |
| near-zero certified coverage | do not pitch as a certification success |

---

## Conceptual notes

The trusted split is **not** used to retrain the model. It is used to certify accepted predictions from a classifier already trained on corrupted labels.

The final guarantee is score-agnostic. MSP, entropy, margin, and energy are proposal scores only; the finite-sample risk certificate comes from the independent clean certification split.
