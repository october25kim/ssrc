# Daily Experiment Run Prompt — SRCC

Read `AGENTS.md` and `CLAUDE.md` before acting.

Today’s objective is to run/extend the AAAI CIFAR-level certification experiments for **Selective Risk Control after Corrupted Training**.

## Start

Run:

```bash
bash scripts/git_start_day.sh
bash scripts/docker_build.sh
bash scripts/docker_smoke.sh
```

If smoke fails, stop and diagnose before running CIFAR training.

## Priority experiment order

1. CIFAR-10 symmetric 35%, seed 0.
2. CIFAR-10 asymmetric 20%, seed 0.
3. Repeat seeds 1 and 2.
4. CIFAR-100 symmetric 35%, seeds 0,1,2.

## Canonical commands

```bash
DATASET=cifar10 NOISE_TYPE=symmetric NOISE_RATE=0.35 SEED=0 \
  bash scripts/docker_train.sh runs/cifar10_sym35_seed0

bash scripts/docker_certify.sh runs/cifar10_sym35_seed0
```

```bash
DATASET=cifar10 NOISE_TYPE=asymmetric NOISE_RATE=0.20 SEED=0 \
  bash scripts/docker_train.sh runs/cifar10_asym20_seed0

bash scripts/docker_certify.sh runs/cifar10_asym20_seed0
```

## Success metric

Inspect `certification_results.csv`. Report:

```text
best certified score_name/gamma at alpha=0.05 and alpha=0.10
cert_risk_ucb
cert_coverage_lcb
test_risk
test_coverage
cert_n/cert_k
```

Interpretation:

```text
cert_coverage_lcb > 0.30 at alpha=0.05: strong
0.15-0.30: usable
only alpha=0.10 works: pragmatic framing
near-zero: certification success claim is risky
```

## End

Run:

```bash
bash scripts/git_end_day.sh "daily srcc experiment update"
```

Do not commit data/runs/outputs/checkpoints.
