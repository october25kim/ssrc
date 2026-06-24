# AGENTS.md — SRCC Journal Version

## Project

This repository supports the journal paper:

**Certifying Accepted Predictions of Classifiers Trained with Corrupted Labels**

Internal acronym:

**SRCC — Selective Risk Certification after Corrupted Training**

Primary target journal:

- **Information Sciences**

Secondary target journal:

- **Neural Networks**

This project is no longer AAAI-first. Use the journal framing: **accepted prediction certification**, **intelligent classifier reliability**, **trusted calibration**, and **finite-sample reliability**.

## Startup rule

Before diagnosing, coding, editing, or running experiments:

1. Read `AGENTS.md`.
2. Read `CLAUDE.md` if present.
3. Summarize the active constraints before making changes.

If any instruction still frames the project as AAAI-first, RC-OWPL, federated learning, pseudo-labeling, open-world SSL, or noisy-label robust training as the primary object, flag the mismatch and apply only a minimal instruction-file correction.

## Paper identity

This is a post-hoc reliability certification paper.

Given a classifier already trained on corrupted labels, certify which deployment-time predictions can be accepted under clean labels using a small trusted calibration set.

Core accepted risk:

```text
R_sel(h) = P(y_hat(X) != Y | A_h(X) = 1)
```

Goal:

```text
R_sel(h) <= alpha
```

while maintaining nontrivial certified accepted coverage.

## Not this project

Do not treat this as:

- noisy-label robust training,
- pseudo-labeling,
- RC-OWPL,
- federated learning,
- generic calibration,
- generic uncertainty thresholding.

Training repair is not the main contribution. Trusted clean data is used for proposal and certification, not for model repair unless running an explicitly labeled baseline.

## Journal framing

Prefer:

- accepted prediction certification,
- post-hoc reliability certification,
- corrupted-trained classifiers,
- intelligent classifier reliability,
- trusted calibration,
- finite-sample accepted-risk certificate,
- certified accepted coverage.

Use “risk control” mainly in technical sections.

## Core pipeline

```text
corrupted-trained classifier
-> risk-buffered proposal
-> independent certification
-> certified accepted coverage
```

The trusted clean set must be split into:

```text
C_prop
C_cert
```

The test set is evaluation-only.

Never use test labels for threshold, score, gamma, proposal, certification, or model selection.

## Proposal logic

For candidate selector `h` on proposal split:

```text
prop_n = number of accepted proposal examples
prop_k = number of accepted proposal errors
prop_risk = prop_k / prop_n
prop_coverage = prop_n / m_prop
```

Risk-buffered proposal:

```text
prop_risk <= gamma * alpha
```

Default gamma values:

```text
0.5, 0.7, 1.0
```

`gamma = 1.0` is the no-buffer baseline.

If `prop_n == 0`, do not treat `prop_risk` as zero. Zero-accept proposals must not win selection.

## Certification logic

On independent certification split:

```text
cert_n = number of accepted certification examples
cert_k = number of accepted certification errors
cert_risk_ucb = Clopper-Pearson UCB(cert_k, cert_n, delta_risk)
cert_coverage_lcb = Clopper-Pearson LCB(cert_n, m_cert, delta_coverage)
```

Certification flag:

```text
certified = cert_n > 0 and cert_risk_ucb <= alpha
```

Certified accepted coverage:

```text
certified_coverage_at_alpha =
    cert_coverage_lcb if certified else 0.0
```

If `cert_n == 0`:

```text
cert_risk_ucb = 1.0
cert_coverage_lcb = 0.0
certified = False
certified_coverage_at_alpha = 0.0
reason = "zero_cert_accepts"
```

Zero accepted samples are never success.

## Delta accounting

Preferred journal default:

```text
delta_total = 0.05
delta_risk = delta_total / 2
delta_coverage = delta_total / 2
delta_allocation = "joint_split"
```

If legacy risk-only mode is used, record:

```text
delta_allocation = "risk_only_legacy"
```

## Multiple-row certificates

No union penalty is valid only when a single selector is selected before certification.

If multiple score/gamma rows are certified and then the best row is chosen using certification results, use simultaneous correction or mark certificates as marginal individual-row certificates.

Required fields:

```text
certificate_scope
n_certified_candidates
```

Allowed scopes:

```text
single_selector
individual_row
simultaneous_rows
```

## Required result schema

Every certification result row should include:

```text
dataset
noise_type
noise_rate
seed
alpha
gamma
score_name
threshold
threshold_direction
prop_n
prop_k
prop_coverage
prop_risk
cert_n
cert_k
cert_risk_ucb
cert_coverage_lcb
certified
certified_coverage_at_alpha
test_coverage
test_risk
delta_total
delta_risk
delta_coverage
delta_allocation
certificate_scope
n_certified_candidates
reason
```

## Reason codes

Use clear reason codes:

```text
certified
no_proposal_candidate
zero_prop_accepts
zero_cert_accepts
insufficient_cert_n
high_cert_error_ucb
not_certified
invalid_split
invalid_delta
numerical_error
```

If `cert_k == 0` but `cert_risk_ucb > alpha`, prefer:

```text
insufficient_cert_n
```

## Required tests

Maintain tests for:

- Clopper-Pearson risk UCB boundary cases,
- Clopper-Pearson coverage LCB boundary cases,
- zero accepted certification convention,
- zero accepted proposal handling,
- zero-error minimum accepted count,
- gamma-buffer proposal logic,
- certified coverage definition,
- split separation,
- multiple-row certificate scope.

## Lightweight verification

Prefer Docker-first checks when available:

```bash
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
bash scripts/docker_cifar_smoke_train.sh
```

Do not launch long CIFAR experiments unless explicitly requested.

## Experiment priorities

Journal minimum:

1. Docker / smoke / fake-logit certification (`scripts/smoke_certification_with_fake_logits.py`, `scripts/docker_smoke.sh`).
2. CIFAR-10 clean.
3. CIFAR-10 symmetric corruption.
4. CIFAR-10 asymmetric corruption.
5. Seeds 0, 1, 2.
6. CIFAR-100 synthetic corruption.
7. Score ablations: MSP, MaxLogit, margin, entropy, energy.
8. Gamma ablations: 0.5, 0.7, 1.0.
9. Baselines: full coverage, naive empirical threshold, no-buffer proposal, LTT/Bonferroni, clean-trained upper bound.

Strong extensions:

1. CIFAR-N or another real noisy-label benchmark.
2. Temperature scaling + SRCC.
3. GCE / Co-teaching / DivideMix + SRCC.
4. Calibration size sensitivity.
5. Failure mode heatmap.

## Forbidden actions

Do not:

- report accuracy as the main metric,
- use test labels for proposal or certification,
- mix proposal and certification splits,
- count zero accepted samples as success,
- hide coverage collapse,
- overclaim from smoke runs,
- present robust training as the main contribution,
- make `T^{-T}` correction the main method,
- perform unrelated refactors,
- commit generated runs or large artifacts.

Do not commit:

```text
runs/
data/
outputs/
checkpoints/
logs/
wandb/
*.pt
*.pth
*.npy
*.npz
```

## Final report format

```text
Implementation summary:
- ...

Files changed:
- ...

Theory / journal-framing items preserved:
- accepted prediction certification:
- Information Sciences-first framing:
- proposal/certification split:
- CP risk UCB:
- coverage LCB:
- risk-buffered proposal:
- zero-accept convention:
- certified_coverage_at_alpha:
- reason codes:

Commands run:
- ...

Key outputs:
- ...

Known limitations:
- ...

Next recommended action:
- ...
```
