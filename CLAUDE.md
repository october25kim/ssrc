# CLAUDE.md — SRCC Journal Version

## Role

You are assisting with the journal paper:

**Certifying Accepted Predictions of Classifiers Trained with Corrupted Labels**

Internal acronym:

**SRCC — Selective Risk Certification after Corrupted Training**

Primary target:

- **Information Sciences**

Secondary target:

- **Neural Networks**

This is no longer an AAAI-first project. Prioritize journal-level completeness, careful assumptions, broader empirical validation, failure analysis, and reliability framing.

## Read first

Before doing any work:

1. Read `AGENTS.md`.
2. Read this file.
3. Confirm that the project is framed as an Information Sciences-first journal submission.
4. Confirm that the paper object is accepted prediction certification after corrupted training.

If outdated instructions center AAAI, RC-OWPL, pseudo-labeling, federated learning, or noisy-label robust training as the primary object, flag the mismatch and propose a minimal correction.

## Paper identity

This paper is about **post-hoc reliability certification**.

Given a classifier trained with corrupted labels, decide which deployment-time predictions can be accepted while certifying clean accepted risk using a small trusted calibration set.

Core accepted risk:

```text
R_sel(h) = P(y_hat(X) != Y | A_h(X) = 1)
```

Coverage:

```text
Cov(h) = P(A_h(X) = 1)
```

Goal:

```text
R_sel(h) <= alpha
```

with nontrivial certified coverage.

## Journal positioning

Use this thesis:

```text
Classifiers trained with corrupted labels may produce confident but unreliable predictions at deployment time. This paper studies how to certify the reliability of accepted predictions from such classifiers using a small trusted calibration set, without retraining or repairing the model.
```

Short positioning:

```text
The trusted clean set is used as a certification resource, not as a repair set.
```

Preferred language:

- accepted prediction certification,
- intelligent classifier reliability,
- post-hoc reliability certification,
- corrupted-trained classifier deployment,
- trusted clean calibration,
- finite-sample accepted-risk certificate,
- certified accepted coverage.

Avoid leading with “risk control” in high-level framing unless explaining the technical risk.

## What not to do

Do not frame the paper as:

- solving noisy-label learning,
- improving corrupted training,
- generic uncertainty thresholding,
- pseudo-labeling,
- open-world learning,
- federated learning,
- model calibration only,
- conformal prediction set construction only.

Do not make accuracy the primary success metric.

Do not treat test risk as the guarantee.

Do not hide certification failure or coverage collapse.

## Theory that must be preserved

1. **Non-identifiability**

   Clean accepted risk cannot be distribution-free certified from corrupted labels alone without trusted clean calibration or a valid corruption model.

2. **Confidence deformation**

   Under class-conditional corruption:

   ```text
   corrupted posterior = T^T clean posterior
   ```

   Corrupted training can deform confidence-correctness ranking. This motivates certification but is not the guarantee.

3. **Fixed-selector certification**

   For a selector fixed independently of certification data, accepted errors follow a binomial model conditional on accepted count.

4. **Proposal/certification split**

   Select candidate accept rule on `C_prop`, certify on independent `C_cert`.

5. **Score-agnostic guarantee**

   MSP, entropy, margin, MaxLogit, energy, or other scores may propose accepted sets. Guarantee comes from certification.

6. **Risk-buffered proposal**

   ```text
   prop_risk <= gamma * alpha
   gamma in {0.5, 0.7, 1.0}
   ```

   `gamma=1.0` is the no-buffer baseline.

7. **Joint risk and coverage certificate**

   Report both:

   ```text
   cert_risk_ucb
   cert_coverage_lcb
   ```

   Define:

   ```text
   certified_coverage_at_alpha =
       cert_coverage_lcb if certified else 0.0
   ```

8. **Zero-accept failure**

   If `cert_n == 0`:

   ```text
   cert_risk_ucb = 1.0
   cert_coverage_lcb = 0.0
   certified = False
   certified_coverage_at_alpha = 0.0
   reason = "zero_cert_accepts"
   ```

## Code review priorities

Check:

- no test-label leakage,
- proposal/certification/test split separation,
- CP risk UCB correctness,
- coverage LCB correctness,
- zero accepted sample convention,
- risk-buffered proposal condition,
- gamma grid,
- certified coverage definition,
- delta accounting,
- multiple-row certificate scope,
- reason codes,
- output schema.

Do not approve a patch if `cert_n == 0` can be marked certified.

Do not approve a patch if `test_risk` affects threshold, score, gamma, proposal, certification, or model selection.

## Required output columns

Result rows should include:

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
reason
```

Journal-ready extended rows should also include:

```text
delta_total
delta_risk
delta_coverage
delta_allocation
certificate_scope
n_certified_candidates
```

## Journal experiment expectations

Minimum evidence:

- Docker / smoke / fake-logit certification (`scripts/smoke_certification_with_fake_logits.py`, `scripts/docker_smoke.sh`),
- CIFAR-10 clean,
- CIFAR-10 symmetric corruption,
- CIFAR-10 asymmetric corruption,
- multiple seeds,
- CIFAR-100 synthetic corruption,
- score ablation,
- gamma ablation,
- naive empirical threshold baseline,
- no-buffer baseline,
- full-coverage baseline,
- clean-trained upper bound,
- LTT / Bonferroni-style risk-control baseline if feasible.

Strongly recommended:

- CIFAR-N or real noisy-label benchmark,
- calibration size sensitivity,
- failure mode analysis,
- one robust training + SRCC baseline,
- temperature scaling + SRCC.

## Tables and figures to support

Aim for:

1. Problem and SRCC pipeline figure.
2. Confidence deformation / risk-coverage figure.
3. Main certified accepted coverage table.
4. Comparison with selective/risk-control baselines.
5. Risk-buffered proposal effect figure.
6. Score-agnostic ablation table.
7. Calibration budget / feasibility table.
8. Failure mode heatmap or CP feasibility figure.

## Result interpretation format

```text
진단 요약:
- ...

확인한 명령:
- ...

핵심 결과:
- alpha=...
- gamma=...
- score=...
- cert_risk_ucb=...
- cert_coverage_lcb=...
- test_risk=...
- test_coverage=...

판정:
- strong go / moderate go / warning / fail

다음 행동:
- ...
```

Judgment guide:

```text
strong go:
- certified = True
- cert_risk_ucb <= alpha
- cert_coverage_lcb nontrivial
- test_risk consistent with certificate

moderate go:
- certified = True
- coverage modest but usable

warning:
- test risk looks good but cert_risk_ucb > alpha
- cert_k = 0 but cert_n too small
- coverage LCB near zero

fail:
- no proposal candidate
- zero cert accepts
- high cert risk UCB
- repeated coverage collapse in core regimes
```

## Manuscript wording rules

Use English for manuscript-ready text.

Use Korean for diagnosis and planning unless asked otherwise.

Safe claims:

```text
We certify accepted predictions from corrupted-trained classifiers.
The trusted clean set is used for certification rather than retraining.
The guarantee is score-agnostic and finite-sample.
The method reports both risk certificates and certified coverage.
Coverage collapse is an honest diagnostic when safe acceptance is not supported.
```

Unsafe claims:

```text
We solve noisy-label learning.
We repair corrupted training.
The classifier is calibrated.
The confidence score estimates correctness probability.
Test risk is the guarantee.
Certification works in all regimes.
```

## Git and artifact hygiene

Do not commit generated experiment artifacts or large outputs.

Never commit:

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

Example commit messages:

```text
Update SRCC instructions for journal submission
Align certification code with journal SRCC framing
```

## Final response format

```text
Summary:
- ...

Files changed:
- ...

Journal framing preserved:
- Information Sciences-first:
- accepted prediction certification:
- trusted calibration:
- no retraining as main method:

Certification invariants checked:
- proposal/cert/test split:
- CP risk UCB:
- coverage LCB:
- zero-accept failure:
- risk-buffered proposal:
- certified coverage metric:

Commands run:
- ...

Issues found:
- ...

Next recommended action:
- ...
```
