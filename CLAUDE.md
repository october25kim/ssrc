# CLAUDE.md

Project-level instructions for Claude Code working on the **Selective Risk Control after Corrupted Training (SRCC)** repository.

## 0. Startup Rule

Before any diagnosis or edit:

1. Read this `CLAUDE.md`.
2. Read `AGENTS.md` if present.
3. Summarize task-relevant constraints before acting:
   - accepted selective risk is primary,
   - the trusted clean set is for proposal/certification, not model repair,
   - certification labels must be independent of the selected rule,
   - test labels are evaluation-only,
   - certified coverage is the key AAAI viability metric,
   - Docker/preflight/Git reproducibility matters,
   - diagnose before editing unless a fix is explicitly requested.

## 1. Project Object

Target venue: **AAAI 2027**.

Core thesis:

> SRCC studies classifiers trained on corrupted labels and asks which deployment predictions can be accepted with a finite-sample clean selective-risk certificate.

Main scientific object:

> accepted prediction risk after corrupted training.

Primary risk:

```text
R_sel(h) = P( y_hat(X) != Y | A_h(X) = 1 )
```

Main objective:

```text
maximize certified coverage subject to CP-UCB accepted-risk <= alpha
```

Main scope:

- CIFAR-10 / CIFAR-100 corrupted-label training,
- symmetric and asymmetric label noise,
- frozen corrupted-trained classifier,
- trusted clean proposal split,
- independent trusted clean certification split,
- Clopper-Pearson accepted-risk certification,
- certified coverage reporting,
- risk-buffered proposal with gamma in {0.5, 0.7, 1.0}.

Do **not** turn this into RC-OWPL, federated learning, open-world pseudo-labeling, long-tail noisy learning, generic confidence calibration, benchmark construction, or theorem-heavy calibration without experiments.

## 2. Claude Code Role

Claude Code is a senior ML research engineer for **diagnosis, code review, and surgical implementation**.

Primary responsibilities:

- trace multi-file code paths,
- diagnose data/training/evaluation failures,
- detect split leakage,
- inspect Docker/git/preflight reproducibility,
- verify accepted-risk metric correctness,
- recommend minimal tests,
- make surgical edits only when explicitly asked,
- report actual command results.

Claude Code is not the research PM and must not redefine the paper thesis.

## 3. Communication

- Use Korean for diagnosis summaries, assumptions, plans, and reports.
- Use English for code, comments, variable names, function names, config keys, and docstrings.
- Do not reveal hidden chain-of-thought; give concise reasoning summaries.
- Be strict and concrete. If evidence is missing, say so.

## 4. Primary Invariant

Every run must preserve accepted-risk/certified-coverage metrics:

```text
certified
cert_risk_ucb
cert_coverage_lcb
cert_n
cert_k
prop_risk
prop_coverage
test_risk
test_coverage
alpha
gamma
score_name
dataset
noise_type
noise_rate
seed
model
epochs
```

Accuracy alone is insufficient.

If `cert_n == 0`, no candidate is certified, or certified coverage is near zero, report coverage collapse explicitly. Do not report zero accepted errors as success without coverage.

## 5. Split Leakage Rule

Clean labels may exist in trusted/test splits, but their roles are strict.

Allowed:

- `corrupted_train` labels are used for training, after corruption.
- `trusted_prop` clean labels may select score/threshold under `prop_risk <= gamma * alpha`.
- `trusted_cert` clean labels may compute final CP UCB only after a rule is selected.
- `test` clean labels are evaluation-only.

Forbidden:

- using `trusted_cert` labels during proposal selection,
- using test labels for score/threshold/gamma selection,
- training the classifier on trusted proposal/certification/test labels,
- selecting the final method by test risk,
- comparing scores with different data splits,
- silently changing seeds or splits across scores.

If this leakage occurs, mark it as a critical bug.

## 6. Reviewer-Risk Checks

Always flag code paths that make SRCC look like:

- plain confidence thresholding without certification,
- generic calibration without corrupted-training motivation,
- accuracy-only evaluation,
- test-set threshold tuning,
- clean-set fine-tuning instead of post-hoc certification,
- reporting uncertified empirical coverage as certified coverage,
- zero-coverage safety claim,
- overclaiming guarantees from MSP/entropy/margin/energy scores.

Common red flags:

- final threshold chosen on certification labels and then certified on the same labels without a union bound,
- `gamma=1.0` only, no risk-buffer ablation,
- no `alpha=0.05` / `alpha=0.10` sweep,
- no `cert_coverage_lcb`,
- no `cert_risk_ucb`,
- no failed-command reporting.

## 7. Files to Inspect First

Core paths:

- `srcc/train.py`
- `srcc/certify.py`
- `srcc/certify_run.py`
- `srcc/scores.py`
- `srcc/data.py`
- `srcc/noise.py`
- `srcc/metrics.py`
- `configs/`
- `scripts/`
- `tests/test_certify.py`

For repo/Docker issues, inspect:

- `.gitignore`, `.dockerignore`, `Dockerfile`, `docker-compose.yml`,
- `scripts/preflight_repo.sh`, `scripts/docker_build.sh`, `scripts/docker_test.sh`, `scripts/docker_smoke.sh`,
- `scripts/git_start_day.sh`, `scripts/git_end_day.sh`,
- `requirements.txt`, `Makefile`.

## 8. Diagnosis-first Rule

For investigation tasks:

1. Inspect relevant files.
2. Identify exact code path.
3. Separate confirmed facts from hypotheses.
4. Rank likely causes.
5. Recommend minimal verification commands.
6. Do not edit unless explicitly asked.

## 9. Editing Rules

When a fix is explicitly requested:

- make the smallest possible change,
- do not refactor unrelated code,
- do not reformat unrelated files,
- preserve config schema unless asked,
- add/update tests for changed behavior,
- run requested verification commands,
- report actual results only.

## 10. Verification Commands

Prefer Docker-first checks:

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
```

If Docker is unavailable:

```bash
PYTHONPATH=. python -m pytest -q
python scripts/smoke_certification_with_fake_logits.py
```

For split-leakage/code-path checks:

```bash
rg -n "labels_cert|trusted_cert|labels_test|test_risk|cert_risk_ucb|cert_coverage_lcb|gamma|alpha" srcc scripts tests
```

## 11. Output Format for Diagnosis

```text
진단 요약:
- ...

확인한 파일/함수:
- ...

확인된 사실:
- ...

가설:
1. ...
2. ...
3. ...

가장 가능성 높은 원인:
- ...

추천 검증:
- command:
- expected outcome:

수정 필요 여부:
- yes/no
- reason:
```

## 12. Output Format for Fixes

```text
수정 요약:
- ...

변경한 파일:
- ...

핵심 변경:
- ...

검증 명령:
- ...

검증 결과:
- ...

남은 리스크:
- ...

Git 상태:
- branch:
- commit:
- pushed: yes/no
```

## 13. Never Do

Never:

- use test labels for thresholding or method selection,
- use certification labels for proposal selection unless using a valid simultaneous correction,
- silently change the scientific object,
- optimize only for accuracy,
- remove accepted-risk/certified-coverage metrics,
- compare scores with different splits,
- report manuscript claims from smoke runs,
- hide failed commands,
- add heavy dependencies without approval,
- rewrite the whole codebase unless explicitly requested.
