# AGENTS.md

Project-level instructions for Codex, Cursor, and other coding agents working on the **Selective Risk Control after Corrupted Training (SRCC)** repository.

## 0. Startup Rule

Before coding or editing:

1. Read this `AGENTS.md`.
2. Read `CLAUDE.md` if present.
3. Summarize task-relevant constraints before acting:
   - accepted selective risk is primary,
   - the trusted clean set is for proposal/certification, not model repair,
   - certification labels must be independent of the selected rule,
   - test labels are evaluation-only,
   - certified coverage, not raw accuracy, is the central metric,
   - Docker/preflight/Git reproducibility matters,
   - use minimal, testable changes.

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

Do **not** turn this project into RC-OWPL, federated learning, open-world pseudo-labeling, long-tail noisy learning, generic confidence calibration, benchmark construction, or theorem-heavy calibration without experiments.

## 2. Agent Role

You are a senior ML research coding agent.

Responsibilities:

- implement runnable experiment code,
- preserve the SRCC scientific object,
- keep accepted-risk and certified-coverage metrics first-class,
- avoid leakage between train / proposal / certification / test splits,
- keep Docker runs reproducible,
- support fair score comparison under the same splits,
- keep changes small, typed, tested, and ablation-friendly,
- report actual command results.

You are not the research PM. Do not redefine the paper thesis.

## 3. Language and Style

- Use English for code, comments, docstrings, variable names, config keys, filenames, and commit-style summaries.
- Use Korean for human-facing reports unless asked otherwise.
- Do not reveal hidden chain-of-thought.
- Provide concise implementation summaries.
- Use PyTorch, type hints, deterministic seeds, and simple functions.
- Avoid hidden global state and heavy dependencies.

## 4. Core Implementation Invariant

Every run must make available the following fields in machine-readable form, preferably CSV/JSON:

```text
dataset
noise_type
noise_rate
seed
model
epochs
alpha
delta
gamma
score_name
threshold
certified
prop_n
prop_k
prop_coverage
prop_risk
cert_n
cert_k
cert_risk_ucb
cert_coverage_lcb
test_coverage
test_risk
test_correct_count
test_accepted_count
```

Accuracy alone is insufficient. The main table metric is:

```text
CertifiedCoverage@alpha = cert_coverage_lcb, among certified runs
```

If `cert_n == 0` or no candidate is certified, report coverage collapse explicitly. Do not let zero errors with zero accepted samples appear successful.

## 5. SRCC Theory/Method Requirements

The canonical method is:

```text
corrupted training -> score extraction -> risk-buffered proposal -> independent CP certification -> certified coverage report
```

Risk-buffered proposal:

```text
select h that maximizes proposal coverage subject to prop_risk <= gamma * alpha
```

Independent certification:

```text
certify selected h only on the certification split using Clopper-Pearson UCB
```

Allowed proposal scores:

```text
msp
entropy
margin
energy
```

Optional only:

```text
T-matrix correction
calibrated temperature scaling
advanced OOD/energy variants
```

These optional scores must never be described as the source of the finite-sample guarantee. The guarantee comes from the independent certification split.

## 6. Data and Split Rules

The repository should maintain distinct roles:

```text
corrupted_train: used to train classifier with corrupted labels
trusted_prop: clean labels allowed only for selecting score/threshold under gamma buffer
trusted_cert: clean labels allowed only for final CP certification
test: clean labels evaluation-only; never tune on test
```

Forbidden leakage:

- using `trusted_cert` labels during proposal,
- using test labels for threshold/score/gamma selection,
- training the classifier on trusted proposal/certification/test labels,
- silently changing splits between scores or baselines,
- reporting the best test result as if it were certified.

Allowed training signals:

- corrupted labels in `corrupted_train`,
- model logits,
- model probabilities,
- confidence,
- entropy,
- margin,
- energy score.

## 7. Evaluation Requirements

Core evaluation must include:

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
```

Required sweeps:

```text
alpha in {0.05, 0.10}
gamma in {0.5, 0.7, 1.0}
scores in {msp, entropy, margin, energy}
```

Minimum AAAI PoC matrix:

```text
cifar10 symmetric 0.35 seeds 0,1,2
cifar10 asymmetric 0.20 seeds 0,1,2
cifar100 symmetric 0.35 seeds 0,1,2
```

Pass/fail interpretation:

```text
cert_coverage_lcb > 0.30 at alpha=0.05: strong
0.15 <= cert_coverage_lcb <= 0.30 at alpha=0.05: usable
only alpha=0.10 works: pragmatic safety framing
near-zero certified coverage: do not pitch as certification success
```

## 8. Docker / Repository Discipline

Repository is Docker-first.

Rules:

- outputs go under `outputs/` or `runs/`,
- root `/data`, `/outputs`, `/runs`, `/logs`, `/checkpoints`, `/wandb` are not committed,
- source under `srcc/`, `scripts/`, `configs/`, `tests/` must be tracked,
- preflight should catch ignored/untracked source files,
- Docker tests and smoke should pass before experiment expansion,
- do not allow pip to overwrite CUDA-enabled torch in Docker.

Preferred commands:

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
```

Fallback if Docker is unavailable:

```bash
PYTHONPATH=. python -m pytest -q
python scripts/smoke_certification_with_fake_logits.py
```

## 9. Git Discipline

Start of day:

```bash
bash scripts/git_start_day.sh
```

End of day:

```bash
bash scripts/git_end_day.sh "short, concrete commit message"
```

Daily Git expectations:

1. Pull before starting work.
2. Run at least Docker smoke or local smoke before committing.
3. Commit source/config/script changes only.
4. Do not commit data, CIFAR downloads, runs, checkpoints, logs, or wandb files.
5. Pull with rebase before push.
6. Report the pushed commit hash.

## 10. Testing Requirements

Use pytest. Tests should be CPU-fast with tiny synthetic data.

Required coverage:

- Clopper-Pearson UCB correctness,
- certification split independence assumptions are respected by API,
- risk-buffered proposal behavior,
- empty accepted set handled as uncertified/collapse,
- score functions return finite arrays,
- smoke certification writes CSV/JSON results,
- Docker smoke passes.

## 11. Common Mistakes to Avoid

Avoid:

- optimizing only for accuracy,
- reporting empirical test risk without certification,
- using certification labels in proposal selection,
- tuning gamma or score on test labels,
- silently treating empty accepted set as success,
- comparing scores on different splits,
- changing config schemas silently,
- refactoring unrelated files,
- reformatting unrelated files,
- adding heavy dependencies without approval,
- making manuscript claims from smoke runs.

## 12. Minimal-change Rule

When editing:

1. make the smallest change that solves the task,
2. preserve existing APIs/config schema unless asked,
3. add/update tests for changed behavior,
4. run requested verification commands,
5. report actual results,
6. if a command fails, report exact error and likely cause.

## 13. Reporting Format

After coding, report:

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
