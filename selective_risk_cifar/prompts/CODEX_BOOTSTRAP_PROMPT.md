# Codex Bootstrap Prompt — SRCC AAAI CIFAR Harness

You are Codex working in a fresh Ubuntu repository for **Selective Risk Control after Corrupted Training**.

Read `AGENTS.md` first and `CLAUDE.md` if present. Follow those instructions as project policy.

## Mission

Build a Docker-first CIFAR-level harness for the AAAI 2027 paper **Selective Risk Control after Corrupted Training**.

The scientific object is **accepted prediction risk after corrupted training**:

```text
R_sel(h) = P(y_hat(X) != Y | A_h(X)=1)
```

The repository must maximize **certified coverage** under:

```text
Clopper-Pearson risk UCB <= alpha
```

Do not drift into RC-OWPL, federated learning, open-world pseudo-labeling, generic calibration, or accuracy-only evaluation.

## Required behavior

Implement or verify:

```text
corrupted CIFAR training
trusted clean proposal split
trusted clean certification split
test split evaluation
MSP / entropy / margin / energy scores
risk-buffered proposal: prop_risk <= gamma * alpha
independent CP certification
certified coverage LCB
CSV/JSON results
Docker scripts
Git daily pull/push scripts
```

## Required files

Ensure these exist and work:

```text
CLAUDE.md
AGENTS.md
Dockerfile
docker-compose.yml
.dockerignore
.gitignore
Makefile
scripts/docker_build.sh
scripts/docker_run.sh
scripts/docker_test.sh
scripts/docker_smoke.sh
scripts/docker_train.sh
scripts/docker_certify.sh
scripts/docker_cifar_smoke_train.sh
scripts/preflight_repo.sh
scripts/git_start_day.sh
scripts/git_end_day.sh
```

## Docker constraints

Use a PyTorch CUDA base image if available. Do not pip-install over CUDA-enabled torch inside Docker. Install non-torch dependencies only from `requirements.txt`.

Mount:

```text
repo -> /workspace
data -> /data
outputs -> /outputs
runs -> /workspace/runs
```

Set:

```text
PYTHONPATH=/workspace
WANDB_DISABLED=true
```

## Git constraints

Start day:

```bash
bash scripts/git_start_day.sh
```

End day:

```bash
bash scripts/git_end_day.sh "message"
```

Never commit data, outputs, runs, logs, checkpoints, wandb, torch checkpoints, numpy logits, or downloaded CIFAR files.

## Required verification

Run:

```bash
python tests/test_certify.py
python scripts/smoke_certification_with_fake_logits.py
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
```

If Docker is unavailable, report that fact and run local fallback only.

## Required final response

Use Korean and include:

```text
수정 요약
변경 파일
검증 명령
검증 결과
다음 실행 명령
Git 상태
남은 리스크
```
