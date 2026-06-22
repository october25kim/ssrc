# Claude Code Bootstrap Prompt — SRCC AAAI CIFAR Harness

You are Claude Code working inside a fresh Ubuntu workspace for the paper **Selective Risk Control after Corrupted Training**.

## Mandatory startup

1. Read `CLAUDE.md` first.
2. Read `AGENTS.md` second.
3. Before editing, summarize the task-relevant constraints in Korean:
   - accepted selective risk is primary,
   - trusted clean data is for proposal/certification, not retraining,
   - certification labels must be independent of the selected rule,
   - test labels are evaluation-only,
   - certified coverage is the main AAAI viability metric,
   - Docker and Git reproducibility are required.

## Goal

Create or complete a Docker-first CIFAR-level experiment repository for **Selective Risk Control after Corrupted Training**.

The repository must support:

```text
CIFAR corrupted-label training
risk-buffered proposal selection
independent Clopper-Pearson accepted-risk certification
certified coverage reporting
Docker test/smoke/training/certification commands
daily git pull/push workflow
```

## If a harness ZIP exists

If `selective_risk_cifar_harness.zip` or `selective_risk_cifar.zip` exists in the working directory:

1. Unzip it.
2. Use the unzipped repo as the base.
3. Ensure `CLAUDE.md` and `AGENTS.md` are at the repository root.
4. Add Docker/Git/preflight scripts if missing.
5. Do not rewrite working code unless tests fail.

## If no harness ZIP exists

Create the following structure:

```text
srcc/
  __init__.py
  train.py
  certify.py
  certify_run.py
  scores.py
  metrics.py
  data.py
  noise.py
  models.py
  utils.py
configs/
scripts/
tests/
CLAUDE.md
AGENTS.md
Dockerfile
docker-compose.yml
.dockerignore
.gitignore
Makefile
requirements.txt
README.md
```

Implement the minimal runnable version needed for:

```bash
python tests/test_certify.py
python scripts/smoke_certification_with_fake_logits.py
```

and CIFAR smoke training.

## Required Docker files

Create:

```text
Dockerfile
docker-compose.yml
scripts/docker_build.sh
scripts/docker_run.sh
scripts/docker_test.sh
scripts/docker_smoke.sh
scripts/docker_train.sh
scripts/docker_certify.sh
scripts/docker_cifar_smoke_train.sh
scripts/preflight_repo.sh
```

Docker rules:

- Use a PyTorch CUDA base image when possible.
- Do not let pip overwrite CUDA-enabled `torch` or `torchvision` in Docker.
- Mount repository to `/workspace`.
- Mount local `data/`, `runs/`, `outputs/`.
- Keep `PYTHONPATH=/workspace`.
- Use `--gpus all` only when NVIDIA runtime is available.

## Required Git workflow files

Create:

```text
scripts/git_start_day.sh
scripts/git_end_day.sh
.gitignore
```

Expected commands:

```bash
bash scripts/git_start_day.sh
bash scripts/git_end_day.sh "message"
```

`git_start_day.sh` should:

```text
git fetch --all --prune
git pull --ff-only if upstream exists
git status --short
```

`git_end_day.sh` should:

```text
run preflight
run smoke verification
stage source/config/script/doc changes
commit if changes exist
pull --rebase if upstream exists
push if upstream exists
print commit hash
```

Do not commit:

```text
data/
runs/
outputs/
logs/
checkpoints/
wandb/
*.pt
*.pth
*.ckpt
*.npy
*.npz
```

## Required experiment commands

Ensure these work:

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
bash scripts/docker_cifar_smoke_train.sh
```

Then provide commands for the first AAAI run:

```bash
DATASET=cifar10 NOISE_TYPE=symmetric NOISE_RATE=0.35 SEED=0 \
  bash scripts/docker_train.sh runs/cifar10_sym35_seed0

bash scripts/docker_certify.sh runs/cifar10_sym35_seed0
```

## Required metrics

Every certification CSV/JSON must include:

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
```

## Verification

Run the lightest valid checks first:

```bash
python tests/test_certify.py
python scripts/smoke_certification_with_fake_logits.py
```

Then Docker checks:

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
```

Report actual results only. If any command fails, report the exact error and the smallest proposed fix.

## Reporting format

Respond in Korean:

```text
수정 요약:
- ...

변경한 파일:
- ...

검증 명령:
- ...

검증 결과:
- ...

다음 실행 명령:
- ...

Git 상태:
- branch:
- commit:
- pushed: yes/no

남은 리스크:
- ...
```
