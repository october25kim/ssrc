# SRCC Ubuntu + Docker + Agent Setup

This repository is prepared for **Selective Risk Control after Corrupted Training**.

## 1. Ubuntu host prerequisites

Install Git and Docker on the host. If you have an NVIDIA GPU, also install NVIDIA Container Toolkit.

Quick check:

```bash
git --version
docker --version
docker compose version || true
nvidia-smi || true
```

## 2. First repository setup

```bash
unzip selective_risk_cifar_dockerized.zip
cd selective_risk_cifar

git init
git add -A
git commit -m "initial srcc cifar harness"
# optional after creating remote:
# git remote add origin git@github.com:<USER>/<REPO>.git
# git push -u origin main
```

## 3. Agent instructions

Keep these files at repository root:

```text
CLAUDE.md
AGENTS.md
```

Claude Code should automatically read `CLAUDE.md`. Codex/Cursor-style agents should be instructed to read `AGENTS.md`. The prompts in `prompts/` explicitly require both files to be read before acting.

## 4. Docker checks

```bash
bash scripts/docker_build.sh
bash scripts/docker_test.sh
bash scripts/docker_smoke.sh
```

Optional CIFAR smoke:

```bash
bash scripts/docker_cifar_smoke_train.sh
```

## 5. Daily Git workflow

Start of day:

```bash
bash scripts/git_start_day.sh
```

End of day:

```bash
bash scripts/git_end_day.sh "daily srcc experiment update"
```

The `.gitignore` excludes data, runs, outputs, checkpoints, logs, wandb, and numpy logits.

## 6. First real run

```bash
DATASET=cifar10 NOISE_TYPE=symmetric NOISE_RATE=0.35 SEED=0 \
  bash scripts/docker_train.sh runs/cifar10_sym35_seed0

bash scripts/docker_certify.sh runs/cifar10_sym35_seed0
```

Then inspect:

```text
runs/cifar10_sym35_seed0/certification_results.csv
```

Primary columns:

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
```

## 7. Agent prompt usage

For Claude Code:

```text
Use prompts/CLAUDE_CODE_BOOTSTRAP_PROMPT.md for initial setup.
Use prompts/DAILY_RUN_PROMPT.md for daily experiment runs.
Use prompts/DEBUG_PROMPT.md for failed commands.
```

For Codex:

```text
Use prompts/CODEX_BOOTSTRAP_PROMPT.md for initial setup.
Use prompts/DAILY_RUN_PROMPT.md for daily experiment runs.
Use prompts/DEBUG_PROMPT.md for failed commands.
```
