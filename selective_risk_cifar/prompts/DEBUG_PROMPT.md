# Debug Prompt — SRCC

Read `CLAUDE.md` and `AGENTS.md` before diagnosing.

A command failed in the SRCC CIFAR harness. Diagnose before editing.

## Required diagnosis format

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

## Must inspect as relevant

```text
srcc/train.py
srcc/data.py
srcc/certify.py
srcc/certify_run.py
srcc/scores.py
scripts/docker_run.sh
scripts/docker_train.sh
scripts/docker_certify.sh
Dockerfile
requirements.txt
```

## Red flags to check

- certification labels used during proposal,
- test labels used for threshold/gamma/score selection,
- `cert_coverage_lcb` missing,
- empty accepted set reported as success,
- Docker pip overwrote CUDA torch,
- CUDA unavailable because `--gpus all` was not passed,
- data/runs accidentally tracked by git.

After the minimal fix, run the smallest relevant verification command and report the exact result.
