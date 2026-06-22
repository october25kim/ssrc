#!/usr/bin/env bash
set -euo pipefail
bash scripts/docker_run.sh "python -m pytest -q"
