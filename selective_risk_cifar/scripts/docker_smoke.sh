#!/usr/bin/env bash
set -euo pipefail
bash scripts/docker_run.sh "python tests/test_certify.py && python scripts/smoke_certification_with_fake_logits.py"
