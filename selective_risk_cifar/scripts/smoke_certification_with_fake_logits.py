from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "smoke_fake"
RUN.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
for split, n in [("prop", 1000), ("cert", 1000), ("test", 2000)]:
    y = rng.integers(0, 10, size=n)
    logits = rng.normal(size=(n, 10)) * 0.8
    for i in range(n):
        logits[i, y[i]] += 3.0 if rng.random() > 0.1 else 0.0
    np.save(RUN / f"logits_{split}.npy", logits)
    np.save(RUN / f"labels_{split}.npy", y)
with (RUN / "metadata.json").open("w") as f:
    json.dump({"smoke_fake_logits": True}, f)

cmd = [
    sys.executable,
    "-m",
    "srcc.certify_run",
    "--run-dir",
    str(RUN),
    "--alpha",
    "0.05",
    "0.10",
    "--delta",
    "0.05",
    "--gammas",
    "0.5",
    "0.7",
    "1.0",
    "--scores",
    "msp",
    "entropy",
    "margin",
    "energy",
    "--num-thresholds",
    "50",
]
subprocess.check_call(cmd, cwd=str(ROOT))
print(f"Smoke outputs in {RUN}")
