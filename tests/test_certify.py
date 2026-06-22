from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running as `python tests/test_certify.py` without installing package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from srcc.certify import cp_lower, cp_upper, run_certification_for_alpha, zero_error_n_min


def test_cp_zero_error_formula():
    alpha = 0.05
    delta = 0.05
    n_min = zero_error_n_min(alpha, delta)
    assert n_min == math.ceil(math.log(delta) / math.log(1 - alpha))
    assert cp_upper(0, n_min, delta) <= alpha + 1e-12
    assert cp_upper(0, n_min - 1, delta) > alpha


def test_cp_bounds_basic():
    assert cp_upper(0, 0, 0.05) == 1.0
    assert cp_lower(0, 100, 0.05) == 0.0
    assert 0.0 < cp_upper(2, 100, 0.05) < 0.1
    assert 0.0 < cp_lower(98, 100, 0.05) < 1.0


def test_certification_pipeline_synthetic():
    rng = np.random.default_rng(0)
    n_prop, n_cert, n_test = 1000, 1000, 2000
    # Low score is safer. The safest 50% have 1% error, rest have 20% error.
    prop_score = rng.random(n_prop)
    cert_score = rng.random(n_cert)
    test_score = rng.random(n_test)

    def make_errors(scores):
        p = np.where(scores <= 0.5, 0.01, 0.20)
        return (rng.random(len(scores)) < p).astype(int)

    prop_errors = make_errors(prop_score)
    cert_errors = make_errors(cert_score)
    test_errors = make_errors(test_score)

    results = run_certification_for_alpha(
        prop_scores={"mock": prop_score},
        prop_errors=prop_errors,
        cert_scores={"mock": cert_score},
        cert_errors=cert_errors,
        test_scores={"mock": test_score},
        test_errors=test_errors,
        alpha=0.05,
        delta=0.05,
        gammas=[0.5],
        num_thresholds=100,
        bonferroni_over_gammas=False,
    )
    assert len(results) == 1
    res = results[0]
    assert res.score_name == "mock"
    assert res.prop_risk <= 0.025 + 1e-12
    # May occasionally fail certification due to finite randomness, but the fields must be well formed.
    assert 0.0 <= res.cert_risk_ucb <= 1.0
    assert 0.0 <= res.cert_coverage_lcb <= 1.0


if __name__ == "__main__":
    test_cp_zero_error_formula()
    test_cp_bounds_basic()
    test_certification_pipeline_synthetic()
    print("ok")
