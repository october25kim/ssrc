from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running as `python tests/test_certify.py` without installing package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from srcc.certify import (
    clopper_pearson_lcb,
    clopper_pearson_ucb,
    cp_lower,
    cp_upper,
    propose_candidate,
    run_certification_for_alpha,
    zero_error_min_n,
    zero_error_n_min,
)


def test_cp_zero_error_formula_and_aliases():
    alpha = 0.05
    delta = 0.05
    n_min = zero_error_min_n(alpha, delta)
    assert zero_error_n_min(alpha, delta) == n_min
    assert n_min == math.ceil(math.log(delta) / math.log(1 - alpha))
    assert clopper_pearson_ucb(0, n_min, delta) <= alpha + 1e-12
    assert cp_upper(0, n_min - 1, delta) > alpha
    assert clopper_pearson_ucb(0, 10, delta) == pytest.approx(1.0 - delta ** (1.0 / 10))


def test_zero_error_min_required_values():
    assert zero_error_min_n(0.05, 0.05) == 59
    assert zero_error_min_n(0.10, 0.05) == 29
    assert zero_error_min_n(0.05, 0.025) == 72
    assert zero_error_min_n(0.10, 0.025) == 36


def test_cp_bounds_boundaries_and_validation():
    assert clopper_pearson_ucb(0, 0, 0.05) == 1.0
    assert clopper_pearson_lcb(0, 0, 0.05) == 0.0
    assert clopper_pearson_ucb(10, 10, 0.05) == 1.0
    assert cp_lower(0, 100, 0.05) == 0.0
    assert 0.0 < clopper_pearson_ucb(2, 100, 0.05) < 0.1
    assert 0.0 < clopper_pearson_lcb(98, 100, 0.05) < 1.0
    with pytest.raises(ValueError):
        clopper_pearson_ucb(2, 1, 0.05)
    with pytest.raises(ValueError):
        clopper_pearson_lcb(0, 10, 1.0)


def test_zero_prop_accepts_cannot_win_proposal():
    scores = np.array([0.1, 0.2, 0.3])
    errors = np.array([0, 0, 0])
    cand = propose_candidate(
        {"mock": scores},
        errors,
        alpha=0.05,
        gamma=1.0,
        num_thresholds=3,
        min_prop_accept=4,
    )
    assert cand is None


def test_no_proposal_candidate_result_schema():
    scores = np.array([0.1, 0.2, 0.3])
    errors = np.array([1, 1, 1])
    results = run_certification_for_alpha(
        prop_scores={"mock": scores},
        prop_errors=errors,
        cert_scores={"mock": scores},
        cert_errors=errors,
        test_scores={"mock": scores},
        test_errors=errors,
        alpha=0.05,
        gammas=[0.5],
        num_thresholds=3,
        min_prop_accept=1,
    )
    res = results[0]
    assert res.reason == "no_proposal_candidate"
    assert res.certified is False
    assert math.isnan(res.prop_risk)
    assert res.certified_coverage_at_alpha == 0.0
    assert res.certificate_scope == "single_selector"


def test_zero_cert_accepts_are_uncertified():
    prop_scores = {"mock": np.array([0.1, 0.2, 0.3, 0.4])}
    prop_errors = np.array([0, 0, 0, 0])
    cert_scores = {"mock": np.array([0.8, 0.9])}
    cert_errors = np.array([0, 0])
    test_scores = {"mock": np.array([0.1, 0.9])}
    test_errors = np.array([0, 1])
    res = run_certification_for_alpha(
        prop_scores,
        prop_errors,
        cert_scores,
        cert_errors,
        test_scores,
        test_errors,
        alpha=0.05,
        gammas=[1.0],
        num_thresholds=4,
    )[0]
    assert res.cert_n == 0
    assert res.cert_risk_ucb == 1.0
    assert res.cert_coverage_lcb == 0.0
    assert res.certified is False
    assert res.certified_coverage_at_alpha == 0.0
    assert res.reason == "zero_cert_accepts"


def test_threshold_direction_for_msp_high_is_safer():
    prop_scores = {"msp": np.array([0.95, 0.9, 0.2, 0.1])}
    prop_errors = np.array([0, 0, 1, 1])
    cand = propose_candidate(prop_scores, prop_errors, alpha=0.05, gamma=1.0, num_thresholds=4)
    assert cand is not None
    assert cand.threshold_direction == ">="
    assert cand.prop_n == 2
    assert cand.prop_k == 0


def test_certification_pipeline_synthetic_joint_scope():
    rng = np.random.default_rng(0)
    n_prop, n_cert, n_test = 1000, 1000, 2000
    # Low mock score is safer. The safest 50% have 1% error, rest have 20% error.
    prop_score = rng.random(n_prop)
    cert_score = rng.random(n_cert)
    test_score = rng.random(n_test)

    def make_errors(scores):
        p = np.where(scores <= 0.5, 0.01, 0.20)
        return (rng.random(len(scores)) < p).astype(int)

    results = run_certification_for_alpha(
        prop_scores={"mock": prop_score},
        prop_errors=make_errors(prop_score),
        cert_scores={"mock": cert_score},
        cert_errors=make_errors(cert_score),
        test_scores={"mock": test_score},
        test_errors=make_errors(test_score),
        alpha=0.05,
        gammas=[0.5, 1.0],
        num_thresholds=100,
        bonferroni_over_gammas=True,
    )
    assert len(results) == 2
    res = results[0]
    assert res.score_name == "mock"
    assert res.prop_risk <= 0.025 + 1e-12
    assert 0.0 <= res.cert_risk_ucb <= 1.0
    assert 0.0 <= res.cert_coverage_lcb <= 1.0
    assert res.certified_coverage_at_alpha == (res.cert_coverage_lcb if res.certified else 0.0)
    assert res.delta_total == 0.05
    assert res.delta_risk == pytest.approx(0.025 / 2)
    assert res.delta_coverage == pytest.approx(0.025 / 2)
    assert res.delta_allocation == "joint_split"
    assert res.certificate_scope == "simultaneous_rows"
    assert res.n_certified_candidates == 2


if __name__ == "__main__":
    test_cp_zero_error_formula_and_aliases()
    test_zero_error_min_required_values()
    test_cp_bounds_boundaries_and_validation()
    test_zero_prop_accepts_cannot_win_proposal()
    test_no_proposal_candidate_result_schema()
    test_zero_cert_accepts_are_uncertified()
    test_threshold_direction_for_msp_high_is_safer()
    test_certification_pipeline_synthetic_joint_scope()
    print("ok")
