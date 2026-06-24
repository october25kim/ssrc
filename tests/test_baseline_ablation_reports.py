from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.build_baseline_ablation_reports import (
    BASELINE_COLUMNS,
    full_coverage_row,
    ltt_bonferroni_row,
    naive_row,
    normalize_baseline_columns,
)
from srcc.scores import risk_scores


def _arrays():
    # Four-class-ish logits with labels arranged so proposal is clean for high maxlogit,
    # certification contains enough errors to exercise CP failure paths.
    return {
        "logits_prop": np.array([[5.0, 0.0], [4.0, 0.0], [0.1, 2.0], [0.2, 1.5]]),
        "labels_prop": np.array([0, 0, 0, 0]),
        "logits_cert": np.array([[5.0, 0.0], [4.0, 0.0], [0.1, 2.0], [0.2, 1.5]]),
        "labels_cert": np.array([0, 1, 0, 1]),
        "logits_test": np.array([[5.0, 0.0], [0.1, 2.0], [0.2, 1.5], [3.0, 0.0]]),
        "labels_test": np.array([0, 0, 0, 1]),
    }


def _meta():
    return {
        "dataset": "toy",
        "regime": "sym50",
        "noise_type": "symmetric",
        "noise_rate": 0.5,
        "seed": 0,
        "run_dir": "runs/toy",
    }


def _deltas():
    return {
        "delta_total": 0.05,
        "delta_risk": 0.05,
        "delta_coverage": 0.05,
        "delta_allocation": "risk_only_legacy",
    }


def test_full_coverage_accepts_all_cert_examples_and_can_fail():
    row = full_coverage_row(_meta(), _arrays(), alpha=0.1, deltas=_deltas())
    assert row["baseline_name"] == "full_coverage"
    assert row["cert_n"] == len(_arrays()["labels_cert"])
    assert row["cert_coverage_lcb"] > 0.45
    assert row["cert_risk_ucb"] > 0.1
    assert row["certified"] is False
    assert row["certified_coverage_at_alpha"] == 0.0


def test_naive_empirical_threshold_uses_proposal_not_cert_or_test_for_selection():
    arrays = _arrays()
    gamma_one = pd.Series({"score_name": "maxlogit", "threshold": 4.0})
    row = naive_row(_meta(), arrays, alpha=0.1, deltas=_deltas(), gamma_one_row=gamma_one, num_thresholds=4)
    assert row["baseline_name"] == "naive_empirical_threshold"
    assert row["prop_risk"] <= 0.1
    prop_mask = risk_scores(arrays["logits_prop"], [row["score_name"]])[row["score_name"]] >= row["threshold"]
    cert_mask = risk_scores(arrays["logits_cert"], [row["score_name"]])[row["score_name"]] >= row["threshold"]
    assert row["prop_n"] == int(prop_mask.sum())
    assert row["cert_n"] == int(cert_mask.sum())
    assert "test" not in row.get("baseline_equivalence", "")


def test_ltt_bonferroni_records_candidate_count_and_divided_deltas():
    row = ltt_bonferroni_row(_meta(), _arrays(), alpha=0.2, deltas=_deltas(), num_thresholds=3)
    assert row["baseline_name"] == "ltt_bonferroni"
    assert row["certificate_scope"] == "simultaneous_rows"
    assert row["n_certified_candidates"] == 15
    assert row["delta_risk_per_row"] == pytest.approx(0.05 / 15)
    assert row["delta_coverage_per_row"] == pytest.approx(0.05 / 15)


def test_maxlogit_score_is_raw_max_logit_and_high_direction():
    logits = np.array([[1.0, 3.0, -2.0], [4.0, 0.0, 2.0]])
    scores = risk_scores(logits, ["maxlogit"])
    assert np.allclose(scores["maxlogit"], np.array([3.0, 4.0]))


def test_report_schema_and_certified_coverage_convention():
    df = pd.DataFrame(
        [
            {
                "certified": True,
                "cert_coverage_lcb": 0.4,
                "certified_coverage_at_alpha": 0.4,
            },
            {
                "certified": False,
                "cert_coverage_lcb": 0.8,
                "certified_coverage_at_alpha": 0.8,
            },
        ]
    )
    out = normalize_baseline_columns(df)
    for col in BASELINE_COLUMNS:
        assert col in out.columns
    assert out.loc[0, "certified_coverage_at_alpha"] == 0.4
    assert out.loc[1, "certified_coverage_at_alpha"] == 0.0
