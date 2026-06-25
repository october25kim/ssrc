from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_calibration_size_sensitivity import (
    actual_size,
    aggregate_sensitivity,
    deterministic_indices,
    rows_for_subset,
)


def _arrays(n_prop: int = 12, n_cert: int = 10, n_test: int = 14):
    def logits(n: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, n)
        return np.column_stack([2.0 - x, x])

    return {
        "logits_prop": logits(n_prop),
        "labels_prop": np.zeros(n_prop, dtype=np.int64),
        "logits_cert": logits(n_cert),
        "labels_cert": np.zeros(n_cert, dtype=np.int64),
        "logits_test": logits(n_test),
        "labels_test": np.zeros(n_test, dtype=np.int64),
    }


def _meta():
    return {
        "dataset": "toy",
        "regime": "clean",
        "noise_type": "clean",
        "noise_rate": 0.0,
        "seed": 0,
    }


def test_subsampling_uses_proposal_and_certification_splits_separately():
    prop_idx = deterministic_indices(50, 10, seed=7, split="prop")
    cert_idx = deterministic_indices(50, 10, seed=7, split="cert")

    assert len(prop_idx) == 10
    assert len(cert_idx) == 10
    assert not np.array_equal(prop_idx, cert_idx)


def test_same_subsample_seed_is_deterministic():
    first = deterministic_indices(50, 10, seed=3, split="prop")
    second = deterministic_indices(50, 10, seed=3, split="prop")

    assert np.array_equal(first, second)


def test_budget_larger_than_available_split_uses_full_split_safely():
    assert actual_size(100, available=12) == 12
    assert actual_size("full", available=12) == 12


def test_no_proposal_candidate_records_attempted_score_name(tmp_path):
    rows = rows_for_subset(
        run_dir=tmp_path / "run",
        arrays=_arrays(),
        meta=_meta(),
        budget=6,
        subsample_seed=0,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
        min_prop_accept=99,
    )

    assert len(rows) == 1
    assert rows[0]["reason"] == "no_proposal_candidate"
    assert rows[0]["score_name"] == "msp"
    assert rows[0]["threshold_direction"] == ">="


def test_rows_do_not_use_test_split_for_selection_and_keep_convention(tmp_path):
    arrays = _arrays()
    rows = rows_for_subset(
        run_dir=tmp_path / "run",
        arrays=arrays,
        meta=_meta(),
        budget=6,
        subsample_seed=0,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
        min_prop_accept=1,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["actual_prop_size"] == 6
    assert row["actual_cert_size"] == 6
    assert "uses_test_for_selection" not in row
    expected = row["cert_coverage_lcb"] if row["certified"] else 0.0
    assert row["certified_coverage_at_alpha"] == expected
    assert 0.0 <= row["test_coverage"] <= 1.0

    changed_test = dict(arrays)
    changed_test["labels_test"] = 1 - arrays["labels_test"]
    changed_rows = rows_for_subset(
        run_dir=tmp_path / "run",
        arrays=changed_test,
        meta=_meta(),
        budget=6,
        subsample_seed=0,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
        min_prop_accept=1,
    )
    changed = changed_rows[0]
    for key in [
        "score_name",
        "threshold",
        "threshold_direction",
        "prop_n",
        "prop_k",
        "cert_n",
        "cert_k",
        "cert_risk_ucb",
        "cert_coverage_lcb",
        "certified",
        "certified_coverage_at_alpha",
        "reason",
    ]:
        assert changed[key] == row[key]
    assert changed["test_risk"] != row["test_risk"]


def test_aggregate_sensitivity_columns_and_means(tmp_path):
    rows = rows_for_subset(
        run_dir=tmp_path / "run",
        arrays=_arrays(),
        meta=_meta(),
        budget="full",
        subsample_seed=0,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
        min_prop_accept=1,
    )
    aggregate = aggregate_sensitivity(pd.DataFrame(rows))

    assert len(aggregate) == 1
    assert aggregate.loc[0, "row_count"] == 1
    assert 0.0 <= aggregate.loc[0, "certification_rate"] <= 1.0
    assert 0.0 <= aggregate.loc[0, "mean_test_coverage"] <= 1.0
