from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_comparison_method_reports import (
    CANONICAL_KEY,
    REQUIRED_ROW_COLUMNS,
    aggregate_rows,
    diagnostic_rows,
    normalize_comparison_rows,
    validate_rows,
)


def _baseline_df() -> pd.DataFrame:
    base = {
        "dataset": "toy",
        "regime": "sym50",
        "noise_type": "symmetric",
        "noise_rate": 0.5,
        "seed": 0,
        "alpha": 0.1,
        "gamma": 1.0,
        "score_name": "msp",
        "threshold": 0.9,
        "threshold_direction": ">=",
        "prop_n": 10,
        "prop_k": 1,
        "prop_coverage": 0.5,
        "prop_risk": 0.1,
        "cert_n": 10,
        "cert_k": 1,
        "cert_risk_ucb": 0.25,
        "cert_coverage_lcb": 0.4,
        "certified": False,
        "certified_coverage_at_alpha": 0.0,
        "test_coverage": 0.6,
        "test_risk": 0.05,
        "delta_total": 0.05,
        "delta_risk": 0.05,
        "delta_coverage": 0.05,
        "delta_allocation": "risk_only_legacy",
        "certificate_scope": "single_selector",
        "n_certified_candidates": 1,
        "delta_risk_per_row": 0.05,
        "delta_coverage_per_row": 0.05,
        "reason": "high_cert_error_ucb",
    }
    rows = []
    for method in ["srcc_main", "full_coverage", "no_buffer_proposal", "naive_empirical_threshold", "ltt_bonferroni"]:
        row = dict(base)
        row["baseline_name"] = method
        row["baseline_equivalence"] = ""
        if method == "full_coverage":
            row["threshold_direction"] = "all"
            row["test_coverage"] = 1.0
        if method == "no_buffer_proposal":
            row["baseline_equivalence"] = "gamma_1_srcc_row"
        if method == "naive_empirical_threshold":
            row["baseline_equivalence"] = "same_proposal_as_gamma_1_no_buffer"
        if method == "ltt_bonferroni":
            row["certificate_scope"] = "simultaneous_rows"
            row["n_certified_candidates"] = 20
            row["delta_risk_per_row"] = 0.0025
            row["delta_coverage_per_row"] = 0.0025
        rows.append(row)
    clean = dict(base)
    clean.update(
        {
            "regime": "clean",
            "noise_rate": 0.0,
            "baseline_name": "clean_trained_upper_bound",
            "baseline_equivalence": "",
            "certified": True,
            "cert_risk_ucb": 0.04,
            "cert_coverage_lcb": 0.8,
            "certified_coverage_at_alpha": 0.8,
            "reason": "certified",
        }
    )
    rows.append(clean)
    return pd.DataFrame(rows)


def test_non_duplication_and_canonical_key_uniqueness():
    rows = normalize_comparison_rows(_baseline_df())
    validate_rows(rows)
    assert not rows.duplicated(subset=CANONICAL_KEY).any()
    naive = rows[rows["baseline_name"].eq("naive_empirical_threshold")].iloc[0]
    assert naive["canonical_baseline_name"] == "no_buffer_proposal"
    assert naive["baseline_equivalence"] == "same_as_gamma_1_no_buffer"
    diagnostic = rows[rows["baseline_name"].eq("uncertified_empirical_threshold")].iloc[0]
    assert diagnostic["canonical_baseline_name"] == "naive_empirical_threshold"


def test_no_test_leakage_and_ltt_bonferroni_metadata():
    rows = normalize_comparison_rows(_baseline_df())
    assert rows["uses_test_for_selection"].eq(False).all()
    ltt = rows[rows["baseline_name"].eq("ltt_bonferroni")].iloc[0]
    assert bool(ltt["uses_certification_for_selection"]) is True
    assert ltt["selection_correction"] == "bonferroni"
    assert ltt["delta_risk_per_row"] == 0.0025
    assert ltt["delta_coverage_per_row"] == 0.0025


def test_full_coverage_and_diagnostics_flags():
    rows = normalize_comparison_rows(_baseline_df())
    full = rows[rows["baseline_name"].eq("full_coverage")].iloc[0]
    assert full["threshold_direction"] == "all"
    assert full["test_coverage"] == 1.0
    assert bool(full["certified"]) is False
    failed = rows[rows["baseline_name"].eq("srcc_main")].iloc[0]
    assert bool(failed["test_looks_good_but_not_certified"]) is True
    assert bool(failed["coverage_collapse"]) is True
    diag = diagnostic_rows(rows)
    assert diag["test_risk_le_alpha_but_cert_risk_ucb_gt_alpha"].any()


def test_schema_and_certified_coverage_convention():
    rows = normalize_comparison_rows(_baseline_df())
    for col in REQUIRED_ROW_COLUMNS:
        assert col in rows.columns
    certified = rows[rows["certified"]]
    uncertified = rows[~rows["certified"]]
    assert (certified["certified_coverage_at_alpha"] == certified["cert_coverage_lcb"]).all()
    assert (uncertified["certified_coverage_at_alpha"] == 0.0).all()


def test_aggregate_columns():
    rows = normalize_comparison_rows(_baseline_df())
    aggregate = aggregate_rows(rows)
    assert "n_seeds_total" in aggregate.columns
    assert "n_seeds_certified" in aggregate.columns
    assert "mean_test_coverage_on_best_certified" in aggregate.columns
