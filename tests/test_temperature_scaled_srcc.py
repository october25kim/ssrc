from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_temperature_scaled_srcc import (
    ROW_COLUMNS,
    argmax_preserved,
    certification_rows_for_scaled_run,
    comparison_rows,
    fit_temperature_grid,
    nll_from_logits,
    scale_arrays,
)
from srcc.scores import prediction


def _arrays():
    return {
        "logits_prop": np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]]),
        "labels_prop": np.array([0, 1, 0, 1]),
        "logits_cert": np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        "labels_cert": np.array([0, 1, 1, 0]),
        "logits_test": np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        "labels_test": np.array([0, 1, 1, 0]),
    }


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "srcc_cifar10_sym50_seed0"
    run_dir.mkdir()
    for name, value in _arrays().items():
        np.save(run_dir / f"{name}.npy", value)
    (run_dir / "metadata.json").write_text(
        '{"dataset":"toy","noise_type":"symmetric","noise_rate":0.5,"seed":0}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "alpha": 0.1,
                "certified": True,
                "certified_coverage_at_alpha": 0.2,
                "cert_coverage_lcb": 0.2,
                "cert_risk_ucb": 0.05,
                "score_name": "msp",
            }
        ]
    ).to_csv(run_dir / "certification_results.csv", index=False)
    return run_dir


def test_temperature_scaling_preserves_argmax_predictions():
    arrays = _arrays()
    scaled = scale_arrays(arrays, temperature=2.5)

    assert argmax_preserved(arrays, scaled)
    assert np.array_equal(prediction(arrays["logits_prop"]), prediction(scaled["logits_prop"]))


def test_grid_temperature_fit_is_no_worse_than_before_on_proposal():
    arrays = _arrays()
    fit = fit_temperature_grid(arrays["logits_prop"], arrays["labels_prop"], grid=np.array([0.5, 1.0, 2.0]))

    assert fit["temperature"] > 0.0
    assert fit["prop_nll_after"] <= fit["prop_nll_before"] + 1e-12
    assert fit["prop_nll_before"] == nll_from_logits(arrays["logits_prop"], arrays["labels_prop"])
    assert fit["temperature_fit_split"] == "prop"
    assert fit["temperature_fit_objective"] == "nll"


def test_certification_rows_schema_and_certified_coverage_convention(tmp_path):
    run_dir = _write_run(tmp_path)
    rows, fit = certification_rows_for_scaled_run(
        run_dir,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
    )

    for col in ROW_COLUMNS:
        assert col in rows.columns
    assert set(rows["method_name"]) == {"temperature_scaled_srcc"}
    assert set(rows["temperature_fit_split"]) == {"prop"}
    assert set(rows["temperature_fit_objective"]) == {"nll"}
    assert fit["prop_nll_after"] <= fit["prop_nll_before"] + 1e-12
    for _, row in rows.iterrows():
        expected = row["cert_coverage_lcb"] if bool(row["certified"]) else 0.0
        assert row["certified_coverage_at_alpha"] == expected
        if row["cert_n"] == 0:
            assert bool(row["certified"]) is False


def test_temperature_fit_uses_proposal_labels_only(tmp_path):
    run_dir = _write_run(tmp_path)
    rows, fit = certification_rows_for_scaled_run(
        run_dir,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
    )
    before = fit["temperature"]

    arrays = _arrays()
    arrays["labels_cert"] = 1 - arrays["labels_cert"]
    arrays["labels_test"] = 1 - arrays["labels_test"]
    for name, value in arrays.items():
        np.save(run_dir / f"{name}.npy", value)
    _, changed_fit = certification_rows_for_scaled_run(
        run_dir,
        alpha_values=[0.1],
        gammas=[1.0],
        scores=["msp"],
        num_thresholds=4,
    )

    assert changed_fit["temperature"] == before
    assert changed_fit["prop_nll_after"] == fit["prop_nll_after"]


def test_comparison_selects_by_certified_coverage_not_test_metrics(tmp_path):
    run_dir = _write_run(tmp_path)
    temp_rows = pd.DataFrame(
        [
            {
                "alpha": 0.1,
                "certified": False,
                "certified_coverage_at_alpha": 0.0,
                "cert_coverage_lcb": 0.9,
                "cert_risk_ucb": 0.2,
                "test_coverage": 1.0,
                "test_risk": 0.0,
                "score_name": "energy",
            },
            {
                "alpha": 0.1,
                "certified": True,
                "certified_coverage_at_alpha": 0.3,
                "cert_coverage_lcb": 0.3,
                "cert_risk_ucb": 0.05,
                "test_coverage": 0.1,
                "test_risk": 0.9,
                "score_name": "msp",
            },
        ]
    )

    rows = comparison_rows(
        run_dir,
        temp_rows,
        {"temperature": 1.0, "prop_nll_before": 1.0, "prop_nll_after": 1.0},
        alpha_values=[0.1],
    )

    assert rows[0]["temp_scaled_certified"] is True
    assert rows[0]["temp_scaled_best_score"] == "msp"
    assert rows[0]["temp_scaled_best_certified_coverage_at_alpha"] == 0.3
