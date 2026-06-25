from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_confidence_deformation import (
    ROW_COLUMNS,
    analyze_split,
    auroc_reason,
    correctness_auroc,
    ranking_score,
    risk_coverage_curve_from_ranking,
)


def test_correctness_auroc_synthetic_ranking():
    correctness = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.8, 0.9])

    assert correctness_auroc(correctness, scores) == pytest.approx(1.0)
    assert correctness_auroc(correctness, -scores) == pytest.approx(0.0)


def test_risk_coverage_curve_values_and_monotonic_coverage():
    correctness = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    curve = risk_coverage_curve_from_ranking(correctness, scores)

    assert np.all(np.diff(curve["coverage"].to_numpy()) > 0)
    assert curve["risk"].to_list() == pytest.approx([0.0, 0.5, 1.0 / 3.0, 0.5])


def test_entropy_and_energy_are_inverted_for_ranking():
    raw = np.array([0.1, 0.9])

    assert np.allclose(ranking_score("entropy", raw), np.array([-0.1, -0.9]))
    assert np.allclose(ranking_score("energy", raw), np.array([-0.1, -0.9]))
    assert np.allclose(ranking_score("msp", raw), raw)


def test_all_correct_or_all_wrong_auroc_is_nan_with_reason():
    all_correct = np.ones(4, dtype=int)
    all_wrong = np.zeros(4, dtype=int)

    assert np.isnan(correctness_auroc(all_correct, np.arange(4)))
    assert np.isnan(correctness_auroc(all_wrong, np.arange(4)))
    assert auroc_reason(all_correct) == "auroc_undefined_all_correct"
    assert auroc_reason(all_wrong) == "auroc_undefined_all_wrong"


def test_analyze_split_output_schema_and_diagnostic_only_fields():
    logits = np.array(
        [
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    labels = np.array([0, 1, 1, 0])
    meta = {
        "dataset": "toy",
        "regime": "sym50",
        "noise_type": "symmetric",
        "noise_rate": 0.5,
        "seed": 0,
    }

    rows, curves = analyze_split(meta, "test", logits, labels, ["msp", "entropy"])
    df = pd.DataFrame(rows)

    for col in ROW_COLUMNS:
        assert col in df.columns
    assert set(df["score_name"]) == {"msp", "entropy"}
    assert set(df["split"]) == {"test"}
    assert "certified" not in df.columns
    assert "threshold" not in df.columns
    assert len(curves) == 2
