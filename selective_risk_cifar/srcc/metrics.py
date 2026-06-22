from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def risk_coverage_curve(scores: np.ndarray, errors: np.ndarray) -> Dict[str, np.ndarray]:
    """Risk-coverage curve for risk scores where lower is safer."""
    scores = np.asarray(scores)
    errors = np.asarray(errors).astype(float)
    order = np.argsort(scores, kind="mergesort")
    e = errors[order]
    n = len(e)
    cum_err = np.cumsum(e)
    counts = np.arange(1, n + 1)
    risk = cum_err / counts
    coverage = counts / n
    return {"coverage": coverage, "risk": risk, "order": order}


def aurc(scores: np.ndarray, errors: np.ndarray) -> float:
    curve = risk_coverage_curve(scores, errors)
    return float(np.mean(curve["risk"]))


def correctness_auroc(scores: np.ndarray, errors: np.ndarray) -> float:
    """AUROC for detecting correctness using negative risk score.

    Returns NaN if only one correctness class is present.
    """
    errors = np.asarray(errors).astype(int)
    correct = 1 - errors
    if len(np.unique(correct)) < 2:
        return float("nan")
    # Larger value should indicate correctness, so use -risk_score.
    return float(roc_auc_score(correct, -np.asarray(scores)))


def coverage_at_risk(scores: np.ndarray, errors: np.ndarray, alpha: float) -> float:
    curve = risk_coverage_curve(scores, errors)
    ok = np.where(curve["risk"] <= alpha)[0]
    if len(ok) == 0:
        return 0.0
    return float(curve["coverage"][ok[-1]])
