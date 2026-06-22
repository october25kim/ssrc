from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _risk_order_scores(scores: np.ndarray, direction: str = "<=") -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if direction == "<=":
        return scores
    if direction == ">=":
        return -scores
    raise ValueError(f"Unknown threshold direction: {direction}")


def risk_coverage_curve(scores: np.ndarray, errors: np.ndarray, direction: str = "<=") -> Dict[str, np.ndarray]:
    """Risk-coverage curve ordered from safest to riskiest examples."""
    order_scores = _risk_order_scores(scores, direction)
    errors = np.asarray(errors).astype(float)
    order = np.argsort(order_scores, kind="mergesort")
    e = errors[order]
    n = len(e)
    cum_err = np.cumsum(e)
    counts = np.arange(1, n + 1)
    risk = cum_err / counts
    coverage = counts / n
    return {"coverage": coverage, "risk": risk, "order": order}


def aurc(scores: np.ndarray, errors: np.ndarray, direction: str = "<=") -> float:
    curve = risk_coverage_curve(scores, errors, direction)
    return float(np.mean(curve["risk"]))


def correctness_auroc(scores: np.ndarray, errors: np.ndarray, direction: str = "<=") -> float:
    """AUROC for detecting correctness."""
    errors = np.asarray(errors).astype(int)
    correct = 1 - errors
    if len(np.unique(correct)) < 2:
        return float("nan")
    safety_score = -_risk_order_scores(scores, direction)
    return float(roc_auc_score(correct, safety_score))


def coverage_at_risk(scores: np.ndarray, errors: np.ndarray, alpha: float, direction: str = "<=") -> float:
    curve = risk_coverage_curve(scores, errors, direction)
    ok = np.where(curve["risk"] <= alpha)[0]
    if len(ok) == 0:
        return 0.0
    return float(curve["coverage"][ok[-1]])
