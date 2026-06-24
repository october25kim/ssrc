from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.special import logsumexp, softmax


EPS = 1e-12


def probabilities(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=1)


def prediction(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1).astype(np.int64)


def risk_scores(logits: np.ndarray, score_names: Iterable[str]) -> Dict[str, np.ndarray]:
    """Compute proposal scores.

    Direction is handled by certification:
    msp/margin accept high scores, entropy/energy accept low scores.
    """
    logits = np.asarray(logits, dtype=np.float64)
    probs = probabilities(logits)
    score_names = list(score_names)
    out: Dict[str, np.ndarray] = {}

    if any(s in score_names for s in ["msp", "entropy", "margin"]):
        conf = probs.max(axis=1)
        if "msp" in score_names:
            out["msp"] = conf
        if "entropy" in score_names:
            ent = -np.sum(probs * np.log(np.clip(probs, EPS, 1.0)), axis=1)
            out["entropy"] = ent / np.log(probs.shape[1])
        if "margin" in score_names:
            part = np.partition(probs, kth=-2, axis=1)
            top2 = part[:, -2:]
            # np.partition does not sort the top2 columns.
            top1 = np.max(top2, axis=1)
            second = np.min(top2, axis=1)
            margin = top1 - second
            out["margin"] = margin

    if "maxlogit" in score_names:
        out["maxlogit"] = logits.max(axis=1)

    if "energy" in score_names:
        # Energy score: lower is generally more ID/confident when defined as -logsumexp(logits).
        out["energy"] = -logsumexp(logits, axis=1)

    unknown = set(score_names) - set(out.keys())
    if unknown:
        raise ValueError(f"Unknown scores requested: {sorted(unknown)}")
    return out


def correctness_errors(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (prediction(logits) != np.asarray(labels, dtype=np.int64)).astype(np.int64)
