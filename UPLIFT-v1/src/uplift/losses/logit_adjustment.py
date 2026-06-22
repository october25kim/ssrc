"""Logit-adjusted learning hook for prior-dependent validation.

The paper convention is:
    s_c^{LA}(x) = s_c(x) - tau * log(prior_c)
"""

from __future__ import annotations

import math
from typing import Iterable

from uplift.priors.counts import normalize_prior


def log_prior_offsets(prior: Iterable[float], tau: float = 1.0, eps: float = 1e-12) -> list[float]:
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    values = normalize_prior(prior, eps=eps)
    return [-tau * math.log(p) for p in values]


def adjust_logits(logits, prior: Iterable[float], tau: float = 1.0):
    offsets = log_prior_offsets(prior, tau=tau)
    if _is_matrix(logits):
        return [[value + offsets[j] for j, value in enumerate(row)] for row in logits]
    return [value + offsets[j] for j, value in enumerate(logits)]


def margin_distortion(clean_prior: Iterable[float], observed_prior: Iterable[float], a: int, b: int, tau: float = 1.0) -> float:
    clean = normalize_prior(clean_prior, eps=1e-12)
    observed = normalize_prior(observed_prior, eps=1e-12)
    if not (0 <= a < len(clean) and 0 <= b < len(clean)):
        raise ValueError("class indices are out of range")
    ratio = (observed[a] / observed[b]) / (clean[a] / clean[b])
    return -tau * math.log(ratio)


def _is_matrix(values) -> bool:
    return bool(values) and isinstance(values[0], (list, tuple))
