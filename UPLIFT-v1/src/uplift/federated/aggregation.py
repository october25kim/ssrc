"""Simple aggregation and counts-only prior estimators."""

from __future__ import annotations

from typing import Iterable, Sequence

from uplift.priors.counts import normalize_prior
from uplift.priors.recovery import RecoveryResult, recover_known_priors, uniform_r_recovery


def average_priors(priors: Iterable[Iterable[float]]) -> list[float]:
    rows = [normalize_prior(row) for row in priors]
    if not rows:
        raise ValueError("priors must contain at least one row")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all priors must have the same length")
    return normalize_prior(sum(row[i] for row in rows) / len(rows) for i in range(width))


def observed_local(observed_priors: Iterable[Iterable[float]]) -> list[list[float]]:
    return [normalize_prior(row) for row in observed_priors]


def global_observed(observed_priors: Iterable[Iterable[float]]) -> list[list[float]]:
    rows = [normalize_prior(row) for row in observed_priors]
    pooled = average_priors(rows)
    return [pooled[:] for _ in rows]


def oracle(clean_priors: Iterable[Iterable[float]]) -> list[list[float]]:
    return [normalize_prior(row) for row in clean_priors]


def uplift_recovery(
    observed_priors: Iterable[Iterable[float]],
    gammas: Sequence[float],
    routing_prior: Iterable[float],
) -> RecoveryResult:
    return recover_known_priors(observed_priors, gammas=gammas, routing_prior=routing_prior)


def uniform_r_baseline(observed_priors: Iterable[Iterable[float]], gammas: Sequence[float]) -> RecoveryResult:
    return uniform_r_recovery(observed_priors, gammas=gammas)


def shrink_to_global(
    observed_prior: Iterable[float],
    global_prior: Iterable[float],
    strength: float = 0.5,
) -> list[float]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    observed = normalize_prior(observed_prior)
    global_values = normalize_prior(global_prior)
    if len(observed) != len(global_values):
        raise ValueError("prior lengths must match")
    return normalize_prior((1.0 - strength) * p + strength * g for p, g in zip(observed, global_values))


def unmix_uniform(observed_prior: Iterable[float], strength: float) -> list[float]:
    if not 0.0 <= strength < 1.0:
        raise ValueError("strength must be in [0, 1)")
    observed = normalize_prior(observed_prior)
    uniform = 1.0 / len(observed)
    return normalize_prior((p - strength * uniform) / (1.0 - strength) for p in observed)
