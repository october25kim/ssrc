"""Counts-only prior recovery for the restricted UPLIFT-U model.

The main v1 model is:
    observed = (1 - gamma) * clean + gamma * routing

This module deliberately works only with class-prior vectors. It does not use
images, features, logits, sample losses, or confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from uplift.priors.counts import l1_distance, normalize_prior


@dataclass(frozen=True)
class RecoveryResult:
    recovered_priors: list[list[float]]
    routing_prior: list[float]
    gammas: list[float]


def recover_known_priors(
    observed_priors: Iterable[Iterable[float]],
    gammas: Sequence[float],
    routing_prior: Iterable[float],
) -> RecoveryResult:
    rows = [normalize_prior(row) for row in observed_priors]
    if not rows:
        raise ValueError("observed_priors must not be empty")
    if len(rows) != len(gammas):
        raise ValueError("one gamma value is required per client")
    routing = normalize_prior(routing_prior)
    recovered: list[list[float]] = []
    for observed, gamma in zip(rows, gammas):
        if len(observed) != len(routing):
            raise ValueError("observed and routing priors must have the same length")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma values must be in [0, 1)")
        clean = [(p - gamma * r) / (1.0 - gamma) for p, r in zip(observed, routing)]
        recovered.append(normalize_prior(clean))
    return RecoveryResult(recovered_priors=recovered, routing_prior=routing, gammas=list(gammas))


def uniform_r_recovery(
    observed_priors: Iterable[Iterable[float]],
    gammas: Sequence[float],
) -> RecoveryResult:
    rows = [normalize_prior(row) for row in observed_priors]
    if not rows:
        raise ValueError("observed_priors must not be empty")
    routing = [1.0 / len(rows[0]) for _ in rows[0]]
    return recover_known_priors(rows, gammas, routing)


def mean_prior_error(estimates: Iterable[Iterable[float]], clean_priors: Iterable[Iterable[float]]) -> float:
    estimate_rows = [normalize_prior(row) for row in estimates]
    clean_rows = [normalize_prior(row) for row in clean_priors]
    if len(estimate_rows) != len(clean_rows):
        raise ValueError("estimate and clean prior counts must match")
    return sum(l1_distance(a, b) for a, b in zip(estimate_rows, clean_rows)) / len(clean_rows)
