"""Counts-only corruption mechanisms for observed class priors."""

from __future__ import annotations

import random
from typing import Iterable

from .counts import counts_to_prior, normalize_prior, validate_counts


def corrupt_prior(
    prior: Iterable[float],
    kind: str = "mix_uniform",
    strength: float = 0.25,
    seed: int | None = None,
) -> list[float]:
    clean = normalize_prior(prior)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    if kind == "none":
        return clean
    if kind == "mix_uniform":
        uniform = 1.0 / len(clean)
        return normalize_prior((1.0 - strength) * p + strength * uniform for p in clean)
    if kind == "shuffle":
        rng = random.Random(seed)
        shuffled = list(clean)
        rng.shuffle(shuffled)
        return normalize_prior((1.0 - strength) * p + strength * q for p, q in zip(clean, shuffled))
    if kind == "drop_head":
        values = list(clean)
        head = max(range(len(values)), key=values.__getitem__)
        values[head] *= 1.0 - strength
        return normalize_prior(values)

    raise ValueError(f"unknown prior corruption kind: {kind}")


def corrupt_counts(
    counts: Iterable[int],
    kind: str = "mix_uniform",
    strength: float = 0.25,
    seed: int | None = None,
) -> list[int]:
    clean_counts = validate_counts(counts)
    total = sum(clean_counts)
    corrupted = corrupt_prior(counts_to_prior(clean_counts), kind=kind, strength=strength, seed=seed)
    rounded = [int(round(p * total)) for p in corrupted]
    delta = total - sum(rounded)
    if delta != 0:
        order = sorted(range(len(corrupted)), key=lambda i: corrupted[i], reverse=delta > 0)
        step = 1 if delta > 0 else -1
        for i in order[: abs(delta)]:
            rounded[i] += step
    return rounded


def identity_transition(num_classes: int) -> list[list[float]]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return [[1.0 if i == j else 0.0 for j in range(num_classes)] for i in range(num_classes)]


def structured_transition(num_classes: int, beta: float) -> list[list[float]]:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    off_diag = beta / (num_classes - 1)
    return [[1.0 - beta if i == j else off_diag for j in range(num_classes)] for i in range(num_classes)]


def apply_transition(prior: Iterable[float], transition: Iterable[Iterable[float]]) -> list[float]:
    clean = normalize_prior(prior)
    matrix = [normalize_prior(row) for row in transition]
    if len(matrix) != len(clean) or any(len(row) != len(clean) for row in matrix):
        raise ValueError("transition must be a square matrix matching the prior length")
    observed = [0.0 for _ in clean]
    for true_idx, mass in enumerate(clean):
        for observed_idx, probability in enumerate(matrix[true_idx]):
            observed[observed_idx] += mass * probability
    return normalize_prior(observed)


def corrupt_open_world_prior(
    clean_prior: Iterable[float],
    gamma: float,
    routing_prior: Iterable[float],
    transition: Iterable[Iterable[float]] | None = None,
) -> list[float]:
    """Apply the UPLIFT-U v1 prior corruption equation."""
    clean = normalize_prior(clean_prior)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    routing = normalize_prior(routing_prior)
    if len(clean) != len(routing):
        raise ValueError("clean and routing priors must have the same length")
    known = clean if transition is None else apply_transition(clean, transition)
    return normalize_prior((1.0 - gamma) * p + gamma * r for p, r in zip(known, routing))
