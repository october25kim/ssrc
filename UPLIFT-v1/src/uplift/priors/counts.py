"""Counts-only class-prior helpers."""

from __future__ import annotations

import random
from typing import Iterable


def validate_counts(counts: Iterable[int]) -> list[int]:
    values = [int(x) for x in counts]
    if not values:
        raise ValueError("counts must contain at least one class")
    if any(x < 0 for x in values):
        raise ValueError("counts must be non-negative")
    if sum(values) <= 0:
        raise ValueError("counts must have positive total mass")
    return values


def normalize_prior(values: Iterable[float], eps: float = 0.0) -> list[float]:
    prior = [max(float(x), 0.0) + eps for x in values]
    if not prior:
        raise ValueError("prior must contain at least one class")
    total = sum(prior)
    if total <= 0.0:
        raise ValueError("prior must have positive total mass")
    return [x / total for x in prior]


def counts_to_prior(counts: Iterable[int]) -> list[float]:
    values = validate_counts(counts)
    total = float(sum(values))
    return [x / total for x in values]


def make_balanced_counts(num_classes: int, total_count: int) -> list[int]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    base = total_count // num_classes
    remainder = total_count % num_classes
    return [base + (1 if i < remainder else 0) for i in range(num_classes)]


def l1_distance(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have the same length")
    return sum(abs(float(a) - float(b)) for a, b in zip(left_values, right_values))


def make_power_law_prior(num_classes: int, rho: float = 1.0) -> list[float]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if rho < 0.0:
        raise ValueError("rho must be non-negative")
    return normalize_prior((rank + 1) ** (-rho) for rank in range(num_classes))


def sample_dirichlet(base_prior: Iterable[float], concentration: float, rng: random.Random) -> list[float]:
    base = normalize_prior(base_prior)
    if concentration <= 0.0:
        raise ValueError("concentration must be positive")
    draws = [rng.gammavariate(max(concentration * p, 1e-12), 1.0) for p in base]
    return normalize_prior(draws)


def sample_multinomial(total_count: int, prior: Iterable[float], rng: random.Random) -> list[int]:
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    probs = normalize_prior(prior)
    counts = [0 for _ in probs]
    cumulative: list[float] = []
    running = 0.0
    for p in probs:
        running += p
        cumulative.append(running)
    cumulative[-1] = 1.0
    for _ in range(total_count):
        u = rng.random()
        for idx, threshold in enumerate(cumulative):
            if u <= threshold:
                counts[idx] += 1
                break
    return counts
