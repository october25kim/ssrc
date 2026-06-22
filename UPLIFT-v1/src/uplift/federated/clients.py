"""Synthetic federated clients for counts-only prior decontamination."""

from __future__ import annotations

from dataclasses import dataclass
import random

from uplift.priors.corruption import corrupt_counts, corrupt_open_world_prior
from uplift.priors.counts import (
    counts_to_prior,
    make_balanced_counts,
    make_power_law_prior,
    normalize_prior,
    sample_dirichlet,
    sample_multinomial,
)


@dataclass(frozen=True)
class ClientPrior:
    client_id: int
    clean_counts: list[int]
    observed_counts: list[int]
    gamma: float = 0.0

    @property
    def clean_prior(self) -> list[float]:
        return counts_to_prior(self.clean_counts)

    @property
    def observed_prior(self) -> list[float]:
        return counts_to_prior(self.observed_counts)


@dataclass(frozen=True)
class FederatedPriorProblem:
    clients: list[ClientPrior]
    global_clean_prior: list[float]
    routing_prior: list[float]


def make_prior_decontamination_problem(
    num_clients: int,
    num_classes: int,
    samples_per_client: int,
    alpha: float,
    rho: float,
    gamma_min: float,
    gamma_max: float,
    routing: str = "head",
    seed: int | None = None,
) -> FederatedPriorProblem:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if samples_per_client <= 0:
        raise ValueError("samples_per_client must be positive")
    if not 0.0 <= gamma_min <= gamma_max < 1.0:
        raise ValueError("gamma bounds must satisfy 0 <= min <= max < 1")
    rng = random.Random(seed)
    global_prior = make_power_law_prior(num_classes, rho=rho)
    routing_prior = make_routing_prior(num_classes, routing)
    clients: list[ClientPrior] = []
    for client_id in range(num_clients):
        clean_prior = sample_dirichlet(global_prior, concentration=alpha, rng=rng)
        gamma = rng.uniform(gamma_min, gamma_max)
        observed_prior = corrupt_open_world_prior(clean_prior, gamma=gamma, routing_prior=routing_prior)
        clean_counts = sample_multinomial(samples_per_client, clean_prior, rng)
        observed_counts = sample_multinomial(samples_per_client, observed_prior, rng)
        clients.append(ClientPrior(client_id=client_id, clean_counts=clean_counts, observed_counts=observed_counts, gamma=gamma))
    return FederatedPriorProblem(clients=clients, global_clean_prior=global_prior, routing_prior=routing_prior)


def make_routing_prior(num_classes: int, routing: str) -> list[float]:
    if routing == "uniform":
        return [1.0 / num_classes for _ in range(num_classes)]
    if routing == "head":
        return make_power_law_prior(num_classes, rho=1.5)
    if routing == "tail":
        return list(reversed(make_power_law_prior(num_classes, rho=1.5)))
    raise ValueError(f"unknown routing prior: {routing}")


# Backward-compatible toy generator used by early smoke tests and examples.
def make_synthetic_clients(
    num_clients: int,
    num_classes: int,
    samples_per_client: int,
    client_skew: float,
    corruption_kind: str,
    corruption_strength: float,
    seed: int | None = None,
) -> list[ClientPrior]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0.0 <= client_skew <= 1.0:
        raise ValueError("client_skew must be in [0, 1]")

    rng = random.Random(seed)
    clients: list[ClientPrior] = []
    for client_id in range(num_clients):
        counts = make_balanced_counts(num_classes, samples_per_client)
        dominant = rng.randrange(num_classes)
        moved = int(samples_per_client * client_skew)
        counts = [max(0, c - moved // max(1, num_classes - 1)) for c in counts]
        counts[dominant] += samples_per_client - sum(counts)
        observed = corrupt_counts(
            counts,
            kind=corruption_kind,
            strength=corruption_strength,
            seed=None if seed is None else seed + client_id,
        )
        clients.append(ClientPrior(client_id=client_id, clean_counts=counts, observed_counts=observed))
    return clients
