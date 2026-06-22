"""Federated counts-only prior decontamination components."""

from .aggregation import average_priors, global_observed, observed_local, uniform_r_baseline, uplift_recovery
from .clients import ClientPrior, FederatedPriorProblem, make_prior_decontamination_problem, make_synthetic_clients

__all__ = [
    "average_priors",
    "global_observed",
    "observed_local",
    "uniform_r_baseline",
    "uplift_recovery",
    "ClientPrior",
    "FederatedPriorProblem",
    "make_prior_decontamination_problem",
    "make_synthetic_clients",
]
