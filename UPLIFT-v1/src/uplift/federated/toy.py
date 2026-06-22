"""Counts-only federated prior decontamination toy experiment."""

from __future__ import annotations

from uplift.federated.aggregation import average_priors, global_observed, observed_local, uniform_r_baseline, uplift_recovery
from uplift.federated.clients import ClientPrior, make_prior_decontamination_problem
from uplift.priors.recovery import mean_prior_error


def run_federated_toy(config: dict) -> dict:
    corruption = config.get("corruption", {})
    problem = make_prior_decontamination_problem(
        num_clients=int(config["num_clients"]),
        num_classes=int(config["num_classes"]),
        samples_per_client=int(config["samples_per_client"]),
        alpha=float(config.get("alpha", 2.0)),
        rho=float(config.get("rho", 1.0)),
        gamma_min=float(corruption.get("gamma_min", corruption.get("gamma", 0.2))),
        gamma_max=float(corruption.get("gamma_max", corruption.get("gamma", 0.2))),
        routing=str(corruption.get("routing", "head")),
        seed=config.get("seed"),
    )
    clients = problem.clients
    clean_priors = [client.clean_prior for client in clients]
    observed_priors = [client.observed_prior for client in clients]
    gammas = [client.gamma for client in clients]

    observed = observed_local(observed_priors)
    global_estimates = global_observed(observed_priors)
    uniform_r = uniform_r_baseline(observed_priors, gammas=gammas).recovered_priors
    uplift = uplift_recovery(observed_priors, gammas=gammas, routing_prior=problem.routing_prior).recovered_priors

    observed_l1 = mean_prior_error(observed, clean_priors)
    uplift_l1 = mean_prior_error(uplift, clean_priors)
    return {
        "num_clients": len(clients),
        "num_classes": int(config["num_classes"]),
        "global_clean_prior": problem.global_clean_prior,
        "routing_prior": problem.routing_prior,
        "global_observed_prior": average_priors(observed_priors),
        "mean_observed_l1": observed_l1,
        "mean_global_observed_l1": mean_prior_error(global_estimates, clean_priors),
        "mean_uniform_r_l1": mean_prior_error(uniform_r, clean_priors),
        "mean_uplift_l1": uplift_l1,
        "delta_recovery": observed_l1 - uplift_l1,
        "mean_gamma": sum(gammas) / len(gammas),
    }


def _mean_l1(clients: list[ClientPrior], estimates: list[list[float]]) -> float:
    return mean_prior_error(estimates, [client.clean_prior for client in clients])
