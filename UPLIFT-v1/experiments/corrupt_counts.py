#!/usr/bin/env python3
"""Run a counts-only UPLIFT-U prior corruption example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.federated.clients import make_routing_prior
from uplift.priors.counts import counts_to_prior, make_balanced_counts, sample_multinomial
from uplift.priors.corruption import corrupt_open_world_prior
from uplift.utils.config import load_simple_yaml
from uplift.utils.logging import print_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/corrupt_counts.yaml")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    corruption = config.get("corruption", {})
    clean_counts = config.get("clean_counts")
    if clean_counts is None:
        clean_counts = make_balanced_counts(int(config["num_classes"]), int(config["total_count"]))
    clean_prior = counts_to_prior(clean_counts)
    routing_prior = make_routing_prior(len(clean_prior), str(corruption.get("routing", "head")))
    observed_prior = corrupt_open_world_prior(
        clean_prior,
        gamma=float(corruption.get("gamma", 0.2)),
        routing_prior=routing_prior,
    )
    import random

    observed_counts = sample_multinomial(sum(clean_counts), observed_prior, random.Random(config.get("seed")))
    print_json(
        {
            "clean_counts": clean_counts,
            "observed_counts": observed_counts,
            "clean_prior": clean_prior,
            "routing_prior": routing_prior,
            "gamma": float(corruption.get("gamma", 0.2)),
            "observed_prior": counts_to_prior(observed_counts),
        }
    )


if __name__ == "__main__":
    main()
