"""Minimal CIFAR validation scaffold.

This is intentionally a scaffold: the ICDM v1 image experiment only validates
whether recovered priors help downstream logit adjustment.
"""

from __future__ import annotations


def describe_cifar_scaffold(config: dict) -> dict:
    dataset = str(config.get("dataset", "CIFAR10"))
    if dataset not in {"CIFAR10", "CIFAR100"}:
        raise ValueError("dataset must be CIFAR10 or CIFAR100")
    known_classes = int(config.get("known_classes", 6 if dataset == "CIFAR10" else 80))
    unknown_classes = int(config.get("unknown_classes", 4 if dataset == "CIFAR10" else 20))
    methods = config.get("methods", ["CE", "LA-Observed", "LA-Recovered", "LA-Oracle"])
    return {
        "dataset": dataset,
        "data_root": str(config.get("data_root", "./data")),
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "num_clients": int(config.get("num_clients", 10)),
        "imbalance_ratio": int(config.get("imbalance_ratio", 50)),
        "unknown_contamination": float(config.get("unknown_contamination", 0.2)),
        "batch_size": int(config.get("batch_size", 128)),
        "methods": methods,
        "metrics": ["balanced_accuracy", "few_shot_accuracy", "auroc", "known_accuracy"],
        "status": "scaffold_only",
        "message": "Install torch/torchvision separately before enabling CIFAR validation.",
    }
