# UPLIFT-U v1

Minimal research scaffold for prior decontamination experiments.

This repository is intentionally small and modular. It contains:

- counts-only prior corruption utilities
- federated toy experiments for prior decontamination
- a minimal CIFAR validation scaffold
- a logit-adjusted learning hook
- configuration-first experiment entry points

No datasets, checkpoints, or generated outputs are committed by default.

## Quick Checks

Run the pure standard-library tests:

```bash
python3 -m unittest discover -s tests
```

Or run inside Docker:

```bash
docker compose run --rm uplift python3 -m unittest discover -s tests
```

Run a counts-only corruption example:

```bash
python3 experiments/corrupt_counts.py --config configs/corrupt_counts.yaml
```

Run a federated toy example:

```bash
python3 experiments/fed_toy.py --config configs/fed_toy.yaml
```

The CIFAR scaffold is deliberately minimal and reports missing optional
dependencies instead of installing anything automatically.
