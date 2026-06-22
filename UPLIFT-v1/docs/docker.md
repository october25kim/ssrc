# Docker Workflow

This project uses a minimal development container. The source tree is mounted
at `/workspace`, so edits in Cursor on the host are immediately visible inside
the container.

## Build

```bash
docker compose build
```

## Start a Shell

```bash
docker compose run --rm uplift
```

## Run Checks

```bash
docker compose run --rm uplift python3 -m unittest discover -s tests
docker compose run --rm uplift python3 experiments/corrupt_counts.py --config configs/corrupt_counts.yaml
docker compose run --rm uplift python3 experiments/fed_toy.py --config configs/fed_toy.yaml
docker compose run --rm uplift python3 experiments/cifar_validate.py --config configs/cifar_minimal.yaml
```

## GPU Note

The Compose file requests NVIDIA GPUs. If Docker is not configured with the
NVIDIA Container Toolkit, remove the `deploy.resources.reservations.devices`
block from `docker-compose.yml` or run the scripts without GPU access.
