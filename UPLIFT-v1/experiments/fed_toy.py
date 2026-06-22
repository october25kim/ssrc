#!/usr/bin/env python3
"""Run the federated prior decontamination toy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.federated.toy import run_federated_toy
from uplift.utils.config import load_simple_yaml
from uplift.utils.logging import print_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fed_toy.yaml")
    args = parser.parse_args()
    print_json(run_federated_toy(load_simple_yaml(args.config)))


if __name__ == "__main__":
    main()
