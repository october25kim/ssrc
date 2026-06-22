"""Small JSON logging helpers for experiment scripts."""

from __future__ import annotations

import json


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
