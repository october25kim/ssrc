"""Seed helpers."""

from __future__ import annotations

import random


def set_python_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(int(seed))
