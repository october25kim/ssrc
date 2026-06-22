from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


CIFAR10_ASYM_MAP: Dict[int, int] = {
    # Semantic-ish pair flips. This maps every class so the realized global
    # corruption rate is close to `noise_rate`.
    0: 2,  # airplane -> bird
    1: 9,  # automobile -> truck
    2: 0,  # bird -> airplane
    3: 5,  # cat -> dog
    4: 7,  # deer -> horse
    5: 3,  # dog -> cat
    6: 3,  # frog -> cat
    7: 4,  # horse -> deer
    8: 9,  # ship -> truck
    9: 1,  # truck -> automobile
}


def corrupt_labels(
    labels: np.ndarray,
    num_classes: int,
    noise_type: str,
    noise_rate: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return noisy labels and a boolean corruption mask.

    Parameters
    ----------
    labels:
        Clean integer labels.
    num_classes:
        Number of known classes.
    noise_type:
        One of {"none", "symmetric", "asymmetric"}.
    noise_rate:
        Probability of corrupting each label.
    seed:
        RNG seed.
    """
    labels = np.asarray(labels, dtype=np.int64)
    noisy = labels.copy()
    mask = np.zeros_like(labels, dtype=bool)

    if noise_type == "none" or noise_rate <= 0:
        return noisy, mask

    rng = np.random.default_rng(seed)
    flip = rng.random(len(labels)) < noise_rate

    if noise_type == "symmetric":
        for idx in np.where(flip)[0]:
            y = int(labels[idx])
            # Draw uniformly from all *other* classes.
            new_y = int(rng.integers(0, num_classes - 1))
            if new_y >= y:
                new_y += 1
            noisy[idx] = new_y
            mask[idx] = True
        return noisy, mask

    if noise_type == "asymmetric":
        if num_classes == 10:
            mapping = CIFAR10_ASYM_MAP
            for idx in np.where(flip)[0]:
                y = int(labels[idx])
                noisy[idx] = mapping[y]
                mask[idx] = True
            return noisy, mask

        # Generic fallback for CIFAR-100: cyclic confusion. This is not a canonical semantic asym noise,
        # but it is useful for stress testing the harness.
        for idx in np.where(flip)[0]:
            y = int(labels[idx])
            noisy[idx] = (y + 1) % num_classes
            mask[idx] = True
        return noisy, mask

    raise ValueError(f"Unknown noise_type={noise_type!r}")
