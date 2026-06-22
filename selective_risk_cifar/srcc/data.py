from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .noise import corrupt_labels


@dataclass
class SplitResult:
    train_idx: np.ndarray
    prop_idx: np.ndarray
    cert_idx: np.ndarray


class NoisyLabelSubset(Dataset):
    """Dataset wrapper that uses noisy labels for training but keeps clean labels available."""

    def __init__(
        self,
        base: Dataset,
        indices: Sequence[int],
        noisy_labels_all: Sequence[int],
        clean_labels_all: Sequence[int],
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.noisy_labels_all = np.asarray(noisy_labels_all, dtype=np.int64)
        self.clean_labels_all = np.asarray(clean_labels_all, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        x, _ = self.base[base_idx]
        noisy_y = int(self.noisy_labels_all[base_idx])
        clean_y = int(self.clean_labels_all[base_idx])
        return x, noisy_y, clean_y, base_idx


class CleanSubset(Dataset):
    """Dataset wrapper returning clean labels."""

    def __init__(self, base: Dataset, indices: Optional[Sequence[int]] = None) -> None:
        self.base = base
        if indices is None:
            self.indices = np.arange(len(base), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        x, y = self.base[base_idx]
        return x, int(y), base_idx


def cifar_transforms(dataset: str, train: bool) -> transforms.Compose:
    dataset = dataset.lower()
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset={dataset!r}")

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_cifar(dataset: str, data_root: str | Path, train: bool, download: bool = True) -> Dataset:
    dataset = dataset.lower()
    transform = cifar_transforms(dataset, train=train)
    if dataset == "cifar10":
        return datasets.CIFAR10(root=str(data_root), train=train, download=download, transform=transform)
    if dataset == "cifar100":
        return datasets.CIFAR100(root=str(data_root), train=train, download=download, transform=transform)
    raise ValueError(f"Unsupported dataset={dataset!r}")


def get_targets(ds: Dataset) -> np.ndarray:
    # torchvision CIFAR stores labels in .targets
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets, dtype=np.int64)
    raise AttributeError("Dataset does not expose .targets")


def stratified_train_prop_cert_split(
    labels: np.ndarray,
    prop_size: int,
    cert_size: int,
    seed: int,
    max_train_samples: Optional[int] = None,
) -> SplitResult:
    """Create stratified train / trusted proposal / trusted certification splits.

    The proposal and certification sets use clean labels and are excluded from corrupted training.
    """
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max() + 1)
    rng = np.random.default_rng(seed)

    prop_parts: List[np.ndarray] = []
    cert_parts: List[np.ndarray] = []
    train_parts: List[np.ndarray] = []

    prop_per_class = np.full(num_classes, prop_size // num_classes, dtype=int)
    prop_per_class[: prop_size % num_classes] += 1
    cert_per_class = np.full(num_classes, cert_size // num_classes, dtype=int)
    cert_per_class[: cert_size % num_classes] += 1

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        n_prop = int(prop_per_class[k])
        n_cert = int(cert_per_class[k])
        if n_prop + n_cert >= len(idx_k):
            raise ValueError(
                f"Class {k} has {len(idx_k)} samples, cannot allocate prop={n_prop}, cert={n_cert}."
            )
        prop_parts.append(idx_k[:n_prop])
        cert_parts.append(idx_k[n_prop : n_prop + n_cert])
        train_parts.append(idx_k[n_prop + n_cert :])

    prop_idx = np.concatenate(prop_parts)
    cert_idx = np.concatenate(cert_parts)
    train_idx = np.concatenate(train_parts)
    rng.shuffle(prop_idx)
    rng.shuffle(cert_idx)
    rng.shuffle(train_idx)

    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(train_idx):
        # Keep this approximately stratified by taking a shuffled global subset. For smoke tests only.
        train_idx = train_idx[:max_train_samples]

    return SplitResult(train_idx=train_idx, prop_idx=prop_idx, cert_idx=cert_idx)


def build_datasets(
    dataset: str,
    data_root: str | Path,
    noise_type: str,
    noise_rate: float,
    trusted_prop_size: int,
    trusted_cert_size: int,
    seed: int,
    max_train_samples: Optional[int] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dict[str, object]]:
    """Build corrupted training set and clean prop/cert/test sets."""
    train_aug = load_cifar(dataset, data_root, train=True, download=download)
    train_eval = load_cifar(dataset, data_root, train=False, download=download)
    # train_eval points at the train split? Need separate object with eval transform and train=True.
    if dataset.lower() == "cifar10":
        train_eval = datasets.CIFAR10(
            root=str(data_root), train=True, download=download, transform=cifar_transforms(dataset, train=False)
        )
        test_eval = datasets.CIFAR10(
            root=str(data_root), train=False, download=download, transform=cifar_transforms(dataset, train=False)
        )
        num_classes = 10
    elif dataset.lower() == "cifar100":
        train_eval = datasets.CIFAR100(
            root=str(data_root), train=True, download=download, transform=cifar_transforms(dataset, train=False)
        )
        test_eval = datasets.CIFAR100(
            root=str(data_root), train=False, download=download, transform=cifar_transforms(dataset, train=False)
        )
        num_classes = 100
    else:
        raise ValueError(dataset)

    clean_labels = get_targets(train_aug)
    split = stratified_train_prop_cert_split(
        clean_labels,
        prop_size=trusted_prop_size,
        cert_size=trusted_cert_size,
        seed=seed,
        max_train_samples=max_train_samples,
    )

    noisy_labels, corruption_mask = corrupt_labels(
        clean_labels,
        num_classes=num_classes,
        noise_type=noise_type,
        noise_rate=noise_rate,
        seed=seed + 12345,
    )

    train_ds = NoisyLabelSubset(train_aug, split.train_idx, noisy_labels, clean_labels)
    prop_ds = CleanSubset(train_eval, split.prop_idx)
    cert_ds = CleanSubset(train_eval, split.cert_idx)
    test_ds = CleanSubset(test_eval, None)

    metadata = {
        "dataset": dataset.lower(),
        "num_classes": num_classes,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "seed": seed,
        "num_train": int(len(train_ds)),
        "num_prop": int(len(prop_ds)),
        "num_cert": int(len(cert_ds)),
        "num_test": int(len(test_ds)),
        "actual_train_corruption_rate": float(corruption_mask[split.train_idx].mean()),
    }
    return train_ds, prop_ds, cert_ds, test_ds, metadata
