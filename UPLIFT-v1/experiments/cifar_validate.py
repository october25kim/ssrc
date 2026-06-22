#!/usr/bin/env python3
"""Minimal CIFAR-10 validation for UPLIFT-U prior decontamination.

This is intentionally small: one compact CNN is trained on a controlled
open-world corrupted known-label training set, then the same logits are evaluated
with different logit-adjustment priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from uplift.federated.clients import make_routing_prior
from uplift.priors.counts import counts_to_prior, normalize_prior
from uplift.priors.recovery import recover_known_priors
from uplift.utils.config import load_simple_yaml


class RelabeledSubset(Dataset):
    def __init__(self, base, pairs: list[tuple[int, int]]):
        self.base = base
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        base_idx, label = self.pairs[idx]
        image, _ = self.base[base_idx]
        return image, label


class SmallCifarCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar_minimal.yaml")
    parser.add_argument("--output", default="results/cifar10_results.csv")
    parser.add_argument("--summary", default="results/cifar10_summary.json")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    result = run_cifar10_validation(config)
    write_rows(Path(args.output), result["rows"])
    Path(args.summary).write_text(json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


def run_cifar10_validation(config: dict) -> dict:
    seed = int(config.get("seed", 19))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = str(config.get("data_root", "./data"))
    known_classes = int(config.get("known_classes", 6))
    unknown_classes = int(config.get("unknown_classes", 4))
    batch_size = int(config.get("batch_size", 128))
    epochs = int(config.get("epochs", 3))
    known_total = int(config.get("known_train_total", 1800))
    test_limit = int(config.get("test_limit", 2000))
    gamma = float(config.get("unknown_contamination", 0.20))
    tau = float(config.get("logit_adjustment", {}).get("tau", 1.0))
    methods = config.get("methods", ["CE", "LA-Observed", "LA-Recovered", "LA-Oracle"])

    if known_classes + unknown_classes > 10:
        raise ValueError("CIFAR10 has only 10 classes")
    known_ids = list(range(known_classes))
    unknown_ids = list(range(known_classes, known_classes + unknown_classes))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_base = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_base = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)

    rng = random.Random(seed)
    clean_counts = make_long_tail_counts(known_classes, known_total, int(config.get("imbalance_ratio", 50)))
    routing_prior = make_routing_prior(known_classes, str(config.get("routing", "head")))
    train_pairs, observed_counts = build_corrupted_train_pairs(
        train_base.targets,
        known_ids,
        unknown_ids,
        clean_counts,
        gamma,
        routing_prior,
        rng,
    )
    rng.shuffle(train_pairs)

    clean_prior = counts_to_prior(clean_counts)
    observed_prior = counts_to_prior(observed_counts)
    recovered_prior = recover_known_priors([observed_prior], gammas=[gamma], routing_prior=routing_prior).recovered_priors[0]
    priors = {
        "CE": None,
        "LA-Observed": observed_prior,
        "LA-Recovered": recovered_prior,
        "LA-Oracle": clean_prior,
    }

    train_loader = DataLoader(RelabeledSubset(train_base, train_pairs), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    known_eval, unknown_eval = build_eval_pairs(test_base.targets, known_ids, unknown_ids, test_limit, rng)
    known_loader = DataLoader(RelabeledSubset(test_base, known_eval), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    unknown_loader = DataLoader(RelabeledSubset(test_base, unknown_eval), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = SmallCifarCNN(known_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"epoch {epoch + 1}/{epochs} loss={train_loss:.4f}")

    rows = []
    summary = {
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "epochs": epochs,
        "clean_counts": clean_counts,
        "observed_counts": observed_counts,
        "clean_prior": clean_prior,
        "observed_prior": observed_prior,
        "recovered_prior": recovered_prior,
        "routing_prior": routing_prior,
        "gamma": gamma,
        "methods": {},
    }
    for method in methods:
        metrics = evaluate_method(model, known_loader, unknown_loader, priors.get(method), tau, clean_counts, device)
        summary["methods"][method] = metrics
        for metric, value in metrics.items():
            rows.append({"method": method, "prior": prior_name(method), "seed": seed, "metric": metric, "value": value, "accuracy": value if metric == "balanced_accuracy" else value})
    return {"rows": rows, "summary": summary}


def make_long_tail_counts(num_classes: int, total: int, imbalance_ratio: int) -> list[int]:
    if num_classes <= 1:
        return [total]
    weights = [imbalance_ratio ** (-i / (num_classes - 1)) for i in range(num_classes)]
    probs = normalize_prior(weights)
    counts = [max(1, int(round(total * p))) for p in probs]
    delta = total - sum(counts)
    order = list(range(num_classes)) if delta > 0 else list(reversed(range(num_classes)))
    for i in range(abs(delta)):
        counts[order[i % num_classes]] += 1 if delta > 0 else -1
    return counts


def build_corrupted_train_pairs(targets, known_ids, unknown_ids, clean_counts, gamma, routing_prior, rng):
    by_class = group_indices(targets)
    pairs: list[tuple[int, int]] = []
    observed_counts = [0 for _ in known_ids]
    for local_label, class_id in enumerate(known_ids):
        selected = sample_indices(by_class[class_id], clean_counts[local_label], rng)
        pairs.extend((idx, local_label) for idx in selected)
        observed_counts[local_label] += len(selected)
    unknown_total = int(round(sum(clean_counts) * gamma / max(1e-12, 1.0 - gamma)))
    unknown_pool = [idx for class_id in unknown_ids for idx in by_class[class_id]]
    routed_labels = sample_labels(unknown_total, routing_prior, rng)
    selected_unknown = sample_indices(unknown_pool, unknown_total, rng)
    for idx, label in zip(selected_unknown, routed_labels):
        pairs.append((idx, label))
        observed_counts[label] += 1
    return pairs, observed_counts


def build_eval_pairs(targets, known_ids, unknown_ids, limit, rng):
    by_class = group_indices(targets)
    per_known = max(1, limit // max(1, len(known_ids)))
    per_unknown = max(1, limit // max(1, len(unknown_ids)))
    known_pairs = []
    for local_label, class_id in enumerate(known_ids):
        known_pairs.extend((idx, local_label) for idx in sample_indices(by_class[class_id], per_known, rng))
    unknown_pairs = []
    for class_id in unknown_ids:
        unknown_pairs.extend((idx, 0) for idx in sample_indices(by_class[class_id], per_unknown, rng))
    return known_pairs, unknown_pairs


def group_indices(targets):
    groups = defaultdict(list)
    for idx, target in enumerate(targets):
        groups[int(target)].append(idx)
    return groups


def sample_indices(indices, count, rng):
    if count <= len(indices):
        return rng.sample(indices, count)
    return [rng.choice(indices) for _ in range(count)]


def sample_labels(count, prior, rng):
    labels = []
    cumulative = []
    total = 0.0
    for p in normalize_prior(prior):
        total += p
        cumulative.append(total)
    cumulative[-1] = 1.0
    for _ in range(count):
        u = rng.random()
        for label, threshold in enumerate(cumulative):
            if u <= threshold:
                labels.append(label)
                break
    return labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total += labels.numel()
    return total_loss / max(1, total)


def evaluate_method(model, known_loader, unknown_loader, prior, tau, clean_counts, device):
    model.eval()
    class_correct = [0 for _ in clean_counts]
    class_total = [0 for _ in clean_counts]
    known_scores = []
    unknown_scores = []
    with torch.no_grad():
        for images, labels in known_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = apply_logit_adjustment(model(images), prior, tau, device)
            pred = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            unknown_scores.extend((-probs.max(dim=1).values).detach().cpu().tolist())
            for label, ok in zip(labels.cpu().tolist(), pred.eq(labels).cpu().tolist()):
                class_total[label] += 1
                class_correct[label] += int(ok)
        for images, _ in unknown_loader:
            images = images.to(device, non_blocking=True)
            logits = apply_logit_adjustment(model(images), prior, tau, device)
            probs = torch.softmax(logits, dim=1)
            unknown_scores.extend((-probs.max(dim=1).values).detach().cpu().tolist())
    recalls = [class_correct[i] / class_total[i] for i in range(len(class_total)) if class_total[i] > 0]
    tail_cut = max(1, len(clean_counts) // 3)
    tail_classes = sorted(range(len(clean_counts)), key=lambda i: clean_counts[i])[:tail_cut]
    tail_recalls = [class_correct[i] / class_total[i] for i in tail_classes if class_total[i] > 0]
    return {
        "balanced_accuracy": sum(recalls) / len(recalls),
        "few_shot_accuracy": sum(tail_recalls) / len(tail_recalls),
        "known_accuracy": sum(class_correct) / max(1, sum(class_total)),
        "auroc": binary_auroc(unknown_scores, known_scores),
    }


def apply_logit_adjustment(logits, prior, tau, device):
    if prior is None:
        return logits
    offsets = torch.tensor([-tau * math.log(max(float(p), 1e-12)) for p in normalize_prior(prior)], device=device, dtype=logits.dtype)
    return logits + offsets


def binary_auroc(pos_scores, neg_scores):
    pairs = [(score, 1) for score in pos_scores] + [(score, 0) for score in neg_scores]
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prior_name(method: str) -> str:
    return {
        "CE": "none",
        "LA-Observed": "observed",
        "LA-Recovered": "uplift",
        "LA-Oracle": "clean",
    }.get(method, method)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "prior", "seed", "metric", "value", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
