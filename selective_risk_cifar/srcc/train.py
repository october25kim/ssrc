from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets
from .models import build_model
from .utils import count_parameters, device_from_arg, ensure_dir, save_json, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct_noisy = 0
    correct_clean = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, noisy_y, clean_y, _ in pbar:
        x = x.to(device, non_blocking=True)
        noisy_y = noisy_y.to(device, non_blocking=True)
        clean_y = clean_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, noisy_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, noisy_y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total += bs
        total_loss += float(loss.detach()) * bs
        pred = logits.argmax(dim=1)
        correct_noisy += int((pred == noisy_y).sum().item())
        correct_clean += int((pred == clean_y).sum().item())
        pbar.set_postfix(loss=total_loss / total, clean_acc=correct_clean / total)

    return {
        "loss": total_loss / max(total, 1),
        "train_noisy_acc": correct_noisy / max(total, 1),
        "train_clean_acc": correct_clean / max(total, 1),
    }


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    labels_list = []
    indices_list = []
    for batch in tqdm(loader, desc="eval", leave=False):
        # CleanSubset returns x, clean_y, base_idx.
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        logits_list.append(logits)
        labels_list.append(y.numpy())
        indices_list.append(idx.numpy())
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0).astype(np.int64),
        np.concatenate(indices_list, axis=0).astype(np.int64),
    )


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = device_from_arg(args.device)

    train_ds, prop_ds, cert_ds, test_ds, metadata = build_datasets(
        dataset=args.dataset,
        data_root=args.data_root,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        trusted_prop_size=args.trusted_prop_size,
        trusted_cert_size=args.trusted_cert_size,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        download=not args.no_download,
    )

    num_classes = int(metadata["num_classes"])
    model = build_model(args.model, num_classes=num_classes).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    prop_loader = DataLoader(
        prop_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    cert_loader = DataLoader(
        cert_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.optimizer)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(args.scheduler)

    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    metadata.update(
        {
            "model": args.model,
            "num_parameters": count_parameters(model),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "amp": use_amp,
            "device": str(device),
        }
    )
    save_json(metadata, out_dir / "metadata.json")

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        if scheduler is not None:
            scheduler.step()
        metrics["epoch"] = epoch
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics['loss']:.4f} "
            f"train_clean_acc={metrics['train_clean_acc']:.4f} train_noisy_acc={metrics['train_noisy_acc']:.4f}"
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "metadata": metadata, "history": history},
                out_dir / f"checkpoint_epoch{epoch}.pt",
            )

    elapsed = time.time() - start
    torch.save(
        {"model": model.state_dict(), "metadata": metadata, "history": history},
        out_dir / "checkpoint_last.pt",
    )
    save_json({"history": history, "elapsed_sec": elapsed}, out_dir / "train_history.json")

    print("Collecting logits for prop/cert/test splits...")
    logits_prop, labels_prop, indices_prop = collect_logits(model, prop_loader, device)
    logits_cert, labels_cert, indices_cert = collect_logits(model, cert_loader, device)
    logits_test, labels_test, indices_test = collect_logits(model, test_loader, device)

    np.save(out_dir / "logits_prop.npy", logits_prop)
    np.save(out_dir / "labels_prop.npy", labels_prop)
    np.save(out_dir / "indices_prop.npy", indices_prop)
    np.save(out_dir / "logits_cert.npy", logits_cert)
    np.save(out_dir / "labels_cert.npy", labels_cert)
    np.save(out_dir / "indices_cert.npy", indices_cert)
    np.save(out_dir / "logits_test.npy", logits_test)
    np.save(out_dir / "labels_test.npy", labels_test)
    np.save(out_dir / "indices_test.npy", indices_test)

    eval_summary = {
        "prop_acc": accuracy_from_logits(logits_prop, labels_prop),
        "cert_acc": accuracy_from_logits(logits_cert, labels_cert),
        "test_acc": accuracy_from_logits(logits_test, labels_test),
    }
    print(json.dumps(eval_summary, indent=2))
    save_json(eval_summary, out_dir / "eval_summary.json")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train corrupted CIFAR classifier and export logits.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--noise-type", choices=["none", "symmetric", "asymmetric"], default="symmetric")
    p.add_argument("--noise-rate", type=float, default=0.35)
    p.add_argument("--trusted-prop-size", type=int, default=2500)
    p.add_argument("--trusted-cert-size", type=int, default=2500)
    p.add_argument("--max-train-samples", type=int, default=None, help="Smoke-test option.")
    p.add_argument("--model", choices=["resnet18", "small_cnn"], default="resnet18")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--no-download", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
