from __future__ import annotations

"""Calibration-size sensitivity for SRCC certification.

This script reuses saved logits/labels only. It subsamples proposal and
certification splits independently, keeps the test split fixed for evaluation,
and then runs the existing SRCC proposal/certification routine.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.build_baseline_ablation_reports import (
    load_json,
    load_run_arrays,
    regime_from_row,
)
from srcc.certify import run_certification_for_alpha, threshold_direction
from srcc.scores import correctness_errors, risk_scores


ROW_COLUMNS = [
    "dataset",
    "regime",
    "noise_type",
    "noise_rate",
    "seed",
    "alpha",
    "gamma",
    "score_name",
    "requested_budget",
    "actual_prop_size",
    "actual_cert_size",
    "subsample_seed",
    "prop_n",
    "prop_k",
    "prop_coverage",
    "prop_risk",
    "cert_n",
    "cert_k",
    "cert_risk_ucb",
    "cert_coverage_lcb",
    "certified",
    "certified_coverage_at_alpha",
    "test_coverage",
    "test_risk",
    "reason",
]

AGG_COLUMNS = [
    "dataset",
    "regime",
    "alpha",
    "requested_budget",
    "gamma",
    "score_name",
    "row_count",
    "certification_rate",
    "mean_certified_coverage_at_alpha",
    "mean_cert_risk_ucb",
    "mean_test_risk",
    "mean_test_coverage",
]


def parse_budget(value: str) -> str | int:
    if value == "full":
        return value
    budget = int(value)
    if budget <= 0:
        raise argparse.ArgumentTypeError("budgets must be positive integers or 'full'")
    return budget


def requested_budget_label(budget: str | int) -> str:
    return str(budget)


def actual_size(budget: str | int, available: int) -> int:
    if budget == "full":
        return int(available)
    return int(min(int(budget), int(available)))


def deterministic_indices(available: int, size: int, seed: int, split: str) -> np.ndarray:
    if size >= available:
        return np.arange(available, dtype=np.int64)
    # Use independent streams for proposal and certification even when the
    # user supplies the same subsample_seed for both splits.
    split_offset = 0 if split == "prop" else 1_000_003
    rng = np.random.default_rng(int(seed) + split_offset)
    return np.sort(rng.choice(available, size=size, replace=False)).astype(np.int64)


def subset_arrays(arrays: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        "logits_test": arrays["logits_test"],
        "labels_test": arrays["labels_test"],
    }
    for split in ["prop", "cert"]:
        idx = indices[split]
        out[f"logits_{split}"] = arrays[f"logits_{split}"][idx]
        out[f"labels_{split}"] = arrays[f"labels_{split}"][idx]
    return out


def metadata_for_run(run_dir: Path) -> Dict[str, object]:
    metadata = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    return {
        "dataset": metadata.get("dataset"),
        "regime": metadata.get("regime") or regime_from_row(metadata),
        "noise_type": metadata.get("noise_type"),
        "noise_rate": metadata.get("noise_rate"),
        "seed": metadata.get("seed"),
    }


def rows_for_subset(
    run_dir: Path,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
    budget: str | int,
    subsample_seed: int,
    alpha_values: Sequence[float],
    gammas: Sequence[float],
    scores: Sequence[str],
    num_thresholds: int,
    min_prop_accept: int,
) -> List[Dict[str, object]]:
    prop_size = actual_size(budget, len(arrays["labels_prop"]))
    cert_size = actual_size(budget, len(arrays["labels_cert"]))
    indices = {
        "prop": deterministic_indices(len(arrays["labels_prop"]), prop_size, subsample_seed, "prop"),
        "cert": deterministic_indices(len(arrays["labels_cert"]), cert_size, subsample_seed, "cert"),
    }
    sub = subset_arrays(arrays, indices)

    prop_errors = correctness_errors(sub["logits_prop"], sub["labels_prop"])
    cert_errors = correctness_errors(sub["logits_cert"], sub["labels_cert"])
    test_errors = correctness_errors(sub["logits_test"], sub["labels_test"])
    prop_scores = risk_scores(sub["logits_prop"], scores)
    cert_scores = risk_scores(sub["logits_cert"], scores)
    test_scores = risk_scores(sub["logits_test"], scores)

    rows: List[Dict[str, object]] = []
    for alpha in alpha_values:
        for score_name in scores:
            results = run_certification_for_alpha(
                prop_scores={score_name: prop_scores[score_name]},
                prop_errors=prop_errors,
                cert_scores={score_name: cert_scores[score_name]},
                cert_errors=cert_errors,
                test_scores={score_name: test_scores[score_name]},
                test_errors=test_errors,
                alpha=float(alpha),
                gammas=gammas,
                num_thresholds=num_thresholds,
                min_prop_accept=min_prop_accept,
                bonferroni_over_gammas=True,
            )
            for result in results:
                row = {
                    **meta,
                    **result.to_dict(),
                    "requested_budget": requested_budget_label(budget),
                    "actual_prop_size": prop_size,
                    "actual_cert_size": cert_size,
                    "subsample_seed": int(subsample_seed),
                    "run_dir": str(run_dir),
                }
                if row.get("score_name") is None:
                    row["score_name"] = score_name
                    row["threshold_direction"] = threshold_direction(score_name)
                row["certified_coverage_at_alpha"] = (
                    row["cert_coverage_lcb"] if bool(row["certified"]) else 0.0
                )
                rows.append(row)
    return rows


def run_sensitivity(
    run_dirs: Sequence[Path],
    budgets: Sequence[str | int],
    subsample_seeds: Sequence[int],
    alpha_values: Sequence[float],
    gammas: Sequence[float],
    scores: Sequence[str],
    num_thresholds: int = 200,
    min_prop_accept: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        arrays = load_run_arrays(run_dir)
        meta = metadata_for_run(run_dir)
        for budget in budgets:
            for subsample_seed in subsample_seeds:
                rows.extend(
                    rows_for_subset(
                        run_dir=run_dir,
                        arrays=arrays,
                        meta=meta,
                        budget=budget,
                        subsample_seed=int(subsample_seed),
                        alpha_values=alpha_values,
                        gammas=gammas,
                        scores=scores,
                        num_thresholds=num_thresholds,
                        min_prop_accept=min_prop_accept,
                    )
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    for col in ROW_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan
    return df[ROW_COLUMNS + [c for c in df.columns if c not in ROW_COLUMNS]]


def aggregate_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)
    df = rows.copy()
    df["certified"] = df["certified"].astype(str).str.lower().isin(["true", "1", "yes"])
    group_cols = ["dataset", "regime", "alpha", "requested_budget", "gamma", "score_name"]
    grouped = []
    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "row_count": int(len(g)),
                "certification_rate": float(g["certified"].mean()),
                "mean_certified_coverage_at_alpha": float(g["certified_coverage_at_alpha"].mean()),
                "mean_cert_risk_ucb": float(pd.to_numeric(g["cert_risk_ucb"], errors="coerce").mean()),
                "mean_test_risk": float(pd.to_numeric(g["test_risk"], errors="coerce").mean()),
                "mean_test_coverage": float(pd.to_numeric(g["test_coverage"], errors="coerce").mean()),
            }
        )
        grouped.append(row)
    return pd.DataFrame(grouped)[AGG_COLUMNS]


def discover_runs(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("srcc_cifar*_seed*"))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SRCC calibration-size sensitivity from saved logits.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--runs", type=Path, nargs="*", default=None)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    parser.add_argument("--scores", type=str, nargs="+", default=["msp", "entropy", "margin", "energy", "maxlogit"])
    parser.add_argument("--budgets", type=parse_budget, nargs="+", default=[250, 500, 1000, 1500, 2000, "full"])
    parser.add_argument("--subsample-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-thresholds", type=int, default=200)
    parser.add_argument("--min-prop-accept", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_dirs = list(args.runs) if args.runs else discover_runs(args.run_root)
    if not run_dirs:
        raise SystemExit(f"No run directories found under {args.run_root}")
    missing = [str(path) for path in run_dirs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing run directories: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_sensitivity(
        run_dirs=run_dirs,
        budgets=args.budgets,
        subsample_seeds=args.subsample_seeds,
        alpha_values=args.alpha,
        gammas=args.gammas,
        scores=args.scores,
        num_thresholds=args.num_thresholds,
        min_prop_accept=args.min_prop_accept,
    )
    aggregate = aggregate_sensitivity(rows)

    rows_path = args.out_dir / "srcc_calibration_size_sensitivity_rows.csv"
    aggregate_path = args.out_dir / "srcc_calibration_size_sensitivity_aggregate.csv"
    rows.to_csv(rows_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    print(f"Saved rows: {rows_path} shape={rows.shape}")
    print(f"Saved aggregate: {aggregate_path} shape={aggregate.shape}")


if __name__ == "__main__":
    main()
