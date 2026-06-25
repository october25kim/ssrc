from __future__ import annotations

"""Post-hoc temperature scaling + SRCC pilot.

Temperature is fit only on the proposal split by minimizing multiclass NLL.
The fitted scalar rescales saved logits and then the usual SRCC proposal /
certification routine is run. The finite-sample guarantee, when present, still
comes only from cert_risk_ucb on the independent certification split.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.build_baseline_ablation_reports import bool_series, regime_from_row
from srcc.certify import best_certified_result, run_certification_for_alpha
from srcc.scores import correctness_errors, prediction, risk_scores


ROW_COLUMNS = [
    "dataset",
    "regime",
    "noise_type",
    "noise_rate",
    "seed",
    "alpha",
    "gamma",
    "score_name",
    "threshold",
    "threshold_direction",
    "method_name",
    "temperature",
    "prop_nll_before",
    "prop_nll_after",
    "temperature_fit_split",
    "temperature_fit_objective",
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
    "delta_total",
    "delta_risk",
    "delta_coverage",
    "delta_allocation",
    "certificate_scope",
    "n_certified_candidates",
    "reason",
]

COMPARISON_COLUMNS = [
    "dataset",
    "regime",
    "seed",
    "alpha",
    "original_best_certified_coverage_at_alpha",
    "temp_scaled_best_certified_coverage_at_alpha",
    "delta_certified_coverage_at_alpha",
    "original_certified",
    "temp_scaled_certified",
    "original_best_score",
    "temp_scaled_best_score",
    "temperature",
    "prop_nll_before",
    "prop_nll_after",
]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_arrays(run_dir: Path) -> Dict[str, np.ndarray]:
    out = {}
    for split in ["prop", "cert", "test"]:
        out[f"logits_{split}"] = np.load(run_dir / f"logits_{split}.npy")
        out[f"labels_{split}"] = np.load(run_dir / f"labels_{split}.npy")
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


def nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have matching lengths")
    if len(labels) == 0:
        return float("nan")
    log_norm = logsumexp(logits, axis=1)
    return float(np.mean(log_norm - logits[np.arange(len(labels)), labels]))


def temperature_grid(num: int = 200, t_min: float = 0.05, t_max: float = 10.0) -> np.ndarray:
    return np.exp(np.linspace(np.log(t_min), np.log(t_max), int(num))).astype(np.float64)


def fit_temperature_grid(
    logits_prop: np.ndarray,
    labels_prop: np.ndarray,
    grid: np.ndarray | None = None,
) -> Dict[str, float]:
    grid = temperature_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if np.any(grid <= 0):
        raise ValueError("temperature grid must be positive")
    before = nll_from_logits(logits_prop, labels_prop)
    losses = np.array([nll_from_logits(logits_prop / t, labels_prop) for t in grid], dtype=np.float64)
    best_idx = int(np.nanargmin(losses))
    best_t = float(grid[best_idx])
    after = float(losses[best_idx])
    if after > before and np.isfinite(before):
        # T=1 may not be exactly present in the log grid. Keep the fit no worse
        # than the unscaled logits by falling back to no scaling.
        best_t = 1.0
        after = before
    return {
        "temperature": best_t,
        "prop_nll_before": float(before),
        "prop_nll_after": float(after),
        "temperature_fit_split": "prop",
        "temperature_fit_objective": "nll",
    }


def scale_arrays(arrays: Dict[str, np.ndarray], temperature: float) -> Dict[str, np.ndarray]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    out = {}
    for split in ["prop", "cert", "test"]:
        out[f"logits_{split}"] = np.asarray(arrays[f"logits_{split}"], dtype=np.float64) / float(temperature)
        out[f"labels_{split}"] = arrays[f"labels_{split}"]
    return out


def argmax_preserved(arrays: Dict[str, np.ndarray], scaled: Dict[str, np.ndarray]) -> bool:
    for split in ["prop", "cert", "test"]:
        if not np.array_equal(prediction(arrays[f"logits_{split}"]), prediction(scaled[f"logits_{split}"])):
            return False
    return True


def certification_rows_for_scaled_run(
    run_dir: Path,
    alpha_values: Sequence[float],
    gammas: Sequence[float],
    scores: Sequence[str],
    num_thresholds: int,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    arrays = load_run_arrays(run_dir)
    fit = fit_temperature_grid(arrays["logits_prop"], arrays["labels_prop"])
    scaled = scale_arrays(arrays, fit["temperature"])
    if not argmax_preserved(arrays, scaled):
        raise RuntimeError(f"Positive scalar temperature changed argmax predictions for {run_dir}")

    prop_errors = correctness_errors(scaled["logits_prop"], scaled["labels_prop"])
    cert_errors = correctness_errors(scaled["logits_cert"], scaled["labels_cert"])
    test_errors = correctness_errors(scaled["logits_test"], scaled["labels_test"])
    prop_scores = risk_scores(scaled["logits_prop"], scores)
    cert_scores = risk_scores(scaled["logits_cert"], scores)
    test_scores = risk_scores(scaled["logits_test"], scores)

    meta = metadata_for_run(run_dir)
    rows: List[Dict[str, object]] = []
    for alpha in alpha_values:
        results = run_certification_for_alpha(
            prop_scores=prop_scores,
            prop_errors=prop_errors,
            cert_scores=cert_scores,
            cert_errors=cert_errors,
            test_scores=test_scores,
            test_errors=test_errors,
            alpha=float(alpha),
            gammas=gammas,
            num_thresholds=num_thresholds,
            bonferroni_over_gammas=True,
        )
        for result in results:
            row = {
                **meta,
                **result.to_dict(),
                "method_name": "temperature_scaled_srcc",
                **fit,
                "run_dir": str(run_dir),
                "guarantee_source": "cert_risk_ucb_on_independent_certification_split",
            }
            row["certified_coverage_at_alpha"] = (
                row["cert_coverage_lcb"] if bool(row["certified"]) else 0.0
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ROW_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan
    return df[ROW_COLUMNS + [c for c in df.columns if c not in ROW_COLUMNS]], fit


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    df["certified"] = bool_series(df["certified"])
    group_cols = ["dataset", "regime", "noise_type", "noise_rate", "alpha", "gamma", "score_name"]
    records = []
    for key, group in df.groupby(group_cols, dropna=False):
        cert = group[group["certified"]]
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "seed_count": int(group["seed"].nunique()),
                "row_count": int(len(group)),
                "certified_seed_count": int(cert["seed"].nunique()),
                "certification_rate": float(group["certified"].mean()),
                "mean_certified_coverage_at_alpha": float(group["certified_coverage_at_alpha"].mean()),
                "mean_cert_risk_ucb": float(pd.to_numeric(group["cert_risk_ucb"], errors="coerce").mean()),
                "mean_test_risk": float(pd.to_numeric(group["test_risk"], errors="coerce").mean()),
                "mean_temperature": float(pd.to_numeric(group["temperature"], errors="coerce").mean()),
                "mean_prop_nll_before": float(pd.to_numeric(group["prop_nll_before"], errors="coerce").mean()),
                "mean_prop_nll_after": float(pd.to_numeric(group["prop_nll_after"], errors="coerce").mean()),
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def best_certified_record(df: pd.DataFrame, alpha: float) -> Dict[str, object]:
    subset = df[pd.to_numeric(df["alpha"], errors="coerce").eq(float(alpha))].copy()
    if subset.empty:
        return {"certified": False, "coverage": 0.0, "score": None}
    subset["certified"] = bool_series(subset["certified"])
    cert = subset[subset["certified"]].copy()
    if cert.empty:
        return {"certified": False, "coverage": 0.0, "score": None}
    cert = cert.sort_values(
        ["certified_coverage_at_alpha", "cert_coverage_lcb", "cert_risk_ucb"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    best = cert.iloc[0]
    return {
        "certified": True,
        "coverage": float(best["certified_coverage_at_alpha"]),
        "score": best.get("score_name"),
    }


def comparison_rows(
    run_dir: Path,
    temp_rows: pd.DataFrame,
    fit: Dict[str, float],
    alpha_values: Sequence[float],
) -> List[Dict[str, object]]:
    meta = metadata_for_run(run_dir)
    original_path = run_dir / "certification_results.csv"
    original = pd.read_csv(original_path) if original_path.exists() else pd.DataFrame()
    rows = []
    for alpha in alpha_values:
        orig_best = best_certified_record(original, alpha) if not original.empty else {
            "certified": False,
            "coverage": 0.0,
            "score": None,
        }
        temp_best = best_certified_record(temp_rows, alpha)
        rows.append(
            {
                "dataset": meta.get("dataset"),
                "regime": meta.get("regime"),
                "seed": meta.get("seed"),
                "alpha": float(alpha),
                "original_best_certified_coverage_at_alpha": orig_best["coverage"],
                "temp_scaled_best_certified_coverage_at_alpha": temp_best["coverage"],
                "delta_certified_coverage_at_alpha": temp_best["coverage"] - orig_best["coverage"],
                "original_certified": bool(orig_best["certified"]),
                "temp_scaled_certified": bool(temp_best["certified"]),
                "original_best_score": orig_best["score"],
                "temp_scaled_best_score": temp_best["score"],
                "temperature": fit["temperature"],
                "prop_nll_before": fit["prop_nll_before"],
                "prop_nll_after": fit["prop_nll_after"],
            }
        )
    return rows


def discover_runs(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("srcc_cifar*_seed*"))


def run_temperature_scaled_srcc(
    run_dirs: Sequence[Path],
    alpha_values: Sequence[float],
    gammas: Sequence[float],
    scores: Sequence[str],
    num_thresholds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_frames = []
    comparison_records = []
    for run_dir in run_dirs:
        rows, fit = certification_rows_for_scaled_run(
            run_dir=run_dir,
            alpha_values=alpha_values,
            gammas=gammas,
            scores=scores,
            num_thresholds=num_thresholds,
        )
        row_frames.append(rows)
        comparison_records.extend(comparison_rows(run_dir, rows, fit, alpha_values))
    all_rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame(columns=ROW_COLUMNS)
    aggregate = aggregate_rows(all_rows)
    comparison = pd.DataFrame(comparison_records)
    for col in COMPARISON_COLUMNS:
        if col not in comparison.columns:
            comparison[col] = math.nan
    return all_rows, aggregate, comparison[COMPARISON_COLUMNS]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling + SRCC pilot from saved logits.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--runs", type=Path, nargs="*", default=None)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    parser.add_argument("--scores", type=str, nargs="+", default=["msp", "entropy", "margin", "energy", "maxlogit"])
    parser.add_argument("--num-thresholds", type=int, default=200)
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

    rows, aggregate, comparison = run_temperature_scaled_srcc(
        run_dirs=run_dirs,
        alpha_values=args.alpha,
        gammas=args.gammas,
        scores=args.scores,
        num_thresholds=args.num_thresholds,
    )
    rows_path = args.out_dir / "srcc_temperature_scaled_rows.csv"
    aggregate_path = args.out_dir / "srcc_temperature_scaled_aggregate.csv"
    comparison_path = args.out_dir / "srcc_temperature_scaled_comparison.csv"
    rows.to_csv(rows_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved rows: {rows_path} shape={rows.shape}")
    print(f"Saved aggregate: {aggregate_path} shape={aggregate.shape}")
    print(f"Saved comparison: {comparison_path} shape={comparison.shape}")
    print("Guarantee note: any guarantee comes from cert_risk_ucb on the independent certification split.")


if __name__ == "__main__":
    main()
