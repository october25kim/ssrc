from __future__ import annotations

"""Diagnostic confidence-deformation analysis for SRCC.

This script uses saved logits and clean labels to evaluate how confidence-like
scores rank clean correctness. These diagnostics are evaluation-only: they do
not select SRCC thresholds, do not certify risk, and are not formal guarantees.

Energy follows the existing SRCC convention in ``srcc.scores`` where lower
energy is safer, so this script negates energy for higher-is-better ranking.
Entropy is handled the same way: lower entropy is safer, so ranking uses
``-entropy``.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.build_baseline_ablation_reports import regime_from_row
from srcc.scores import prediction, risk_scores


ROW_COLUMNS = [
    "dataset",
    "regime",
    "noise_type",
    "noise_rate",
    "seed",
    "split",
    "score_name",
    "correctness_auroc",
    "aurc",
    "coverage_at_test_risk_0_05",
    "coverage_at_test_risk_0_10",
    "risk_at_coverage_0_10",
    "risk_at_coverage_0_25",
    "risk_at_coverage_0_50",
    "risk_at_coverage_0_75",
    "risk_at_coverage_1_00",
    "n_examples",
    "n_correct",
    "accuracy",
    "reason",
]

AGG_COLUMNS = [
    "dataset",
    "regime",
    "score_name",
    "mean_correctness_auroc",
    "std_correctness_auroc",
    "mean_aurc",
    "std_aurc",
    "mean_coverage_at_test_risk_0_05",
    "mean_coverage_at_test_risk_0_10",
    "n_seeds",
]

FIXED_COVERAGES = [0.10, 0.25, 0.50, 0.75, 1.00]
TARGET_RISKS = [0.05, 0.10]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_for_run(run_dir: Path) -> Dict[str, object]:
    metadata = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    return {
        "dataset": metadata.get("dataset"),
        "regime": metadata.get("regime") or regime_from_row(metadata),
        "noise_type": metadata.get("noise_type"),
        "noise_rate": metadata.get("noise_rate"),
        "seed": metadata.get("seed"),
    }


def ranking_score(score_name: str, raw_score: np.ndarray) -> np.ndarray:
    """Return a score where larger values mean more likely correct."""
    raw = np.asarray(raw_score, dtype=float)
    if score_name in {"msp", "margin", "maxlogit"}:
        return raw
    if score_name in {"entropy", "energy"}:
        return -raw
    raise ValueError(f"Unknown score for ranking direction: {score_name}")


def correctness_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = prediction(logits)
    return (pred == np.asarray(labels, dtype=np.int64)).astype(np.int64)


def auroc_rank_statistic(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via average ranks, with ties assigned average rank."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def correctness_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, scores))
    except Exception:
        return float(auroc_rank_statistic(y, scores))


def risk_coverage_curve_from_ranking(correctness: np.ndarray, rank_scores: np.ndarray) -> pd.DataFrame:
    correct = np.asarray(correctness, dtype=np.int64)
    scores = np.asarray(rank_scores, dtype=float)
    if len(correct) != len(scores):
        raise ValueError("correctness and rank_scores must have the same length")
    order = np.argsort(-scores, kind="mergesort")
    ordered_correct = correct[order]
    errors = 1 - ordered_correct
    counts = np.arange(1, len(errors) + 1, dtype=float)
    risk = np.cumsum(errors) / counts
    coverage = counts / float(len(errors)) if len(errors) else np.array([], dtype=float)
    return pd.DataFrame({"coverage": coverage, "risk": risk, "rank": counts.astype(int)})


def aurc_from_curve(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    return float(pd.to_numeric(curve["risk"], errors="coerce").mean())


def coverage_at_empirical_risk(curve: pd.DataFrame, alpha: float) -> float:
    if curve.empty:
        return 0.0
    ok = curve[pd.to_numeric(curve["risk"], errors="coerce") <= float(alpha)]
    if ok.empty:
        return 0.0
    return float(ok["coverage"].max())


def risk_at_coverage(curve: pd.DataFrame, coverage_level: float) -> float:
    if curve.empty:
        return float("nan")
    coverage = pd.to_numeric(curve["coverage"], errors="coerce").to_numpy()
    idx = int(np.searchsorted(coverage, float(coverage_level), side="left"))
    idx = min(max(idx, 0), len(curve) - 1)
    return float(curve.iloc[idx]["risk"])


def auroc_reason(correctness: np.ndarray) -> str:
    n_correct = int(np.asarray(correctness, dtype=np.int64).sum())
    n = int(len(correctness))
    if n == 0:
        return "empty_split"
    if n_correct == n:
        return "auroc_undefined_all_correct"
    if n_correct == 0:
        return "auroc_undefined_all_wrong"
    return "ok"


def analyze_split(
    meta: Dict[str, object],
    split: str,
    logits: np.ndarray,
    labels: np.ndarray,
    score_names: Sequence[str],
) -> tuple[List[Dict[str, object]], List[pd.DataFrame]]:
    correctness = correctness_from_logits(logits, labels)
    scores = risk_scores(logits, score_names)
    rows: List[Dict[str, object]] = []
    curves: List[pd.DataFrame] = []
    reason = auroc_reason(correctness)
    n_examples = int(len(correctness))
    n_correct = int(correctness.sum())
    accuracy = n_correct / n_examples if n_examples else float("nan")

    for score_name in score_names:
        rank_scores = ranking_score(score_name, scores[score_name])
        curve = risk_coverage_curve_from_ranking(correctness, rank_scores)
        curve.insert(0, "score_name", score_name)
        curve.insert(0, "split", split)
        for key, value in reversed(list(meta.items())):
            curve.insert(0, key, value)
        curves.append(curve)

        row = {
            **meta,
            "split": split,
            "score_name": score_name,
            "correctness_auroc": correctness_auroc(correctness, rank_scores),
            "aurc": aurc_from_curve(curve),
            "coverage_at_test_risk_0_05": coverage_at_empirical_risk(curve, 0.05),
            "coverage_at_test_risk_0_10": coverage_at_empirical_risk(curve, 0.10),
            "risk_at_coverage_0_10": risk_at_coverage(curve, 0.10),
            "risk_at_coverage_0_25": risk_at_coverage(curve, 0.25),
            "risk_at_coverage_0_50": risk_at_coverage(curve, 0.50),
            "risk_at_coverage_0_75": risk_at_coverage(curve, 0.75),
            "risk_at_coverage_1_00": risk_at_coverage(curve, 1.00),
            "n_examples": n_examples,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "reason": reason,
        }
        rows.append(row)
    return rows, curves


def analyze_run(run_dir: Path, score_names: Sequence[str], splits: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = metadata_for_run(run_dir)
    meta["run_dir"] = str(run_dir)
    all_rows: List[Dict[str, object]] = []
    all_curves: List[pd.DataFrame] = []
    for split in splits:
        logits_path = run_dir / f"logits_{split}.npy"
        labels_path = run_dir / f"labels_{split}.npy"
        if not logits_path.exists() or not labels_path.exists():
            continue
        rows, curves = analyze_split(
            meta=meta,
            split=split,
            logits=np.load(logits_path),
            labels=np.load(labels_path),
            score_names=score_names,
        )
        all_rows.extend(rows)
        all_curves.extend(curves)
    rows_df = pd.DataFrame(all_rows)
    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    return rows_df, curves_df


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)
    test_rows = rows[rows["split"].eq("test")].copy()
    grouped = []
    for key, group in test_rows.groupby(["dataset", "regime", "score_name"], dropna=False):
        row = dict(zip(["dataset", "regime", "score_name"], key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "mean_correctness_auroc": float(pd.to_numeric(group["correctness_auroc"], errors="coerce").mean()),
                "std_correctness_auroc": float(pd.to_numeric(group["correctness_auroc"], errors="coerce").std(ddof=1)),
                "mean_aurc": float(pd.to_numeric(group["aurc"], errors="coerce").mean()),
                "std_aurc": float(pd.to_numeric(group["aurc"], errors="coerce").std(ddof=1)),
                "mean_coverage_at_test_risk_0_05": float(
                    pd.to_numeric(group["coverage_at_test_risk_0_05"], errors="coerce").mean()
                ),
                "mean_coverage_at_test_risk_0_10": float(
                    pd.to_numeric(group["coverage_at_test_risk_0_10"], errors="coerce").mean()
                ),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
        grouped.append(row)
    return pd.DataFrame(grouped)[AGG_COLUMNS]


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame(
            columns=["dataset", "regime", "score_name", "coverage_level", "mean_risk", "std_risk", "n_seeds"]
        )
    test_curves = curves[curves["split"].eq("test")].copy()
    rows = []
    for (dataset, regime, score_name, seed), group in test_curves.groupby(
        ["dataset", "regime", "score_name", "seed"], dropna=False
    ):
        for cov in FIXED_COVERAGES:
            rows.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "score_name": score_name,
                    "seed": seed,
                    "coverage_level": cov,
                    "risk": risk_at_coverage(group, cov),
                }
            )
    point_df = pd.DataFrame(rows)
    summary = []
    for key, group in point_df.groupby(["dataset", "regime", "score_name", "coverage_level"], dropna=False):
        summary.append(
            {
                "dataset": key[0],
                "regime": key[1],
                "score_name": key[2],
                "coverage_level": key[3],
                "mean_risk": float(pd.to_numeric(group["risk"], errors="coerce").mean()),
                "std_risk": float(pd.to_numeric(group["risk"], errors="coerce").std(ddof=1)),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(summary)


def plot_metric_bars(aggregate: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping {out_path}: matplotlib import failed: {exc}", file=sys.stderr)
        return
    if aggregate.empty:
        return
    df = aggregate.sort_values(["dataset", "regime", "score_name"], kind="mergesort").copy()
    labels = [f"{r.dataset}\n{r.regime}\n{r.score_name}" for r in df.itertuples(index=False)]
    values = pd.to_numeric(df[metric], errors="coerce").to_numpy()
    width = max(10.0, min(22.0, 0.32 * len(df)))
    fig, ax = plt.subplots(figsize=(width, 5.0))
    ax.bar(np.arange(len(df)), np.nan_to_num(values, nan=0.0), color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_risk_coverage(curves: pd.DataFrame, dataset: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping {out_path}: matplotlib import failed: {exc}", file=sys.stderr)
        return
    df = curves[(curves["split"].eq("test")) & (curves["dataset"].astype(str).eq(dataset))].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for (regime, score_name), group in df.groupby(["regime", "score_name"], dropna=False):
        # Downsample each seed curve then average by nearest coverage grid.
        grid = np.linspace(0.01, 1.0, 100)
        seed_risks = []
        for _, seed_group in group.groupby("seed", dropna=False):
            coverage = seed_group["coverage"].to_numpy(dtype=float)
            risk = seed_group["risk"].to_numpy(dtype=float)
            idx = np.searchsorted(coverage, grid, side="left")
            idx = np.clip(idx, 0, len(risk) - 1)
            seed_risks.append(risk[idx])
        mean_risk = np.mean(np.vstack(seed_risks), axis=0)
        ax.plot(grid, mean_risk, linewidth=1.2, label=f"{regime}/{score_name}")
    ax.set_title(f"Evaluation-only risk-coverage diagnostics: {dataset}")
    ax.set_xlabel("Test coverage")
    ax.set_ylabel("Empirical test selective risk")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def discover_runs(run_root: Path, pattern: str) -> List[Path]:
    return sorted([p for p in run_root.glob(pattern) if p.is_dir()])


def build_outputs(
    run_root: Path,
    out_dir: Path,
    runs_pattern: str,
    score_names: Sequence[str],
    splits: Sequence[str],
) -> Dict[str, Path]:
    run_dirs = discover_runs(run_root, runs_pattern)
    if not run_dirs:
        raise SystemExit(f"No runs matched {run_root / runs_pattern}")
    out_dir.mkdir(parents=True, exist_ok=True)

    row_frames = []
    curve_frames = []
    for run_dir in run_dirs:
        rows, curves = analyze_run(run_dir, score_names=score_names, splits=splits)
        if not rows.empty:
            row_frames.append(rows)
        if not curves.empty:
            curve_frames.append(curves)
    rows_df = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame(columns=ROW_COLUMNS)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    for col in ROW_COLUMNS:
        if col not in rows_df.columns:
            rows_df[col] = math.nan
    rows_df = rows_df[ROW_COLUMNS + [c for c in rows_df.columns if c not in ROW_COLUMNS]]

    aggregate = aggregate_rows(rows_df)
    curve_summary = summarize_curves(curves_df)

    outputs = {
        "rows": out_dir / "confidence_deformation_rows.csv",
        "aggregate": out_dir / "confidence_deformation_aggregate.csv",
        "curves": out_dir / "risk_coverage_curves.csv",
        "curve_summary": out_dir / "risk_coverage_summary.csv",
        "auroc_fig": out_dir / "correctness_auroc_by_regime.png",
        "aurc_fig": out_dir / "aurc_by_regime.png",
        "cifar10_curve_fig": out_dir / "risk_coverage_cifar10.png",
        "cifar100_curve_fig": out_dir / "risk_coverage_cifar100.png",
    }
    rows_df.to_csv(outputs["rows"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    curves_df.to_csv(outputs["curves"], index=False)
    curve_summary.to_csv(outputs["curve_summary"], index=False)

    plot_metric_bars(aggregate, "mean_correctness_auroc", outputs["auroc_fig"], "Correctness AUROC by regime")
    plot_metric_bars(aggregate, "mean_aurc", outputs["aurc_fig"], "AURC by regime (lower is better)")
    plot_risk_coverage(curves_df, "cifar10", outputs["cifar10_curve_fig"])
    plot_risk_coverage(curves_df, "cifar100", outputs["cifar100_curve_fig"])
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation-only confidence deformation diagnostics for SRCC.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/confidence_deformation"))
    parser.add_argument("--runs-pattern", type=str, default="srcc_cifar*_seed*")
    parser.add_argument("--scores", type=str, nargs="+", default=["msp", "entropy", "margin", "energy", "maxlogit"])
    parser.add_argument("--splits", type=str, nargs="+", default=["test"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outputs = build_outputs(
        run_root=args.run_root,
        out_dir=args.out_dir,
        runs_pattern=args.runs_pattern,
        score_names=args.scores,
        splits=args.splits,
    )
    for name, path in outputs.items():
        exists = path.exists()
        suffix = "" if exists else " (not generated)"
        print(f"Saved {name}: {path}{suffix}")
    print("Note: confidence deformation metrics are diagnostic/evaluation-only, not SRCC guarantees.")


if __name__ == "__main__":
    main()
