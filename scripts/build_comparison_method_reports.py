from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.build_baseline_ablation_reports import build_reports


COMPARISON_METHODS = [
    "srcc_main",
    "full_coverage",
    "no_buffer_proposal",
    "naive_empirical_threshold",
    "uncertified_empirical_threshold",
    "ltt_bonferroni",
    "clean_trained_upper_bound",
]

REQUIRED_ROW_COLUMNS = [
    "dataset",
    "regime",
    "noise_type",
    "noise_rate",
    "seed",
    "alpha",
    "baseline_name",
    "canonical_baseline_name",
    "baseline_equivalence",
    "method_family",
    "method_source",
    "gamma",
    "score_name",
    "threshold",
    "threshold_direction",
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
    "delta_risk_per_row",
    "delta_coverage_per_row",
    "reason",
    "uses_test_for_selection",
    "uses_certification_for_selection",
    "selection_correction",
    "is_primary_srcc",
    "test_looks_good_but_not_certified",
    "coverage_collapse",
]

CANONICAL_KEY = [
    "dataset",
    "regime",
    "noise_rate",
    "seed",
    "alpha",
    "baseline_name",
    "canonical_baseline_name",
]


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def ensure_baseline_report(run_root: Path, report_root: Path) -> Path:
    path = report_root / "srcc_baseline_comparison.csv"
    if path.exists():
        return path
    build_reports(run_root, report_root)
    if not path.exists():
        raise SystemExit(f"Missing required report after rebuild: {path}")
    return path


def load_baseline_rows(run_root: Path, report_root: Path) -> pd.DataFrame:
    path = ensure_baseline_report(run_root, report_root)
    df = pd.read_csv(path)
    if "baseline_name" not in df.columns:
        raise SystemExit(f"{path} has no baseline_name column")
    df["certified"] = bool_series(df["certified"])
    return df


def _clean_equivalence(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    if text in {"gamma_1_srcc_row", "same_proposal_as_gamma_1_no_buffer"}:
        return {
            "gamma_1_srcc_row": "gamma_1_no_buffer",
            "same_proposal_as_gamma_1_no_buffer": "same_as_gamma_1_no_buffer",
        }[text]
    return text


def add_method_metadata(df: pd.DataFrame, coverage_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["baseline_equivalence"] = out.get("baseline_equivalence", "").map(_clean_equivalence)
    out["canonical_baseline_name"] = out["baseline_name"]
    same_as_no_buffer = out["baseline_equivalence"].eq("same_as_gamma_1_no_buffer")
    out.loc[same_as_no_buffer, "canonical_baseline_name"] = "no_buffer_proposal"
    same_as_naive = out["baseline_equivalence"].eq("same_selector_as_naive_empirical_threshold")
    out.loc[same_as_naive, "canonical_baseline_name"] = "naive_empirical_threshold"

    out["method_family"] = "comparison"
    out.loc[out["baseline_name"].eq("srcc_main"), "method_family"] = "srcc"
    out.loc[out["baseline_name"].eq("clean_trained_upper_bound"), "method_family"] = "reference"
    out.loc[out["baseline_name"].eq("uncertified_empirical_threshold"), "method_family"] = "diagnostic"

    out["method_source"] = "reused_from_baseline_report"
    out.loc[out["baseline_name"].eq("no_buffer_proposal"), "method_source"] = "derived_from_srcc_gamma_1"
    out.loc[out["baseline_name"].eq("naive_empirical_threshold"), "method_source"] = "reused_proposal_selected_naive"
    out.loc[out["baseline_name"].eq("uncertified_empirical_threshold"), "method_source"] = "derived_from_naive_empirical_threshold"
    out.loc[out["baseline_name"].eq("clean_trained_upper_bound"), "method_source"] = "reused_from_clean_upper_bound"

    out["uses_test_for_selection"] = False
    out["uses_certification_for_selection"] = out["baseline_name"].eq("ltt_bonferroni")
    out["selection_correction"] = "proposal_split"
    out.loc[out["baseline_name"].eq("full_coverage"), "selection_correction"] = "none_needed_fixed_selector"
    out.loc[out["baseline_name"].eq("ltt_bonferroni"), "selection_correction"] = "bonferroni"
    out.loc[out["baseline_name"].eq("clean_trained_upper_bound"), "selection_correction"] = "proposal_split"
    out["is_primary_srcc"] = out["baseline_name"].eq("srcc_main")

    out["test_looks_good_but_not_certified"] = (
        ~out["certified"]
        & (pd.to_numeric(out["test_risk"], errors="coerce") <= pd.to_numeric(out["alpha"], errors="coerce"))
        & (pd.to_numeric(out["cert_risk_ucb"], errors="coerce") > pd.to_numeric(out["alpha"], errors="coerce"))
    )
    out["coverage_collapse"] = (
        pd.to_numeric(out["certified_coverage_at_alpha"], errors="coerce").fillna(0.0).eq(0.0)
        | (pd.to_numeric(out["cert_coverage_lcb"], errors="coerce").fillna(0.0) < coverage_threshold)
    )

    certified_cov = pd.to_numeric(out["certified_coverage_at_alpha"], errors="coerce").fillna(0.0)
    cert_lcb = pd.to_numeric(out["cert_coverage_lcb"], errors="coerce").fillna(0.0)
    out["certified_coverage_at_alpha"] = np.where(out["certified"], cert_lcb, 0.0)
    out["certified_coverage_at_alpha"] = out["certified_coverage_at_alpha"].astype(float)
    return out


def add_uncertified_empirical_threshold(df: pd.DataFrame) -> pd.DataFrame:
    naive = df[df["baseline_name"].eq("naive_empirical_threshold")].copy()
    if naive.empty:
        return df
    naive["baseline_name"] = "uncertified_empirical_threshold"
    naive["canonical_baseline_name"] = "naive_empirical_threshold"
    naive["baseline_equivalence"] = "same_selector_as_naive_empirical_threshold"
    naive["method_family"] = "diagnostic"
    naive["method_source"] = "derived_from_naive_empirical_threshold"
    naive["selection_correction"] = "proposal_split"
    naive["is_primary_srcc"] = False
    return pd.concat([df, naive], ignore_index=True, sort=False)


def normalize_comparison_rows(df: pd.DataFrame, coverage_threshold: float = 0.01) -> pd.DataFrame:
    wanted = df[df["baseline_name"].isin(COMPARISON_METHODS[:-1])].copy()
    clean = df[df["baseline_name"].eq("clean_trained_upper_bound")].copy()
    wanted = pd.concat([wanted, clean], ignore_index=True, sort=False)
    wanted = add_method_metadata(wanted, coverage_threshold)
    wanted = add_uncertified_empirical_threshold(wanted)
    wanted = add_method_metadata(wanted, coverage_threshold)

    for col in REQUIRED_ROW_COLUMNS:
        if col not in wanted.columns:
            wanted[col] = math.nan

    wanted = wanted.sort_values(
        by=[
            "dataset",
            "regime",
            "noise_rate",
            "seed",
            "alpha",
            "baseline_name",
            "certified",
            "certified_coverage_at_alpha",
            "cert_risk_ucb",
        ],
        ascending=[True, True, True, True, True, True, False, False, True],
        kind="mergesort",
    )
    wanted = wanted.drop_duplicates(subset=CANONICAL_KEY, keep="first")
    wanted = wanted.sort_values(CANONICAL_KEY, kind="mergesort").reset_index(drop=True)
    return wanted[REQUIRED_ROW_COLUMNS + [c for c in wanted.columns if c not in REQUIRED_ROW_COLUMNS]]


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "regime", "alpha", "baseline_name"]
    records = []
    for key, group in rows.groupby(group_cols, dropna=False):
        certified = group[group["certified"]]
        seed_values = sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int).unique().tolist())
        certified_seed_values = sorted(pd.to_numeric(certified["seed"], errors="coerce").dropna().astype(int).unique().tolist())
        record = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        record.update(
            {
                "n_seeds_total": int(len(seed_values)),
                "n_seeds_certified": int(len(certified_seed_values)),
                "certified_seeds": " ".join(str(s) for s in certified_seed_values),
                "mean_certified_coverage_all_seeds": float(group["certified_coverage_at_alpha"].mean()),
                "min_certified_coverage_all_seeds": float(group["certified_coverage_at_alpha"].min()),
                "mean_certified_coverage_certified_only": float(certified["certified_coverage_at_alpha"].mean()) if len(certified) else math.nan,
                "mean_cert_risk_ucb_on_best_certified": float(certified["cert_risk_ucb"].mean()) if len(certified) else math.nan,
                "mean_test_risk_on_best_certified": float(certified["test_risk"].mean()) if len(certified) else math.nan,
                "mean_test_coverage_on_best_certified": float(certified["test_coverage"].mean()) if len(certified) else math.nan,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def pivot_rows(aggregate: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["mean_certified_coverage_all_seeds", "n_seeds_certified"]
    pivot = aggregate.pivot_table(
        index=["dataset", "regime", "alpha"],
        columns="baseline_name",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{method}_{metric}" for metric, method in pivot.columns]
    return pivot.reset_index()


def diagnostic_rows(rows: pd.DataFrame) -> pd.DataFrame:
    failed = rows[~rows["certified"]].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "regime",
                "alpha",
                "baseline_name",
                "best_failed_cert_risk_ucb",
                "best_failed_cert_coverage_lcb",
                "failure_reason",
                "test_risk_le_alpha_but_cert_risk_ucb_gt_alpha",
            ]
        )
    records = []
    for key, group in failed.groupby(["dataset", "regime", "alpha", "baseline_name"], dropna=False):
        best = group.sort_values(
            by=["cert_coverage_lcb", "cert_risk_ucb"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        record = dict(zip(["dataset", "regime", "alpha", "baseline_name"], key if isinstance(key, tuple) else (key,)))
        record.update(
            {
                "best_failed_cert_risk_ucb": best.get("cert_risk_ucb"),
                "best_failed_cert_coverage_lcb": best.get("cert_coverage_lcb"),
                "failure_reason": best.get("reason"),
                "test_risk_le_alpha_but_cert_risk_ucb_gt_alpha": bool(group["test_looks_good_but_not_certified"].any()),
                "coverage_collapse_count": int(group["coverage_collapse"].sum()),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def validate_rows(rows: pd.DataFrame) -> None:
    if rows["uses_test_for_selection"].any():
        raise ValueError("Comparison rows must never use test split for selection.")
    duplicated = rows.duplicated(subset=CANONICAL_KEY, keep=False)
    if duplicated.any():
        dupes = rows.loc[duplicated, CANONICAL_KEY].to_dict("records")
        raise ValueError(f"Duplicate canonical comparison keys: {dupes[:5]}")
    bad_cov = rows["certified_coverage_at_alpha"] != rows["cert_coverage_lcb"].where(rows["certified"], 0.0)
    if bad_cov.any():
        raise ValueError("Invalid certified_coverage_at_alpha convention in comparison rows.")


def build_comparison_reports(
    run_root: Path,
    report_root: Path,
    out_dir: Path,
    coverage_threshold: float = 0.01,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_rows(run_root, report_root)
    rows = normalize_comparison_rows(baseline, coverage_threshold=coverage_threshold)
    validate_rows(rows)
    aggregate = aggregate_rows(rows)
    pivot = pivot_rows(aggregate)
    diagnostics = diagnostic_rows(rows)

    outputs = {
        "rows": out_dir / "srcc_method_comparison_rows.csv",
        "aggregate": out_dir / "srcc_method_comparison_aggregate.csv",
        "pivot": out_dir / "srcc_method_comparison_pivot.csv",
        "diagnostics": out_dir / "srcc_method_comparison_diagnostics.csv",
    }
    rows.to_csv(outputs["rows"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    pivot.to_csv(outputs["pivot"], index=False)
    diagnostics.to_csv(outputs["diagnostics"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-duplicating SRCC comparison-method reports.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--coverage-collapse-threshold", type=float, default=0.01)
    args = parser.parse_args()
    outputs = build_comparison_reports(
        run_root=args.run_root,
        report_root=args.report_root,
        out_dir=args.out_dir,
        coverage_threshold=args.coverage_collapse_threshold,
    )
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
