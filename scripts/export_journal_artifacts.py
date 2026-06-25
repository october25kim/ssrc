from __future__ import annotations

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


INPUT_FILES = {
    "main": "srcc_main_journal_table.csv",
    "baseline_rows": "srcc_baseline_comparison.csv",
    "score": "srcc_score_ablation.csv",
    "gamma": "srcc_gamma_ablation.csv",
    "failure": "srcc_failure_mode_summary.csv",
    "method_aggregate": "srcc_method_comparison_aggregate.csv",
    "method_diagnostics": "srcc_method_comparison_diagnostics.csv",
}

PRIMARY_METRIC = "mean_certified_coverage_all_seeds"
CERTIFIED_ONLY_METRIC = "mean_certified_coverage_certified_seeds"
CERTIFIED_ONLY_AGG_METRIC = "mean_certified_coverage_certified_only"


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def require_columns(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")


def read_csv(report_root: Path, name: str) -> pd.DataFrame:
    path = report_root / INPUT_FILES[name]
    if not path.exists():
        raise SystemExit(f"Missing required input report: {path}")
    return pd.read_csv(path)


def sort_existing(df: pd.DataFrame, preferred: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        return df.reset_index(drop=True)
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def clean_seed_list(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def export_main_table(report_root: Path, out_dir: Path) -> Path:
    path = report_root / INPUT_FILES["main"]
    df = read_csv(report_root, "main")
    require_columns(
        df,
        path,
        [
            "dataset",
            "regime",
            "alpha",
            "baseline_name",
            "seed_count",
            "certified_seed_count",
            PRIMARY_METRIC,
            CERTIFIED_ONLY_METRIC,
        ],
    )
    out = sort_existing(df, ["dataset", "regime", "alpha", "baseline_name"])
    dest = out_dir / "main_certified_coverage_table.csv"
    out.to_csv(dest, index=False)
    return dest


def export_baseline_table(report_root: Path, out_dir: Path) -> Path:
    agg_path = report_root / INPUT_FILES["method_aggregate"]
    diag_path = report_root / INPUT_FILES["method_diagnostics"]
    agg = read_csv(report_root, "method_aggregate")
    diag = read_csv(report_root, "method_diagnostics")
    require_columns(
        agg,
        agg_path,
        [
            "dataset",
            "regime",
            "alpha",
            "baseline_name",
            "n_seeds_total",
            "n_seeds_certified",
            PRIMARY_METRIC,
            CERTIFIED_ONLY_AGG_METRIC,
        ],
    )
    require_columns(
        diag,
        diag_path,
        ["dataset", "regime", "alpha", "baseline_name", "failure_reason"],
    )
    out = agg.copy()
    out["certified_seeds"] = out.get("certified_seeds", "").map(clean_seed_list)
    out = out.rename(
        columns={
            "n_seeds_total": "seed_count",
            "n_seeds_certified": "certified_seed_count",
            CERTIFIED_ONLY_AGG_METRIC: CERTIFIED_ONLY_METRIC,
        }
    )
    diagnostics = diag[
        [
            c
            for c in [
                "dataset",
                "regime",
                "alpha",
                "baseline_name",
                "best_failed_cert_risk_ucb",
                "best_failed_cert_coverage_lcb",
                "failure_reason",
                "test_risk_le_alpha_but_cert_risk_ucb_gt_alpha",
                "coverage_collapse_count",
            ]
            if c in diag.columns
        ]
    ]
    out = out.merge(diagnostics, on=["dataset", "regime", "alpha", "baseline_name"], how="left")
    out = sort_existing(out, ["dataset", "regime", "alpha", "baseline_name"])
    dest = out_dir / "baseline_comparison_table.csv"
    out.to_csv(dest, index=False)
    return dest


def export_simple_table(
    report_root: Path,
    out_dir: Path,
    input_name: str,
    output_name: str,
    sort_cols: Sequence[str],
    required_cols: Sequence[str],
) -> Path:
    path = report_root / INPUT_FILES[input_name]
    df = read_csv(report_root, input_name)
    require_columns(df, path, required_cols)
    out = sort_existing(df, sort_cols)
    dest = out_dir / output_name
    out.to_csv(dest, index=False)
    return dest



def _flatten_column_name(prefix: str, value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return f"{prefix}{text}"


def corrupted_alpha005_label(row: pd.Series) -> str:
    regime = str(row.get("regime", "")).lower()
    alpha = pd.to_numeric(pd.Series([row.get("alpha")]), errors="coerce").iloc[0]
    if regime != "clean" and pd.notna(alpha) and abs(float(alpha) - 0.05) <= 1e-12:
        return "alpha=0.05 corrupted feasibility limit"
    return ""


def build_failure_heatmap_table(report_root: Path) -> pd.DataFrame:
    main_path = report_root / INPUT_FILES["main"]
    failure_path = report_root / INPUT_FILES["failure"]
    diagnostics_path = report_root / INPUT_FILES["method_diagnostics"]
    main = read_csv(report_root, "main")
    failure = read_csv(report_root, "failure")
    diagnostics = read_csv(report_root, "method_diagnostics")

    require_columns(
        main,
        main_path,
        ["dataset", "regime", "alpha", "baseline_name", "seed_count", "certified_seed_count"],
    )
    require_columns(
        failure,
        failure_path,
        ["dataset", "regime", "alpha", "reason", "seed_count"],
    )
    require_columns(
        diagnostics,
        diagnostics_path,
        [
            "dataset",
            "regime",
            "alpha",
            "baseline_name",
            "test_risk_le_alpha_but_cert_risk_ucb_gt_alpha",
            "coverage_collapse_count",
        ],
    )

    key_cols = ["dataset", "regime", "alpha"]
    base = main[key_cols].drop_duplicates().copy()
    base = sort_existing(base, key_cols)
    base["feasibility_limit_label"] = base.apply(corrupted_alpha005_label, axis=1)
    base["note"] = np.where(
        base["feasibility_limit_label"].eq(""),
        "failure counts are certification outcomes; test metrics are diagnostic only",
        "alpha=0.05 corrupted failures are feasibility limits; test metrics are diagnostic only",
    )

    main_counts = main.copy()
    main_counts["seed_count"] = pd.to_numeric(main_counts["seed_count"], errors="coerce").fillna(0).astype(int)
    main_counts["certified_seed_count"] = pd.to_numeric(
        main_counts["certified_seed_count"], errors="coerce"
    ).fillna(0).astype(int)
    main_counts["failed_seed_count"] = (
        main_counts["seed_count"] - main_counts["certified_seed_count"]
    ).clip(lower=0)
    baseline_pivot = main_counts.pivot_table(
        index=key_cols,
        columns="baseline_name",
        values="failed_seed_count",
        aggfunc="sum",
        fill_value=0,
    )
    baseline_pivot = baseline_pivot.rename(
        columns={c: _flatten_column_name("baseline_failed_seed_count_", c) for c in baseline_pivot.columns}
    ).reset_index()

    failure_counts = failure.copy()
    failure_counts["seed_count"] = pd.to_numeric(failure_counts["seed_count"], errors="coerce").fillna(0).astype(int)
    reason_pivot = failure_counts.pivot_table(
        index=key_cols,
        columns="reason",
        values="seed_count",
        aggfunc="sum",
        fill_value=0,
    )
    reason_pivot = reason_pivot.rename(
        columns={c: _flatten_column_name("reason_failed_seed_count_", c) for c in reason_pivot.columns}
    ).reset_index()

    diag = diagnostics.copy()
    diag["test_good_but_not_certified_count"] = bool_series(
        diag["test_risk_le_alpha_but_cert_risk_ucb_gt_alpha"]
    ).astype(int)
    diag["coverage_collapse_count"] = pd.to_numeric(
        diag["coverage_collapse_count"], errors="coerce"
    ).fillna(0).astype(int)
    diag_summary = diag.groupby(key_cols, dropna=False).agg(
        test_looks_good_but_not_certified_count=("test_good_but_not_certified_count", "sum"),
        coverage_collapse_count=("coverage_collapse_count", "sum"),
    ).reset_index()

    out = base.merge(baseline_pivot, on=key_cols, how="left")
    out = out.merge(reason_pivot, on=key_cols, how="left")
    out = out.merge(diag_summary, on=key_cols, how="left")
    numeric_cols = [
        c
        for c in out.columns
        if c.endswith("_count")
        or c.startswith("baseline_failed_seed_count_")
        or c.startswith("reason_failed_seed_count_")
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return sort_existing(out, key_cols)


def export_failure_heatmap_table(report_root: Path, out_dir: Path) -> Path:
    out = build_failure_heatmap_table(report_root)
    dest = out_dir / "failure_mode_heatmap.csv"
    out.to_csv(dest, index=False)
    return dest


def export_test_good_but_not_certified(report_root: Path, out_dir: Path) -> Path:
    path = report_root / INPUT_FILES["baseline_rows"]
    df = read_csv(report_root, "baseline_rows")
    require_columns(
        df,
        path,
        [
            "dataset",
            "regime",
            "noise_type",
            "noise_rate",
            "seed",
            "alpha",
            "baseline_name",
            "certified",
            "cert_risk_ucb",
            "cert_coverage_lcb",
            "certified_coverage_at_alpha",
            "test_risk",
            "test_coverage",
            "reason",
        ],
    )
    certified = bool_series(df["certified"])
    alpha = pd.to_numeric(df["alpha"], errors="coerce")
    cert_risk_ucb = pd.to_numeric(df["cert_risk_ucb"], errors="coerce")
    test_risk = pd.to_numeric(df["test_risk"], errors="coerce")
    mask = (~certified) & test_risk.notna() & alpha.notna() & cert_risk_ucb.notna()
    mask &= (test_risk <= alpha) & (cert_risk_ucb > alpha)
    wanted_cols = [
        "dataset",
        "regime",
        "noise_type",
        "noise_rate",
        "seed",
        "alpha",
        "baseline_name",
        "gamma",
        "score_name",
        "cert_n",
        "cert_k",
        "cert_risk_ucb",
        "cert_coverage_lcb",
        "certified",
        "certified_coverage_at_alpha",
        "test_risk",
        "test_coverage",
        "reason",
        "certificate_scope",
        "n_certified_candidates",
    ]
    out = df.loc[mask, [c for c in wanted_cols if c in df.columns]].copy()
    out = sort_existing(
        out,
        ["dataset", "regime", "alpha", "baseline_name", "seed", "cert_risk_ucb"],
    )
    dest = out_dir / "test_good_but_not_certified_table.csv"
    out.to_csv(dest, index=False)
    return dest


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"Skipping optional figures: matplotlib import failed: {exc}", file=sys.stderr)
        return None


def figure_label(row: pd.Series, extra: str | None = None) -> str:
    bits = [str(row["dataset"]), str(row["regime"]), f"alpha={row['alpha']}"]
    if extra:
        bits.append(extra)
    return "\n".join(bits)


def save_bar_figure(
    df: pd.DataFrame,
    out_path: Path,
    category_col: str,
    title: str,
    plt,
) -> None:
    require_columns(df, out_path, ["dataset", "regime", "alpha", category_col, PRIMARY_METRIC])
    plot_df = sort_existing(df, ["dataset", "regime", "alpha", category_col]).copy()
    labels = [figure_label(row, f"{category_col}={row[category_col]}") for _, row in plot_df.iterrows()]
    values = pd.to_numeric(plot_df[PRIMARY_METRIC], errors="coerce").fillna(0.0).to_numpy()
    colors = np.where(values > 0, "#4c78a8", "#b8b8b8")

    width = max(9.0, min(18.0, 0.35 * max(len(values), 1)))
    fig, ax = plt.subplots(figsize=(width, 5.0))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Mean certified accepted coverage, all seeds")
    ax.set_ylim(0.0, min(1.0, max(0.05, float(np.nanmax(values)) * 1.2 if len(values) else 1.0)))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_failure_heatmap(df: pd.DataFrame, out_path: Path, plt) -> None:
    require_columns(df, out_path, ["dataset", "regime", "alpha"])
    baseline_cols = [c for c in df.columns if c.startswith("baseline_failed_seed_count_")]
    if not baseline_cols:
        return
    plot_df = sort_existing(df.copy(), ["dataset", "regime", "alpha"])
    labels = []
    for row in plot_df.itertuples(index=False):
        label = f"{row.dataset}\n{row.regime}\nalpha={row.alpha}"
        note = getattr(row, "feasibility_limit_label", "")
        if isinstance(note, str) and note:
            label += "\nfeasibility limit"
        labels.append(label)
    matrix = plot_df[baseline_cols].to_numpy(dtype=float)
    col_labels = [c.replace("baseline_failed_seed_count_", "") for c in baseline_cols]

    fig_width = max(8.0, min(16.0, 1.4 * len(col_labels)))
    fig_height = max(4.5, min(12.0, 0.55 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("Certification failure heatmap: failed seed counts by baseline")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Failed seed count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def export_optional_figures(report_root: Path, out_dir: Path) -> List[Path]:
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    outputs: List[Path] = []
    gamma = read_csv(report_root, "gamma")
    baseline = read_csv(report_root, "method_aggregate").rename(
        columns={CERTIFIED_ONLY_AGG_METRIC: CERTIFIED_ONLY_METRIC}
    )
    failure_heatmap = build_failure_heatmap_table(report_root)

    gamma_out = out_dir / "gamma_ablation_coverage.png"
    save_bar_figure(gamma, gamma_out, "gamma", "Risk-buffered proposal effect", plt)
    outputs.append(gamma_out)

    baseline_out = out_dir / "baseline_comparison_coverage.png"
    save_bar_figure(baseline, baseline_out, "baseline_name", "Baseline comparison", plt)
    outputs.append(baseline_out)

    failure_out = out_dir / "failure_mode_heatmap.png"
    save_failure_heatmap(failure_heatmap, failure_out, plt)
    if failure_out.exists():
        outputs.append(failure_out)

    return outputs


def validate_no_test_as_guarantee(paths: Iterable[Path]) -> None:
    for path in paths:
        df = pd.read_csv(path)
        if "test_risk" in df.columns and PRIMARY_METRIC not in df.columns:
            continue
        if PRIMARY_METRIC in df.columns:
            values = pd.to_numeric(df[PRIMARY_METRIC], errors="coerce")
            if values.isna().all() and len(df):
                raise SystemExit(f"{path} has no usable {PRIMARY_METRIC} values")


def build_artifacts(report_root: Path, out_dir: Path, make_figures: bool = True) -> Dict[str, List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_outputs = [
        export_main_table(report_root, out_dir),
        export_baseline_table(report_root, out_dir),
        export_simple_table(
            report_root,
            out_dir,
            input_name="gamma",
            output_name="gamma_ablation_table.csv",
            sort_cols=["dataset", "regime", "alpha", "gamma"],
            required_cols=[
                "dataset",
                "regime",
                "alpha",
                "gamma",
                "seed_count",
                "certified_seed_count",
                PRIMARY_METRIC,
                CERTIFIED_ONLY_METRIC,
            ],
        ),
        export_simple_table(
            report_root,
            out_dir,
            input_name="score",
            output_name="score_ablation_table.csv",
            sort_cols=["dataset", "regime", "alpha", "score_name"],
            required_cols=[
                "dataset",
                "regime",
                "alpha",
                "score_name",
                "seed_count",
                "certified_seed_count",
                PRIMARY_METRIC,
                CERTIFIED_ONLY_METRIC,
            ],
        ),
        export_failure_heatmap_table(report_root, out_dir),
        export_simple_table(
            report_root,
            out_dir,
            input_name="failure",
            output_name="failure_mode_table.csv",
            sort_cols=["dataset", "regime", "alpha", "reason"],
            required_cols=[
                "dataset",
                "regime",
                "alpha",
                "reason",
                "count",
                "seed_count",
                "best_failed_cert_risk_ucb",
                "best_failed_cert_coverage_lcb",
            ],
        ),
        export_test_good_but_not_certified(report_root, out_dir),
    ]
    validate_no_test_as_guarantee(csv_outputs)
    figure_outputs = export_optional_figures(report_root, out_dir) if make_figures else []
    return {"csv": csv_outputs, "figures": figure_outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export journal-ready SRCC tables and optional figures.")
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/journal_artifacts"))
    parser.add_argument("--no-figures", action="store_true", help="Skip optional PNG figure generation.")
    args = parser.parse_args()

    outputs = build_artifacts(args.report_root, args.out_dir, make_figures=not args.no_figures)
    for path in outputs["csv"]:
        print(f"Saved table: {path}")
    for path in outputs["figures"]:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
