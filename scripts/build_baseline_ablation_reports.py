from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from srcc.certify import (
    Candidate,
    acceptance_mask,
    certify_candidate,
    certification_reason,
    clopper_pearson_lcb,
    clopper_pearson_ucb,
    empirical_counts,
    make_thresholds,
    propose_candidate,
    run_certification_for_alpha,
    threshold_direction,
    zero_error_min_n,
)
from srcc.scores import correctness_errors, risk_scores

SCORES = ["msp", "entropy", "margin", "energy", "maxlogit"]
BASELINE_COLUMNS = [
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
    "baseline_name",
    "baseline_equivalence",
    "prop_coverage",
    "prop_risk",
    "prop_n",
    "prop_k",
    "cert_n",
    "cert_k",
    "cert_risk_ucb",
    "cert_coverage_lcb",
    "certified",
    "certified_coverage_at_alpha",
    "test_coverage",
    "test_risk",
    "reason",
    "delta_total",
    "delta_risk",
    "delta_coverage",
    "delta_allocation",
    "certificate_scope",
    "n_certified_candidates",
    "delta_risk_per_row",
    "delta_coverage_per_row",
    "run_dir",
]


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def regime_from_row(row: Dict[str, object]) -> str:
    noise_type = str(row.get("noise_type") or "").lower()
    noise_rate = float(row.get("noise_rate") or 0.0)
    if noise_type in {"clean", "none"} or noise_rate == 0.0:
        return "clean"
    pct = int(round(noise_rate * 100))
    if noise_type.startswith("sym"):
        return f"sym{pct}"
    if noise_type.startswith("asym"):
        return f"asym{pct}"
    return f"{noise_type}{pct}" if noise_type else "unknown"


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def read_existing_rows(run_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_root.glob("srcc_cifar*_seed*/certification_results.csv")):
        df = pd.read_csv(path)
        df["run_dir"] = str(path.parent)
        if "regime" not in df.columns:
            df["regime"] = [regime_from_row(r) for r in df.to_dict("records")]
        df["certified"] = bool_series(df["certified"])
        df["baseline_name"] = "srcc_main_row"
        frames.append(df)
    if not frames:
        raise SystemExit(f"No SRCC CIFAR certification_results.csv files found under {run_root}")
    return pd.concat(frames, ignore_index=True)


def best_row(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(
        by=["certified", "certified_coverage_at_alpha", "cert_coverage_lcb", "cert_risk_ucb", "prop_coverage"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    )
    return ordered.iloc[0].copy()


def base_deltas(existing_rows: pd.DataFrame) -> Dict[str, float | str]:
    row = existing_rows.iloc[0]
    delta_total = float(row.get("delta_total", 0.05))
    allocation = str(row.get("delta_allocation", "joint_split"))
    if allocation == "risk_only_legacy":
        return {
            "delta_total": delta_total,
            "delta_risk": delta_total,
            "delta_coverage": delta_total,
            "delta_allocation": allocation,
        }
    return {
        "delta_total": delta_total,
        "delta_risk": float(row.get("delta_risk", delta_total / 2.0)),
        "delta_coverage": float(row.get("delta_coverage", delta_total / 2.0)),
        "delta_allocation": allocation,
    }


def load_run_arrays(run_dir: Path) -> Dict[str, np.ndarray]:
    out = {}
    for split in ["prop", "cert", "test"]:
        out[f"logits_{split}"] = np.load(run_dir / f"logits_{split}.npy")
        out[f"labels_{split}"] = np.load(run_dir / f"labels_{split}.npy")
    return out


def metadata_for_run(run_dir: Path, existing_rows: pd.DataFrame) -> Dict[str, object]:
    metadata = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    row = existing_rows.iloc[0].to_dict()
    merged = {**row, **metadata}
    return {
        "dataset": merged.get("dataset"),
        "regime": row.get("regime") or regime_from_row(merged),
        "noise_type": merged.get("noise_type"),
        "noise_rate": merged.get("noise_rate"),
        "seed": merged.get("seed"),
        "model": merged.get("model"),
        "epochs": merged.get("epochs"),
        "run_dir": str(run_dir),
    }


def row_from_result(result, meta: Dict[str, object], baseline_name: str, equivalence: str = "") -> Dict[str, object]:
    row = {**meta, **result.to_dict()}
    row["baseline_name"] = baseline_name
    row["baseline_equivalence"] = equivalence
    row["delta_risk_per_row"] = row.get("delta_risk")
    row["delta_coverage_per_row"] = row.get("delta_coverage")
    return row


def full_coverage_row(
    meta: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    alpha: float,
    deltas: Dict[str, float | str],
) -> Dict[str, object]:
    prop_errors = correctness_errors(arrays["logits_prop"], arrays["labels_prop"])
    cert_errors = correctness_errors(arrays["logits_cert"], arrays["labels_cert"])
    test_errors = correctness_errors(arrays["logits_test"], arrays["labels_test"])
    prop_n = int(len(prop_errors))
    prop_k = int(prop_errors.sum())
    cert_n = int(len(cert_errors))
    cert_k = int(cert_errors.sum())
    test_n = int(len(test_errors))
    test_k = int(test_errors.sum())
    delta_risk = float(deltas["delta_risk"])
    delta_coverage = float(deltas["delta_coverage"])
    cert_ucb = clopper_pearson_ucb(cert_k, cert_n, delta_risk)
    cert_lcb = clopper_pearson_lcb(cert_n, cert_n, delta_coverage)
    certified = bool(cert_n > 0 and cert_ucb <= alpha)
    reason = certification_reason(cert_n, cert_k, cert_ucb, alpha)
    return {
        **meta,
        "alpha": float(alpha),
        "gamma": math.nan,
        "score_name": "full_coverage",
        "threshold": math.nan,
        "threshold_direction": "all",
        "baseline_name": "full_coverage",
        "baseline_equivalence": "",
        "prop_coverage": 1.0 if prop_n else 0.0,
        "prop_risk": prop_k / prop_n if prop_n else math.nan,
        "prop_n": prop_n,
        "prop_k": prop_k,
        "cert_n": cert_n,
        "cert_k": cert_k,
        "cert_risk_ucb": cert_ucb,
        "cert_coverage_lcb": cert_lcb,
        "certified": certified,
        "certified_coverage_at_alpha": cert_lcb if certified else 0.0,
        "test_coverage": 1.0 if test_n else 0.0,
        "test_risk": test_k / test_n if test_n else math.nan,
        "reason": reason,
        "delta_total": float(deltas["delta_total"]),
        "delta_risk": delta_risk,
        "delta_coverage": delta_coverage,
        "delta_allocation": str(deltas["delta_allocation"]),
        "certificate_scope": "single_selector",
        "n_certified_candidates": 1,
        "delta_risk_per_row": delta_risk,
        "delta_coverage_per_row": delta_coverage,
    }


def naive_row(
    meta: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    alpha: float,
    deltas: Dict[str, float | str],
    gamma_one_row: Optional[pd.Series],
    num_thresholds: int,
) -> Dict[str, object]:
    prop_errors = correctness_errors(arrays["logits_prop"], arrays["labels_prop"])
    cert_errors = correctness_errors(arrays["logits_cert"], arrays["labels_cert"])
    test_errors = correctness_errors(arrays["logits_test"], arrays["labels_test"])
    prop_scores = risk_scores(arrays["logits_prop"], SCORES)
    cert_scores = risk_scores(arrays["logits_cert"], SCORES)
    test_scores = risk_scores(arrays["logits_test"], SCORES)
    cand = propose_candidate(prop_scores, prop_errors, alpha=alpha, gamma=1.0, num_thresholds=num_thresholds)
    if cand is None:
        row = full_coverage_row(meta, arrays, alpha, deltas)
        row.update(
            {
                "baseline_name": "naive_empirical_threshold",
                "score_name": None,
                "threshold_direction": None,
                "prop_coverage": 0.0,
                "prop_risk": math.nan,
                "prop_n": 0,
                "prop_k": 0,
                "cert_n": 0,
                "cert_k": 0,
                "cert_risk_ucb": 1.0,
                "cert_coverage_lcb": 0.0,
                "certified": False,
                "certified_coverage_at_alpha": 0.0,
                "test_coverage": 0.0,
                "test_risk": math.nan,
                "reason": "no_proposal_candidate",
            }
        )
        return row
    result = certify_candidate(
        cand,
        cert_scores,
        cert_errors,
        test_scores,
        test_errors,
        alpha=alpha,
        delta_risk=float(deltas["delta_risk"]),
        delta_coverage=float(deltas["delta_coverage"]),
        delta_total=float(deltas["delta_total"]),
        delta_allocation=str(deltas["delta_allocation"]),
        certificate_scope="single_selector",
        n_certified_candidates=1,
    )
    equivalence = ""
    if gamma_one_row is not None:
        same_score = result.score_name == gamma_one_row.get("score_name")
        same_threshold = np.isclose(float(result.threshold), float(gamma_one_row.get("threshold")), equal_nan=True)
        if same_score and same_threshold:
            equivalence = "same_proposal_as_gamma_1_no_buffer"
    return row_from_result(result, meta, "naive_empirical_threshold", equivalence)


def ltt_bonferroni_row(
    meta: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    alpha: float,
    deltas: Dict[str, float | str],
    num_thresholds: int,
) -> Dict[str, object]:
    prop_errors = correctness_errors(arrays["logits_prop"], arrays["labels_prop"])
    cert_errors = correctness_errors(arrays["logits_cert"], arrays["labels_cert"])
    test_errors = correctness_errors(arrays["logits_test"], arrays["labels_test"])
    prop_scores = risk_scores(arrays["logits_prop"], SCORES)
    cert_scores = risk_scores(arrays["logits_cert"], SCORES)
    test_scores = risk_scores(arrays["logits_test"], SCORES)

    candidates: List[Candidate] = []
    for score_name, scores in prop_scores.items():
        direction = threshold_direction(score_name)
        for threshold in make_thresholds(scores, num_thresholds=num_thresholds):
            n, k, cov, risk = empirical_counts(scores, prop_errors, float(threshold), direction)
            candidates.append(
                Candidate(
                    score_name=score_name,
                    threshold=float(threshold),
                    threshold_direction=direction,
                    gamma=math.nan,
                    alpha=float(alpha),
                    prop_coverage=float(cov),
                    prop_risk=float(risk) if math.isfinite(risk) else math.nan,
                    prop_n=int(n),
                    prop_k=int(k),
                )
            )
    j = max(1, len(candidates))
    row_delta_risk = float(deltas["delta_risk"]) / j
    row_delta_coverage = float(deltas["delta_coverage"]) / j
    results = [
        certify_candidate(
            cand,
            cert_scores,
            cert_errors,
            test_scores,
            test_errors,
            alpha=alpha,
            delta_risk=row_delta_risk,
            delta_coverage=row_delta_coverage,
            delta_total=float(deltas["delta_total"]),
            delta_allocation=str(deltas["delta_allocation"]),
            certificate_scope="simultaneous_rows",
            n_certified_candidates=j,
        )
        for cand in candidates
    ]
    if not results:
        return {**meta, "alpha": alpha, "baseline_name": "ltt_bonferroni", "reason": "no_proposal_candidate"}
    chosen = best_result_from_records([r.to_dict() for r in results])
    chosen["baseline_name"] = "ltt_bonferroni"
    chosen["baseline_equivalence"] = ""
    chosen["delta_risk_per_row"] = row_delta_risk
    chosen["delta_coverage_per_row"] = row_delta_coverage
    return {**meta, **chosen}


def best_result_from_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    df = pd.DataFrame(records)
    df["certified"] = bool_series(df["certified"])
    return best_row(df).to_dict()


def srcc_main_rows(all_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "regime", "noise_type", "noise_rate", "seed", "alpha"]
    rows = []
    for key, group in all_rows.groupby(keys, dropna=False):
        row = best_row(group).to_dict()
        for col, value in zip(keys, key if isinstance(key, tuple) else (key,)):
            row[col] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["baseline_name"] = "srcc_main"
    out["baseline_equivalence"] = ""
    out["delta_risk_per_row"] = out["delta_risk"]
    out["delta_coverage_per_row"] = out["delta_coverage"]
    return out


def no_buffer_rows(all_rows: pd.DataFrame) -> pd.DataFrame:
    subset = all_rows[np.isclose(all_rows["gamma"].astype(float), 1.0)].copy()
    subset["baseline_name"] = "no_buffer_proposal"
    subset["baseline_equivalence"] = "gamma_1_srcc_row"
    subset["delta_risk_per_row"] = subset["delta_risk"]
    subset["delta_coverage_per_row"] = subset["delta_coverage"]
    return subset


def build_score_rows(run_groups: Iterable[tuple[Path, pd.DataFrame]], num_thresholds: int) -> pd.DataFrame:
    rows = []
    for run_dir, existing in run_groups:
        arrays = load_run_arrays(run_dir)
        meta = metadata_for_run(run_dir, existing)
        deltas = base_deltas(existing)
        prop_errors = correctness_errors(arrays["logits_prop"], arrays["labels_prop"])
        cert_errors = correctness_errors(arrays["logits_cert"], arrays["labels_cert"])
        test_errors = correctness_errors(arrays["logits_test"], arrays["labels_test"])
        all_scores = {
            split: risk_scores(arrays[f"logits_{split}"], SCORES)
            for split in ["prop", "cert", "test"]
        }
        for alpha in sorted(existing["alpha"].astype(float).unique()):
            for score_name in SCORES:
                results = run_certification_for_alpha(
                    prop_scores={score_name: all_scores["prop"][score_name]},
                    prop_errors=prop_errors,
                    cert_scores={score_name: all_scores["cert"][score_name]},
                    cert_errors=cert_errors,
                    test_scores={score_name: all_scores["test"][score_name]},
                    test_errors=test_errors,
                    alpha=float(alpha),
                    delta_total=float(deltas["delta_total"]),
                    delta_risk=float(deltas["delta_risk"]),
                    delta_coverage=float(deltas["delta_coverage"]),
                    delta_allocation=str(deltas["delta_allocation"]),
                    gammas=[0.5, 0.7, 1.0],
                    num_thresholds=num_thresholds,
                    bonferroni_over_gammas=True,
                )
                rows.append({**meta, **best_result_from_records([r.to_dict() for r in results])})
    return pd.DataFrame(rows)


def aggregate_certification(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["certified"] = bool_series(df["certified"])
    grouped = []
    for key, g in df.groupby(group_cols, dropna=False):
        cert = g[g["certified"]]
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "seed_count": int(g["seed"].nunique()),
                "row_count": int(len(g)),
                "certified_seed_count": int(cert["seed"].nunique()),
                "mean_certified_coverage_all_seeds": float(g["certified_coverage_at_alpha"].mean()),
                "mean_certified_coverage_certified_seeds": float(cert["certified_coverage_at_alpha"].mean()) if len(cert) else math.nan,
                "mean_cert_risk_ucb_certified_rows": float(cert["cert_risk_ucb"].mean()) if len(cert) else math.nan,
                "mean_test_risk_certified_rows": float(cert["test_risk"].mean()) if len(cert) else math.nan,
            }
        )
        grouped.append(row)
    return pd.DataFrame(grouped)


def failure_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    failed = all_rows[~bool_series(all_rows["certified"])].copy()
    if failed.empty:
        return pd.DataFrame(columns=["dataset", "regime", "alpha", "reason", "count"])
    group_cols = ["dataset", "regime", "noise_type", "noise_rate", "alpha", "reason"]
    rows = []
    for key, g in failed.groupby(group_cols, dropna=False):
        best = best_row(g).to_dict()
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "count": int(len(g)),
                "seed_count": int(g["seed"].nunique()),
                "best_failed_score_name": best.get("score_name"),
                "best_failed_gamma": best.get("gamma"),
                "best_failed_cert_risk_ucb": best.get("cert_risk_ucb"),
                "best_failed_cert_coverage_lcb": best.get("cert_coverage_lcb"),
                "best_failed_test_risk": best.get("test_risk"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BASELINE_COLUMNS:
        if col not in out.columns:
            out[col] = math.nan
    out["certified"] = bool_series(out["certified"])
    bad_cov = out["certified_coverage_at_alpha"].where(out["certified"], 0.0)
    out["certified_coverage_at_alpha"] = bad_cov
    return out[BASELINE_COLUMNS + [c for c in out.columns if c not in BASELINE_COLUMNS]]


def build_reports(run_root: Path, out_dir: Path, num_thresholds: int = 200) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_existing_rows(run_root)
    run_groups = [(Path(run_dir), g.copy()) for run_dir, g in all_rows.groupby("run_dir", dropna=False)]

    baseline_parts = [srcc_main_rows(all_rows), no_buffer_rows(all_rows)]
    derived_rows = []
    for run_dir, existing in run_groups:
        arrays = load_run_arrays(run_dir)
        meta = metadata_for_run(run_dir, existing)
        deltas = base_deltas(existing)
        for alpha in sorted(existing["alpha"].astype(float).unique()):
            gamma_one = existing[(existing["alpha"].astype(float) == alpha) & np.isclose(existing["gamma"].astype(float), 1.0)]
            gamma_one_row = gamma_one.iloc[0] if len(gamma_one) else None
            derived_rows.append(full_coverage_row(meta, arrays, float(alpha), deltas))
            derived_rows.append(naive_row(meta, arrays, float(alpha), deltas, gamma_one_row, num_thresholds))
            derived_rows.append(ltt_bonferroni_row(meta, arrays, float(alpha), deltas, num_thresholds))
    baseline_parts.append(pd.DataFrame(derived_rows))
    clean_reference = srcc_main_rows(all_rows[all_rows["regime"] == "clean"].copy())
    if not clean_reference.empty:
        clean_reference["baseline_name"] = "clean_trained_upper_bound"
        baseline_parts.append(clean_reference)

    baseline_df = normalize_baseline_columns(pd.concat(baseline_parts, ignore_index=True, sort=False))
    score_rows = build_score_rows(run_groups, num_thresholds=num_thresholds)
    score_summary = aggregate_certification(score_rows, ["dataset", "regime", "noise_type", "noise_rate", "alpha", "score_name"])
    gamma_summary = aggregate_certification(all_rows, ["dataset", "regime", "noise_type", "noise_rate", "alpha", "gamma"])
    gamma_summary["gamma_label"] = np.where(np.isclose(gamma_summary["gamma"].astype(float), 1.0), "no_buffer", "buffered")
    clean_df = baseline_df[baseline_df["baseline_name"] == "clean_trained_upper_bound"].copy()
    failures = failure_summary(all_rows)
    main_table = aggregate_certification(
        baseline_df,
        ["dataset", "regime", "noise_type", "noise_rate", "alpha", "baseline_name"],
    )

    outputs = {
        "baseline": out_dir / "srcc_baseline_comparison.csv",
        "score": out_dir / "srcc_score_ablation.csv",
        "gamma": out_dir / "srcc_gamma_ablation.csv",
        "clean": out_dir / "srcc_clean_upper_bound.csv",
        "failure": out_dir / "srcc_failure_mode_summary.csv",
        "main": out_dir / "srcc_main_journal_table.csv",
    }
    baseline_df.to_csv(outputs["baseline"], index=False)
    score_summary.to_csv(outputs["score"], index=False)
    gamma_summary.to_csv(outputs["gamma"], index=False)
    clean_df.to_csv(outputs["clean"], index=False)
    failures.to_csv(outputs["failure"], index=False)
    main_table.to_csv(outputs["main"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SRCC baseline and ablation reports from saved CIFAR runs.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--num-thresholds", type=int, default=200)
    args = parser.parse_args()
    outputs = build_reports(args.run_root, args.out_dir, args.num_thresholds)
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
