from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .certify import best_certified_result, run_certification_for_alpha, threshold_direction, zero_error_min_n
from .metrics import aurc, correctness_auroc, coverage_at_risk
from .scores import correctness_errors, risk_scores
from .utils import ensure_dir, load_json, save_json


def load_logits_and_labels(run_dir: Path):
    data = {}
    for split in ["prop", "cert", "test"]:
        data[f"logits_{split}"] = np.load(run_dir / f"logits_{split}.npy")
        data[f"labels_{split}"] = np.load(run_dir / f"labels_{split}.npy")
    return data


def score_diagnostics(score_dict: Dict[str, np.ndarray], errors: np.ndarray, alpha_values: List[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, scores in score_dict.items():
        d = {
            "threshold_direction": threshold_direction(name),
            "aurc": aurc(scores, errors, threshold_direction(name)),
            "correctness_auroc": correctness_auroc(scores, errors, threshold_direction(name)),
        }
        for alpha in alpha_values:
            d[f"oracle_coverage_at_risk_{alpha}"] = coverage_at_risk(scores, errors, alpha, threshold_direction(name))
        out[name] = d
    return out


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(args.out_dir or run_dir)
    data = load_logits_and_labels(run_dir)

    prop_errors = correctness_errors(data["logits_prop"], data["labels_prop"])
    cert_errors = correctness_errors(data["logits_cert"], data["labels_cert"])
    test_errors = correctness_errors(data["logits_test"], data["labels_test"])

    prop_scores = risk_scores(data["logits_prop"], args.scores)
    cert_scores = risk_scores(data["logits_cert"], args.scores)
    test_scores = risk_scores(data["logits_test"], args.scores)

    all_results = []
    for alpha in args.alpha:
        results = run_certification_for_alpha(
            prop_scores=prop_scores,
            prop_errors=prop_errors,
            cert_scores=cert_scores,
            cert_errors=cert_errors,
            test_scores=test_scores,
            test_errors=test_errors,
            alpha=alpha,
            delta=args.delta,
            delta_total=args.delta_total,
            delta_risk=args.delta_risk,
            delta_coverage=args.delta_coverage,
            delta_allocation=args.delta_allocation,
            gammas=args.gammas,
            num_thresholds=args.num_thresholds,
            min_prop_accept=args.min_prop_accept,
            bonferroni_over_gammas=not args.no_bonferroni_over_gammas,
        )
        all_results.extend(results)

    metadata = load_json(run_dir / "metadata.json") if (run_dir / "metadata.json").exists() else {}
    row_metadata = {
        "dataset": metadata.get("dataset"),
        "noise_type": metadata.get("noise_type"),
        "noise_rate": metadata.get("noise_rate"),
        "seed": metadata.get("seed"),
        "model": metadata.get("model"),
        "epochs": metadata.get("epochs"),
    }
    rows = [{**row_metadata, **r.to_dict()} for r in all_results]
    df = pd.DataFrame(rows)

    diagnostics = {
        "prop": score_diagnostics(prop_scores, prop_errors, list(args.alpha)),
        "cert": score_diagnostics(cert_scores, cert_errors, list(args.alpha)),
        "test": score_diagnostics(test_scores, test_errors, list(args.alpha)),
    }

    feasibility = []
    if args.delta is not None and args.delta_allocation is None:
        base_delta_risk = args.delta if args.delta_risk is None else args.delta_risk
        base_delta_coverage = args.delta if args.delta_coverage is None else args.delta_coverage
        delta_total = args.delta if args.delta_total is None else args.delta_total
        delta_allocation = "risk_only_legacy"
    else:
        delta_total = 0.05 if args.delta_total is None else args.delta_total
        base_delta_risk = delta_total / 2 if args.delta_risk is None else args.delta_risk
        base_delta_coverage = delta_total / 2 if args.delta_coverage is None else args.delta_coverage
        delta_allocation = args.delta_allocation or "joint_split"
    row_divisor = len(args.gammas) if (not args.no_bonferroni_over_gammas and len(args.gammas) > 1) else 1
    row_delta_risk = base_delta_risk / row_divisor
    row_delta_coverage = base_delta_coverage / row_divisor
    for alpha in args.alpha:
        feasibility.append(
            {
                "alpha": alpha,
                "delta_total": delta_total,
                "delta_risk_per_row": row_delta_risk,
                "delta_coverage_per_row": row_delta_coverage,
                "delta_allocation": delta_allocation,
                "zero_error_min_cert_n": zero_error_min_n(alpha, row_delta_risk),
            }
        )

    report = {
        "run_dir": str(run_dir),
        "alpha": list(args.alpha),
        "delta": args.delta,
        "delta_total": delta_total,
        "delta_risk": base_delta_risk,
        "delta_coverage": base_delta_coverage,
        "delta_allocation": delta_allocation,
        "gammas": list(args.gammas),
        "scores": list(args.scores),
        "num_thresholds": args.num_thresholds,
        "bonferroni_over_gammas": not args.no_bonferroni_over_gammas,
        "feasibility": feasibility,
        "results": rows,
        "diagnostics": diagnostics,
    }

    if metadata:
        report["metadata"] = metadata

    out_json = out_dir / args.output_json
    out_csv = out_dir / args.output_csv
    save_json(report, out_json)
    df.to_csv(out_csv, index=False)

    display_cols = [
        "dataset",
        "noise_type",
        "noise_rate",
        "seed",
        "model",
        "epochs",
        "alpha",
        "gamma",
        "certified",
        "score_name",
        "threshold",
        "threshold_direction",
        "prop_coverage",
        "prop_risk",
        "cert_n",
        "cert_k",
        "cert_risk_ucb",
        "cert_coverage_lcb",
        "certified_coverage_at_alpha",
        "test_coverage",
        "test_risk",
        "certificate_scope",
        "reason",
    ]
    print("\nCertification results")
    print(df[display_cols].to_string(index=False))

    for alpha in args.alpha:
        subset = [r for r in all_results if r.alpha == alpha]
        best = best_certified_result(subset)
        if best is None:
            print(f"\nalpha={alpha}: no certified result")
        else:
            print(
                f"\nalpha={alpha}: best certified row gamma={best.gamma}, score={best.score_name}, "
                f"certified_cov_at_alpha={best.certified_coverage_at_alpha:.4f}, "
                f"cert_cov_lcb={best.cert_coverage_lcb:.4f}, test_cov={best.test_coverage:.4f}, "
                f"test_risk={best.test_risk:.4f}, risk_ucb={best.cert_risk_ucb:.4f}"
            )

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run risk-buffered accepted-risk certification.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--delta", type=float, default=None, help="Legacy risk-only delta. Prefer --delta-total for joint certificates.")
    p.add_argument("--delta-total", type=float, default=None)
    p.add_argument("--delta-risk", type=float, default=None)
    p.add_argument("--delta-coverage", type=float, default=None)
    p.add_argument("--delta-allocation", type=str, choices=["joint_split", "risk_only", "marginal"], default=None)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    p.add_argument("--scores", type=str, nargs="+", default=["msp", "entropy", "margin", "energy"])
    p.add_argument("--num-thresholds", type=int, default=200)
    p.add_argument("--min-prop-accept", type=int, default=1)
    p.add_argument("--no-bonferroni-over-gammas", action="store_true")
    p.add_argument("--output-json", type=str, default="certification_report.json")
    p.add_argument("--output-csv", type=str, default="certification_results.csv")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
