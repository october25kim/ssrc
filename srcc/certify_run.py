from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .certify import best_certified_result, run_certification_for_alpha, zero_error_n_min
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
            "aurc": aurc(scores, errors),
            "correctness_auroc": correctness_auroc(scores, errors),
        }
        for alpha in alpha_values:
            d[f"oracle_coverage_at_risk_{alpha}"] = coverage_at_risk(scores, errors, alpha)
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
            gammas=args.gammas,
            num_thresholds=args.num_thresholds,
            min_prop_accept=args.min_prop_accept,
            bonferroni_over_gammas=not args.no_bonferroni_over_gammas,
        )
        all_results.extend(results)

    rows = [r.to_dict() for r in all_results]
    df = pd.DataFrame(rows)

    diagnostics = {
        "prop": score_diagnostics(prop_scores, prop_errors, list(args.alpha)),
        "cert": score_diagnostics(cert_scores, cert_errors, list(args.alpha)),
        "test": score_diagnostics(test_scores, test_errors, list(args.alpha)),
    }

    feasibility = []
    delta_each = args.delta / len(args.gammas) if not args.no_bonferroni_over_gammas else args.delta
    for alpha in args.alpha:
        feasibility.append(
            {
                "alpha": alpha,
                "delta_effective_per_gamma": delta_each,
                "zero_error_n_min": zero_error_n_min(alpha, delta_each),
            }
        )

    report = {
        "run_dir": str(run_dir),
        "alpha": list(args.alpha),
        "delta": args.delta,
        "gammas": list(args.gammas),
        "scores": list(args.scores),
        "num_thresholds": args.num_thresholds,
        "bonferroni_over_gammas": not args.no_bonferroni_over_gammas,
        "feasibility": feasibility,
        "results": rows,
        "diagnostics": diagnostics,
    }

    if (run_dir / "metadata.json").exists():
        report["metadata"] = load_json(run_dir / "metadata.json")

    out_json = out_dir / args.output_json
    out_csv = out_dir / args.output_csv
    save_json(report, out_json)
    df.to_csv(out_csv, index=False)

    display_cols = [
        "alpha",
        "gamma",
        "certified",
        "score_name",
        "prop_coverage",
        "prop_risk",
        "cert_n",
        "cert_k",
        "cert_risk_ucb",
        "cert_coverage_lcb",
        "test_coverage",
        "test_risk",
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
                f"\nalpha={alpha}: best certified gamma={best.gamma}, score={best.score_name}, "
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
    p.add_argument("--delta", type=float, default=0.05)
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
