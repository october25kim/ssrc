from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=str, default="runs")
    p.add_argument("--out", type=str, default="runs/aggregate_certification_results.csv")
    args = p.parse_args()

    paths = sorted(Path(args.run_root).glob("*/certification_results.csv"))
    if not paths:
        raise SystemExit(f"No certification_results.csv files found under {args.run_root}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df.insert(0, "run", path.parent.name)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

    cols = ["run", "alpha", "gamma", "certified", "score_name", "cert_coverage_lcb", "test_coverage", "test_risk"]
    print(all_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
