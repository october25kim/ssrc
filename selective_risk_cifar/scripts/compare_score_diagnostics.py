from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Compare AURC/AUROC diagnostics across certification reports.")
    p.add_argument("reports", nargs="+", help="Paths to certification_report.json")
    p.add_argument("--split", choices=["prop", "cert", "test"], default="test")
    args = p.parse_args()

    rows = []
    for rp in args.reports:
        path = Path(rp)
        with path.open() as f:
            rep = json.load(f)
        run = Path(rep.get("run_dir", path.parent)).name
        diag = rep["diagnostics"][args.split]
        for score, vals in diag.items():
            row = {"run": run, "score": score}
            row.update(vals)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
