#!/usr/bin/env python3
"""Plot counts-only UPLIFT-U sweep results with matplotlib."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


AXES = ["K", "alpha", "gamma"]
AXIS_TITLES = {
    "K": "Clients K",
    "alpha": "Heterogeneity alpha",
    "gamma": "Unknown mass gamma",
}
METRICS = [
    ("observed_l1", "Observed", "#4b5563", "o"),
    ("uniform_r_l1", "Uniform-r", "#d97706", "s"),
    ("uplift_l1", "UPLIFT-U", "#2563eb", "^"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/counts_sweep.csv")
    parser.add_argument("--output-prefix", default="results/fig_counts_sweep_matplotlib")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    max_y = max(max(row[key] for key, _, _, _ in METRICS) for row in rows) * 1.08

    for ax, axis in zip(axes, AXES):
        subset = [row for row in rows if row["axis"] == axis]
        x = [row["value"] for row in subset]
        for key, label, color, marker in METRICS:
            y = [row[key] for row in subset]
            ax.plot(x, y, marker=marker, linewidth=2.0, markersize=4.5, color=color, label=label)
        ax.set_title(AXIS_TITLES[axis], fontsize=11, weight="bold")
        ax.set_xlabel(axis)
        ax.set_ylim(0.0, max_y)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean L1 prior recovery error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 1.06))
    fig.suptitle("Counts-only prior decontamination sweep", x=0.02, y=1.07, ha="left", fontsize=13, weight="bold")
    fig.text(0.02, -0.02, "Lower is better. UPLIFT-U uses shared routing prior under the restricted T=I model.", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "axis": row["axis"],
                "value": float(row["value"]),
                "observed_l1": float(row["observed_l1"]),
                "uniform_r_l1": float(row["uniform_r_l1"]),
                "uplift_l1": float(row["uplift_l1"]),
                "delta_recovery": float(row["delta_recovery"]),
            }
            for row in reader
        ]


if __name__ == "__main__":
    main()
