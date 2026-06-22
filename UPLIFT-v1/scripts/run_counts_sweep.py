#!/usr/bin/env python3
"""Run small counts-only UPLIFT-U sweeps and write CSV/SVG figures.

This script is intentionally dependency-free. It produces draft figures for the
ICDM v1 counts-only evidence section without using images, features, logits, or
sample-level losses.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.federated.toy import run_federated_toy


BASE_CONFIG = {
    "num_clients": 10,
    "num_classes": 10,
    "samples_per_client": 1000,
    "alpha": 2.0,
    "rho": 1.0,
    "corruption": {"gamma_min": 0.2, "gamma_max": 0.2, "routing": "head"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(run_axis("K", [2, 5, 10, 20, 40], args.seeds))
    rows.extend(run_axis("alpha", [0.5, 1.0, 2.0, 5.0, 10.0], args.seeds))
    rows.extend(run_axis("gamma", [0.0, 0.1, 0.2, 0.3, 0.4], args.seeds))

    csv_path = output_dir / "counts_sweep.csv"
    write_csv(csv_path, rows)

    svg_path = output_dir / "fig_counts_sweep.svg"
    write_svg(svg_path, rows)

    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")


def run_axis(axis: str, values: Iterable[float], num_seeds: int) -> list[dict]:
    axis_rows: list[dict] = []
    for value in values:
        metrics = []
        for seed_offset in range(num_seeds):
            config = clone_config(BASE_CONFIG)
            config["seed"] = 1000 + seed_offset
            if axis == "K":
                config["num_clients"] = int(value)
            elif axis == "alpha":
                config["alpha"] = float(value)
            elif axis == "gamma":
                config["corruption"]["gamma_min"] = float(value)
                config["corruption"]["gamma_max"] = float(value)
            else:
                raise ValueError(f"unknown axis: {axis}")
            metrics.append(run_federated_toy(config))
        axis_rows.append(
            {
                "axis": axis,
                "value": value,
                "observed_l1": mean(row["mean_observed_l1"] for row in metrics),
                "uniform_r_l1": mean(row["mean_uniform_r_l1"] for row in metrics),
                "uplift_l1": mean(row["mean_uplift_l1"] for row in metrics),
                "delta_recovery": mean(row["delta_recovery"] for row in metrics),
            }
        )
    return axis_rows


def clone_config(config: dict) -> dict:
    return {key: (value.copy() if isinstance(value, dict) else value) for key, value in config.items()}


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["axis", "value", "observed_l1", "uniform_r_l1", "uplift_l1", "delta_recovery"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict]) -> None:
    width, height = 1080, 360
    margin = 52
    panel_w = width / 3
    max_y = max(max(row["observed_l1"], row["uniform_r_l1"], row["uplift_l1"]) for row in rows) * 1.08
    colors = {"observed_l1": "#4b5563", "uniform_r_l1": "#d97706", "uplift_l1": "#2563eb"}
    labels = {"observed_l1": "Observed", "uniform_r_l1": "Uniform-r", "uplift_l1": "UPLIFT-U"}
    axis_titles = {"K": "Clients K", "alpha": "Heterogeneity alpha", "gamma": "Unknown mass gamma"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="Arial" font-size="18" font-weight="700">Counts-only prior recovery sweep</text>',
        '<text x="24" y="50" font-family="Arial" font-size="12" fill="#4b5563">Metric: mean L1 error between estimated and clean client priors. Lower is better.</text>',
    ]

    for panel_idx, axis in enumerate(["K", "alpha", "gamma"]):
        subset = [row for row in rows if row["axis"] == axis]
        x0 = panel_idx * panel_w + margin
        y0 = 76
        plot_w = panel_w - 86
        plot_h = 220
        parts.append(f'<text x="{x0}" y="70" font-family="Arial" font-size="14" font-weight="700">{axis_titles[axis]}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#111827" stroke-width="1"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#111827" stroke-width="1"/>')
        for frac in [0.0, 0.5, 1.0]:
            y = y0 + plot_h - frac * plot_h
            label = max_y * frac
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + plot_w}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
            parts.append(f'<text x="{x0 - 8}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#6b7280">{label:.2f}</text>')
        values = [float(row["value"]) for row in subset]
        min_x, max_x = min(values), max(values)
        for metric in ["observed_l1", "uniform_r_l1", "uplift_l1"]:
            points = []
            for row in subset:
                x = scale(float(row["value"]), min_x, max_x, x0, x0 + plot_w)
                y = y0 + plot_h - (row[metric] / max_y) * plot_h
                points.append((x, y))
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{point_text}" fill="none" stroke="{colors[metric]}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[metric]}"/>')
        for row in subset:
            x = scale(float(row["value"]), min_x, max_x, x0, x0 + plot_w)
            parts.append(f'<text x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">{format_value(row["value"])}</text>')

    legend_x = 774
    for idx, metric in enumerate(["observed_l1", "uniform_r_l1", "uplift_l1"]):
        x = legend_x + idx * 96
        parts.append(f'<line x1="{x}" y1="28" x2="{x + 24}" y2="28" stroke="{colors[metric]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 30}" y="32" font-family="Arial" font-size="12" fill="#111827">{labels[metric]}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def scale(value: float, min_value: float, max_value: float, left: float, right: float) -> float:
    if max_value == min_value:
        return (left + right) / 2
    return left + (value - min_value) / (max_value - min_value) * (right - left)


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


if __name__ == "__main__":
    main()
