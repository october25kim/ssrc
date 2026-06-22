#!/usr/bin/env python3
"""Create a dependency-free SVG figure from CIFAR-10 result CSV files.

Expected CSV columns:
    method,prior,seed,accuracy

Example:
    python3 scripts/plot_cifar10_results.py \
        --input results/cifar10_results.csv \
        --output figures/cifar10_accuracy.svg
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


COLORS = {
    "clean": "#4C78A8",
    "corrupted": "#F58518",
    "decontaminated": "#54A24B",
    "observed": "#F58518",
    "uplift": "#54A24B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with method,prior,seed,accuracy columns")
    parser.add_argument("--output", default="figures/cifar10_accuracy.svg")
    parser.add_argument("--title", default="CIFAR-10 Prior Decontamination")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    summary = summarize(rows)
    svg = render_grouped_bar_svg(summary, title=args.title)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"wrote {output}")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "prior", "seed", "accuracy"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "method": row["method"].strip(),
                    "prior": row["prior"].strip(),
                    "seed": row["seed"].strip(),
                    "accuracy": float(row["accuracy"]),
                }
            )
    if not rows:
        raise ValueError("input CSV contains no result rows")
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["prior"])].append(row["accuracy"])

    summary = {}
    for key, values in grouped.items():
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            stderr = math.sqrt(var) / math.sqrt(len(values))
        else:
            stderr = 0.0
        summary[key] = {"mean": mean, "stderr": stderr, "n": float(len(values))}
    return summary


def render_grouped_bar_svg(summary: dict[tuple[str, str], dict[str, float]], title: str) -> str:
    methods = sorted({method for method, _ in summary})
    priors = sorted({prior for _, prior in summary})
    width, height = 920, 520
    left, right, top, bottom = 92, 32, 64, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(1.0, max(item["mean"] + item["stderr"] for item in summary.values()) * 1.08)

    parts = [
        svg_header(width, height),
        f'<text x="{width / 2}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]

    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_h - value / ymax * plot_h
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

    group_w = plot_w / max(1, len(methods))
    bar_gap = 8
    bar_w = min(48, (group_w - 28) / max(1, len(priors)) - bar_gap)

    for i, method in enumerate(methods):
        group_x = left + i * group_w
        center = group_x + group_w / 2
        parts.append(f'<text x="{center:.1f}" y="{height - 42}" text-anchor="middle" class="label">{escape(method)}</text>')
        for j, prior in enumerate(priors):
            item = summary.get((method, prior))
            if item is None:
                continue
            x = center - (len(priors) * (bar_w + bar_gap) - bar_gap) / 2 + j * (bar_w + bar_gap)
            bar_h = item["mean"] / ymax * plot_h
            y = top + plot_h - bar_h
            color = COLORS.get(prior.lower(), "#B279A2")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            if item["stderr"] > 0:
                err = item["stderr"] / ymax * plot_h
                cx = x + bar_w / 2
                parts.append(f'<line x1="{cx:.1f}" y1="{y - err:.1f}" x2="{cx:.1f}" y2="{y + err:.1f}" class="err"/>')
                parts.append(f'<line x1="{cx - 6:.1f}" y1="{y - err:.1f}" x2="{cx + 6:.1f}" y2="{y - err:.1f}" class="err"/>')
                parts.append(f'<line x1="{cx - 6:.1f}" y1="{y + err:.1f}" x2="{cx + 6:.1f}" y2="{y + err:.1f}" class="err"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{item["mean"]:.3f}</text>')

    parts.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">Accuracy</text>')
    parts.extend(render_legend(priors, left, height - 24))
    parts.append("</svg>")
    return "\n".join(parts)


def render_legend(priors: list[str], x: int, y: int) -> list[str]:
    parts = []
    cursor = x
    for prior in priors:
        color = COLORS.get(prior.lower(), "#B279A2")
        parts.append(f'<rect x="{cursor}" y="{y - 11}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text x="{cursor + 18}" y="{y}" class="legend">{escape(prior)}</text>')
        cursor += 112
    return parts


def svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 22px sans-serif; fill: #111827; }}
  .axis {{ stroke: #111827; stroke-width: 1.2; }}
  .grid {{ stroke: #E5E7EB; stroke-width: 1; }}
  .tick {{ font: 12px sans-serif; fill: #4B5563; }}
  .label {{ font: 13px sans-serif; fill: #111827; }}
  .legend {{ font: 13px sans-serif; fill: #111827; }}
  .value {{ font: 11px sans-serif; fill: #111827; }}
  .err {{ stroke: #111827; stroke-width: 1.2; }}
</style>
<rect width="100%" height="100%" fill="#FFFFFF"/>'''


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
