"""Render publication-safe Season 0 figures without changing frozen estimators.

The primary visualization module is frozen with the analysis protocol.  This
module only changes display scales and label placement for two figures whose
perfectly separated Bradley--Terry fit would otherwise create unreadable axes.
Exact estimates remain printed and no value used by the analysis is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .season0_visualization import (
    BLUE,
    GOLD,
    GREEN,
    GRID,
    INK,
    PAPER,
    VisualizationError,
    _e,
    _header,
    _load,
    _svg_start,
    _write,
)


def _rated_rows(analysis: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = analysis.get("model_leaderboard")
    if not isinstance(rows, list):
        raise VisualizationError("analysis has no model leaderboard")
    rated = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("rating") is not None
        and row.get("rating_lower") is not None
        and row.get("rating_upper") is not None
    ]
    if not rated:
        raise VisualizationError("model leaderboard has no fitted ratings")
    return rated


def _nonseparated_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return rows with a non-degenerate fitted interval for display scaling."""

    stable = [row for row in rows if float(row["rating_upper"]) - float(row["rating_lower"]) >= 1.0]
    return stable or list(rows)


def render_model_forest(analysis: Mapping[str, Any], path: Path) -> None:
    """Render a forest plot with separated fits clipped, labelled, and disclosed."""

    rated = _rated_rows(analysis)
    stable = _nonseparated_rows(rated)
    low = math.floor((min(float(row["rating_lower"]) for row in stable) - 25) / 50) * 50
    high = math.ceil((max(float(row["rating_upper"]) for row in stable) + 25) / 50) * 50
    high = high if high > low else low + 100
    width = 1600
    row_height = 66
    height = 340 + row_height * len(rated)
    plot_left, plot_right = 550, 1370

    def x(value: float) -> float:
        bounded = min(high, max(low, value))
        return plot_left + (bounded - low) / (high - low) * (plot_right - plot_left)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 model ranking",
        "Bradley-Terry ratings with 95 percent confidence intervals; "
        "separated fits are clipped for display.",
    )
    _header(
        lines,
        "Model Arena",
        "Bradley–Terry preference · 95% CI · Epicure enabled for every contestant",
        "FlavourBench · Season 0",
    )
    axis_y = 236
    tick_step = 250
    first_tick = math.ceil(low / tick_step) * tick_step
    for tick in range(first_tick, int(high) + 1, tick_step):
        tick_x = x(float(tick))
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="{axis_y}" x2="{tick_x:.1f}" '
                f'y2="{height - 94}" stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{tick_x:.1f}" y="{axis_y - 12}" class="axis" '
                f'text-anchor="middle">{tick}</text>',
            ]
        )
    if low <= 1000 <= high:
        lines.append(
            f'<line x1="{x(1000):.1f}" y1="{axis_y}" x2="{x(1000):.1f}" '
            f'y2="{height - 94}" stroke="{INK}" stroke-width="1.5" opacity=".42"/>'
        )

    for index, row in enumerate(rated):
        y_value = 278 + index * row_height
        if index % 2 == 0:
            lines.append(
                f'<rect x="54" y="{y_value - 31}" width="1492" height="58" rx="10" fill="#F4F2EC"/>'
            )
        provider = str(row.get("provider") or "")
        badge_color = GREEN if provider == "bedrock" else BLUE
        badge_label = "AWS" if provider == "bedrock" else "OR"
        rating = float(row["rating"])
        lower = float(row["rating_lower"])
        upper = float(row["rating_upper"])
        clipped_left = upper < low
        clipped_right = lower > high
        value_label = f"{rating:.0f}"
        if clipped_left or clipped_right:
            value_label += " sep."

        lines.extend(
            [
                f'<text x="78" y="{y_value + 6}" class="value">{index + 1:02d}</text>',
                f'<text x="126" y="{y_value - 3}" class="label">{_e(row["display_name"])}</text>',
                f'<text x="126" y="{y_value + 19}" class="small">'
                f"n={int(row.get('comparisons') or 0)} · "
                f"invalid {100 * float(row.get('invalid_response_rate') or 0):.1f}%</text>",
                f'<rect x="438" y="{y_value - 19}" width="82" height="24" rx="12" '
                f'fill="{badge_color}"/>',
                f'<text x="479" y="{y_value - 2}" class="badge" fill="{PAPER}" '
                f'text-anchor="middle">{badge_label}</text>',
            ]
        )
        if clipped_left:
            lines.append(
                f'<polygon points="{plot_left},{y_value} {plot_left + 15},{y_value - 10} '
                f'{plot_left + 15},{y_value + 10}" fill="{BLUE}" stroke="{PAPER}" '
                'stroke-width="3"/>'
            )
        elif clipped_right:
            lines.append(
                f'<polygon points="{plot_right},{y_value} {plot_right - 15},{y_value - 10} '
                f'{plot_right - 15},{y_value + 10}" fill="{BLUE}" stroke="{PAPER}" '
                'stroke-width="3"/>'
            )
        else:
            lines.extend(
                [
                    f'<line x1="{x(lower):.1f}" y1="{y_value}" x2="{x(upper):.1f}" '
                    f'y2="{y_value}" stroke="{INK}" stroke-width="4" '
                    'stroke-linecap="round" opacity=".55"/>',
                    f'<circle cx="{x(rating):.1f}" cy="{y_value}" r="9" fill="{BLUE}" '
                    f'stroke="{PAPER}" stroke-width="4"/>',
                ]
            )
        lines.append(f'<text x="1400" y="{y_value + 6}" class="value">{value_label}</text>')

    lines.extend(
        [
            f'<line x1="76" y1="{height - 74}" x2="1524" y2="{height - 74}" stroke="{GRID}"/>',
            f'<text x="76" y="{height - 47}" class="small">'
            "Automated-judge cohort; all ranks are provisional. "
            "Ties are half-wins; both-bad is excluded. "
            "Out-of-range separated fits are clipped at the boundary; "
            "exact ratings remain printed.</text>",
        ]
    )
    _write(path, lines)


def _boxes_overlap(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    return not (
        first[2] + 5 < second[0]
        or second[2] + 5 < first[0]
        or first[3] + 4 < second[1]
        or second[3] + 4 < first[1]
    )


def _place_label(
    point_x: float,
    point_y: float,
    label: str,
    occupied: list[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, str, tuple[float, float, float, float]]:
    """Place a deterministic compact label while avoiding prior labels."""

    left, right, top, bottom = bounds
    width = max(58.0, 7.15 * len(label))
    candidates = (
        (16.0, -14.0, "start"),
        (16.0, 27.0, "start"),
        (-16.0, -14.0, "end"),
        (-16.0, 27.0, "end"),
        (16.0, -34.0, "start"),
        (-16.0, -34.0, "end"),
        (16.0, 47.0, "start"),
        (-16.0, 47.0, "end"),
    )
    fallback: tuple[float, float, str, tuple[float, float, float, float]] | None = None
    for dx, dy, anchor in candidates:
        label_x = point_x + dx
        label_y = point_y + dy
        box_left = label_x if anchor == "start" else label_x - width
        box = (box_left, label_y - 14, box_left + width, label_y + 4)
        placement = (label_x, label_y, anchor, box)
        if fallback is None:
            fallback = placement
        if box[0] < left or box[2] > right or box[1] < top or box[3] > bottom:
            continue
        if any(_boxes_overlap(box, previous) for previous in occupied):
            continue
        occupied.append(box)
        return placement
    assert fallback is not None
    occupied.append(fallback[3])
    return fallback


def render_quality_cost(analysis: Mapping[str, Any], path: Path) -> None:
    """Render quality versus cost with a readable robust y scale and collision-aware labels."""

    rows = _rated_rows(analysis)
    points = [row for row in rows if float(row.get("mean_arm_cost_usd") or 0) > 0]
    if not points:
        raise VisualizationError("analysis has no quality-cost points")
    stable = _nonseparated_rows(points)
    costs = [float(row["mean_arm_cost_usd"]) for row in points]
    log_low = math.floor(math.log10(min(costs)) * 2) / 2
    log_high = math.ceil(math.log10(max(costs)) * 2) / 2
    rating_low = math.floor((min(float(row["rating"]) for row in stable) - 50) / 100) * 100
    rating_high = math.ceil((max(float(row["rating"]) for row in stable) + 50) / 100) * 100
    width, height = 1600, 1080
    left, right, top, bottom = 210, 1470, 245, 880

    def x(cost: float) -> float:
        return left + (math.log10(cost) - log_low) / (log_high - log_low) * (right - left)

    def y(rating: float) -> float:
        bounded = min(rating_high, max(rating_low, rating))
        return bottom - (bounded - rating_low) / (rating_high - rating_low) * (bottom - top)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 quality cost frontier",
        "Mean response cost versus Bradley-Terry rating; separated fits are clipped for display.",
    )
    _header(
        lines,
        "Quality × cost",
        "Higher preference is better · lower mean arm cost is better · log cost axis",
        "FlavourBench · Season 0",
    )
    for exponent_half in range(round(log_low * 2), round(log_high * 2) + 1):
        exponent = exponent_half / 2
        tick = 10**exponent
        tick_x = x(tick)
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="{top}" x2="{tick_x:.1f}" y2="{bottom}" '
                f'stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{tick_x:.1f}" y="{bottom + 32}" class="axis" '
                f'text-anchor="middle">${tick:.4g}</text>',
            ]
        )
    tick_step = 200
    first_tick = math.ceil(rating_low / tick_step) * tick_step
    for tick in range(first_tick, int(rating_high) + 1, tick_step):
        tick_y = y(float(tick))
        lines.extend(
            [
                f'<line x1="{left}" y1="{tick_y:.1f}" x2="{right}" y2="{tick_y:.1f}" '
                f'stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{left - 22}" y="{tick_y + 5:.1f}" class="axis" '
                f'text-anchor="end">{tick}</text>',
            ]
        )

    frontier: list[Mapping[str, Any]] = []
    best_rating = -math.inf
    for row in sorted(points, key=lambda item: float(item["mean_arm_cost_usd"])):
        if float(row["rating"]) > best_rating:
            frontier.append(row)
            best_rating = float(row["rating"])
    if len(frontier) > 1:
        path_points = " ".join(
            f"{x(float(row['mean_arm_cost_usd'])):.1f},{y(float(row['rating'])):.1f}"
            for row in frontier
        )
        lines.append(
            f'<polyline points="{path_points}" fill="none" stroke="{GOLD}" stroke-width="4" '
            'stroke-dasharray="10 8" opacity=".8"/>'
        )

    occupied: list[tuple[float, float, float, float]] = []
    for row in points:
        point_x = x(float(row["mean_arm_cost_usd"]))
        rating = float(row["rating"])
        point_y = y(rating)
        color = GREEN if row.get("provider") == "bedrock" else BLUE
        separated = float(row["rating_upper"]) - float(row["rating_lower"]) < 1.0
        label = str(row["display_name"])
        if rating < rating_low or rating > rating_high or separated:
            label += f" ({rating:.0f}; sep.)"
        if rating < rating_low:
            lines.append(
                f'<polygon points="{point_x - 9:.1f},{bottom - 14} {point_x + 9:.1f},{bottom - 14} '
                f'{point_x:.1f},{bottom}" fill="{color}" stroke="{PAPER}" stroke-width="3"/>'
            )
        elif rating > rating_high:
            lines.append(
                f'<polygon points="{point_x - 9:.1f},{top + 14} {point_x + 9:.1f},{top + 14} '
                f'{point_x:.1f},{top}" fill="{color}" stroke="{PAPER}" stroke-width="3"/>'
            )
        else:
            lines.append(
                f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="10" fill="{color}" '
                f'stroke="{PAPER}" stroke-width="4"/>'
            )
        label_x, label_y, anchor, _box = _place_label(
            point_x,
            point_y,
            label,
            occupied,
            (left + 4, right - 4, top + 4, bottom - 4),
        )
        lines.extend(
            [
                f'<line x1="{point_x:.1f}" y1="{point_y:.1f}" x2="{label_x:.1f}" '
                f'y2="{label_y - 5:.1f}" stroke="{GRID}" stroke-width="1.5"/>',
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="small" '
                f'text-anchor="{anchor}">{_e(label)}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 76}" class="label" '
            'text-anchor="middle">Mean cost per response arm (USD, log scale)</text>',
            f'<text x="72" y="{(top + bottom) / 2:.1f}" class="label" text-anchor="middle" '
            f'transform="rotate(-90 72 {(top + bottom) / 2:.1f})">Model Arena rating</text>',
            f'<text x="210" y="{height - 64}" class="small">'
            "Dashed gold line marks the observed Pareto frontier. Separated fits are clipped "
            "at the plot boundary and printed exactly; cost means cover all attempted arms.</text>",
        ]
    )
    _write(path, lines)


def render_all(analysis: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "model_forest": output_dir / "season0-model-arena.svg",
        "quality_cost": output_dir / "season0-quality-cost.svg",
    }
    render_model_forest(analysis, paths["model_forest"])
    render_quality_cost(analysis, paths["quality_cost"])
    return {key: str(path) for key, path in paths.items()}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_all(_load(args.analysis), args.output_dir), indent=2))


if __name__ == "__main__":
    run()
