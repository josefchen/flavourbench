"""Render deterministic publication and web SVGs from a Season 0 analysis artifact."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

INK = "#17221F"
MUTED = "#65706C"
BG = "#F7F5EF"
PAPER = "#FFFDF8"
GRID = "#D8DDD7"
BLUE = "#355CFF"
GREEN = "#139C6B"
GOLD = "#D59628"
RED = "#CE5A4B"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
DIMENSION_LABELS = {
    "task_completion": "Task completion",
    "constraint_compliance": "Constraint compliance",
    "coherence": "Coherence",
    "sensory_promise": "Sensory promise",
    "cookability": "Cookability",
    "clarity": "Clarity",
    "originality": "Originality",
    "evidence_use": "Evidence use",
    "calibration": "Calibration",
}


class VisualizationError(RuntimeError):
    """The analysis artifact cannot support a truthful visualization."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise VisualizationError("analysis file is not a JSON object")
    claimed = value.get("artifact_sha256")
    actual = sha256_json({key: item for key, item in value.items() if key != "artifact_sha256"})
    if claimed != actual:
        raise VisualizationError("analysis artifact hash mismatch")
    return value


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_e(title)}</title>',
        f'<desc id="desc">{_e(description)}</desc>',
        "<style>"
        ".title{font-family:Arial,sans-serif;font-size:42px;font-weight:700;fill:#17221F}"
        ".subtitle{font-family:Arial,sans-serif;font-size:19px;font-weight:400;fill:#65706C}"
        ".label{font-family:Arial,sans-serif;font-size:18px;font-weight:600;fill:#17221F}"
        ".small{font-family:Arial,sans-serif;font-size:14px;font-weight:400;fill:#65706C}"
        ".axis{font-family:Arial,sans-serif;font-size:14px;font-weight:500;fill:#65706C}"
        ".value{font-family:monospace;font-size:16px;font-weight:700;fill:#17221F}"
        ".badge{font-family:Arial,sans-serif;font-size:11px;font-weight:700;letter-spacing:1.2px}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<rect x="28" y="28" width="{width - 56}" height="{height - 56}" rx="26" '
        f'fill="{PAPER}" stroke="{GRID}"/>',
    ]


def _header(lines: list[str], title: str, subtitle: str, kicker: str) -> None:
    lines.extend(
        [
            f'<text x="76" y="84" class="small" fill="{BLUE}">{_e(kicker.upper())}</text>',
            f'<text x="76" y="132" class="title">{_e(title)}</text>',
            f'<text x="76" y="166" class="subtitle">{_e(subtitle)}</text>',
            f'<line x1="76" y1="190" x2="1524" y2="190" stroke="{GRID}"/>',
        ]
    )


def _write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*lines, "</svg>", ""]), encoding="utf-8")


def render_model_forest(analysis: Mapping[str, Any], path: Path) -> None:
    rows = analysis.get("model_leaderboard")
    if not isinstance(rows, list) or not rows:
        raise VisualizationError("analysis has no model leaderboard")
    rated = [row for row in rows if isinstance(row, Mapping) and row.get("rating") is not None]
    if not rated:
        raise VisualizationError("model leaderboard has no fitted ratings")
    lows = [float(row["rating_lower"]) for row in rated]
    highs = [float(row["rating_upper"]) for row in rated]
    low = math.floor((min(lows) - 25) / 50) * 50
    high = math.ceil((max(highs) + 25) / 50) * 50
    high = high if high > low else low + 100
    width = 1600
    row_height = 66
    height = 300 + row_height * len(rated)
    plot_left, plot_right = 550, 1370

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 model ranking",
        "Bradley-Terry ratings with 95 percent confidence intervals for the automated cohort.",
    )
    _header(
        lines,
        "Model Arena",
        "Bradley–Terry preference · 95% CI · Epicure enabled for every contestant",
        "FlavourBench · Season 0",
    )
    axis_y = 236
    for tick in range(int(low), int(high) + 1, 50):
        tick_x = x(float(tick))
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="{axis_y}" x2="{tick_x:.1f}" '
            f'y2="{height - 94}" stroke="{GRID}" stroke-dasharray="3 7"/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="{axis_y - 12}" class="axis" text-anchor="middle">{tick}</text>'
        )
    if low <= 1000 <= high:
        lines.append(
            f'<line x1="{x(1000):.1f}" y1="{axis_y}" x2="{x(1000):.1f}" '
            f'y2="{height - 94}" stroke="{INK}" stroke-width="1.5" opacity=".42"/>'
        )
    for index, row in enumerate(rated):
        y = 278 + index * row_height
        if index % 2 == 0:
            lines.append(
                f'<rect x="54" y="{y - 31}" width="1492" height="58" rx="10" fill="#F4F2EC"/>'
            )
        provider = str(row.get("provider") or "")
        badge_color = GREEN if provider == "bedrock" else BLUE
        lines.extend(
            [
                f'<text x="78" y="{y + 6}" class="value">{index + 1:02d}</text>',
                f'<text x="126" y="{y - 3}" class="label">{_e(row["display_name"])}</text>',
                f'<text x="126" y="{y + 19}" class="small">n={int(row.get("comparisons") or 0)} · '
                f"invalid {100 * float(row.get('invalid_response_rate') or 0):.1f}%</text>",
                f'<rect x="438" y="{y - 19}" width="82" height="24" rx="12" fill="{badge_color}" opacity=".12"/>',
                f'<text x="479" y="{y - 2}" class="badge" fill="{badge_color}" text-anchor="middle">{_e(provider)}</text>',
                f'<line x1="{x(float(row["rating_lower"])):.1f}" y1="{y}" '
                f'x2="{x(float(row["rating_upper"])):.1f}" y2="{y}" stroke="{INK}" stroke-width="4" '
                'stroke-linecap="round" opacity=".55"/>',
                f'<circle cx="{x(float(row["rating"])):.1f}" cy="{y}" r="9" fill="{BLUE}" stroke="{PAPER}" stroke-width="4"/>',
                f'<text x="1400" y="{y + 6}" class="value">{float(row["rating"]):.0f}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="76" y1="{height - 74}" x2="1524" y2="{height - 74}" stroke="{GRID}"/>',
            f'<text x="76" y="{height - 43}" class="small">Automated judge cohort is reported separately from public and expert-human cohorts. '
            "Ties count as half-wins; both-bad outcomes are excluded from fitting.</text>",
        ]
    )
    _write(path, lines)


def render_uplift_forest(analysis: Mapping[str, Any], path: Path) -> None:
    rows = analysis.get("uplift_leaderboard")
    if not isinstance(rows, list) or not rows:
        raise VisualizationError("analysis has no uplift leaderboard")
    width = 1600
    row_height = 66
    height = 300 + row_height * len(rows)
    plot_left, plot_right = 510, 1280
    values = [
        float(row[key])
        for row in rows
        if isinstance(row, Mapping)
        for key in ("interval_lower", "interval_upper")
    ]
    low = min(0.35, min(values) - 0.03)
    high = max(0.65, max(values) + 0.03)

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 Epicure uplift",
        "Paired Epicure win share with tie-aware confidence intervals by model.",
    )
    _header(
        lines,
        "Epicure Uplift",
        "Same model, same task, Epicure off vs on · paired tie-aware estimate",
        "FlavourBench · Season 0",
    )
    for tick_index in range(7):
        tick = low + (high - low) * tick_index / 6
        tick_x = x(tick)
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="236" x2="{tick_x:.1f}" y2="{height - 94}" '
            f'stroke="{GRID}" stroke-dasharray="3 7"/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="224" class="axis" text-anchor="middle">{tick:.2f}</text>'
        )
    lines.append(
        f'<line x1="{x(0.5):.1f}" y1="236" x2="{x(0.5):.1f}" y2="{height - 94}" '
        f'stroke="{INK}" stroke-width="2" opacity=".55"/>'
    )
    for index, row in enumerate(rows):
        y = 278 + index * row_height
        if index % 2 == 0:
            lines.append(
                f'<rect x="54" y="{y - 31}" width="1492" height="58" rx="10" fill="#F4F2EC"/>'
            )
        wins = int(row.get("epicure_wins") or 0)
        ties = int(row.get("ties") or 0)
        losses = int(row.get("unaided_wins") or 0)
        lines.extend(
            [
                f'<text x="78" y="{y + 6}" class="value">{index + 1:02d}</text>',
                f'<text x="126" y="{y - 3}" class="label">{_e(row["display_name"])}</text>',
                f'<text x="126" y="{y + 19}" class="small">W/T/L {wins}/{ties}/{losses} · n={int(row.get("comparisons") or 0)}</text>',
                f'<line x1="{x(float(row["interval_lower"])):.1f}" y1="{y}" '
                f'x2="{x(float(row["interval_upper"])):.1f}" y2="{y}" stroke="{INK}" stroke-width="4" '
                'stroke-linecap="round" opacity=".55"/>',
                f'<circle cx="{x(float(row["epicure_win_share"])):.1f}" cy="{y}" r="9" fill="{GREEN}" stroke="{PAPER}" stroke-width="4"/>',
                f'<text x="1310" y="{y + 6}" class="value">{float(row["epicure_win_share"]):.3f}</text>',
            ]
        )
    lines.append(
        f'<text x="76" y="{height - 43}" class="small">0.50 is no directional uplift. Positive values favor Epicure-on; '
        "intervals profile the three-outcome multinomial likelihood.</text>"
    )
    _write(path, lines)


def render_dimension_uplift(analysis: Mapping[str, Any], path: Path) -> None:
    rows = analysis.get("panel_uplift_dimensions")
    if not isinstance(rows, list) or not rows:
        raise VisualizationError("analysis has no panel-level dimension uplift")
    analyzable = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("mean_delta") is not None
        and row.get("lower") is not None
        and row.get("upper") is not None
    ]
    if not analyzable:
        raise VisualizationError("dimension uplift has no analyzable intervals")
    maximum = max(
        abs(float(row[key])) for row in analyzable for key in ("lower", "upper", "mean_delta")
    )
    bound = max(0.25, math.ceil(maximum * 10) / 10)
    width = 1600
    row_height = 70
    height = 310 + row_height * len(analyzable)
    plot_left, plot_right = 520, 1300

    def x(value: float) -> float:
        return plot_left + (value + bound) / (2 * bound) * (plot_right - plot_left)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 quality-dimension uplift",
        "Task-clustered Epicure-on minus Epicure-off rubric score differences across the frozen model panel.",
    )
    _header(
        lines,
        "Where Epicure changes answer quality",
        "Automated rubric delta · Epicure on minus off · task-clustered 95% CI",
        "FlavourBench · Season 0",
    )
    for tick_index in range(7):
        tick = -bound + (2 * bound * tick_index / 6)
        tick_x = x(tick)
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="238" x2="{tick_x:.1f}" '
                f'y2="{height - 92}" stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{tick_x:.1f}" y="224" class="axis" text-anchor="middle">{tick:+.2f}</text>',
            ]
        )
    lines.append(
        f'<line x1="{x(0):.1f}" y1="238" x2="{x(0):.1f}" y2="{height - 92}" '
        f'stroke="{INK}" stroke-width="2" opacity=".58"/>'
    )
    for index, row in enumerate(analyzable):
        y = 282 + index * row_height
        if index % 2 == 0:
            lines.append(
                f'<rect x="54" y="{y - 32}" width="1492" height="62" rx="10" fill="#F4F2EC"/>'
            )
        dimension = str(row["dimension"])
        label = DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())
        mean = float(row["mean_delta"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        lines.extend(
            [
                f'<text x="82" y="{y + 6}" class="label">{_e(label)}</text>',
                f'<text x="342" y="{y + 6}" class="small">n={int(row.get("comparisons") or 0)} · '
                f"{int(row.get('task_clusters') or 0)} tasks</text>",
                f'<line x1="{x(lower):.1f}" y1="{y}" x2="{x(upper):.1f}" y2="{y}" '
                f'stroke="{INK}" stroke-width="4" stroke-linecap="round" opacity=".55"/>',
                f'<circle cx="{x(mean):.1f}" cy="{y}" r="9" fill="{GREEN}" '
                f'stroke="{PAPER}" stroke-width="4"/>',
                f'<text x="1332" y="{y + 6}" class="value">{mean:+.3f}</text>',
            ]
        )
    lines.append(
        f'<text x="76" y="{height - 43}" class="small">Positive values favor Epicure-on. '
        "Scores are automated-judge measurements on a 1–5 rubric, not chef sensory trials.</text>"
    )
    _write(path, lines)


def _mix(first: tuple[int, int, int], second: tuple[int, int, int], amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    rgb = tuple(round(a + (b - a) * amount) for a, b in zip(first, second, strict=True))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def render_family_heatmap(analysis: Mapping[str, Any], path: Path) -> None:
    by_family = analysis.get("model_leaderboard_by_family")
    global_rows = analysis.get("model_leaderboard")
    if not isinstance(by_family, Mapping) or not isinstance(global_rows, list):
        raise VisualizationError("analysis has no family rankings")
    models = [str(row["season_model_id"]) for row in global_rows if isinstance(row, Mapping)]
    names = {
        str(row["season_model_id"]): str(row["display_name"])
        for row in global_rows
        if isinstance(row, Mapping)
    }
    ratings: dict[tuple[str, str], float | None] = {}
    for family, rows in by_family.items():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    value = row.get("rating")
                    ratings[(str(row["season_model_id"]), str(family))] = (
                        float(value) if value is not None else None
                    )
    width, height = 1500, 1050
    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 family heatmap",
        "Within-family Bradley-Terry ratings for twelve frontier models and four culinary task families.",
    )
    _header(
        lines,
        "Capability profile",
        "Within-family Bradley–Terry rating · columns are centered independently",
        "FlavourBench · Season 0",
    )
    left, top, cell_w, cell_h = 520, 270, 220, 55
    families = [family for family in FAMILIES if family in by_family]
    for column, family in enumerate(families):
        lines.append(
            f'<text x="{left + column * cell_w + cell_w / 2:.1f}" y="234" class="label" '
            f'text-anchor="middle">{_e(family.title())}</text>'
        )
    for index, model_id in enumerate(models):
        y = top + index * cell_h
        lines.append(f'<text x="88" y="{y + 35}" class="label">{_e(names[model_id])}</text>')
        for column, family in enumerate(families):
            value = ratings.get((model_id, family))
            x = left + column * cell_w
            if value is None:
                fill, label = "#ECEDE9", "—"
            else:
                deviation = max(-160.0, min(160.0, value - 1000)) / 160
                fill = (
                    _mix((255, 253, 248), (53, 92, 255), deviation)
                    if deviation >= 0
                    else _mix((255, 253, 248), (206, 90, 75), -deviation)
                )
                label = f"{value:.0f}"
            lines.extend(
                [
                    f'<rect x="{x + 5}" y="{y + 5}" width="{cell_w - 10}" height="{cell_h - 10}" '
                    f'rx="10" fill="{fill}"/>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + 36}" class="value" '
                    f'text-anchor="middle">{label}</text>',
                ]
            )
    lines.append(
        f'<text x="88" y="{height - 60}" class="small">Blue indicates above-family mean; red below-family mean. '
        "Do not compare absolute color intensity across columns.</text>"
    )
    _write(path, lines)


def render_quality_cost(analysis: Mapping[str, Any], path: Path) -> None:
    rows = analysis.get("model_leaderboard")
    if not isinstance(rows, list):
        raise VisualizationError("analysis has no model leaderboard")
    points = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("rating") is not None
        and float(row.get("mean_arm_cost_usd") or 0) > 0
    ]
    costs = [float(row["mean_arm_cost_usd"]) for row in points]
    ratings = [float(row["rating"]) for row in points]
    log_low = math.floor(math.log10(min(costs)) * 2) / 2
    log_high = math.ceil(math.log10(max(costs)) * 2) / 2
    rating_low = math.floor((min(ratings) - 30) / 50) * 50
    rating_high = math.ceil((max(ratings) + 30) / 50) * 50
    width, height = 1600, 1080
    left, right, top, bottom = 210, 1470, 245, 900

    def x(cost: float) -> float:
        return left + (math.log10(cost) - log_low) / (log_high - log_low) * (right - left)

    def y(rating: float) -> float:
        return bottom - (rating - rating_low) / (rating_high - rating_low) * (bottom - top)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 quality cost frontier",
        "Mean response cost versus Bradley-Terry model-arena rating.",
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
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="{top}" x2="{tick_x:.1f}" y2="{bottom}" '
            f'stroke="{GRID}" stroke-dasharray="3 7"/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="{bottom + 32}" class="axis" text-anchor="middle">${tick:.4g}</text>'
        )
    for tick in range(int(rating_low), int(rating_high) + 1, 50):
        tick_y = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{tick_y:.1f}" x2="{right}" y2="{tick_y:.1f}" '
            f'stroke="{GRID}" stroke-dasharray="3 7"/>'
        )
        lines.append(
            f'<text x="{left - 22}" y="{tick_y + 5:.1f}" class="axis" text-anchor="end">{tick}</text>'
        )
    frontier = []
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
    for index, row in enumerate(points):
        point_x = x(float(row["mean_arm_cost_usd"]))
        point_y = y(float(row["rating"]))
        color = GREEN if row.get("provider") == "bedrock" else BLUE
        anchor = "start" if index % 2 == 0 else "end"
        dx = 14 if anchor == "start" else -14
        lines.extend(
            [
                f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="10" fill="{color}" '
                f'stroke="{PAPER}" stroke-width="4"/>',
                f'<text x="{point_x + dx:.1f}" y="{point_y - 12:.1f}" class="small" '
                f'text-anchor="{anchor}">{_e(row["display_name"])}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 76}" class="label" text-anchor="middle">Mean cost per response arm (USD, log scale)</text>',
            f'<text x="72" y="{(top + bottom) / 2:.1f}" class="label" text-anchor="middle" '
            f'transform="rotate(-90 72 {(top + bottom) / 2:.1f})">Model Arena rating</text>',
            f'<text x="210" y="{height - 64}" class="small">Dashed gold line marks the observed Pareto frontier. '
            "Means cover all attempted arms; unattributed possible-delivery calls retain frozen reservations. Bedrock rates await AWS CUR cross-check.</text>",
        ]
    )
    _write(path, lines)


def render_adoption_uplift(analysis: Mapping[str, Any], path: Path) -> None:
    uplift_rows = analysis.get("uplift_leaderboard")
    operational = analysis.get("operational_metrics")
    if not isinstance(uplift_rows, list) or not isinstance(operational, Mapping):
        raise VisualizationError("analysis has no uplift or operational metrics")
    points = []
    for row in uplift_rows:
        if not isinstance(row, Mapping) or row.get("epicure_win_share") is None:
            continue
        model_id = str(row.get("season_model_id") or "")
        metrics = operational.get(model_id)
        if not isinstance(metrics, Mapping) or metrics.get("epicure_on_tool_use_rate") is None:
            continue
        points.append((row, metrics))
    if not points:
        raise VisualizationError("analysis has no tool-adoption/uplift points")

    interval_values = [
        float(row[key])
        for row, _metrics in points
        for key in ("interval_lower", "interval_upper")
        if row.get(key) is not None
    ]
    low = max(0.0, math.floor((min([0.5, *interval_values]) - 0.03) * 20) / 20)
    high = min(1.0, math.ceil((max([0.5, *interval_values]) + 0.03) * 20) / 20)
    if high <= low:
        low, high = 0.4, 0.6
    width, height = 1600, 1060
    left, right, top, bottom = 210, 1450, 255, 865

    def x(value: float) -> float:
        return left + value * (right - left)

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    lines = _svg_start(
        width,
        height,
        "FlavourBench Season 0 Epicure adoption and uplift",
        "Observed Epicure tool-use rate versus paired automated-judge uplift with confidence intervals.",
    )
    _header(
        lines,
        "Access is not adoption",
        "Observed tool uptake × paired Epicure preference · descriptive, not a mediator effect",
        "FlavourBench · Season 0",
    )
    for tick_index in range(6):
        tick = tick_index / 5
        tick_x = x(tick)
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="{top}" x2="{tick_x:.1f}" y2="{bottom}" '
                f'stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{tick_x:.1f}" y="{bottom + 34}" class="axis" text-anchor="middle">{tick:.0%}</text>',
            ]
        )
    y_tick_count = max(2, round((high - low) / 0.05))
    for tick_index in range(y_tick_count + 1):
        tick = low + (high - low) * tick_index / y_tick_count
        tick_y = y(tick)
        lines.extend(
            [
                f'<line x1="{left}" y1="{tick_y:.1f}" x2="{right}" y2="{tick_y:.1f}" '
                f'stroke="{GRID}" stroke-dasharray="3 7"/>',
                f'<text x="{left - 22}" y="{tick_y + 5:.1f}" class="axis" text-anchor="end">{tick:.2f}</text>',
            ]
        )
    if low <= 0.5 <= high:
        lines.append(
            f'<line x1="{left}" y1="{y(0.5):.1f}" x2="{right}" y2="{y(0.5):.1f}" '
            f'stroke="{INK}" stroke-width="2" opacity=".58"/>'
        )
        lines.append(
            f'<text x="{right - 4}" y="{y(0.5) - 10:.1f}" class="small" text-anchor="end">no directional uplift</text>'
        )

    ordered = sorted(
        points,
        key=lambda item: (
            float(item[1]["epicure_on_tool_use_rate"]),
            float(item[0]["epicure_win_share"]),
            str(item[0]["display_name"]),
        ),
    )
    for index, (row, metrics) in enumerate(ordered):
        uptake = float(metrics["epicure_on_tool_use_rate"])
        estimate = float(row["epicure_win_share"])
        lower = float(row["interval_lower"])
        upper = float(row["interval_upper"])
        point_x, point_y = x(uptake), y(estimate)
        provider = str(metrics.get("provider") or "")
        color = GREEN if provider == "bedrock" else BLUE
        anchor = "end" if uptake > 0.68 else "start"
        dx = -14 if anchor == "end" else 14
        dy = -16 if index % 2 == 0 else 26
        lines.extend(
            [
                f'<line x1="{point_x:.1f}" y1="{y(lower):.1f}" x2="{point_x:.1f}" '
                f'y2="{y(upper):.1f}" stroke="{INK}" stroke-width="4" stroke-linecap="round" opacity=".5"/>',
                f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="10" fill="{color}" '
                f'stroke="{PAPER}" stroke-width="4"/>',
                f'<text x="{point_x + dx:.1f}" y="{point_y + dy:.1f}" class="small" '
                f'text-anchor="{anchor}">{_e(row["display_name"])}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{bottom + 78}" class="label" text-anchor="middle">Epicure-enabled arms with at least one real tool call</text>',
            f'<text x="70" y="{(top + bottom) / 2:.1f}" class="label" text-anchor="middle" '
            f'transform="rotate(-90 70 {(top + bottom) / 2:.1f})">Paired Epicure win share</text>',
            f'<circle cx="1110" cy="{height - 103}" r="7" fill="{GREEN}"/><text x="1125" y="{height - 98}" class="small">Bedrock target route</text>',
            f'<circle cx="1305" cy="{height - 103}" r="7" fill="{BLUE}"/><text x="1320" y="{height - 98}" class="small">OpenRouter target route</text>',
            f'<text x="210" y="{height - 52}" class="small">Vertical bars are paired 95% intervals. Uptake is post-treatment behavior: this plot does not identify the causal effect of invoking a tool.</text>',
        ]
    )
    _write(path, lines)


def render_architecture(path: Path) -> None:
    width, height = 1800, 1160
    lines = _svg_start(
        width,
        height,
        "FlavourBench causal benchmark architecture",
        "Two-track architecture from human-origin tasks through frozen model execution, Epicure, blinded judging, and separate statistical outputs.",
    )
    _header(
        lines,
        "Two estimands, one evidence chain",
        "Every arrow is hash-bound · every paid call is journaled · cohorts never silently pool",
        "FlavourBench · Benchmark architecture",
    )
    lines.insert(
        2,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#65706C"/></marker></defs>',
    )

    def box(x: int, y: int, w: int, h: int, title: str, body: Sequence[str], color: str) -> None:
        pale = {
            INK: "#EEF0ED",
            BLUE: "#EEF1FF",
            GREEN: "#EAF7F1",
            GOLD: "#FFF4DE",
        }[color]
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{pale}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{x + 26}" y="{y + 40}" class="label" fill="{color}">{_e(title)}</text>',
            ]
        )
        for index, item in enumerate(body):
            lines.append(
                f'<text x="{x + 26}" y="{y + 70 + index * 23}" class="small">{_e(item)}</text>'
            )

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        lines.append(
            f'<path d="M{x1},{y1} C{x1},{(y1 + y2) / 2:.1f} {x2},{(y1 + y2) / 2:.1f} {x2},{y2}" '
            f'fill="none" stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )

    box(
        610,
        220,
        580,
        112,
        "Human-origin task registry",
        [
            "120 frozen questions · 30 per family",
            "accepted answers are non-binding references · legacy data quarantined",
        ],
        INK,
    )
    box(
        120,
        400,
        700,
        126,
        "Track A · Model Arena",
        [
            "(Mi, Epicure on)  vs  (Mj, Epicure on)",
            "12 frozen endpoints · balanced matching design · up to 720 pairs",
        ],
        BLUE,
    )
    box(
        980,
        400,
        700,
        126,
        "Track B · Epicure Uplift",
        [
            "(Mi, Epicure off)  vs  (Mi, Epicure on)",
            "same task + model + decoding contract · up to 1,440 paired contrasts",
            "access is the treatment · invocation is observed, never used to reclassify",
        ],
        GREEN,
    )
    box(
        120,
        610,
        460,
        144,
        "Frozen execution",
        [
            "Bedrock primary · OpenRouter exact routes",
            "8 tool rounds · 32 calls · one final answer",
            "identity, tokens, latency, cost, retries",
        ],
        GOLD,
    )
    box(
        670,
        610,
        460,
        144,
        "Epicure MCP intervention",
        [
            "1,790 ingredients · 300 dimensions",
            "13 real read-only tools · complete trace",
            "similarity is evidence, never ground truth",
        ],
        GREEN,
    )
    box(
        1220,
        610,
        460,
        144,
        "Immutable evidence store",
        [
            "request-start journal before inference",
            "append-only arms + superseding corrections",
            "prompt · schema · tool · app · bundle hashes",
        ],
        INK,
    )
    box(
        120,
        842,
        460,
        146,
        "Automated cohort",
        [
            "4 Bedrock judges · 3 model families",
            "original + swapped orientation",
            "self-judgments excluded from primary",
        ],
        BLUE,
    )
    box(
        670,
        842,
        460,
        146,
        "Reserved human cohorts",
        [
            "not collected in the current automated release",
            "future public preference + qualified expert rubric",
            "always reported separately from automated judges",
        ],
        GOLD,
    )
    box(
        1220,
        842,
        460,
        146,
        "Transparent outputs",
        [
            "Bradley–Terry model preference",
            "paired uplift W/T/L + intervals",
            "quality · reliability · cost · latency",
        ],
        GREEN,
    )
    arrow(900, 332, 470, 400)
    arrow(900, 332, 1330, 400)
    arrow(470, 526, 350, 610)
    arrow(1330, 526, 500, 610)
    lines.extend(
        [
            '<path d="M580,682 L670,682" fill="none" stroke="#65706C" stroke-width="2.5" marker-end="url(#arrow)"/>',
            '<text x="625" y="666" class="small" text-anchor="middle">on arms</text>',
            '<path d="M1130,682 L1220,682" fill="none" stroke="#65706C" stroke-width="2.5" marker-end="url(#arrow)"/>',
            '<path d="M580,730 C820,790 1020,790 1220,730" fill="none" stroke="#65706C" stroke-width="2.5" marker-end="url(#arrow)"/>',
        ]
    )
    arrow(1450, 754, 350, 842)
    arrow(1450, 754, 900, 842)
    lines.extend(
        [
            '<path d="M580,915 C620,915 620,1018 690,1018 L1150,1018 C1190,1018 1190,915 1220,915" fill="none" stroke="#65706C" stroke-width="2.5" marker-end="url(#arrow)"/>',
            '<path d="M1130,915 C1170,915 1190,915 1220,915" fill="none" stroke="#65706C" stroke-width="2.5" marker-end="url(#arrow)"/>',
            '<text x="900" y="1080" class="small" text-anchor="middle">The model track estimates system quality. The paired track isolates the Epicure treatment. Operational metrics remain separate.</text>',
        ]
    )
    _write(path, lines)


def render_all(analysis: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "model_forest": output_dir / "season0-model-arena.svg",
        "uplift_forest": output_dir / "season0-epicure-uplift.svg",
        "dimension_uplift": output_dir / "season0-dimension-uplift.svg",
        "family_heatmap": output_dir / "season0-family-heatmap.svg",
        "quality_cost": output_dir / "season0-quality-cost.svg",
        "adoption_uplift": output_dir / "season0-adoption-uplift.svg",
        "architecture": output_dir / "season0-architecture.svg",
    }
    render_model_forest(analysis, paths["model_forest"])
    render_uplift_forest(analysis, paths["uplift_forest"])
    render_dimension_uplift(analysis, paths["dimension_uplift"])
    render_family_heatmap(analysis, paths["family_heatmap"])
    render_quality_cost(analysis, paths["quality_cost"])
    render_adoption_uplift(analysis, paths["adoption_uplift"])
    render_architecture(paths["architecture"])
    return {key: str(path) for key, path in paths.items()}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_all(_load(args.analysis), args.output_dir), indent=2))


if __name__ == "__main__":
    run()
