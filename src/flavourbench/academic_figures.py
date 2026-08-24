"""Render publication-grade vector figures for the retrospective pilot audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .real_task_bank import sha256_json

BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
NAVY = "#184E77"
INK = "#202124"
MID = "#667085"
LIGHT = "#E7EBF0"
PALE = "#F5F7FA"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _clean_axis(axis: mpl.axes.Axes, *, grid: str | None = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT, linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)


def _save(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        facecolor="white",
        transparent=False,
        metadata={
            "Title": path.stem,
            "Author": "Josef Chen",
            "Creator": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_evidence_flow(data_dir: Path, output: Path) -> dict[str, Any]:
    rows = _read_csv(data_dir / "pilot-flow.csv")
    values = {(row["panel"], row["stage"]): int(row["count"]) for row in rows}
    task_labels = [
        "Source pool",
        "LLM-curator agreement",
        "Frozen sample",
        "Human reviewed",
    ]
    task_values = [
        values[("task", "source_candidates")],
        values[("task", "strict_llm_curator_agreement")],
        values[("task", "selected_tasks")],
        values[("task", "qualified_human_reviewed")],
    ]
    arm_labels = ["Attempted", "Collector accepted", "Normal finish", "Effective failure"]
    arm_values = [
        values[("evaluation", "response_arms")],
        values[("evaluation", "collector_accepted_arms")],
        values[("evaluation", "effective_complete_arms")],
        values[("evaluation", "effective_failed_arms")],
    ]
    comparison_labels = ["Planned", "Original cohort", "Effective cohort", "Consensus"]
    comparison_values = [
        values[("evaluation", "planned_comparisons")],
        values[("evaluation", "source_judging_admitted_comparisons")],
        values[("evaluation", "effective_judging_admitted_comparisons")],
        values[("evaluation", "primary_consensus")],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.1), gridspec_kw={"wspace": 0.92})
    panels = [
        ("a  Task evidence, N=979", task_labels, task_values, 979),
        ("b  Response arms, N=2,880", arm_labels, arm_values, 2880),
        ("c  Comparisons, N=2,160", comparison_labels, comparison_values, 2160),
    ]
    for axis, (title, labels, values, denominator) in zip(axes, panels, strict=True):
        y = np.arange(len(labels))[::-1]
        widths = np.asarray(values, dtype=float) / denominator
        colors = [BLUE] * len(labels)
        if title.startswith("b"):
            colors[-1] = ORANGE
        if title.startswith("a"):
            colors[-1] = ORANGE
        axis.barh(y, widths, color=colors, height=0.54, zorder=2)
        axis.set_yticks(y, labels)
        axis.set_xlim(0, 1.03)
        axis.set_xticks([0, 0.5, 1.0], ["0", "50", "100%"])
        axis.set_title(title, loc="left", pad=7)
        for yi, value, width in zip(y, values, widths, strict=True):
            if width >= 0.29:
                axis.text(
                    width - 0.025,
                    yi,
                    f"{value:,}",
                    va="center",
                    ha="right",
                    fontsize=8.0,
                    color="white",
                    fontweight="bold",
                )
            else:
                axis.text(
                    max(width + 0.025, 0.04),
                    yi,
                    f"{value:,}",
                    va="center",
                    ha="left",
                    fontsize=8.0,
                )
        _clean_axis(axis)
    fig.text(
        0.5,
        -0.01,
        "Panels have distinct denominators. Response arms are reused across comparison tracks.",
        ha="center",
        color=MID,
        fontsize=8.0,
    )
    _save(fig, output)
    return {
        "figure": output.name,
        "question": "Where did evidence leave the retrospective evaluation pipeline?",
        "chart_type": "three-panel normalized attrition bars",
        "denominators": {
            "tasks": task_values[0],
            "response_arms": arm_values[0],
            "comparisons": comparison_values[0],
        },
        "source": "pilot-flow.csv",
    }


def render_measurement_integrity(data_dir: Path, output: Path) -> dict[str, Any]:
    all_rows = _read_csv(data_dir / "pilot-measurement-integrity.csv")
    rows = [row for row in all_rows if row["panel"] == "judge"]
    judges = list(dict.fromkeys(row["label"] for row in rows))
    metrics = [
        ("both_orientations_complete", "Both orientations / submitted", SKY, "o"),
        ("agreement_given_completion", "Agreement / completed pairs", NAVY, "s"),
        ("eligible_vote_yield", "Eligible votes / submitted", BLUE, "D"),
    ]
    value = {(row["label"], row["metric"]): row for row in rows}

    fig, axis = plt.subplots(figsize=(7.15, 2.55))
    y = np.arange(len(judges))[::-1]
    offsets = np.linspace(0.18, -0.18, len(metrics))
    for offset, (metric, label, color, marker) in zip(offsets, metrics, strict=True):
        metric_rows = [value[(judge, metric)] for judge in judges]
        x = np.asarray([100 * float(row["rate"]) for row in metric_rows])
        lower = np.asarray([100 * float(row["wilson_lower_95"]) for row in metric_rows])
        upper = np.asarray([100 * float(row["wilson_upper_95"]) for row in metric_rows])
        axis.errorbar(
            x,
            y + offset,
            xerr=np.vstack((x - lower, upper - x)),
            fmt=marker,
            markersize=4.8,
            color=color,
            ecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.35,
            linewidth=0.9,
            capsize=1.8,
            label=label,
        )
        for yi, row in zip(y + offset, metric_rows, strict=True):
            axis.text(
                102.0,
                yi,
                f"{int(row['numerator'])}/{int(row['denominator'])}",
                va="center",
                ha="left",
                fontsize=8.0,
                color=MID,
            )
    axis.set_yticks(y, judges)
    axis.set_xlim(0, 121)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Rate (%) with Wilson 95% interval; exact n/N at right")
    axis.set_title(
        "Historical judge-pair completion, orientation agreement, and eligible-vote yield",
        loc="left",
        pad=22,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.12),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    primary = next(
        row
        for row in all_rows
        if row["panel"] == "diagnostic" and row["label"] == "Primary consensus"
    )
    primary_rate = 100 * float(primary["rate"])
    axis.axvline(primary_rate, color=ORANGE, linestyle=(0, (3, 2)), linewidth=1.0)
    axis.text(
        primary_rate + 1.0,
        -0.58,
        (
            f"Primary consensus: {primary_rate:.1f}% "
            f"({int(primary['numerator'])}/{int(primary['denominator'])})"
        ),
        color=ORANGE,
        fontsize=8.0,
        va="bottom",
    )
    _clean_axis(axis)
    _save(fig, output)
    return {
        "figure": output.name,
        "question": "How much judge evidence survived orientation and eligibility rules?",
        "chart_type": "grouped dot-and-interval plot",
        "interval": "Wilson 95% intervals",
        "primary_consensus_denominator": int(primary["denominator"]),
        "source": "pilot-measurement-integrity.csv",
    }


def render_model_stability(data_dir: Path, output: Path) -> dict[str, Any]:
    rows = _read_csv(data_dir / "pilot-model-uncertainty.csv")
    rows.sort(key=lambda row: row["season_model_id"])
    display_prefixes = {
        "Anthropic Claude ": "Claude ",
        "OpenAI GPT-": "GPT-",
        "OpenAI gpt-": "gpt-",
        "Google Gemini ": "Gemini ",
        "Mistral Devstral ": "Devstral ",
    }
    names = []
    for row in rows:
        name = row["model"]
        for prefix, replacement in display_prefixes.items():
            if name.startswith(prefix):
                name = replacement + name.removeprefix(prefix)
                break
        if row["complete_separation"] == "True":
            name += "†"
        names.append(name)
    medians = np.asarray([float(row["bootstrap_median"]) - 1000 for row in rows])
    lower = np.asarray([float(row["bootstrap_lower"]) - 1000 for row in rows])
    upper = np.asarray([float(row["bootstrap_upper"]) - 1000 for row in rows])
    n = np.asarray([int(row["n"]) for row in rows])
    separated = np.asarray([row["complete_separation"] == "True" for row in rows])

    fig, axis = plt.subplots(figsize=(7.15, 3.8))
    y = np.arange(len(rows))[::-1]
    axis.axvline(0, color=MID, linewidth=0.8, zorder=1)
    for yi, median, lo, hi, is_separated in zip(y, medians, lower, upper, separated, strict=True):
        if is_separated:
            axis.plot(
                [lo, hi],
                [yi, yi],
                color=ORANGE,
                linewidth=1.2,
                linestyle="--",
                zorder=2,
            )
            axis.scatter(
                [median],
                [yi],
                s=32,
                marker="D",
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=1.1,
                zorder=3,
            )
            continue
        axis.plot([lo, hi], [yi, yi], color=BLUE, linewidth=1.2, zorder=2)
        axis.scatter(
            [median],
            [yi],
            s=28,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=1.0,
            zorder=3,
        )
    axis.set_yticks(y, names)
    axis.set_xlabel("Task-bootstrap ridge BT rating minus 1,000 (arbitrary origin)")
    axis.set_title(
        "Historical pilot endpoint uncertainty (frozen manifest order)",
        loc="left",
        pad=8,
    )
    axis.set_xlim(min(lower) - 300, max(upper) + 520)
    table_x = max(upper) + 180
    axis.text(
        table_x,
        y.max() + 0.72,
        "valid n",
        fontsize=8.0,
        color=MID,
        ha="left",
    )
    for yi, ni in zip(y, n, strict=True):
        axis.text(table_x, yi, f"{ni}", va="center", fontsize=8.0)
    fig.subplots_adjust(bottom=0.11)
    _clean_axis(axis)
    _save(fig, output)
    return {
        "figure": output.name,
        "question": "Does the sparse automated-consensus graph resolve an endpoint ordering?",
        "chart_type": "manifest-ordered uncertainty forest plot",
        "interval": "95% observed-task bootstrap range, 1,000 replicates",
        "ordering": "frozen season_model_id, not an estimated rank",
        "source": "pilot-model-uncertainty.csv",
    }


def render_uplift_reliability(data_dir: Path, output: Path) -> dict[str, Any]:
    uplift = _read_csv(data_dir / "pilot-epicure-robustness.csv")
    reliability = _read_csv(data_dir / "pilot-condition-reliability.csv")[0]
    attrition = {
        row["outcome"]: int(row["count"])
        for row in _read_csv(data_dir / "pilot-condition-attrition.csv")
    }
    labels = [
        "Cell weighted",
        "Equal task means",
        "Equal endpoint means",
        "Image-dependent item removed",
        "Cross-family admitted subset",
    ]
    estimate = np.asarray([float(row["estimate"]) for row in uplift])
    lower = np.asarray([float(row["lower"]) for row in uplift])
    upper = np.asarray([float(row["upper"]) for row in uplift])
    n = [int(row["n"]) for row in uplift]

    fig = plt.figure(figsize=(7.15, 2.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.42, 1], wspace=0.5)
    left = fig.add_subplot(outer[0, 0])
    right_grid = outer[0, 1].subgridspec(
        2,
        1,
        height_ratios=[1.35, 0.72],
        hspace=0.62,
    )
    matrix_axis = fig.add_subplot(right_grid[0, 0])
    risk_axis = fig.add_subplot(right_grid[1, 0])
    y = np.arange(len(labels))[::-1]
    left.axvline(0.5, color=MID, linewidth=0.8)
    row_colors = [BLUE, BLUE, BLUE, MID, ORANGE]
    for yi, point, lo, hi, color in zip(y, estimate, lower, upper, row_colors, strict=True):
        left.errorbar(
            [point],
            [yi],
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.5,
            capsize=2.2,
            linewidth=1.1,
        )
    left.set_yticks(y, labels)
    left.set_xlim(max(0.0, float(lower.min()) - 0.04), min(1.0, float(upper.max()) + 0.06))
    left.set_xlabel("Tool-available preference share")
    left.set_title(
        "a  Paired preference estimates across\n    weighting and admission rules",
        loc="left",
        pad=8,
    )
    for yi, hi, ni in zip(y, upper, n, strict=True):
        left.text(hi + 0.006, yi, f"n={ni}", va="center", fontsize=8.0, color=MID)
    left.axhline(1.5, color=LIGHT, linewidth=0.8)
    left.axhline(0.5, color=LIGHT, linewidth=0.8)
    _clean_axis(left)

    attempted = int(reliability["attempted_cells"])
    matrix = [
        [attrition["both_success"], attrition["off_failed_on_success"]],
        [attrition["on_failed_off_success"], attrition["both_failed"]],
    ]
    cell_styles = [
        [("#E8F3FA", BLUE), ("#F3F8FC", NAVY)],
        [("#FFF1E8", ORANGE), ("#F1F3F5", MID)],
    ]
    for row_index, row in enumerate(matrix):
        for column_index, count in enumerate(row):
            face, edge = cell_styles[row_index][column_index]
            matrix_axis.add_patch(
                Rectangle(
                    (column_index, 1 - row_index),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.9,
                )
            )
            matrix_axis.text(
                column_index + 0.5,
                1.5 - row_index,
                f"{count:,}\n({100 * count / attempted:.1f}%)",
                ha="center",
                va="center",
                fontsize=8.0,
                color=INK,
            )
    matrix_axis.set_xlim(0, 2)
    matrix_axis.set_ylim(0, 2)
    matrix_axis.set_xticks(
        [0.5, 1.5],
        ["Tool-unavailable\ncompleted", "Tool-unavailable\nfailed"],
    )
    matrix_axis.xaxis.tick_top()
    matrix_axis.set_yticks(
        [1.5, 0.5],
        ["Tool-available\ncompleted", "Tool-available\nfailed"],
    )
    matrix_axis.tick_params(length=0, pad=2)
    matrix_axis.set_title("b  Matched response outcomes", loc="left", pad=28)
    matrix_axis.spines[:].set_visible(False)

    rd = float(reliability["realized_success_proportion_difference"])
    lo = float(reliability["lower_95_task_bootstrap"])
    hi = float(reliability["upper_95_task_bootstrap"])
    risk_axis.errorbar(
        [rd],
        [0],
        xerr=[[rd - lo], [hi - rd]],
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        capsize=2.5,
        markersize=4.8,
    )
    risk_axis.axvline(0, color=MID, linewidth=0.8)
    risk_axis.set_xlim(-0.2, 0.03)
    risk_axis.set_yticks([])
    risk_axis.set_xticks([-0.2, -0.1, 0], ["−20", "−10", "0"])
    risk_axis.set_xlabel(
        "Tool-available minus tool-unavailable realized completion proportion (pp)"
    )
    risk_axis.text(
        rd,
        0.78,
        f"{100 * rd:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
        transform=risk_axis.get_xaxis_transform(),
        va="center",
        ha="center",
        color=ORANGE,
        fontsize=8.0,
    )
    _clean_axis(risk_axis, grid="x")
    _save(fig, output)
    return {
        "figure": output.name,
        "question": (
            "How sensitive is the paired preference contrast, and what reliability cost "
            "is hidden by complete-case analysis?"
        ),
        "chart_type": "forest plot and matched-outcome decomposition",
        "sources": [
            "pilot-epicure-robustness.csv",
            "pilot-condition-reliability.csv",
            "pilot-condition-attrition.csv",
        ],
    }


def render_all(data_dir: Path, output_dir: Path) -> Path:
    _configure()
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        render_evidence_flow(data_dir, output_dir / "pilot-evidence-flow.pdf"),
        render_measurement_integrity(data_dir, output_dir / "pilot-judge-survival.pdf"),
        render_model_stability(data_dir, output_dir / "pilot-model-stability.pdf"),
        render_uplift_reliability(data_dir, output_dir / "pilot-uplift-reliability.pdf"),
    ]
    source_names = sorted(
        {
            source
            for spec in specifications
            for source in (
                [spec["source"]]
                if isinstance(spec.get("source"), str)
                else list(spec.get("sources") or [])
            )
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "flavourbench-academic-figure-provenance-v2",
        "renderer": {
            "library": "matplotlib",
            "version": mpl.__version__,
            "background": "white",
            "format": "vector-pdf",
            "font_embedding": "TrueType",
        },
        "figures": specifications,
        "sources": {name: _file_sha256(data_dir / name) for name in source_names},
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    provenance = output_dir / "pilot-academic-figure-provenance.json"
    provenance.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return provenance


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    provenance = render_all(args.data_dir, args.output_dir)
    print(json.dumps({"provenance": str(provenance)}, indent=2))


if __name__ == "__main__":
    run()
