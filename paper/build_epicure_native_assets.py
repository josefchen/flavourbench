#!/usr/bin/env python3
"""Build paper tables and vector figures from the public Epicure-native release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


class AssetError(RuntimeError):
    """The release is incomplete or inconsistent with the paper contract."""


SHORT_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol Pro",
    "openai/gpt-5.6-terra-pro": "GPT-5.6 Terra Pro",
    "openai/gpt-5.6-luna-pro": "GPT-5.6 Luna Pro",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-opus-5": "Claude Opus 5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.6-flash": "Gemini 3.6 Flash",
    "x-ai/grok-4.5": "Grok 4.5",
    "moonshotai/kimi-k3": "Kimi K3",
    "qwen/qwen3.8-max": "Qwen 3.8 Max",
    "z-ai/glm-5.2": "GLM 5.2",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "minimax/minimax-m3": "MiniMax M3",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
    "mistralai/mistral-large-2512": "Mistral Large 2512",
    "tencent/hy3": "Tencent HY 3",
    "cohere/command-a-plus-05-2026": "Command A Plus",
    "cohere/command-a-reasoning-08-2025": "Command A Reasoning",
}

FAMILIES = ("substitution", "composition", "cookability", "evidence")
FAMILY_LABELS = ("Substitution", "Composition", "Cookability", "Evidence")
BLUE = "#1769AA"
GOLD = "#E6A11A"
TEAL = "#168C7A"
RED = "#C75450"
CHARCOAL = "#262B33"
LIGHT = "#E9EDF2"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_release(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AssetError("release must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssetError("release root must be an object")
    payload = dict(value)
    recorded = str(payload.pop("artifact_sha256", ""))
    if recorded != _sha256(payload):
        raise AssetError("release content address does not verify")
    counts = value.get("counts") or {}
    if (
        value.get("release_status") != "complete_public_automated_leaderboard"
        or counts.get("models") != 20
        or counts.get("tasks") != 32
        or counts.get("assigned_arms") != 1280
        or len(value.get("observations") or []) != 1280
    ):
        raise AssetError("release does not contain the complete 20 by 32 paired grid")
    return value


def _tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#9CA6B3",
            "axes.linewidth": 0.7,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(figure: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _ranked_rows(release: dict[str, Any]) -> list[dict[str, Any]]:
    rows = release.get("leaderboard", {}).get("models")
    if not isinstance(rows, list) or len(rows) != 20:
        raise AssetError("leaderboard has no exact 20-row ranking")
    if [row.get("rank") for row in rows] != list(range(1, 21)):
        raise AssetError("leaderboard ranks are incomplete")
    return rows


def _short(model_id: str) -> str:
    return SHORT_NAMES.get(model_id, model_id.rsplit("/", 1)[-1])


def _paired_exact_p(release: dict[str, Any], left_model: str, right_model: str) -> float:
    outcomes: dict[tuple[str, str], bool] = {}
    for observation in release["observations"]:
        if observation["condition"] != "epicure_off":
            continue
        key = (str(observation["model_id"]), str(observation["task_id"]))
        if key in outcomes:
            raise AssetError("duplicate Model only observation in paired test")
        outcomes[key] = bool(observation["correct"])

    task_ids = [str(task["task_id"]) for task in release["tasks"]]
    if len(task_ids) != 32 or len(set(task_ids)) != 32:
        raise AssetError("paired test requires the exact 32-task panel")
    left_only = right_only = 0
    for task_id in task_ids:
        left = outcomes.get((left_model, task_id))
        right = outcomes.get((right_model, task_id))
        if left is None or right is None:
            raise AssetError("paired test is missing a Model only observation")
        left_only += int(left and not right)
        right_only += int(right and not left)
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def _score_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["epicure_benchmark_score"]),
            _short(str(row["model_id"])).casefold(),
        ),
    )


def _score_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    previous_score: float | None = None
    current_rank = 0
    for position, row in enumerate(_score_display_rows(rows), start=1):
        score = float(row["epicure_benchmark_score"])
        if previous_score is None or score != previous_score:
            current_rank = position
            previous_score = score
        ranks[str(row["model_id"])] = current_rank
    return ranks


def _leaderboard_table(rows: list[dict[str, Any]]) -> str:
    score_ranks = _score_ranks(rows)
    lines = [
        r"\begin{tabular}{@{}r l r r c r@{}}",
        r"\toprule",
        r"Score rank & Model & Correct & FB Score & Wilson 95\% & Parsed \\",
        r"\midrule",
    ]
    for row in _score_display_rows(rows):
        off = row["conditions"]["epicure_off"]
        lower, upper = (100 * float(value) for value in off["wilson_95"])
        rank = str(score_ranks[str(row["model_id"])]) if off["reliability"] > 0 else "DNF"
        lines.append(
            f"{rank} & {_tex(_short(row['model_id']))} & "
            f"{off['correct']}/32 & {off['accuracy_percent']:.1f} & "
            f"{lower:.1f} to {upper:.1f} & {off['parseable_answers']}/32 \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _route_table(release: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    route_by_model = {model["model_id"]: model for model in release["models"]}
    lines = [
        r"\begin{tabular}{@{}r l l@{}}",
        r"\toprule",
        r"Slot & Model & Execution route \\",
        r"\midrule",
    ]
    for row in rows:
        model = route_by_model[row["model_id"]]
        backend = str(model["execution_backend"])
        route = str(model["provider_route"])
        label = f"{backend}: {route}" if route and route != backend else backend
        lines.append(
            f"{int(str(model['slot_id']).rsplit('-', 1)[-1])} & "
            f"{_tex(_short(row['model_id']))} & {_tex(label)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _family_table(rows: list[dict[str, Any]], release: dict[str, Any]) -> str:
    tools_by_family: dict[str, set[str]] = defaultdict(set)
    for task in release["tasks"]:
        call = task.get("reference_tool_call") or {}
        tools_by_family[str(task["family"])].add(str(call.get("name") or ""))
    lines = [
        r"\begin{tabular}{@{}l l r r r@{}}",
        r"\toprule",
        r"Family & Epicure operation & Tasks & Model only & Model + Epicure \\",
        r"\midrule",
    ]
    for family, label in zip(FAMILIES, FAMILY_LABELS, strict=True):
        off = np.mean([row["conditions"]["epicure_off"]["family_accuracy"][family] for row in rows])
        on = np.mean([row["conditions"]["epicure_on"]["family_accuracy"][family] for row in rows])
        tool = ", ".join(sorted(tools_by_family[family]))
        lines.append(
            f"{label} & \\texttt{{{_tex(tool)}}} & 8 & {100 * off:.1f} & {100 * on:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _macros(release: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    aggregate = release["leaderboard"]["aggregate"]
    top = rows[0]
    completed_rows = [
        row for row in rows if float(row["conditions"]["epicure_off"]["reliability"]) > 0
    ]
    bottom = completed_rows[-1]
    cohere = [row for row in rows if row["model_id"].startswith("cohere/")]
    score_ranks = _score_ranks(rows)
    best_uplift = max(rows, key=lambda row: float(row["uplift_percentage_points"]))
    worst_uplift = min(rows, key=lambda row: float(row["uplift_percentage_points"]))
    worst_base_reliability = min(
        float(row["conditions"]["epicure_off"]["reliability"]) for row in rows
    )
    worst_base_reliability_models = [
        row
        for row in rows
        if float(row["conditions"]["epicure_off"]["reliability"]) == worst_base_reliability
    ]
    positive_uplift = sum(float(row["uplift_percentage_points"]) > 0 for row in rows)
    negative_uplift = sum(float(row["uplift_percentage_points"]) < 0 for row in rows)
    zero_uplift = len(rows) - positive_uplift - negative_uplift
    worst_reliability_names = _tex(
        " and ".join(_short(row["model_id"]) for row in worst_base_reliability_models)
    )
    cohere_plus = next(row for row in cohere if "plus" in row["model_id"])
    cohere_reasoning = next(row for row in cohere if "reasoning" in row["model_id"])
    assisted_eligible = [
        row for row in rows if int(row["conditions"]["epicure_on"]["normal_completions"]) > 0
    ]
    assisted_ceiling = sum(
        float(row["conditions"]["epicure_on"]["accuracy_percent"]) == 100
        for row in assisted_eligible
    )
    assisted_accuracy = (
        100
        * sum(int(row["conditions"]["epicure_on"]["correct"]) for row in assisted_eligible)
        / (32 * len(assisted_eligible))
    )
    score_gain_correlation = float(
        np.corrcoef(
            [float(row["epicure_benchmark_score"]) for row in assisted_eligible],
            [float(row["uplift_percentage_points"]) for row in assisted_eligible],
        )[0, 1]
    )
    top_vs_second_p = _paired_exact_p(release, top["model_id"], rows[1]["model_id"])
    top_vs_third_p = _paired_exact_p(release, top["model_id"], rows[2]["model_id"])
    pairwise = [
        (_paired_exact_p(release, top["model_id"], row["model_id"]), row) for row in rows[1:]
    ]
    adjusted: list[tuple[float, dict[str, Any]]] = []
    running_adjusted = 0.0
    for position, (raw_p, row) in enumerate(sorted(pairwise, key=lambda item: item[0])):
        candidate = min(1.0, raw_p * (len(pairwise) - position))
        running_adjusted = max(running_adjusted, candidate)
        adjusted.append((running_adjusted, row))
    separated = [row for adjusted_p, row in adjusted if adjusted_p <= 0.05]
    separated.sort(key=lambda row: -float(row["epicure_benchmark_score"]))
    separated_names = _tex(", ".join(_short(row["model_id"]) for row in separated))
    top_interval = [100 * float(value) for value in top["conditions"]["epicure_off"]["wilson_95"]]
    lines = [
        f"\\newcommand{{\\FBReleaseSHA}}{{\\texttt{{{release['artifact_sha256'][:12]}\\ldots}}}}",
        r"\newcommand{\FBModels}{20}",
        r"\newcommand{\FBTasks}{32}",
        r"\newcommand{\FBPairs}{640}",
        r"\newcommand{\FBArms}{1280}",
        f"\\newcommand{{\\FBObservedArms}}{{{release['counts']['observed_response_arms']}}}",
        f"\\newcommand{{\\FBMissingArms}}{{{1280 - release['counts']['observed_response_arms']}}}",
        f"\\newcommand{{\\FBTopModel}}{{{_tex(_short(top['model_id']))}}}",
        f"\\newcommand{{\\FBTopScore}}{{{top['epicure_benchmark_score']:.1f}}}",
        f"\\newcommand{{\\FBTopCorrect}}{{{top['conditions']['epicure_off']['correct']}}}",
        f"\\newcommand{{\\FBTopWilsonLow}}{{{top_interval[0]:.1f}}}",
        f"\\newcommand{{\\FBTopWilsonHigh}}{{{top_interval[1]:.1f}}}",
        f"\\newcommand{{\\FBTopVsSecondP}}{{{top_vs_second_p:.3f}}}",
        f"\\newcommand{{\\FBTopVsThirdP}}{{{top_vs_third_p:.3f}}}",
        f"\\newcommand{{\\FBLeaderHolmSeparated}}{{{len(separated)}}}",
        f"\\newcommand{{\\FBLeaderHolmComparisons}}{{{len(pairwise)}}}",
        f"\\newcommand{{\\FBLeaderHolmSeparatedNames}}{{{separated_names}}}",
        f"\\newcommand{{\\FBTopOnCorrect}}{{{top['conditions']['epicure_on']['correct']}}}",
        f"\\newcommand{{\\FBTopOnScore}}{{{top['conditions']['epicure_on']['accuracy_percent']:.1f}}}",
        f"\\newcommand{{\\FBTopUplift}}{{{top['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBBottomModel}}{{{_tex(_short(bottom['model_id']))}}}",
        f"\\newcommand{{\\FBBottomScore}}{{{bottom['epicure_benchmark_score']:.1f}}}",
        f"\\newcommand{{\\FBBottomCorrect}}{{{bottom['conditions']['epicure_off']['correct']}}}",
        f"\\newcommand{{\\FBAggregateOff}}{{{100 * aggregate['epicure_off_correct'] / 640:.1f}}}",
        f"\\newcommand{{\\FBAggregateOn}}{{{100 * aggregate['epicure_on_correct'] / 640:.1f}}}",
        f"\\newcommand{{\\FBAggregateUplift}}{{{aggregate['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBToolCalls}}{{{aggregate['epicure_tool_calls']}}}",
        f"\\newcommand{{\\FBReferenceMatches}}{{{aggregate['reference_tool_match_pairs']}}}",
        f"\\newcommand{{\\FBObservedCost}}{{{aggregate['observed_cost_usd']:.2f}}}",
        f"\\newcommand{{\\FBAssistedEligibleModels}}{{{len(assisted_eligible)}}}",
        f"\\newcommand{{\\FBAssistedCeilingModels}}{{{assisted_ceiling}}}",
        f"\\newcommand{{\\FBAssistedEligibleAccuracy}}{{{assisted_accuracy:.1f}}}",
        f"\\newcommand{{\\FBScoreGainCorrelation}}{{{score_gain_correlation:.3f}}}",
        f"\\newcommand{{\\FBPositiveUpliftModels}}{{{positive_uplift}}}",
        f"\\newcommand{{\\FBNegativeUpliftModels}}{{{negative_uplift}}}",
        f"\\newcommand{{\\FBZeroUpliftModels}}{{{zero_uplift}}}",
        f"\\newcommand{{\\FBBestUpliftModel}}{{{_tex(_short(best_uplift['model_id']))}}}",
        f"\\newcommand{{\\FBBestUplift}}{{{best_uplift['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBWorstUpliftModel}}{{{_tex(_short(worst_uplift['model_id']))}}}",
        f"\\newcommand{{\\FBWorstUplift}}{{{worst_uplift['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBWorstBaseReliabilityModels}}{{{worst_reliability_names}}}",
        f"\\newcommand{{\\FBWorstBaseReliability}}{{{100 * worst_base_reliability:.1f}}}",
        f"\\newcommand{{\\FBCoherePlusRank}}{{{score_ranks[cohere_plus['model_id']]}}}",
        f"\\newcommand{{\\FBCohereReasoningRank}}{{{score_ranks[cohere_reasoning['model_id']]}}}",
        f"\\newcommand{{\\FBCoherePlusScore}}{{{cohere_plus['epicure_benchmark_score']:.1f}}}",
        f"\\newcommand{{\\FBCoherePlusUplift}}{{{cohere_plus['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBCohereReasoningScore}}{{{cohere_reasoning['epicure_benchmark_score']:.1f}}}",
        f"\\newcommand{{\\FBCohereReasoningUplift}}{{{cohere_reasoning['uplift_percentage_points']:+.1f}}}",
        f"\\newcommand{{\\FBKimiRank}}{{{score_ranks['moonshotai/kimi-k3']}}}",
        f"\\newcommand{{\\FBQwenRank}}{{{score_ranks['qwen/qwen3.8-max']}}}",
        f"\\newcommand{{\\FBGLMRank}}{{{score_ranks['z-ai/glm-5.2']}}}",
    ]
    return "\n".join(lines)


def _dumbbell(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = list(reversed(rows))
    names = [_short(row["model_id"]) for row in ordered]
    off = np.array([row["conditions"]["epicure_off"]["accuracy_percent"] for row in ordered])
    on = np.array([row["conditions"]["epicure_on"]["accuracy_percent"] for row in ordered])
    figure, axis = plt.subplots(figsize=(7.2, 7.4))
    y = np.arange(len(rows))
    for index, (left, right) in enumerate(zip(off, on, strict=True)):
        color = TEAL if right > left else RED if right < left else "#A7AFB9"
        axis.plot(
            [left, right],
            [index, index],
            color=color,
            linewidth=2.2,
            alpha=0.75,
            zorder=1,
        )
    axis.scatter(off, y, s=35, color=BLUE, label="Model only", zorder=3)
    axis.scatter(on, y, s=42, color=GOLD, marker="D", label="Model + Epicure", zorder=3)
    axis.set_yticks(y, labels=names)
    axis.set_xlim(0, 103)
    axis.set_xlabel("Exact-choice accuracy (%)")
    aggregate_gain = float(np.mean(on - off))
    axis.set_title(
        f"Epicure raises panel accuracy by {aggregate_gain:.1f} points",
        loc="left",
        weight="bold",
    )
    axis.grid(axis="x", color="#DDE2E8", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(loc="lower right", frameon=False, ncol=2)
    _save(figure, output / "frontier-score-dumbbell")


def _score_forest(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = list(reversed(rows))
    names = [_short(row["model_id"]) for row in ordered]
    point = np.array([row["conditions"]["epicure_off"]["accuracy_percent"] for row in ordered])
    intervals = np.array([row["conditions"]["epicure_off"]["wilson_95"] for row in ordered]) * 100
    errors = np.vstack((point - intervals[:, 0], intervals[:, 1] - point))
    completed = np.array([row["conditions"]["epicure_off"]["reliability"] > 0 for row in ordered])
    figure, axis = plt.subplots(figsize=(7.2, 7.4))
    y = np.arange(len(rows))
    axis.errorbar(
        point[completed],
        y[completed],
        xerr=errors[:, completed],
        fmt="o",
        color=BLUE,
        ecolor="#8BAAC7",
        elinewidth=1.7,
        capsize=2.5,
        markersize=5.2,
    )
    if not np.all(completed):
        axis.scatter(
            point[~completed],
            y[~completed],
            marker="x",
            s=45,
            linewidth=1.5,
            color="#7B8491",
            label="No parseable answer",
            zorder=4,
        )
    axis.axvline(25, color=RED, linestyle="--", linewidth=1.1, label="Chance (25%)")
    axis.set_yticks(y, labels=names)
    axis.set_xlim(0, 103)
    axis.set_xlabel("FlavourBench Score (%)")
    axis.set_title(
        "FlavourBench Score on the common 32-task panel",
        loc="left",
        weight="bold",
    )
    axis.grid(axis="x", color="#DDE2E8", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(frameon=False, loc="lower right")
    _save(figure, output / "frontier-score-forest")


def _heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    matrices = []
    for condition in ("epicure_off", "epicure_on"):
        matrices.append(
            np.array(
                [
                    [row["conditions"][condition]["family_accuracy"][family] for family in FAMILIES]
                    for row in rows
                ]
            )
        )
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 8.1), sharey=True, constrained_layout=True)
    image = None
    for axis, matrix, title in zip(axes, matrices, ("Model only", "Model + Epicure"), strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        axis.set_xticks(range(4), labels=FAMILY_LABELS, rotation=28, ha="right")
        axis.set_yticks(range(20), labels=[_short(row["model_id"]) for row in rows])
        axis.set_title(title, loc="left", weight="bold")
        for row_index in range(20):
            for column_index in range(4):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.57 else CHARCOAL,
                )
        axis.tick_params(length=0)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Accuracy")
    figure.suptitle("The same score can hide different family profiles", weight="bold")
    _save(figure, output / "frontier-family-heatmap")


def _paired_matrix(release: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    task_ids = [task["task_id"] for task in release["tasks"]]
    family_by_task = {task["task_id"]: task["family"] for task in release["tasks"]}
    task_ids.sort(key=lambda task_id: (FAMILIES.index(family_by_task[task_id]), task_id))
    observed = {
        (row["model_id"], row["task_id"], row["condition"]): bool(row["correct"])
        for row in release["observations"]
    }
    matrix = np.zeros((20, 32), dtype=int)
    for row_index, row in enumerate(rows):
        for column_index, task_id in enumerate(task_ids):
            off = observed[(row["model_id"], task_id, "epicure_off")]
            on = observed[(row["model_id"], task_id, "epicure_on")]
            matrix[row_index, column_index] = 3 if off and on else 1 if off else 2 if on else 0
    cmap = ListedColormap(["#E5E8EC", RED, GOLD, BLUE])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    figure, axis = plt.subplots(figsize=(11.4, 6.9))
    axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axis.set_yticks(range(20), labels=[_short(row["model_id"]) for row in rows])
    axis.set_xticks([3.5, 11.5, 19.5, 27.5], labels=FAMILY_LABELS)
    axis.tick_params(length=0)
    for boundary in (7.5, 15.5, 23.5):
        axis.axvline(boundary, color="white", linewidth=2.5)
    axis.set_title("Every cell compares the same model and task", loc="left", weight="bold")
    legend = [
        Line2D([0], [0], marker="s", linestyle="", color=color, markersize=9, label=label)
        for color, label in (
            ("#E5E8EC", "Neither correct"),
            (RED, "Model only"),
            (GOLD, "Model + Epicure only"),
            (BLUE, "Both correct"),
        )
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
    )
    _save(figure, output / "frontier-paired-outcome-matrix")


def _latency_uplift(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    label_offsets = {
        "openai/gpt-5.6-sol-pro": (7, 8, "left"),
        "anthropic/claude-sonnet-5": (-8, -11, "right"),
        "deepseek/deepseek-v4-pro": (-8, 10, "right"),
        "google/gemini-3.6-flash": (6, 8, "left"),
        "moonshotai/kimi-k3": (6, -10, "left"),
        "cohere/command-a-plus-05-2026": (6, 8, "left"),
        "cohere/command-a-reasoning-08-2025": (6, 7, "left"),
        "x-ai/grok-4.5": (6, 8, "left"),
        "mistralai/mistral-large-2512": (6, 7, "left"),
        "nvidia/nemotron-3-ultra-550b-a55b": (6, 7, "left"),
    }
    observed_latencies = [
        float(value) / 1000
        for row in rows
        for value in (
            row["conditions"]["epicure_off"]["median_latency_ms"],
            row["conditions"]["epicure_on"]["median_latency_ms"],
        )
        if value is not None
    ]
    if not observed_latencies:
        raise AssetError("release contains no observed latency values")
    unavailable_floor = max(0.05, min(observed_latencies) / 2)
    unavailable_plotted = False
    for row in rows:
        off = row["conditions"]["epicure_off"]
        on = row["conditions"]["epicure_on"]
        arm_latencies = [
            value
            for value in (off["median_latency_ms"], on["median_latency_ms"])
            if value is not None
        ]
        latency_missing = not arm_latencies
        latency = unavailable_floor if latency_missing else float(np.mean(arm_latencies)) / 1000
        is_direct = row["execution_backend"] in {"cohere_direct", "kimi_direct"}
        color = TEAL if is_direct else BLUE
        axis.scatter(
            row["epicure_benchmark_score"],
            latency,
            s=58,
            color="#7B8491" if latency_missing else color,
            marker="x" if latency_missing else "o",
            alpha=0.82,
            edgecolor="white" if not latency_missing else None,
            linewidth=1.3 if latency_missing else 0.7,
        )
        unavailable_plotted = unavailable_plotted or latency_missing
        if row["rank"] <= 4 or row["rank"] >= 18 or row["execution_backend"] != "openrouter":
            dx, dy, horizontal_alignment = label_offsets.get(row["model_id"], (4, 4, "left"))
            axis.annotate(
                _short(row["model_id"]),
                (row["epicure_benchmark_score"], latency),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha=horizontal_alignment,
            )
    axis.set_yscale("log")
    axis.set_xlabel("FlavourBench Score")
    axis.set_ylabel("Median response latency (seconds, log scale)")
    axis.set_title(
        "Score and response time",
        loc="left",
        weight="bold",
    )
    axis.grid(color="#DDE2E8", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="OpenRouter"),
        Line2D([0], [0], marker="o", linestyle="", color=TEAL, label="Direct provider"),
    ]
    if unavailable_plotted:
        legend.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="",
                color="#7B8491",
                label="No observed latency",
            )
        )
    axis.legend(handles=legend, frameon=False, loc="upper right")
    _save(figure, output / "frontier-latency-uplift")


def _social_summary(release: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    figure = plt.figure(figsize=(12, 6.75), facecolor="white")
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1.7, 1), left=0.08, right=0.97, top=0.86, bottom=0.1
    )
    axis = figure.add_subplot(grid[0, 0])
    summary = figure.add_subplot(grid[0, 1])
    ordered = list(reversed(rows))
    names = [_short(row["model_id"]) for row in ordered]
    scores = [float(row["epicure_benchmark_score"]) for row in ordered]
    colors = [
        GOLD
        if row["rank"] == 1
        else TEAL
        if row["execution_backend"] in {"cohere_direct", "kimi_direct"}
        else BLUE
        for row in ordered
    ]
    bars = axis.barh(range(20), scores, color=colors, height=0.68)
    axis.axvline(25, color=RED, linestyle="--", linewidth=1.0, alpha=0.8)
    axis.text(25.8, 19.25, "chance", color=RED, fontsize=7.2, va="top")
    axis.set_yticks(range(20), labels=names)
    axis.set_xlim(0, 100)
    axis.set_xlabel("FlavourBench Score")
    axis.grid(axis="x", color="#E1E5EA", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=7.5)
    for bar, score in zip(bars, scores, strict=True):
        axis.text(
            min(score + 1.2, 96),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            fontsize=7.2,
            color=CHARCOAL,
        )

    summary.axis("off")
    summary.text(0, 0.96, "THE RESULT", fontsize=10, color=TEAL, weight="bold")
    summary.text(
        0,
        0.83,
        _short(rows[0]["model_id"]),
        fontsize=22,
        color=CHARCOAL,
        weight="bold",
        wrap=True,
    )
    summary.text(
        0,
        0.74,
        f"highest score in this run: {rows[0]['epicure_benchmark_score']:.1f}",
        fontsize=13,
        color=BLUE,
        weight="bold",
    )
    summary.text(0, 0.58, "20", fontsize=30, weight="bold", color=CHARCOAL)
    summary.text(0.2, 0.6, "frontier models", fontsize=11, color=CHARCOAL)
    summary.text(0, 0.46, "32", fontsize=30, weight="bold", color=CHARCOAL)
    summary.text(0.2, 0.48, "Epicure-generated tasks", fontsize=11, color=CHARCOAL)
    summary.text(0, 0.34, "640", fontsize=30, weight="bold", color=CHARCOAL)
    summary.text(0.25, 0.36, "matched model and Epicure pairs", fontsize=11, color=CHARCOAL)
    summary.text(
        0,
        0.19,
        f"{100 * rows[0]['conditions']['epicure_off']['wilson_95'][0]:.1f} to "
        f"{100 * rows[0]['conditions']['epicure_off']['wilson_95'][1]:.1f}%",
        fontsize=21,
        color=BLUE,
        weight="bold",
    )
    summary.text(0, 0.12, "leader Wilson 95% interval", fontsize=11, color=CHARCOAL)
    summary.text(
        0,
        0.01,
        "Score only ranks  |  exact offline replay  |  direct Kimi and Cohere",
        fontsize=8.5,
        color="#68717D",
    )
    figure.suptitle(
        "FlavourBench: executable culinary reasoning\nwithout a model judge",
        x=0.08,
        y=0.96,
        ha="left",
        fontsize=21,
        color=CHARCOAL,
        weight="bold",
    )
    _save(figure, output / "frontier-social-summary")


def _case_studies(
    release: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    task_by_id = {task["task_id"]: task for task in release["tasks"]}
    observations = {
        (row["model_id"], row["task_id"], row["condition"]): row for row in release["observations"]
    }
    cases: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_tasks = sorted(
            task["task_id"] for task in release["tasks"] if task["family"] == family
        )
        selected: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
        for task_id in family_tasks:
            for leaderboard_row in rows:
                off = observations[(leaderboard_row["model_id"], task_id, "epicure_off")]
                on = observations[(leaderboard_row["model_id"], task_id, "epicure_on")]
                if not off["correct"] and on["correct"] and on["tool_trace"]:
                    selected = (leaderboard_row, off, on)
                    break
            if selected is not None:
                break
        if selected is None:
            leaderboard_row = rows[0]
            task_id = family_tasks[0]
            selected = (
                leaderboard_row,
                observations[(leaderboard_row["model_id"], task_id, "epicure_off")],
                observations[(leaderboard_row["model_id"], task_id, "epicure_on")],
            )
        leaderboard_row, off, on = selected
        task = task_by_id[off["task_id"]]
        cases.append(
            {
                "family": family,
                "model_id": leaderboard_row["model_id"],
                "model": _short(leaderboard_row["model_id"]),
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "choices": task["choices"],
                "expected_choice": task["expected_choice"],
                "reference_tool_result": task["reference_tool_result"],
                "off_answer": off["answer_markdown"],
                "on_answer": on["answer_markdown"],
                "tool_trace": on["tool_trace"],
            }
        )
    lines: list[str] = []
    for index, case in enumerate(cases, start=1):
        tool = case["tool_trace"][0] if case["tool_trace"] else {"name": "none", "arguments": {}}
        arguments = json.dumps(tool["arguments"], ensure_ascii=False, sort_keys=True)
        reference_result = json.dumps(
            case["reference_tool_result"], ensure_ascii=False, sort_keys=True
        )
        lines.extend(
            [
                rf"\paragraph{{Case {index}: {_tex(case['family'].title())} "
                rf"with {_tex(case['model'])}.}}",
                rf"\textbf{{Prompt.}} {_tex(case['prompt'])}",
                "",
                rf"\textbf{{Model only.}} \texttt{{{_tex(case['off_answer'])}}} \quad "
                rf"\textbf{{Model + Epicure.}} \texttt{{{_tex(case['on_answer'])}}}",
                "",
                rf"\textbf{{Epicure call.}} \texttt{{{_tex(tool['name'])}}}"
                rf"\allowbreak\texttt{{({_tex(arguments)})}}. "
                rf"The answer key is \texttt{{FINAL\_CHOICE: {_tex(case['expected_choice'])}}}.",
                "",
                rf"\textbf{{Epicure result.}} "
                rf"{{\ttfamily\footnotesize\sloppy {_tex(reference_result)}}}",
                "",
            ]
        )
    return "\n".join(lines), cases


def build(release_path: Path, generated: Path, figures: Path) -> None:
    release = _read_release(release_path)
    rows = _ranked_rows(release)
    display_rows = _score_display_rows(rows)
    completed_rows = [
        row for row in rows if float(row["conditions"]["epicure_off"]["reliability"]) > 0
    ]
    _configure_plots()
    _write(generated / "epicure-native-macros.tex", _macros(release, rows))
    _write(generated / "epicure-native-leaderboard-table.tex", _leaderboard_table(display_rows))
    _write(generated / "epicure-native-route-table.tex", _route_table(release, rows))
    _write(generated / "epicure-native-family-table.tex", _family_table(rows, release))
    cases_tex, cases = _case_studies(release, rows)
    _write(generated / "epicure-native-case-studies.tex", cases_tex)
    _write(
        generated / "epicure-native-case-studies.json",
        json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True),
    )
    _dumbbell(display_rows, figures)
    _score_forest(display_rows, figures)
    _heatmap(display_rows, figures)
    _paired_matrix(release, display_rows, figures)
    _latency_uplift(display_rows, figures)
    _social_summary(release, display_rows, figures)
    manifest = {
        "schema_version": "flavourbench-epicure-native-paper-assets-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "generated_files": sorted(
            [
                f"generated/epicure-native/{path.name}"
                for path in generated.glob("epicure-native-*")
                if path.is_file() and not path.name.startswith("epicure-native-paper-assets-")
            ]
            + [
                f"figures/epicure-native/{path.name}"
                for path in figures.glob("frontier-*")
                if path.is_file()
            ]
        ),
    }
    manifest["artifact_sha256"] = _sha256(manifest)
    _write(
        generated / f"epicure-native-paper-assets-{manifest['artifact_sha256']}.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
    )
    print(
        json.dumps(
            {
                "status": "built",
                "release_artifact_sha256": release["artifact_sha256"],
                "top_model": _short(rows[0]["model_id"]),
                "top_score": rows[0]["epicure_benchmark_score"],
                "bottom_ranked_model": _short(completed_rows[-1]["model_id"]),
                "bottom_ranked_score": completed_rows[-1]["epicure_benchmark_score"],
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("generated/epicure-native"),
    )
    parser.add_argument("--figures", type=Path, default=Path("figures/epicure-native"))
    args = parser.parse_args()
    build(args.release, args.generated, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
