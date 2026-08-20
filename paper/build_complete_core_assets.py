#!/usr/bin/env python3
"""Build paper tables and figures from the final complete-core release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from flavourbench.epicure_selection_common_core_analysis_v1 import load_complete_common_core
from flavourbench.epicure_selection_complete_core_plan_v84 import (
    CORE_FAMILIES,
    selected_task_ids,
    verify_plan,
)
from flavourbench.epicure_selection_complete_core_sources_v1 import source_graph
from flavourbench.epicure_selection_powered_joint_analysis_v1 import combine_panel_data
from flavourbench.epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID

BLUE = "#1769AA"
GOLD = "#E6A11A"
TEAL = "#168C7A"
RED = "#C75450"
CHARCOAL = "#262B33"
LIGHT = "#E9EDF2"
FAMILY_LABELS = {
    "substitution": "Substitution",
    "pairing": "Pairing",
    "constraint": "Constraints",
}
SHORT_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol Pro",
    "openai/gpt-5.6-terra-pro": "GPT-5.6 Terra Pro",
    "openai/gpt-5.6-luna-pro": "GPT-5.6 Luna Pro",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "anthropic/claude-opus-5": "Claude Opus 5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.6-flash": "Gemini 3.6 Flash",
    "x-ai/grok-4.6": "Grok 4.6",
    "moonshotai/kimi-k3": "Kimi K3",
    "qwen/qwen3.8-max": "Qwen 3.8 Max",
    "qwen/qwen3.8-2.4t-a95b": "Qwen3.8 A95B",
    "z-ai/glm-5.2": "GLM 5.2",
    "z-ai/glm-5.3": "GLM 5.3",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro 0813",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "bytedance-seed/seed-2-1-turbo": "Seed 2.1 Turbo",
    "thinkingmachines/inkling": "Inkling",
    "minimax/minimax-m3": "MiniMax M3",
    "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
    "mistralai/mistral-large-2512": "Mistral Large 3",
    "tencent/hy3": "Tencent HY 3",
    "cohere/command-a": "Command A",
    "cohere/command-r-plus-08-2024": "Command R+",
    "meta/muse-spark-1.2": "Muse Spark 1.2",
    "meta/muse-glimmer-30b": "Muse Glimmer 30B",
}


class CompleteCoreAssetError(RuntimeError):
    """The final paper inputs or generated assets are inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompleteCoreAssetError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompleteCoreAssetError(f"input is not a JSON object: {path}")
    return value


def _read_release(path: Path) -> dict[str, Any]:
    release = _load(path)
    payload = dict(release)
    recorded = str(payload.pop("artifact_sha256", ""))
    analysis = release.get("analysis") or {}
    models = analysis.get("models") or []
    pairs = analysis.get("pairwise_comparisons") or []
    if not (
        recorded == _sha256(payload)
        and release.get("schema_version") == "flavourbench-complete-common-core-release-v1"
        and release.get("status") == "final_complete_common_core"
        and len(models) == 27
        and len(pairs) == 351
        and all((row.get("coverage") or {}).get("valid_scored") == 534 for row in models)
    ):
        raise CompleteCoreAssetError("release is not the final 27-model complete core")
    return release


def _short(model_id: str) -> str:
    return SHORT_NAMES.get(model_id, model_id.rsplit("/", 1)[-1])


def _tex(value: object) -> str:
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
    return "".join(replacements.get(character, character) for character in str(value))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _ranked(release: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        release["analysis"]["models"],
        key=lambda row: (int(row["point_estimate_rank"]), str(row["model_id"])),
    )


def _selected_tasks(
    plan: Mapping[str, Any], taskset_1: Mapping[str, Any], taskset_2: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    left, right = selected_task_ids(plan)
    by_id = {str(task["task_id"]): task for task in [*taskset_1["tasks"], *taskset_2["tasks"]]}
    return [by_id[task_id] for task_id in (*left, *right)]


def _macros(
    release: Mapping[str, Any], plan: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> str:
    models = _ranked(release)
    top = models[0]
    analysis = release["analysis"]
    replication = analysis["panel_replication"]
    values = {
        "FBModels": len(models),
        "FBTasks": len(tasks),
        "FBTasksPerFamily": len(tasks) // len(CORE_FAMILIES),
        "FBTasksPerPanelFamily": len(tasks) // (2 * len(CORE_FAMILIES)),
        "FBPanelCount": 2,
        "FBFamilies": len(CORE_FAMILIES),
        "FBUniqueAnchors": len({str(task["anchor_ingredient"]) for task in tasks}),
        "FBSelectionsPerTask": 56,
        "FBPrefrozenScores": len(tasks) * 56,
        "FBPrimaryCells": len(models) * len(tasks),
        "FBPairs": len(analysis["pairwise_comparisons"]),
        "FBSignificantPairs": analysis["resolved_pair_count"],
        "FBBootstrapResamples": analysis["inference"]["bootstrap_resamples"],
        "FBPermutationResamples": analysis["inference"]["permutation_resamples"],
        "FBTopModel": _short(str(top["model_id"])),
        "FBTopScore": f"{float(top['flavourbench_score']):.1f}",
        "FBTopCILow": f"{float(top['score_simultaneous_95_ci'][0]):.1f}",
        "FBTopCIHigh": f"{float(top['score_simultaneous_95_ci'][1]):.1f}",
        "FBTopGroup": top["statistical_rank_group"],
        "FBDefinitiveTop": "yes" if analysis["definitive_top_model_id"] else "no",
        "FBPanelPearson": f"{float(replication['score_pearson']):.2f}",
        "FBPanelSpearman": f"{float(replication['rank_spearman']):.2f}",
        "FBIndependentClusters": analysis["inference"]["independent_cluster_count"],
        "FBCompleteCells": sum(row["coverage"]["valid_scored"] for row in models),
        "FBPlanDigest": str(plan["artifact_sha256"])[:12],
    }
    return "\n".join(f"\\newcommand{{\\{key}}}{{{_tex(value)}}}" for key, value in values.items())


def _leaderboard_table(release: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}r l r c c c@{}}",
        r"\toprule",
        r"Rank & Model & FB Score & simultaneous 95\% CI & rank 95\% CI & group \\",
        r"\midrule",
    ]
    for row in _ranked(release):
        score_ci = row["score_simultaneous_95_ci"]
        rank_ci = row["bootstrap_rank_95_interval"]
        lines.append(
            f"{row['point_estimate_rank']} & {_tex(_short(str(row['model_id'])))} & "
            f"{float(row['flavourbench_score']):.1f} & "
            f"[{float(score_ci[0]):.1f}, {float(score_ci[1]):.1f}] & "
            f"[{rank_ci[0]}, {rank_ci[1]}] & {row['statistical_rank_group']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _family_table(release: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r r@{}}",
        r"\toprule",
        r"Model & Substitution & Pairing & Constraints \\",
        r"\midrule",
    ]
    for row in _ranked(release):
        scores = row["family_scores"]
        lines.append(
            f"{_tex(_short(str(row['model_id'])))} & "
            + " & ".join(f"{float(scores[family]):.1f}" for family in CORE_FAMILIES)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _route_table(
    release: Mapping[str, Any], panel_1_plan: Mapping[str, Any], panel_2_plan: Mapping[str, Any]
) -> str:
    left = {str(row["model_id"]): row for row in panel_1_plan["roster"]["models"]}
    right = {str(row["model_id"]): row for row in panel_2_plan["roster"]["models"]}
    lines = [
        r"\begin{tabular}{@{}l l l l@{}}",
        r"\toprule",
        r"Model & Backend & Panel 1 route & Panel 2 route \\",
        r"\midrule",
    ]
    for row in _ranked(release):
        model_id = str(row["model_id"])
        lines.append(
            f"{_tex(_short(model_id))} & {_tex(right[model_id]['execution_backend'])} & "
            f"{_tex(left[model_id]['provider_tag'])} & {_tex(right[model_id]['provider_tag'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _task_table(tasks: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r r@{}}",
        r"\toprule",
        r"Family & Exact chance & Top gap & Distinct scores \\",
        r"\midrule",
    ]
    for family in CORE_FAMILIES:
        rows = [task for task in tasks if task["family"] == family]
        chance = np.mean([float(task["chance_score_bps"]) / 100 for task in rows])
        gap = np.median([float(task["optimal_margin_bps"]) / 100 for task in rows])
        distinct = np.median([len(set(task["selection_scores_bps"].values())) for task in rows])
        lines.append(f"{FAMILY_LABELS[family]} & {chance:.1f} & {gap:.1f} & {distinct:.0f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#A5ADB8",
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


def _save(figure: mpl.figure.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.pdf", facecolor="white")
    figure.savefig(directory / f"{name}.png", dpi=240, facecolor="white")
    plt.close(figure)


def _leaderboard_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = list(reversed(_ranked(release)))
    scores = np.asarray([row["flavourbench_score"] for row in rows])
    intervals = np.asarray([row["score_simultaneous_95_ci"] for row in rows])
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(7.7, max(7.0, 1.1 + 0.27 * len(rows))))
    axis.hlines(y, intervals[:, 0], intervals[:, 1], color=LIGHT, linewidth=6, zorder=1)
    axis.scatter(scores, y, c=BLUE, s=38, edgecolor="white", linewidth=0.7, zorder=2)
    for index, row in enumerate(rows):
        axis.text(
            intervals[index, 1] + 0.3,
            index,
            f"{scores[index]:.1f}  G{row['statistical_rank_group']}",
            va="center",
            fontsize=7.2,
        )
    chance = float(np.mean([row["chance_comparison"]["exact_chance_score"] for row in rows]))
    axis.axvline(chance, color=GOLD, linestyle="--", linewidth=1.2, label="Mean exact chance")
    axis.set_yticks(y, [_short(str(row["model_id"])) for row in rows])
    axis.set_xlabel("FlavourBench Score (0–100)")
    axis.set_title("27 frontier models on the identical 534-task core", loc="left")
    axis.grid(axis="x", color="#EFF1F4", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    _save(figure, output, "complete-core-leaderboard-forest")


def _family_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = _ranked(release)
    matrix = np.asarray(
        [[row["family_scores"][family] for family in CORE_FAMILIES] for row in rows]
    )
    figure, axis = plt.subplots(figsize=(6.5, max(7.0, 1.1 + 0.27 * len(rows))))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 68 else CHARCOAL,
            )
    axis.set_xticks(
        range(len(CORE_FAMILIES)),
        [FAMILY_LABELS[family] for family in CORE_FAMILIES],
    )
    axis.set_yticks(range(len(rows)), [_short(str(row["model_id"])) for row in rows])
    axis.set_title("Performance decomposes by culinary decision family", loc="left")
    figure.colorbar(image, ax=axis, label="Mean Epicure score", fraction=0.04, pad=0.03)
    _save(figure, output, "complete-core-family-heatmap")


def _pairwise_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = _ranked(release)
    index = {str(row["model_id"]): position for position, row in enumerate(rows)}
    matrix = np.zeros((len(rows), len(rows)), dtype=int)
    for comparison in release["analysis"]["pairwise_comparisons"]:
        left = index[str(comparison["left_model_id"])]
        right = index[str(comparison["right_model_id"])]
        if not comparison["holm_significant"]:
            continue
        sign = 1 if float(comparison["mean_difference"]) > 0 else -1
        matrix[left, right] = sign
        matrix[right, left] = -sign
    figure, axis = plt.subplots(figsize=(7.5, 7.1))
    axis.imshow(matrix, cmap=ListedColormap([RED, "#E6E8EC", TEAL]), vmin=-1, vmax=1)
    labels = [_short(str(row["model_id"])) for row in rows]
    axis.set_xticks(range(len(rows)), labels, rotation=90, fontsize=6.2)
    axis.set_yticks(range(len(rows)), labels, fontsize=6.2)
    axis.set_xlabel("Column model")
    axis.set_ylabel("Row model")
    axis.set_title("Holm-controlled paired comparisons", loc="left")
    axis.text(
        0,
        -1.9,
        "Teal: row higher   Red: row lower   Grey: unresolved at familywise α=.05",
        fontsize=7.5,
    )
    _save(figure, output, "complete-core-pairwise-matrix")


def _panel_figure(release: Mapping[str, Any], output: Path) -> None:
    diagnostic = release["analysis"]["panel_replication"]
    rows = diagnostic["models"]
    left = np.asarray([float(row["panel_1"]) for row in rows])
    right = np.asarray([float(row["panel_2"]) for row in rows])
    lower = float(min(left.min(), right.min())) - 1.5
    upper = float(max(left.max(), right.max())) + 1.5
    figure, axis = plt.subplots(figsize=(6.3, 5.5))
    axis.plot([lower, upper], [lower, upper], color=LIGHT, linewidth=1.4)
    axis.scatter(left, right, s=46, c=TEAL, edgecolor="white", linewidth=0.7)
    for row, x_value, y_value in zip(rows, left, right, strict=True):
        axis.annotate(
            _short(str(row["model_id"])),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.2,
        )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Panel 1 FlavourBench Score")
    axis.set_ylabel("Panel 2 FlavourBench Score")
    axis.set_title("Independent task panels test ranking stability", loc="left")
    axis.text(
        0.01,
        0.99,
        f"Pearson $r$={diagnostic['score_pearson']:.2f}   "
        f"Spearman $\\rho$={diagnostic['rank_spearman']:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axis.grid(color="#EFF1F4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output, "complete-core-panel-replication")


def _rank_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = list(reversed(_ranked(release)))
    point = np.asarray([row["point_estimate_rank"] for row in rows])
    intervals = np.asarray([row["bootstrap_rank_95_interval"] for row in rows])
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(7.2, max(7.0, 1.1 + 0.27 * len(rows))))
    axis.hlines(y, intervals[:, 0], intervals[:, 1], color=LIGHT, linewidth=6)
    axis.scatter(point, y, c=TEAL, s=36, edgecolor="white", linewidth=0.7)
    axis.set_yticks(y, [_short(str(row["model_id"])) for row in rows])
    axis.set_xlim(len(rows) + 0.5, 0.5)
    axis.set_xticks(range(1, len(rows) + 1, 2))
    axis.set_xlabel("Rank (lower is better)")
    axis.set_title("Rank uncertainty remains visible behind point estimates", loc="left")
    axis.grid(axis="x", color="#EFF1F4", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    _save(figure, output, "complete-core-rank-intervals")


def _case_studies(
    root: Path,
    release: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    graph = source_graph(root)
    tasks_1, tasks_2 = selected_task_ids(plan)
    allowed = {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})}
    panel_1 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_1_plan,
        taskset=graph.panel_1_taskset,
        repeat_panel=graph.panel_1_repeat,
        task_ids=tasks_1,
        model_sources=graph.panel_1_model_sources,
        allowed_source_roster_differences=allowed,
    )
    panel_2 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_2_plan,
        taskset=graph.panel_2_taskset,
        repeat_panel=graph.panel_2_repeat,
        task_ids=tasks_2,
        model_sources=graph.panel_2_model_sources,
        allowed_source_roster_differences=allowed,
    )
    data = combine_panel_data(panel_1, panel_2, panel="joint_primary_complete_core")
    task_by_id = {
        str(task["task_id"]): task
        for task in [*graph.panel_1_taskset["tasks"], *graph.panel_2_taskset["tasks"]]
    }
    ranked = _ranked(release)
    high_id = str(ranked[0]["model_id"])
    low_id = str(ranked[-1]["model_id"])
    high = data.model_ids.index(high_id)
    low = data.model_ids.index(low_id)
    cases: list[dict[str, Any]] = []
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}l X X X@{}}",
        r"\toprule",
        r"Family & Prompt & Higher-ranked selection & Lower-ranked selection \\",
        r"\midrule",
    ]
    for family in CORE_FAMILIES:
        candidates = [index for index, value in enumerate(data.families) if value == family]
        selected_index = max(
            candidates, key=lambda index: data.scores[high, index] - data.scores[low, index]
        )
        task = task_by_id[data.task_ids[selected_index]]
        high_selection = str(data.selections[high][selected_index])
        low_selection = str(data.selections[low][selected_index])
        high_ingredients = [
            str(task["choices"][label]).replace("_", " ") for label in high_selection
        ]
        low_ingredients = [str(task["choices"][label]).replace("_", " ") for label in low_selection]
        question = str(task["prompt"]).split("\n\nCandidates:", 1)[0].splitlines()[-1]
        case = {
            "task_id": task["task_id"],
            "family": family,
            "prompt": task["prompt"],
            "choices": task["choices"],
            "optimal_selection": task["optimal_selection"],
            "higher_ranked": {
                "model_id": high_id,
                "selection": high_selection,
                "ingredients": high_ingredients,
                "score": float(data.scores[high, selected_index]),
            },
            "lower_ranked": {
                "model_id": low_id,
                "selection": low_selection,
                "ingredients": low_ingredients,
                "score": float(data.scores[low, selected_index]),
            },
            "selection_rule": "largest observed score gap within family; illustrative only",
        }
        cases.append(case)
        lines.append(
            f"{FAMILY_LABELS[family]} & {_tex(question)} & "
            f"{_tex(', '.join(high_ingredients))} ({case['higher_ranked']['score']:.0f}) & "
            f"{_tex(', '.join(low_ingredients))} ({case['lower_ranked']['score']:.0f}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    document = {
        "schema_version": "flavourbench-complete-core-paper-examples-v1",
        "higher_ranked_model_id": high_id,
        "lower_ranked_model_id": low_id,
        "cases": cases,
    }
    document["artifact_sha256"] = _sha256(document)
    return document, "\n".join(lines)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--taskset-secondary", type=Path, required=True)
    parser.add_argument("--panel-1-plan", type=Path, required=True)
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = _read_release(args.release)
    plan = _load(args.plan)
    taskset_1 = _load(args.taskset)
    taskset_2 = _load(args.taskset_secondary)
    panel_1_plan = _load(args.panel_1_plan)
    panel_2_plan = _load(args.panel_2_plan)
    if not verify_plan(plan) or release["inputs"]["complete_core_plan"] != {
        "semantic_sha256": plan["artifact_sha256"],
        "physical_sha256": _sha256_file(args.plan),
    }:
        raise CompleteCoreAssetError("release and complete-core plan differ")
    tasks = _selected_tasks(plan, taskset_1, taskset_2)
    args.generated_directory.mkdir(parents=True, exist_ok=True)
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    _write(args.generated_directory / "complete-core-macros.tex", _macros(release, plan, tasks))
    _write(
        args.generated_directory / "complete-core-leaderboard-table.tex",
        _leaderboard_table(release),
    )
    _write(args.generated_directory / "complete-core-family-table.tex", _family_table(release))
    _write(
        args.generated_directory / "complete-core-route-table.tex",
        _route_table(release, panel_1_plan, panel_2_plan),
    )
    _write(args.generated_directory / "complete-core-task-table.tex", _task_table(tasks))
    examples, example_table = _case_studies(root, release, plan)
    _write(args.generated_directory / "complete-core-examples-table.tex", example_table)
    _write(
        args.generated_directory / "complete-core-examples.json",
        json.dumps(examples, ensure_ascii=False, indent=2, sort_keys=True),
    )
    _configure_plots()
    _leaderboard_figure(release, args.figure_directory)
    _family_figure(release, args.figure_directory)
    _pairwise_figure(release, args.figure_directory)
    _panel_figure(release, args.figure_directory)
    _rank_figure(release, args.figure_directory)
    files = sorted(
        [path for path in args.generated_directory.glob("complete-core-*") if path.is_file()]
        + [path for path in args.figure_directory.glob("complete-core-*") if path.is_file()]
    )
    inventory = {
        "schema_version": "flavourbench-complete-core-paper-assets-v1",
        "release_semantic_sha256": release["artifact_sha256"],
        "release_physical_sha256": _sha256_file(args.release),
        "plan_semantic_sha256": plan["artifact_sha256"],
        "plan_physical_sha256": _sha256_file(args.plan),
        "files": [
            {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    inventory["artifact_sha256"] = _sha256(inventory)
    destination = args.generated_directory / (
        f"complete-core-paper-assets-{inventory['artifact_sha256']}.json"
    )
    _write(destination, json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True))
    print(destination)


if __name__ == "__main__":
    run()
