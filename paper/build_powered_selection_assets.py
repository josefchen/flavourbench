#!/usr/bin/env python3
"""Build final paper tables and figures from the powered selection release."""

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

FAMILIES = ("substitution", "pairing", "constraint", "cultural_composition")
FAMILY_LABELS = ("Substitution", "Pairing", "Constraints", "Regional composition")
BLUE = "#1769AA"
GOLD = "#E6A11A"
TEAL = "#168C7A"
RED = "#C75450"
CHARCOAL = "#262B33"
LIGHT = "#E9EDF2"
SHORT_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol Pro",
    "openai/gpt-5.6-terra-pro": "GPT-5.6 Terra Pro",
    "openai/gpt-5.6-luna-pro": "GPT-5.6 Luna Pro",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
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
    "nvidia/nemotron-3.5-lightning": "Nemotron 3.5 Lightning",
    "mistralai/mistral-large-2512": "Mistral Large 3",
    "tencent/hy3": "Tencent HY 3",
    "cohere/command-a": "Command A",
    "cohere/command-r-plus-08-2024": "Command R+",
    "cohere/command-a-plus-05-2026": "Command A Plus",
    "cohere/command-a-reasoning-08-2025": "Command A Reasoning",
    "meta/muse-spark-1.2": "Muse Spark 1.2",
    "meta/muse-glimmer-30b": "Muse Glimmer 30B",
    "x-ai/grok-4.6": "Grok 4.6",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro 0813",
    "qwen/qwen3.8-2.4t-a95b": "Qwen3.8 A95B",
    "bytedance-seed/seed-2-1-turbo": "Seed 2.1 Turbo",
    "thinkingmachines/inkling": "Inkling",
}


class PoweredAssetError(RuntimeError):
    """The final release is incomplete or inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PoweredAssetError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PoweredAssetError(f"input is not a JSON object: {path}")
    return value


def _read_release(path: Path) -> dict[str, Any]:
    release = _load(path)
    payload = dict(release)
    recorded = str(payload.pop("artifact_sha256", ""))
    analysis = release.get("analysis") or {}
    models = analysis.get("models") or []
    pairwise = analysis.get("pairwise_comparisons") or []
    repeats = analysis.get("repeatability") or []
    model_count = len(models)
    if (
        recorded != _sha256(payload)
        or release.get("schema_version") != "flavourbench-selection-powered-release-v1"
        or release.get("status") != "final_complete"
        or model_count < 2
        or len(pairwise) != model_count * (model_count - 1) // 2
        or len(repeats) != model_count
    ):
        raise PoweredAssetError("release is not the complete powered statistical release")
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


def _save(figure: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    plt.close(figure)


def _ranked_models(release: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = list(release["analysis"]["models"])
    return sorted(
        models,
        key=lambda row: (
            row["point_estimate_rank"] is None,
            row["point_estimate_rank"] or 10_000,
            str(row["model_id"]),
        ),
    )


def _macros(release: Mapping[str, Any], taskset: Mapping[str, Any]) -> str:
    models = _ranked_models(release)
    top = models[0]
    eligible = [row for row in models if row["eligible"]]
    eligible_ids = {str(row["model_id"]) for row in eligible}
    eligible_pairs = [
        row
        for row in release["analysis"]["pairwise_comparisons"]
        if str(row["left_model_id"]) in eligible_ids and str(row["right_model_id"]) in eligible_ids
    ]
    significant = sum(
        bool(row["holm_significant"]) for row in release["analysis"]["pairwise_comparisons"]
    )
    significant_eligible = sum(bool(row["holm_significant"]) for row in eligible_pairs)
    above_chance = sum(
        bool(row["chance_comparison"]["holm_significant_above_chance"]) for row in models
    )
    above_chance_eligible = sum(
        bool(row["chance_comparison"]["holm_significant_above_chance"]) for row in eligible
    )
    total_complete = sum(int(row["availability"]["completed"]) for row in models)
    total_parseable = sum(int(row["availability"]["parseable"]) for row in models)
    repeat = top["repeatability"]
    top_ci = top["score_simultaneous_95_ci"]
    definitive = release["analysis"]["definitive_top_model_id"]
    values = {
        "FBModels": len(models),
        "FBTasks": taskset["counts"]["tasks"],
        "FBFamilies": len(FAMILIES),
        "FBSelectionsPerTask": taskset["counts"]["scored_combinations_per_task"],
        "FBPrefrozenScores": taskset["counts"]["total_prefrozen_selection_scores"],
        "FBPrimaryCells": release["inputs"]["primary_responses"]["count"],
        "FBRepeatCells": release["inputs"]["repeat_responses"]["count"],
        "FBTotalCells": (
            release["inputs"]["primary_responses"]["count"]
            + release["inputs"]["repeat_responses"]["count"]
        ),
        "FBEligibleModels": len(eligible),
        "FBCompletedCells": total_complete,
        "FBParseableCells": total_parseable,
        "FBTopModel": _short(str(top["model_id"])),
        "FBTopScore": f"{float(top['flavourbench_score']):.1f}",
        "FBTopCILow": f"{float(top_ci[0]):.1f}",
        "FBTopCIHigh": f"{float(top_ci[1]):.1f}",
        "FBTopGroup": top["statistical_rank_group"],
        "FBTopRepeat": f"{float(repeat['mean_ingredient_set_jaccard']):.2f}",
        "FBSignificantPairs": significant,
        "FBPairs": len(release["analysis"]["pairwise_comparisons"]),
        "FBSignificantEligiblePairs": significant_eligible,
        "FBEligiblePairs": len(eligible_pairs),
        "FBAboveChance": above_chance,
        "FBAboveChanceEligible": above_chance_eligible,
        "FBDefinitiveTop": "yes" if definitive else "no",
        "FBBootstrapResamples": release["analysis"]["inference"]["bootstrap_resamples"],
        "FBPermutationResamples": release["analysis"]["inference"]["permutation_resamples"],
    }
    return "\n".join(f"\\newcommand{{\\{key}}}{{{_tex(value)}}}" for key, value in values.items())


def _leaderboard_table(release: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}r l r c c r r@{}}",
        r"\toprule",
        r"Rank & Model & FB Score & simultaneous 95\% CI & group & completed & repeat \\",
        r"\midrule",
    ]
    for row in _ranked_models(release):
        ci = row["score_simultaneous_95_ci"]
        rank = row["point_estimate_rank"] if row["eligible"] else "DNF"
        lines.append(
            f"{rank} & {_tex(_short(str(row['model_id'])))} & "
            f"{float(row['flavourbench_score']):.1f} & "
            f"[{float(ci[0]):.1f}, {float(ci[1]):.1f}] & "
            f"{row['statistical_rank_group'] or '--'} & "
            f"{row['availability']['completed']}/640 & "
            f"{float(row['repeatability']['mean_ingredient_set_jaccard']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _route_table(release: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    by_id = {str(row["model"]["id"]): row for row in manifest["models"]}
    lines = [
        r"\begin{tabular}{@{}l l l@{}}",
        r"\toprule",
        r"Model & Backend & Frozen route \\",
        r"\midrule",
    ]
    for row in _ranked_models(release):
        entry = by_id[str(row["model_id"])]
        lines.append(
            f"{_tex(_short(str(row['model_id'])))} & "
            f"{_tex(entry['execution_route']['selected_backend'])} & "
            f"{_tex(entry['endpoint']['tag'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _family_table(release: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r r r@{}}",
        r"\toprule",
        r"Model & Substitution & Pairing & Constraints & Regional \\",
        r"\midrule",
    ]
    for row in _ranked_models(release):
        scores = row["family_scores"]
        values = [float(scores[family]) for family in FAMILIES]
        lines.append(
            f"{_tex(_short(str(row['model_id'])))} & "
            + " & ".join(f"{value:.1f}" for value in values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _task_diagnostics_table(taskset: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}l r r r@{}}",
        r"\toprule",
        r"Family & Chance & Top gap & Unique scores \\",
        r"\midrule",
    ]
    for family, label in zip(FAMILIES, FAMILY_LABELS, strict=True):
        tasks = [task for task in taskset["tasks"] if task["family"] == family]
        chance = np.mean([float(task["chance_score_bps"]) / 100 for task in tasks])
        margin = np.median([float(task["optimal_margin_bps"]) / 100 for task in tasks])
        unique = np.median([len(set(task["selection_scores_bps"].values())) for task in tasks])
        lines.append(f"{_tex(label)} & {chance:.1f} & {margin:.1f} & {unique:.0f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _response_map(
    run_directory: Path,
    expected: int,
    *,
    expected_model_ids: set[str] | None = None,
    recovery_run_directory: Path | None = None,
    recovery_model_id: str | None = None,
    response_sources: Mapping[str, Path] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    paths = sorted((run_directory / "responses" / "primary").glob("*/response-*.json"))
    if (recovery_run_directory is None) != (recovery_model_id is None):
        raise PoweredAssetError("recovery response source is incomplete")
    replacements = dict(response_sources or {})
    if recovery_run_directory is not None:
        if recovery_model_id in replacements:
            raise PoweredAssetError("response model has two replacement sources")
        replacements[recovery_model_id] = recovery_run_directory
    final_models = set(expected_model_ids or ())
    if final_models and not set(replacements).issubset(final_models):
        raise PoweredAssetError("replacement response model is outside the final roster")
    base_paths = []
    for path in paths:
        row = _load(path)
        model_id = str(row.get("model_id") or "")
        if model_id not in replacements and (not final_models or model_id in final_models):
            base_paths.append(path)
    replacement_paths: list[Path] = []
    for model_id, directory in sorted(replacements.items()):
        model_paths = [
            path
            for path in sorted((directory / "responses" / "primary").glob("*/response-*.json"))
            if _load(path).get("model_id") == model_id
        ]
        if not model_paths:
            raise PoweredAssetError(f"replacement response source is empty for {model_id}")
        replacement_paths.extend(model_paths)
    paths = base_paths + replacement_paths
    if len(paths) != expected:
        raise PoweredAssetError(
            f"primary response panel is incomplete: observed {len(paths)}, expected {expected}"
        )
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        row = _load(path)
        payload = dict(row)
        recorded = str(payload.pop("artifact_sha256", ""))
        key = (str(row.get("model_id") or ""), str(row.get("task_id") or ""))
        if recorded != _sha256(payload) or key in output:
            raise PoweredAssetError("primary response integrity failed")
        output[key] = row
    return output


def _question(prompt: str) -> str:
    before_candidates = prompt.split("\n\nCandidates:", 1)[0]
    lines = [line.strip() for line in before_candidates.splitlines() if line.strip()]
    return lines[-1]


def _selection_ingredients(task: Mapping[str, Any], selection: str | None) -> list[str]:
    if not selection:
        return []
    return [str(task["choices"][label]).replace("_", " ") for label in selection]


def _case_studies(
    release: Mapping[str, Any],
    taskset: Mapping[str, Any],
    run_directory: Path,
    *,
    recovery_run_directory: Path | None = None,
    recovery_model_id: str | None = None,
    response_sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], str]:
    rows = [row for row in _ranked_models(release) if row["eligible"]]
    top = rows[0]
    lower = rows[-1]
    responses = _response_map(
        run_directory,
        int(release["inputs"]["primary_responses"]["count"]),
        expected_model_ids={str(row["model_id"]) for row in release["analysis"]["models"]},
        recovery_run_directory=recovery_run_directory,
        recovery_model_id=recovery_model_id,
        response_sources=response_sources,
    )
    selected_tasks = [
        sorted(
            (task for task in taskset["tasks"] if task["family"] == family),
            key=lambda task: str(task["task_id"]),
        )[0]
        for family in FAMILIES
    ]
    cases: list[dict[str, Any]] = []
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}l X X X@{}}",
        r"\toprule",
        r"Family & Prompt & Higher-ranked response & Lower-ranked response \\",
        r"\midrule",
    ]
    for task in selected_tasks:
        top_response = responses[(str(top["model_id"]), str(task["task_id"]))]
        lower_response = responses[(str(lower["model_id"]), str(task["task_id"]))]
        optimum = str(task["optimal_selection"])
        case = {
            "task_id": task["task_id"],
            "family": task["family"],
            "prompt": task["prompt"],
            "choices": task["choices"],
            "optimal_selection": optimum,
            "optimal_ingredients": _selection_ingredients(task, optimum),
            "higher_ranked": {
                "model_id": top["model_id"],
                "answer_markdown": (top_response.get("generation") or {}).get("answer_markdown"),
                "observed_selection": top_response["scoring"]["observed_selection"],
                "selected_ingredients": _selection_ingredients(
                    task, top_response["scoring"]["observed_selection"]
                ),
                "score": top_response["scoring"]["score"],
            },
            "lower_ranked": {
                "model_id": lower["model_id"],
                "answer_markdown": (lower_response.get("generation") or {}).get("answer_markdown"),
                "observed_selection": lower_response["scoring"]["observed_selection"],
                "selected_ingredients": _selection_ingredients(
                    task, lower_response["scoring"]["observed_selection"]
                ),
                "score": lower_response["scoring"]["score"],
            },
        }
        cases.append(case)
        top_text = ", ".join(case["higher_ranked"]["selected_ingredients"]) or "invalid"
        lower_text = ", ".join(case["lower_ranked"]["selected_ingredients"]) or "invalid"
        lines.append(
            f"{_tex(FAMILY_LABELS[FAMILIES.index(str(task['family']))])} & "
            f"{_tex(_question(str(task['prompt'])))} & "
            f"{_tex(top_text)} ({float(case['higher_ranked']['score']):.0f}) & "
            f"{_tex(lower_text)} ({float(case['lower_ranked']['score']):.0f}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    document = {
        "schema_version": "flavourbench-powered-paper-case-studies-v1",
        "selection": "lexicographically first frozen task in each family",
        "higher_ranked_model_id": top["model_id"],
        "lower_ranked_model_id": lower["model_id"],
        "cases": cases,
    }
    document["artifact_sha256"] = _sha256(document)
    return document, "\n".join(lines)


def _leaderboard_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = list(reversed(_ranked_models(release)))
    scores = np.asarray([row["flavourbench_score"] for row in rows])
    intervals = np.asarray([row["score_simultaneous_95_ci"] for row in rows])
    labels = [_short(str(row["model_id"])) for row in rows]
    colors = [BLUE if row["eligible"] else "#9AA1AA" for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, max(7.0, 1.15 + 0.26 * len(rows))))
    y = np.arange(len(rows))
    axis.hlines(y, intervals[:, 0], intervals[:, 1], color=LIGHT, linewidth=6, zorder=1)
    axis.scatter(scores, y, c=colors, s=38, edgecolor="white", linewidth=0.7, zorder=2)
    for index, row in enumerate(rows):
        axis.text(
            intervals[index, 1] + 0.35,
            index,
            f"{scores[index]:.1f}  G{row['statistical_rank_group'] or '–'}",
            va="center",
            fontsize=7.5,
        )
    chance = float(rows[0]["chance_comparison"]["exact_chance_score"])
    axis.axvline(chance, color=GOLD, linestyle="--", linewidth=1.2, label="Exact chance")
    axis.set_yticks(y, labels)
    axis.set_xlabel("FlavourBench Score (0–100)")
    axis.set_title("Frontier models under executable culinary ground truth", loc="left")
    axis.grid(axis="x", color="#EFF1F4", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    _save(figure, output / "powered-leaderboard-forest")


def _family_heatmap(release: Mapping[str, Any], output: Path) -> None:
    rows = _ranked_models(release)
    matrix = np.asarray(
        [[row["family_scores"][family] for family in FAMILIES] for row in rows], dtype=float
    )
    figure, axis = plt.subplots(figsize=(6.6, max(7.0, 1.15 + 0.26 * len(rows))))
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
    axis.set_xticks(range(len(FAMILIES)), FAMILY_LABELS, rotation=25, ha="right")
    axis.set_yticks(range(len(rows)), [_short(str(row["model_id"])) for row in rows])
    axis.set_title("Culinary strengths differ by task family", loc="left")
    figure.colorbar(image, ax=axis, label="Mean Epicure score", fraction=0.04, pad=0.03)
    _save(figure, output / "powered-family-heatmap")


def _pairwise_matrix(release: Mapping[str, Any], output: Path) -> None:
    rows = _ranked_models(release)
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
    figure, axis = plt.subplots(figsize=(7.4, 7.0))
    cmap = ListedColormap([RED, "#E6E8EC", TEAL])
    axis.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    labels = [_short(str(row["model_id"])) for row in rows]
    axis.set_xticks(range(len(rows)), labels, rotation=90, fontsize=6.5)
    axis.set_yticks(range(len(rows)), labels, fontsize=6.5)
    axis.set_xlabel("Column model")
    axis.set_ylabel("Row model")
    axis.set_title("Holm-controlled paired comparisons", loc="left")
    axis.text(
        0,
        -1.9,
        "Teal: row higher   Red: row lower   Grey: unresolved at familywise α=.05",
        fontsize=7.5,
    )
    _save(figure, output / "powered-pairwise-matrix")


def _repeatability_figure(release: Mapping[str, Any], output: Path) -> None:
    rows = _ranked_models(release)
    x = np.asarray([row["flavourbench_score"] for row in rows])
    y = np.asarray([row["repeatability"]["mean_ingredient_set_jaccard"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.scatter(x, y, s=42, c=BLUE, edgecolor="white", linewidth=0.7)
    for row, x_value, y_value in zip(rows, x, y, strict=True):
        axis.annotate(
            _short(str(row["model_id"])),
            (x_value, y_value),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axis.axhline(0.8, color=GOLD, linestyle="--", linewidth=1.1, label="Repeat floor")
    axis.set_xlabel("FlavourBench Score")
    axis.set_ylabel("Ingredient-set Jaccard agreement")
    axis.set_ylim(0, 1.03)
    axis.set_title("Capability and answer stability are separate measurements", loc="left")
    axis.grid(color="#EFF1F4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    _save(figure, output / "powered-repeatability-scatter")


def _inventory(output_directory: Path, generated_directory: Path) -> dict[str, Any]:
    files = sorted(
        [path for path in output_directory.glob("powered-*.*") if path.is_file()]
        + [
            path
            for path in generated_directory.glob("powered-*.*")
            if path.is_file() and not path.name.startswith("powered-paper-assets-")
        ]
    )
    return {
        "schema_version": "flavourbench-powered-paper-assets-v1",
        "files": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--recovery-run-directory", type=Path)
    parser.add_argument("--recovery-model-id")
    parser.add_argument(
        "--response-source",
        action="append",
        default=[],
        metavar="MODEL_ID=RUN_DIRECTORY",
    )
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--generated-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    release = _read_release(args.release)
    taskset = _load(args.taskset)
    manifest = _load(args.manifest)
    response_sources: dict[str, Path] = {}
    for spec in args.response_source:
        model_id, separator, directory = spec.partition("=")
        if not separator or not model_id or not directory or model_id in response_sources:
            raise PoweredAssetError(f"invalid or duplicate response source: {spec}")
        response_sources[model_id] = Path(directory)
    if release["inputs"]["taskset"]["semantic_sha256"] != taskset.get("artifact_sha256"):
        raise PoweredAssetError("release and taskset differ")
    if release["inputs"]["plan"]["semantic_sha256"] != release["analysis"].get("plan_sha256"):
        raise PoweredAssetError("release analysis plan binding differs")
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    args.generated_directory.mkdir(parents=True, exist_ok=True)
    _configure_plots()
    _write(args.generated_directory / "powered-macros.tex", _macros(release, taskset))
    _write(
        args.generated_directory / "powered-leaderboard-table.tex",
        _leaderboard_table(release),
    )
    _write(args.generated_directory / "powered-route-table.tex", _route_table(release, manifest))
    _write(args.generated_directory / "powered-family-table.tex", _family_table(release))
    _write(
        args.generated_directory / "powered-task-diagnostics-table.tex",
        _task_diagnostics_table(taskset),
    )
    cases, case_table = _case_studies(
        release,
        taskset,
        args.run_directory,
        recovery_run_directory=args.recovery_run_directory,
        recovery_model_id=args.recovery_model_id,
        response_sources=response_sources,
    )
    _write(args.generated_directory / "powered-case-studies-table.tex", case_table)
    _write(
        args.generated_directory / "powered-case-studies.json",
        json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True),
    )
    _leaderboard_figure(release, args.figure_directory)
    _family_heatmap(release, args.figure_directory)
    _pairwise_matrix(release, args.figure_directory)
    _repeatability_figure(release, args.figure_directory)
    inventory = _inventory(args.figure_directory, args.generated_directory)
    inventory.update(
        {
            "release_semantic_sha256": release["artifact_sha256"],
            "release_physical_sha256": _sha256_file(args.release),
            "taskset_semantic_sha256": taskset["artifact_sha256"],
            "taskset_physical_sha256": _sha256_file(args.taskset),
            "manifest_semantic_sha256": manifest["content_address"]["digest"],
            "manifest_physical_sha256": _sha256_file(args.manifest),
        }
    )
    inventory["artifact_sha256"] = _sha256(inventory)
    destination = (
        args.generated_directory / f"powered-paper-assets-{inventory['artifact_sha256']}.json"
    )
    destination.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    run()
