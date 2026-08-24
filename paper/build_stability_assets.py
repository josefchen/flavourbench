#!/usr/bin/env python3
"""Build deterministic task-count stability and crossed-design variance assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILIES = ("substitution", "pairing", "constraint")
PANELS = ("panel_1", "panel_2")
TASK_COUNTS = (30, 60, 90, 150, 270, 534)
DEFAULT_REPLICATES = 5_000
DEFAULT_SEED = 20260824
SCHEMA_VERSION = "flavourbench-task-count-stability-v1"


class StabilityBuildError(RuntimeError):
    """The complete-core matrix or generated stability artifact is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StabilityBuildError(f"input is not a regular file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise StabilityBuildError(f"input has no JSON object rows: {path}")
    return rows


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _spearman_from_ranks(left: np.ndarray, right: np.ndarray) -> float:
    count = len(left)
    squared = float(np.square(left - right).sum())
    return 1.0 - 6.0 * squared / (count * (count * count - 1))


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
    }


def load_matrix(
    *,
    tasks_path: Path,
    observations_path: Path,
    leaderboard_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    tasks = _rows(tasks_path)
    leaderboard = sorted(_rows(leaderboard_path), key=lambda row: int(row["point_estimate_rank"]))
    observations = _rows(observations_path)
    if len(tasks) != 534 or len(leaderboard) != 27 or len(observations) != 14_418:
        raise StabilityBuildError("complete-core input cardinality differs")
    task_index = {str(row["task_id"]): index for index, row in enumerate(tasks)}
    model_index = {str(row["model_id"]): index for index, row in enumerate(leaderboard)}
    if len(task_index) != 534 or len(model_index) != 27:
        raise StabilityBuildError("task or model identifiers are not unique")
    matrix = np.full((27, 534), np.nan, dtype=float)
    for row in observations:
        response = row.get("response")
        scoring = row.get("release_scoring")
        if not isinstance(response, Mapping) or not isinstance(scoring, Mapping):
            raise StabilityBuildError("observation lacks response or release scoring")
        model_id = str(response.get("model_id"))
        task_id = str(response.get("task_id"))
        try:
            model_position = model_index[model_id]
            task_position = task_index[task_id]
        except KeyError as error:
            raise StabilityBuildError(
                f"observation identifier is not in the release: {error}"
            ) from error
        if np.isfinite(matrix[model_position, task_position]):
            raise StabilityBuildError(f"duplicate observation: {model_id}/{task_id}")
        matrix[model_position, task_position] = float(scoring["score"])
    if not np.isfinite(matrix).all():
        raise StabilityBuildError("complete-core matrix contains missing or non-finite scores")
    for model_position, row in enumerate(leaderboard):
        family_score = np.mean(
            [
                np.mean(
                    matrix[
                        model_position,
                        [index for index, task in enumerate(tasks) if task["family"] == family],
                    ]
                )
                for family in FAMILIES
            ]
        )
        if not np.isclose(family_score, float(row["flavourbench_score"]), atol=1e-10):
            raise StabilityBuildError(f"matrix does not reproduce score for {row['model_id']}")
    return tasks, leaderboard, matrix


def task_count_stability(
    *,
    tasks: Sequence[Mapping[str, Any]],
    leaderboard: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    significant_pairs: Sequence[tuple[int, int]],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        strata[(str(task["family"]), str(task["release_panel"]))].append(index)
    expected_strata = {(family, panel) for family in FAMILIES for panel in PANELS}
    if set(strata) != expected_strata or any(len(indices) != 89 for indices in strata.values()):
        raise StabilityBuildError("complete-core task strata are not six balanced groups of 89")
    full_scores = matrix.mean(axis=1)
    full_ranks = _rank(full_scores)
    full_order = np.argsort(-full_scores, kind="stable")
    triangle = np.triu_indices(len(leaderboard), k=1)
    full_pair_sign = np.sign(full_scores[triangle[0]] - full_scores[triangle[1]])
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for task_count in TASK_COUNTS:
        per_stratum = task_count // 6
        if task_count % 6 or per_stratum > 89:
            raise StabilityBuildError(f"task count is not compatible with the design: {task_count}")
        repetitions = 1 if task_count == 534 else replicates
        metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(repetitions):
            selected: list[int] = []
            for family in FAMILIES:
                for panel in PANELS:
                    indices = strata[(family, panel)]
                    if per_stratum == 89:
                        selected.extend(indices)
                    else:
                        selected.extend(
                            int(value)
                            for value in rng.choice(indices, size=per_stratum, replace=False)
                        )
            scores = matrix[:, selected].mean(axis=1)
            ranks = _rank(scores)
            order = np.argsort(-scores, kind="stable")
            pair_sign = np.sign(scores[triangle[0]] - scores[triangle[1]])
            metrics["rank_spearman"].append(_spearman_from_ranks(ranks, full_ranks))
            metrics["all_pair_order_agreement"].append(float(np.mean(pair_sign == full_pair_sign)))
            metrics["top_1_preserved"].append(float(order[0] == full_order[0]))
            metrics["top_5_overlap"].append(
                len(set(order[:5]).intersection(int(value) for value in full_order[:5])) / 5
            )
            metrics["mean_absolute_score_error"].append(
                float(np.mean(np.abs(scores - full_scores)))
            )
            metrics["maximum_absolute_score_error"].append(
                float(np.max(np.abs(scores - full_scores)))
            )
            if significant_pairs:
                metrics["resolved_pair_direction_agreement"].append(
                    float(
                        np.mean(
                            [
                                np.sign(scores[left] - scores[right])
                                == np.sign(full_scores[left] - full_scores[right])
                                for left, right in significant_pairs
                            ]
                        )
                    )
                )
        output.append(
            {
                "tasks": task_count,
                "tasks_per_family_panel_stratum": per_stratum,
                "subsample_replicates": repetitions,
                "metrics": {name: _summary(values) for name, values in sorted(metrics.items())},
            }
        )
    return output


def variance_partition(*, tasks: Sequence[Mapping[str, Any]], matrix: np.ndarray) -> dict[str, Any]:
    model_count, task_count = matrix.shape
    grand = float(matrix.mean())
    model_means = matrix.mean(axis=1)
    task_means = matrix.mean(axis=0)
    model_ss = float(task_count * np.square(model_means - grand).sum())
    task_ss = float(model_count * np.square(task_means - grand).sum())
    residual = matrix - model_means[:, None] - task_means[None, :] + grand
    interaction_ss = float(np.square(residual).sum())
    total_ss = float(np.square(matrix - grand).sum())

    family_means = {
        family: float(
            matrix[
                :, [index for index, task in enumerate(tasks) if task["family"] == family]
            ].mean()
        )
        for family in FAMILIES
    }
    panel_means = {
        panel: float(
            matrix[
                :,
                [index for index, task in enumerate(tasks) if task["release_panel"] == panel],
            ].mean()
        )
        for panel in PANELS
    }
    cell_means = {
        (family, panel): float(
            matrix[
                :,
                [
                    index
                    for index, task in enumerate(tasks)
                    if task["family"] == family and task["release_panel"] == panel
                ],
            ].mean()
        )
        for family in FAMILIES
        for panel in PANELS
    }
    family_ss = float(
        model_count * 178 * sum((family_means[family] - grand) ** 2 for family in FAMILIES)
    )
    panel_ss = float(model_count * 267 * sum((panel_means[panel] - grand) ** 2 for panel in PANELS))
    family_panel_ss = float(
        model_count
        * 89
        * sum(
            (cell_means[(family, panel)] - family_means[family] - panel_means[panel] + grand) ** 2
            for family in FAMILIES
            for panel in PANELS
        )
    )
    within_cell_task_ss = task_ss - family_ss - panel_ss - family_panel_ss
    if not np.isclose(total_ss, model_ss + task_ss + interaction_ss, rtol=1e-10):
        raise StabilityBuildError("crossed-design sums of squares do not close")
    components = [
        ("model", model_ss),
        ("family", family_ss),
        ("panel", panel_ss),
        ("family_by_panel", family_panel_ss),
        ("task_within_family_panel", within_cell_task_ss),
        ("model_by_task", interaction_ss),
    ]
    model_ms = model_ss / (model_count - 1)
    task_ms = task_ss / (task_count - 1)
    interaction_ms = interaction_ss / ((model_count - 1) * (task_count - 1))
    model_variance = max(0.0, (model_ms - interaction_ms) / task_count)
    task_variance = max(0.0, (task_ms - interaction_ms) / model_count)
    interaction_variance = interaction_ms
    relative_g = model_variance / (model_variance + interaction_variance / task_count)
    tasks_for_g_90 = int(np.ceil(9.0 * interaction_variance / model_variance))
    return {
        "grand_mean": grand,
        "total_sum_squares": total_ss,
        "sum_squares_partition": [
            {
                "component": name,
                "sum_squares": value,
                "fraction_of_total": value / total_ss,
            }
            for name, value in components
        ],
        "random_effect_descriptive_components": {
            "model": model_variance,
            "task": task_variance,
            "model_by_task": interaction_variance,
        },
        "relative_decision_generalizability_at_534_tasks": relative_g,
        "estimated_balanced_tasks_for_relative_g_0_90": tasks_for_g_90,
        "interpretation": (
            "Descriptive crossed-design generalizability for relative endpoint comparisons; "
            "it is not a claim that the 27 endpoints or public tasks are random population samples."
        ),
    }


def build_analysis(
    *,
    tasks_path: Path,
    observations_path: Path,
    leaderboard_path: Path,
    pairwise_path: Path,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    tasks, leaderboard, matrix = load_matrix(
        tasks_path=tasks_path,
        observations_path=observations_path,
        leaderboard_path=leaderboard_path,
    )
    model_index = {str(row["model_id"]): index for index, row in enumerate(leaderboard)}
    significant_pairs = [
        (model_index[str(row["left_model_id"])], model_index[str(row["right_model_id"])])
        for row in _rows(pairwise_path)
        if bool(row["holm_significant"])
    ]
    if len(significant_pairs) != 101:
        raise StabilityBuildError("Holm-significant pair inventory differs")
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "retrospective_precision_and_stability_analysis",
        "seed": seed,
        "requested_subsample_replicates": replicates,
        "design": {
            "models": len(leaderboard),
            "tasks": len(tasks),
            "family_panel_strata": 6,
            "tasks_per_stratum": 89,
            "sampling": "without_replacement_within_each_family_panel_stratum",
            "reference": "complete_534_task_point_order",
        },
        "inputs": {
            "tasks_sha256": _sha256(tasks_path),
            "observations_sha256": _sha256(observations_path),
            "leaderboard_sha256": _sha256(leaderboard_path),
            "pairwise_sha256": _sha256(pairwise_path),
        },
        "task_count_stability": task_count_stability(
            tasks=tasks,
            leaderboard=leaderboard,
            matrix=matrix,
            significant_pairs=significant_pairs,
            replicates=replicates,
            seed=seed,
        ),
        "variance_partition": variance_partition(tasks=tasks, matrix=matrix),
        "claim_boundary": (
            "Subsampling quantifies stability relative to this release's full point order. It is "
            "not a post-hoc power calculation and does not prove external culinary validity."
        ),
    }
    analysis["artifact_sha256"] = hashlib.sha256(_canonical(analysis)).hexdigest()
    return analysis


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_tables(directory: Path, analysis: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stability_path = directory / "complete-core-task-count-stability.csv"
    metric_names = sorted(analysis["task_count_stability"][0]["metrics"])
    with stability_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tasks", "tasks_per_stratum", "replicates"]
            + [
                f"{metric}_{stat}"
                for metric in metric_names
                for stat in ("mean", "median", "p2_5", "p97_5")
            ]
        )
        for row in analysis["task_count_stability"]:
            writer.writerow(
                [row["tasks"], row["tasks_per_family_panel_stratum"], row["subsample_replicates"]]
                + [
                    row["metrics"][metric][stat]
                    for metric in metric_names
                    for stat in ("mean", "median", "p2_5", "p97_5")
                ]
            )
    variance_path = directory / "complete-core-variance-partition.csv"
    with variance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("component", "sum_squares", "fraction_of_total")
        )
        writer.writeheader()
        writer.writerows(analysis["variance_partition"]["sum_squares_partition"])


def _write_dataset_views(directory: Path, analysis: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = str(analysis["artifact_sha256"])
    stability_rows = []
    for row in analysis["task_count_stability"]:
        flattened: dict[str, Any] = {
            "analysis_artifact_sha256": artifact,
            "tasks": row["tasks"],
            "tasks_per_stratum": row["tasks_per_family_panel_stratum"],
            "replicates": row["subsample_replicates"],
        }
        for metric, summary in row["metrics"].items():
            for statistic, value in summary.items():
                flattened[f"{metric}_{statistic}"] = value
        stability_rows.append(flattened)
    variance_rows = [
        {"analysis_artifact_sha256": artifact, **row}
        for row in analysis["variance_partition"]["sum_squares_partition"]
    ]
    for name, rows in (
        ("task_count_stability.jsonl", stability_rows),
        ("variance_partition.jsonl", variance_rows),
    ):
        payload = b"".join(_canonical(row) + b"\n" for row in rows)
        (directory / name).write_bytes(payload)


def _write_macros(path: Path, analysis: Mapping[str, Any]) -> None:
    row_270 = next(row for row in analysis["task_count_stability"] if row["tasks"] == 270)
    rank = row_270["metrics"]["rank_spearman"]
    variance = analysis["variance_partition"]
    top_one = float(row_270["metrics"]["top_1_preserved"]["mean"]) * 100
    top_five = float(row_270["metrics"]["top_5_overlap"]["median"]) * 100
    lines = [
        "% Generated by build_stability_assets.py; do not edit.",
        rf"\newcommand{{\FBStabilityReplicates}}{{{int(analysis['requested_subsample_replicates']):,}}}",
        rf"\newcommand{{\FBGeneralizability}}{{{float(variance['relative_decision_generalizability_at_534_tasks']):.3f}}}",
        rf"\newcommand{{\FBTasksForGNinety}}{{{int(variance['estimated_balanced_tasks_for_relative_g_0_90'])}}}",
        rf"\newcommand{{\FBHalfTaskCount}}{{{int(row_270['tasks'])}}}",
        rf"\newcommand{{\FBHalfRankMedian}}{{{float(rank['median']):.3f}}}",
        rf"\newcommand{{\FBHalfRankLow}}{{{float(rank['p2_5']):.3f}}}",
        rf"\newcommand{{\FBHalfRankHigh}}{{{float(rank['p97_5']):.3f}}}",
        rf"\newcommand{{\FBHalfTopOne}}{{{top_one:.1f}\%}}",
        rf"\newcommand{{\FBHalfTopFive}}{{{top_five:.0f}\%}}",
        rf"\newcommand{{\FBStabilityArtifact}}{{{analysis['artifact_sha256']}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(figure_directory: Path, analysis: Mapping[str, Any]) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1787565600")
    rows = analysis["task_count_stability"]
    counts = np.asarray([row["tasks"] for row in rows], dtype=float)
    rust = "#A83D34"
    ink = "#161817"
    muted = "#68706C"
    rule = "#DDE1DE"
    paper = "#F6F7F5"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": paper,
            "figure.facecolor": paper,
            "axes.edgecolor": rule,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.titlecolor": ink,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    metrics = (
        ("rank_spearman", "Rank correlation to 534-task order"),
        ("top_5_overlap", "Top-five overlap"),
    )
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        median = np.asarray([row["metrics"][metric]["median"] for row in rows])
        low = np.asarray([row["metrics"][metric]["p2_5"] for row in rows])
        high = np.asarray([row["metrics"][metric]["p97_5"] for row in rows])
        axis.fill_between(counts, low, high, color=rust, alpha=0.13, linewidth=0)
        axis.plot(counts, median, color=rust, marker="o", linewidth=2.2, markersize=5)
        axis.axhline(1.0, color=rule, linewidth=1)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlabel("Balanced tasks")
        axis.set_ylim(max(0.0, float(low.min()) - 0.03), 1.015)
        axis.set_xticks(counts)
        axis.grid(axis="y", color=rule, linewidth=0.8)
    figure.suptitle(
        "How much of the full leaderboard survives smaller task sets?",
        x=0.01,
        ha="left",
        color=ink,
        fontsize=15,
        fontweight="bold",
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    stem = figure_directory / "complete-core-task-count-stability"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "FlavourBench", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--dataset-figure-directory", type=Path)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise StabilityBuildError("replicates must be positive")
    analysis = build_analysis(
        tasks_path=args.tasks,
        observations_path=args.observations,
        leaderboard_path=args.leaderboard,
        pairwise_path=args.pairwise,
        replicates=args.replicates,
        seed=args.seed,
    )
    _write_json(args.generated_directory / "complete-core-stability-analysis.json", analysis)
    _write_tables(args.generated_directory, analysis)
    _write_macros(args.generated_directory / "complete-core-stability-macros.tex", analysis)
    if args.dataset_directory is not None:
        _write_json(args.dataset_directory / "complete-core-stability-analysis.json", analysis)
        _write_tables(args.dataset_directory, analysis)
        _write_dataset_views(args.dataset_directory, analysis)
    _plot(args.figure_directory, analysis)
    if args.dataset_figure_directory is not None:
        _plot(args.dataset_figure_directory, analysis)
    print(
        "built stability analysis "
        f"{analysis['artifact_sha256']} with {args.replicates:,} subsamples per task count"
    )


if __name__ == "__main__":
    main()
