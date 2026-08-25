#!/usr/bin/env python3
"""Build task-selection, score-definition, and family-weight robustness assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flavourbench.epicure_selection_complete_core_plan_v84 import (
    _task_order_key,
    selected_task_ids,
    verify_plan,
)
from flavourbench.epicure_selection_complete_core_sources_v1 import (
    load_full_primary_panels,
    source_graph,
)
from flavourbench.epicure_selection_powered_analysis import PanelData

FAMILIES = ("substitution", "pairing", "constraint")
PANELS = ("panel_1", "panel_2")
TASKS_PER_STRATUM = 89
SCHEMA_VERSION = "flavourbench-selection-robustness-v1"
DEFAULT_RANDOM_SUBSET_DRAWS = 20_000
DEFAULT_RANDOM_SUBSET_SEED = 20260825
DEFAULT_PLAN = Path(
    "benchmark/powered-v84/plan/epicure-selection-joint-analysis-plan-"
    "2ba71c793c8d4b97eed863ee83fd770b429fdefdffebdeafb241672f634ee507.json"
)


class SelectionRobustnessError(RuntimeError):
    """The frozen candidate panels or a derived robustness result is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRobustnessError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRobustnessError(f"input is not a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array, kind="stable")
    ranks = np.empty(len(array), dtype=np.int64)
    ranks[order] = np.arange(1, len(array) + 1)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    count = len(left_rank)
    if count < 2:
        return 1.0
    squared = float(np.square(left_rank - right_rank).sum())
    return 1.0 - 6.0 * squared / (count * (count * count - 1))


def _pair_order_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    triangle = np.triu_indices(len(left_array), k=1)
    return float(
        np.mean(
            np.sign(left_array[triangle[0]] - left_array[triangle[1]])
            == np.sign(right_array[triangle[0]] - right_array[triangle[1]])
        )
    )


def _standardized_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if not len(left_array) or not len(right_array):
        return 0.0
    left_variance = np.var(left_array, ddof=1 if len(left_array) > 1 else 0)
    right_variance = np.var(right_array, ddof=1 if len(right_array) > 1 else 0)
    pooled = np.sqrt((left_variance + right_variance) / 2.0)
    difference = float(np.mean(left_array) - np.mean(right_array))
    return 0.0 if pooled <= 1e-12 else difference / float(pooled)


def _total_variation(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    return 0.5 * sum(
        abs(left_counts[label] / len(left) - right_counts[label] / len(right)) for label in labels
    )


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for position, index in enumerate(order):
        running = max(running, (count - position) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return list(map(float, adjusted))


def _short_name(name: str) -> str:
    value = name.split(":", 1)[-1].strip()
    replacements = {
        "GPT-5.6": "5.6",
        "Claude ": "",
        "Gemini ": "Gemini ",
        "DeepSeek ": "DS ",
        "Command R+ (08-2024)": "Command R+",
        "Qwen3.8 2.4T A95B": "Qwen 3.8 A95B",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    return value


def _task_lookup(taskset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output = {str(task["task_id"]): task for task in taskset["tasks"]}
    if len(output) != 640:
        raise SelectionRobustnessError("candidate task set does not contain 640 unique tasks")
    return output


def _indices(data: PanelData) -> dict[str, int]:
    output = {task_id: index for index, task_id in enumerate(data.task_ids)}
    if len(output) != len(data.task_ids):
        raise SelectionRobustnessError("candidate panel contains duplicate task IDs")
    return output


def _selected_by_stratum(plan: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, ...]]:
    output: dict[tuple[str, str], tuple[str, ...]] = {}
    for panel in PANELS:
        selected = plan["common_core"]["panels"][panel]["selected_task_ids_by_family"]
        for family in FAMILIES:
            task_ids = tuple(str(value) for value in selected[family])
            if len(task_ids) != TASKS_PER_STRATUM or len(set(task_ids)) != len(task_ids):
                raise SelectionRobustnessError("frozen stratum does not contain 89 unique tasks")
            output[(panel, family)] = task_ids
    return output


def _balanced_scores(
    panels: Mapping[str, PanelData],
    selected: Mapping[tuple[str, str], Sequence[str]],
    model_indices: Sequence[int] | None = None,
) -> np.ndarray:
    indices = (
        np.arange(len(next(iter(panels.values())).model_ids), dtype=int)
        if model_indices is None
        else np.asarray(model_indices, dtype=int)
    )
    parts: list[np.ndarray] = []
    for panel in PANELS:
        data = panels[panel]
        task_index = _indices(data)
        for family in FAMILIES:
            positions = [task_index[task_id] for task_id in selected[(panel, family)]]
            parts.append(data.scores[np.ix_(indices, positions)].mean(axis=1))
    return np.stack(parts, axis=1).mean(axis=1)


def _select_for_roster(
    panels: Mapping[str, PanelData],
    tasksets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    included_model_indices: Sequence[int],
) -> tuple[dict[tuple[str, str], tuple[str, ...]], dict[tuple[str, str], int]]:
    selected: dict[tuple[str, str], tuple[str, ...]] = {}
    available: dict[tuple[str, str], int] = {}
    roster = np.asarray(included_model_indices, dtype=int)
    for panel in PANELS:
        data = panels[panel]
        valid = np.all((data.completed & data.parseable)[roster], axis=0)
        for family in FAMILIES:
            pool = [
                task_id
                for task_id, is_valid in zip(data.task_ids, valid, strict=True)
                if is_valid and tasksets[panel][task_id]["family"] == family
            ]
            pool.sort(
                key=lambda task_id: _task_order_key(
                    panel=panel,
                    family=family,
                    task_id=task_id,
                )
            )
            available[(panel, family)] = len(pool)
            if len(pool) < TASKS_PER_STRATUM:
                raise SelectionRobustnessError(
                    f"{panel}/{family} has only {len(pool)} tasks for a leave-one-out roster"
                )
            selected[(panel, family)] = tuple(pool[:TASKS_PER_STRATUM])
    return selected, available


def _leave_one_model_out(
    *,
    panels: Mapping[str, PanelData],
    tasksets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    official: Mapping[tuple[str, str], Sequence[str]],
    model_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    model_ids = panels["panel_1"].model_ids
    official_scores = _balanced_scores(panels, official)
    official_ids = {task_id for values in official.values() for task_id in values}
    output: list[dict[str, Any]] = []
    for omitted_index, omitted_model_id in enumerate(model_ids):
        included = [index for index in range(len(model_ids)) if index != omitted_index]
        selected, available = _select_for_roster(panels, tasksets, included)
        selected_ids = {task_id for values in selected.values() for task_id in values}
        baseline = official_scores[included]
        alternative = _balanced_scores(panels, selected, included)
        baseline_order = np.argsort(-baseline, kind="stable")
        alternative_order = np.argsort(-alternative, kind="stable")
        output.append(
            {
                "omitted_model_id": omitted_model_id,
                "omitted_model_name": model_names[omitted_model_id],
                "selected_task_overlap": len(official_ids & selected_ids),
                "selected_task_overlap_fraction": len(official_ids & selected_ids)
                / len(official_ids),
                "minimum_available_tasks_per_stratum": min(available.values()),
                "rank_spearman": _spearman(baseline, alternative),
                "pair_order_agreement": _pair_order_agreement(baseline, alternative),
                "mean_absolute_score_shift": float(np.mean(np.abs(alternative - baseline))),
                "maximum_absolute_score_shift": float(np.max(np.abs(alternative - baseline))),
                "official_point_leader": model_ids[included[int(baseline_order[0])]],
                "leave_one_out_point_leader": model_ids[included[int(alternative_order[0])]],
                "point_leader_preserved": bool(baseline_order[0] == alternative_order[0]),
            }
        )
    return output


def _task_metrics(
    task: Mapping[str, Any],
    reference_score: float,
) -> dict[str, float]:
    scores = np.asarray(list(task["selection_scores_bps"].values()), dtype=float)
    return {
        "chance_score": float(task["chance_score_bps"]) / 100.0,
        "optimal_margin": float(task["optimal_margin_bps"]) / 100.0,
        "distinct_score_count": float(len(np.unique(scores))),
        "zero_portfolio_share": float(np.mean(scores == 0.0)),
        "score_map_standard_deviation": float(np.std(scores, ddof=0) / 100.0),
        "reference_26_model_mean_score": float(reference_score),
    }


def _selection_profile(
    *,
    panels: Mapping[str, PanelData],
    tasksets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    official: Mapping[tuple[str, str], Sequence[str]],
    fable_index: int,
    random_subset_draws: int,
    random_subset_seed: int,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    gate_comparisons: list[dict[str, Any]] = []
    category_distances: list[dict[str, Any]] = []
    reference_indices = [
        index for index in range(len(panels["panel_1"].model_ids)) if index != fable_index
    ]
    tested_rows: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(PANELS):
        data = panels[panel]
        task_index = _indices(data)
        validity = data.completed & data.parseable
        reference_matrix = data.scores[np.asarray(reference_indices, dtype=int)].astype(float)
        reference_validity = validity[np.asarray(reference_indices, dtype=int)]
        reference_matrix[~reference_validity] = np.nan
        reference_means = np.nanmean(reference_matrix, axis=0)
        fable_valid = validity[fable_index]
        for family_index, family in enumerate(FAMILIES):
            family_ids = [
                task_id for task_id in data.task_ids if tasksets[panel][task_id]["family"] == family
            ]
            selected_ids = set(official[(panel, family)])
            not_selected_ids = [task_id for task_id in family_ids if task_id not in selected_ids]
            if len(family_ids) != 160 or len(selected_ids) != 89 or len(not_selected_ids) != 71:
                raise SelectionRobustnessError("candidate or selected family cardinality differs")
            rows = {
                task_id: _task_metrics(
                    tasksets[panel][task_id],
                    reference_means[task_index[task_id]],
                )
                for task_id in family_ids
            }
            selected_mask = np.asarray(
                [task_id in selected_ids for task_id in family_ids], dtype=bool
            )
            rng = np.random.default_rng(
                random_subset_seed + panel_index * len(FAMILIES) + family_index
            )
            random_indices = np.stack(
                [
                    rng.choice(len(family_ids), size=TASKS_PER_STRATUM, replace=False)
                    for _ in range(random_subset_draws)
                ],
                axis=0,
            )
            for metric in rows[family_ids[0]]:
                metric_values = np.asarray(
                    [rows[task_id][metric] for task_id in family_ids], dtype=float
                )
                left = metric_values[selected_mask]
                right = metric_values[~selected_mask]
                observed_difference = float(np.mean(left) - np.mean(right))
                random_left_mean = metric_values[random_indices].mean(axis=1)
                random_right_mean = (
                    float(metric_values.sum()) - TASKS_PER_STRATUM * random_left_mean
                ) / (len(family_ids) - TASKS_PER_STRATUM)
                empirical_p = (
                    1
                    + int(
                        np.count_nonzero(
                            np.abs(random_left_mean - random_right_mean)
                            >= abs(observed_difference) - 1e-12
                        )
                    )
                ) / (random_subset_draws + 1)
                comparison = {
                    "panel": panel,
                    "family": family,
                    "metric": metric,
                    "selected_mean": float(np.mean(left)),
                    "not_selected_mean": float(np.mean(right)),
                    "standardized_difference": _standardized_difference(left, right),
                    "random_subset_two_sided_p": empirical_p,
                }
                comparisons.append(comparison)
                tested_rows.append(comparison)
                valid_ids = [task_id for task_id in family_ids if fable_valid[task_index[task_id]]]
                invalid_ids = [
                    task_id for task_id in family_ids if not fable_valid[task_index[task_id]]
                ]
                if valid_ids and invalid_ids:
                    gate_comparisons.append(
                        {
                            "panel": panel,
                            "family": family,
                            "metric": metric,
                            "fable_valid_tasks": len(valid_ids),
                            "fable_invalid_tasks": len(invalid_ids),
                            "valid_mean": float(
                                np.mean([rows[task_id][metric] for task_id in valid_ids])
                            ),
                            "invalid_mean": float(
                                np.mean([rows[task_id][metric] for task_id in invalid_ids])
                            ),
                            "standardized_difference": _standardized_difference(
                                [rows[task_id][metric] for task_id in valid_ids],
                                [rows[task_id][metric] for task_id in invalid_ids],
                            ),
                        }
                    )
            categories = np.asarray(
                [str(tasksets[panel][task_id]["primary_category"]) for task_id in family_ids],
                dtype=object,
            )
            observed_tv = _total_variation(
                list(categories[selected_mask]), list(categories[~selected_mask])
            )
            category_codes = np.unique(categories, return_inverse=True)[1]
            random_tv = np.zeros(random_subset_draws, dtype=float)
            for code in np.unique(category_codes):
                total = int(np.count_nonzero(category_codes == code))
                selected_counts = np.count_nonzero(category_codes[random_indices] == code, axis=1)
                random_tv += np.abs(
                    selected_counts / TASKS_PER_STRATUM
                    - (total - selected_counts) / (len(family_ids) - TASKS_PER_STRATUM)
                )
            random_tv *= 0.5
            category_row = {
                "panel": panel,
                "family": family,
                "selected_vs_not_selected_total_variation": observed_tv,
                "random_subset_two_sided_p": (
                    1 + int(np.count_nonzero(random_tv >= observed_tv - 1e-12))
                )
                / (random_subset_draws + 1),
            }
            category_distances.append(category_row)
            tested_rows.append(category_row)
    adjusted = _holm_adjust([float(row["random_subset_two_sided_p"]) for row in tested_rows])
    for row, value in zip(tested_rows, adjusted, strict=True):
        row["random_subset_holm_p"] = value
        row["random_subset_holm_significant"] = value < 0.05
    qualified_gate = [
        row
        for row in gate_comparisons
        if int(row["fable_valid_tasks"]) >= 10 and int(row["fable_invalid_tasks"]) >= 10
    ]
    return {
        "candidate_tasks_in_included_families": 2 * 3 * 160,
        "selected_tasks": 2 * 3 * TASKS_PER_STRATUM,
        "selected_vs_not_selected": comparisons,
        "validity_gate_fable_valid_vs_invalid": gate_comparisons,
        "category_balance": category_distances,
        "random_subset_draws_per_stratum": random_subset_draws,
        "random_subset_seed": random_subset_seed,
        "random_subset_hypotheses": len(tested_rows),
        "random_subset_holm_resolved_characteristics": sum(
            bool(row["random_subset_holm_significant"]) for row in tested_rows
        ),
        "maximum_absolute_selected_standardized_difference": max(
            abs(float(row["standardized_difference"])) for row in comparisons
        ),
        "maximum_absolute_validity_gate_standardized_difference": max(
            abs(float(row["standardized_difference"])) for row in gate_comparisons
        ),
        "maximum_absolute_validity_gate_standardized_difference_minimum_10_tasks_per_group": max(
            abs(float(row["standardized_difference"])) for row in qualified_gate
        ),
        "maximum_category_total_variation": max(
            float(row["selected_vs_not_selected_total_variation"]) for row in category_distances
        ),
    }


def _selection_metric_values(task: Mapping[str, Any], selection: str) -> dict[str, float]:
    scores = np.asarray(list(task["selection_scores_bps"].values()), dtype=float)
    observed = float(task["selection_scores_bps"][selection])
    chance = float(task["chance_score_bps"])
    equal_other = int(np.count_nonzero(scores == observed)) - 1
    lower = int(np.count_nonzero(scores < observed))
    percentile = 100.0 * (lower + 0.5 * equal_other) / (len(scores) - 1)
    return {
        "flavourbench_score": observed / 100.0,
        "chance_adjusted_gain": 100.0 * (observed - chance) / (10_000.0 - chance),
        "action_percentile": percentile,
        "exact_optimum_rate": 100.0 if observed == 10_000.0 else 0.0,
    }


def _metric_sensitivity(
    *,
    panels: Mapping[str, PanelData],
    tasksets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    official: Mapping[tuple[str, str], Sequence[str]],
    model_names: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    metric_names = (
        "flavourbench_score",
        "chance_adjusted_gain",
        "action_percentile",
        "exact_optimum_rate",
    )
    values = {
        metric: np.zeros(len(panels["panel_1"].model_ids), dtype=float) for metric in metric_names
    }
    for panel in PANELS:
        data = panels[panel]
        task_index = _indices(data)
        for family in FAMILIES:
            task_ids = official[(panel, family)]
            for model_index in range(len(data.model_ids)):
                cell_metrics = [
                    _selection_metric_values(
                        tasksets[panel][task_id],
                        str(data.selections[model_index][task_index[task_id]]),
                    )
                    for task_id in task_ids
                ]
                for metric in metric_names:
                    values[metric][model_index] += (
                        float(np.mean([row[metric] for row in cell_metrics])) / 6.0
                    )
    baseline = values["flavourbench_score"]
    baseline_top = set(np.argsort(-baseline, kind="stable")[:5])
    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        scores = values[metric]
        order = np.argsort(-scores, kind="stable")
        rows.append(
            {
                "metric": metric,
                "rank_spearman_with_flavourbench": _spearman(baseline, scores),
                "pair_order_agreement_with_flavourbench": _pair_order_agreement(baseline, scores),
                "point_leader_model_id": panels["panel_1"].model_ids[int(order[0])],
                "point_leader_model_name": model_names[panels["panel_1"].model_ids[int(order[0])]],
                "top_five_overlap_with_flavourbench": len(
                    baseline_top & set(int(value) for value in order[:5])
                )
                / 5.0,
            }
        )
    return rows, {metric: list(map(float, score)) for metric, score in values.items()}


def _family_weight_sensitivity(
    *,
    panels: Mapping[str, PanelData],
    official: Mapping[tuple[str, str], Sequence[str]],
    model_names: Mapping[str, str],
) -> dict[str, Any]:
    family_scores = np.zeros((len(panels["panel_1"].model_ids), len(FAMILIES)))
    for family_index, family in enumerate(FAMILIES):
        parts = []
        for panel in PANELS:
            data = panels[panel]
            task_index = _indices(data)
            positions = [task_index[task_id] for task_id in official[(panel, family)]]
            parts.append(data.scores[:, positions].mean(axis=1))
        family_scores[:, family_index] = np.stack(parts, axis=1).mean(axis=1)
    equal_scores = family_scores.mean(axis=1)
    weights = np.asarray(
        [
            (first / 100.0, second / 100.0, third / 100.0)
            for first in range(20, 51)
            for second in range(20, 51)
            for third in (100 - first - second,)
            if 20 <= third <= 50
        ],
        dtype=float,
    )
    weighted = weights @ family_scores.T
    rhos = np.asarray([_spearman(equal_scores, row) for row in weighted], dtype=float)
    agreements = np.asarray(
        [_pair_order_agreement(equal_scores, row) for row in weighted], dtype=float
    )
    leaders = np.argmax(weighted, axis=1)
    leader_counts = Counter(int(value) for value in leaders)
    equal_leader = int(np.argmax(equal_scores))
    rank_matrix = np.stack([_ranks(row) for row in weighted], axis=0)
    model_ids = panels["panel_1"].model_ids
    return {
        "weight_grid_definition": (
            "all one-percentage-point three-family weights summing to one, with every family "
            "between 0.20 and 0.50"
        ),
        "grid_points": len(weights),
        "rank_spearman": {
            "minimum": float(np.min(rhos)),
            "median": float(np.median(rhos)),
            "maximum": float(np.max(rhos)),
        },
        "pair_order_agreement": {
            "minimum": float(np.min(agreements)),
            "median": float(np.median(agreements)),
            "maximum": float(np.max(agreements)),
        },
        "equal_weight_point_leader": model_ids[equal_leader],
        "equal_weight_leader_retention_share": leader_counts[equal_leader] / len(weights),
        "point_leader_frequencies": [
            {
                "model_id": model_ids[index],
                "model_name": model_names[model_ids[index]],
                "grid_points": count,
                "share": count / len(weights),
            }
            for index, count in sorted(
                leader_counts.items(), key=lambda item: (-item[1], model_ids[item[0]])
            )
        ],
        "model_rank_ranges": [
            {
                "model_id": model_id,
                "model_name": model_names[model_id],
                "minimum_rank": int(np.min(rank_matrix[:, index])),
                "maximum_rank": int(np.max(rank_matrix[:, index])),
            }
            for index, model_id in enumerate(model_ids)
        ],
    }


def build_analysis(*, root: Path, plan_path: Path) -> dict[str, Any]:
    root = root.resolve()
    plan_path = plan_path.resolve()
    plan = _load(plan_path)
    if not verify_plan(plan):
        raise SelectionRobustnessError("complete-core plan failed verification")
    graph = source_graph(root)
    panel_1, panel_2 = load_full_primary_panels(graph)
    if panel_1.model_ids != panel_2.model_ids:
        raise SelectionRobustnessError("candidate-panel model rosters differ")
    panels = {"panel_1": panel_1, "panel_2": panel_2}
    tasksets = {
        "panel_1": _task_lookup(graph.panel_1_taskset),
        "panel_2": _task_lookup(graph.panel_2_taskset),
    }
    official = _selected_by_stratum(plan)
    expected_1, expected_2 = selected_task_ids(plan)
    observed_1 = tuple(task_id for family in FAMILIES for task_id in official[("panel_1", family)])
    observed_2 = tuple(task_id for family in FAMILIES for task_id in official[("panel_2", family)])
    if observed_1 != expected_1:
        raise SelectionRobustnessError("panel-1 selected task order differs from the plan")
    if observed_2 != expected_2:
        raise SelectionRobustnessError("panel-2 selected task order differs from the plan")
    model_names = {str(row["model_id"]): str(row["model_name"]) for row in plan["roster"]["models"]}
    fable_model_id = "anthropic/claude-fable-5"
    fable_index = panel_1.model_ids.index(fable_model_id)
    leave_one_out = _leave_one_model_out(
        panels=panels,
        tasksets=tasksets,
        official=official,
        model_names=model_names,
    )
    metric_rows, metric_values = _metric_sensitivity(
        panels=panels,
        tasksets=tasksets,
        official=official,
        model_names=model_names,
    )
    analysis: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_collection_robustness_analysis",
        "inputs": {
            "builder_physical_sha256": _file_sha256(Path(__file__).resolve()),
            "complete_core_plan_physical_sha256": _file_sha256(plan_path),
            "complete_core_plan_semantic_sha256": plan["artifact_sha256"],
            "panel_1_taskset_semantic_sha256": graph.panel_1_taskset["artifact_sha256"],
            "panel_2_taskset_semantic_sha256": graph.panel_2_taskset["artifact_sha256"],
            "panel_1_validity_matrix_sha256": plan["common_core"]["panels"]["panel_1"][
                "validity_matrix_sha256"
            ],
            "panel_2_validity_matrix_sha256": plan["common_core"]["panels"]["panel_2"][
                "validity_matrix_sha256"
            ],
        },
        "design": {
            "models": len(panel_1.model_ids),
            "candidate_tasks_per_panel": len(panel_1.task_ids),
            "included_families": list(FAMILIES),
            "selected_tasks_per_family_panel": TASKS_PER_STRATUM,
            "primary_selected_tasks": sum(len(value) for value in official.values()),
            "selection_rule": "same fixed SHA-256 order as the primary complete-core plan",
        },
        "selection_profile": _selection_profile(
            panels=panels,
            tasksets=tasksets,
            official=official,
            fable_index=fable_index,
            random_subset_draws=DEFAULT_RANDOM_SUBSET_DRAWS,
            random_subset_seed=DEFAULT_RANDOM_SUBSET_SEED,
        ),
        "leave_one_model_out": leave_one_out,
        "most_influential_omission": min(
            leave_one_out,
            key=lambda row: (float(row["selected_task_overlap_fraction"]), row["omitted_model_id"]),
        ),
        "score_definition_sensitivity": metric_rows,
        "score_definition_model_values": [
            {
                "model_id": model_id,
                "model_name": model_names[model_id],
                **{metric: values[index] for metric, values in metric_values.items()},
            }
            for index, model_id in enumerate(panel_1.model_ids)
        ],
        "family_weight_sensitivity": _family_weight_sensitivity(
            panels=panels,
            official=official,
            model_names=model_names,
        ),
        "claim_boundary": (
            "These are post-collection sensitivity diagnostics. They test dependence on the "
            "complete-core roster, observable task-map characteristics, score definition, and "
            "moderate family reweighting. They do not establish external culinary validity or "
            "replace the prespecified complete-core inference."
        ),
    }
    analysis["artifact_sha256"] = hashlib.sha256(_canonical(analysis)).hexdigest()
    return analysis


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csvs(directory: Path, analysis: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "complete-core-leave-one-model-out.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = analysis["leave_one_model_out"]
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "complete-core-score-definition-sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = analysis["score_definition_sensitivity"]
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _write_tex(directory: Path, analysis: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    omission = analysis["most_influential_omission"]
    profile = analysis["selection_profile"]
    weights = analysis["family_weight_sensitivity"]
    alternatives = [
        row
        for row in analysis["score_definition_sensitivity"]
        if row["metric"] != "flavourbench_score"
    ]
    minimum_metric_rho = min(float(row["rank_spearman_with_flavourbench"]) for row in alternatives)
    minimum_metric_agreement = min(
        float(row["pair_order_agreement_with_flavourbench"]) for row in alternatives
    )
    omitted_name = _latex_escape(str(omission["omitted_model_name"]).split(":", 1)[-1].strip())
    omitted_overlap = 100 * float(omission["selected_task_overlap_fraction"])
    omitted_pair_agreement = 100 * float(omission["pair_order_agreement"])
    lines = [
        "% Generated by build_selection_robustness_assets.py; do not edit.",
        rf"\newcommand{{\FBLOOModel}}{{{omitted_name}}}",
        rf"\newcommand{{\FBLOOOverlap}}{{{omitted_overlap:.1f}\%}}",
        rf"\newcommand{{\FBLOORankRho}}{{{float(omission['rank_spearman']):.3f}}}",
        rf"\newcommand{{\FBLOOPairAgreement}}{{{omitted_pair_agreement:.1f}\%}}",
        rf"\newcommand{{\FBLOOMeanShift}}{{{float(omission['mean_absolute_score_shift']):.2f}}}",
        rf"\newcommand{{\FBLOOMaxShift}}{{{float(omission['maximum_absolute_score_shift']):.2f}}}",
        rf"\newcommand{{\FBSelectionMaxSMD}}{{{float(profile['maximum_absolute_selected_standardized_difference']):.2f}}}",
        rf"\newcommand{{\FBMetricMinRho}}{{{minimum_metric_rho:.3f}}}",
        rf"\newcommand{{\FBMetricMinAgreement}}{{{100 * minimum_metric_agreement:.1f}\%}}",
        rf"\newcommand{{\FBWeightGridPoints}}{{{int(weights['grid_points'])}}}",
        rf"\newcommand{{\FBWeightMinRho}}{{{float(weights['rank_spearman']['minimum']):.3f}}}",
        rf"\newcommand{{\FBWeightMedianRho}}{{{float(weights['rank_spearman']['median']):.3f}}}",
        rf"\newcommand{{\FBRobustnessArtifact}}{{{analysis['artifact_sha256']}}}",
    ]
    (directory / "complete-core-robustness-macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    labels = {
        "flavourbench_score": "FlavourBench Score",
        "chance_adjusted_gain": "Chance-adjusted gain",
        "action_percentile": "Action percentile",
        "exact_optimum_rate": "Exact-optimum rate",
    }
    table = [
        r"\begin{tabular}{@{}l r r l@{}}",
        r"\toprule",
        r"Score summary & Rank $\rho$ & Pair order & Point leader \\",
        r"\midrule",
    ]
    for row in analysis["score_definition_sensitivity"]:
        table.append(
            f"{labels[str(row['metric'])]} & "
            f"{float(row['rank_spearman_with_flavourbench']):.3f} & "
            f"{100 * float(row['pair_order_agreement_with_flavourbench']):.1f}\\% & "
            f"{_latex_escape(str(row['point_leader_model_name']).split(':', 1)[-1].strip())} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (directory / "complete-core-score-sensitivity-table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


def _plot(directory: Path, analysis: Mapping[str, Any]) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1787628000")
    ink = "#171A18"
    muted = "#68706C"
    rule = "#DDE1DE"
    rust = "#A83D34"
    teal = "#356C64"
    paper = "#F6F7F5"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": paper,
            "figure.facecolor": paper,
            "axes.edgecolor": rule,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": ink,
            "axes.titlecolor": ink,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6.4), constrained_layout=True)
    loo = sorted(
        analysis["leave_one_model_out"],
        key=lambda row: (
            float(row["selected_task_overlap_fraction"]),
            float(row["rank_spearman"]),
            str(row["omitted_model_name"]),
        ),
    )
    y = np.arange(len(loo))
    overlap = np.asarray([100 * float(row["selected_task_overlap_fraction"]) for row in loo])
    colors = [rust if index == 0 else teal for index in range(len(loo))]
    axes[0].hlines(y, 0, overlap, color=rule, linewidth=1.2)
    axes[0].scatter(overlap, y, color=colors, s=25, zorder=3)
    axes[0].set_yticks(
        y,
        [_short_name(str(row["omitted_model_name"])) for row in loo],
        fontsize=7.2,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 102)
    axes[0].set_xlabel("Official tasks retained (%)")
    axes[0].set_title("Task-set dependence on the model roster", loc="left", fontsize=11.5)
    axes[0].grid(axis="x", color=rule, linewidth=0.8)

    alternatives = [
        row
        for row in analysis["score_definition_sensitivity"]
        if row["metric"] != "flavourbench_score"
    ]
    labels = {
        "chance_adjusted_gain": "Chance-adjusted gain",
        "action_percentile": "Action percentile",
        "exact_optimum_rate": "Exact-optimum rate",
    }
    y_alt = np.arange(len(alternatives))
    rho = np.asarray([float(row["rank_spearman_with_flavourbench"]) for row in alternatives])
    pair = np.asarray(
        [float(row["pair_order_agreement_with_flavourbench"]) for row in alternatives]
    )
    axes[1].scatter(rho, y_alt - 0.10, color=rust, s=42, label="Rank correlation")
    axes[1].scatter(pair, y_alt + 0.10, color=teal, s=42, label="Pair-order agreement")
    axes[1].set_yticks(y_alt, [labels[str(row["metric"])] for row in alternatives], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(0.7, float(min(rho.min(), pair.min())) - 0.03), 1.01)
    axes[1].set_xlabel("Agreement with the primary score")
    axes[1].set_title("Alternative score definitions", loc="left", fontsize=11.5)
    axes[1].grid(axis="x", color=rule, linewidth=0.8)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    figure.suptitle(
        "Does the leaderboard depend on one task filter or one score formula?",
        x=0.01,
        ha="left",
        color=ink,
        fontsize=15,
        fontweight="bold",
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / "complete-core-selection-robustness"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "FlavourBench", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--generated-directory", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--dataset-figure-directory", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    analysis = build_analysis(root=root, plan_path=plan_path)
    output_name = "complete-core-selection-robustness.json"
    _write_json(args.generated_directory / output_name, analysis)
    _write_csvs(args.generated_directory, analysis)
    _write_tex(args.generated_directory, analysis)
    _plot(args.figure_directory, analysis)
    if args.dataset_directory is not None:
        _write_json(args.dataset_directory / output_name, analysis)
        _write_csvs(args.dataset_directory, analysis)
    if args.dataset_figure_directory is not None:
        _plot(args.dataset_figure_directory, analysis)
    print(
        "built selection robustness analysis "
        f"{analysis['artifact_sha256']} from two 640-task candidate panels"
    )


if __name__ == "__main__":
    main()
