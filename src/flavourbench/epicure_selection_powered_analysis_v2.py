"""Analyze the anchor-free panel with quality and coverage as separate endpoints."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_selection_powered_analysis import (
    PanelData,
    SelectionPoweredAnalysisError,
    _cohen_dz,
    _ingredient_set,
    _load,
    _percentile_interval,
    _rank_intervals,
    _sha256,
    _sha256_bytes,
    _sha256_file,
    _statistical_groups,
    _write_content_addressed,
    holm_adjust,
    load_panel,
)
from .epicure_selection_powered_plan_v44 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V44
from .epicure_selection_powered_plan_v44 import verify_plan as verify_plan_v44
from .epicure_selection_powered_plan_v45 import verify_plan as verify_plan_v45
from .epicure_selection_powered_plan_v46 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V46
from .epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from .epicure_selection_powered_plan_v47 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V47
from .epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from .epicure_selection_powered_plan_v49 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V49
from .epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from .epicure_selection_powered_plan_v50 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V50
from .epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50
from .epicure_selection_powered_plan_v52 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V52
from .epicure_selection_powered_plan_v52 import verify_plan as verify_plan_v52
from .epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel as verify_repeat_panel_replication_2,
)
from .epicure_selection_repeat_panel_v2 import verify_repeat_panel
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID, QWEN_MODEL_ID
from .epicure_selection_route_manifest_v52 import (
    DEEPSEEK_FLASH_MODEL_ID,
    LUNA_MODEL_ID,
)
from .epicure_selection_route_manifest_v52 import (
    REPLACEMENT_MODEL_IDS as PANEL_2_REPLACEMENT_MODEL_IDS,
)
from .epicure_selection_taskset_replication_v1 import (
    verify_taskset as verify_taskset_replication_2,
)
from .epicure_selection_taskset_v1 import FAMILIES
from .epicure_selection_taskset_v2 import verify_taskset

ANALYSIS_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-v2-success-only"
RELEASE_SCHEMA_VERSION = "flavourbench-selection-powered-release-v2-anchor-free"


def _valid_family_summary(
    values: np.ndarray, valid: np.ndarray, families: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    family_values = np.asarray(families, dtype=object)
    if matrix.ndim != 2 or matrix.shape != mask.shape or matrix.shape[1] != len(families):
        raise SelectionPoweredAnalysisError("available-case score matrices differ")
    means: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        indices = np.flatnonzero(family_values == family)
        family_mask = mask[:, indices]
        count = family_mask.sum(axis=1)
        if np.any(count == 0):
            raise SelectionPoweredAnalysisError(
                f"a model has no successful parseable {family} response"
            )
        counts[family] = count
        means[family] = (matrix[:, indices] * family_mask).sum(axis=1) / count
    point = np.stack([means[family] for family in FAMILIES], axis=1).mean(axis=1)
    return point, means, counts


def missing_cell_score_bounds(
    values: np.ndarray, valid: np.ndarray, families: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Bound the scheduled-panel score without assigning a primary score to failures.

    The lower endpoint assigns every missing cell zero and the upper endpoint
    assigns every missing cell 100.  These are sensitivity bounds only; the
    benchmark's primary estimand remains the available-case family macro mean.
    """

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    family_values = np.asarray(families, dtype=object)
    if matrix.ndim != 2 or matrix.shape != mask.shape or matrix.shape[1] != len(families):
        raise SelectionPoweredAnalysisError("missing-cell score matrices differ")
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for family in FAMILIES:
        indices = np.flatnonzero(family_values == family)
        if not len(indices):
            raise SelectionPoweredAnalysisError(f"missing task family: {family}")
        observed_sum = (matrix[:, indices] * mask[:, indices]).sum(axis=1)
        missing = (~mask[:, indices]).sum(axis=1)
        lower_parts.append(observed_sum / len(indices))
        upper_parts.append((observed_sum + 100.0 * missing) / len(indices))
    return (
        np.stack(lower_parts, axis=1).mean(axis=1),
        np.stack(upper_parts, axis=1).mean(axis=1),
    )


def family_stratified_available_bootstrap(
    values: np.ndarray,
    valid: np.ndarray,
    families: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    """Resample tasks by family, then average each row's observed cells only."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    _valid_family_summary(matrix, mask, families)
    if resamples <= 0:
        raise SelectionPoweredAnalysisError("bootstrap resamples must be positive")
    family_values = np.asarray(families, dtype=object)
    rng = np.random.default_rng(seed)
    output = np.zeros((resamples, matrix.shape[0]), dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        width = stop - start
        batch = np.zeros((width, matrix.shape[0]), dtype=np.float64)
        for family in FAMILIES:
            indices = np.flatnonzero(family_values == family)
            family_matrix = matrix[:, indices]
            family_mask = mask[:, indices]
            draws = rng.integers(0, len(indices), size=(width, len(indices)))
            for _ in range(100):
                selected_mask = np.take(family_mask, draws, axis=1)
                denominators = selected_mask.sum(axis=2)
                bad = np.any(denominators == 0, axis=0)
                if not np.any(bad):
                    break
                draws[bad] = rng.integers(0, len(indices), size=(int(bad.sum()), len(indices)))
            else:
                raise SelectionPoweredAnalysisError("bootstrap could not sample every model")
            selected_values = np.take(family_matrix, draws, axis=1)
            numerators = (selected_values * selected_mask).sum(axis=2)
            batch += (numerators / denominators).T / len(FAMILIES)
        output[start:stop] = batch
    return output


def anchor_cluster_available_bootstrap(
    values: np.ndarray,
    valid: np.ndarray,
    families: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Resample ingredient anchors while retaining every task in each anchor cluster."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    _valid_family_summary(matrix, mask, families)
    if resamples <= 0:
        raise SelectionPoweredAnalysisError("bootstrap resamples must be positive")
    if len(cluster_ids) != matrix.shape[1]:
        raise SelectionPoweredAnalysisError("anchor cluster vector differs from tasks")
    ordered_clusters = tuple(dict.fromkeys(str(value) for value in cluster_ids))
    if not ordered_clusters or any(not value for value in ordered_clusters):
        raise SelectionPoweredAnalysisError("anchor cluster IDs must be nonempty")
    cluster_index = {value: index for index, value in enumerate(ordered_clusters)}
    task_clusters = np.asarray([cluster_index[str(value)] for value in cluster_ids])
    family_values = np.asarray(families, dtype=object)
    family_totals: list[tuple[np.ndarray, np.ndarray]] = []
    for family in FAMILIES:
        numerators = np.zeros((matrix.shape[0], len(ordered_clusters)), dtype=np.float64)
        denominators = np.zeros_like(numerators)
        for task_index in np.flatnonzero(family_values == family):
            anchor_index = task_clusters[task_index]
            numerators[:, anchor_index] += matrix[:, task_index] * mask[:, task_index]
            denominators[:, anchor_index] += mask[:, task_index]
        family_totals.append((numerators, denominators))
    rng = np.random.default_rng(seed)
    output = np.zeros((resamples, matrix.shape[0]), dtype=np.float64)
    probabilities = np.full(len(ordered_clusters), 1.0 / len(ordered_clusters), dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        width = stop - start
        cluster_counts = rng.multinomial(len(ordered_clusters), probabilities, size=width)
        batch = np.zeros((width, matrix.shape[0]), dtype=np.float64)
        for _ in range(100):
            bad = np.zeros(width, dtype=bool)
            for _, denominators in family_totals:
                sampled = denominators @ cluster_counts.T
                bad |= np.any(sampled == 0, axis=0)
            if not np.any(bad):
                break
            cluster_counts[bad] = rng.multinomial(
                len(ordered_clusters),
                probabilities,
                size=int(bad.sum()),
            )
        else:
            raise SelectionPoweredAnalysisError(
                "anchor bootstrap could not sample every model and family"
            )
        for numerators, denominators in family_totals:
            sampled_denominators = denominators @ cluster_counts.T
            sampled_numerators = numerators @ cluster_counts.T
            batch += (sampled_numerators / sampled_denominators).T / len(FAMILIES)
        output[start:stop] = batch
    return output


def weighted_sign_flip_pvalues(
    values: np.ndarray,
    valid: np.ndarray,
    families: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired sign flips with equal total weight assigned to every family."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    _valid_family_summary(matrix, mask, families)
    family_values = np.asarray(families, dtype=object)
    weights = np.zeros_like(matrix)
    for family in FAMILIES:
        indices = np.flatnonzero(family_values == family)
        counts = mask[:, indices].sum(axis=1)
        weights[:, indices] = mask[:, indices] / (len(FAMILIES) * counts[:, None])
    weighted = matrix * weights
    observed = weighted.sum(axis=1)
    exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    threshold = np.abs(observed)
    for start in range(0, resamples, batch_size):
        width = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(width, matrix.shape[1]), dtype=np.int8)
        signed = signs.astype(np.float64) * 2.0 - 1.0
        null = signed @ weighted.T
        exceed += np.count_nonzero(np.abs(null) >= threshold[None, :] - 1e-12, axis=0)
    return observed, (exceed + 1.0) / (resamples + 1.0)


def anchor_cluster_weighted_sign_flip_pvalues(
    values: np.ndarray,
    valid: np.ndarray,
    families: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip whole anchor clusters with equal total weight assigned to every family."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    _valid_family_summary(matrix, mask, families)
    if resamples <= 0:
        raise SelectionPoweredAnalysisError("sign-flip resamples must be positive")
    if len(cluster_ids) != matrix.shape[1]:
        raise SelectionPoweredAnalysisError("anchor cluster vector differs from tasks")
    ordered_clusters = tuple(dict.fromkeys(str(value) for value in cluster_ids))
    if not ordered_clusters or any(not value for value in ordered_clusters):
        raise SelectionPoweredAnalysisError("anchor cluster IDs must be nonempty")
    cluster_index = {value: index for index, value in enumerate(ordered_clusters)}
    family_values = np.asarray(families, dtype=object)
    weights = np.zeros_like(matrix)
    for family in FAMILIES:
        indices = np.flatnonzero(family_values == family)
        counts = mask[:, indices].sum(axis=1)
        weights[:, indices] = mask[:, indices] / (len(FAMILIES) * counts[:, None])
    cluster_weighted = np.zeros((matrix.shape[0], len(ordered_clusters)), dtype=np.float64)
    for task_index, cluster_id in enumerate(cluster_ids):
        cluster_weighted[:, cluster_index[str(cluster_id)]] += (
            matrix[:, task_index] * weights[:, task_index]
        )
    observed = cluster_weighted.sum(axis=1)
    exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    threshold = np.abs(observed)
    for start in range(0, resamples, batch_size):
        width = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(width, len(ordered_clusters)), dtype=np.int8)
        signed = signs.astype(np.float64) * 2.0 - 1.0
        null = signed @ cluster_weighted.T
        exceed += np.count_nonzero(np.abs(null) >= threshold[None, :] - 1e-12, axis=0)
    return observed, (exceed + 1.0) / (resamples + 1.0)


def analyze_repeatability(
    *,
    primary: PanelData,
    repeat: PanelData,
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    bootstrap_resamples: int,
    seed: int,
    cluster_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if primary.model_ids != repeat.model_ids:
        raise SelectionPoweredAnalysisError("primary and repeat rosters differ")
    primary_index = {task_id: index for index, task_id in enumerate(primary.task_ids)}
    primary_tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
    repeat_tasks = {str(task["task_id"]): task for task in repeat_panel["tasks"]}
    valid = np.zeros_like(repeat.completed, dtype=bool)
    jaccard = np.zeros_like(repeat.scores)
    exact = np.zeros_like(repeat.scores)
    score_delta = np.zeros_like(repeat.scores)
    for repeat_index, repeat_task_id in enumerate(repeat.task_ids):
        repeated_task = repeat_tasks[repeat_task_id]
        original_task_id = str(repeated_task["original_task_id"])
        original_index = primary_index[original_task_id]
        original_task = primary_tasks[original_task_id]
        for model_index in range(len(primary.model_ids)):
            is_valid = bool(
                primary.completed[model_index, original_index]
                and primary.parseable[model_index, original_index]
                and repeat.completed[model_index, repeat_index]
                and repeat.parseable[model_index, repeat_index]
            )
            valid[model_index, repeat_index] = is_valid
            if not is_valid:
                continue
            left = _ingredient_set(original_task, primary.selections[model_index][original_index])
            right = _ingredient_set(repeated_task, repeat.selections[model_index][repeat_index])
            union = left | right
            jaccard[model_index, repeat_index] = len(left & right) / len(union)
            exact[model_index, repeat_index] = float(left == right)
            score_delta[model_index, repeat_index] = abs(
                primary.scores[model_index, original_index]
                - repeat.scores[model_index, repeat_index]
            )
    family_values = np.asarray(repeat.families, dtype=object)
    counts = {
        family: valid[:, np.flatnonzero(family_values == family)].sum(axis=1) for family in FAMILIES
    }
    estimable = np.logical_and.reduce([counts[family] > 0 for family in FAMILIES])
    estimable_indices = np.flatnonzero(estimable)
    point = np.full(len(primary.model_ids), np.nan, dtype=np.float64)
    exact_point = np.full(len(primary.model_ids), np.nan, dtype=np.float64)
    delta_point = np.full(len(primary.model_ids), np.nan, dtype=np.float64)
    intervals: list[list[float] | None] = [None] * len(primary.model_ids)
    if len(estimable_indices):
        valid_subset = valid[estimable_indices]
        point_subset, _, _ = _valid_family_summary(
            jaccard[estimable_indices], valid_subset, repeat.families
        )
        exact_subset, _, _ = _valid_family_summary(
            exact[estimable_indices], valid_subset, repeat.families
        )
        delta_subset, _, _ = _valid_family_summary(
            score_delta[estimable_indices], valid_subset, repeat.families
        )
        point[estimable_indices] = point_subset
        exact_point[estimable_indices] = exact_subset
        delta_point[estimable_indices] = delta_subset
        if cluster_ids is None:
            bootstrap = family_stratified_available_bootstrap(
                jaccard[estimable_indices],
                valid_subset,
                repeat.families,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        else:
            bootstrap = anchor_cluster_available_bootstrap(
                jaccard[estimable_indices],
                valid_subset,
                repeat.families,
                cluster_ids,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        for subset_index, model_index in enumerate(estimable_indices):
            intervals[model_index] = _percentile_interval(bootstrap[:, subset_index])
    return [
        {
            "model_id": model_id,
            "repeatability_status": (
                "estimated_equal_family_macro"
                if estimable[index]
                else "not_estimable_missing_family_pairs"
            ),
            "scheduled": len(repeat.task_ids),
            "completed": int(repeat.completed[index].sum()),
            "parseable": int(repeat.parseable[index].sum()),
            "valid_primary_repeat_pairs": int(valid[index].sum()),
            "valid_pairs_per_family": {family: int(counts[family][index]) for family in FAMILIES},
            "mean_ingredient_set_jaccard": (float(point[index]) if estimable[index] else None),
            "jaccard_pointwise_95_ci": intervals[index],
            "exact_ingredient_set_match_rate": (
                float(exact_point[index]) if estimable[index] else None
            ),
            "mean_absolute_score_difference": (
                float(delta_point[index]) if estimable[index] else None
            ),
        }
        for index, model_id in enumerate(primary.model_ids)
    ]


def analyze_panels(
    *,
    primary: PanelData,
    taskset: Mapping[str, Any],
    plan: Mapping[str, Any],
    repeat: PanelData | None = None,
    repeat_panel: Mapping[str, Any] | None = None,
    bootstrap_resamples: int | None = None,
    permutation_resamples: int | None = None,
    cluster_ids: Sequence[str] | None = None,
    repeat_cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    inference = plan["inference"]
    bootstrap_count = int(bootstrap_resamples or inference["bootstrap_resamples"])
    permutation_count = int(permutation_resamples or inference["permutation_resamples"])
    seed = int(inference["seed"])
    valid = primary.completed & primary.parseable
    point, family_scores, family_counts = _valid_family_summary(
        primary.scores, valid, primary.families
    )
    missing_lower, missing_upper = missing_cell_score_bounds(
        primary.scores, valid, primary.families
    )
    if cluster_ids is None:
        score_bootstrap = family_stratified_available_bootstrap(
            primary.scores,
            valid,
            primary.families,
            resamples=bootstrap_count,
            seed=seed,
        )
        sign_flip = weighted_sign_flip_pvalues
    else:
        if len(cluster_ids) != len(primary.task_ids):
            raise SelectionPoweredAnalysisError("primary anchor clusters differ from tasks")
        score_bootstrap = anchor_cluster_available_bootstrap(
            primary.scores,
            valid,
            primary.families,
            cluster_ids,
            resamples=bootstrap_count,
            seed=seed,
        )
        sign_flip = anchor_cluster_weighted_sign_flip_pvalues
    standard_errors = np.std(score_bootstrap, axis=0, ddof=1)
    safe_se = np.where(standard_errors > 0, standard_errors, 1.0)
    max_t = np.max(np.abs((score_bootstrap - point[None, :]) / safe_se[None, :]), axis=1)
    max_t_critical = float(np.quantile(max_t, 0.95))

    task_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    chance = np.asarray(
        [float(task_by_id[task_id]["chance_score_bps"]) / 100 for task_id in primary.task_ids]
    )
    chance_matrix = np.broadcast_to(chance, primary.scores.shape)
    chance_point, _, _ = _valid_family_summary(chance_matrix, valid, primary.families)
    bootstrap_arguments = (cluster_ids,) if cluster_ids is not None else ()
    chance_bootstrap = (
        anchor_cluster_available_bootstrap(
            chance_matrix,
            valid,
            primary.families,
            cluster_ids,
            resamples=bootstrap_count,
            seed=seed,
        )
        if cluster_ids is not None
        else family_stratified_available_bootstrap(
            chance_matrix,
            valid,
            primary.families,
            resamples=bootstrap_count,
            seed=seed,
        )
    )
    chance_observed, chance_raw = sign_flip(
        primary.scores - chance_matrix,
        valid,
        primary.families,
        *bootstrap_arguments,
        resamples=permutation_count,
        seed=seed + 1,
    )
    chance_adjusted = holm_adjust(chance_raw)

    left_indices: list[int] = []
    right_indices: list[int] = []
    pair_values: list[np.ndarray] = []
    pair_valid: list[np.ndarray] = []
    for left in range(len(primary.model_ids)):
        for right in range(left + 1, len(primary.model_ids)):
            left_indices.append(left)
            right_indices.append(right)
            pair_values.append(primary.scores[left] - primary.scores[right])
            pair_valid.append(valid[left] & valid[right])
    pair_matrix = np.asarray(pair_values)
    pair_mask = np.asarray(pair_valid)
    pair_observed, raw_pvalues = sign_flip(
        pair_matrix,
        pair_mask,
        primary.families,
        *bootstrap_arguments,
        resamples=permutation_count,
        seed=seed + 2,
    )
    pair_bootstrap = (
        anchor_cluster_available_bootstrap(
            pair_matrix,
            pair_mask,
            primary.families,
            cluster_ids,
            resamples=bootstrap_count,
            seed=seed + 3,
        )
        if cluster_ids is not None
        else family_stratified_available_bootstrap(
            pair_matrix,
            pair_mask,
            primary.families,
            resamples=bootstrap_count,
            seed=seed + 3,
        )
    )
    adjusted_pvalues = holm_adjust(raw_pvalues)
    family_values = np.asarray(primary.families, dtype=object)
    pairwise: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(left_indices, right_indices, strict=True)):
        pairwise.append(
            {
                "left_index": left,
                "right_index": right,
                "left_model_id": primary.model_ids[left],
                "right_model_id": primary.model_ids[right],
                "shared_valid_tasks": int(pair_mask[index].sum()),
                "shared_valid_tasks_per_family": {
                    family: int((pair_mask[index] & (family_values == family)).sum())
                    for family in FAMILIES
                },
                "mean_difference": float(pair_observed[index]),
                "bootstrap_95_ci": _percentile_interval(pair_bootstrap[:, index]),
                "cohen_dz": _cohen_dz(pair_matrix[index, pair_mask[index]]),
                "sign_flip_p": float(raw_pvalues[index]),
                "holm_p": float(adjusted_pvalues[index]),
                "holm_significant": bool(adjusted_pvalues[index] < 0.05),
                "direction": (
                    "left_higher"
                    if pair_observed[index] > 0
                    else "right_higher"
                    if pair_observed[index] < 0
                    else "tie"
                ),
            }
        )

    all_models = np.ones(len(primary.model_ids), dtype=bool)
    rank_intervals = _rank_intervals(score_bootstrap, all_models)
    groups = _statistical_groups(point, all_models, pairwise)
    point_order = sorted(range(len(primary.model_ids)), key=lambda index: (-point[index], index))
    point_ranks = {index: rank + 1 for rank, index in enumerate(point_order)}

    repeat_results: list[dict[str, Any]] | None = None
    repeat_by_model: dict[str, Mapping[str, Any]] = {}
    if repeat is not None:
        if repeat_panel is None:
            raise SelectionPoweredAnalysisError("repeat data requires its task panel")
        repeat_results = analyze_repeatability(
            primary=primary,
            repeat=repeat,
            taskset=taskset,
            repeat_panel=repeat_panel,
            bootstrap_resamples=bootstrap_count,
            seed=seed + 4,
            cluster_ids=repeat_cluster_ids,
        )
        repeat_by_model = {str(row["model_id"]): row for row in repeat_results}

    models: list[dict[str, Any]] = []
    for index, model_id in enumerate(primary.model_ids):
        half_width = max_t_critical * standard_errors[index]
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_name": primary.model_names[index],
            "slot_id": primary.slot_ids[index],
            "score_status": "scored",
            "coverage": {
                "scheduled": len(primary.task_ids),
                "completed": int(primary.completed[index].sum()),
                "parseable": int(primary.parseable[index].sum()),
                "valid_scored": int(valid[index].sum()),
                "completion_rate": float(primary.completed[index].mean()),
                "parseable_rate": float(primary.parseable[index].mean()),
                "valid_scored_rate": float(valid[index].mean()),
                "valid_scored_per_family": {
                    family: int(family_counts[family][index]) for family in FAMILIES
                },
            },
            "flavourbench_score": float(point[index]),
            "failure_exclusion_sensitivity": {
                "scheduled_panel_worst_best_bounds": [
                    float(missing_lower[index]),
                    float(missing_upper[index]),
                ],
                "primary_score_uses_either_endpoint": False,
                "purpose": "bound conclusions under arbitrary missing-cell outcomes",
            },
            "family_scores": {family: float(family_scores[family][index]) for family in FAMILIES},
            "score_standard_error": float(standard_errors[index]),
            "score_pointwise_95_ci": _percentile_interval(score_bootstrap[:, index]),
            "score_simultaneous_95_ci": [
                float(point[index] - half_width),
                float(point[index] + half_width),
            ],
            "point_estimate_rank": point_ranks[index],
            "bootstrap_rank_95_interval": rank_intervals[index],
            "statistical_rank_group": groups[index],
            "chance_comparison": {
                "exact_chance_score_on_valid_tasks": float(chance_point[index]),
                "mean_difference": float(chance_observed[index]),
                "bootstrap_95_ci": _percentile_interval(
                    score_bootstrap[:, index] - chance_bootstrap[:, index]
                ),
                "sign_flip_p": float(chance_raw[index]),
                "holm_p": float(chance_adjusted[index]),
                "holm_significant_above_chance": bool(
                    chance_observed[index] > 0 and chance_adjusted[index] < 0.05
                ),
            },
        }
        if model_id in repeat_by_model:
            row["repeatability"] = dict(repeat_by_model[model_id])
        models.append(row)

    definitive_top_model_id: str | None = None
    if repeat_results is not None:
        leader = point_order[0]
        comparisons = [row for row in pairwise if leader in {row["left_index"], row["right_index"]}]
        leader_beats_all = len(comparisons) == len(primary.model_ids) - 1 and all(
            row["holm_significant"]
            and (
                (row["left_index"] == leader and row["mean_difference"] > 0)
                or (row["right_index"] == leader and row["mean_difference"] < 0)
            )
            for row in comparisons
        )
        repeat_value = repeat_by_model[primary.model_ids[leader]]["mean_ingredient_set_jaccard"]
        repeat_ok = repeat_value is not None and float(repeat_value) >= float(
            plan["repeatability"]["acceptance_floor"]
        )
        if leader_beats_all and repeat_ok:
            definitive_top_model_id = primary.model_ids[leader]

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "final_complete" if repeat is not None else "primary_complete_repeat_pending",
        "plan_sha256": plan["artifact_sha256"],
        "estimand": plan["outcomes"]["primary_definition"],
        "quality_scope": "successful_parseable_responses_only",
        "failure_handling": "excluded_from_quality_score_and_retained_in_coverage",
        "dnf_classification": False,
        "models": models,
        "pairwise_comparisons": pairwise,
        "repeatability": repeat_results,
        "definitive_top_model_id": definitive_top_model_id,
        "inference": {
            "bootstrap_resamples": bootstrap_count,
            "permutation_resamples": permutation_count,
            "familywise_alpha": 0.05,
            "pairwise_hypotheses": len(pairwise),
            "chance_hypotheses": len(primary.model_ids),
            "max_t_critical_value": max_t_critical,
            "seed": seed,
            "independence_unit": "anchor_ingredient" if cluster_ids is not None else "task",
            "independent_cluster_count": (
                len(set(cluster_ids)) if cluster_ids is not None else len(primary.task_ids)
            ),
            "shared_anchor_tasks_move_together": cluster_ids is not None,
            "pairwise_missingness": "shared successful parseable tasks only",
            "missing_cell_sensitivity": (
                "per-model scheduled-panel bounds assign excluded cells 0 or 100; "
                "neither endpoint enters the primary score"
            ),
        },
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _leaderboard_csv(analysis: Mapping[str, Any]) -> bytes:
    rows = []
    for model in analysis["models"]:
        repeat = model.get("repeatability") or {}
        coverage = model["coverage"]
        missing_bounds = model["failure_exclusion_sensitivity"]["scheduled_panel_worst_best_bounds"]
        rows.append(
            {
                "rank": model["point_estimate_rank"],
                "rank_group": model["statistical_rank_group"],
                "model": model["model_name"],
                "model_id": model["model_id"],
                "flavourbench_score": f"{model['flavourbench_score']:.6f}",
                "simultaneous_ci_low": f"{model['score_simultaneous_95_ci'][0]:.6f}",
                "simultaneous_ci_high": f"{model['score_simultaneous_95_ci'][1]:.6f}",
                "valid_scored": coverage["valid_scored"],
                "scheduled": coverage["scheduled"],
                "coverage": f"{coverage['valid_scored_rate']:.6f}",
                "missing_score_lower": f"{missing_bounds[0]:.6f}",
                "missing_score_upper": f"{missing_bounds[1]:.6f}",
                "completed": coverage["completed"],
                "parseable": coverage["parseable"],
                "repeat_jaccard": (
                    f"{repeat['mean_ingredient_set_jaccard']:.6f}"
                    if repeat.get("mean_ingredient_set_jaccard") is not None
                    else ""
                ),
            }
        )
    rows.sort(key=lambda row: (row["rank"], row["model_id"]))
    return _csv_bytes(rows, tuple(rows[0]))


def _pairwise_csv(analysis: Mapping[str, Any]) -> bytes:
    rows = [
        {
            "left_model_id": row["left_model_id"],
            "right_model_id": row["right_model_id"],
            "shared_valid_tasks": row["shared_valid_tasks"],
            "mean_difference": f"{row['mean_difference']:.6f}",
            "ci_low": f"{row['bootstrap_95_ci'][0]:.6f}",
            "ci_high": f"{row['bootstrap_95_ci'][1]:.6f}",
            "cohen_dz": "" if row["cohen_dz"] is None else f"{row['cohen_dz']:.8f}",
            "sign_flip_p": f"{row['sign_flip_p']:.10g}",
            "holm_p": f"{row['holm_p']:.10g}",
            "holm_significant": str(row["holm_significant"]).lower(),
            "direction": row["direction"],
        }
        for row in analysis["pairwise_comparisons"]
    ]
    return _csv_bytes(rows, tuple(rows[0]))


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--source-plan-v44", type=Path)
    parser.add_argument("--qwen-plan-v45", type=Path)
    parser.add_argument("--qwen-run-directory", type=Path)
    parser.add_argument("--fable-run-directory", type=Path)
    parser.add_argument("--source-plan-v49", type=Path)
    parser.add_argument("--luna-run-directory", type=Path)
    parser.add_argument("--deepseek-flash-run-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args(argv)
    taskset = _load(args.taskset)
    repeat_document = _load(args.repeat_panel)
    plan = _load(args.plan)
    plan_schema = plan.get("schema_version")
    if plan_schema in {
        PLAN_SCHEMA_VERSION_V46,
        PLAN_SCHEMA_VERSION_V49,
        PLAN_SCHEMA_VERSION_V52,
    }:
        taskset_valid = verify_taskset_replication_2(taskset)
        repeat_valid = verify_repeat_panel_replication_2(repeat_document, taskset=taskset)
        plan_valid = (
            verify_plan_v52(plan)
            if plan_schema == PLAN_SCHEMA_VERSION_V52
            else verify_plan_v49(plan)
            if plan_schema == PLAN_SCHEMA_VERSION_V49
            else verify_plan_v46(plan)
        )
    elif plan_schema in {PLAN_SCHEMA_VERSION_V44, PLAN_SCHEMA_VERSION_V47, PLAN_SCHEMA_VERSION_V50}:
        taskset_valid = verify_taskset(taskset)
        repeat_valid = verify_repeat_panel(repeat_document, taskset=taskset)
        plan_valid = (
            verify_plan_v50(plan)
            if plan_schema == PLAN_SCHEMA_VERSION_V50
            else verify_plan_v47(plan)
            if plan_schema == PLAN_SCHEMA_VERSION_V47
            else verify_plan_v44(plan)
        )
    else:
        taskset_valid = repeat_valid = plan_valid = False
    if not taskset_valid or not repeat_valid or not plan_valid:
        raise SelectionPoweredAnalysisError("analysis inputs failed semantic verification")
    for label, document, path in (
        ("taskset", taskset, args.taskset),
        ("repeat_panel", repeat_document, args.repeat_panel),
    ):
        recorded = plan["inputs"][label]
        if recorded["semantic_sha256"] != document["artifact_sha256"] or recorded[
            "physical_sha256"
        ] != _sha256_file(path):
            raise SelectionPoweredAnalysisError(f"plan {label} pin differs from exact input")

    composite_arguments = (
        args.source_plan_v44,
        args.qwen_plan_v45,
        args.qwen_run_directory,
    )
    panel_2_replacement_arguments = (
        args.source_plan_v49,
        args.luna_run_directory,
        args.deepseek_flash_run_directory,
    )
    model_sources: dict[str, tuple[Path, Mapping[str, Any]]] | None = None
    response_source_lineage: dict[str, Any]
    if plan_schema == PLAN_SCHEMA_VERSION_V52:
        if any(value is None for value in panel_2_replacement_arguments):
            raise SelectionPoweredAnalysisError(
                "v52 panel composite requires the exact v49 plan and both replacement runs"
            )
        if any(value is not None for value in (*composite_arguments, args.fable_run_directory)):
            raise SelectionPoweredAnalysisError("panel-1 composite sources cannot be used with v52")
        source_plan_path = args.source_plan_v49
        assert source_plan_path is not None
        source_plan = _load(source_plan_path)
        source_pin = plan["inputs"]["plan_v49_predecessor"]
        if (
            not verify_plan_v49(source_plan)
            or source_pin["semantic_sha256"] != source_plan["artifact_sha256"]
            or source_pin["physical_sha256"] != _sha256_file(source_plan_path)
        ):
            raise SelectionPoweredAnalysisError("v49 panel-2 source plan binding failed")
        assert args.luna_run_directory is not None
        assert args.deepseek_flash_run_directory is not None
        model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
        model_sources = {model_id: (args.run_directory, source_plan) for model_id in model_ids}
        model_sources[LUNA_MODEL_ID] = (args.luna_run_directory, plan)
        model_sources[DEEPSEEK_FLASH_MODEL_ID] = (args.deepseek_flash_run_directory, plan)
        response_source_lineage = {
            "schema_version": "flavourbench-score-blind-panel-2-composite-v1",
            "base_plan_sha256": source_plan["artifact_sha256"],
            "base_run_model_ids": [
                model_id for model_id in model_ids if model_id not in PANEL_2_REPLACEMENT_MODEL_IDS
            ],
            "replacement_plan_sha256": plan["artifact_sha256"],
            "replacement_run_model_ids": PANEL_2_REPLACEMENT_MODEL_IDS,
            "superseded_luna_responses_used": False,
            "superseded_deepseek_flash_responses_used": False,
            "cross_route_response_pooling": False,
            "selective_failed_cell_retry": False,
        }
    elif plan_schema in {PLAN_SCHEMA_VERSION_V47, PLAN_SCHEMA_VERSION_V50}:
        if any(value is not None for value in panel_2_replacement_arguments):
            raise SelectionPoweredAnalysisError(
                "panel-2 replacement sources cannot be used with a panel-1 composite"
            )
        if any(value is None for value in composite_arguments):
            raise SelectionPoweredAnalysisError(
                "panel composite requires the exact v44 source plan and v45 Qwen source"
            )
        if plan_schema == PLAN_SCHEMA_VERSION_V50 and args.fable_run_directory is None:
            raise SelectionPoweredAnalysisError("v50 requires the exact Fable replacement source")
        if plan_schema == PLAN_SCHEMA_VERSION_V47 and args.fable_run_directory is not None:
            raise SelectionPoweredAnalysisError("Fable replacement source requires a v50 plan")
        source_plan_path = args.source_plan_v44
        qwen_plan_path = args.qwen_plan_v45
        qwen_run_directory = args.qwen_run_directory
        assert source_plan_path is not None
        assert qwen_plan_path is not None
        assert qwen_run_directory is not None
        source_plan = _load(source_plan_path)
        qwen_plan = _load(qwen_plan_path)
        source_pin = plan["inputs"]["plan_v44_predecessor"]
        qwen_pin = plan["inputs"]["plan_v45_qwen_source"]
        if (
            not verify_plan_v44(source_plan)
            or source_pin["semantic_sha256"] != source_plan["artifact_sha256"]
            or source_pin["physical_sha256"] != _sha256_file(source_plan_path)
        ):
            raise SelectionPoweredAnalysisError("v44 panel-1 source plan binding failed")
        if (
            not verify_plan_v45(qwen_plan)
            or qwen_pin["semantic_sha256"] != qwen_plan["artifact_sha256"]
            or qwen_pin["physical_sha256"] != _sha256_file(qwen_plan_path)
        ):
            raise SelectionPoweredAnalysisError("v45 Qwen source plan binding failed")
        model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
        model_sources = {model_id: (args.run_directory, source_plan) for model_id in model_ids}
        model_sources[QWEN_MODEL_ID] = (qwen_run_directory, qwen_plan)
        if plan_schema == PLAN_SCHEMA_VERSION_V50:
            assert args.fable_run_directory is not None
            model_sources[FABLE_MODEL_ID] = (args.fable_run_directory, plan)
        response_source_lineage = {
            "schema_version": (
                "flavourbench-score-blind-panel-composite-v2"
                if plan_schema == PLAN_SCHEMA_VERSION_V50
                else "flavourbench-score-blind-panel-composite-v1"
            ),
            "base_plan_sha256": source_plan["artifact_sha256"],
            "base_run_model_ids": [
                model_id
                for model_id in model_ids
                if model_id
                not in (
                    {QWEN_MODEL_ID, FABLE_MODEL_ID}
                    if plan_schema == PLAN_SCHEMA_VERSION_V50
                    else {QWEN_MODEL_ID}
                )
            ],
            "replacement_plan_sha256": qwen_plan["artifact_sha256"],
            "replacement_run_model_ids": (
                [QWEN_MODEL_ID, FABLE_MODEL_ID]
                if plan_schema == PLAN_SCHEMA_VERSION_V50
                else [QWEN_MODEL_ID]
            ),
            "fable_replacement_plan_sha256": (
                plan["artifact_sha256"] if plan_schema == PLAN_SCHEMA_VERSION_V50 else None
            ),
            "superseded_qwen_responses_used": False,
            "superseded_fable_responses_used": False,
            "cross_route_response_pooling": False,
        }
    else:
        if any(
            value is not None
            for value in (
                *composite_arguments,
                args.fable_run_directory,
                *panel_2_replacement_arguments,
            )
        ):
            raise SelectionPoweredAnalysisError("composite source arguments require a v47 plan")
        response_source_lineage = {
            "schema_version": "flavourbench-single-fresh-response-source-v1",
            "plan_sha256": plan["artifact_sha256"],
            "model_ids": [str(row["model_id"]) for row in plan["roster"]["models"]],
            "predecessor_responses_used": False,
        }
    primary = load_panel(
        run_directory=args.run_directory,
        panel="primary",
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat_document,
        model_sources=model_sources,
    )
    repeat = None
    if not args.primary_only:
        repeat = load_panel(
            run_directory=args.run_directory,
            panel="repeat",
            plan=plan,
            taskset=taskset,
            repeat_panel=repeat_document,
            model_sources=model_sources,
        )
    analysis = analyze_panels(
        primary=primary,
        repeat=repeat,
        taskset=taskset,
        repeat_panel=repeat_document,
        plan=plan,
    )
    leaderboard_bytes = _leaderboard_csv(analysis)
    pairwise_bytes = _pairwise_csv(analysis)
    leaderboard_path = _write_content_addressed(
        args.output_directory, "flavourbench-leaderboard-table", leaderboard_bytes
    )
    pairwise_path = _write_content_addressed(
        args.output_directory, "flavourbench-pairwise-table", pairwise_bytes
    )
    release: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": analysis["status"],
        "benchmark": "FlavourBench",
        "track": "Epicure-scored combinatorial culinary decisions",
        "inputs": {
            "plan": {
                "semantic_sha256": plan["artifact_sha256"],
                "physical_sha256": _sha256_file(args.plan),
            },
            "taskset": {
                "semantic_sha256": taskset["artifact_sha256"],
                "physical_sha256": _sha256_file(args.taskset),
            },
            "repeat_panel": {
                "semantic_sha256": repeat_document["artifact_sha256"],
                "physical_sha256": _sha256_file(args.repeat_panel),
            },
            "primary_responses": {
                "count": len(primary.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(primary.response_artifact_sha256s)),
                "spend_micros": primary.spend_micros,
            },
            "repeat_responses": (
                {
                    "count": len(repeat.response_artifact_sha256s),
                    "artifact_set_sha256": _sha256(list(repeat.response_artifact_sha256s)),
                    "spend_micros": repeat.spend_micros,
                }
                if repeat is not None
                else None
            ),
            "model_response_sources": response_source_lineage,
        },
        "tables": {
            "leaderboard": {
                "filename": leaderboard_path.name,
                "sha256": _sha256_bytes(leaderboard_bytes),
            },
            "pairwise": {
                "filename": pairwise_path.name,
                "sha256": _sha256_bytes(pairwise_bytes),
            },
        },
        "analysis": analysis,
        "claim_boundary": plan["claim_boundary"],
    }
    release["artifact_sha256"] = _sha256(release)
    release_bytes = (
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    release_path = _write_content_addressed(
        args.output_directory,
        "flavourbench-powered-release",
        release_bytes,
        address=release["artifact_sha256"],
    )
    print(
        json.dumps(
            {
                "release": str(release_path),
                "artifact_sha256": release["artifact_sha256"],
                "leaderboard": str(leaderboard_path),
                "pairwise": str(pairwise_path),
                "status": release["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
