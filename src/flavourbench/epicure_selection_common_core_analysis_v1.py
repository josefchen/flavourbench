"""Analyze a complete, score-blindly frozen FlavourBench common core."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_selection_powered_analysis import (
    PanelData,
    SelectionPoweredAnalysisError,
    _cohen_dz,
    _percentile_interval,
    _rank_intervals,
    _sha256,
    _statistical_groups,
    _verify_semantic,
    _zero_scoring,
    holm_adjust,
)
from .epicure_selection_taskset_v1 import score_answer
from .selection_response_parser_v3 import score_answer_v3

type SourceSpec = (
    tuple[Path | Sequence[Path], Mapping[str, Any]] | Sequence[tuple[Path, Mapping[str, Any]]]
)


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredAnalysisError(f"input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionPoweredAnalysisError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionPoweredAnalysisError(f"input is not a JSON object: {path}")
    return value


def _source_items(value: SourceSpec) -> tuple[tuple[Path, Mapping[str, Any]], ...]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        directory_value, plan = value
        directories = (
            (directory_value,) if isinstance(directory_value, Path) else tuple(directory_value)
        )
        items = tuple((directory, plan) for directory in directories)
    else:
        items = tuple(value)
    if not items or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], Path)
        or not isinstance(item[1], Mapping)
        for item in items
    ):
        raise SelectionPoweredAnalysisError("response source order is malformed")
    directories = tuple(item[0] for item in items)
    if len(set(directories)) != len(directories):
        raise SelectionPoweredAnalysisError("response source directory is duplicated")
    return items


def load_complete_common_core(
    *,
    panel: str,
    plan: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    task_ids: Sequence[str],
    model_sources: Mapping[str, SourceSpec],
    allowed_source_roster_differences: Mapping[str, frozenset[str]] | None = None,
    analysis_score_function: Callable[[Mapping[str, Any], str], Mapping[str, Any]] = (
        score_answer_v3
    ),
) -> PanelData:
    """Load only frozen common-core tasks and reject any missing/invalid ranked cell."""

    if panel != "primary":
        raise SelectionPoweredAnalysisError("common-core ranking uses the primary panel")
    roster = list(plan["roster"]["models"])
    all_tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
    ordered_tasks = tuple(str(task_id) for task_id in task_ids)
    if not ordered_tasks or len(set(ordered_tasks)) != len(ordered_tasks):
        raise SelectionPoweredAnalysisError("common-core task order is empty or duplicated")
    if any(task_id not in all_tasks for task_id in ordered_tasks):
        raise SelectionPoweredAnalysisError("common-core task is absent from its taskset")
    tasks = {task_id: all_tasks[task_id] for task_id in ordered_tasks}
    model_by_id = {str(row["model_id"]): row for row in roster}
    if set(model_sources) != set(model_by_id):
        raise SelectionPoweredAnalysisError("common-core response-source roster differs")

    candidates: dict[tuple[str, str], list[tuple[int, Path, dict[str, Any]]]] = {}
    analysis_scoring: dict[str, Mapping[str, Any]] = {}
    for model_id, roster_row in model_by_id.items():
        for priority, (directory, source_plan) in enumerate(_source_items(model_sources[model_id])):
            source_rows = {str(row["model_id"]): row for row in source_plan["roster"]["models"]}
            source_row = source_rows.get(model_id)
            if source_row is None:
                raise SelectionPoweredAnalysisError(
                    f"{model_id} is absent from a common-core source plan"
                )
            differing_fields = {
                key
                for key in set(source_row) | set(roster_row)
                if source_row.get(key) != roster_row.get(key)
            }
            allowed = (allowed_source_roster_differences or {}).get(model_id, frozenset())
            if not differing_fields <= allowed:
                raise SelectionPoweredAnalysisError(
                    f"{model_id} source roster binding differs: {sorted(differing_fields)}"
                )
            seen_in_source: set[str] = set()
            response_directory = directory / "responses" / panel / str(source_row["slot_id"])
            for path in sorted(response_directory.glob("response-*.json")):
                document = _load(path)
                task_id = str(document.get("task_id") or "")
                if task_id not in tasks:
                    continue
                if task_id in seen_in_source:
                    raise SelectionPoweredAnalysisError(
                        f"response cell is duplicated within one source: {(model_id, task_id)}"
                    )
                seen_in_source.add(task_id)
                if not _verify_semantic(document):
                    raise SelectionPoweredAnalysisError(f"response semantic hash failed: {path}")
                artifact = str(document["artifact_sha256"])
                cell_id = str(document.get("cell_id") or "")
                if path.name != f"response-{cell_id}-{artifact}.json":
                    raise SelectionPoweredAnalysisError(
                        f"response filename is not content addressed: {path}"
                    )
                task = tasks[task_id]
                exact = {
                    "schema_version": "flavourbench-powered-response-v1",
                    "panel": panel,
                    "plan_sha256": source_plan["artifact_sha256"],
                    "manifest_sha256": source_plan["inputs"]["route_manifest"]["semantic_sha256"],
                    "taskset_sha256": taskset["artifact_sha256"],
                    "repeat_panel_sha256": repeat_panel["artifact_sha256"],
                    "family": task["family"],
                    "model_id": model_id,
                    "slot_id": source_row["slot_id"],
                    "model_name": source_row["model_name"],
                    "canonical_model_slug": source_row["canonical_model_slug"],
                    "execution_backend": source_row["execution_backend"],
                    "endpoint_execution_sha256": source_row["endpoint_execution_sha256"],
                    "backend_contract_sha256": source_row["backend_contract_sha256"],
                    "prompt_sha256": task["prompt_sha256"],
                    "optimal_selection": task["optimal_selection"],
                    "original_task_id": task.get("original_task_id"),
                }
                if any(document.get(key) != value for key, value in exact.items()):
                    raise SelectionPoweredAnalysisError(
                        f"response binding differs from frozen inputs: {path}"
                    )
                if path.parent.name != document["slot_id"]:
                    raise SelectionPoweredAnalysisError(
                        f"response is stored under the wrong slot: {path}"
                    )
                status = document.get("status")
                generation = document.get("generation")
                if status == "completed":
                    if not isinstance(generation, dict) or not isinstance(
                        generation.get("answer_markdown"), str
                    ):
                        raise SelectionPoweredAnalysisError("completed response lacks answer bytes")
                    historical = score_answer(task, generation["answer_markdown"])
                    rescored = analysis_score_function(task, generation["answer_markdown"])
                elif status == "failed":
                    historical = rescored = _zero_scoring(task)
                else:
                    raise SelectionPoweredAnalysisError(f"unsupported response status: {status}")
                if document.get("scoring") != historical:
                    raise SelectionPoweredAnalysisError(
                        f"historical response score does not reproduce: {path}"
                    )
                analysis_scoring[artifact] = rescored
                candidates.setdefault((model_id, task_id), []).append((priority, path, document))

    expected = {(model_id, task_id) for model_id in model_by_id for task_id in ordered_tasks}
    if set(candidates) != expected:
        missing = sorted(expected - set(candidates))
        raise SelectionPoweredAnalysisError(
            f"common-core response key set is incomplete: {missing[:3]}"
        )

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[str] = []
    spend_micros = 0
    for key, rows in candidates.items():
        ordered = sorted(rows, key=lambda value: (value[0], str(value[1])))
        valid = [
            row
            for row in ordered
            if row[2]["status"] == "completed"
            and analysis_scoring[str(row[2]["artifact_sha256"])]["parseable"] is True
        ]
        if not valid:
            raise SelectionPoweredAnalysisError(f"common-core cell is not valid: {key}")
        document = valid[0][2]
        selected[key] = document
        artifacts.append(str(document["artifact_sha256"]))
        spend_micros += int((document.get("generation") or {}).get("cost_micros") or 0)

    model_ids = tuple(model_by_id)
    scores = np.empty((len(model_ids), len(ordered_tasks)), dtype=np.float64)
    selections: list[tuple[str | None, ...]] = []
    for model_index, model_id in enumerate(model_ids):
        model_selections: list[str | None] = []
        for task_index, task_id in enumerate(ordered_tasks):
            document = selected[(model_id, task_id)]
            scoring = analysis_scoring[str(document["artifact_sha256"])]
            scores[model_index, task_index] = float(scoring["score"])
            model_selections.append(str(scoring["observed_selection"]))
        selections.append(tuple(model_selections))
    complete = np.ones_like(scores, dtype=bool)
    return PanelData(
        panel=panel,
        model_ids=model_ids,
        model_names=tuple(str(model_by_id[model_id]["model_name"]) for model_id in model_ids),
        slot_ids=tuple(str(model_by_id[model_id]["slot_id"]) for model_id in model_ids),
        task_ids=ordered_tasks,
        families=tuple(str(tasks[task_id]["family"]) for task_id in ordered_tasks),
        scores=scores,
        completed=complete.copy(),
        parseable=complete.copy(),
        selections=tuple(selections),
        response_artifact_sha256s=tuple(sorted(artifacts)),
        spend_micros=spend_micros,
    )


def equal_family_mean(
    values: np.ndarray,
    families: Sequence[str],
    family_order: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != len(families) or not family_order:
        raise SelectionPoweredAnalysisError("common-core family matrix differs")
    family_values = np.asarray(families, dtype=object)
    parts: list[np.ndarray] = []
    for family in family_order:
        indices = np.flatnonzero(family_values == family)
        if not len(indices):
            raise SelectionPoweredAnalysisError(f"missing common-core family: {family}")
        parts.append(matrix[:, indices].mean(axis=1))
    return np.stack(parts, axis=1).mean(axis=1)


def anchor_cluster_bootstrap(
    values: np.ndarray,
    families: Sequence[str],
    family_order: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != len(families) or len(cluster_ids) != len(families):
        raise SelectionPoweredAnalysisError("common-core bootstrap vectors differ")
    if resamples <= 0:
        raise SelectionPoweredAnalysisError("bootstrap resamples must be positive")
    ordered_clusters = tuple(dict.fromkeys(str(value) for value in cluster_ids))
    if not ordered_clusters or any(not value for value in ordered_clusters):
        raise SelectionPoweredAnalysisError("anchor cluster IDs must be nonempty")
    cluster_index = {value: index for index, value in enumerate(ordered_clusters)}
    task_clusters = np.asarray([cluster_index[str(value)] for value in cluster_ids])
    family_values = np.asarray(families, dtype=object)
    totals: list[tuple[np.ndarray, np.ndarray]] = []
    for family in family_order:
        numerators = np.zeros((matrix.shape[0], len(ordered_clusters)), dtype=np.float64)
        denominators = np.zeros(len(ordered_clusters), dtype=np.float64)
        for task_index in np.flatnonzero(family_values == family):
            anchor_index = task_clusters[task_index]
            numerators[:, anchor_index] += matrix[:, task_index]
            denominators[anchor_index] += 1.0
        totals.append((numerators, denominators))
    rng = np.random.default_rng(seed)
    output = np.empty((resamples, matrix.shape[0]), dtype=np.float64)
    probabilities = np.full(len(ordered_clusters), 1.0 / len(ordered_clusters))
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        counts = rng.multinomial(len(ordered_clusters), probabilities, size=stop - start)
        batch = np.zeros((stop - start, matrix.shape[0]), dtype=np.float64)
        for _ in range(100):
            bad = np.zeros(stop - start, dtype=bool)
            for _, denominators in totals:
                bad |= (denominators @ counts.T) == 0
            if not np.any(bad):
                break
            counts[bad] = rng.multinomial(len(ordered_clusters), probabilities, size=int(bad.sum()))
        else:
            raise SelectionPoweredAnalysisError("bootstrap sampled an empty family")
        for numerators, denominators in totals:
            batch += ((numerators @ counts.T) / (denominators @ counts.T)[None, :]).T / len(
                family_order
            )
        output[start:stop] = batch
    return output


def anchor_cluster_sign_flip(
    values: np.ndarray,
    families: Sequence[str],
    family_order: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != len(families) or len(cluster_ids) != len(families):
        raise SelectionPoweredAnalysisError("common-core sign-flip vectors differ")
    ordered_clusters = tuple(dict.fromkeys(str(value) for value in cluster_ids))
    cluster_index = {value: index for index, value in enumerate(ordered_clusters)}
    family_values = np.asarray(families, dtype=object)
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    for family in family_order:
        indices = np.flatnonzero(family_values == family)
        if not len(indices):
            raise SelectionPoweredAnalysisError(f"missing common-core family: {family}")
        weights[indices] = 1.0 / (len(family_order) * len(indices))
    clustered = np.zeros((matrix.shape[0], len(ordered_clusters)), dtype=np.float64)
    for task_index, cluster_id in enumerate(cluster_ids):
        clustered[:, cluster_index[str(cluster_id)]] += matrix[:, task_index] * weights[task_index]
    observed = clustered.sum(axis=1)
    threshold = np.abs(observed)
    exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, batch_size):
        width = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(width, len(ordered_clusters)), dtype=np.int8)
        null = (signs.astype(np.float64) * 2.0 - 1.0) @ clustered.T
        exceed += np.count_nonzero(np.abs(null) >= threshold[None, :] - 1e-12, axis=0)
    return observed, (exceed + 1.0) / (resamples + 1.0)


def _panel_stability(
    data: PanelData,
    panel_ids: Sequence[str],
    family_order: Sequence[str],
) -> dict[str, Any]:
    if len(panel_ids) != len(data.task_ids) or len(set(panel_ids)) != 2:
        raise SelectionPoweredAnalysisError("common-core panel labels differ")
    panel_values = np.asarray(panel_ids, dtype=object)
    scores: dict[str, np.ndarray] = {}
    for panel in dict.fromkeys(panel_ids):
        indices = np.flatnonzero(panel_values == panel)
        scores[panel] = equal_family_mean(
            data.scores[:, indices],
            np.asarray(data.families, dtype=object)[indices],
            family_order,
        )
    left, right = tuple(scores)
    left_values = scores[left]
    right_values = scores[right]
    left_rank = np.argsort(np.argsort(left_values, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right_values, kind="stable"), kind="stable")
    return {
        "status": "prespecified_descriptive_replication_diagnostic",
        "panels": [left, right],
        "score_pearson": float(np.corrcoef(left_values, right_values)[0, 1]),
        "rank_spearman": float(np.corrcoef(left_rank, right_rank)[0, 1]),
        "models": [
            {
                "model_id": model_id,
                left: float(left_values[index]),
                right: float(right_values[index]),
                "difference": float(right_values[index] - left_values[index]),
            }
            for index, model_id in enumerate(data.model_ids)
        ],
    }


def analyze_complete_common_core(
    *,
    primary: PanelData,
    taskset: Mapping[str, Any],
    plan: Mapping[str, Any],
    family_order: Sequence[str],
    cluster_ids: Sequence[str],
    panel_ids: Sequence[str],
    bootstrap_resamples: int | None = None,
    permutation_resamples: int | None = None,
) -> dict[str, Any]:
    """Compute complete-case scores, simultaneous bands, and all paired tests."""

    if not np.all(primary.completed & primary.parseable):
        raise SelectionPoweredAnalysisError("common-core matrix is not completely valid")
    inference = plan["inference"]
    bootstrap_count = int(bootstrap_resamples or inference["bootstrap_resamples"])
    permutation_count = int(permutation_resamples or inference["permutation_resamples"])
    seed = int(inference["seed"])
    point = equal_family_mean(primary.scores, primary.families, family_order)
    bootstrap = anchor_cluster_bootstrap(
        primary.scores,
        primary.families,
        family_order,
        cluster_ids,
        resamples=bootstrap_count,
        seed=seed,
    )
    standard_errors = np.std(bootstrap, axis=0, ddof=1)
    safe_se = np.where(standard_errors > 0, standard_errors, 1.0)
    max_t = np.max(np.abs((bootstrap - point[None, :]) / safe_se[None, :]), axis=1)
    max_t_critical = float(np.quantile(max_t, 0.95))

    task_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    chance = np.asarray(
        [float(task_by_id[task_id]["chance_score_bps"]) / 100 for task_id in primary.task_ids]
    )
    chance_matrix = np.broadcast_to(chance, primary.scores.shape)
    chance_point = equal_family_mean(chance_matrix, primary.families, family_order)
    chance_bootstrap = anchor_cluster_bootstrap(
        chance_matrix,
        primary.families,
        family_order,
        cluster_ids,
        resamples=bootstrap_count,
        seed=seed,
    )
    chance_observed, chance_raw = anchor_cluster_sign_flip(
        primary.scores - chance_matrix,
        primary.families,
        family_order,
        cluster_ids,
        resamples=permutation_count,
        seed=seed + 1,
    )
    chance_adjusted = holm_adjust(chance_raw)

    left_indices: list[int] = []
    right_indices: list[int] = []
    differences: list[np.ndarray] = []
    for left in range(len(primary.model_ids)):
        for right in range(left + 1, len(primary.model_ids)):
            left_indices.append(left)
            right_indices.append(right)
            differences.append(primary.scores[left] - primary.scores[right])
    pair_matrix = np.asarray(differences)
    pair_observed, pair_raw = anchor_cluster_sign_flip(
        pair_matrix,
        primary.families,
        family_order,
        cluster_ids,
        resamples=permutation_count,
        seed=seed + 2,
    )
    pair_bootstrap = anchor_cluster_bootstrap(
        pair_matrix,
        primary.families,
        family_order,
        cluster_ids,
        resamples=bootstrap_count,
        seed=seed + 3,
    )
    pair_adjusted = holm_adjust(pair_raw)
    family_values = np.asarray(primary.families, dtype=object)
    pairwise: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(left_indices, right_indices, strict=True)):
        pairwise.append(
            {
                "left_index": left,
                "right_index": right,
                "left_model_id": primary.model_ids[left],
                "right_model_id": primary.model_ids[right],
                "shared_valid_tasks": len(primary.task_ids),
                "shared_valid_tasks_per_family": {
                    family: int(np.count_nonzero(family_values == family))
                    for family in family_order
                },
                "mean_difference": float(pair_observed[index]),
                "bootstrap_95_ci": _percentile_interval(pair_bootstrap[:, index]),
                "cohen_dz": _cohen_dz(pair_matrix[index]),
                "sign_flip_p": float(pair_raw[index]),
                "holm_p": float(pair_adjusted[index]),
                "holm_significant": bool(pair_adjusted[index] < 0.05),
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
    rank_intervals = _rank_intervals(bootstrap, all_models)
    groups = _statistical_groups(point, all_models, pairwise)
    point_order = sorted(range(len(primary.model_ids)), key=lambda index: (-point[index], index))
    point_ranks = {model_index: rank + 1 for rank, model_index in enumerate(point_order)}
    models: list[dict[str, Any]] = []
    for index, model_id in enumerate(primary.model_ids):
        half_width = max_t_critical * standard_errors[index]
        family_scores = {
            family: float(primary.scores[index, np.flatnonzero(family_values == family)].mean())
            for family in family_order
        }
        models.append(
            {
                "model_id": model_id,
                "model_name": primary.model_names[index],
                "slot_id": primary.slot_ids[index],
                "score_status": "scored_complete_common_core",
                "coverage": {
                    "scheduled": len(primary.task_ids),
                    "completed": len(primary.task_ids),
                    "parseable": len(primary.task_ids),
                    "valid_scored": len(primary.task_ids),
                    "valid_scored_rate": 1.0,
                    "valid_scored_per_family": {
                        family: int(np.count_nonzero(family_values == family))
                        for family in family_order
                    },
                },
                "flavourbench_score": float(point[index]),
                "family_scores": family_scores,
                "score_standard_error": float(standard_errors[index]),
                "score_pointwise_95_ci": _percentile_interval(bootstrap[:, index]),
                "score_simultaneous_95_ci": [
                    float(point[index] - half_width),
                    float(point[index] + half_width),
                ],
                "point_estimate_rank": point_ranks[index],
                "bootstrap_rank_95_interval": rank_intervals[index],
                "statistical_rank_group": groups[index],
                "chance_comparison": {
                    "exact_chance_score": float(chance_point[index]),
                    "mean_difference": float(chance_observed[index]),
                    "bootstrap_95_ci": _percentile_interval(
                        bootstrap[:, index] - chance_bootstrap[:, index]
                    ),
                    "sign_flip_p": float(chance_raw[index]),
                    "holm_p": float(chance_adjusted[index]),
                    "holm_significant_above_chance": bool(
                        chance_observed[index] > 0 and chance_adjusted[index] < 0.05
                    ),
                },
            }
        )

    leader = point_order[0]
    leader_comparisons = [
        row for row in pairwise if leader in {row["left_index"], row["right_index"]}
    ]
    unique_top = len(leader_comparisons) == len(primary.model_ids) - 1 and all(
        row["holm_significant"]
        and (
            (row["left_index"] == leader and row["mean_difference"] > 0)
            or (row["right_index"] == leader and row["mean_difference"] < 0)
        )
        for row in leader_comparisons
    )
    return {
        "schema_version": "flavourbench-selection-common-core-analysis-v1",
        "status": "final_complete_common_core",
        "plan_sha256": plan["artifact_sha256"],
        "estimand": plan["common_core"]["estimand_label"],
        "quality_scope": "complete_valid_score_blindly_selected_common_core",
        "failure_handling": "no failed or unparseable cell enters the analytic matrix",
        "dnf_rows_emitted": False,
        "models": models,
        "pairwise_comparisons": pairwise,
        "resolved_pair_count": int(sum(row["holm_significant"] for row in pairwise)),
        "definitive_top_model_id": primary.model_ids[leader] if unique_top else None,
        "panel_replication": _panel_stability(primary, panel_ids, family_order),
        "response_artifact_count": len(primary.response_artifact_sha256s),
        "response_artifact_set_sha256": _sha256(list(primary.response_artifact_sha256s)),
        "inference": {
            "bootstrap_resamples": bootstrap_count,
            "permutation_resamples": permutation_count,
            "familywise_alpha": 0.05,
            "pairwise_hypotheses": len(pairwise),
            "chance_hypotheses": len(primary.model_ids),
            "max_t_critical_value": max_t_critical,
            "seed": seed,
            "independence_unit": "anchor_ingredient",
            "independent_cluster_count": len(set(cluster_ids)),
            "shared_anchor_tasks_move_together": True,
            "complete_case_matrix": True,
        },
    }
