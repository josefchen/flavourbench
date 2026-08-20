"""Analyze the frozen powered Epicure-selection FlavourBench run.

The implementation follows the analysis plan frozen before primary collection:
family-stratified shared-task bootstrap intervals, simultaneous max-t score
bands, all paired sign-flip tests with Holm correction, intention-to-evaluate
zeroes, and a label-permutation repeatability panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_selection_powered_plan import (
    FAMILIES,
    MODEL_COUNT,
    REPEAT_TASK_COUNT,
    TASK_COUNT,
    verify_repeat_panel,
)
from .epicure_selection_powered_plan import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V17
from .epicure_selection_powered_plan import verify_plan as verify_plan_v17
from .epicure_selection_powered_plan_v18 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V18
from .epicure_selection_powered_plan_v18 import verify_plan as verify_plan_v18
from .epicure_selection_powered_plan_v19 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V19
from .epicure_selection_powered_plan_v19 import verify_plan as verify_plan_v19
from .epicure_selection_powered_plan_v20 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V20
from .epicure_selection_powered_plan_v20 import verify_plan as verify_plan_v20
from .epicure_selection_powered_plan_v21 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V21
from .epicure_selection_powered_plan_v21 import verify_plan as verify_plan_v21
from .epicure_selection_powered_plan_v22 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V22
from .epicure_selection_powered_plan_v22 import verify_plan as verify_plan_v22
from .epicure_selection_powered_plan_v23 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V23
from .epicure_selection_powered_plan_v23 import verify_plan as verify_plan_v23
from .epicure_selection_powered_plan_v24 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V24
from .epicure_selection_powered_plan_v24 import verify_plan as verify_plan_v24
from .epicure_selection_powered_plan_v25 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V25
from .epicure_selection_powered_plan_v25 import verify_plan as verify_plan_v25
from .epicure_selection_powered_plan_v26 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V26
from .epicure_selection_powered_plan_v26 import verify_plan as verify_plan_v26
from .epicure_selection_powered_plan_v27 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V27
from .epicure_selection_powered_plan_v27 import verify_plan as verify_plan_v27
from .epicure_selection_powered_plan_v28 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V28
from .epicure_selection_powered_plan_v28 import verify_plan as verify_plan_v28
from .epicure_selection_powered_plan_v29 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V29
from .epicure_selection_powered_plan_v29 import verify_plan as verify_plan_v29
from .epicure_selection_powered_plan_v30 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V30
from .epicure_selection_powered_plan_v30 import verify_plan as verify_plan_v30
from .epicure_selection_powered_plan_v31 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V31
from .epicure_selection_powered_plan_v31 import verify_plan as verify_plan_v31
from .epicure_selection_powered_plan_v32 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V32
from .epicure_selection_powered_plan_v32 import verify_plan as verify_plan_v32
from .epicure_selection_powered_plan_v33 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V33
from .epicure_selection_powered_plan_v33 import verify_plan as verify_plan_v33
from .epicure_selection_powered_plan_v34 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V34
from .epicure_selection_powered_plan_v34 import verify_plan as verify_plan_v34
from .epicure_selection_powered_plan_v35 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V35
from .epicure_selection_powered_plan_v35 import verify_plan as verify_plan_v35
from .epicure_selection_powered_plan_v36 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V36
from .epicure_selection_powered_plan_v36 import verify_plan as verify_plan_v36
from .epicure_selection_powered_plan_v37 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V37
from .epicure_selection_powered_plan_v37 import verify_plan as verify_plan_v37
from .epicure_selection_powered_plan_v38 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V38
from .epicure_selection_powered_plan_v38 import verify_plan as verify_plan_v38
from .epicure_selection_powered_plan_v39 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V39
from .epicure_selection_powered_plan_v39 import verify_plan as verify_plan_v39
from .epicure_selection_powered_plan_v40 import DEEPSEEK_ID as V40_DEEPSEEK_ID
from .epicure_selection_powered_plan_v40 import FABLE_ID as V40_FABLE_ID
from .epicure_selection_powered_plan_v40 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V40
from .epicure_selection_powered_plan_v40 import verify_plan as verify_plan_v40
from .epicure_selection_powered_plan_v41 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V41
from .epicure_selection_powered_plan_v41 import verify_plan as verify_plan_v41
from .epicure_selection_powered_plan_v42 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V42
from .epicure_selection_powered_plan_v42 import verify_plan as verify_plan_v42
from .epicure_selection_powered_plan_v43 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V43
from .epicure_selection_powered_plan_v43 import verify_plan as verify_plan_v43
from .epicure_selection_taskset_v1 import score_answer, verify_taskset

ANALYSIS_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-v1"
RELEASE_SCHEMA_VERSION = "flavourbench-selection-powered-release-v1"


class SelectionPoweredAnalysisError(RuntimeError):
    """The frozen powered-run analysis failed closed."""


@dataclass(frozen=True)
class PanelData:
    """Verified response panel aligned as model by task matrices."""

    panel: str
    model_ids: tuple[str, ...]
    model_names: tuple[str, ...]
    slot_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    families: tuple[str, ...]
    scores: np.ndarray
    completed: np.ndarray
    parseable: np.ndarray
    selections: tuple[tuple[str | None, ...], ...]
    response_artifact_sha256s: tuple[str, ...]
    spend_micros: int


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _verify_semantic(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def _zero_scoring(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observed_selection": None,
        "optimal_selection": task["optimal_selection"],
        "parseable": False,
        "score_bps": 0,
        "score": 0.0,
        "optimal": False,
    }


def load_panel(
    *,
    run_directory: Path,
    panel: str,
    plan: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    model_sources: Mapping[
        str,
        tuple[Path | Sequence[Path], Mapping[str, Any]] | Sequence[tuple[Path, Mapping[str, Any]]],
    ]
    | None = None,
    allowed_source_roster_differences: Mapping[str, frozenset[str]] | None = None,
    analysis_score_function: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
) -> PanelData:
    """Load, content-verify, rescore, and align one complete response panel.

    Stored scoring is always reproduced with the historical parser that wrote
    each immutable response artifact.  A later, explicitly frozen analysis
    parser may then be supplied for candidate selection and score matrices;
    this never rewrites or weakens verification of the source artifacts.
    """
    if panel not in {"primary", "repeat"}:
        raise SelectionPoweredAnalysisError("panel must be primary or repeat")
    roster = list(plan["roster"]["models"])
    tasks = list(taskset["tasks"] if panel == "primary" else repeat_panel["tasks"])
    expected_model_count = int(plan.get("roster", {}).get("model_count", MODEL_COUNT))
    if len(roster) != expected_model_count:
        raise SelectionPoweredAnalysisError("analysis plan roster has the wrong cardinality")
    if len(tasks) != (TASK_COUNT if panel == "primary" else REPEAT_TASK_COUNT):
        raise SelectionPoweredAnalysisError("analysis task panel has the wrong cardinality")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    model_by_id = {str(row["model_id"]): row for row in roster}
    source_plan_by_path: dict[Path, Mapping[str, Any]] = {}
    paths: list[Path] = []
    source_priority_by_path: dict[Path, int] = {}
    for model_id, roster_row in model_by_id.items():
        source_value = (
            model_sources.get(model_id, (run_directory, plan))
            if model_sources is not None
            else (run_directory, plan)
        )
        if (
            isinstance(source_value, tuple)
            and len(source_value) == 2
            and isinstance(source_value[1], Mapping)
        ):
            source_directory_value, source_plan = source_value
            source_directories = (
                (source_directory_value,)
                if isinstance(source_directory_value, Path)
                else tuple(source_directory_value)
            )
            source_items = tuple((directory, source_plan) for directory in source_directories)
        else:
            source_items = tuple(source_value)
        if not source_items or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], Path)
            or not isinstance(item[1], Mapping)
            for item in source_items
        ):
            raise SelectionPoweredAnalysisError(
                f"{model_id} response source order is empty, duplicated, or malformed"
            )
        source_directories = tuple(item[0] for item in source_items)
        if len(set(source_directories)) != len(source_directories):
            raise SelectionPoweredAnalysisError(
                f"{model_id} response source order is empty, duplicated, or malformed"
            )
        model_paths: list[Path] = []
        for priority, (source_directory, source_plan) in enumerate(source_items):
            source_rows = {str(row["model_id"]): row for row in source_plan["roster"]["models"]}
            source_row = source_rows.get(model_id)
            if source_row is None:
                raise SelectionPoweredAnalysisError(
                    f"{model_id} is absent from a response source plan"
                )
            differing_fields = {
                key
                for key in set(source_row) | set(roster_row)
                if source_row.get(key) != roster_row.get(key)
            }
            allowed_differences = (allowed_source_roster_differences or {}).get(
                model_id, frozenset()
            )
            if not differing_fields <= allowed_differences:
                raise SelectionPoweredAnalysisError(
                    f"{model_id} source roster binding differs from the analysis plan"
                )
            directory_paths = sorted(
                (source_directory / "responses" / panel / str(source_row["slot_id"])).glob(
                    "response-*.json"
                )
            )
            for path in directory_paths:
                if path in source_priority_by_path:
                    raise SelectionPoweredAnalysisError(
                        f"response path is reused by multiple sources: {path}"
                    )
                source_priority_by_path[path] = priority
                source_plan_by_path[path] = source_plan
            model_paths.extend(directory_paths)
        if len(model_paths) < len(tasks):
            raise SelectionPoweredAnalysisError(
                f"{panel} panel for {model_id} is incomplete: "
                f"observed {len(model_paths)} candidate responses, expected at least {len(tasks)}"
            )
        paths.extend(model_paths)

    candidates_by_key: dict[tuple[str, str], list[tuple[int, Path, dict[str, Any]]]] = {}
    analysis_scoring_by_artifact: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        document = _load(path)
        if not _verify_semantic(document):
            raise SelectionPoweredAnalysisError(f"response semantic hash failed: {path}")
        artifact = str(document["artifact_sha256"])
        cell_id = str(document.get("cell_id") or "")
        if path.name != f"response-{cell_id}-{artifact}.json":
            raise SelectionPoweredAnalysisError(
                f"response filename is not content addressed: {path}"
            )
        model_id = str(document.get("model_id") or "")
        task_id = str(document.get("task_id") or "")
        key = (model_id, task_id)
        if model_id not in model_by_id or task_id not in task_by_id:
            raise SelectionPoweredAnalysisError(f"unexpected response cell: {key}")
        roster_row = model_by_id[model_id]
        source_plan = source_plan_by_path[path]
        task = task_by_id[task_id]
        exact_fields = {
            "schema_version": "flavourbench-powered-response-v1",
            "panel": panel,
            "plan_sha256": source_plan["artifact_sha256"],
            "manifest_sha256": source_plan["inputs"]["route_manifest"]["semantic_sha256"],
            "taskset_sha256": taskset["artifact_sha256"],
            "repeat_panel_sha256": repeat_panel["artifact_sha256"],
            "family": task["family"],
            "slot_id": roster_row["slot_id"],
            "model_name": roster_row["model_name"],
            "canonical_model_slug": roster_row["canonical_model_slug"],
            "execution_backend": roster_row["execution_backend"],
            "endpoint_execution_sha256": roster_row["endpoint_execution_sha256"],
            "backend_contract_sha256": roster_row["backend_contract_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "optimal_selection": task["optimal_selection"],
            "original_task_id": task.get("original_task_id"),
        }
        if any(document.get(field) != value for field, value in exact_fields.items()):
            raise SelectionPoweredAnalysisError(
                f"response binding differs from frozen inputs: {path}"
            )
        if path.parent.name != document["slot_id"]:
            raise SelectionPoweredAnalysisError(f"response is stored under the wrong slot: {path}")
        status = document.get("status")
        if status not in {"completed", "failed"}:
            raise SelectionPoweredAnalysisError(f"unsupported response status: {status}")
        generation = document.get("generation")
        if status == "completed":
            if not isinstance(generation, dict) or not isinstance(
                generation.get("answer_markdown"), str
            ):
                raise SelectionPoweredAnalysisError("completed response lacks answer bytes")
            rescored = score_answer(task, generation["answer_markdown"])
        else:
            rescored = _zero_scoring(task)
        if document.get("scoring") != rescored:
            raise SelectionPoweredAnalysisError(f"response score does not reproduce: {path}")
        analysis_scoring = (
            (analysis_score_function or score_answer)(task, generation["answer_markdown"])
            if status == "completed"
            else _zero_scoring(task)
        )
        required_scoring_fields = {
            "observed_selection",
            "optimal_selection",
            "parseable",
            "score_bps",
            "score",
            "optimal",
        }
        if (
            not isinstance(analysis_scoring, Mapping)
            or set(analysis_scoring) != required_scoring_fields
            or analysis_scoring.get("optimal_selection") != task["optimal_selection"]
        ):
            raise SelectionPoweredAnalysisError("analysis scorer returned an invalid record")
        analysis_scoring_by_artifact[artifact] = analysis_scoring
        priority = source_priority_by_path[path]
        candidates = candidates_by_key.setdefault(key, [])
        if any(existing_priority == priority for existing_priority, _, _ in candidates):
            raise SelectionPoweredAnalysisError(
                f"response cell is duplicated within one source directory: {key}"
            )
        candidates.append((priority, path, document))

    ordered_models = tuple(str(row["model_id"]) for row in roster)
    ordered_tasks = tuple(str(task["task_id"]) for task in tasks)
    expected_keys = {(model, task) for model in ordered_models for task in ordered_tasks}
    if set(candidates_by_key) != expected_keys:
        raise SelectionPoweredAnalysisError(f"{panel} response key set is incomplete")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[str] = []
    spend_micros = 0
    for key, candidates in candidates_by_key.items():
        ordered = sorted(candidates, key=lambda value: (value[0], str(value[1])))
        valid = [
            candidate
            for candidate in ordered
            if candidate[2]["status"] == "completed"
            and analysis_scoring_by_artifact[str(candidate[2]["artifact_sha256"])]["parseable"]
            is True
        ]
        completed = [candidate for candidate in ordered if candidate[2]["status"] == "completed"]
        _, _, selected = (valid or completed or ordered)[0]
        by_key[key] = selected
        artifacts.append(str(selected["artifact_sha256"]))
        spend_micros += int((selected.get("generation") or {}).get("cost_micros") or 0)
    scores = np.zeros((len(ordered_models), len(ordered_tasks)), dtype=np.float64)
    completed = np.zeros_like(scores, dtype=bool)
    parseable = np.zeros_like(scores, dtype=bool)
    selections: list[tuple[str | None, ...]] = []
    for model_index, model_id in enumerate(ordered_models):
        model_selections: list[str | None] = []
        for task_index, task_id in enumerate(ordered_tasks):
            row = by_key[(model_id, task_id)]
            scoring = analysis_scoring_by_artifact[str(row["artifact_sha256"])]
            scores[model_index, task_index] = float(scoring["score"])
            completed[model_index, task_index] = row["status"] == "completed"
            parseable[model_index, task_index] = bool(scoring["parseable"])
            model_selections.append(scoring["observed_selection"])
        selections.append(tuple(model_selections))
    return PanelData(
        panel=panel,
        model_ids=ordered_models,
        model_names=tuple(str(row["model_name"]) for row in roster),
        slot_ids=tuple(str(row["slot_id"]) for row in roster),
        task_ids=ordered_tasks,
        families=tuple(str(task["family"]) for task in tasks),
        scores=scores,
        completed=completed,
        parseable=parseable,
        selections=tuple(selections),
        response_artifact_sha256s=tuple(sorted(artifacts)),
        spend_micros=spend_micros,
    )


def family_macro_mean(values: np.ndarray, families: Sequence[str]) -> np.ndarray:
    """Return equal-family means for one or more rows of task values."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != len(families):
        raise SelectionPoweredAnalysisError("value matrix and family vector differ")
    means = []
    family_values = np.asarray(families, dtype=object)
    for family in FAMILIES:
        indices = np.flatnonzero(family_values == family)
        if not len(indices):
            raise SelectionPoweredAnalysisError(f"missing task family: {family}")
        means.append(matrix[:, indices].mean(axis=1))
    return np.stack(means, axis=1).mean(axis=1)


def family_stratified_bootstrap(
    values: np.ndarray,
    families: Sequence[str],
    *,
    resamples: int,
    seed: int,
    batch_size: int = 500,
) -> np.ndarray:
    """Shared-task, equal-family bootstrap for every row in ``values``."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if resamples <= 0:
        raise SelectionPoweredAnalysisError("bootstrap resamples must be positive")
    family_values = np.asarray(families, dtype=object)
    indices = [np.flatnonzero(family_values == family) for family in FAMILIES]
    if any(not len(value) for value in indices):
        raise SelectionPoweredAnalysisError("bootstrap family is empty")
    rng = np.random.default_rng(seed)
    output = np.empty((resamples, matrix.shape[0]), dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        width = stop - start
        batch = np.zeros((width, matrix.shape[0]), dtype=np.float64)
        for family_indices in indices:
            draws = rng.integers(0, len(family_indices), size=(width, len(family_indices)))
            selected = matrix[:, family_indices[draws]]
            batch += selected.mean(axis=2).T / len(FAMILIES)
        output[start:stop] = batch
    return output


def paired_sign_flip_pvalues(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Monte Carlo paired sign-flip tests with a plus-one correction."""
    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if resamples <= 0 or matrix.shape[1] == 0:
        raise SelectionPoweredAnalysisError("invalid permutation test shape")
    observed = matrix.mean(axis=1)
    exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    threshold = np.abs(observed)
    for start in range(0, resamples, batch_size):
        width = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(width, matrix.shape[1]), dtype=np.int8)
        signed = signs.astype(np.float64) * 2.0 - 1.0
        null = signed @ matrix.T / matrix.shape[1]
        exceed += np.count_nonzero(np.abs(null) >= threshold[None, :] - 1e-12, axis=0)
    return observed, (exceed + 1.0) / (resamples + 1.0)


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Return monotone Holm-adjusted p-values in original order."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise SelectionPoweredAnalysisError("invalid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _rank_intervals(bootstrap_scores: np.ndarray, eligible: np.ndarray) -> dict[int, list[int]]:
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return {}
    subset = bootstrap_scores[:, indices]
    order = np.argsort(-subset, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, len(indices) + 1)[None, :]
    output: dict[int, list[int]] = {}
    for local, model_index in enumerate(indices):
        output[int(model_index)] = [
            int(np.quantile(ranks[:, local], 0.025, method="lower")),
            int(np.quantile(ranks[:, local], 0.975, method="higher")),
        ]
    return output


def _statistical_groups(
    point_scores: np.ndarray,
    eligible: np.ndarray,
    pairwise: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    significant: set[frozenset[int]] = {
        frozenset((int(row["left_index"]), int(row["right_index"])))
        for row in pairwise
        if row["holm_significant"]
    }
    ordered = sorted(np.flatnonzero(eligible), key=lambda index: (-point_scores[index], index))
    groups: dict[int, int] = {}
    current: list[int] = []
    group = 0
    for raw_index in ordered:
        index = int(raw_index)
        if not current or all(frozenset((index, other)) not in significant for other in current):
            if not current:
                group += 1
            current.append(index)
        else:
            group += 1
            current = [index]
        groups[index] = group
    return groups


def _cohen_dz(values: np.ndarray) -> float | None:
    sd = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    if sd <= 0:
        return 0.0 if mean == 0 else None
    return mean / sd


def _ingredient_set(task: Mapping[str, Any], selection: str | None) -> frozenset[str]:
    if selection is None:
        return frozenset()
    return frozenset(str(task["choices"][label]) for label in selection)


def analyze_repeatability(
    *,
    primary: PanelData,
    repeat: PanelData,
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare selected ingredient sets, never answer-label strings."""
    if primary.model_ids != repeat.model_ids:
        raise SelectionPoweredAnalysisError("primary and repeat rosters differ")
    primary_index = {task_id: index for index, task_id in enumerate(primary.task_ids)}
    primary_tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
    repeat_tasks = {str(task["task_id"]): task for task in repeat_panel["tasks"]}
    jaccard = np.zeros_like(repeat.scores)
    exact = np.zeros_like(repeat.scores)
    score_delta = np.zeros_like(repeat.scores)
    for repeat_task_index, repeat_task_id in enumerate(repeat.task_ids):
        repeated_task = repeat_tasks[repeat_task_id]
        original_task_id = str(repeated_task["original_task_id"])
        original_task_index = primary_index[original_task_id]
        original_task = primary_tasks[original_task_id]
        for model_index in range(len(primary.model_ids)):
            left = _ingredient_set(
                original_task, primary.selections[model_index][original_task_index]
            )
            right = _ingredient_set(
                repeated_task, repeat.selections[model_index][repeat_task_index]
            )
            union = left | right
            jaccard[model_index, repeat_task_index] = (
                len(left & right) / len(union) if union else 0.0
            )
            exact[model_index, repeat_task_index] = float(bool(left) and left == right)
            score_delta[model_index, repeat_task_index] = abs(
                primary.scores[model_index, original_task_index]
                - repeat.scores[model_index, repeat_task_index]
            )
    jaccard_bootstrap = family_stratified_bootstrap(
        jaccard,
        repeat.families,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    output: list[dict[str, Any]] = []
    for index, model_id in enumerate(primary.model_ids):
        output.append(
            {
                "model_id": model_id,
                "tasks": len(repeat.task_ids),
                "completed": int(repeat.completed[index].sum()),
                "parseable": int(repeat.parseable[index].sum()),
                "mean_ingredient_set_jaccard": float(
                    family_macro_mean(jaccard[index], repeat.families)[0]
                ),
                "jaccard_pointwise_95_ci": _percentile_interval(jaccard_bootstrap[:, index]),
                "exact_ingredient_set_match_rate": float(
                    family_macro_mean(exact[index], repeat.families)[0]
                ),
                "mean_absolute_score_difference": float(
                    family_macro_mean(score_delta[index], repeat.families)[0]
                ),
            }
        )
    return output


def analyze_panels(
    *,
    primary: PanelData,
    taskset: Mapping[str, Any],
    plan: Mapping[str, Any],
    repeat: PanelData | None = None,
    repeat_panel: Mapping[str, Any] | None = None,
    bootstrap_resamples: int | None = None,
    permutation_resamples: int | None = None,
) -> dict[str, Any]:
    """Compute every frozen primary and repeatability estimand."""
    inference = plan["inference"]
    bootstrap_count = int(bootstrap_resamples or inference["bootstrap_resamples"])
    permutation_count = int(permutation_resamples or inference["permutation_resamples"])
    seed = int(inference["seed"])
    task_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    chance = np.asarray(
        [float(task_by_id[task_id]["chance_score_bps"]) / 100 for task_id in primary.task_ids],
        dtype=np.float64,
    )
    point = family_macro_mean(primary.scores, primary.families)
    family_values = np.asarray(primary.families, dtype=object)
    family_scores = {
        family: primary.scores[:, family_values == family].mean(axis=1) for family in FAMILIES
    }
    all_bootstrap = family_stratified_bootstrap(
        np.vstack((primary.scores, chance[None, :])),
        primary.families,
        resamples=bootstrap_count,
        seed=seed,
    )
    score_bootstrap = all_bootstrap[:, :-1]
    chance_bootstrap = all_bootstrap[:, -1]
    standard_errors = np.std(score_bootstrap, axis=0, ddof=1)
    safe_se = np.where(standard_errors > 0, standard_errors, 1.0)
    max_t = np.max(np.abs((score_bootstrap - point[None, :]) / safe_se[None, :]), axis=1)
    max_t_critical = float(np.quantile(max_t, 0.95))

    left_indices: list[int] = []
    right_indices: list[int] = []
    differences: list[np.ndarray] = []
    for left in range(len(primary.model_ids)):
        for right in range(left + 1, len(primary.model_ids)):
            left_indices.append(left)
            right_indices.append(right)
            differences.append(primary.scores[left] - primary.scores[right])
    pair_matrix = np.asarray(differences, dtype=np.float64)
    observed, raw_pvalues = paired_sign_flip_pvalues(
        pair_matrix,
        resamples=permutation_count,
        seed=seed + 1,
    )
    adjusted_pvalues = holm_adjust(raw_pvalues)
    pairwise: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(zip(left_indices, right_indices, strict=True)):
        bootstrap_difference = score_bootstrap[:, left] - score_bootstrap[:, right]
        pairwise.append(
            {
                "left_index": left,
                "right_index": right,
                "left_model_id": primary.model_ids[left],
                "right_model_id": primary.model_ids[right],
                "mean_difference": float(observed[pair_index]),
                "bootstrap_95_ci": _percentile_interval(bootstrap_difference),
                "cohen_dz": _cohen_dz(pair_matrix[pair_index]),
                "sign_flip_p": float(raw_pvalues[pair_index]),
                "holm_p": float(adjusted_pvalues[pair_index]),
                "holm_significant": bool(adjusted_pvalues[pair_index] < 0.05),
                "direction": (
                    "left_higher"
                    if observed[pair_index] > 0
                    else "right_higher"
                    if observed[pair_index] < 0
                    else "tie"
                ),
            }
        )

    chance_differences = primary.scores - chance[None, :]
    chance_observed, chance_raw = paired_sign_flip_pvalues(
        chance_differences,
        resamples=permutation_count,
        seed=seed + 2,
    )
    chance_adjusted = holm_adjust(chance_raw)
    completed_count = primary.completed.sum(axis=1)
    eligible = completed_count >= int(plan["eligibility"]["minimum_completed_tasks"])
    rank_intervals = _rank_intervals(score_bootstrap, eligible)
    groups = _statistical_groups(point, eligible, pairwise)
    point_order = sorted(np.flatnonzero(eligible), key=lambda index: (-point[index], index))
    point_ranks = {int(index): rank + 1 for rank, index in enumerate(point_order)}

    repeat_results: list[dict[str, Any]] | None = None
    repeat_by_model: dict[str, Mapping[str, Any]] = {}
    if repeat is not None:
        if repeat_panel is None:
            raise SelectionPoweredAnalysisError("repeat data requires its frozen task panel")
        repeat_results = analyze_repeatability(
            primary=primary,
            repeat=repeat,
            taskset=taskset,
            repeat_panel=repeat_panel,
            bootstrap_resamples=bootstrap_count,
            seed=seed + 3,
        )
        repeat_by_model = {str(row["model_id"]): row for row in repeat_results}

    models: list[dict[str, Any]] = []
    for index, model_id in enumerate(primary.model_ids):
        simultaneous_half_width = max_t_critical * standard_errors[index]
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_name": primary.model_names[index],
            "slot_id": primary.slot_ids[index],
            "eligible": bool(eligible[index]),
            "availability": {
                "scheduled": len(primary.task_ids),
                "completed": int(completed_count[index]),
                "parseable": int(primary.parseable[index].sum()),
                "completion_rate": float(primary.completed[index].mean()),
                "parseable_rate": float(primary.parseable[index].mean()),
            },
            "flavourbench_score": float(point[index]),
            "family_scores": {family: float(family_scores[family][index]) for family in FAMILIES},
            "score_standard_error": float(standard_errors[index]),
            "score_pointwise_95_ci": _percentile_interval(score_bootstrap[:, index]),
            "score_simultaneous_95_ci": [
                float(point[index] - simultaneous_half_width),
                float(point[index] + simultaneous_half_width),
            ],
            "point_estimate_rank": point_ranks.get(index),
            "bootstrap_rank_95_interval": rank_intervals.get(index),
            "statistical_rank_group": groups.get(index),
            "chance_comparison": {
                "exact_chance_score": float(family_macro_mean(chance, primary.families)[0]),
                "mean_difference": float(chance_observed[index]),
                "bootstrap_95_ci": _percentile_interval(
                    score_bootstrap[:, index] - chance_bootstrap
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
    if point_order and repeat_results is not None:
        leader = int(point_order[0])
        comparisons = [
            row
            for row in pairwise
            if leader in {row["left_index"], row["right_index"]}
            and eligible[int(row["left_index"])]
            and eligible[int(row["right_index"])]
        ]
        leader_beats_all = len(comparisons) == int(eligible.sum()) - 1 and all(
            row["holm_significant"]
            and (
                (row["left_index"] == leader and row["mean_difference"] > 0)
                or (row["right_index"] == leader and row["mean_difference"] < 0)
            )
            for row in comparisons
        )
        repeat_ok = float(
            repeat_by_model[primary.model_ids[leader]]["mean_ingredient_set_jaccard"]
        ) >= float(plan["repeatability"]["acceptance_floor"])
        if leader_beats_all and repeat_ok:
            definitive_top_model_id = primary.model_ids[leader]

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "final_complete" if repeat is not None else "primary_complete_repeat_pending",
        "plan_sha256": plan["artifact_sha256"],
        "estimand": plan["outcomes"]["primary_definition"],
        "intention_to_evaluate": True,
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
        rows.append(
            {
                "rank": model["point_estimate_rank"],
                "rank_group": model["statistical_rank_group"],
                "model": model["model_name"],
                "model_id": model["model_id"],
                "flavourbench_score": f"{model['flavourbench_score']:.6f}",
                "simultaneous_ci_low": f"{model['score_simultaneous_95_ci'][0]:.6f}",
                "simultaneous_ci_high": f"{model['score_simultaneous_95_ci'][1]:.6f}",
                "completed": model["availability"]["completed"],
                "parseable": model["availability"]["parseable"],
                "repeat_jaccard": (
                    f"{repeat['mean_ingredient_set_jaccard']:.6f}" if repeat else ""
                ),
                "eligible": str(model["eligible"]).lower(),
            }
        )
    rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or math.inf, row["model_id"]))
    return _csv_bytes(
        rows,
        (
            "rank",
            "rank_group",
            "model",
            "model_id",
            "flavourbench_score",
            "simultaneous_ci_low",
            "simultaneous_ci_high",
            "completed",
            "parseable",
            "repeat_jaccard",
            "eligible",
        ),
    )


def _pairwise_csv(analysis: Mapping[str, Any]) -> bytes:
    rows = []
    for row in analysis["pairwise_comparisons"]:
        rows.append(
            {
                "left_model_id": row["left_model_id"],
                "right_model_id": row["right_model_id"],
                "mean_difference": f"{row['mean_difference']:.6f}",
                "ci_low": f"{row['bootstrap_95_ci'][0]:.6f}",
                "ci_high": f"{row['bootstrap_95_ci'][1]:.6f}",
                "cohen_dz": "" if row["cohen_dz"] is None else f"{row['cohen_dz']:.8f}",
                "sign_flip_p": f"{row['sign_flip_p']:.10g}",
                "holm_p": f"{row['holm_p']:.10g}",
                "holm_significant": str(row["holm_significant"]).lower(),
                "direction": row["direction"],
            }
        )
    return _csv_bytes(
        rows,
        (
            "left_model_id",
            "right_model_id",
            "mean_difference",
            "ci_low",
            "ci_high",
            "cohen_dz",
            "sign_flip_p",
            "holm_p",
            "holm_significant",
            "direction",
        ),
    )


def _write_content_addressed(
    directory: Path, prefix: str, data: bytes, *, address: str | None = None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = address or _sha256_bytes(data)
    destination = directory / f"{prefix}-{digest}{'.csv' if prefix.endswith('table') else '.json'}"
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != data:
            raise SelectionPoweredAnalysisError("content-addressed output conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--predecessor-plan", type=Path)
    parser.add_argument("--base-plan", type=Path)
    parser.add_argument("--deepseek-plan", type=Path)
    parser.add_argument("--recovery-run-directory", type=Path)
    parser.add_argument("--frontier-run-directory", type=Path)
    parser.add_argument("--frontier-plan", type=Path)
    parser.add_argument("--deepseek-run-directory", type=Path)
    parser.add_argument("--cohere-run-directory", type=Path)
    parser.add_argument("--cohere-plan", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args(argv)
    taskset = _load(args.taskset)
    repeat_document = _load(args.repeat_panel)
    plan = _load(args.plan)
    plan_schema = plan.get("schema_version")
    plan_verifiers = {
        PLAN_SCHEMA_VERSION_V17: verify_plan_v17,
        PLAN_SCHEMA_VERSION_V18: verify_plan_v18,
        PLAN_SCHEMA_VERSION_V19: verify_plan_v19,
        PLAN_SCHEMA_VERSION_V20: verify_plan_v20,
        PLAN_SCHEMA_VERSION_V21: verify_plan_v21,
        PLAN_SCHEMA_VERSION_V22: verify_plan_v22,
        PLAN_SCHEMA_VERSION_V23: verify_plan_v23,
        PLAN_SCHEMA_VERSION_V24: verify_plan_v24,
        PLAN_SCHEMA_VERSION_V25: verify_plan_v25,
        PLAN_SCHEMA_VERSION_V26: verify_plan_v26,
        PLAN_SCHEMA_VERSION_V27: verify_plan_v27,
        PLAN_SCHEMA_VERSION_V28: verify_plan_v28,
        PLAN_SCHEMA_VERSION_V29: verify_plan_v29,
        PLAN_SCHEMA_VERSION_V30: verify_plan_v30,
        PLAN_SCHEMA_VERSION_V31: verify_plan_v31,
        PLAN_SCHEMA_VERSION_V32: verify_plan_v32,
        PLAN_SCHEMA_VERSION_V33: verify_plan_v33,
        PLAN_SCHEMA_VERSION_V34: verify_plan_v34,
        PLAN_SCHEMA_VERSION_V35: verify_plan_v35,
        PLAN_SCHEMA_VERSION_V36: verify_plan_v36,
        PLAN_SCHEMA_VERSION_V37: verify_plan_v37,
        PLAN_SCHEMA_VERSION_V38: verify_plan_v38,
        PLAN_SCHEMA_VERSION_V39: verify_plan_v39,
        PLAN_SCHEMA_VERSION_V40: verify_plan_v40,
        PLAN_SCHEMA_VERSION_V41: verify_plan_v41,
        PLAN_SCHEMA_VERSION_V42: verify_plan_v42,
        PLAN_SCHEMA_VERSION_V43: verify_plan_v43,
    }
    plan_valid = plan_schema in plan_verifiers and plan_verifiers[plan_schema](plan)
    if (
        not verify_taskset(taskset)
        or not verify_repeat_panel(repeat_document, taskset=taskset)
        or not plan_valid
    ):
        raise SelectionPoweredAnalysisError("analysis inputs failed semantic verification")
    exact_inputs = {
        "taskset": (taskset["artifact_sha256"], _sha256_file(args.taskset)),
        "repeat_panel": (repeat_document["artifact_sha256"], _sha256_file(args.repeat_panel)),
    }
    for label, (semantic, physical) in exact_inputs.items():
        recorded = plan["inputs"][label]
        if recorded["semantic_sha256"] != semantic or recorded["physical_sha256"] != physical:
            raise SelectionPoweredAnalysisError(f"plan {label} pin differs from exact input")

    composite_requested = (
        args.predecessor_plan is not None
        or args.base_plan is not None
        or args.deepseek_plan is not None
        or args.recovery_run_directory is not None
        or args.frontier_run_directory is not None
        or args.frontier_plan is not None
        or args.deepseek_run_directory is not None
        or args.cohere_run_directory is not None
        or args.cohere_plan is not None
    )
    model_sources: dict[str, tuple[Path, Mapping[str, Any]]] | None = None
    response_source_lineage: dict[str, Any] | None = None
    if composite_requested:
        if plan_schema not in {
            PLAN_SCHEMA_VERSION_V32,
            PLAN_SCHEMA_VERSION_V33,
            PLAN_SCHEMA_VERSION_V35,
            PLAN_SCHEMA_VERSION_V36,
            PLAN_SCHEMA_VERSION_V37,
            PLAN_SCHEMA_VERSION_V38,
            PLAN_SCHEMA_VERSION_V39,
            PLAN_SCHEMA_VERSION_V40,
            PLAN_SCHEMA_VERSION_V41,
            PLAN_SCHEMA_VERSION_V42,
            PLAN_SCHEMA_VERSION_V43,
        }:
            raise SelectionPoweredAnalysisError("composite recovery requires a recovery plan")
        if args.predecessor_plan is None or args.recovery_run_directory is None:
            raise SelectionPoweredAnalysisError(
                "composite analysis requires both predecessor plan and recovery run directory"
            )
        if plan_schema in {PLAN_SCHEMA_VERSION_V42, PLAN_SCHEMA_VERSION_V43}:
            if (
                args.base_plan is None
                or args.cohere_plan is None
                or args.cohere_run_directory is None
                or args.frontier_plan is None
                or args.frontier_run_directory is None
                or args.deepseek_plan is None
                or args.deepseek_run_directory is None
            ):
                raise SelectionPoweredAnalysisError(
                    "v42/v43 composite analysis requires base, Cohere, v38 frontier, "
                    "v39 DeepSeek, and v41 predecessor sources"
                )
            predecessor_plan = _load(args.predecessor_plan)
            if plan_schema == PLAN_SCHEMA_VERSION_V43:
                predecessor_pin = plan["inputs"]["plan_v42_predecessor"]
                predecessor_valid = verify_plan_v42(predecessor_plan)
                predecessor_label = "v43 predecessor v42"
            else:
                predecessor_pin = plan["inputs"]["plan_v41_predecessor"]
                predecessor_valid = verify_plan_v41(predecessor_plan)
                predecessor_label = "v42 predecessor v41"
            if (
                not predecessor_valid
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError(f"{predecessor_label} binding failed")
            deepseek_plan = _load(args.deepseek_plan)
            deepseek_pin = plan["inputs"]["plan_v39_predecessor"]
            if (
                not verify_plan_v39(deepseek_plan)
                or deepseek_pin["semantic_sha256"] != deepseek_plan["artifact_sha256"]
                or deepseek_pin["physical_sha256"] != _sha256_file(args.deepseek_plan)
            ):
                raise SelectionPoweredAnalysisError("v42/v43 DeepSeek v39 binding failed")
            frontier_plan = _load(args.frontier_plan)
            frontier_pin = plan["inputs"]["plan_v38_predecessor"]
            if (
                not verify_plan_v38(frontier_plan)
                or frontier_pin["semantic_sha256"] != frontier_plan["artifact_sha256"]
                or frontier_pin["physical_sha256"] != _sha256_file(args.frontier_plan)
            ):
                raise SelectionPoweredAnalysisError("v42/v43 frontier v38 binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["retained_base_response_source_plan"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v42/v43 base v31 binding failed")
            cohere_plan = _load(args.cohere_plan)
            cohere_pin = plan["inputs"]["retained_cohere_response_source_plan"]
            if (
                not verify_plan_v35(cohere_plan)
                or cohere_pin["semantic_sha256"] != cohere_plan["artifact_sha256"]
                or cohere_pin["physical_sha256"] != _sha256_file(args.cohere_plan)
            ):
                raise SelectionPoweredAnalysisError("v42/v43 Cohere v35 binding failed")
            successor = plan["execution"]["frontier_refresh_successor"]
            base_ids = {str(value) for value in successor["retained_base_model_ids"]}
            cohere_ids = {str(value) for value in successor["retained_cohere_model_ids"]}
            frontier_ids = {str(value) for value in successor["retained_v38_new_model_ids"]}
            deepseek_ids = {str(value) for value in successor["retained_v39_new_model_ids"]}
            rerun_ids = {str(value) for value in successor["rerun_model_ids"]}
            roster_ids = {str(row["model_id"]) for row in plan["roster"]["models"]}
            groups = (base_ids, cohere_ids, frontier_ids, deepseek_ids, rerun_ids)
            if (
                tuple(map(len, groups)) != (16, 2, 6, 1, 1)
                or deepseek_ids != {V40_DEEPSEEK_ID}
                or rerun_ids != {V40_FABLE_ID}
                or any(
                    left & right
                    for index, left in enumerate(groups)
                    for right in groups[index + 1 :]
                )
                or set().union(*groups) != roster_ids
            ):
                raise SelectionPoweredAnalysisError("v42/v43 response-source partition is invalid")
            model_sources = {model_id: (args.run_directory, base_plan) for model_id in base_ids}
            model_sources.update(
                {model_id: (args.cohere_run_directory, cohere_plan) for model_id in cohere_ids}
            )
            model_sources.update(
                {
                    model_id: (args.frontier_run_directory, frontier_plan)
                    for model_id in frontier_ids
                }
            )
            model_sources.update(
                {
                    model_id: (args.deepseek_run_directory, deepseek_plan)
                    for model_id in deepseek_ids
                }
            )
            model_sources.update(
                {model_id: (args.recovery_run_directory, plan) for model_id in rerun_ids}
            )
            response_source_lineage = {
                "schema_version": (
                    "flavourbench-selection-composite-response-sources-v8"
                    if plan_schema == PLAN_SCHEMA_VERSION_V43
                    else "flavourbench-selection-composite-response-sources-v7"
                ),
                "base_model_ids": sorted(base_ids),
                "base_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "cohere_model_ids": sorted(cohere_ids),
                "cohere_plan": {
                    "semantic_sha256": cohere_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.cohere_plan),
                },
                "frontier_model_ids": sorted(frontier_ids),
                "frontier_plan": {
                    "semantic_sha256": frontier_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.frontier_plan),
                },
                "deepseek_model_ids": sorted(deepseek_ids),
                "deepseek_plan": {
                    "semantic_sha256": deepseek_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.deepseek_plan),
                },
                "successor_model_ids": sorted(rerun_ids),
                "successor_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_model_responses_used": False,
                "v38_deepseek_responses_used_as_score_data": False,
                "v38_fable_responses_used_as_score_data": False,
                "v39_deepseek_responses_used_as_score_data": True,
                "v40_fable_responses_used_as_score_data": False,
                "v41_fable_responses_used_as_score_data": False,
                "v42_fable_responses_used_as_score_data": (plan_schema == PLAN_SCHEMA_VERSION_V42),
                "v43_fable_responses_used_as_score_data": (plan_schema == PLAN_SCHEMA_VERSION_V43),
                "route_probe_responses_used_as_score_data": False,
            }
        elif plan_schema == PLAN_SCHEMA_VERSION_V40:
            if (
                args.base_plan is None
                or args.cohere_plan is None
                or args.cohere_run_directory is None
                or args.frontier_plan is None
                or args.frontier_run_directory is None
                or args.deepseek_run_directory is None
            ):
                raise SelectionPoweredAnalysisError(
                    "v40 composite analysis requires base, Cohere, v38 frontier, "
                    "and v39 DeepSeek sources"
                )
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = plan["inputs"]["plan_v39_predecessor"]
            if (
                not verify_plan_v39(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("v40 predecessor v39 binding failed")
            frontier_plan = _load(args.frontier_plan)
            frontier_pin = predecessor_plan["inputs"]["plan_v38_predecessor"]
            if (
                not verify_plan_v38(frontier_plan)
                or frontier_pin["semantic_sha256"] != frontier_plan["artifact_sha256"]
                or frontier_pin["physical_sha256"] != _sha256_file(args.frontier_plan)
            ):
                raise SelectionPoweredAnalysisError("v40 frontier v38 binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["retained_base_response_source_plan"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v40 base v31 binding failed")
            cohere_plan = _load(args.cohere_plan)
            cohere_pin = plan["inputs"]["retained_cohere_response_source_plan"]
            if (
                not verify_plan_v35(cohere_plan)
                or cohere_pin["semantic_sha256"] != cohere_plan["artifact_sha256"]
                or cohere_pin["physical_sha256"] != _sha256_file(args.cohere_plan)
            ):
                raise SelectionPoweredAnalysisError("v40 Cohere v35 binding failed")
            successor = plan["execution"]["frontier_refresh_successor"]
            base_ids = {str(value) for value in successor["retained_base_model_ids"]}
            cohere_ids = {str(value) for value in successor["retained_cohere_model_ids"]}
            frontier_ids = {str(value) for value in successor["retained_v38_new_model_ids"]}
            deepseek_ids = {str(value) for value in successor["retained_v39_new_model_ids"]}
            rerun_ids = {str(value) for value in successor["rerun_model_ids"]}
            roster_ids = {str(row["model_id"]) for row in plan["roster"]["models"]}
            groups = (base_ids, cohere_ids, frontier_ids, deepseek_ids, rerun_ids)
            if (
                tuple(map(len, groups)) != (16, 2, 6, 1, 1)
                or deepseek_ids != {V40_DEEPSEEK_ID}
                or rerun_ids != {V40_FABLE_ID}
                or any(
                    left & right
                    for index, left in enumerate(groups)
                    for right in groups[index + 1 :]
                )
                or set().union(*groups) != roster_ids
            ):
                raise SelectionPoweredAnalysisError("v40 response-source partition is invalid")
            model_sources = {model_id: (args.run_directory, base_plan) for model_id in base_ids}
            model_sources.update(
                {model_id: (args.cohere_run_directory, cohere_plan) for model_id in cohere_ids}
            )
            model_sources.update(
                {
                    model_id: (args.frontier_run_directory, frontier_plan)
                    for model_id in frontier_ids
                }
            )
            model_sources.update(
                {
                    model_id: (args.deepseek_run_directory, predecessor_plan)
                    for model_id in deepseek_ids
                }
            )
            model_sources.update(
                {model_id: (args.recovery_run_directory, plan) for model_id in rerun_ids}
            )
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v6",
                "base_model_ids": sorted(base_ids),
                "base_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "cohere_model_ids": sorted(cohere_ids),
                "cohere_plan": {
                    "semantic_sha256": cohere_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.cohere_plan),
                },
                "frontier_model_ids": sorted(frontier_ids),
                "frontier_plan": {
                    "semantic_sha256": frontier_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.frontier_plan),
                },
                "deepseek_model_ids": sorted(deepseek_ids),
                "deepseek_plan": {
                    "semantic_sha256": predecessor_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.predecessor_plan),
                },
                "successor_model_ids": sorted(rerun_ids),
                "successor_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_model_responses_used": False,
                "v38_deepseek_responses_used_as_score_data": False,
                "v38_fable_responses_used_as_score_data": False,
                "v39_deepseek_responses_used_as_score_data": True,
            }
        elif plan_schema == PLAN_SCHEMA_VERSION_V39:
            if (
                args.base_plan is None
                or args.cohere_plan is None
                or args.cohere_run_directory is None
                or args.frontier_run_directory is None
            ):
                raise SelectionPoweredAnalysisError(
                    "v39 composite analysis requires base, Cohere, and v38 frontier sources"
                )
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = plan["inputs"]["plan_v38_predecessor"]
            if (
                not verify_plan_v38(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("v39 predecessor plan binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["retained_base_response_source_plan"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v39 base v31 binding failed")
            cohere_plan = _load(args.cohere_plan)
            cohere_pin = plan["inputs"]["retained_cohere_response_source_plan"]
            if (
                not verify_plan_v35(cohere_plan)
                or cohere_pin["semantic_sha256"] != cohere_plan["artifact_sha256"]
                or cohere_pin["physical_sha256"] != _sha256_file(args.cohere_plan)
            ):
                raise SelectionPoweredAnalysisError("v39 Cohere v35 binding failed")
            successor = plan["execution"]["frontier_refresh_successor"]
            base_ids = {str(value) for value in successor["retained_base_model_ids"]}
            cohere_ids = {str(value) for value in successor["retained_cohere_model_ids"]}
            frontier_ids = {str(value) for value in successor["retained_v38_new_model_ids"]}
            rerun_ids = {str(value) for value in successor["rerun_model_ids"]}
            roster_ids = {str(row["model_id"]) for row in plan["roster"]["models"]}
            groups = (base_ids, cohere_ids, frontier_ids, rerun_ids)
            if (
                tuple(map(len, groups)) != (16, 2, 7, 1)
                or any(
                    left & right
                    for index, left in enumerate(groups)
                    for right in groups[index + 1 :]
                )
                or set().union(*groups) != roster_ids
            ):
                raise SelectionPoweredAnalysisError("v39 response-source partition is invalid")
            model_sources = {model_id: (args.run_directory, base_plan) for model_id in base_ids}
            model_sources.update(
                {model_id: (args.cohere_run_directory, cohere_plan) for model_id in cohere_ids}
            )
            model_sources.update(
                {
                    model_id: (args.frontier_run_directory, predecessor_plan)
                    for model_id in frontier_ids
                }
            )
            model_sources.update(
                {model_id: (args.recovery_run_directory, plan) for model_id in rerun_ids}
            )
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v5",
                "base_model_ids": sorted(base_ids),
                "base_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "cohere_model_ids": sorted(cohere_ids),
                "cohere_plan": {
                    "semantic_sha256": cohere_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.cohere_plan),
                },
                "frontier_model_ids": sorted(frontier_ids),
                "frontier_plan": {
                    "semantic_sha256": predecessor_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.predecessor_plan),
                },
                "successor_model_ids": sorted(rerun_ids),
                "successor_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_model_responses_used": False,
                "v38_deepseek_responses_used_as_score_data": False,
            }
        elif plan_schema in {PLAN_SCHEMA_VERSION_V37, PLAN_SCHEMA_VERSION_V38}:
            if (
                args.base_plan is None
                or args.cohere_plan is None
                or args.cohere_run_directory is None
            ):
                raise SelectionPoweredAnalysisError(
                    "v37 composite analysis requires base and Cohere plans/runs"
                )
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = (
                plan["inputs"]["plan_v36_predecessor"]
                if plan_schema == PLAN_SCHEMA_VERSION_V37
                else plan["inputs"]["plan_v37_predecessor"]
            )
            predecessor_verifier = (
                verify_plan_v36 if plan_schema == PLAN_SCHEMA_VERSION_V37 else verify_plan_v37
            )
            if (
                not predecessor_verifier(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("frontier predecessor plan binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["retained_base_response_source_plan"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v37 base v31 binding failed")
            cohere_plan = _load(args.cohere_plan)
            cohere_pin = plan["inputs"]["retained_cohere_response_source_plan"]
            if (
                not verify_plan_v35(cohere_plan)
                or cohere_pin["semantic_sha256"] != cohere_plan["artifact_sha256"]
                or cohere_pin["physical_sha256"] != _sha256_file(args.cohere_plan)
            ):
                raise SelectionPoweredAnalysisError("v37 Cohere v35 binding failed")
            successor = plan["execution"]["frontier_refresh_successor"]
            base_ids = set(str(value) for value in successor["retained_base_model_ids"])
            cohere_ids = set(str(value) for value in successor["retained_cohere_model_ids"])
            new_ids = set(str(value) for value in successor["new_model_ids"])
            roster_ids = {str(row["model_id"]) for row in plan["roster"]["models"]}
            if (
                len(base_ids) != 16
                or len(cohere_ids) != 2
                or len(new_ids) != 8
                or base_ids & cohere_ids
                or base_ids & new_ids
                or cohere_ids & new_ids
                or base_ids | cohere_ids | new_ids != roster_ids
            ):
                raise SelectionPoweredAnalysisError("v37 response-source partition is invalid")
            model_sources = {model_id: (args.run_directory, base_plan) for model_id in base_ids}
            model_sources.update(
                {model_id: (args.cohere_run_directory, cohere_plan) for model_id in cohere_ids}
            )
            model_sources.update(
                {model_id: (args.recovery_run_directory, plan) for model_id in new_ids}
            )
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v4",
                "base_model_ids": sorted(base_ids),
                "base_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "cohere_model_ids": sorted(cohere_ids),
                "cohere_plan": {
                    "semantic_sha256": cohere_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.cohere_plan),
                },
                "successor_model_ids": sorted(new_ids),
                "successor_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_model_responses_used": False,
                "v36_pilot_responses_used_as_score_data": False,
            }
        elif plan_schema == PLAN_SCHEMA_VERSION_V36:
            if args.base_plan is None:
                raise SelectionPoweredAnalysisError(
                    "v36 composite analysis requires the exact retained v31 source plan"
                )
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = plan["inputs"]["plan_v35_predecessor"]
            if (
                not verify_plan_v35(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("v36 predecessor v35 plan binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["retained_response_source_plan"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v36 retained v31 plan binding failed")
            successor = plan["execution"]["frontier_refresh_successor"]
            retained_ids = set(str(value) for value in successor["retained_model_ids"])
            new_ids = set(str(value) for value in successor["new_model_ids"])
            roster_model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
            if (
                len(retained_ids) != 16
                or len(new_ids) != 10
                or retained_ids & new_ids
                or retained_ids | new_ids != set(roster_model_ids)
            ):
                raise SelectionPoweredAnalysisError("v36 response-source partition is invalid")
            model_sources = {model_id: (args.run_directory, base_plan) for model_id in retained_ids}
            model_sources.update(
                {model_id: (args.recovery_run_directory, plan) for model_id in new_ids}
            )
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v3",
                "retained_model_ids": sorted(retained_ids),
                "retained_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "successor_model_ids": sorted(new_ids),
                "successor_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "successor_run_directory_role": "ten complete new or replacement blocks",
                "cross_provider_response_pooling_within_model": False,
                "superseded_model_responses_used": False,
            }
        elif plan_schema == PLAN_SCHEMA_VERSION_V35:
            if (
                args.base_plan is None
                or args.deepseek_plan is None
                or args.cohere_run_directory is None
            ):
                raise SelectionPoweredAnalysisError(
                    "v35 composite analysis requires v31 base, v33 DeepSeek, and Cohere sources"
                )
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = plan["inputs"]["plan_v34_predecessor"]
            if (
                not verify_plan_v34(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("v35 predecessor v34 plan binding failed")
            deepseek_plan = _load(args.deepseek_plan)
            deepseek_pin = plan["inputs"]["plan_v33_predecessor"]
            if (
                not verify_plan_v33(deepseek_plan)
                or deepseek_pin["semantic_sha256"] != deepseek_plan["artifact_sha256"]
                or deepseek_pin["physical_sha256"] != _sha256_file(args.deepseek_plan)
            ):
                raise SelectionPoweredAnalysisError("v35 DeepSeek v33 plan binding failed")
            base_plan = _load(args.base_plan)
            base_pin = plan["inputs"]["plan_v31_predecessor"]
            if (
                not verify_plan_v31(base_plan)
                or base_pin["semantic_sha256"] != base_plan["artifact_sha256"]
                or base_pin["physical_sha256"] != _sha256_file(args.base_plan)
            ):
                raise SelectionPoweredAnalysisError("v35 base v31 plan binding failed")
            roster_model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
            deepseek_model_id = str(plan["execution"]["deepseek_route_recovery"]["model_id"])
            cohere_model_ids = tuple(
                str(model_id)
                for model_id in plan["execution"]["cohere_route_successor"]["successor_model_ids"]
            )
            replacement_ids = {deepseek_model_id, *cohere_model_ids}
            if (
                len(cohere_model_ids) != 2
                or len(replacement_ids) != 3
                or not replacement_ids <= set(roster_model_ids)
            ):
                raise SelectionPoweredAnalysisError("v35 replacement model set is invalid")
            model_sources = {
                model_id: (args.run_directory, base_plan) for model_id in roster_model_ids
            }
            model_sources[deepseek_model_id] = (args.recovery_run_directory, deepseek_plan)
            for model_id in cohere_model_ids:
                model_sources[model_id] = (args.cohere_run_directory, plan)
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v2",
                "base_models": sorted(set(roster_model_ids) - replacement_ids),
                "base_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.base_plan),
                },
                "deepseek_model_id": deepseek_model_id,
                "deepseek_plan": {
                    "semantic_sha256": deepseek_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.deepseek_plan),
                },
                "deepseek_calibration_plan": {
                    **deepseek_plan["inputs"]["plan_v32_predecessor"],
                    "responses_used_as_score_data": False,
                },
                "cohere_model_ids": sorted(cohere_model_ids),
                "cohere_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cohere_calibration_plan": {
                    "semantic_sha256": predecessor_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.predecessor_plan),
                    "responses_used_as_score_data": False,
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_deepseek_responses_used": False,
                "superseded_direct_cohere_responses_used": False,
                "cohere_transport_responses_used_as_score_data": False,
            }
        else:
            if args.deepseek_plan is not None or args.cohere_run_directory is not None:
                raise SelectionPoweredAnalysisError(
                    "DeepSeek-plan and Cohere-run arguments are reserved for v35"
                )
            if plan_schema == PLAN_SCHEMA_VERSION_V33 and args.base_plan is None:
                raise SelectionPoweredAnalysisError(
                    "v33 composite recovery requires the v31 base plan"
                )
            if plan_schema == PLAN_SCHEMA_VERSION_V32 and args.base_plan is not None:
                raise SelectionPoweredAnalysisError("v32 predecessor is already the base plan")
            predecessor_plan = _load(args.predecessor_plan)
            predecessor_pin = (
                plan["inputs"]["plan_v31_predecessor"]
                if plan_schema == PLAN_SCHEMA_VERSION_V32
                else plan["inputs"]["plan_v32_predecessor"]
            )
            predecessor_verifier = (
                verify_plan_v31 if plan_schema == PLAN_SCHEMA_VERSION_V32 else verify_plan_v32
            )
            if (
                not predecessor_verifier(predecessor_plan)
                or predecessor_pin["semantic_sha256"] != predecessor_plan["artifact_sha256"]
                or predecessor_pin["physical_sha256"] != _sha256_file(args.predecessor_plan)
            ):
                raise SelectionPoweredAnalysisError("recovery predecessor plan binding failed")
            recovery_model_id = str(plan["execution"]["deepseek_route_recovery"]["model_id"])
            roster_model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
            base_plan = predecessor_plan
            base_plan_path = args.predecessor_plan
            base_run = args.run_directory
            if plan_schema == PLAN_SCHEMA_VERSION_V33:
                v31_pin = predecessor_plan["inputs"]["plan_v31_predecessor"]
                base_plan = _load(args.base_plan)
                base_plan_path = args.base_plan
                if (
                    not verify_plan_v31(base_plan)
                    or base_plan["artifact_sha256"] != v31_pin["semantic_sha256"]
                    or _sha256_file(args.base_plan) != v31_pin["physical_sha256"]
                ):
                    raise SelectionPoweredAnalysisError("v33 base v31 plan binding failed")
            model_sources = {model_id: (base_run, base_plan) for model_id in roster_model_ids}
            model_sources[recovery_model_id] = (args.recovery_run_directory, plan)
            response_source_lineage = {
                "schema_version": "flavourbench-selection-composite-response-sources-v1",
                "predecessor_models": sorted(
                    model_id for model_id in roster_model_ids if model_id != recovery_model_id
                ),
                "predecessor_plan": {
                    "semantic_sha256": base_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(base_plan_path),
                },
                "recovery_model_id": recovery_model_id,
                "recovery_plan": {
                    "semantic_sha256": plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.plan),
                },
                "cross_provider_response_pooling_within_model": False,
                "superseded_deepseek_responses_used": False,
            }
            if plan_schema == PLAN_SCHEMA_VERSION_V33:
                response_source_lineage["recovery_calibration_plan"] = {
                    "semantic_sha256": predecessor_plan["artifact_sha256"],
                    "physical_sha256": _sha256_file(args.predecessor_plan),
                    "responses_used_as_score_data": False,
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
