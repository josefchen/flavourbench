"""Freeze the complete-response 27-model FlavourBench common core."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_selection_powered_analysis import PanelData, _sha256, _sha256_file
from .epicure_selection_powered_plan_v83 import (
    CORE_FAMILIES,
    MODEL_COUNT,
    PAIRWISE_HYPOTHESES,
    TASKS_PER_PANEL_FAMILY,
)
from .epicure_selection_powered_plan_v83 import (
    verify_plan as verify_plan_v83,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v84"
PLAN_VERSION = "flavourbench-selection-27x534-complete-common-core-v84"
SELECTION_SEED = "flavourbench-all-model-complete-common-core-v1"
TASKS_PER_PANEL = len(CORE_FAMILIES) * TASKS_PER_PANEL_FAMILY
PRIMARY_TASKS = 2 * TASKS_PER_PANEL


class SelectionCompleteCorePlanV84Error(RuntimeError):
    """The complete-response common-core plan failed verification."""


def _task_order_key(*, panel: str, family: str, task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\0{panel}\0{family}\0{task_id}".encode()).hexdigest()


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _freeze_panel(
    *,
    panel: str,
    data: PanelData,
    taskset: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if data.panel != "primary":
        raise SelectionCompleteCorePlanV84Error("only primary responses may enter the score")
    if len(data.model_ids) != MODEL_COUNT or len(set(data.model_ids)) != MODEL_COUNT:
        raise SelectionCompleteCorePlanV84Error("common-core roster cardinality differs")
    if data.scores.shape != (MODEL_COUNT, len(data.task_ids)):
        raise SelectionCompleteCorePlanV84Error("common-core response matrix differs")
    if data.completed.shape != data.scores.shape or data.parseable.shape != data.scores.shape:
        raise SelectionCompleteCorePlanV84Error("common-core validity matrix differs")

    task_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    if set(data.task_ids) != set(task_by_id):
        raise SelectionCompleteCorePlanV84Error("panel tasks differ from the frozen taskset")
    valid = np.all(data.completed & data.parseable, axis=0)
    selected_by_family: dict[str, list[str]] = {}
    available_by_family: dict[str, int] = {}
    selected_tasks: list[Mapping[str, Any]] = []
    for family in CORE_FAMILIES:
        pool = [
            task_id
            for task_id, is_valid in zip(data.task_ids, valid, strict=True)
            if is_valid and task_by_id[task_id]["family"] == family
        ]
        pool.sort(
            key=lambda task_id: _task_order_key(
                panel=panel,
                family=family,
                task_id=task_id,
            )
        )
        available_by_family[family] = len(pool)
        if len(pool) < TASKS_PER_PANEL_FAMILY:
            raise SelectionCompleteCorePlanV84Error(
                f"{panel} has only {len(pool)} complete {family} tasks; "
                f"{TASKS_PER_PANEL_FAMILY} are required"
            )
        selected = pool[:TASKS_PER_PANEL_FAMILY]
        selected_by_family[family] = selected
        selected_tasks.extend(task_by_id[task_id] for task_id in selected)

    selected_ids = [str(task["task_id"]) for task in selected_tasks]
    invalid_counts = {
        model_id: int(np.count_nonzero(~(data.completed[index] & data.parseable[index])))
        for index, model_id in enumerate(data.model_ids)
    }
    record = {
        "panel": panel,
        "available_complete_tasks_by_family": available_by_family,
        "selected_task_ids_by_family": selected_by_family,
        "selected_task_ids_sha256": _sha256(selected_ids),
        "selected_primary_tasks": len(selected_ids),
        "response_artifact_count": len(data.response_artifact_sha256s),
        "response_artifact_set_sha256": _sha256(list(data.response_artifact_sha256s)),
        "validity_matrix_sha256": _sha256(
            {
                "model_ids": list(data.model_ids),
                "task_ids": list(data.task_ids),
                "completed": data.completed.astype(int).tolist(),
                "parseable": data.parseable.astype(int).tolist(),
            }
        ),
        "invalid_candidate_cells_by_model": invalid_counts,
        "selected_invalid_cells": 0,
    }
    return record, selected_tasks


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_data: PanelData,
    panel_1_taskset: Mapping[str, Any],
    panel_1_taskset_path: Path,
    panel_2_data: PanelData,
    panel_2_taskset: Mapping[str, Any],
    panel_2_taskset_path: Path,
    parser_path: Path,
) -> dict[str, Any]:
    """Build a score-blind task freeze from response validity alone."""

    if not verify_plan_v83(predecessor):
        raise SelectionCompleteCorePlanV84Error("v84 requires the exact v83 predecessor")
    if parser_path.is_symlink() or not parser_path.is_file():
        raise SelectionCompleteCorePlanV84Error("analysis parser must be a regular file")
    roster_ids = tuple(str(row["model_id"]) for row in predecessor["roster"]["models"])
    if panel_1_data.model_ids != roster_ids or panel_2_data.model_ids != roster_ids:
        raise SelectionCompleteCorePlanV84Error("response panels differ from the ranked roster")

    panel_1, selected_1 = _freeze_panel(
        panel="panel_1",
        data=panel_1_data,
        taskset=panel_1_taskset,
    )
    panel_2, selected_2 = _freeze_panel(
        panel="panel_2",
        data=panel_2_data,
        taskset=panel_2_taskset,
    )
    selected_tasks = [*selected_1, *selected_2]
    anchors = [str(task["anchor_ingredient"]) for task in selected_tasks]
    family_counts = {
        family: sum(task["family"] == family for task in selected_tasks) for family in CORE_FAMILIES
    }

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256", None)
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "complete_common_core_frozen_before_quality_analysis"
    document["inputs"].update(
        {
            "fable_common_core_plan_v83": _pin(predecessor, predecessor_path),
            "complete_core_panel_1_taskset": _pin(panel_1_taskset, panel_1_taskset_path),
            "complete_core_panel_2_taskset": _pin(panel_2_taskset, panel_2_taskset_path),
            "complete_core_analysis_parser": {
                "physical_sha256": _sha256_file(parser_path),
            },
        }
    )
    document["design"] = {
        "panel_count": 2,
        "ranked_models": MODEL_COUNT,
        "primary_tasks_per_panel": TASKS_PER_PANEL,
        "primary_tasks_per_model": PRIMARY_TASKS,
        "primary_model_task_cells": MODEL_COUNT * PRIMARY_TASKS,
        "family_task_counts": family_counts,
        "independence_unit": "anchor_ingredient",
        "unique_anchor_clusters": len(set(anchors)),
        "tasks_sharing_an_anchor_are_one_bootstrap_cluster": True,
        "complete_response_matrix_required": True,
        "failed_or_unparseable_cells_in_primary_score": 0,
    }
    document["inference"] = {
        "bootstrap_resamples": 50_000,
        "permutation_resamples": 100_000,
        "seed": 20260820,
        "familywise_alpha": 0.05,
        "score_intervals": "anchor-cluster bootstrap with equal panel and family weights",
        "score_uncertainty": "pointwise percentile and simultaneous max-t 95% intervals",
        "paired_tests": (
            "all 351 two-sided anchor-cluster sign-flip tests on the same 534 complete "
            "tasks, with Holm familywise correction"
        ),
        "chance_tests": (
            "anchor-cluster sign flips of taskwise score minus exact combinatorial chance, "
            "with Holm correction across 27 models"
        ),
        "pairwise_missingness": "none; every ranked contrast uses the same 534 tasks",
        "rank_display": "point ranks, simultaneous score intervals, bootstrap rank intervals, "
        "and Holm-derived statistical rank groups",
        "panel_replication_summary": "panel-specific scores and cross-panel correlations",
        "no_result_dependent_test_selection": True,
    }
    document["outcomes"] = {
        "primary_name": "FlavourBench Score",
        "primary_definition": (
            "equal-panel, equal-family mean Epicure score on the frozen 534-task complete "
            "common core"
        ),
        "task_score": "prefrozen continuous lookup over all three-of-eight selections",
        "chance": "exact task-specific mean across all 56 possible selections",
        "failed_content_filtered_or_unparseable": (
            "preserved as transport evidence but absent from the frozen complete common core"
        ),
        "failed_cells_retained_in_provenance": True,
        "failed_cells_scored_as_zero": False,
        "dnf_classification": False,
        "epicure_is_the_executable_reference_not_a_ranked_model": True,
    }
    document["eligibility"] = {
        "ranked_models": MODEL_COUNT,
        "required_valid_tasks_per_model": PRIMARY_TASKS,
        "every_ranked_model_has_identical_task_count": True,
        "missing_or_invalid_ranked_cells": 0,
        "all_ranked_models_eligible": True,
    }
    document["repeatability"] = {
        "status": "separate_supplementary_diagnostic",
        "used_for_primary_ranking": False,
        "reason": "the primary ranking is defined only on the complete 534-task common core",
    }
    document["power"] = {
        "status": "report_observed_precision_not_post_hoc_power",
        "primary_evidence": "simultaneous confidence bands and Holm-adjusted paired contrasts",
        "familywise_comparisons": PAIRWISE_HYPOTHESES,
        "prespecified_bootstrap_resamples": 50_000,
        "prespecified_permutation_resamples": 100_000,
    }
    document["common_core"] = {
        "schema_version": "flavourbench-complete-common-core-v2",
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "within each panel and family, retain tasks with a completed parser-v3-valid "
            "response from every ranked model, hash-order them, and take the first 89"
        ),
        "panels": {"panel_1": panel_1, "panel_2": panel_2},
        "included_families": list(CORE_FAMILIES),
        "excluded_family": "cultural_composition",
        "excluded_family_reason": (
            "the 27-model response-valid intersection cannot support the same balanced "
            "89-task-per-panel cell in this family"
        ),
        "ranked_model_ids": list(roster_ids),
        "ranked_models": MODEL_COUNT,
        "primary_tasks_per_model": PRIMARY_TASKS,
        "model_task_cells": MODEL_COUNT * PRIMARY_TASKS,
        "pairwise_hypotheses": PAIRWISE_HYPOTHESES,
        "panel_weighting": "equal",
        "family_weighting_within_panel": "equal",
        "response_validity_definition": "status completed and parser-v3 parseable",
        "failures_and_unparseable_responses_scored_as_zero": False,
        "requires_one_valid_response_for_every_ranked_model_and_selected_task": True,
        "selection_uses_status_and_parseability_only": True,
        "quality_scores_or_observed_selections_used_for_task_selection": False,
        "selection_is_conditioned_on_all_ranked_models_response_validity": True,
        "selection_is_conditioned_on_fable_response_validity_only": False,
        "estimand_label": "27-model three-family complete-common-core performance",
        "repeat_responses_used_for_primary_ranking": False,
        "full_four_family_26_model_analysis_retained_separately": True,
        "all_response_artifacts_preserved": True,
    }
    document["source_rules"].update(
        {
            "ranked_response_set_all_model_complete_common_core": True,
            "common_core_quality_scores_or_selections_inspected": False,
            "failed_or_unparseable_responses_used_as_score_data": False,
            "task_selection_uses_only_completion_and_parser_validity": True,
            "every_pairwise_contrast_uses_the_same_tasks": True,
        }
    )
    document["claim_boundary"] = (
        "Epicure alignment on a frozen 534-task culinary selection common core; "
        "not universal model quality"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionCompleteCorePlanV84Error("constructed v84 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        core = document["common_core"]
        panels = core["panels"]
    except (KeyError, TypeError):
        return False
    panel_valid = True
    for panel in ("panel_1", "panel_2"):
        record = panels.get(panel) or {}
        selected = record.get("selected_task_ids_by_family") or {}
        flat = [task_id for family in CORE_FAMILIES for task_id in selected.get(family, [])]
        panel_valid = panel_valid and bool(
            set(selected) == set(CORE_FAMILIES)
            and all(len(selected[family]) == TASKS_PER_PANEL_FAMILY for family in CORE_FAMILIES)
            and len(flat) == TASKS_PER_PANEL
            and len(set(flat)) == TASKS_PER_PANEL
            and record.get("selected_task_ids_sha256") == _sha256(flat)
            and record.get("selected_primary_tasks") == TASKS_PER_PANEL
            and record.get("selected_invalid_cells") == 0
            and all(
                int((record.get("available_complete_tasks_by_family") or {}).get(family, 0))
                >= TASKS_PER_PANEL_FAMILY
                for family in CORE_FAMILIES
            )
            and len(str(record.get("response_artifact_set_sha256") or "")) == 64
            and len(str(record.get("validity_matrix_sha256") or "")) == 64
        )
    roster_ids = [str(row["model_id"]) for row in document.get("roster", {}).get("models", [])]
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status") == "complete_common_core_frozen_before_quality_analysis"
        and recorded == _sha256(payload)
        and panel_valid
        and len(roster_ids) == MODEL_COUNT
        and len(set(roster_ids)) == MODEL_COUNT
        and core.get("ranked_model_ids") == roster_ids
        and core.get("ranked_models") == MODEL_COUNT
        and core.get("primary_tasks_per_model") == PRIMARY_TASKS
        and core.get("model_task_cells") == MODEL_COUNT * PRIMARY_TASKS
        and core.get("pairwise_hypotheses") == PAIRWISE_HYPOTHESES
        and core.get("included_families") == list(CORE_FAMILIES)
        and core.get("failures_and_unparseable_responses_scored_as_zero") is False
        and core.get("requires_one_valid_response_for_every_ranked_model_and_selected_task") is True
        and core.get("quality_scores_or_observed_selections_used_for_task_selection") is False
        and core.get("selection_is_conditioned_on_all_ranked_models_response_validity") is True
        and core.get("repeat_responses_used_for_primary_ranking") is False
        and document.get("eligibility", {}).get("all_ranked_models_eligible") is True
        and document.get("design", {}).get("primary_model_task_cells")
        == MODEL_COUNT * PRIMARY_TASKS
        and document.get("inference", {}).get("pairwise_missingness")
        == "none; every ranked contrast uses the same 534 tasks"
    )


def selected_task_ids(document: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exact panel task orders from a verified v84 plan."""

    if not verify_plan(document):
        raise SelectionCompleteCorePlanV84Error("cannot read an invalid v84 plan")
    output: list[tuple[str, ...]] = []
    for panel in ("panel_1", "panel_2"):
        selected = document["common_core"]["panels"][panel]["selected_task_ids_by_family"]
        output.append(tuple(task_id for family in CORE_FAMILIES for task_id in selected[family]))
    return output[0], output[1]


def write_plan(document: Mapping[str, Any], directory: Path) -> Path:
    """Publish one content-addressed plan without replacing an existing file."""

    if not verify_plan(document):
        raise SelectionCompleteCorePlanV84Error("cannot write an invalid v84 plan")
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != payload:
            raise SelectionCompleteCorePlanV84Error("content-addressed v84 plan conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
