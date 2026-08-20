"""Freeze the score-blind Fable-compatible 27-model common-core estimand."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_powered_runner import _semantic_valid
from .epicure_selection_powered_plan_v54 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from .epicure_selection_powered_plan_v55 import verify_plan as verify_plan_v55
from .epicure_selection_powered_plan_v77 import PLAN_SCHEMA_VERSION as PREDECESSOR_SCHEMA
from .epicure_selection_powered_plan_v77 import PLAN_VERSION as PREDECESSOR_VERSION
from .epicure_selection_powered_plan_v77 import verify_plan as verify_plan_v77
from .epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel as verify_repeat_panel_replication,
)
from .epicure_selection_repeat_panel_v2 import verify_repeat_panel as verify_repeat_panel_v2
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID
from .epicure_selection_taskset_replication_v1 import (
    verify_taskset as verify_taskset_replication,
)
from .epicure_selection_taskset_v1 import score_answer
from .epicure_selection_taskset_v2 import verify_taskset as verify_taskset_v2
from .selection_response_parser_v3 import PARSER_SCHEMA_VERSION, parse_final_selection_v3

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v83"
PLAN_VERSION = "flavourbench-selection-27x534-fable-compatible-common-core-v83"
CORE_FAMILIES = ("substitution", "pairing", "constraint")
TASKS_PER_PANEL_FAMILY = 89
TASKS_PER_PANEL = len(CORE_FAMILIES) * TASKS_PER_PANEL_FAMILY
PRIMARY_TASKS = 2 * TASKS_PER_PANEL
MODEL_COUNT = 27
PAIRWISE_HYPOTHESES = MODEL_COUNT * (MODEL_COUNT - 1) // 2


class SelectionPoweredPlanV83Error(RuntimeError):
    """The Fable-compatible common-core plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV83Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV83Error("plan input is not a JSON object")
    return value


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _task_order_key(*, panel: str, family: str, task_id: str) -> str:
    return hashlib.sha256(
        f"flavourbench-fable-common-core-v1\0{panel}\0{family}\0{task_id}".encode()
    ).hexdigest()


def _response_inventory(
    *,
    panel: str,
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    source_directories: Sequence[Path],
) -> tuple[set[str], dict[str, Any]]:
    tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
    rows = [row for row in source_plan["roster"]["models"] if row.get("model_id") == FABLE_MODEL_ID]
    if len(rows) != 1:
        raise SelectionPoweredPlanV83Error("Fable source-plan row is not unique")
    row = rows[0]
    records: list[dict[str, Any]] = []
    valid_by_task: dict[str, tuple[int, str]] = {}
    source_counts: list[dict[str, Any]] = []
    for priority, directory in enumerate(source_directories):
        paths = sorted(
            (directory / "responses" / "primary" / str(row["slot_id"])).glob("response-*.json")
        )
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        for path in paths:
            document = _load(path)
            if not _semantic_valid(document):
                raise SelectionPoweredPlanV83Error(f"response semantic hash failed: {path}")
            artifact = str(document["artifact_sha256"])
            cell_id = str(document.get("cell_id") or "")
            if path.name != f"response-{cell_id}-{artifact}.json":
                raise SelectionPoweredPlanV83Error("response filename is not content addressed")
            task_id = str(document.get("task_id") or "")
            if task_id not in tasks or task_id in seen:
                raise SelectionPoweredPlanV83Error("unexpected or duplicated Fable response cell")
            seen.add(task_id)
            task = tasks[task_id]
            exact = {
                "schema_version": "flavourbench-powered-response-v1",
                "panel": "primary",
                "plan_sha256": source_plan["artifact_sha256"],
                "manifest_sha256": source_plan["inputs"]["route_manifest"]["semantic_sha256"],
                "taskset_sha256": taskset["artifact_sha256"],
                "repeat_panel_sha256": repeat_panel["artifact_sha256"],
                "family": task["family"],
                "model_id": FABLE_MODEL_ID,
                "slot_id": row["slot_id"],
                "model_name": row["model_name"],
                "canonical_model_slug": row["canonical_model_slug"],
                "execution_backend": row["execution_backend"],
                "endpoint_execution_sha256": row["endpoint_execution_sha256"],
                "backend_contract_sha256": row["backend_contract_sha256"],
                "prompt_sha256": task["prompt_sha256"],
                "optimal_selection": task["optimal_selection"],
                "original_task_id": task.get("original_task_id"),
            }
            if any(document.get(key) != value for key, value in exact.items()):
                raise SelectionPoweredPlanV83Error("Fable response binding differs from inputs")
            status = document.get("status")
            if status not in {"completed", "failed"}:
                raise SelectionPoweredPlanV83Error("unsupported Fable response status")
            valid = False
            if status == "completed":
                answer = (document.get("generation") or {}).get("answer_markdown")
                if not isinstance(answer, str):
                    raise SelectionPoweredPlanV83Error("completed Fable response lacks answer")
                if document.get("scoring") != score_answer(task, answer):
                    raise SelectionPoweredPlanV83Error(
                        "historical Fable scoring does not reproduce"
                    )
                valid = parse_final_selection_v3(task, answer) is not None
            counts["valid" if valid else str(status)] += 1
            records.append(
                {
                    "artifact_sha256": artifact,
                    "filename": path.name,
                    "physical_sha256": _sha256_file(path),
                    "source_priority": priority,
                    "status": status,
                    "task_id": task_id,
                    "valid_under_parser_v3": valid,
                }
            )
            if valid and task_id not in valid_by_task:
                valid_by_task[task_id] = (priority, artifact)
        source_counts.append(
            {
                "source_priority": priority,
                "response_artifacts": len(paths),
                "valid_responses": counts["valid"],
                "failed_responses": counts["failed"],
                "completed_unparseable_responses": counts["completed"],
            }
        )
    records.sort(key=lambda item: (item["source_priority"], item["filename"]))
    return set(valid_by_task), {
        "panel": panel,
        "source_directory_count": len(source_directories),
        "source_counts": source_counts,
        "response_artifact_count": len(records),
        "response_artifact_set_sha256": _sha256(records),
        "valid_unique_primary_tasks": len(valid_by_task),
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_path: Path,
    panel_1_taskset: Mapping[str, Any],
    panel_1_taskset_path: Path,
    panel_1_repeat: Mapping[str, Any],
    panel_1_repeat_path: Path,
    panel_1_source_plan: Mapping[str, Any],
    panel_1_source_plan_path: Path,
    panel_1_source_directories: Sequence[Path],
    panel_2_taskset: Mapping[str, Any],
    panel_2_taskset_path: Path,
    panel_2_repeat: Mapping[str, Any],
    panel_2_repeat_path: Path,
    panel_2_source_plan: Mapping[str, Any],
    panel_2_source_plan_path: Path,
    panel_2_source_directories: Sequence[Path],
    parser_path: Path,
) -> dict[str, Any]:
    if not verify_plan_v77(predecessor):
        raise SelectionPoweredPlanV83Error("v83 requires the exact v77 predecessor")
    if not (
        verify_taskset_v2(panel_1_taskset)
        and verify_repeat_panel_v2(panel_1_repeat, taskset=panel_1_taskset)
        and verify_plan_v55(panel_1_source_plan)
        and verify_taskset_replication(panel_2_taskset)
        and verify_repeat_panel_replication(panel_2_repeat, taskset=panel_2_taskset)
        and verify_plan_v54(panel_2_source_plan)
    ):
        raise SelectionPoweredPlanV83Error("v83 source inputs failed verification")
    if parser_path.is_symlink() or not parser_path.is_file():
        raise SelectionPoweredPlanV83Error("parser source must be regular and non-symlink")

    panels: dict[str, Any] = {}
    for panel, taskset, repeat, source_plan, directories in (
        (
            "panel_1",
            panel_1_taskset,
            panel_1_repeat,
            panel_1_source_plan,
            panel_1_source_directories,
        ),
        (
            "panel_2",
            panel_2_taskset,
            panel_2_repeat,
            panel_2_source_plan,
            panel_2_source_directories,
        ),
    ):
        valid, inventory = _response_inventory(
            panel=panel,
            taskset=taskset,
            repeat_panel=repeat,
            source_plan=source_plan,
            source_directories=directories,
        )
        tasks = {str(task["task_id"]): task for task in taskset["tasks"]}
        selected: dict[str, list[str]] = {}
        available: dict[str, int] = {}
        for family in CORE_FAMILIES:
            pool = [
                task_id
                for task_id in valid
                if task_id in tasks and tasks[task_id]["family"] == family
            ]
            pool.sort(
                key=lambda task_id: _task_order_key(panel=panel, family=family, task_id=task_id)
            )
            available[family] = len(pool)
            if len(pool) < TASKS_PER_PANEL_FAMILY:
                raise SelectionPoweredPlanV83Error(
                    f"{panel} lacks {TASKS_PER_PANEL_FAMILY} valid {family} tasks"
                )
            selected[family] = pool[:TASKS_PER_PANEL_FAMILY]
        task_ids = [task_id for family in CORE_FAMILIES for task_id in selected[family]]
        panels[panel] = {
            **inventory,
            "available_valid_tasks_by_family": available,
            "selected_task_ids_by_family": selected,
            "selected_task_ids_sha256": _sha256(task_ids),
            "selected_primary_tasks": len(task_ids),
        }

    document = copy.deepcopy(predecessor)
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "fable_compatible_common_core_frozen_before_quality_analysis"
    document["inputs"].update(
        {
            "joint_plan_v77_predecessor": _pin(predecessor, predecessor_path),
            "panel_1_taskset_v2": _pin(panel_1_taskset, panel_1_taskset_path),
            "panel_1_repeat_v2": _pin(panel_1_repeat, panel_1_repeat_path),
            "panel_1_fable_source_plan": _pin(panel_1_source_plan, panel_1_source_plan_path),
            "panel_2_taskset_replication": _pin(panel_2_taskset, panel_2_taskset_path),
            "panel_2_repeat_replication": _pin(panel_2_repeat, panel_2_repeat_path),
            "panel_2_fable_source_plan": _pin(panel_2_source_plan, panel_2_source_plan_path),
            "analysis_parser_v3": {
                "schema_version": PARSER_SCHEMA_VERSION,
                "physical_sha256": _sha256_file(parser_path),
            },
        }
    )
    document["roster"]["ranked_model_count"] = MODEL_COUNT
    document["roster"]["pairwise_hypotheses"] = PAIRWISE_HYPOTHESES
    document["common_core"] = {
        "schema_version": "flavourbench-fable-compatible-common-core-v1",
        "selection_seed": "flavourbench-fable-common-core-v1",
        "selection_rule": (
            "within each panel and family, hash-order tasks with a completed response "
            "parseable under parser v3; take the first 89"
        ),
        "panels": panels,
        "included_families": list(CORE_FAMILIES),
        "excluded_family": "cultural_composition",
        "excluded_family_reason": (
            "Fable has fewer than 89 valid tasks in each panel; retain cultural tasks in the "
            "full 26-model four-family supplement rather than score them as failures"
        ),
        "ranked_model_ids": [str(row["model_id"]) for row in document["roster"]["models"]],
        "ranked_models": MODEL_COUNT,
        "primary_tasks_per_model": PRIMARY_TASKS,
        "model_task_cells": MODEL_COUNT * PRIMARY_TASKS,
        "pairwise_hypotheses": PAIRWISE_HYPOTHESES,
        "panel_weighting": "equal",
        "family_weighting_within_panel": "equal",
        "failures_and_unparseable_responses_scored_as_zero": False,
        "requires_one_valid_response_for_every_ranked_model_and_selected_task": True,
        "fable_validity_screen_uses_status_and_parseability_only": True,
        "quality_scores_or_observed_selections_used_for_task_selection": False,
        "selection_is_conditioned_on_fable_response_validity": True,
        "estimand_label": "Fable-compatible three-family common-core performance",
        "repeat_responses_used_for_primary_ranking": False,
        "full_four_family_26_model_analysis_retained_separately": True,
        "all_response_artifacts_preserved": True,
    }
    document["source_rules"].update(
        {
            "ranked_response_set_fable_compatible_common_core": True,
            "common_core_quality_scores_or_selections_inspected": False,
            "failed_or_unparseable_responses_used_as_score_data": False,
            "full_panel_and_common_core_estimands_must_not_be_conflated": True,
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV83Error("constructed v83 plan failed verification")
    return document


def _as_v77(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = copy.deepcopy(document)
    prior.pop("artifact_sha256", None)
    prior["schema_version"] = PREDECESSOR_SCHEMA
    prior["plan_version"] = PREDECESSOR_VERSION
    prior["status"] = "joint_source_lineage_frozen_after_coverage_before_quality_analysis"
    for key in (
        "joint_plan_v77_predecessor",
        "panel_1_taskset_v2",
        "panel_1_repeat_v2",
        "panel_1_fable_source_plan",
        "panel_2_taskset_replication",
        "panel_2_repeat_replication",
        "panel_2_fable_source_plan",
        "analysis_parser_v3",
    ):
        prior["inputs"].pop(key)
    prior["roster"].pop("ranked_model_count")
    prior["roster"]["pairwise_hypotheses"] = PAIRWISE_HYPOTHESES
    prior.pop("common_core")
    for key in (
        "ranked_response_set_fable_compatible_common_core",
        "common_core_quality_scores_or_selections_inspected",
        "failed_or_unparseable_responses_used_as_score_data",
        "full_panel_and_common_core_estimands_must_not_be_conflated",
    ):
        prior["source_rules"].pop(key)
    prior["artifact_sha256"] = _sha256(prior)
    return prior


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        core = document["common_core"]
        panels = core["panels"]
        inputs = document["inputs"]
    except (KeyError, TypeError):
        return False
    pins = [
        inputs[key]
        for key in (
            "joint_plan_v77_predecessor",
            "panel_1_taskset_v2",
            "panel_1_repeat_v2",
            "panel_1_fable_source_plan",
            "panel_2_taskset_replication",
            "panel_2_repeat_replication",
            "panel_2_fable_source_plan",
        )
    ]
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
            and record.get("selected_primary_tasks") == TASKS_PER_PANEL
            and record.get("selected_task_ids_sha256") == _sha256(flat)
            and all(
                int((record.get("available_valid_tasks_by_family") or {}).get(family, 0))
                >= TASKS_PER_PANEL_FAMILY
                for family in CORE_FAMILIES
            )
            and isinstance(record.get("response_artifact_set_sha256"), str)
            and len(record["response_artifact_set_sha256"]) == 64
        )
    roster_ids = [str(row["model_id"]) for row in document.get("roster", {}).get("models", [])]
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and document.get("status") == "fable_compatible_common_core_frozen_before_quality_analysis"
        and recorded == _sha256(payload)
        and verify_plan_v77(_as_v77(document))
        and panel_valid
        and len(roster_ids) == MODEL_COUNT
        and len(set(roster_ids)) == MODEL_COUNT
        and FABLE_MODEL_ID in roster_ids
        and document["roster"].get("ranked_model_count") == MODEL_COUNT
        and document["roster"].get("pairwise_hypotheses") == PAIRWISE_HYPOTHESES
        and core.get("ranked_model_ids") == roster_ids
        and core.get("ranked_models") == MODEL_COUNT
        and core.get("primary_tasks_per_model") == PRIMARY_TASKS
        and core.get("model_task_cells") == MODEL_COUNT * PRIMARY_TASKS
        and core.get("pairwise_hypotheses") == PAIRWISE_HYPOTHESES
        and core.get("included_families") == list(CORE_FAMILIES)
        and core.get("excluded_family") == "cultural_composition"
        and core.get("failures_and_unparseable_responses_scored_as_zero") is False
        and core.get("requires_one_valid_response_for_every_ranked_model_and_selected_task") is True
        and core.get("quality_scores_or_observed_selections_used_for_task_selection") is False
        and core.get("selection_is_conditioned_on_fable_response_validity") is True
        and core.get("repeat_responses_used_for_primary_ranking") is False
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in pins
        )
        and inputs.get("analysis_parser_v3", {}).get("schema_version") == PARSER_SCHEMA_VERSION
        and len(inputs.get("analysis_parser_v3", {}).get("physical_sha256", "")) == 64
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV83Error("content-addressed v83 plan conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
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
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--panel-1-taskset", type=Path, required=True)
    parser.add_argument("--panel-1-repeat", type=Path, required=True)
    parser.add_argument("--panel-1-source-plan", type=Path, required=True)
    parser.add_argument("--panel-1-source-directory", type=Path, action="append", required=True)
    parser.add_argument("--panel-2-taskset", type=Path, required=True)
    parser.add_argument("--panel-2-repeat", type=Path, required=True)
    parser.add_argument("--panel-2-source-plan", type=Path, required=True)
    parser.add_argument("--panel-2-source-directory", type=Path, action="append", required=True)
    parser.add_argument("--parser-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            build_plan(
                predecessor=_load(args.predecessor),
                predecessor_path=args.predecessor,
                panel_1_taskset=_load(args.panel_1_taskset),
                panel_1_taskset_path=args.panel_1_taskset,
                panel_1_repeat=_load(args.panel_1_repeat),
                panel_1_repeat_path=args.panel_1_repeat,
                panel_1_source_plan=_load(args.panel_1_source_plan),
                panel_1_source_plan_path=args.panel_1_source_plan,
                panel_1_source_directories=args.panel_1_source_directory,
                panel_2_taskset=_load(args.panel_2_taskset),
                panel_2_taskset_path=args.panel_2_taskset,
                panel_2_repeat=_load(args.panel_2_repeat),
                panel_2_repeat_path=args.panel_2_repeat,
                panel_2_source_plan=_load(args.panel_2_source_plan),
                panel_2_source_plan_path=args.panel_2_source_plan,
                panel_2_source_directories=args.panel_2_source_directory,
                parser_path=args.parser_source,
            ),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
