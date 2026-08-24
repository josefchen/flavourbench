"""Freeze the blinded human protocol for the 24-task effort study.

This module is intentionally planning-only.  It reads the existing real-human
development-task dossier, applies the current task quarantine, and writes a
content-addressed allocation.  It never contacts a model provider, Epicure,
or a reviewer service and it never creates a human judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_frontier_task_quarantine import build_quarantine_artifact

SCHEMA_VERSION = "flavourbench-reasoning-effort-human-evaluation-protocol-v1"
BALLOT_SCHEMA_VERSION = "flavourbench-reasoning-effort-human-ballot-v1"
TASK_DOSSIER_SHA256 = "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119"
QUARANTINE_SHA256 = "e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15"
TASK_SELECTION_ARTIFACT_SHA256 = "ebb62d92fc0589d1b8473b72682e8c17f6f90d7d2b0b2fd23e9064d63958044a"
SELECTED_TASK_SET_SHA256 = "825526a211edd98a242f6eb5706114f2dc9c9921cc051c85b83d133a1bcbd682"
WAVE_ORDER_SHA256 = "50ad328c7276399cc7933b97fd6bf6ad04aa5859979d8b60561f1359dab3dafd"
RANDOMIZATION_NONCE = "flavourbench-effort-human-v2-blinded-sides-2026-08-04"
ORDER_NONCE = "flavourbench-effort-human-v2-block-order-2026-08-04"
BOOTSTRAP_NONCE = "flavourbench-effort-human-v2-task-bootstrap-2026-08-04"

FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONDITIONS = ("epicure_off", "epicure_on")
COHORTS = ("culinary_expert", "general_public")
REPLICATION_SLOTS = (1, 2, 3)

ENDPOINTS: tuple[dict[str, Any], ...] = (
    {
        "endpoint_id": "sonnet",
        "model_id": "anthropic/claude-sonnet-5",
        "canonical_model_slug": "anthropic/claude-sonnet-5-20260630",
        "provider_endpoint": "anthropic",
        "actual_provider_name": "Anthropic",
        "provider_controls": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "only": ["anthropic"],
            "require_parameters": True,
        },
        "supported_variants": ["explicit_low", "explicit_high"],
        "provider_default_in_study": False,
    },
    {
        "endpoint_id": "gemini",
        "model_id": "google/gemini-3.6-flash",
        "canonical_model_slug": "google/gemini-3.6-flash-20260721",
        "provider_endpoint": "google-ai-studio/flex",
        "actual_provider_name": "Google AI Studio",
        "provider_controls": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "only": ["google-ai-studio/flex"],
            "require_parameters": True,
        },
        "supported_variants": ["explicit_low", "provider_default", "explicit_high"],
        "provider_default_effort": "medium",
        "provider_default_in_study": True,
    },
    {
        "endpoint_id": "deepseek",
        "model_id": "deepseek/deepseek-v4-flash-0731",
        "canonical_model_slug": "deepseek/deepseek-v4-flash-20260731",
        "provider_endpoint": "deepinfra/fp4",
        "actual_provider_name": "DeepInfra",
        "provider_controls": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "only": ["deepinfra/fp4"],
            "require_parameters": True,
        },
        "supported_variants": ["explicit_low", "explicit_high"],
        "provider_default_in_study": False,
    },
)

VARIANTS = {
    "explicit_low": {
        "intermediate_reasoning_effort": "low",
        "final_reasoning_effort": "low",
        "request_semantics": "reasoning_effort_explicit_low",
    },
    "provider_default": {
        "intermediate_reasoning_effort": None,
        "final_reasoning_effort": None,
        "request_semantics": "reasoning_parameter_omitted",
    },
    "explicit_high": {
        "intermediate_reasoning_effort": "high",
        "final_reasoning_effort": "high",
        "request_semantics": "reasoning_effort_explicit_high",
    },
}

COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "comparison_id": "sonnet5_low_high_off",
        "endpoint_id": "sonnet",
        "condition": "epicure_off",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "sonnet5_low_high_on",
        "endpoint_id": "sonnet",
        "condition": "epicure_on",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "gemini36_low_high_off",
        "endpoint_id": "gemini",
        "condition": "epicure_off",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "gemini36_low_high_on",
        "endpoint_id": "gemini",
        "condition": "epicure_on",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "deepseekv4_low_high_off",
        "endpoint_id": "deepseek",
        "condition": "epicure_off",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "deepseekv4_low_high_on",
        "endpoint_id": "deepseek",
        "condition": "epicure_on",
        "lower_variant": "explicit_low",
        "upper_variant": "explicit_high",
        "analysis_role": "primary",
    },
    {
        "comparison_id": "gemini36_low_default_off",
        "endpoint_id": "gemini",
        "condition": "epicure_off",
        "lower_variant": "explicit_low",
        "upper_variant": "provider_default",
        "analysis_role": "secondary",
    },
    {
        "comparison_id": "gemini36_low_default_on",
        "endpoint_id": "gemini",
        "condition": "epicure_on",
        "lower_variant": "explicit_low",
        "upper_variant": "provider_default",
        "analysis_role": "secondary",
    },
    {
        "comparison_id": "gemini36_default_high_off",
        "endpoint_id": "gemini",
        "condition": "epicure_off",
        "lower_variant": "provider_default",
        "upper_variant": "explicit_high",
        "analysis_role": "secondary",
    },
    {
        "comparison_id": "gemini36_default_high_on",
        "endpoint_id": "gemini",
        "condition": "epicure_on",
        "lower_variant": "provider_default",
        "upper_variant": "explicit_high",
        "analysis_role": "secondary",
    },
)

RUBRIC_DIMENSIONS = (
    "task_completion",
    "constraint_compliance",
    "coherence",
    "sensory_promise",
    "cookability",
    "clarity",
    "originality",
    "evidence_use",
    "calibration",
)


class HumanProtocolError(ValueError):
    """The frozen protocol or one of its immutable inputs is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HumanProtocolError(f"input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HumanProtocolError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise HumanProtocolError(f"JSON input is not an object: {path}")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise HumanProtocolError(f"input is outside repository root: {path}") from error


def _verified_dossier(path: Path) -> dict[str, Any]:
    dossier = _read_object(path)
    digest = str(dossier.get("artifact_sha256") or "")
    payload = {key: value for key, value in dossier.items() if key != "artifact_sha256"}
    if digest != TASK_DOSSIER_SHA256 or _sha256(payload) != digest:
        raise HumanProtocolError("task dossier content address differs")
    if dossier.get("counts", {}).get("per_family") != {family: 10 for family in FAMILIES}:
        raise HumanProtocolError("task dossier no longer has ten candidates per family")
    return dossier


def _verified_executor_plan(path: Path, *, repo_root: Path) -> dict[str, Any]:
    from .reasoning_effort_full_study_v1 import PLAN_SCHEMA, validate_plan

    plan = _read_object(path)
    digest = str(plan.get("artifact_sha256") or "")
    payload = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    if plan.get("schema_version") != PLAN_SCHEMA or _sha256(payload) != digest:
        raise HumanProtocolError("executor study plan content address differs")
    validate_plan(plan, repo_root=repo_root)
    return plan


def _reasoning_task_wave_binding(executor_plan: Mapping[str, Any]) -> dict[str, Any]:
    from .reasoning_effort_full_study_v1 import canonical_task_wave_identity

    identity = canonical_task_wave_identity(
        tasks=executor_plan.get("tasks") or [],
        waves=executor_plan.get("task_waves") or [],
    )
    if identity != executor_plan.get("task_wave_identity"):
        raise HumanProtocolError("executor task-wave identity does not rederive")
    task_selection_sha256 = str(
        executor_plan.get("source_artifacts", {})
        .get("task_selection", {})
        .get("semantic_sha256", "")
    )
    if (
        task_selection_sha256 != TASK_SELECTION_ARTIFACT_SHA256
        or identity["selected_task_set_sha256"] != SELECTED_TASK_SET_SHA256
        or identity["wave_order_sha256"] != WAVE_ORDER_SHA256
    ):
        raise HumanProtocolError("executor task selection or wave order changed")
    return {
        "study_plan_sha256": executor_plan["artifact_sha256"],
        "task_selection_artifact_sha256": task_selection_sha256,
        "selected_task_set_sha256": identity["selected_task_set_sha256"],
        "wave_order_sha256": identity["wave_order_sha256"],
        "selected_tasks": identity["selected_tasks"],
        "ordered_waves": identity["ordered_waves"],
    }


def _selected_tasks(
    dossier: Mapping[str, Any], executor_plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    quarantine = build_quarantine_artifact()
    if quarantine.get("artifact_sha256") != QUARANTINE_SHA256:
        raise HumanProtocolError("task quarantine content address differs")
    excluded = {str(row["task_id"]) for row in quarantine["records"]}
    excluded |= {str(row["task_id"]) for row in dossier.get("specialist_quarantine", [])}
    excluded |= {str(row["task_id"]) for row in dossier.get("surface_dependency_quarantine", [])}
    candidates: dict[str, dict[str, Any]] = {}
    for raw in dossier.get("tasks", []):
        if not isinstance(raw, Mapping):
            raise HumanProtocolError("task dossier contains a non-object row")
        task_id = str(raw.get("task_id") or "")
        family = str(raw.get("family") or "")
        source_license = str(raw.get("source_license") or "")
        surface = raw.get("surface_dependency_screen") or {}
        if (
            family not in FAMILIES
            or task_id in excluded
            or raw.get("confirmatory_eligible") is not False
            or raw.get("rank_eligible") is not False
            or surface.get("status") != "pass"
            or not source_license.startswith("CC BY-SA ")
        ):
            continue
        candidates[task_id] = {
            "task_id": task_id,
            "family": family,
            "task_sha256": str(raw.get("task_sha256") or ""),
            "prompt_sha256": str(raw.get("prompt_sha256") or ""),
            "source_license": source_license,
            "source_created_utc": str(raw.get("source_created_utc") or ""),
            "confirmatory_eligible": False,
            "rank_eligible": False,
        }
    waves = executor_plan.get("task_waves") or []
    if (
        len(waves) != 24
        or len({str(row.get("wave_id") or "") for row in waves}) != 24
        or [row.get("wave_id") for row in waves] != executor_plan.get("wave_execution_order")
    ):
        raise HumanProtocolError("executor plan does not have 24 unique ordered waves")
    chosen: list[dict[str, Any]] = []
    for index, wave in enumerate(waves):
        task_id = str(wave.get("task_id") or "")
        source = candidates.get(task_id)
        if source is None or source["family"] != wave.get("task_family"):
            raise HumanProtocolError("executor wave task is not an eligible dossier task")
        if (
            wave.get("matched_pairs") != 7
            or wave.get("response_arms") != 14
            or len(wave.get("work_item_ids") or []) != 7
        ):
            raise HumanProtocolError("executor task wave is not seven pairs and fourteen arms")
        chosen.append(
            {
                **source,
                "task_wave_id": str(wave["wave_id"]),
                "task_wave_ordinal": int(wave["wave_ordinal"]),
                "task_index": index,
                "executor_work_item_ids": list(wave["work_item_ids"]),
                "generation_pairs": 7,
                "generation_response_arms": 14,
            }
        )
    if Counter(row["family"] for row in chosen) != Counter({family: 6 for family in FAMILIES}):
        raise HumanProtocolError("executor task order is not six tasks per family")
    return chosen


def _arm_id(*, executor_arm_id: str, work_item_id: str) -> str:
    return _sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "executor_arm_id": executor_arm_id,
            "executor_work_item_id": work_item_id,
        }
    )


def _build_graph(
    tasks: Sequence[Mapping[str, Any]], executor_plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_by_id = {row["endpoint_id"]: row for row in ENDPOINTS}
    plan_models = executor_plan.get("models") or {}
    for endpoint_id, endpoint in endpoint_by_id.items():
        observed = plan_models.get(endpoint_id) or {}
        for field in (
            "model_id",
            "canonical_model_slug",
            "provider_endpoint",
            "actual_provider_name",
        ):
            if observed.get(field) != endpoint[field]:
                raise HumanProtocolError(f"executor endpoint differs for {endpoint_id}:{field}")
    task_by_id = {str(row["task_id"]): row for row in tasks}
    work_lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for work_item in executor_plan.get("work_items") or []:
        coordinate = work_item.get("route_coordinate") or {}
        key = (
            str(coordinate.get("task_id") or ""),
            str(coordinate.get("endpoint_id") or ""),
            str(coordinate.get("variant_id") or ""),
        )
        if key in work_lookup:
            raise HumanProtocolError("executor work-item coordinate is duplicated")
        work_lookup[key] = work_item
    if len(work_lookup) != 168:
        raise HumanProtocolError("executor plan does not contain 168 generation pairs")
    arms: list[dict[str, Any]] = []
    arm_lookup: dict[tuple[str, str, str, str], str] = {}
    work_id_lookup: dict[tuple[str, str, str], str] = {}
    for (task_id, endpoint_id, variant), work_item in work_lookup.items():
        task = task_by_id.get(task_id)
        endpoint = endpoint_by_id.get(endpoint_id)
        if task is None or endpoint is None or variant not in endpoint["supported_variants"]:
            raise HumanProtocolError("executor work item is outside the human study graph")
        work_item_id = str(work_item.get("work_item_id") or "")
        work_id_lookup[(task_id, endpoint_id, variant)] = work_item_id
        executor_arm_ids = {
            str(arm_id).rsplit(":", 1)[-1]: str(arm_id) for arm_id in work_item.get("arm_ids") or []
        }
        if set(executor_arm_ids) != set(CONDITIONS):
            raise HumanProtocolError("executor work item does not bind off/on generation arms")
        for condition in CONDITIONS:
            executor_arm_id = executor_arm_ids[condition]
            coordinate_id = _arm_id(executor_arm_id=executor_arm_id, work_item_id=work_item_id)
            key = (task_id, endpoint_id, condition, variant)
            arm_lookup[key] = coordinate_id
            arms.append(
                {
                    "arm_coordinate_id": coordinate_id,
                    "executor_arm_id": executor_arm_id,
                    "executor_work_item_id": work_item_id,
                    "task_id": task_id,
                    "task_sha256": task["task_sha256"],
                    "task_wave_id": task["task_wave_id"],
                    "family": task["family"],
                    "endpoint_id": endpoint_id,
                    "canonical_model_slug": endpoint["canonical_model_slug"],
                    "provider_endpoint": endpoint["provider_endpoint"],
                    "condition": condition,
                    "variant": variant,
                    "variant_contract": VARIANTS[variant],
                    "response_artifact_sha256": None,
                    "generation_id": None,
                    "status": "planned_not_generated",
                }
            )
    if len(arms) != 336:
        raise HumanProtocolError("executor plan does not map to 336 response arms")
    comparison_lookup = {
        (
            row["endpoint_id"],
            row["condition"],
            row["lower_variant"],
            row["upper_variant"],
        ): (index, row)
        for index, row in enumerate(COMPARISONS)
    }
    executor_presentations = executor_plan.get("human_evaluation", {}).get("presentations") or []
    if len(executor_presentations) != 240:
        raise HumanProtocolError("executor human graph is not 240 unique presentations")
    cells: list[dict[str, Any]] = []
    for executor_presentation in executor_presentations:
        task_id = str(executor_presentation.get("task_id") or "")
        task = task_by_id.get(task_id)
        endpoint_id = str(executor_presentation.get("endpoint_id") or "")
        endpoint = endpoint_by_id.get(endpoint_id)
        lower_variant = str(executor_presentation.get("first_variant") or "")
        upper_variant = str(executor_presentation.get("second_variant") or "")
        condition = str(executor_presentation.get("condition") or "")
        match = comparison_lookup.get((endpoint_id, condition, lower_variant, upper_variant))
        if task is None or endpoint is None or match is None:
            raise HumanProtocolError("executor human presentation differs from contrasts")
        comparison_index, comparison = match
        lower_key = (task_id, endpoint_id, condition, lower_variant)
        upper_key = (task_id, endpoint_id, condition, upper_variant)
        coordinate = {
            "task_id": task_id,
            "task_sha256": task["task_sha256"],
            "endpoint_id": endpoint_id,
            "canonical_model_slug": endpoint["canonical_model_slug"],
            "provider_endpoint": endpoint["provider_endpoint"],
            "condition": condition,
            "comparison_id": comparison["comparison_id"],
            "lower_variant": lower_variant,
            "upper_variant": upper_variant,
            "analysis_role": comparison["analysis_role"],
        }
        lower_work_id = work_id_lookup[(task_id, endpoint_id, lower_variant)]
        base_lower_left = executor_presentation.get("left_work_item_id") == lower_work_id
        cells.append(
            {
                "cell_id": str(executor_presentation["presentation_id"]),
                "executor_presentation_id": str(executor_presentation["presentation_id"]),
                "executor_base_orientation": ("lower_left" if base_lower_left else "upper_left"),
                **coordinate,
                "family": task["family"],
                "task_wave_id": task["task_wave_id"],
                "task_index": task["task_index"],
                "comparison_index": comparison_index,
                "base_block_id": (
                    f"base-block-{(int(task['task_index']) + comparison_index) % 12 + 1:02d}"
                ),
                "lower_arm_coordinate_id": arm_lookup[lower_key],
                "upper_arm_coordinate_id": arm_lookup[upper_key],
                "planned_judgments_per_cohort": 3,
            }
        )
    if len({str(row["cell_id"]) for row in cells}) != 240:
        raise HumanProtocolError("executor presentation IDs are not unique")
    return arms, cells


def _orientation(cell: Mapping[str, Any], cohort: str, replication_slot: int) -> str:
    bit = int(cell["executor_base_orientation"] == "upper_left")
    if replication_slot == 2:
        bit = 1 - bit
    elif replication_slot == 3 and cohort == "general_public":
        bit = 1 - bit
    return "lower_left" if bit == 0 else "upper_left"


def _presentation(
    *,
    cell: Mapping[str, Any],
    cohort: str,
    replication_slot: int,
    assignment_block_id: str,
    orientation: str,
) -> dict[str, Any]:
    if orientation == "lower_left":
        left = cell["lower_arm_coordinate_id"]
        right = cell["upper_arm_coordinate_id"]
    else:
        left = cell["upper_arm_coordinate_id"]
        right = cell["lower_arm_coordinate_id"]
    presentation_id = _sha256(
        {
            "cell_id": cell["cell_id"],
            "cohort": cohort,
            "replication_slot": replication_slot,
            "assignment_block_id": assignment_block_id,
            "orientation": orientation,
            "repeat": False,
        }
    )
    return {
        "presentation_id": presentation_id,
        "cell_id": cell["cell_id"],
        "task_id": cell["task_id"],
        "family": cell["family"],
        "task_wave_id": cell["task_wave_id"],
        "base_block_id": cell["base_block_id"],
        "assignment_block_id": assignment_block_id,
        "cohort": cohort,
        "replication_slot": replication_slot,
        "analysis_role": cell["analysis_role"],
        "blind_left_arm_token": left,
        "blind_right_arm_token": right,
        "sealed_orientation": orientation,
        "is_repeat": False,
        "repeat_source_presentation_id": None,
        "counts_in_effect_estimate": True,
        "sequence_position": None,
    }


def _repeat(source: Mapping[str, Any], *, repeat_number: int) -> dict[str, Any]:
    repeat_id = _sha256(
        {
            "source_presentation_id": source["presentation_id"],
            "repeat_number": repeat_number,
            "repeat_policy": "exact_pair_position_swapped",
        }
    )
    return {
        **source,
        "presentation_id": repeat_id,
        "blind_left_arm_token": source["blind_right_arm_token"],
        "blind_right_arm_token": source["blind_left_arm_token"],
        "sealed_orientation": (
            "upper_left" if source["sealed_orientation"] == "lower_left" else "lower_left"
        ),
        "is_repeat": True,
        "repeat_source_presentation_id": source["presentation_id"],
        "counts_in_effect_estimate": False,
        "sequence_position": None,
    }


def _build_presentations(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_block: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_block[str(cell["base_block_id"])].append(cell)
    if set(by_block) != {f"base-block-{index:02d}" for index in range(1, 13)}:
        raise HumanProtocolError("presentation graph did not fill all twelve base blocks")
    presentations: list[dict[str, Any]] = []
    assignment_blocks: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for replication_slot in REPLICATION_SLOTS:
            for base_block_id in sorted(by_block):
                block_cells = by_block[base_block_id]
                if len(block_cells) != 20 or len({row["task_id"] for row in block_cells}) != 20:
                    raise HumanProtocolError("a base block is not twenty distinct tasks")
                assignment_block_id = f"{cohort}-rep-{replication_slot}-{base_block_id}"
                originals = [
                    _presentation(
                        cell=cell,
                        cohort=cohort,
                        replication_slot=replication_slot,
                        assignment_block_id=assignment_block_id,
                        orientation=_orientation(cell, cohort, replication_slot),
                    )
                    for cell in block_cells
                ]
                originals.sort(
                    key=lambda row: hashlib.sha256(
                        f"{ORDER_NONCE}|{assignment_block_id}|{row['presentation_id']}".encode()
                    ).hexdigest()
                )
                repeat_one = _repeat(originals[0], repeat_number=1)
                repeat_two = _repeat(originals[7], repeat_number=2)
                ordered = [*originals[:14], repeat_one, *originals[14:], repeat_two]
                for position, row in enumerate(ordered, start=1):
                    row["sequence_position"] = position
                    presentations.append(row)
                assignment_blocks.append(
                    {
                        "assignment_block_id": assignment_block_id,
                        "base_block_id": base_block_id,
                        "cohort": cohort,
                        "replication_slot": replication_slot,
                        "original_presentations": 20,
                        "position_swapped_repeats": 2,
                        "presentation_count": 22,
                        "repeat_sequence_positions": [15, 22],
                        "repeat_source_sequence_positions": [1, 8],
                        "minimum_intervening_presentations": 13,
                        "one_distinct_reviewer_required": True,
                    }
                )
    return presentations, assignment_blocks


def build_protocol(
    *,
    repo_root: Path,
    task_dossier_path: Path,
    executor_plan_path: Path,
    protocol_schema_path: Path,
    ballot_schema_path: Path,
) -> dict[str, Any]:
    dossier = _verified_dossier(task_dossier_path)
    executor_plan = _verified_executor_plan(executor_plan_path, repo_root=repo_root)
    task_wave_binding = _reasoning_task_wave_binding(executor_plan)
    selection_reference = executor_plan["source_artifacts"]["task_selection"]
    selection_path = repo_root / selection_reference["path"]
    selection_document = _read_object(selection_path)
    selection_payload = {
        key: value for key, value in selection_document.items() if key != "artifact_sha256"
    }
    if (
        selection_document.get("artifact_sha256") != TASK_SELECTION_ARTIFACT_SHA256
        or _sha256(selection_payload) != TASK_SELECTION_ARTIFACT_SHA256
        or _file_sha256(selection_path) != selection_reference["file_sha256"]
    ):
        raise HumanProtocolError("executor task-selection artifact differs")
    selected_tasks = _selected_tasks(dossier, executor_plan)
    arms, cells = _build_graph(selected_tasks, executor_plan)
    presentations, assignment_blocks = _build_presentations(cells)
    task_counts = Counter(row["family"] for row in selected_tasks)
    cell_counts = Counter(row["analysis_role"] for row in cells)
    presentation_counts = Counter((row["cohort"], bool(row["is_repeat"])) for row in presentations)
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_role": "pre_generation_blinded_human_evaluation_allocation",
        "status": "frozen_planning_only_no_generations_no_judgments",
        "study_id": "reasoning-effort-sensitivity-successor-24-task-v2",
        "reasoning_task_wave_binding": task_wave_binding,
        "supersedes": {
            "artifact_sha256": ("42fb1b5ea606034d4eb62eb813c957b87ffee44392e1c8f11322bf61fe7002ea"),
            "reason": (
                "the predecessor was bound to the retired executor plan whose first "
                "block stopped before generation on a benchmark-pipeline parser defect"
            ),
        },
        "study_scope": {
            "benchmark": "FlavourBench",
            "track": "reasoning_effort_sensitivity",
            "development_only": True,
            "rank_eligible": False,
            "confirmatory": False,
            "official_leaderboard_admission": False,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
            "provider_calls_made_by_protocol_freeze": 0,
            "mcp_calls_made_by_protocol_freeze": 0,
            "human_records_created_by_protocol_freeze": 0,
        },
        "source_bindings": {
            "task_dossier": {
                "path": _relative(repo_root, task_dossier_path),
                "semantic_sha256": TASK_DOSSIER_SHA256,
                "file_sha256": _file_sha256(task_dossier_path),
            },
            "executor_study_plan": {
                "path": _relative(repo_root, executor_plan_path),
                "semantic_sha256": executor_plan["artifact_sha256"],
                "file_sha256": _file_sha256(executor_plan_path),
                "schema_version": executor_plan["schema_version"],
            },
            "executor_task_selection": dict(executor_plan["source_artifacts"]["task_selection"]),
            "task_quarantine": {
                "semantic_sha256": QUARANTINE_SHA256,
                "quarantined_task_ids": sorted(
                    row["task_id"] for row in build_quarantine_artifact()["records"]
                ),
            },
            "protocol_schema": {
                "path": _relative(repo_root, protocol_schema_path),
                "file_sha256": _file_sha256(protocol_schema_path),
            },
            "ballot_schema": {
                "path": _relative(repo_root, ballot_schema_path),
                "file_sha256": _file_sha256(ballot_schema_path),
                "schema_version": BALLOT_SCHEMA_VERSION,
            },
        },
        "task_selection": {
            "selection_seed": selection_document["selection_seed"],
            "executor_task_selection_artifact_sha256": (TASK_SELECTION_ARTIFACT_SHA256),
            "selection_is_outcome_blind": True,
            "source_quality_outcomes_used": 0,
            "source_candidates_per_family": 10,
            "selected_per_family": 6,
            "excluded_current_quarantine": True,
            "excluded_specialist_scope_and_surface_dependency_holds": True,
            "accepted_source_licenses": ["CC BY-SA 3.0", "CC BY-SA 4.0"],
            "task_count_by_family": dict(task_counts),
            "selected_task_set_sha256": SELECTED_TASK_SET_SHA256,
            "wave_order_sha256": WAVE_ORDER_SHA256,
            "unique_atomic_task_waves": 24,
            "generation_pairs_per_wave": 7,
            "generation_response_arms_per_wave": 14,
        },
        "tasks": selected_tasks,
        "endpoints": list(ENDPOINTS),
        "variant_contracts": VARIANTS,
        "epicure_conditions": list(CONDITIONS),
        "comparison_contracts": list(COMPARISONS),
        "arm_coordinates": arms,
        "comparison_cells": cells,
        "presentation_allocation": {
            "randomization_nonce": RANDOMIZATION_NONCE,
            "base_side_assignment": "exact_executor_presentation_orientation",
            "replication_side_schedule": (
                "slot_one_executor_orientation_slot_two_mirrored_slot_three_"
                "expert_executor_and_public_mirrored"
            ),
            "ordering_nonce": ORDER_NONCE,
            "review_delivery_must_not_expose_sealed_fields": [
                "endpoint_id",
                "canonical_model_slug",
                "provider_endpoint",
                "condition",
                "variant",
                "sealed_orientation",
                "repeat_source_presentation_id",
            ],
            "identity_reveal": "only_after_the_ballot_is_sealed",
            "allocation_release": "after_all_planned_primary_ballots_are_sealed_or_withdrawn",
            "cohorts": list(COHORTS),
            "cohorts_are_analyzed_separately": True,
            "cross_cohort_pooling_permitted": False,
            "reviewer_cohort_contract": {
                "culinary_expert": {
                    "minimum_evidence": (
                        "documented_culinary_credential_or_five_years_relevant_practice"
                    ),
                    "qualified_families_required": list(FAMILIES),
                    "author_or_study_affiliation_recorded_privately": True,
                    "affiliated_reviewer_sensitivity_reported_separately": True,
                },
                "general_public": {
                    "minimum_age": 18,
                    "working_language_comprehension_required": True,
                    "culinary_credential_required": False,
                },
                "one_person_one_cohort": True,
                "reviewer_identity_in_research_export": "season_specific_hmac_only",
                "raw_identity_in_research_export": False,
                "study_authors_cannot_count_as_independent_experts": True,
            },
            "target_independent_judgments_per_cell_per_cohort": 3,
            "one_assignment_block_per_reviewer_maximum": True,
            "same_reviewer_may_not_fill_multiple_replication_slots": True,
            "base_blocks": 12,
            "assignment_blocks": assignment_blocks,
            "repeat_policy": {
                "type": "exact_position_swapped_repeat",
                "repeats_per_assignment_block": 2,
                "repeat_ballots_enter_effect_estimates": False,
                "minimum_intervening_presentations": 13,
                "consistency_outputs": [
                    "preference_agreement",
                    "position_corrected_cohen_kappa",
                    "rubric_mean_absolute_difference",
                ],
                "reviewer_exclusion_is_not_automatic": True,
                "exclusion_requires_predeclared_quality_review_without_outcome_access": True,
            },
        },
        "presentations": presentations,
        "ballot_contract": {
            "pre_answer_task_validity_must_be_sealed": True,
            "task_validity_decisions": ["valid", "revise", "exclude"],
            "preference_choices": ["left", "right", "tie", "both_bad"],
            "rubric_dimensions": list(RUBRIC_DIMENSIONS),
            "rubric_scale": {
                "minimum": 1,
                "maximum": 5,
                "higher_is_better": True,
                "null_allowed_only_if_schema_marks_not_applicable": True,
            },
            "confidence_scale": {"minimum": 1, "maximum": 5},
            "comparative_rationale_required": True,
            "failure_tags_are_arm_specific": True,
            "identity_leakage_flag_required": True,
            "complete_ballots_are_append_only": True,
            "corrections_require_superseding_ballots": True,
        },
        "failure_and_missingness": {
            "technical_arm_failure": (
                "do_not_deliver_the_pair_do_not_accept_a_preference_retain_reliability_event"
            ),
            "provider_or_mcp_retry_after_ambiguous_success": False,
            "replacement_generation_after_output_inspection": False,
            "invalid_task_decision": "no_preference_requested_ballot_is_scheduled_missing",
            "incomplete_ballot": "exclude_without_imputation_and_count_by_reason",
            "both_bad": "exclude_from_preference_point_estimate_and_count_as_observed_failure",
            "repeat_ballot": "quality_control_only",
            "required_missingness_table_dimensions": [
                "endpoint_id",
                "condition",
                "variant",
                "family",
                "cohort",
                "missingness_reason",
            ],
            "missingness_reasons": [
                "lower_arm_generation_failed",
                "upper_arm_generation_failed",
                "identity_or_route_mismatch",
                "task_marked_revise_or_exclude_before_answer_reveal",
                "reviewer_withdrew",
                "ballot_incomplete",
                "both_bad_no_directional_preference",
                "administrative_not_assigned",
            ],
        },
        "analysis_plan": {
            "unit_of_randomization": "presentation_within_frozen_cell_and_reviewer_slot",
            "primary_estimands": (
                "endpoint_specific_high_vs_low_preference_by_epicure_condition_and_cohort"
            ),
            "secondary_estimands": (
                "gemini_default_medium_vs_low_and_high_vs_default_by_condition_and_cohort"
            ),
            "preference_points_for_upper_variant": {
                "upper_variant_win": 1.0,
                "tie": 0.5,
                "lower_variant_win": 0.0,
                "both_bad": None,
            },
            "complete_case_rule": (
                "a_task_cell_contributes_only_when_both_real_arms_pass_all_execution_gates_"
                "and_all_three_distinct_reviewers_in_the_cohort_submit_valid_directional_or_tie_ballots"
            ),
            "complete_case_weighting": "equal_family_then_equal_task_then_equal_ballot",
            "worst_best_bounds": {
                "denominator": "all_three_frozen_reviewer_slots_for_all_24_tasks",
                "worst_case_upper_variant_points_for_each_missing_slot": 0.0,
                "best_case_upper_variant_points_for_each_missing_slot": 1.0,
                "both_bad_is_bounded_as_missing_direction": True,
                "bounds_reported_even_when_complete_case_estimate_is_suppressed": True,
            },
            "task_cluster_inference": {
                "primary_cluster": "task_id",
                "family_stratified": True,
                "resamples": 20000,
                "seed_sha256": hashlib.sha256(BOOTSTRAP_NONCE.encode()).hexdigest(),
                "resampling_rule": "sample_six_task_clusters_with_replacement_within_each_family",
                "all_ballots_and_reused_response_arms_for_sampled_task_move_together": True,
                "confidence_interval": "two_sided_95_percent_percentile",
                "minimum_complete_task_clusters_per_family": 4,
                "rater_cluster_sensitivity": "crossed_task_by_reviewer_pigeonhole_bootstrap",
                "ballots_treated_as_independent_rows": False,
            },
            "multiplicity": {
                "primary_family": "six_endpoint_by_condition_tests_within_each_cohort",
                "primary_adjustment": "holm_two_sided",
                "gemini_secondary_family": (
                    "four_ordered_contrast_by_condition_tests_within_each_cohort"
                ),
                "secondary_adjustment": "holm_two_sided",
            },
            "rubrics": "report_each_dimension_separately_no_opaque_composite",
            "cohort_pooling": False,
            "cross_endpoint_pooling": False,
            "repeats_enter_effect_estimates": False,
        },
        "counts": {
            "tasks": len(selected_tasks),
            "tasks_per_family": dict(task_counts),
            "arm_coordinates": len(arms),
            "comparison_cells": len(cells),
            "primary_comparison_cells": cell_counts["primary"],
            "secondary_comparison_cells": cell_counts["secondary"],
            "assignment_blocks": len(assignment_blocks),
            "original_presentations": sum(not row["is_repeat"] for row in presentations),
            "position_swapped_repeats": sum(row["is_repeat"] for row in presentations),
            "presentations_total": len(presentations),
            "original_presentations_by_cohort": {
                cohort: presentation_counts[(cohort, False)] for cohort in COHORTS
            },
            "repeats_by_cohort": {
                cohort: presentation_counts[(cohort, True)] for cohort in COHORTS
            },
            "provider_calls": 0,
            "mcp_calls": 0,
            "human_records": 0,
        },
        "claim_boundary": {
            "supports_execution": False,
            "supports_model_quality_claims": False,
            "supports_reasoning_effort_effect_claims": False,
            "supports_epicure_uplift_claims": False,
            "supports_official_ranking": False,
            "precommits_human_evaluation_before_generation": True,
            "must_remain_labelled_development_only": True,
        },
    }
    return {**protocol, "artifact_sha256": _sha256(protocol)}


def verify_protocol(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise HumanProtocolError("unexpected protocol schema version")
    digest = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if len(digest) != 64 or _sha256(payload) != digest:
        raise HumanProtocolError("protocol content address does not verify")
    counts = document.get("counts") or {}
    expected = {
        "tasks": 24,
        "arm_coordinates": 336,
        "comparison_cells": 240,
        "primary_comparison_cells": 144,
        "secondary_comparison_cells": 96,
        "assignment_blocks": 72,
        "original_presentations": 1440,
        "position_swapped_repeats": 144,
        "presentations_total": 1584,
        "provider_calls": 0,
        "mcp_calls": 0,
        "human_records": 0,
    }
    if any(counts.get(key) != value for key, value in expected.items()):
        raise HumanProtocolError("protocol count contract differs")
    tasks = document.get("tasks") or []
    if Counter(row.get("family") for row in tasks) != Counter({family: 6 for family in FAMILIES}):
        raise HumanProtocolError("protocol is not balanced at six tasks per family")
    if (
        len({row.get("task_wave_id") for row in tasks}) != 24
        or any(row.get("generation_pairs") != 7 for row in tasks)
        or any(row.get("generation_response_arms") != 14 for row in tasks)
    ):
        raise HumanProtocolError("protocol does not bind 24 atomic seven-pair waves")
    quarantined = set(
        document.get("source_bindings", {})
        .get("task_quarantine", {})
        .get("quarantined_task_ids", [])
    )
    if quarantined & {str(row.get("task_id")) for row in tasks}:
        raise HumanProtocolError("protocol admits a quarantined task")
    if any(row.get("confirmatory_eligible") is not False for row in tasks):
        raise HumanProtocolError("protocol promotes an unvalidated task")
    cells = document.get("comparison_cells") or []
    if len({row.get("cell_id") for row in cells}) != 240:
        raise HumanProtocolError("comparison cell IDs are not unique")
    presentations = document.get("presentations") or []
    if len({row.get("presentation_id") for row in presentations}) != 1584:
        raise HumanProtocolError("presentation IDs are not unique")
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in presentations:
        if not row.get("is_repeat"):
            by_cell[str(row.get("cell_id"))].append(row)
    for rows in by_cell.values():
        if len(rows) != 6:
            raise HumanProtocolError("each comparison cell must have six original slots")
        left_lower = Counter(row.get("sealed_orientation") for row in rows)
        if left_lower != Counter({"lower_left": 3, "upper_left": 3}):
            raise HumanProtocolError("each cell must balance the two sides three-to-three")


def verify_cross_artifact(document: Mapping[str, Any], executor_plan: Mapping[str, Any]) -> None:
    """Recompute the human graph from the executor plan instead of trusting claims."""

    verify_protocol(document)
    binding = _reasoning_task_wave_binding(executor_plan)
    if document.get("reasoning_task_wave_binding") != binding:
        raise HumanProtocolError("human protocol binds a different task-wave study")
    source_plan = document.get("source_bindings", {}).get("executor_study_plan", {})
    if source_plan.get("semantic_sha256") != executor_plan.get("artifact_sha256"):
        raise HumanProtocolError("human protocol source binding has another executor plan")

    tasks = document.get("tasks") or []
    task_rows = [
        (
            row.get("task_wave_ordinal"),
            row.get("task_wave_id"),
            row.get("task_id"),
            row.get("family"),
            row.get("prompt_sha256"),
        )
        for row in tasks
    ]
    expected_task_rows = [
        (
            wave.get("wave_ordinal"),
            wave.get("wave_id"),
            wave.get("task_id"),
            wave.get("task_family"),
            wave.get("prompt_sha256"),
        )
        for wave in executor_plan.get("task_waves") or []
    ]
    if task_rows != expected_task_rows:
        raise HumanProtocolError("human task rows differ from executor wave order")

    plan_work_items = executor_plan.get("work_items") or []
    plan_work_ids = {str(row.get("work_item_id") or "") for row in plan_work_items}
    plan_arm_ids = {str(arm_id) for row in plan_work_items for arm_id in row.get("arm_ids") or []}
    arms = document.get("arm_coordinates") or []
    if {str(row.get("executor_work_item_id") or "") for row in arms} != plan_work_ids or {
        str(row.get("executor_arm_id") or "") for row in arms
    } != plan_arm_ids:
        raise HumanProtocolError("human arm coordinates differ from executor work items")
    by_wave: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for arm in arms:
        by_wave[str(arm.get("task_wave_id") or "")].append(arm)
        expected_arm_id = _arm_id(
            executor_arm_id=str(arm.get("executor_arm_id") or ""),
            work_item_id=str(arm.get("executor_work_item_id") or ""),
        )
        if arm.get("arm_coordinate_id") != expected_arm_id:
            raise HumanProtocolError("a human arm coordinate was not recomputed correctly")
    if any(
        len(rows) != 14 or len({str(row.get("executor_work_item_id") or "") for row in rows}) != 7
        for rows in by_wave.values()
    ):
        raise HumanProtocolError("a human task wave is not fourteen arms in seven pairs")

    arm_lookup = {
        (
            row["task_id"],
            row["endpoint_id"],
            row["variant"],
            row["condition"],
        ): row
        for row in arms
    }
    cells = {str(row["cell_id"]): row for row in document.get("comparison_cells") or []}
    plan_presentations = executor_plan.get("human_evaluation", {}).get("presentations") or []
    if set(cells) != {str(row["presentation_id"]) for row in plan_presentations}:
        raise HumanProtocolError("human comparison cell IDs differ from executor graph")
    for planned in plan_presentations:
        cell = cells[str(planned["presentation_id"])]
        coordinate = (
            planned["task_id"],
            planned["endpoint_id"],
            planned["first_variant"],
            planned["condition"],
        )
        upper_coordinate = (
            planned["task_id"],
            planned["endpoint_id"],
            planned["second_variant"],
            planned["condition"],
        )
        lower_arm = arm_lookup[coordinate]
        upper_arm = arm_lookup[upper_coordinate]
        lower_left = planned["left_work_item_id"] == lower_arm["executor_work_item_id"]
        expected_orientation = "lower_left" if lower_left else "upper_left"
        if (
            cell.get("lower_arm_coordinate_id") != lower_arm["arm_coordinate_id"]
            or cell.get("upper_arm_coordinate_id") != upper_arm["arm_coordinate_id"]
            or cell.get("executor_base_orientation") != expected_orientation
            or cell.get("task_wave_id") != planned.get("wave_id")
        ):
            raise HumanProtocolError("human comparison cell differs from executor presentation")


def write_protocol(document: Mapping[str, Any], output_dir: Path) -> Path:
    verify_protocol(document)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"reasoning-effort-human-protocol-{document['artifact_sha256']}.json"
    rendered = _canonical(document) + b"\n"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise HumanProtocolError("content-addressed output conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(".."))
    parser.add_argument("--task-dossier", type=Path, required=True)
    parser.add_argument("--executor-plan", type=Path, required=True)
    parser.add_argument("--protocol-schema", type=Path, required=True)
    parser.add_argument("--ballot-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    executor_plan_path = arguments.executor_plan.resolve()
    executor_plan = _verified_executor_plan(executor_plan_path, repo_root=repo_root)
    document = build_protocol(
        repo_root=repo_root,
        task_dossier_path=arguments.task_dossier.resolve(),
        executor_plan_path=executor_plan_path,
        protocol_schema_path=arguments.protocol_schema.resolve(),
        ballot_schema_path=arguments.ballot_schema.resolve(),
    )
    verify_cross_artifact(document, executor_plan)
    path = write_protocol(document, arguments.output_dir.resolve())
    print(
        json.dumps(
            {
                "artifact_sha256": document["artifact_sha256"],
                "status": document["status"],
                "counts": document["counts"],
                "output": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
