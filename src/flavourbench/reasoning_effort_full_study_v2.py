"""Freeze the import-safe successor to the reasoning-effort task-wave study.

The predecessor remains immutable.  This module reuses its real tasks,
manifests, endpoint contracts, and forecasts, but assigns a new study identity,
execution root, namespace, and every execution-facing identifier.  It binds the
zero-call import incident and recovery receipt that retired the predecessor.
"""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_full_study_v1 as predecessor
from . import reasoning_effort_human_protocol as human
from .reasoning_effort_source_closure_v2 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)

PLAN_SCHEMA = "flavourbench-reasoning-effort-task-wave-plan-v3"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-task-wave-preflight-v3"
BOUND_PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-bound-admission-preflight-v3"
HUMAN_PROTOCOL_SCHEMA = "flavourbench-reasoning-effort-human-evaluation-protocol-v2"

FREEZE_NONCE = "reasoning-effort-task-waves-v4-import-safe-2026-08-04"
NAMESPACE = uuid.UUID("84fd10c7-f3af-49a9-ae72-b58f467275a4")
STUDY_ID = "frontier-reasoning-effort-task-waves-v4-import-safe"
ROOT_ID = "reasoning-effort-task-waves-v4-import-safe"
# Artifact schemas advance from predecessor v2 to v3.  The execution campaign
# itself is the fourth task-wave identity, so its human confirmation is V4 and
# cannot authorize any v3-root command.
CONFIRMATION = "RUN_REASONING_EFFORT_V4_ONE_COMPLETE_FAMILY_BLOCK"

PREDECESSOR_PLAN_SHA256 = (
    "99b8f70ae81aa3a7b7e79a45bb4253cb58d26306f90ab5b9c4f09a6938f1a301"
)
EARLIER_RETIRED_PLAN_SHA256 = (
    "03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c"
)
INCIDENT_SHA256 = "2385d025f33b3286ba48f36e7e493be49ce5a55a07ee94d88f757b130ae88ea3"
RECOVERY_RECEIPT_SHA256 = (
    "ab6d988c1d63473163bb4a0ec821923e3703778f93af0c3ee1fab1a12d258eeb"
)
PREDECESSOR_HUMAN_SHA256 = (
    "cd2a234f617158304a5eb4efed1c6e34198cd857f2de124b10dee09fdec370a8"
)

PREDECESSOR_PLAN_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-task-waves-v3/"
    f"plan/reasoning-effort-task-wave-plan-v2-{PREDECESSOR_PLAN_SHA256}.json"
)
EARLIER_RETIRED_PLAN_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/"
    f"plan/reasoning-effort-task-wave-plan-v2-{EARLIER_RETIRED_PLAN_SHA256}.json"
)
INCIDENT_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-task-waves-v3/"
    "import-incident-recovery-v1/reasoning-effort-import-pipeline-incident-"
    f"{INCIDENT_SHA256}.json"
)
RECOVERY_RECEIPT_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-task-waves-v3/"
    "import-incident-recovery-v1/reasoning-effort-import-recovery-receipt-"
    f"{RECOVERY_RECEIPT_SHA256}.json"
)
PREDECESSOR_HUMAN_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v2/"
    f"reasoning-effort-human-protocol-{PREDECESSOR_HUMAN_SHA256}.json"
)

ENDPOINTS = predecessor.ENDPOINTS
TASK_FAMILIES = predecessor.TASK_FAMILIES
CURRENT_EXPOSURE_USD = predecessor.CURRENT_EXPOSURE_USD
ADMISSION_CEILING_USD = predecessor.ADMISSION_CEILING_USD
HARD_CAP_USD = predecessor.HARD_CAP_USD


class FullStudyError(RuntimeError):
    """A successor identity, source, protocol, or preflight predicate failed."""


_sha256 = predecessor._sha256
_file_sha256 = predecessor._file_sha256
_regular_json = predecessor._regular_json
_decimal_text = predecessor._decimal_text
_relative = predecessor._relative
_file_ref = predecessor._file_ref
_write_artifact = predecessor._write_artifact


def _verified_artifact(repo_root: Path, relative: str, semantic_sha256: str) -> dict[str, Any]:
    path = repo_root / relative
    document = _regular_json(path)
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if document.get("artifact_sha256") != semantic_sha256 or _sha256(body) != semantic_sha256:
        raise FullStudyError(f"supersession artifact differs: {relative}")
    return document


def _attempt_slots(run_id: str, route_cell_id: str) -> list[dict[str, Any]]:
    coordinates: list[tuple[str, str, int]] = []
    off = f"{run_id}:epicure_off"
    on = f"{run_id}:epicure_on"
    for phase in ("planning", "evidence_decision", "final"):
        coordinates.extend((off, phase, attempt) for attempt in (0, 1))
    for phase in ("planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"):
        coordinates.extend((on, phase, attempt) for attempt in (0, 1))
    coordinates.append((on, "mcp_session", 0))
    for round_index in range(3):
        for call_index in range(13):
            coordinates.append((on, f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt,
            "attempt_id": str(
                uuid.uuid5(
                    NAMESPACE,
                    f"{FREEZE_NONCE}:{route_cell_id}:{arm_id}:{phase}:{attempt}",
                )
            ),
        }
        for arm_id, phase, attempt in coordinates
    ]


def _identity_sets(plan: Mapping[str, Any]) -> dict[str, set[str]]:
    items = plan.get("work_items") or []
    return {
        "work_item_ids": {str(item["work_item_id"]) for item in items},
        "run_ids": {str(item["run_id"]) for item in items},
        "arm_ids": {str(arm) for item in items for arm in item.get("arm_ids") or []},
        "attempt_ids": {
            str(slot["attempt_id"])
            for item in items
            for slot in item.get("attempt_slots") or []
        },
        "wave_ids": {str(wave["wave_id"]) for wave in plan.get("task_waves") or []},
        "block_ids": {
            str(block["admission_block_id"]) for block in plan.get("admission_blocks") or []
        },
        "presentation_ids": {
            str(row["presentation_id"])
            for row in (plan.get("human_evaluation") or {}).get("presentations") or []
        },
    }


def _human_presentations(
    *, waves: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    item_map = {
        (
            item["route_coordinate"]["endpoint_id"],
            item["route_coordinate"]["task_id"],
            item["route_coordinate"]["variant_id"],
        ): item
        for item in items
    }
    contrasts = {
        "sonnet": (("explicit_low", "explicit_high", "primary_low_high"),),
        "gemini": (
            ("explicit_low", "explicit_high", "primary_low_high"),
            ("explicit_low", "provider_default", "secondary_low_default"),
            ("provider_default", "explicit_high", "secondary_default_high"),
        ),
        "deepseek": (("explicit_low", "explicit_high", "primary_low_high"),),
    }
    records: list[dict[str, Any]] = []
    for wave in waves:
        task_id = str(wave["task_id"])
        for endpoint_id, endpoint_contrasts in contrasts.items():
            for first, second, contrast in endpoint_contrasts:
                for condition in ("epicure_off", "epicure_on"):
                    coordinate = {
                        "schema_version": "flavourbench-reasoning-presentation-v3",
                        "freeze_nonce": FREEZE_NONCE,
                        "wave_id": wave["wave_id"],
                        "task_id": task_id,
                        "task_family": wave["task_family"],
                        "endpoint_id": endpoint_id,
                        "condition": condition,
                        "contrast": contrast,
                        "first_variant": first,
                        "second_variant": second,
                    }
                    records.append(
                        {
                            **coordinate,
                            "presentation_id": _sha256(coordinate),
                            "first_work_item_id": item_map[(endpoint_id, task_id, first)][
                                "work_item_id"
                            ],
                            "second_work_item_id": item_map[(endpoint_id, task_id, second)][
                                "work_item_id"
                            ],
                        }
                    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["endpoint_id"]),
                str(record["task_family"]),
                str(record["condition"]),
                str(record["contrast"]),
            )
        ].append(record)
    final: list[dict[str, Any]] = []
    for values in grouped.values():
        values.sort(
            key=lambda value: _sha256(
                {"side": FREEZE_NONCE, "presentation_id": value["presentation_id"]}
            )
        )
        if len(values) != 6:
            raise FullStudyError("successor presentation stratum is not six tasks")
        for index, value in enumerate(values):
            first_left = index < 3
            final.append(
                {
                    **value,
                    "left_work_item_id": value[
                        "first_work_item_id" if first_left else "second_work_item_id"
                    ],
                    "right_work_item_id": value[
                        "second_work_item_id" if first_left else "first_work_item_id"
                    ],
                    "first_variant_on_left": first_left,
                }
            )
    return sorted(final, key=lambda value: str(value["presentation_id"]))


def build_plan(*, repo_root: Path) -> dict[str, Any]:
    """Derive the successor plan without writing or contacting any service."""

    predecessor_plan = _verified_artifact(
        repo_root, PREDECESSOR_PLAN_PATH, PREDECESSOR_PLAN_SHA256
    )
    predecessor.validate_plan(predecessor_plan, repo_root=repo_root)
    _verified_artifact(
        repo_root, EARLIER_RETIRED_PLAN_PATH, EARLIER_RETIRED_PLAN_SHA256
    )
    incident = _verified_artifact(repo_root, INCIDENT_PATH, INCIDENT_SHA256)
    recovery = _verified_artifact(
        repo_root, RECOVERY_RECEIPT_PATH, RECOVERY_RECEIPT_SHA256
    )
    if (
        incident.get("study_plan_sha256") != PREDECESSOR_PLAN_SHA256
        or incident.get("impact", {}).get("provider_completion_requests") != 0
        or incident.get("impact", {}).get("epicure_calls") != 0
        or recovery.get("study_plan_sha256") != PREDECESSOR_PLAN_SHA256
        or recovery.get("reservation_released") is not True
        or recovery.get("replay_permitted") is not False
        or recovery.get("provider_completion_requests") != 0
        or recovery.get("epicure_calls") != 0
    ):
        raise FullStudyError("import incident or zero-call recovery binding differs")

    old_by_coordinate = {
        (
            str(item["route_coordinate"]["task_id"]),
            str(item["route_coordinate"]["endpoint_id"]),
            str(item["route_coordinate"]["variant_id"]),
        ): item
        for item in predecessor_plan["work_items"]
    }
    if len(old_by_coordinate) != 168:
        raise FullStudyError("predecessor route coordinates are not unique")

    work_items: list[dict[str, Any]] = []
    new_by_coordinate: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, old in old_by_coordinate.items():
        coordinate = {
            **copy.deepcopy(dict(old["route_coordinate"])),
            "schema_version": "flavourbench-reasoning-effort-task-wave-coordinate-v3",
            "freeze_nonce": FREEZE_NONCE,
            "superseded_plan_sha256": PREDECESSOR_PLAN_SHA256,
            "supersession_incident_sha256": INCIDENT_SHA256,
            "supersession_recovery_receipt_sha256": RECOVERY_RECEIPT_SHA256,
        }
        route_cell_id = _sha256(coordinate)
        work_item_id = _sha256(
            {
                "route_cell_id": route_cell_id,
                "role": "reasoning-effort-task-wave-v3-import-safe",
                "namespace": str(NAMESPACE),
            }
        )
        run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
        item = {
            **copy.deepcopy(dict(old)),
            "route_coordinate": coordinate,
            "route_cell_id": route_cell_id,
            "work_item_id": work_item_id,
            "run_id": run_id,
            "arm_ids": [f"{run_id}:epicure_off", f"{run_id}:epicure_on"],
            "attempt_slots": _attempt_slots(run_id, route_cell_id),
            "supersedes_identifiers": {
                "work_item_id": old["work_item_id"],
                "route_cell_id": old["route_cell_id"],
                "run_id": old["run_id"],
            },
            "same_identifier_replay_permitted": False,
        }
        work_items.append(item)
        new_by_coordinate[key] = item
    work_items.sort(key=lambda item: str(item["work_item_id"]))

    waves: list[dict[str, Any]] = []
    for old_wave in predecessor_plan["task_waves"]:
        task_id = str(old_wave["task_id"])
        members = [item for key, item in new_by_coordinate.items() if key[0] == task_id]
        members.sort(
            key=lambda item: _sha256(
                {
                    "wave": FREEZE_NONCE,
                    "task_id": task_id,
                    "work_item_id": item["work_item_id"],
                }
            )
        )
        coordinate = {
            "schema_version": "flavourbench-reasoning-effort-task-wave-v3",
            "freeze_nonce": FREEZE_NONCE,
            "wave_ordinal": old_wave["wave_ordinal"],
            "task_id": task_id,
            "task_family": old_wave["task_family"],
            "prompt_sha256": old_wave["prompt_sha256"],
            "work_item_ids": [item["work_item_id"] for item in members],
        }
        waves.append(
            {
                **coordinate,
                "wave_id": _sha256(coordinate),
                "worst_case_reserve_usd": old_wave["worst_case_reserve_usd"],
                "response_arms": 14,
                "matched_pairs": 7,
                "individually_admissible": False,
                "admitted_only_as_member_of_family_block": True,
                "superseded_wave_id": old_wave["wave_id"],
            }
        )

    blocks: list[dict[str, Any]] = []
    for block_index in range(6):
        block_waves = waves[block_index * 4 : (block_index + 1) * 4]
        coordinate = {
            "schema_version": "flavourbench-reasoning-effort-family-block-v3",
            "freeze_nonce": FREEZE_NONCE,
            "block_ordinal": block_index + 1,
            "wave_ids": [wave["wave_id"] for wave in block_waves],
            "task_ids": [wave["task_id"] for wave in block_waves],
            "work_item_ids": [
                item_id for wave in block_waves for item_id in wave["work_item_ids"]
            ],
        }
        blocks.append(
            {
                **coordinate,
                "admission_block_id": _sha256(coordinate),
                "task_families": [wave["task_family"] for wave in block_waves],
                "worst_case_reserve_usd": _decimal_text(
                    sum(Decimal(wave["worst_case_reserve_usd"]) for wave in block_waves)
                ),
                "tasks": 4,
                "matched_pairs": 28,
                "response_arms": 56,
                "atomic_admission": True,
                "partial_block_start_permitted": False,
                "superseded_block_id": predecessor_plan["admission_blocks"][block_index][
                    "admission_block_id"
                ],
            }
        )

    source_artifacts = copy.deepcopy(dict(predecessor_plan["source_artifacts"]))
    source_artifacts.update(
        {
            "superseded_plan": _file_ref(repo_root, repo_root / PREDECESSOR_PLAN_PATH),
            "earlier_retired_plan": _file_ref(
                repo_root, repo_root / EARLIER_RETIRED_PLAN_PATH
            ),
            "import_pipeline_incident": _file_ref(repo_root, repo_root / INCIDENT_PATH),
            "import_recovery_receipt": _file_ref(
                repo_root, repo_root / RECOVERY_RECEIPT_PATH
            ),
        }
    )
    source_closure = build_source_closure(repo_root=repo_root)
    presentations = _human_presentations(waves=waves, items=work_items)
    repeat_ids = [
        value["presentation_id"]
        for value in sorted(
            presentations,
            key=lambda value: _sha256(
                {"repeat": FREEZE_NONCE, "presentation_id": value["presentation_id"]}
            ),
        )[:24]
    ]
    task_identity = predecessor.canonical_task_wave_identity(
        tasks=predecessor_plan["tasks"], waves=waves
    )
    plan = {
        **copy.deepcopy(dict(predecessor_plan)),
        "schema_version": PLAN_SCHEMA,
        "record_role": "import_safe_route_qualified_randomized_task_wave_sensitivity",
        "status": "frozen_not_executed_successor_roots_empty",
        "study_id": STUDY_ID,
        "freeze_nonce": FREEZE_NONCE,
        "root_id": ROOT_ID,
        "supersedes": {
            "plans": [PREDECESSOR_PLAN_SHA256, EARLIER_RETIRED_PLAN_SHA256],
            "immediate_predecessor_plan_sha256": PREDECESSOR_PLAN_SHA256,
            "pipeline_incident_sha256": INCIDENT_SHA256,
            "zero_call_recovery_receipt_sha256": RECOVERY_RECEIPT_SHA256,
            "reason": (
                "the immediate predecessor imported a nonexistent live-smoke symbol after "
                "catalog attestation, reservation, and the first durable item start; the "
                "zero-call recovery terminalized all 28 scheduled pairs and released the reserve"
            ),
        },
        "source_artifacts": source_artifacts,
        "source_code": source_closure,
        "work_items": work_items,
        "task_waves": waves,
        "wave_execution_order": [wave["wave_id"] for wave in waves],
        "admission_blocks": blocks,
        "block_execution_order": [block["admission_block_id"] for block in blocks],
        "task_wave_identity": task_identity,
        "execution_roots": {
            "coordinator": (
                "flavourbench/artifacts/season1/current-quality-run/"
                f"{ROOT_ID}/runs/coordinator"
            ),
            "endpoints": {
                endpoint_id: (
                    "flavourbench/artifacts/season1/current-quality-run/"
                    f"{ROOT_ID}/runs/{endpoint_id}"
                )
                for endpoint_id in ENDPOINTS
            },
        },
        "execution": {
            **copy.deepcopy(dict(predecessor_plan["execution"])),
            "module": "flavourbench.reasoning_effort_full_study_executor_v2",
            "confirmation": CONFIRMATION,
            "all_168_runtime_arguments_validated_before_any_external_or_durable_operation": True,
            "post_start_operation_failure_durably_classified": True,
        },
        "failure_policy": {
            **copy.deepcopy(dict(predecessor_plan["failure_policy"])),
            "pre_external_all_item_runtime_construction_required": True,
            "pipeline_failure_after_reservation_retains_or_terminalizes_reserve": True,
            "pipeline_failure_after_item_start_durably_classified": True,
        },
        "human_evaluation": {
            **copy.deepcopy(dict(predecessor_plan["human_evaluation"])),
            "presentations": presentations,
            "position_swapped_repeat_presentation_ids": repeat_ids,
        },
        "calls_made_by_freeze": {
            "provider_completions": 0,
            "epicure": 0,
            "catalog_gets": 0,
        },
    }
    plan.pop("artifact_sha256", None)
    plan["artifact_sha256"] = _sha256(plan)
    validate_plan(plan, repo_root=repo_root)
    return plan


def validate_plan(plan: Mapping[str, Any], *, repo_root: Path) -> None:
    body = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("artifact_sha256") != _sha256(body):
        raise FullStudyError("successor plan schema or content address failed")
    if (
        plan.get("freeze_nonce") != FREEZE_NONCE
        or plan.get("study_id") != STUDY_ID
        or plan.get("root_id") != ROOT_ID
        or plan.get("execution", {}).get("module")
        != "flavourbench.reasoning_effort_full_study_executor_v2"
    ):
        raise FullStudyError("successor study, root, or executor identity differs")
    try:
        verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    except SourceClosureError as error:
        raise FullStudyError(f"successor source closure does not rederive: {error}") from error
    supersedes = plan.get("supersedes") or {}
    if (
        supersedes.get("plans")
        != [PREDECESSOR_PLAN_SHA256, EARLIER_RETIRED_PLAN_SHA256]
        or supersedes.get("pipeline_incident_sha256") != INCIDENT_SHA256
        or supersedes.get("zero_call_recovery_receipt_sha256") != RECOVERY_RECEIPT_SHA256
    ):
        raise FullStudyError("successor does not bind both retired plans and recovery")
    for key, digest in (
        ("import_pipeline_incident", INCIDENT_SHA256),
        ("import_recovery_receipt", RECOVERY_RECEIPT_SHA256),
    ):
        reference = plan.get("source_artifacts", {}).get(key) or {}
        path = repo_root / str(reference.get("path") or "")
        if (
            reference.get("semantic_sha256") != digest
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != reference.get("bytes")
            or _file_sha256(path) != reference.get("file_sha256")
        ):
            raise FullStudyError(f"successor supersession source differs: {key}")
    items = plan.get("work_items") or []
    waves = plan.get("task_waves") or []
    blocks = plan.get("admission_blocks") or []
    presentations = (plan.get("human_evaluation") or {}).get("presentations") or []
    if (
        len(items) != 168
        or len(waves) != 24
        or len(blocks) != 6
        or len(presentations) != 240
        or len(_identity_sets(plan)["attempt_ids"]) != 168 * 56
        or Counter(task["family"] for task in plan.get("tasks") or [])
        != Counter({family: 6 for family in TASK_FAMILIES})
    ):
        raise FullStudyError("successor study counts differ")
    item_ids = _identity_sets(plan)["work_item_ids"]
    if any(
        len(wave.get("work_item_ids") or []) != 7
        or not set(wave["work_item_ids"]) <= item_ids
        for wave in waves
    ):
        raise FullStudyError("successor wave membership differs")
    if any(
        len(block.get("work_item_ids") or []) != 28
        or Counter(block.get("task_families") or [])
        != Counter({family: 1 for family in TASK_FAMILIES})
        for block in blocks
    ):
        raise FullStudyError("successor block balance differs")
    for relative in plan["execution_roots"]["endpoints"].values():
        if ROOT_ID not in str(relative):
            raise FullStudyError("successor endpoint root identity differs")
    if ROOT_ID not in str(plan["execution_roots"]["coordinator"]):
        raise FullStudyError("successor coordinator root identity differs")
    current_ids = _identity_sets(plan)
    for relative in (PREDECESSOR_PLAN_PATH, EARLIER_RETIRED_PLAN_PATH):
        retired_ids = _identity_sets(_regular_json(repo_root / relative))
        for identity_type in current_ids:
            if current_ids[identity_type] & retired_ids[identity_type]:
                raise FullStudyError(
                    f"successor reuses a retired {identity_type.removesuffix('s')}"
                )
    if plan.get("task_wave_identity") != predecessor.canonical_task_wave_identity(
        tasks=plan.get("tasks") or [], waves=waves
    ):
        raise FullStudyError("successor task-wave identity does not rederive")


def successor_roots(*, plan: Mapping[str, Any], repo_root: Path) -> list[Path]:
    return [
        repo_root / str(plan["execution_roots"]["coordinator"]),
        *(
            repo_root / str(relative)
            for relative in plan["execution_roots"]["endpoints"].values()
        ),
    ]


def assert_successor_roots_empty(*, plan: Mapping[str, Any], repo_root: Path) -> None:
    files = [
        path
        for root in successor_roots(plan=plan, repo_root=repo_root)
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".lock")
    ]
    if files:
        raise FullStudyError("successor execution roots are not empty")


def build_human_protocol(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Rebind the full blinded allocation to the successor identifiers."""

    validate_plan(plan, repo_root=repo_root)
    previous = _verified_artifact(
        repo_root, PREDECESSOR_HUMAN_PATH, PREDECESSOR_HUMAN_SHA256
    )
    dossier_path = repo_root / previous["source_bindings"]["task_dossier"]["path"]
    dossier = human._verified_dossier(dossier_path)
    selected_tasks = human._selected_tasks(dossier, plan)
    arms, cells = human._build_graph(selected_tasks, plan)
    presentations, assignment_blocks = human._build_presentations(cells)
    protocol = {
        **copy.deepcopy(previous),
        "schema_version": HUMAN_PROTOCOL_SCHEMA,
        "study_id": f"{STUDY_ID}-human-v3",
        "supersedes": {
            "artifact_sha256": PREDECESSOR_HUMAN_SHA256,
            "reason": "all executor-facing identifiers were regenerated after import recovery",
            "pipeline_incident_sha256": INCIDENT_SHA256,
            "zero_call_recovery_receipt_sha256": RECOVERY_RECEIPT_SHA256,
        },
        "reasoning_task_wave_binding": {
            "study_plan_sha256": plan["artifact_sha256"],
            "task_selection_artifact_sha256": plan["source_artifacts"]["task_selection"][
                "semantic_sha256"
            ],
            **copy.deepcopy(dict(plan["task_wave_identity"])),
        },
        "tasks": selected_tasks,
        "arm_coordinates": arms,
        "comparison_cells": cells,
        "presentations": presentations,
        "presentation_allocation": {
            **copy.deepcopy(dict(previous["presentation_allocation"])),
            "assignment_blocks": assignment_blocks,
        },
        "source_bindings": {
            **copy.deepcopy(dict(previous["source_bindings"])),
            "executor_study_plan": {
                "path": None,
                "semantic_sha256": plan["artifact_sha256"],
                "file_sha256": None,
                "schema_version": PLAN_SCHEMA,
            },
            "superseded_human_protocol": _file_ref(
                repo_root, repo_root / PREDECESSOR_HUMAN_PATH
            ),
            "import_pipeline_incident": _file_ref(repo_root, repo_root / INCIDENT_PATH),
            "import_recovery_receipt": _file_ref(
                repo_root, repo_root / RECOVERY_RECEIPT_PATH
            ),
        },
        "task_selection": {
            **copy.deepcopy(dict(previous["task_selection"])),
            "wave_order_sha256": plan["task_wave_identity"]["wave_order_sha256"],
        },
    }
    protocol.pop("artifact_sha256", None)
    protocol["artifact_sha256"] = _sha256(protocol)
    verify_human_protocol_binding(plan=plan, human_protocol=protocol)
    return protocol


def verify_human_protocol_binding(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any]
) -> None:
    body = {
        key: value for key, value in human_protocol.items() if key != "artifact_sha256"
    }
    if (
        human_protocol.get("schema_version") != HUMAN_PROTOCOL_SCHEMA
        or human_protocol.get("artifact_sha256") != _sha256(body)
    ):
        raise FullStudyError("successor human protocol content address failed")
    binding = human_protocol.get("reasoning_task_wave_binding") or {}
    if (
        binding.get("study_plan_sha256") != plan["artifact_sha256"]
        or binding.get("selected_task_set_sha256")
        != plan["task_wave_identity"]["selected_task_set_sha256"]
        or binding.get("wave_order_sha256")
        != plan["task_wave_identity"]["wave_order_sha256"]
        or binding.get("ordered_waves") != plan["task_wave_identity"]["ordered_waves"]
    ):
        raise FullStudyError("successor human task-wave binding differs")
    plan_items = {str(item["work_item_id"]): item for item in plan["work_items"]}
    expected_arms = {
        (
            str(arm_id),
            item_id,
            str(item["route_coordinate"]["task_id"]),
            str(item["route_coordinate"]["endpoint_id"]),
            str(arm_id).rsplit(":", 1)[-1],
            str(item["route_coordinate"]["variant_id"]),
        )
        for item_id, item in plan_items.items()
        for arm_id in item["arm_ids"]
    }
    arms = human_protocol.get("arm_coordinates") or []
    observed_arms = {
        (
            str(arm.get("executor_arm_id")),
            str(arm.get("executor_work_item_id")),
            str(arm.get("task_id")),
            str(arm.get("endpoint_id")),
            str(arm.get("condition")),
            str(arm.get("variant")),
        )
        for arm in arms
        if isinstance(arm, Mapping)
    }
    cells = human_protocol.get("comparison_cells") or []
    plan_presentations = {
        str(row["presentation_id"]) for row in plan["human_evaluation"]["presentations"]
    }
    if (
        len(arms) != 336
        or observed_arms != expected_arms
        or len(cells) != 240
        or {str(cell.get("executor_presentation_id")) for cell in cells}
        != plan_presentations
        or len(human_protocol.get("presentations") or []) != 1584
    ):
        raise FullStudyError("successor human arm, cell, or presentation graph differs")


def build_preflight(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    assert_successor_roots_empty(plan=plan, repo_root=repo_root)
    first_reserve = Decimal(plan["admission_blocks"][0]["worst_case_reserve_usd"])
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "zero_call_import_safe_successor_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "decision": "awaiting_cross_bound_human_protocol",
        "checks": {
            "successor_execution_roots_empty": True,
            "all_execution_identifiers_fresh": True,
            "import_incident_bound": INCIDENT_SHA256,
            "zero_call_recovery_receipt_bound": RECOVERY_RECEIPT_SHA256,
            "planned_runtime_argument_constructions": 168,
            "first_family_block_reserve_usd": _decimal_text(first_reserve),
            "first_family_block_projected_usd": _decimal_text(
                CURRENT_EXPOSURE_USD + first_reserve
            ),
            "below_85_percent_admission": CURRENT_EXPOSURE_USD + first_reserve
            <= ADMISSION_CEILING_USD,
            "human_protocol_frozen_and_cross_verified": False,
        },
        "calls_made": {"provider_completions": 0, "epicure": 0, "catalog_gets": 0},
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def build_bound_preflight(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    preliminary = build_preflight(plan=plan, repo_root=repo_root)
    payload = {
        "schema_version": BOUND_PREFLIGHT_SCHEMA,
        "record_role": "cross_bound_zero_call_import_safe_successor_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "decision": "first_family_block_admissible",
        "checks": {
            **dict(preliminary["checks"]),
            "human_protocol_frozen_and_cross_verified": True,
            "human_arm_coordinates_verified": 336,
            "human_comparison_cells_verified": 240,
            "all_168_runtime_arguments_must_validate_before_any_catalog_get": True,
        },
        "calls_made": {"provider_completions": 0, "epicure": 0, "catalog_gets": 0},
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def verify_bound_preflight(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
) -> None:
    verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    body = {
        key: value for key, value in bound_preflight.items() if key != "artifact_sha256"
    }
    if (
        bound_preflight.get("schema_version") != BOUND_PREFLIGHT_SCHEMA
        or bound_preflight.get("artifact_sha256") != _sha256(body)
        or bound_preflight.get("study_plan_sha256") != plan["artifact_sha256"]
        or bound_preflight.get("human_protocol_sha256")
        != human_protocol["artifact_sha256"]
        or bound_preflight.get("decision") != "first_family_block_admissible"
    ):
        raise FullStudyError("successor bound preflight is absent or invalid")


def freeze(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    """Write plan, human protocol, and preflights; never contact a service."""

    plan = build_plan(repo_root=repo_root)
    plan_path = _write_artifact(
        output_dir / "plan", "reasoning-effort-task-wave-plan-v3", plan
    )
    plan = _regular_json(plan_path)
    protocol = build_human_protocol(plan=plan, repo_root=repo_root)
    protocol["source_bindings"]["executor_study_plan"] = {
        "path": _relative(repo_root, plan_path),
        "semantic_sha256": plan["artifact_sha256"],
        "file_sha256": _file_sha256(plan_path),
        "schema_version": PLAN_SCHEMA,
    }
    protocol.pop("artifact_sha256", None)
    protocol["artifact_sha256"] = _sha256(protocol)
    verify_human_protocol_binding(plan=plan, human_protocol=protocol)
    human_path = _write_artifact(
        output_dir / "human-protocol", "reasoning-effort-human-protocol-v3", protocol
    )
    protocol = _regular_json(human_path)
    preflight = build_preflight(plan=plan, repo_root=repo_root)
    preflight_path = _write_artifact(
        output_dir / "preflight", "reasoning-effort-task-wave-preflight-v3", preflight
    )
    bound = build_bound_preflight(
        plan=plan, human_protocol=protocol, repo_root=repo_root
    )
    bound_path = _write_artifact(
        output_dir / "bound-preflight",
        "reasoning-effort-bound-admission-preflight-v3",
        bound,
    )
    return {
        "plan": plan_path,
        "human_protocol": human_path,
        "preflight": preflight_path,
        "bound_preflight": bound_path,
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    result = freeze(
        repo_root=arguments.repo_root.resolve(), output_dir=arguments.output_dir.resolve()
    )
    print(json.dumps({key: str(path) for key, path in result.items()}, indent=2))


if __name__ == "__main__":
    run()
