"""Freeze the crash-safe successor to the V4 reasoning-effort campaign.

V4 is retained byte-for-byte as a retired NO-GO design.  This successor keeps
its estimand and human allocation, but regenerates every execution-facing
identifier and requires canonical per-work-item reservations in the shared
frontier ledger.
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

from . import reasoning_effort_full_study_v2 as v4
from . import reasoning_effort_human_protocol as human
from .reasoning_effort_source_closure_v5 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)

PLAN_SCHEMA = "flavourbench-reasoning-effort-task-wave-plan-v5"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-task-wave-preflight-v5"
BOUND_PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-bound-preflight-v5"
HUMAN_PROTOCOL_SCHEMA = "flavourbench-reasoning-effort-human-evaluation-protocol-v5"
GOVERNANCE_GO_SCHEMA = "flavourbench-reasoning-effort-independent-go-v1"

FREEZE_NONCE = "reasoning-effort-task-waves-v5-shared-ledger-crash-safe-2026-08-04"
NAMESPACE = uuid.UUID("785008d7-06c4-4f11-93a6-83c30ec91d8e")
STUDY_ID = "frontier-reasoning-effort-task-waves-v5-shared-ledger-crash-safe"
ROOT_ID = "reasoning-effort-task-waves-v5-shared-ledger-crash-safe"
CONFIRMATION = "RUN_REASONING_EFFORT_V5_ONE_CRASH_SAFE_FAMILY_BLOCK"

V4_PLAN_SHA256 = "60ea1a21395faf98fd6919b5a6393d55f920aa5f438be4dda086d592738eb131"
V4_HUMAN_SHA256 = "303b96e252b7b696cc2a522f9986d0b0c0ce4e4084c073e07293328e9817f2e7"
V3_INCIDENT_SHA256 = "2385d025f33b3286ba48f36e7e493be49ce5a55a07ee94d88f757b130ae88ea3"
V3_RECOVERY_SHA256 = "ab6d988c1d63473163bb4a0ec821923e3703778f93af0c3ee1fab1a12d258eeb"

V4_PLAN_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v4-import-safe/plan/"
    f"reasoning-effort-task-wave-plan-v3-{V4_PLAN_SHA256}.json"
)
V4_HUMAN_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v4-import-safe/human-protocol/"
    f"reasoning-effort-human-protocol-v3-{V4_HUMAN_SHA256}.json"
)
V3_INCIDENT_PATH = v4.INCIDENT_PATH
V3_RECOVERY_PATH = v4.RECOVERY_RECEIPT_PATH
V3_PLAN_PATH = v4.PREDECESSOR_PLAN_PATH
V3_PLAN_SHA256 = v4.PREDECESSOR_PLAN_SHA256
V2_PLAN_PATH = v4.EARLIER_RETIRED_PLAN_PATH
V2_PLAN_SHA256 = v4.EARLIER_RETIRED_PLAN_SHA256

GLOBAL_LEDGER_PATH = "flavourbench/artifacts/frontier-contract/ledger.jsonl"
GLOBAL_SOURCE_PATH = "flavourbench/artifacts/live-smoke"
GLOBAL_LEDGER_ANCHOR_SEQUENCE = 29
GLOBAL_LEDGER_ANCHOR_HEAD_SHA256 = (
    "9bd3ce01c661c09b5a64ce863982428ac21edb6c09ca09b21d7f0530e3aa53ee"
)
GLOBAL_LEDGER_ANCHOR_FILE_SHA256 = (
    "82dea23af42c2b26b6c489beb35ac5f09e560fae10a0fb7df6875bd44851b29f"
)

ENDPOINTS = v4.ENDPOINTS
TASK_FAMILIES = v4.TASK_FAMILIES
CURRENT_EXPOSURE_USD = v4.CURRENT_EXPOSURE_USD
ADMISSION_CEILING_USD = v4.ADMISSION_CEILING_USD
HARD_CAP_USD = v4.HARD_CAP_USD

_sha256 = v4._sha256
_file_sha256 = v4._file_sha256
_regular_json = v4._regular_json
_decimal_text = v4._decimal_text
_relative = v4._relative
_file_ref = v4._file_ref
_write_artifact = v4._write_artifact
verify_manifest_content_address = v4.predecessor.verify_manifest_content_address
pair_audit = v4.predecessor.pair_audit


class FullStudyError(RuntimeError):
    """A V5 identity, source, estimand, or admission binding failed."""


def _verified_artifact(
    repo_root: Path, relative: str, semantic_sha256: str
) -> dict[str, Any]:
    path = repo_root / relative
    document = _regular_json(path)
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("artifact_sha256") != semantic_sha256
        or _sha256(body) != semantic_sha256
        or path.is_symlink()
    ):
        raise FullStudyError(f"frozen predecessor differs: {relative}")
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
            str(block["admission_block_id"])
            for block in plan.get("admission_blocks") or []
        },
        "presentation_ids": {
            str(row["presentation_id"])
            for row in (plan.get("human_evaluation") or {}).get("presentations") or []
        },
    }


def _canonical_presentations(
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
        for endpoint_id, endpoint_contrasts in contrasts.items():
            for first, second, contrast in endpoint_contrasts:
                for condition in ("epicure_off", "epicure_on"):
                    coordinate = {
                        "schema_version": "flavourbench-reasoning-presentation-v5",
                        "freeze_nonce": FREEZE_NONCE,
                        "wave_id": wave["wave_id"],
                        "task_id": wave["task_id"],
                        "task_family": wave["task_family"],
                        "endpoint_id": endpoint_id,
                        "condition": condition,
                        "contrast": contrast,
                        "first_variant": first,
                        "second_variant": second,
                    }
                    first_item = item_map[(endpoint_id, wave["task_id"], first)]
                    second_item = item_map[(endpoint_id, wave["task_id"], second)]
                    records.append(
                        {
                            **coordinate,
                            "presentation_id": _sha256(coordinate),
                            "first_work_item_id": first_item["work_item_id"],
                            "second_work_item_id": second_item["work_item_id"],
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
            raise FullStudyError("presentation stratum is not six tasks")
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
    predecessor = _verified_artifact(repo_root, V4_PLAN_PATH, V4_PLAN_SHA256)
    old_to_new: dict[str, str] = {}
    work_items: list[dict[str, Any]] = []
    for old in predecessor["work_items"]:
        coordinate = copy.deepcopy(old["route_coordinate"])
        coordinate.update(
            {
                "schema_version": "flavourbench-reasoning-effort-task-wave-coordinate-v5",
                "freeze_nonce": FREEZE_NONCE,
                "superseded_plan_sha256": V4_PLAN_SHA256,
            }
        )
        route_cell_id = _sha256(coordinate)
        run_id = str(uuid.uuid5(NAMESPACE, f"{FREEZE_NONCE}:{route_cell_id}:run"))
        work_identity = {
            "schema_version": "flavourbench-reasoning-effort-work-item-identity-v5",
            "freeze_nonce": FREEZE_NONCE,
            "route_cell_id": route_cell_id,
            "run_id": run_id,
            "task_id": coordinate["task_id"],
            "endpoint_id": coordinate["endpoint_id"],
            "variant_id": coordinate["variant_id"],
        }
        work_item_id = _sha256(work_identity)
        old_to_new[str(old["work_item_id"])] = work_item_id
        work_items.append(
            {
                **copy.deepcopy(old),
                "route_cell_id": route_cell_id,
                "route_coordinate": coordinate,
                "run_id": run_id,
                "arm_ids": [f"{run_id}:epicure_off", f"{run_id}:epicure_on"],
                "attempt_slots": _attempt_slots(run_id, route_cell_id),
                "work_item_id": work_item_id,
                "supersedes_identifiers": {
                    "route_cell_id": old["route_cell_id"],
                    "run_id": old["run_id"],
                    "work_item_id": old["work_item_id"],
                },
            }
        )

    wave_map: dict[str, str] = {}
    waves: list[dict[str, Any]] = []
    new_items = {str(item["work_item_id"]): item for item in work_items}
    for old in predecessor["task_waves"]:
        wave_item_ids = [old_to_new[value] for value in old["work_item_ids"]]
        wave_reserve = sum(
            (Decimal(new_items[value]["worst_case_reserve_usd"]) for value in wave_item_ids),
            Decimal(0),
        )
        body = {
            **copy.deepcopy(old),
            "schema_version": "flavourbench-reasoning-effort-task-wave-v5",
            "freeze_nonce": FREEZE_NONCE,
            "superseded_wave_id": old["wave_id"],
            "work_item_ids": wave_item_ids,
            "worst_case_reserve_usd": _decimal_text(wave_reserve),
        }
        body.pop("wave_id", None)
        body["wave_id"] = _sha256(body)
        wave_map[str(old["wave_id"])] = body["wave_id"]
        waves.append(body)

    blocks: list[dict[str, Any]] = []
    for old in predecessor["admission_blocks"]:
        block_item_ids = [old_to_new[value] for value in old["work_item_ids"]]
        block_reserve = sum(
            (Decimal(new_items[value]["worst_case_reserve_usd"]) for value in block_item_ids),
            Decimal(0),
        )
        body = {
            **copy.deepcopy(old),
            "schema_version": "flavourbench-reasoning-effort-family-block-v5",
            "freeze_nonce": FREEZE_NONCE,
            "superseded_block_id": old["admission_block_id"],
            "wave_ids": [wave_map[value] for value in old["wave_ids"]],
            "work_item_ids": block_item_ids,
            "worst_case_reserve_usd": _decimal_text(block_reserve),
            "canonical_global_reservations_required": 28,
            "local_admission_binds_exact_global_reservation_shas": True,
        }
        body.pop("admission_block_id", None)
        body["admission_block_id"] = _sha256(body)
        blocks.append(body)

    presentations = _canonical_presentations(waves=waves, items=work_items)
    repeat_ids = [
        row["presentation_id"]
        for row in sorted(
            presentations,
            key=lambda row: _sha256(
                {"repeat": FREEZE_NONCE, "presentation_id": row["presentation_id"]}
            ),
        )[:24]
    ]
    source_artifacts = copy.deepcopy(predecessor["source_artifacts"])
    source_artifacts.update(
        {
            "retired_v4_no_go_plan": _file_ref(repo_root, repo_root / V4_PLAN_PATH),
            "retired_v4_human_protocol": _file_ref(
                repo_root, repo_root / V4_HUMAN_PATH
            ),
            "v3_import_pipeline_incident": _file_ref(
                repo_root, repo_root / V3_INCIDENT_PATH
            ),
            "v3_zero_call_recovery": _file_ref(
                repo_root, repo_root / V3_RECOVERY_PATH
            ),
        }
    )
    task_identity = v4.predecessor.canonical_task_wave_identity(
        tasks=predecessor["tasks"], waves=waves
    )
    plan = {
        **copy.deepcopy(predecessor),
        "schema_version": PLAN_SCHEMA,
        "record_role": "shared_ledger_crash_safe_reasoning_effort_successor",
        "status": "frozen_not_executed_independent_go_required",
        "study_id": STUDY_ID,
        "freeze_nonce": FREEZE_NONCE,
        "root_id": ROOT_ID,
        "supersedes": {
            "retired_v4_plan_sha256": V4_PLAN_SHA256,
            "retired_v4_decision": "no_go",
            "retired_v4_reason": (
                "V4 did not create canonical shared-ledger per-item reservations and "
                "could release unsupported post-start zero-cost paths"
            ),
            "v3_import_pipeline_incident_sha256": V3_INCIDENT_SHA256,
            "v3_zero_call_recovery_sha256": V3_RECOVERY_SHA256,
        },
        "source_artifacts": source_artifacts,
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
                endpoint: (
                    "flavourbench/artifacts/season1/current-quality-run/"
                    f"{ROOT_ID}/runs/{endpoint}"
                )
                for endpoint in ENDPOINTS
            },
            "canonical_global_reservation_ledger": GLOBAL_LEDGER_PATH,
            "canonical_global_source": GLOBAL_SOURCE_PATH,
        },
        "canonical_global_ledger_anchor": {
            "sequence": GLOBAL_LEDGER_ANCHOR_SEQUENCE,
            "head_entry_sha256": GLOBAL_LEDGER_ANCHOR_HEAD_SHA256,
            "physical_file_sha256_at_freeze": GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
            "baseline_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
        },
        "execution": {
            **copy.deepcopy(predecessor["execution"]),
            "module": "flavourbench.reasoning_effort_full_study_executor_v5",
            "confirmation": CONFIRMATION,
            "canonical_global_reservation_per_work_item": True,
            "canonical_source_before_local_terminal": True,
            "independent_go_artifact_required": True,
            "all_block_runtime_arguments_bound_after_attestation_before_reservation": True,
        },
        "failure_policy": {
            **copy.deepcopy(predecessor["failure_policy"]),
            "post_start_without_source_retains_canonical_reservation": True,
            "unsupported_zero_cost_release_permitted": False,
            "crash_cut_recovery_without_identifier_replay": True,
        },
        "human_evaluation": {
            **copy.deepcopy(predecessor["human_evaluation"]),
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
    plan["source_code"] = build_source_closure(repo_root=repo_root)
    plan["artifact_sha256"] = _sha256(plan)
    validate_plan(plan, repo_root=repo_root)
    return plan


def validate_plan(plan: Mapping[str, Any], *, repo_root: Path) -> None:
    body = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("artifact_sha256") != _sha256(
        body
    ):
        raise FullStudyError("V5 plan schema or content address failed")
    if (
        plan.get("freeze_nonce") != FREEZE_NONCE
        or plan.get("study_id") != STUDY_ID
        or plan.get("root_id") != ROOT_ID
        or plan.get("execution", {}).get("module")
        != "flavourbench.reasoning_effort_full_study_executor_v5"
        or plan.get("supersedes", {}).get("retired_v4_decision") != "no_go"
        or plan.get("supersedes", {}).get("retired_v4_plan_sha256") != V4_PLAN_SHA256
    ):
        raise FullStudyError("V5 identity or retired V4 NO-GO binding differs")
    try:
        verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    except SourceClosureError as error:
        raise FullStudyError(f"V5 source closure does not rederive: {error}") from error
    for key, digest in (
        ("retired_v4_no_go_plan", V4_PLAN_SHA256),
        ("retired_v4_human_protocol", V4_HUMAN_SHA256),
        ("v3_import_pipeline_incident", V3_INCIDENT_SHA256),
        ("v3_zero_call_recovery", V3_RECOVERY_SHA256),
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
            raise FullStudyError(f"V5 predecessor source differs: {key}")
    items = plan.get("work_items") or []
    waves = plan.get("task_waves") or []
    blocks = plan.get("admission_blocks") or []
    canonical = (plan.get("human_evaluation") or {}).get("presentations") or []
    if (
        len(items) != 168
        or len(waves) != 24
        or len(blocks) != 6
        or len(canonical) != 240
        or sum(row["contrast"] == "primary_low_high" for row in canonical) != 144
        or sum(row["contrast"] != "primary_low_high" for row in canonical) != 96
        or len(_identity_sets(plan)["attempt_ids"]) != 168 * 56
        or Counter(task["family"] for task in plan.get("tasks") or [])
        != Counter({family: 6 for family in TASK_FAMILIES})
    ):
        raise FullStudyError("V5 estimand counts differ")
    item_reserves = {
        str(item["work_item_id"]): Decimal(item["worst_case_reserve_usd"])
        for item in items
    }
    if any(
        len(block.get("work_item_ids") or []) != 28
        or block.get("canonical_global_reservations_required") != 28
        or Decimal(block["worst_case_reserve_usd"])
        != sum(
            (item_reserves[item_id] for item_id in block["work_item_ids"]),
            Decimal(0),
        )
        or Counter(block.get("task_families") or [])
        != Counter({family: 1 for family in TASK_FAMILIES})
        for block in blocks
    ):
        raise FullStudyError("V5 block structure differs")
    if any(
        Decimal(wave["worst_case_reserve_usd"])
        != sum((item_reserves[item_id] for item_id in wave["work_item_ids"]), Decimal(0))
        for wave in waves
    ):
        raise FullStudyError("V5 wave reservation is not the exact item sum")
    expected_anchor = {
        "sequence": GLOBAL_LEDGER_ANCHOR_SEQUENCE,
        "head_entry_sha256": GLOBAL_LEDGER_ANCHOR_HEAD_SHA256,
        "physical_file_sha256_at_freeze": GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
        "baseline_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
    }
    if plan.get("canonical_global_ledger_anchor") != expected_anchor:
        raise FullStudyError("V5 canonical global-ledger anchor differs")
    if plan.get("task_wave_identity") != v4.predecessor.canonical_task_wave_identity(
        tasks=plan.get("tasks") or [], waves=waves
    ):
        raise FullStudyError("V5 task-wave identity does not rederive")
    current = _identity_sets(plan)
    for label, relative, digest in (
        ("V4", V4_PLAN_PATH, V4_PLAN_SHA256),
        ("V3", V3_PLAN_PATH, V3_PLAN_SHA256),
        ("V2", V2_PLAN_PATH, V2_PLAN_SHA256),
    ):
        retired = _identity_sets(_verified_artifact(repo_root, relative, digest))
        for identity_type in current:
            if current[identity_type] & retired[identity_type]:
                raise FullStudyError(f"V5 reuses a {label} {identity_type}")


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
        raise FullStudyError("V5 execution roots are not empty")


def build_human_protocol(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    previous = _verified_artifact(repo_root, V4_HUMAN_PATH, V4_HUMAN_SHA256)
    dossier = human._verified_dossier(
        repo_root / previous["source_bindings"]["task_dossier"]["path"]
    )
    selected = human._selected_tasks(dossier, plan)
    arms, cells = human._build_graph(selected, plan)
    presentations, assignment_blocks = human._build_presentations(cells)
    protocol = {
        **copy.deepcopy(previous),
        "schema_version": HUMAN_PROTOCOL_SCHEMA,
        "study_id": f"{STUDY_ID}-human-v5",
        "supersedes": {
            "artifact_sha256": V4_HUMAN_SHA256,
            "reason": "V4 executor and presentation identifiers were retired NO-GO",
            "v4_plan_sha256": V4_PLAN_SHA256,
        },
        "reasoning_task_wave_binding": {
            "study_plan_sha256": plan["artifact_sha256"],
            "task_selection_artifact_sha256": plan["source_artifacts"]["task_selection"][
                "semantic_sha256"
            ],
            **copy.deepcopy(plan["task_wave_identity"]),
        },
        "tasks": selected,
        "arm_coordinates": arms,
        "comparison_cells": cells,
        "presentations": presentations,
        "presentation_allocation": {
            **copy.deepcopy(previous["presentation_allocation"]),
            "assignment_blocks": assignment_blocks,
        },
        "source_bindings": {
            **copy.deepcopy(previous["source_bindings"]),
            "executor_study_plan": {
                "path": None,
                "semantic_sha256": plan["artifact_sha256"],
                "file_sha256": None,
                "schema_version": PLAN_SCHEMA,
            },
            "retired_v4_human_protocol": _file_ref(
                repo_root, repo_root / V4_HUMAN_PATH
            ),
        },
    }
    protocol.pop("artifact_sha256", None)
    protocol["artifact_sha256"] = _sha256(protocol)
    verify_human_protocol_binding(plan=plan, human_protocol=protocol)
    if {
        str(row["presentation_id"]) for row in protocol["presentations"]
    } & {str(row["presentation_id"]) for row in previous["presentations"]}:
        raise FullStudyError("V5 reuses a V4 assignment presentation identifier")
    return protocol


def verify_human_protocol_binding(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any]
) -> None:
    body = {
        key: value for key, value in human_protocol.items() if key != "artifact_sha256"
    }
    binding = human_protocol.get("reasoning_task_wave_binding") or {}
    if (
        human_protocol.get("schema_version") != HUMAN_PROTOCOL_SCHEMA
        or human_protocol.get("artifact_sha256") != _sha256(body)
        or binding.get("study_plan_sha256") != plan["artifact_sha256"]
        or binding.get("selected_task_set_sha256")
        != plan["task_wave_identity"]["selected_task_set_sha256"]
        or binding.get("wave_order_sha256")
        != plan["task_wave_identity"]["wave_order_sha256"]
    ):
        raise FullStudyError("V5 human protocol binding differs")
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
    observed_arms = {
        (
            str(arm.get("executor_arm_id")),
            str(arm.get("executor_work_item_id")),
            str(arm.get("task_id")),
            str(arm.get("endpoint_id")),
            str(arm.get("condition")),
            str(arm.get("variant")),
        )
        for arm in human_protocol.get("arm_coordinates") or []
        if isinstance(arm, Mapping)
    }
    cells = human_protocol.get("comparison_cells") or []
    if (
        len(observed_arms) != 336
        or observed_arms != expected_arms
        or len(cells) != 240
        or len(human_protocol.get("presentations") or []) != 1584
        or (human_protocol.get("counts") or {}).get("original_presentations") != 1440
        or (human_protocol.get("counts") or {}).get("position_swapped_repeats") != 144
        or {str(cell.get("executor_presentation_id")) for cell in cells}
        != {
            str(row["presentation_id"])
            for row in plan["human_evaluation"]["presentations"]
        }
    ):
        raise FullStudyError("V5 arm, cell, or assignment graph differs")


def build_preflight(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    assert_successor_roots_empty(plan=plan, repo_root=repo_root)
    first = Decimal(plan["admission_blocks"][0]["worst_case_reserve_usd"])
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "zero_call_crash_safe_successor_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "decision": "technical_preflight_pass_independent_go_not_supplied",
        "checks": {
            "retired_v4_no_go_bound": V4_PLAN_SHA256,
            "successor_roots_empty": True,
            "canonical_global_ledger_anchor": plan["canonical_global_ledger_anchor"],
            "first_block_reserve_usd": _decimal_text(first),
            "first_block_projected_usd": _decimal_text(CURRENT_EXPOSURE_USD + first),
            "below_85_percent_admission": CURRENT_EXPOSURE_USD + first
            <= ADMISSION_CEILING_USD,
            "independent_go_required": True,
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
        "record_role": "cross_bound_crash_safe_successor_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "preliminary_preflight_sha256": preliminary["artifact_sha256"],
        "decision": "technical_pass_independent_go_required_before_live_execution",
        "checks": {
            **preliminary["checks"],
            "human_protocol_cross_verified": True,
            "canonical_comparison_cells": 240,
            "assignment_presentations": 1584,
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
        or bound_preflight.get("decision")
        != "technical_pass_independent_go_required_before_live_execution"
    ):
        raise FullStudyError("V5 bound preflight is absent or invalid")


def verify_governance_go(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
) -> None:
    body = {key: value for key, value in governance_go.items() if key != "artifact_sha256"}
    if (
        governance_go.get("schema_version") != GOVERNANCE_GO_SCHEMA
        or governance_go.get("artifact_sha256") != _sha256(body)
        or governance_go.get("decision") != "go_for_exactly_one_family_block"
        or governance_go.get("study_plan_sha256") != plan["artifact_sha256"]
        or governance_go.get("human_protocol_sha256") != human_protocol["artifact_sha256"]
        or governance_go.get("bound_preflight_sha256")
        != bound_preflight["artifact_sha256"]
        or governance_go.get("reviewer_is_executor") is not False
        or governance_go.get("provider_or_epicure_calls_made_by_review") is not False
        or governance_go.get("maximum_family_blocks") != 1
    ):
        raise FullStudyError("an exact independent one-block GO artifact is required")


def freeze(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    plan = build_plan(repo_root=repo_root)
    plan_path = _write_artifact(output_dir / "plan", "reasoning-effort-plan-v5", plan)
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
        output_dir / "human-protocol", "reasoning-effort-human-protocol-v5", protocol
    )
    protocol = _regular_json(human_path)
    preflight = build_preflight(plan=plan, repo_root=repo_root)
    preflight_path = _write_artifact(
        output_dir / "preflight", "reasoning-effort-preflight-v5", preflight
    )
    bound = build_bound_preflight(
        plan=plan, human_protocol=protocol, repo_root=repo_root
    )
    bound_path = _write_artifact(
        output_dir / "bound-preflight", "reasoning-effort-bound-preflight-v5", bound
    )
    return {
        "plan": plan_path,
        "human_protocol": human_path,
        "preflight": preflight_path,
        "bound_preflight": bound_path,
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = freeze(
        repo_root=arguments.repo_root.resolve(), output_dir=arguments.output_dir.resolve()
    )
    print(json.dumps({key: str(path) for key, path in result.items()}, indent=2))


if __name__ == "__main__":
    run()
