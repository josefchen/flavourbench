"""Freeze the V7 successor to the independently rejected V6 study.

V7 preserves the V6 estimand and exact Decimal budget while retiring every
execution-facing V6 identity.  It remains inert until a different independent
reviewer issues an exact content-addressed one-block GO.
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

from . import reasoning_effort_full_study_v6 as v6
from . import reasoning_effort_human_protocol as human
from .reasoning_effort_source_closure_v7 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)

PLAN_SCHEMA = "flavourbench-reasoning-effort-task-wave-plan-v7"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-task-wave-preflight-v7"
BOUND_PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-bound-preflight-v7"
HUMAN_PROTOCOL_SCHEMA = "flavourbench-reasoning-effort-human-evaluation-protocol-v7"
GOVERNANCE_GO_SCHEMA = "flavourbench-reasoning-effort-independent-go-v3"

FREEZE_NONCE = "reasoning-effort-task-waves-v7-v6-independent-no-go-remediation-2026-08-04"
NAMESPACE = uuid.UUID("49f82d79-d630-4e41-9ca7-a07900442d72")
STUDY_ID = "frontier-reasoning-effort-task-waves-v7-v6-independent-no-go-remediation"
ROOT_ID = "reasoning-effort-task-waves-v7-v6-independent-no-go-remediation"
CONFIRMATION = "RUN_REASONING_EFFORT_V7_ONE_INDEPENDENTLY_REVIEWED_FAMILY_BLOCK"

V6_PLAN_SHA256 = "92c5584d635ea6fd42c19cc8f0ce31dd9c72e1410af96fd081cd533df02744ff"
V6_PLAN_FILE_SHA256 = "8f3c9f3c142e9d68966498658b38a6f3a0130658498b104eac496aa1d0d8402f"
V6_HUMAN_SHA256 = "367daef3caf31fa7697a7d7765b61902588dc1031c59b8ccc99d4ff13199ced0"
V6_HUMAN_FILE_SHA256 = "86195472c1707113ec60b08f3af446fe62cd223924cb7e3cd74753d311cdcc0f"
V6_PREFLIGHT_SHA256 = "cc413904cd5153f1f3d7c9c2012fd281c3d95d7e97b3725ea98be3c3ad78ba01"
V6_PREFLIGHT_FILE_SHA256 = "6cecc9892b992d18606760f8ce17b66558b90f91792240410852acdffdee7e69"
V6_BOUND_SHA256 = "704ba199b573f38c8e03656e5dd12f713257067f9e4db7aca6e89d4de0df85f9"
V6_BOUND_FILE_SHA256 = "63155695a8f28258d913d1a2a9138f2191e4e1b71af5ae702bdf8fe989d8b380"
V6_NO_GO_SHA256 = "7fda72011d031a83af707f3caa5778a7af305b9a8f777d3e0503122034ba3cf1"
V6_NO_GO_FILE_SHA256 = "217fde602aea9ac4ae4864444d2a8373eb49eec1e5eaee5ef01dda48013b357b"

V6_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v6-independent-no-go-remediation"
)
V6_PLAN_PATH = f"{V6_ROOT}/plan/reasoning-effort-plan-v6-{V6_PLAN_SHA256}.json"
V6_HUMAN_PATH = (
    f"{V6_ROOT}/human-protocol/reasoning-effort-human-protocol-v6-{V6_HUMAN_SHA256}.json"
)
V6_PREFLIGHT_PATH = f"{V6_ROOT}/preflight/reasoning-effort-preflight-v6-{V6_PREFLIGHT_SHA256}.json"
V6_BOUND_PATH = (
    f"{V6_ROOT}/bound-preflight/reasoning-effort-bound-preflight-v6-{V6_BOUND_SHA256}.json"
)
V6_NO_GO_PATH = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v6-independent-no-go-v1/"
    f"reasoning-effort-v6-independent-no-go-{V6_NO_GO_SHA256}.json"
)

GLOBAL_LEDGER_PATH = v6.GLOBAL_LEDGER_PATH
GLOBAL_SOURCE_PATH = v6.GLOBAL_SOURCE_PATH
GLOBAL_LEDGER_ANCHOR_SEQUENCE = v6.GLOBAL_LEDGER_ANCHOR_SEQUENCE
GLOBAL_LEDGER_ANCHOR_HEAD_SHA256 = v6.GLOBAL_LEDGER_ANCHOR_HEAD_SHA256
GLOBAL_LEDGER_ANCHOR_FILE_SHA256 = v6.GLOBAL_LEDGER_ANCHOR_FILE_SHA256

ENDPOINTS = v6.ENDPOINTS
TASK_FAMILIES = v6.TASK_FAMILIES
CURRENT_EXPOSURE_USD = v6.CURRENT_EXPOSURE_USD
ADMISSION_CEILING_USD = v6.ADMISSION_CEILING_USD
HARD_CAP_USD = v6.HARD_CAP_USD

_sha256 = v6._sha256
_file_sha256 = v6._file_sha256
_regular_json = v6._regular_json
_decimal_text = v6._decimal_text
_relative = v6._relative
_file_ref = v6._file_ref
_write_artifact = v6._write_artifact
_exact_sum = v6._exact_sum
_exact_add = v6._exact_add
verify_manifest_content_address = v6.verify_manifest_content_address
pair_audit = v6.pair_audit


class FullStudyError(RuntimeError):
    """A V7 identity, source, budget, or admission binding failed."""


def _verified_artifact(
    repo_root: Path,
    relative: str,
    semantic_sha256: str,
    physical_sha256: str | None = None,
) -> dict[str, Any]:
    path = repo_root / relative
    document = _regular_json(path)
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("artifact_sha256") != semantic_sha256
        or _sha256(body) != semantic_sha256
        or path.is_symlink()
        or (physical_sha256 is not None and _file_sha256(path) != physical_sha256)
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
            str(slot["attempt_id"]) for item in items for slot in item.get("attempt_slots") or []
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
                        "schema_version": "flavourbench-reasoning-presentation-v7",
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
            raise FullStudyError("V7 presentation stratum is not six tasks")
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
    predecessor = _verified_artifact(repo_root, V6_PLAN_PATH, V6_PLAN_SHA256, V6_PLAN_FILE_SHA256)
    _verified_artifact(repo_root, V6_HUMAN_PATH, V6_HUMAN_SHA256, V6_HUMAN_FILE_SHA256)
    _verified_artifact(repo_root, V6_PREFLIGHT_PATH, V6_PREFLIGHT_SHA256, V6_PREFLIGHT_FILE_SHA256)
    _verified_artifact(repo_root, V6_BOUND_PATH, V6_BOUND_SHA256, V6_BOUND_FILE_SHA256)
    no_go = _verified_artifact(repo_root, V6_NO_GO_PATH, V6_NO_GO_SHA256, V6_NO_GO_FILE_SHA256)
    if no_go.get("decision") != "no_go_for_first_family_balanced_block":
        raise FullStudyError("V6 independent artifact is not an exact NO-GO")

    old_to_new: dict[str, str] = {}
    work_items: list[dict[str, Any]] = []
    for old in predecessor["work_items"]:
        coordinate = copy.deepcopy(old["route_coordinate"])
        coordinate.update(
            {
                "schema_version": "flavourbench-reasoning-effort-task-wave-coordinate-v7",
                "freeze_nonce": FREEZE_NONCE,
                "superseded_plan_sha256": V6_PLAN_SHA256,
                "superseded_no_go_sha256": V6_NO_GO_SHA256,
            }
        )
        route_cell_id = _sha256(coordinate)
        run_id = str(uuid.uuid5(NAMESPACE, f"{FREEZE_NONCE}:{route_cell_id}:run"))
        work_identity = {
            "schema_version": "flavourbench-reasoning-effort-work-item-identity-v7",
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

    items_by_id = {str(item["work_item_id"]): item for item in work_items}
    wave_map: dict[str, str] = {}
    waves: list[dict[str, Any]] = []
    for old in predecessor["task_waves"]:
        item_ids = [old_to_new[value] for value in old["work_item_ids"]]
        reserve = _exact_sum(
            [Decimal(items_by_id[value]["worst_case_reserve_usd"]) for value in item_ids]
        )
        body = {
            **copy.deepcopy(old),
            "schema_version": "flavourbench-reasoning-effort-task-wave-v7",
            "freeze_nonce": FREEZE_NONCE,
            "superseded_wave_id": old["wave_id"],
            "work_item_ids": item_ids,
            "worst_case_reserve_usd": _decimal_text(reserve),
        }
        body.pop("wave_id", None)
        body["wave_id"] = _sha256(body)
        wave_map[str(old["wave_id"])] = body["wave_id"]
        waves.append(body)

    blocks: list[dict[str, Any]] = []
    for old in predecessor["admission_blocks"]:
        item_ids = [old_to_new[value] for value in old["work_item_ids"]]
        reserve = _exact_sum(
            [Decimal(items_by_id[value]["worst_case_reserve_usd"]) for value in item_ids]
        )
        body = {
            **copy.deepcopy(old),
            "schema_version": "flavourbench-reasoning-effort-family-block-v7",
            "freeze_nonce": FREEZE_NONCE,
            "superseded_block_id": old["admission_block_id"],
            "wave_ids": [wave_map[value] for value in old["wave_ids"]],
            "work_item_ids": item_ids,
            "worst_case_reserve_usd": _decimal_text(reserve),
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
            "retired_v6_no_go_plan": _file_ref(repo_root, repo_root / V6_PLAN_PATH),
            "retired_v6_human_protocol": _file_ref(repo_root, repo_root / V6_HUMAN_PATH),
            "retired_v6_preflight": _file_ref(repo_root, repo_root / V6_PREFLIGHT_PATH),
            "retired_v6_bound_preflight": _file_ref(repo_root, repo_root / V6_BOUND_PATH),
            "v6_independent_no_go": _file_ref(repo_root, repo_root / V6_NO_GO_PATH),
        }
    )
    task_identity = v6.v5.v4.predecessor.canonical_task_wave_identity(
        tasks=predecessor["tasks"], waves=waves
    )
    total_reserve = _exact_sum([Decimal(item["worst_case_reserve_usd"]) for item in work_items])
    first_reserve = Decimal(blocks[0]["worst_case_reserve_usd"])
    budget = copy.deepcopy(predecessor["budget"])
    budget.update(
        {
            "first_block_worst_case_usd": _decimal_text(first_reserve),
            "all_24_waves_worst_case_usd": _decimal_text(total_reserve),
            "all_24_waves_projected_usd": _decimal_text(
                _exact_add(CURRENT_EXPOSURE_USD, total_reserve)
            ),
            "current_total_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
            "all_reserve_and_projection_values_derived_from_exact_decimal_sums": True,
        }
    )
    plan = {
        **copy.deepcopy(predecessor),
        "schema_version": PLAN_SCHEMA,
        "record_role": "v6_independent_no_go_remediated_reasoning_effort_successor",
        "status": "frozen_not_executed_independent_go_required",
        "study_id": STUDY_ID,
        "freeze_nonce": FREEZE_NONCE,
        "root_id": ROOT_ID,
        "supersedes": {
            "retired_v6_plan_sha256": V6_PLAN_SHA256,
            "retired_v6_plan_file_sha256": V6_PLAN_FILE_SHA256,
            "retired_v6_human_sha256": V6_HUMAN_SHA256,
            "retired_v6_human_file_sha256": V6_HUMAN_FILE_SHA256,
            "retired_v6_preflight_sha256": V6_PREFLIGHT_SHA256,
            "retired_v6_preflight_file_sha256": V6_PREFLIGHT_FILE_SHA256,
            "retired_v6_bound_sha256": V6_BOUND_SHA256,
            "retired_v6_bound_file_sha256": V6_BOUND_FILE_SHA256,
            "v6_independent_no_go_sha256": V6_NO_GO_SHA256,
            "v6_independent_no_go_file_sha256": V6_NO_GO_FILE_SHA256,
            "retired_v6_decision": "no_go",
            "closed_blocker_count": 2,
        },
        "source_artifacts": source_artifacts,
        "work_items": work_items,
        "task_waves": waves,
        "wave_execution_order": [wave["wave_id"] for wave in waves],
        "admission_blocks": blocks,
        "block_execution_order": [block["admission_block_id"] for block in blocks],
        "task_wave_identity": task_identity,
        "budget": budget,
        "execution_roots": {
            "coordinator": (
                f"flavourbench/artifacts/season1/current-quality-run/{ROOT_ID}/runs/coordinator"
            ),
            "endpoints": {
                endpoint: (
                    f"flavourbench/artifacts/season1/current-quality-run/{ROOT_ID}/runs/{endpoint}"
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
            "module": "flavourbench.reasoning_effort_full_study_executor_v7",
            "confirmation": CONFIRMATION,
            "independent_go_artifact_required": True,
            "every_endpoint_incident_replay_rederives_canonical_disposition": True,
            "normal_prestart_failures_use_coordinator_only_no_delivery_state": True,
            "normal_exception_and_process_crash_cut_matrix_required": True,
        },
        "failure_policy": {
            **copy.deepcopy(predecessor["failure_policy"]),
            "stale_endpoint_incident_disposition_rejected": True,
            "endpoint_incident_requires_durable_item_start": True,
            "prestart_no_delivery_never_appends_endpoint_event": True,
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
    supersedes = plan.get("supersedes") or {}
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("artifact_sha256") != _sha256(body)
        or plan.get("freeze_nonce") != FREEZE_NONCE
        or plan.get("study_id") != STUDY_ID
        or plan.get("root_id") != ROOT_ID
        or plan.get("status") != "frozen_not_executed_independent_go_required"
        or plan.get("execution", {}).get("module")
        != "flavourbench.reasoning_effort_full_study_executor_v7"
        or supersedes.get("retired_v6_decision") != "no_go"
        or supersedes.get("v6_independent_no_go_sha256") != V6_NO_GO_SHA256
    ):
        raise FullStudyError("V7 identity or V6 NO-GO binding differs")
    try:
        verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    except SourceClosureError as error:
        raise FullStudyError(f"V7 source closure does not rederive: {error}") from error
    references = {
        "retired_v6_no_go_plan": (V6_PLAN_SHA256, V6_PLAN_FILE_SHA256),
        "retired_v6_human_protocol": (V6_HUMAN_SHA256, V6_HUMAN_FILE_SHA256),
        "retired_v6_preflight": (V6_PREFLIGHT_SHA256, V6_PREFLIGHT_FILE_SHA256),
        "retired_v6_bound_preflight": (V6_BOUND_SHA256, V6_BOUND_FILE_SHA256),
        "v6_independent_no_go": (V6_NO_GO_SHA256, V6_NO_GO_FILE_SHA256),
    }
    for key, (semantic, physical) in references.items():
        reference = plan.get("source_artifacts", {}).get(key) or {}
        path = repo_root / str(reference.get("path") or "")
        if (
            reference.get("semantic_sha256") != semantic
            or reference.get("file_sha256") != physical
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != reference.get("bytes")
            or _file_sha256(path) != physical
        ):
            raise FullStudyError(f"V7 predecessor source differs: {key}")
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
        raise FullStudyError("V7 estimand counts differ")
    reserves = {
        str(item["work_item_id"]): Decimal(item["worst_case_reserve_usd"]) for item in items
    }
    for collection in (waves, blocks):
        for row in collection:
            expected = _exact_sum([reserves[item_id] for item_id in row["work_item_ids"]])
            if Decimal(row["worst_case_reserve_usd"]) != expected:
                raise FullStudyError("V7 wave/block reserve is not an exact item sum")
    if any(
        len(block.get("work_item_ids") or []) != 28
        or block.get("canonical_global_reservations_required") != 28
        or Counter(block.get("task_families") or [])
        != Counter({family: 1 for family in TASK_FAMILIES})
        for block in blocks
    ):
        raise FullStudyError("V7 block structure differs")
    total = _exact_sum(list(reserves.values()))
    budget = plan.get("budget") or {}
    if (
        Decimal(budget.get("first_block_worst_case_usd", "NaN"))
        != Decimal(blocks[0]["worst_case_reserve_usd"])
        or Decimal(budget.get("all_24_waves_worst_case_usd", "NaN")) != total
        or Decimal(budget.get("all_24_waves_projected_usd", "NaN"))
        != _exact_add(CURRENT_EXPOSURE_USD, total)
        or budget.get("all_reserve_and_projection_values_derived_from_exact_decimal_sums")
        is not True
    ):
        raise FullStudyError("V7 exact Decimal budget summary differs")
    expected_anchor = {
        "sequence": GLOBAL_LEDGER_ANCHOR_SEQUENCE,
        "head_entry_sha256": GLOBAL_LEDGER_ANCHOR_HEAD_SHA256,
        "physical_file_sha256_at_freeze": GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
        "baseline_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
    }
    if plan.get("canonical_global_ledger_anchor") != expected_anchor:
        raise FullStudyError("V7 canonical global-ledger anchor differs")
    if plan.get("task_wave_identity") != v6.v5.v4.predecessor.canonical_task_wave_identity(
        tasks=plan.get("tasks") or [], waves=waves
    ):
        raise FullStudyError("V7 task-wave identity does not rederive")
    current = _identity_sets(plan)
    historical = (
        ("V6", V6_PLAN_PATH, V6_PLAN_SHA256, V6_PLAN_FILE_SHA256),
        ("V5", v6.V5_PLAN_PATH, v6.V5_PLAN_SHA256, v6.V5_PLAN_FILE_SHA256),
        ("V4", v6.v5.V4_PLAN_PATH, v6.v5.V4_PLAN_SHA256, None),
        ("V3", v6.v5.V3_PLAN_PATH, v6.v5.V3_PLAN_SHA256, None),
        ("V2", v6.v5.V2_PLAN_PATH, v6.v5.V2_PLAN_SHA256, None),
    )
    for label, relative, semantic, physical in historical:
        retired = _identity_sets(_verified_artifact(repo_root, relative, semantic, physical))
        for identity_type in current:
            if current[identity_type] & retired[identity_type]:
                raise FullStudyError(f"V7 reuses a {label} {identity_type}")


def successor_roots(*, plan: Mapping[str, Any], repo_root: Path) -> list[Path]:
    return [
        repo_root / str(plan["execution_roots"]["coordinator"]),
        *(repo_root / str(relative) for relative in plan["execution_roots"]["endpoints"].values()),
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
        raise FullStudyError("V7 execution roots are not empty")


def build_human_protocol(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    previous = _verified_artifact(repo_root, V6_HUMAN_PATH, V6_HUMAN_SHA256, V6_HUMAN_FILE_SHA256)
    dossier = human._verified_dossier(
        repo_root / previous["source_bindings"]["task_dossier"]["path"]
    )
    selected = human._selected_tasks(dossier, plan)
    arms, cells = human._build_graph(selected, plan)
    presentations, assignment_blocks = human._build_presentations(cells)
    protocol = {
        **copy.deepcopy(previous),
        "schema_version": HUMAN_PROTOCOL_SCHEMA,
        "study_id": f"{STUDY_ID}-human-v7",
        "supersedes": {
            "artifact_sha256": V6_HUMAN_SHA256,
            "artifact_file_sha256": V6_HUMAN_FILE_SHA256,
            "reason": "V6 was independently rejected and all identities were retired",
            "v6_plan_sha256": V6_PLAN_SHA256,
            "v6_independent_no_go_sha256": V6_NO_GO_SHA256,
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
            "retired_v6_human_protocol": _file_ref(repo_root, repo_root / V6_HUMAN_PATH),
            "v6_independent_no_go": _file_ref(repo_root, repo_root / V6_NO_GO_PATH),
        },
    }
    protocol.pop("artifact_sha256", None)
    protocol["artifact_sha256"] = _sha256(protocol)
    verify_human_protocol_binding(plan=plan, human_protocol=protocol)
    if {str(row["presentation_id"]) for row in protocol["presentations"]} & {
        str(row["presentation_id"]) for row in previous["presentations"]
    }:
        raise FullStudyError("V7 reuses a V6 assignment presentation identifier")
    return protocol


def verify_human_protocol_binding(
    *, plan: Mapping[str, Any], human_protocol: Mapping[str, Any]
) -> None:
    body = {key: value for key, value in human_protocol.items() if key != "artifact_sha256"}
    binding = human_protocol.get("reasoning_task_wave_binding") or {}
    if (
        human_protocol.get("schema_version") != HUMAN_PROTOCOL_SCHEMA
        or human_protocol.get("artifact_sha256") != _sha256(body)
        or binding.get("study_plan_sha256") != plan["artifact_sha256"]
        or binding.get("selected_task_set_sha256")
        != plan["task_wave_identity"]["selected_task_set_sha256"]
        or binding.get("wave_order_sha256") != plan["task_wave_identity"]["wave_order_sha256"]
    ):
        raise FullStudyError("V7 human protocol binding differs")
    items = {str(item["work_item_id"]): item for item in plan["work_items"]}
    expected_arms = {
        (
            str(arm_id),
            item_id,
            str(item["route_coordinate"]["task_id"]),
            str(item["route_coordinate"]["endpoint_id"]),
            str(arm_id).rsplit(":", 1)[-1],
            str(item["route_coordinate"]["variant_id"]),
        )
        for item_id, item in items.items()
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
        != {str(row["presentation_id"]) for row in plan["human_evaluation"]["presentations"]}
    ):
        raise FullStudyError("V7 arm, cell, or assignment graph differs")


def build_preflight(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    assert_successor_roots_empty(plan=plan, repo_root=repo_root)
    first = Decimal(plan["admission_blocks"][0]["worst_case_reserve_usd"])
    projected = _exact_add(CURRENT_EXPOSURE_USD, first)
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "zero_call_v6_no_go_remediated_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "decision": "technical_preflight_pass_independent_go_not_supplied",
        "checks": {
            "retired_v6_plan_bound": V6_PLAN_SHA256,
            "v6_independent_no_go_bound": V6_NO_GO_SHA256,
            "successor_roots_empty": True,
            "canonical_global_ledger_anchor": plan["canonical_global_ledger_anchor"],
            "first_block_reserve_usd": _decimal_text(first),
            "first_block_projected_usd": _decimal_text(projected),
            "below_85_percent_admission": projected <= ADMISSION_CEILING_USD,
            "independent_go_required": True,
            "closed_v6_blockers": 2,
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
        "record_role": "cross_bound_v6_no_go_remediated_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "preliminary_preflight_sha256": preliminary["artifact_sha256"],
        "v6_independent_no_go_sha256": V6_NO_GO_SHA256,
        "decision": "technical_pass_different_independent_go_required_before_execution",
        "checks": {
            **preliminary["checks"],
            "human_protocol_cross_verified": True,
            "canonical_comparison_cells": 240,
            "assignment_presentations": 1584,
            "reviewer_must_differ_from_v6_reviewer_and_v7_builder": True,
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
    body = {key: value for key, value in bound_preflight.items() if key != "artifact_sha256"}
    if (
        bound_preflight.get("schema_version") != BOUND_PREFLIGHT_SCHEMA
        or bound_preflight.get("artifact_sha256") != _sha256(body)
        or bound_preflight.get("study_plan_sha256") != plan["artifact_sha256"]
        or bound_preflight.get("human_protocol_sha256") != human_protocol["artifact_sha256"]
        or bound_preflight.get("v6_independent_no_go_sha256") != V6_NO_GO_SHA256
        or bound_preflight.get("decision")
        != "technical_pass_different_independent_go_required_before_execution"
    ):
        raise FullStudyError("V7 bound preflight is absent or invalid")


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
        or governance_go.get("bound_preflight_sha256") != bound_preflight["artifact_sha256"]
        or governance_go.get("reviewer_is_executor") is not False
        or governance_go.get("reviewer_is_v6_independent_reviewer") is not False
        or governance_go.get("reviewer_is_v7_builder") is not False
        or governance_go.get("reviewed_v6_no_go_sha256") != V6_NO_GO_SHA256
        or governance_go.get("provider_or_epicure_calls_made_by_review") is not False
        or governance_go.get("maximum_family_blocks") != 1
    ):
        raise FullStudyError("a different independent exact one-block GO is required")


def freeze(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    plan = build_plan(repo_root=repo_root)
    plan_path = _write_artifact(output_dir / "plan", "reasoning-effort-plan-v7", plan)
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
        output_dir / "human-protocol", "reasoning-effort-human-protocol-v7", protocol
    )
    protocol = _regular_json(human_path)
    preflight = build_preflight(plan=plan, repo_root=repo_root)
    preflight_path = _write_artifact(
        output_dir / "preflight", "reasoning-effort-preflight-v7", preflight
    )
    bound = build_bound_preflight(plan=plan, human_protocol=protocol, repo_root=repo_root)
    bound_path = _write_artifact(
        output_dir / "bound-preflight", "reasoning-effort-bound-preflight-v7", bound
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
