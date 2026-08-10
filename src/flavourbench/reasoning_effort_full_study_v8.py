"""Freeze the V8 successor to the stopped V7 pre-reservation attempt.

V8 preserves the V7 estimand and exact Decimal budget while retiring every
execution-facing V7 identity.  It remains inert until a different independent
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

from . import reasoning_effort_full_study_v7 as v7
from . import reasoning_effort_human_protocol as human
from .reasoning_effort_route_gate_v5 import (
    raw_endpoint_contract,
    semantic_endpoint_contract,
)
from .reasoning_effort_source_closure_v8 import (
    SourceClosureError,
    build_source_closure,
    verify_source_closure,
)

PLAN_SCHEMA = "flavourbench-reasoning-effort-task-wave-plan-v8"
PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-task-wave-preflight-v8"
BOUND_PREFLIGHT_SCHEMA = "flavourbench-reasoning-effort-bound-preflight-v8"
HUMAN_PROTOCOL_SCHEMA = "flavourbench-reasoning-effort-human-evaluation-protocol-v8"
GOVERNANCE_GO_SCHEMA = "flavourbench-reasoning-effort-independent-go-v4"
PREFLIGHT_INCIDENT_SCHEMA = "flavourbench-reasoning-effort-v7-preflight-drift-incident-v1"

FREEZE_NONCE = "reasoning-effort-task-waves-v8-monotone-capacity-attestation-2026-08-08"
NAMESPACE = uuid.UUID("0d370940-86fc-41ac-83ce-67d83e647d37")
STUDY_ID = "frontier-reasoning-effort-task-waves-v8-monotone-capacity-attestation"
ROOT_ID = "reasoning-effort-task-waves-v8-monotone-capacity-attestation"
CONFIRMATION = "RUN_REASONING_EFFORT_V8_ONE_INDEPENDENTLY_REVIEWED_FAMILY_BLOCK"

V7_PLAN_SHA256 = "81d123dbfa33932c5aaadf2f6829fa3cbee1c25bc9be2b1491c515604a4fc71a"
V7_PLAN_FILE_SHA256 = "746ec7900f76e9346162473dab857a489bc7c7dde1c613bb5869daf24df4103f"
V7_HUMAN_SHA256 = "435299f5218e1ed2c7e235b9d1235a4a77d0acfad063966d5f1a05b37c6aa77c"
V7_HUMAN_FILE_SHA256 = "4634ce12974582697b977dcbd83175dda49a0f01da7f9a46aebc745ae61b8e46"
V7_PREFLIGHT_SHA256 = "17c44c278012b033e427eb952c2b982816ddd743d8c6ac73504825d296dde052"
V7_PREFLIGHT_FILE_SHA256 = "0ce5a42896d2b6ce999dc4183ed9407efa7cea6af9f2adb7d235d688d927e3b1"
V7_BOUND_SHA256 = "ecda8c84db5e93ae7ff21b278beb8d33e39fe80055c9c03edae1e62914236f6e"
V7_BOUND_FILE_SHA256 = "184bb23a359263b94bd503315e369eae8f22094d7afd4351133c995fcd33a5ae"
V7_PI_GO_SHA256 = "8344a3476353e60c39246104104d7e0b5de27346c1329f21d2cfc83d1324df1c"
V7_PI_GO_FILE_SHA256 = "3f973476770a55446bae16a5dd5c5927a35484f539f705567448d10524b9b154"

V7_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v7-v6-independent-no-go-remediation"
)
V7_PLAN_PATH = f"{V7_ROOT}/plan/reasoning-effort-plan-v7-{V7_PLAN_SHA256}.json"
V7_HUMAN_PATH = (
    f"{V7_ROOT}/human-protocol/reasoning-effort-human-protocol-v7-{V7_HUMAN_SHA256}.json"
)
V7_PREFLIGHT_PATH = f"{V7_ROOT}/preflight/reasoning-effort-preflight-v7-{V7_PREFLIGHT_SHA256}.json"
V7_BOUND_PATH = (
    f"{V7_ROOT}/bound-preflight/reasoning-effort-bound-preflight-v7-{V7_BOUND_SHA256}.json"
)
V7_PI_GO_PATH = (
    f"{V7_ROOT}/governance/reasoning-effort-v7-human-pi-one-block-go-"
    f"{V7_PI_GO_SHA256}.json"
)
INCIDENT_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v8-monotone-capacity-attestation/preflight-incident"
)

OBSERVED_CONTRACT_SHA256 = {
    "deepseek": "9e14df569e70ac4f0a810504f99079b88f0b3527463e6f82fa4cf1fbb5f81270",
    "gemini": "887e66e3cef94c22cbe1ee41f38b8d3ea44c18b65118e435f1b5c40fa09a1324",
    "sonnet": "2f56d24a84c24eb794a683152169d5387e905b75c272b70fc8630ba38481453a",
}
MONOTONE_CAPACITY_FIELDS = ("context_length", "max_completion_tokens")

GLOBAL_LEDGER_PATH = v7.GLOBAL_LEDGER_PATH
GLOBAL_SOURCE_PATH = v7.GLOBAL_SOURCE_PATH
GLOBAL_LEDGER_ANCHOR_SEQUENCE = v7.GLOBAL_LEDGER_ANCHOR_SEQUENCE
GLOBAL_LEDGER_ANCHOR_HEAD_SHA256 = v7.GLOBAL_LEDGER_ANCHOR_HEAD_SHA256
GLOBAL_LEDGER_ANCHOR_FILE_SHA256 = v7.GLOBAL_LEDGER_ANCHOR_FILE_SHA256

ENDPOINTS = v7.ENDPOINTS
TASK_FAMILIES = v7.TASK_FAMILIES
CURRENT_EXPOSURE_USD = v7.CURRENT_EXPOSURE_USD
ADMISSION_CEILING_USD = v7.ADMISSION_CEILING_USD
HARD_CAP_USD = v7.HARD_CAP_USD

_sha256 = v7._sha256
_file_sha256 = v7._file_sha256
_regular_json = v7._regular_json
_decimal_text = v7._decimal_text
_relative = v7._relative
_file_ref = v7._file_ref
_write_artifact = v7._write_artifact
_exact_sum = v7._exact_sum
_exact_add = v7._exact_add
verify_manifest_content_address = v7.verify_manifest_content_address


class FullStudyError(RuntimeError):
    """A V8 identity, source, budget, or admission binding failed."""


def validate_monotone_capacity_contract(
    *, frozen: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept only non-decreasing advertised capacity and exact other semantics."""

    frozen_keys = set(map(str, frozen))
    observed_keys = set(map(str, observed))
    if frozen_keys != observed_keys:
        raise FullStudyError("endpoint semantic fields were added or removed")
    changed: list[dict[str, Any]] = []
    for key in sorted(frozen_keys):
        old = frozen[key]
        new = observed[key]
        if key in MONOTONE_CAPACITY_FIELDS:
            if (
                isinstance(old, bool)
                or isinstance(new, bool)
                or not isinstance(old, int)
                or not isinstance(new, int)
                or new < old
            ):
                raise FullStudyError(f"{key} is not a monotone integer capacity")
            if new != old:
                changed.append({"field": key, "frozen": old, "observed": new})
        elif new != old:
            raise FullStudyError(f"non-capacity endpoint semantic field changed: {key}")
    for key in (
        "model_id",
        "provider_name",
        "tag",
        "pricing",
        "pricing_normalization",
        "quantization",
        "supported_parameters",
    ):
        if observed.get(key) != frozen.get(key):
            raise FullStudyError(f"required exact endpoint field changed: {key}")
    return {
        "decision": "accepted_exact_or_monotone_capacity_only",
        "allowed_fields": list(MONOTONE_CAPACITY_FIELDS),
        "changed_fields": changed,
        "identity_provider_pricing_quantization_and_parameters_exact": True,
        "request_and_output_caps_changed": False,
    }


def build_preflight_incident(*, repo_root: Path) -> dict[str, Any]:
    """Reify the stopped V7 catalog drift observation without replaying it."""

    plan = _verified_artifact(
        repo_root, V7_PLAN_PATH, V7_PLAN_SHA256, V7_PLAN_FILE_SHA256
    )
    pi_go = _verified_artifact(
        repo_root, V7_PI_GO_PATH, V7_PI_GO_SHA256, V7_PI_GO_FILE_SHA256
    )
    contracts: dict[str, Any] = {}
    for endpoint_id in sorted(ENDPOINTS):
        frozen = copy.deepcopy(plan["models"][endpoint_id]["semantic_execution_contract"])
        observed = copy.deepcopy(frozen)
        if endpoint_id == "deepseek":
            observed["max_completion_tokens"] = 384000
        frozen_sha = _sha256(frozen)
        observed_sha = _sha256(observed)
        if (
            frozen_sha
            != plan["models"][endpoint_id]["semantic_execution_contract_sha256"]
            or observed_sha != OBSERVED_CONTRACT_SHA256[endpoint_id]
        ):
            raise FullStudyError("V7 observed endpoint contract does not rederive")
        contracts[endpoint_id] = {
            "frozen_contract": frozen,
            "frozen_contract_sha256": frozen_sha,
            "observed_contract": observed,
            "observed_contract_sha256": observed_sha,
            "diff": validate_monotone_capacity_contract(
                frozen=frozen, observed=observed
            )["changed_fields"],
        }
    ledger = repo_root / GLOBAL_LEDGER_PATH
    ledger_lines = ledger.read_text(encoding="utf-8").splitlines()
    run_root = repo_root / f"{V7_ROOT}/runs"
    run_files = (
        [path for path in run_root.rglob("*") if path.is_file()]
        if run_root.exists()
        else []
    )
    if (
        _file_sha256(ledger) != GLOBAL_LEDGER_ANCHOR_FILE_SHA256
        or len(ledger_lines) != GLOBAL_LEDGER_ANCHOR_SEQUENCE
        or run_files
    ):
        raise FullStudyError("failed V7 preflight did not preserve its zero-side-effect boundary")
    payload = {
        "schema_version": PREFLIGHT_INCIDENT_SCHEMA,
        "record_role": "append_only_v7_pre_reservation_catalog_drift_incident",
        "observed_on": "2026-08-08",
        "decision": "stopped_before_reservation_due_to_exact_contract_drift",
        "v7_bindings": {
            "study_plan_sha256": V7_PLAN_SHA256,
            "study_plan_file_sha256": V7_PLAN_FILE_SHA256,
            "human_pi_one_block_go_sha256": pi_go["artifact_sha256"],
            "confirmation_present_but_no_work_admitted": True,
        },
        "endpoint_contracts": contracts,
        "exact_observed_diff": {
            "endpoint_id": "deepseek",
            "field": "max_completion_tokens",
            "frozen": 65536,
            "observed": 384000,
            "all_other_endpoint_semantics_exact": True,
            "sonnet_changed": False,
            "gemini_changed": False,
        },
        "attempt_boundary": {
            "catalog_http_gets": 6,
            "provider_completion_requests": 0,
            "epicure_calls": 0,
            "reservations_created": 0,
            "run_files_created": 0,
            "quality_observations_created": 0,
        },
        "canonical_state_after_stop": {
            "ledger_path": GLOBAL_LEDGER_PATH,
            "ledger_file_sha256": _file_sha256(ledger),
            "ledger_entries": len(ledger_lines),
            "v7_run_root_absent": not run_root.exists(),
        },
        "interpretation": (
            "The provider advertised a larger completion capacity. No model request, "
            "Epicure request, reservation, or run-state write occurred."
        ),
        "official": False,
        "rank_eligible": False,
    }
    return {**payload, "artifact_sha256": _sha256(payload)}


def freeze_preflight_incident(*, repo_root: Path, output_dir: Path) -> Path:
    incident = build_preflight_incident(repo_root=repo_root)
    return _write_artifact(
        output_dir,
        "reasoning-effort-v7-preflight-drift-incident",
        incident,
    )


def pair_audit(
    *, plan: Mapping[str, Any], item: Mapping[str, Any], source_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Retain the V7 pair audit while accepting only the frozen monotone rule."""

    pair = v7.pair_audit(plan=plan, item=item, source_path=source_path, repo_root=repo_root)
    source = _regular_json(source_path)
    observed = semantic_endpoint_contract(
        raw_endpoint_contract(source.get("endpoint_contract") or {})
    )
    endpoint_id = str(item["route_coordinate"]["endpoint_id"])
    frozen = plan["models"][endpoint_id]["semantic_execution_contract"]
    try:
        attestation = validate_monotone_capacity_contract(
            frozen=frozen, observed=observed
        )
    except FullStudyError as error:
        attestation = {"decision": "rejected", "reason": str(error)}
    else:
        pair["failures"] = [
            value
            for value in pair.get("failures") or []
            if value
            not in {
                "source_endpoint_semantic_contract_differs_from_task_wave_freeze",
                "source_endpoint_semantic_contract_differs_from_v5_freeze",
            }
        ]
    pair["capacity_attestation_v8"] = {
        **attestation,
        "frozen_contract_sha256": _sha256(frozen),
        "observed_contract_sha256": _sha256(observed),
        "frozen_request_caps": {
            "max_intermediate_tokens": plan["common_protocol"]["max_intermediate_tokens"],
            "max_output_tokens": plan["common_protocol"]["max_output_tokens"],
        },
    }
    pair["failures"] = sorted(set(pair.get("failures") or []))
    pair["decision"] = "passed_all_predicates" if not pair["failures"] else "failed"
    return pair


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
                        "schema_version": "flavourbench-reasoning-presentation-v8",
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
            raise FullStudyError("V8 presentation stratum is not six tasks")
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
    predecessor = _verified_artifact(repo_root, V7_PLAN_PATH, V7_PLAN_SHA256, V7_PLAN_FILE_SHA256)
    _verified_artifact(repo_root, V7_HUMAN_PATH, V7_HUMAN_SHA256, V7_HUMAN_FILE_SHA256)
    _verified_artifact(repo_root, V7_PREFLIGHT_PATH, V7_PREFLIGHT_SHA256, V7_PREFLIGHT_FILE_SHA256)
    _verified_artifact(repo_root, V7_BOUND_PATH, V7_BOUND_SHA256, V7_BOUND_FILE_SHA256)
    incident_expected = build_preflight_incident(repo_root=repo_root)
    incident_path = (
        repo_root
        / INCIDENT_ROOT
        / (
            "reasoning-effort-v7-preflight-drift-incident-"
            f"{incident_expected['artifact_sha256']}.json"
        )
    )
    incident = _verified_artifact(
        repo_root,
        _relative(repo_root, incident_path),
        incident_expected["artifact_sha256"],
        _file_sha256(incident_path),
    )
    if incident.get("decision") != "stopped_before_reservation_due_to_exact_contract_drift":
        raise FullStudyError("V7 preflight incident is not the exact stopped attempt")

    old_to_new: dict[str, str] = {}
    work_items: list[dict[str, Any]] = []
    for old in predecessor["work_items"]:
        coordinate = copy.deepcopy(old["route_coordinate"])
        coordinate.update(
            {
                "schema_version": "flavourbench-reasoning-effort-task-wave-coordinate-v8",
                "freeze_nonce": FREEZE_NONCE,
                "superseded_plan_sha256": V7_PLAN_SHA256,
                "superseded_preflight_incident_sha256": incident["artifact_sha256"],
            }
        )
        route_cell_id = _sha256(coordinate)
        run_id = str(uuid.uuid5(NAMESPACE, f"{FREEZE_NONCE}:{route_cell_id}:run"))
        work_identity = {
            "schema_version": "flavourbench-reasoning-effort-work-item-identity-v8",
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
            "schema_version": "flavourbench-reasoning-effort-task-wave-v8",
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
            "schema_version": "flavourbench-reasoning-effort-family-block-v8",
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
            "retired_v7_plan": _file_ref(repo_root, repo_root / V7_PLAN_PATH),
            "retired_v7_human_protocol": _file_ref(repo_root, repo_root / V7_HUMAN_PATH),
            "retired_v7_preflight": _file_ref(repo_root, repo_root / V7_PREFLIGHT_PATH),
            "retired_v7_bound_preflight": _file_ref(repo_root, repo_root / V7_BOUND_PATH),
            "v7_preflight_drift_incident": _file_ref(repo_root, incident_path),
            "v7_human_pi_go": _file_ref(repo_root, repo_root / V7_PI_GO_PATH),
        }
    )
    task_identity = v7.v6.v5.v4.predecessor.canonical_task_wave_identity(
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
        "record_role": "v7_preflight_capacity_drift_remediated_reasoning_effort_successor",
        "status": "frozen_not_executed_independent_go_required",
        "study_id": STUDY_ID,
        "freeze_nonce": FREEZE_NONCE,
        "root_id": ROOT_ID,
        "supersedes": {
            "retired_v7_plan_sha256": V7_PLAN_SHA256,
            "retired_v7_plan_file_sha256": V7_PLAN_FILE_SHA256,
            "retired_v7_human_sha256": V7_HUMAN_SHA256,
            "retired_v7_human_file_sha256": V7_HUMAN_FILE_SHA256,
            "retired_v7_preflight_sha256": V7_PREFLIGHT_SHA256,
            "retired_v7_preflight_file_sha256": V7_PREFLIGHT_FILE_SHA256,
            "retired_v7_bound_sha256": V7_BOUND_SHA256,
            "retired_v7_bound_file_sha256": V7_BOUND_FILE_SHA256,
            "v7_preflight_incident_sha256": incident["artifact_sha256"],
            "v7_preflight_incident_file_sha256": _file_sha256(incident_path),
            "retired_v7_decision": "stopped_before_reservation",
            "retired_v7_human_pi_go_sha256": V7_PI_GO_SHA256,
            "fresh_v8_authorization_required": True,
        },
        "source_artifacts": source_artifacts,
        "work_items": work_items,
        "task_waves": waves,
        "wave_execution_order": [wave["wave_id"] for wave in waves],
        "admission_blocks": blocks,
        "block_execution_order": [block["admission_block_id"] for block in blocks],
        "task_wave_identity": task_identity,
        "endpoint_capacity_attestation_policy": {
            "rule": "exact_all_semantics_except_monotone_capacity",
            "monotone_non_decreasing_fields": list(MONOTONE_CAPACITY_FIELDS),
            "exact_fields": [
                "model_id",
                "provider_name",
                "tag",
                "pricing",
                "pricing_normalization",
                "quantization",
                "supported_parameters",
            ],
            "additional_or_missing_semantic_fields_allowed": False,
            "request_caps_frozen": {
                "max_intermediate_tokens": predecessor["common_protocol"][
                    "max_intermediate_tokens"
                ],
                "max_output_tokens": predecessor["common_protocol"]["max_output_tokens"],
            },
            "provider_advertised_capacity_does_not_change_request_caps": True,
            "v7_incident_sha256": incident["artifact_sha256"],
        },
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
            "module": "flavourbench.reasoning_effort_full_study_executor_v8",
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
        != "flavourbench.reasoning_effort_full_study_executor_v8"
        or supersedes.get("retired_v7_decision") != "stopped_before_reservation"
        or supersedes.get("fresh_v8_authorization_required") is not True
    ):
        raise FullStudyError("V8 identity or V7 stopped-preflight binding differs")
    try:
        verify_source_closure(expected=plan.get("source_code") or {}, repo_root=repo_root)
    except SourceClosureError as error:
        raise FullStudyError(f"V8 source closure does not rederive: {error}") from error
    incident = build_preflight_incident(repo_root=repo_root)
    incident_reference = plan.get("source_artifacts", {}).get(
        "v7_preflight_drift_incident"
    ) or {}
    references = {
        "retired_v7_plan": (V7_PLAN_SHA256, V7_PLAN_FILE_SHA256),
        "retired_v7_human_protocol": (V7_HUMAN_SHA256, V7_HUMAN_FILE_SHA256),
        "retired_v7_preflight": (V7_PREFLIGHT_SHA256, V7_PREFLIGHT_FILE_SHA256),
        "retired_v7_bound_preflight": (V7_BOUND_SHA256, V7_BOUND_FILE_SHA256),
        "v7_human_pi_go": (V7_PI_GO_SHA256, V7_PI_GO_FILE_SHA256),
        "v7_preflight_drift_incident": (
            incident["artifact_sha256"],
            str(incident_reference.get("file_sha256") or ""),
        ),
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
            raise FullStudyError(f"V8 predecessor source differs: {key}")
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
        raise FullStudyError("V8 estimand counts differ")
    reserves = {
        str(item["work_item_id"]): Decimal(item["worst_case_reserve_usd"]) for item in items
    }
    for collection in (waves, blocks):
        for row in collection:
            expected = _exact_sum([reserves[item_id] for item_id in row["work_item_ids"]])
            if Decimal(row["worst_case_reserve_usd"]) != expected:
                raise FullStudyError("V8 wave/block reserve is not an exact item sum")
    if any(
        len(block.get("work_item_ids") or []) != 28
        or block.get("canonical_global_reservations_required") != 28
        or Counter(block.get("task_families") or [])
        != Counter({family: 1 for family in TASK_FAMILIES})
        for block in blocks
    ):
        raise FullStudyError("V8 block structure differs")
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
        raise FullStudyError("V8 exact Decimal budget summary differs")
    expected_anchor = {
        "sequence": GLOBAL_LEDGER_ANCHOR_SEQUENCE,
        "head_entry_sha256": GLOBAL_LEDGER_ANCHOR_HEAD_SHA256,
        "physical_file_sha256_at_freeze": GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
        "baseline_exposure_usd": _decimal_text(CURRENT_EXPOSURE_USD),
    }
    if plan.get("canonical_global_ledger_anchor") != expected_anchor:
        raise FullStudyError("V8 canonical global-ledger anchor differs")
    policy = plan.get("endpoint_capacity_attestation_policy") or {}
    if (
        policy.get("monotone_non_decreasing_fields")
        != list(MONOTONE_CAPACITY_FIELDS)
        or policy.get("additional_or_missing_semantic_fields_allowed") is not False
        or policy.get("request_caps_frozen")
        != {"max_intermediate_tokens": 8192, "max_output_tokens": 8192}
        or supersedes.get("v7_preflight_incident_sha256")
        != incident["artifact_sha256"]
    ):
        raise FullStudyError("V8 monotone capacity or request-cap policy differs")
    for endpoint_id, model in plan.get("models", {}).items():
        frozen = model.get("semantic_execution_contract") or {}
        if _sha256(frozen) != model.get("semantic_execution_contract_sha256"):
            raise FullStudyError(f"V8 frozen model contract differs: {endpoint_id}")
    if plan.get("task_wave_identity") != v7.v6.v5.v4.predecessor.canonical_task_wave_identity(
        tasks=plan.get("tasks") or [], waves=waves
    ):
        raise FullStudyError("V8 task-wave identity does not rederive")
    current = _identity_sets(plan)
    historical = (
        ("V7", V7_PLAN_PATH, V7_PLAN_SHA256, V7_PLAN_FILE_SHA256),
        ("V6", v7.V6_PLAN_PATH, v7.V6_PLAN_SHA256, v7.V6_PLAN_FILE_SHA256),
        ("V5", v7.v6.V5_PLAN_PATH, v7.v6.V5_PLAN_SHA256, v7.v6.V5_PLAN_FILE_SHA256),
        ("V4", v7.v6.v5.V4_PLAN_PATH, v7.v6.v5.V4_PLAN_SHA256, None),
        ("V3", v7.v6.v5.V3_PLAN_PATH, v7.v6.v5.V3_PLAN_SHA256, None),
        ("V2", v7.v6.v5.V2_PLAN_PATH, v7.v6.v5.V2_PLAN_SHA256, None),
    )
    for label, relative, semantic, physical in historical:
        retired = _identity_sets(_verified_artifact(repo_root, relative, semantic, physical))
        for identity_type in current:
            if current[identity_type] & retired[identity_type]:
                raise FullStudyError(f"V8 reuses a {label} {identity_type}")


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
        raise FullStudyError("V8 execution roots are not empty")


def build_human_protocol(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    previous = _verified_artifact(repo_root, V7_HUMAN_PATH, V7_HUMAN_SHA256, V7_HUMAN_FILE_SHA256)
    dossier = human._verified_dossier(
        repo_root / previous["source_bindings"]["task_dossier"]["path"]
    )
    selected = human._selected_tasks(dossier, plan)
    arms, cells = human._build_graph(selected, plan)
    presentations, assignment_blocks = human._build_presentations(cells)
    protocol = {
        **copy.deepcopy(previous),
        "schema_version": HUMAN_PROTOCOL_SCHEMA,
        "study_id": f"{STUDY_ID}-human-v8",
        "supersedes": {
            "artifact_sha256": V7_HUMAN_SHA256,
            "artifact_file_sha256": V7_HUMAN_FILE_SHA256,
            "reason": "V7 stopped before reservation after monotone provider capacity drift",
            "v7_plan_sha256": V7_PLAN_SHA256,
            "v7_preflight_incident_sha256": plan["supersedes"][
                "v7_preflight_incident_sha256"
            ],
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
            "retired_v7_human_protocol": _file_ref(repo_root, repo_root / V7_HUMAN_PATH),
            "v7_preflight_drift_incident": copy.deepcopy(
                plan["source_artifacts"]["v7_preflight_drift_incident"]
            ),
        },
    }
    protocol.pop("artifact_sha256", None)
    protocol["artifact_sha256"] = _sha256(protocol)
    verify_human_protocol_binding(plan=plan, human_protocol=protocol)
    if {str(row["presentation_id"]) for row in protocol["presentations"]} & {
        str(row["presentation_id"]) for row in previous["presentations"]
    }:
        raise FullStudyError("V8 reuses a V7 assignment presentation identifier")
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
        raise FullStudyError("V8 human protocol binding differs")
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
        raise FullStudyError("V8 arm, cell, or assignment graph differs")


def build_preflight(*, plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    validate_plan(plan, repo_root=repo_root)
    assert_successor_roots_empty(plan=plan, repo_root=repo_root)
    first = Decimal(plan["admission_blocks"][0]["worst_case_reserve_usd"])
    projected = _exact_add(CURRENT_EXPOSURE_USD, first)
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "record_role": "zero_call_v7_capacity_drift_remediated_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "decision": "technical_preflight_pass_independent_go_not_supplied",
        "checks": {
            "retired_v7_plan_bound": V7_PLAN_SHA256,
            "v7_preflight_incident_bound": plan["supersedes"][
                "v7_preflight_incident_sha256"
            ],
            "successor_roots_empty": True,
            "canonical_global_ledger_anchor": plan["canonical_global_ledger_anchor"],
            "first_block_reserve_usd": _decimal_text(first),
            "first_block_projected_usd": _decimal_text(projected),
            "below_85_percent_admission": projected <= ADMISSION_CEILING_USD,
            "independent_go_required": True,
            "monotone_capacity_rule_predeclared": True,
            "request_and_output_caps_remain_8192": True,
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
        "record_role": "cross_bound_v7_capacity_drift_remediated_preflight",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "preliminary_preflight_sha256": preliminary["artifact_sha256"],
        "v7_preflight_incident_sha256": plan["supersedes"][
            "v7_preflight_incident_sha256"
        ],
        "decision": "technical_pass_different_independent_go_required_before_execution",
        "checks": {
            **preliminary["checks"],
            "human_protocol_cross_verified": True,
            "canonical_comparison_cells": 240,
            "assignment_presentations": 1584,
            "fresh_transparent_pi_authorization_required": True,
            "reviewer_must_not_be_v8_builder_or_executor": True,
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
        or bound_preflight.get("v7_preflight_incident_sha256")
        != plan["supersedes"]["v7_preflight_incident_sha256"]
        or bound_preflight.get("decision")
        != "technical_pass_different_independent_go_required_before_execution"
    ):
        raise FullStudyError("V8 bound preflight is absent or invalid")


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
        or governance_go.get("reviewer_is_v8_builder") is not False
        or governance_go.get("reviewed_v7_preflight_incident_sha256")
        != plan["supersedes"]["v7_preflight_incident_sha256"]
        or governance_go.get("authorization_is_transparent_human_pi_record") is not True
        or governance_go.get("provider_or_epicure_calls_made_by_review") is not False
        or governance_go.get("maximum_family_blocks") != 1
    ):
        raise FullStudyError("a fresh transparent human-PI exact one-block GO is required")


def freeze(*, repo_root: Path, output_dir: Path) -> dict[str, Path]:
    incident_path = freeze_preflight_incident(
        repo_root=repo_root, output_dir=output_dir / "preflight-incident"
    )
    plan = build_plan(repo_root=repo_root)
    plan_path = _write_artifact(output_dir / "plan", "reasoning-effort-plan-v8", plan)
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
        output_dir / "human-protocol", "reasoning-effort-human-protocol-v8", protocol
    )
    protocol = _regular_json(human_path)
    preflight = build_preflight(plan=plan, repo_root=repo_root)
    preflight_path = _write_artifact(
        output_dir / "preflight", "reasoning-effort-preflight-v8", preflight
    )
    bound = build_bound_preflight(plan=plan, human_protocol=protocol, repo_root=repo_root)
    bound_path = _write_artifact(
        output_dir / "bound-preflight", "reasoning-effort-bound-preflight-v8", bound
    )
    return {
        "preflight_incident": incident_path,
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
