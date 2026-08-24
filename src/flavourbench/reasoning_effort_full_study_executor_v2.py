"""Execute one import-safe successor reasoning-effort family block.

The entire 168-item runtime surface is constructed and validated before a
catalog request, lock acquisition, directory creation, ledger append, or budget
reservation.  Once an item start is durable, every subsequent failure is
classified from source/journal evidence or recorded as a durable pipeline
incident; an identifier is never replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_full_study_executor_v1 as retired_executor
from . import reasoning_effort_full_study_v2 as study

LEDGER_SCHEMA = retired_executor.LEDGER_SCHEMA
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-family-block-execution-receipt-v3"
ATTESTATION_SCHEMA = "flavourbench-reasoning-effort-family-block-attestations-v3"
OPERATION_INCIDENT_SCHEMA = "flavourbench-reasoning-effort-operation-incident-v1"

FullStudyExecutionError = retired_executor.FullStudyExecutionError
_load_ledger = retired_executor._load_ledger
_append_ledger = retired_executor._append_ledger
_ledger_lock = retired_executor._ledger_lock
_item_map = retired_executor._item_map
_wave_map = retired_executor._wave_map
_block_map = retired_executor._block_map
_item_wave_id = retired_executor._item_wave_id
_roots = retired_executor._roots
_manifest = retired_executor._manifest
_prospective_policy_from_manifest = retired_executor._prospective_policy_from_manifest
_policy = retired_executor._policy
_coordinator_state = retired_executor._coordinator_state
_endpoint_state = retired_executor._endpoint_state
_journal_evidence = retired_executor._journal_evidence
_source_for_item = retired_executor._source_for_item
_source_terminal_payload = retired_executor._source_terminal_payload
_global_state = retired_executor._global_state
_accounting = retired_executor._accounting
_append_item_terminal = retired_executor._append_item_terminal
_append_uncertain_incident = retired_executor._append_uncertain_incident
_terminalize_block = retired_executor._terminalize_block

FailureInjector = Callable[[str, str | None], None]


def _live_args(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    repo_root: Path,
    source: Path,
    raw_endpoint_sha256: str,
) -> argparse.Namespace:
    # The predecessor imported a nonexistent alias at this boundary.  Keep the
    # explicit alias local so import resolution is exercised by the all-item
    # pre-external validation pass.
    from .live_smoke import CONFIRMATION as LIVE_SMOKE_CONFIRMATION

    coordinate = item["route_coordinate"]
    policy = _policy(plan, item, repo_root)
    return argparse.Namespace(
        confirm=LIVE_SMOKE_CONFIRMATION,
        cap_usd=Decimal(item["worst_case_reserve_usd"]),
        model_id=coordinate["model_id"],
        provider_slug=coordinate["provider_endpoint"],
        prompt=item["task"]["prompt"],
        category=item["task"]["family"],
        skip_tool_contract=True,
        contract_only=False,
        condition=None,
        plain_text_final=True,
        tool_catalog_bytes_bound=policy.tool_catalog_bytes_bound,
        require_epicure_call=True,
        evidence_protocol=policy.evidence_protocol,
        intermediate_reasoning_effort=coordinate["intermediate_reasoning_effort"],
        final_reasoning_effort=coordinate["final_reasoning_effort"],
        output_dir=str(source),
        candidate_manifest_sha256=item["manifest"]["semantic_sha256"],
        sequential_arms=False,
        dataset_work_item_id=item["work_item_id"],
        dataset_task_id=item["task"]["task_id"],
        expected_canonical_model_slug=coordinate["canonical_model_slug"],
        expected_endpoint_execution_sha256=raw_endpoint_sha256,
        expected_execution_policy_sha256=policy.sha256,
        expected_epicure_release_id=plan["epicure"]["release_id"],
        expected_epicure_bundle_sha256=plan["epicure"]["bundle_sha256"],
        expected_epicure_application_sha256=plan["epicure"]["application_sha256"],
        expected_epicure_tool_schema_sha256=plan["epicure"]["tool_schema_sha256"],
        frozen_run_id=item["run_id"],
        frozen_attempt_slots=item["attempt_slots"],
    )


def _validate_live_args(
    *, plan: Mapping[str, Any], item: Mapping[str, Any], args: argparse.Namespace, policy: Any
) -> None:
    coordinate = item["route_coordinate"]
    if (
        args.dataset_work_item_id != item["work_item_id"]
        or args.frozen_run_id != item["run_id"]
        or args.frozen_attempt_slots != item["attempt_slots"]
        or args.model_id != coordinate["model_id"]
        or args.provider_slug != coordinate["provider_endpoint"]
        or args.expected_canonical_model_slug != coordinate["canonical_model_slug"]
        or args.expected_execution_policy_sha256 != coordinate["execution_policy_sha256"]
        or args.intermediate_reasoning_effort
        != coordinate["intermediate_reasoning_effort"]
        or args.final_reasoning_effort != coordinate["final_reasoning_effort"]
        or args.evidence_protocol != "matched_evidence_v2"
        or args.sequential_arms is not False
        or args.require_epicure_call is not True
        or args.plain_text_final is not True
        or args.cap_usd != Decimal(item["worst_case_reserve_usd"])
        or policy.pair_arm_scheduling != "concurrent"
        or policy.max_tool_calls_total != 13
        or policy.max_tool_calls_per_round != 13
        or policy.max_tool_rounds != 3
        or policy.max_provider_attempts != 2
        or len(item.get("attempt_slots") or []) != 56
        or len(str(args.expected_endpoint_execution_sha256)) != 64
    ):
        raise FullStudyExecutionError(
            f"runtime arguments differ for work item {item['work_item_id']}"
        )
    if (
        args.expected_epicure_release_id != plan["epicure"]["release_id"]
        or args.expected_epicure_bundle_sha256 != plan["epicure"]["bundle_sha256"]
        or args.expected_epicure_application_sha256
        != plan["epicure"]["application_sha256"]
        or args.expected_epicure_tool_schema_sha256
        != plan["epicure"]["tool_schema_sha256"]
    ):
        raise FullStudyExecutionError("runtime Epicure identity differs")


def prepare_all_runtime_items(
    *, plan: Mapping[str, Any], repo_root: Path
) -> dict[str, tuple[Any, argparse.Namespace]]:
    """Construct all policies and live args without creating or contacting anything."""

    items = _item_map(plan)
    _, endpoint_roots = _roots(plan, repo_root)
    prepared: dict[str, tuple[Any, argparse.Namespace]] = {}
    for item_id in sorted(items):
        item = items[item_id]
        endpoint_id = str(item["route_coordinate"]["endpoint_id"])
        policy = _policy(plan, item, repo_root)
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source=endpoint_roots[endpoint_id] / "source",
            raw_endpoint_sha256="0" * 64,
        )
        _validate_live_args(plan=plan, item=item, args=args, policy=policy)
        prepared[item_id] = (policy, args)
    if len(prepared) != 168:
        raise FullStudyExecutionError("all-item runtime preparation did not cover 168 items")
    return prepared


async def _attest_all_endpoints(
    *, plan: Mapping[str, Any], api_base: str, api_key: str
) -> list[dict[str, Any]]:
    return await retired_executor._attest_all_endpoints(
        plan=plan, api_base=api_base, api_key=api_key
    )


def _require_live_environment_before_reservation() -> None:
    from .reasoning_effort_route_gate_v4 import (
        _require_live_environment_before_reservation as require,
    )

    require()


async def _invoke_live_pair(
    *, args: argparse.Namespace, policy: Any, endpoint: Mapping[str, Any]
) -> None:
    from .config import get_settings
    from .frontier_contract_runner import AdmissionDenied
    from .live_smoke import live_smoke
    from .reasoning_effort_route_gate_v4 import _policy_environment

    with _policy_environment(policy=policy, endpoint=endpoint):
        settings = get_settings()
        if settings.execution_mode != "live" or not settings.live_authorized:
            raise AdmissionDenied("live authority changed after block reservation")
        await live_smoke(args)


def _inject(
    failure_injector: FailureInjector | None, stage: str, item_id: str | None = None
) -> None:
    if failure_injector is not None:
        failure_injector(stage, item_id)


def _error_record(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
    }


def _write_operation_incident(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    reservation: Mapping[str, Any],
    coordinator_root: Path,
    coordinator_ledger: Path,
    repo_root: Path,
    stage: str,
    error: BaseException,
    item: Mapping[str, Any] | None,
    journal_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entries = _load_ledger(coordinator_ledger, role="coordinator")
    payload = {
        "schema_version": OPERATION_INCIDENT_SCHEMA,
        "record_role": "durable_post_reservation_pipeline_incident",
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "block_reservation_entry_sha256": reservation["entry_sha256"],
        "stage": stage,
        "work_item_id": item.get("work_item_id") if item else None,
        "journal_evidence": dict(journal_evidence or {}),
        **_error_record(error),
        "coordinator_ledger_head_entry_sha256": (
            entries[-1]["entry_sha256"] if entries else None
        ),
        "full_block_reserve_retained_usd": block["worst_case_reserve_usd"],
        "replay_permitted": False,
        "provider_substitution_performed": False,
        "rank_eligible": False,
    }
    path = study._write_artifact(
        coordinator_root / "operation-incidents",
        "reasoning-effort-operation-incident",
        payload,
    )
    return study._file_ref(repo_root, path)


def _append_block_pipeline_incident(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    reservation: Mapping[str, Any],
    coordinator_ledger: Path,
    stage: str,
    error: BaseException,
    item: Mapping[str, Any] | None,
    journal_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_execution_incident",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            "incident": "successor_pipeline_operation_failure",
            "stage": stage,
            "work_item_id": item.get("work_item_id") if item else None,
            "journal_evidence": dict(journal_evidence or {}),
            **_error_record(error),
            "full_family_block_reserve_retained_usd": block["worst_case_reserve_usd"],
            "replay_permitted": False,
        },
    )


def _durable_block_failure(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    reservation: Mapping[str, Any],
    coordinator_root: Path,
    coordinator_ledger: Path,
    repo_root: Path,
    stage: str,
    error: BaseException,
    item: Mapping[str, Any] | None = None,
    journal_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger_event = _append_block_pipeline_incident(
        plan=plan,
        block=block,
        reservation=reservation,
        coordinator_ledger=coordinator_ledger,
        stage=stage,
        error=error,
        item=item,
        journal_evidence=journal_evidence,
    )
    incident_ref = _write_operation_incident(
        plan=plan,
        block=block,
        reservation=reservation,
        coordinator_root=coordinator_root,
        coordinator_ledger=coordinator_ledger,
        repo_root=repo_root,
        stage=stage,
        error=error,
        item=item,
        journal_evidence=journal_evidence,
    )
    return {
        "decision": "durable_pipeline_incident_stop",
        "stage": stage,
        "work_item_id": item.get("work_item_id") if item else None,
        "coordinator_incident_entry_sha256": ledger_event["entry_sha256"],
        "operation_incident": incident_ref,
    }


def _pre_generation_payload(
    *, evidence: Mapping[str, Any], error: BaseException | None
) -> dict[str, Any]:
    text = str(error) if error is not None else "no source after pair invocation"
    return {
        "disposition": "pre_generation_failure_zero_cost",
        "actual_cost_usd": "0",
        "journal_evidence": dict(evidence),
        "error_type": type(error).__name__ if error is not None else "MissingSource",
        "error_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _classify_started_item(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    repo_root: Path,
    coordinator_root: Path,
    coordinator_ledger: Path,
    endpoint_root: Path,
    endpoint_ledger: Path,
    invocation_error: BaseException | None,
) -> dict[str, Any]:
    """Classify a durable item start without ever replaying its identifier."""

    item_id = str(item["work_item_id"])
    stage = "post_start_source_and_journal_classification"
    try:
        source = _source_for_item(endpoint_root, item_id)
        if source is not None:
            payload = _source_terminal_payload(
                plan=plan,
                item=item,
                source_record=source,
                repo_root=repo_root,
                endpoint_root=endpoint_root,
            )
            _append_item_terminal(
                plan=plan,
                block=block,
                item=item,
                reservation=reservation,
                endpoint_ledger=endpoint_ledger,
                coordinator_ledger=coordinator_ledger,
                payload=payload,
            )
            return {
                "work_item_id": item_id,
                "decision": payload["disposition"],
                "error_type": type(invocation_error).__name__ if invocation_error else None,
            }
        evidence = _journal_evidence(endpoint_root / "source", item_id)
        if evidence["request_started_count"]:
            incident = invocation_error or FullStudyExecutionError(
                "provider request started without a reconciled source"
            )
            entry = _append_uncertain_incident(
                plan=plan,
                block=block,
                item=item,
                reservation=reservation,
                endpoint_ledger=endpoint_ledger,
                coordinator_ledger=coordinator_ledger,
                evidence=evidence,
                error=incident,
            )
            return {
                "work_item_id": item_id,
                "decision": "request_started_no_source_stop",
                "incident_entry_sha256": entry["entry_sha256"],
            }
        payload = _pre_generation_payload(evidence=evidence, error=invocation_error)
        _append_item_terminal(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            payload=payload,
        )
        return {
            "work_item_id": item_id,
            "decision": "pre_generation_failure_zero_cost",
        }
    except BaseException as error:
        evidence: dict[str, Any]
        try:
            evidence = _journal_evidence(endpoint_root / "source", item_id)
        except BaseException as evidence_error:
            evidence = {
                "journal_count": None,
                "request_started_count": None,
                "journals": [],
                "evidence_error": _error_record(evidence_error),
            }
        return _durable_block_failure(
            plan=plan,
            block=block,
            reservation=reservation,
            coordinator_root=coordinator_root,
            coordinator_ledger=coordinator_ledger,
            repo_root=repo_root,
            stage=stage,
            error=error,
            item=item,
            journal_evidence=evidence,
        )


def _recover_started_prefix(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    reservation: Mapping[str, Any],
    repo_root: Path,
    coordinator_root: Path,
    coordinator_ledger: Path,
    endpoint_roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    state = _coordinator_state(
        plan, _load_ledger(coordinator_ledger, role="coordinator")
    )
    items = _item_map(plan)
    for item_id in block["work_item_ids"]:
        if item_id in state["terminals"]:
            continue
        item = items[item_id]
        endpoint_id = str(item["route_coordinate"]["endpoint_id"])
        endpoint_root = endpoint_roots[endpoint_id]
        endpoint_ledger = endpoint_root / "ledger.jsonl"
        endpoint_state = _endpoint_state(
            _load_ledger(endpoint_ledger, role="endpoint")
        )
        if item_id not in endpoint_state["started"]:
            continue
        outcome = _classify_started_item(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            repo_root=repo_root,
            coordinator_root=coordinator_root,
            coordinator_ledger=coordinator_ledger,
            endpoint_root=endpoint_root,
            endpoint_ledger=endpoint_ledger,
            invocation_error=FullStudyExecutionError("recovered durable item start"),
        )
        outcomes.append(outcome)
        if outcome["decision"] in {
            "request_started_no_source_stop",
            "durable_pipeline_incident_stop",
        }:
            break
        state = _coordinator_state(
            plan, _load_ledger(coordinator_ledger, role="coordinator")
        )
    return outcomes


async def execute_one_block(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    repo_root: Path,
    api_base: str,
    api_key: str,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    from .frontier_contract_runner import AdmissionDenied, _exclusive_runner_lock

    # These operations are read-only and network-free.  In particular,
    # prepare_all_runtime_items resolves the live-smoke confirmation alias and
    # materializes every policy/Namespace before the first durable side effect.
    study.validate_plan(plan, repo_root=repo_root)
    study.verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    study.verify_bound_preflight(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
    )
    prepared = prepare_all_runtime_items(plan=plan, repo_root=repo_root)
    _inject(failure_injector, "after_all_runtime_validation_before_any_side_effect")

    _require_live_environment_before_reservation()
    coordinator_root, endpoint_roots = _roots(plan, repo_root)
    coordinator_ledger = coordinator_root / "ledger.jsonl"
    global_ledger = repo_root / "flavourbench/artifacts/frontier-contract/ledger.jsonl"
    item_map = _item_map(plan)
    block_map = _block_map(plan)
    outcomes: list[dict[str, Any]] = []
    attest_ref: dict[str, Any] | None = None
    new_block_admitted = False
    provider_pair_invocations = 0
    block: Mapping[str, Any] | None = None
    reservation: Mapping[str, Any] | None = None

    with _exclusive_runner_lock(global_ledger):
        with _ledger_lock(coordinator_ledger):
            entries = _load_ledger(coordinator_ledger, role="coordinator")
            state = _coordinator_state(plan, entries)
            accounting = _accounting(
                plan=plan,
                repo_root=repo_root,
                coordinator_ledger=coordinator_ledger,
                endpoint_roots=endpoint_roots,
            )
            active_id = state["active_block_id"]
            if active_id and state["incidents"].get(active_id):
                block = block_map[active_id]
                outcomes.append(
                    {
                        "decision": "blocked_active_block_incident",
                        "admission_block_id": active_id,
                    }
                )
            elif active_id:
                if accounting["active_block_resume_allowed"] is not True:
                    raise AdmissionDenied("active successor block is not safe to recover")
                block = block_map[active_id]
                reservation = state["reservations"][active_id]
                outcomes.extend(
                    _recover_started_prefix(
                        plan=plan,
                        block=block,
                        reservation=reservation,
                        repo_root=repo_root,
                        coordinator_root=coordinator_root,
                        coordinator_ledger=coordinator_ledger,
                        endpoint_roots=endpoint_roots,
                    )
                )
            elif accounting["new_block_admission_allowed"] is True:
                block = block_map[str(accounting["next_block_id"])]
            else:
                outcomes.append({"decision": "no_new_block_admitted", "budget": accounting})

            state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            if block is not None and not state["incidents"].get(block["admission_block_id"]):
                block_id = str(block["admission_block_id"])
                remaining = [
                    item_id
                    for item_id in block["work_item_ids"]
                    if item_id not in state["terminals"]
                ]
                if remaining:
                    _inject(failure_injector, "before_catalog_attestation")
                    attestations = await _attest_all_endpoints(
                        plan=plan, api_base=api_base, api_key=api_key
                    )
                    _inject(
                        failure_injector,
                        "after_catalog_attestation_before_attestation_artifact",
                    )
                    attestation_document = {
                        "schema_version": ATTESTATION_SCHEMA,
                        "record_role": "successor_all_endpoint_pre_block_attestation",
                        "study_plan_sha256": plan["artifact_sha256"],
                        "admission_block_id": block_id,
                        "wave_ids": block["wave_ids"],
                        "records": attestations,
                        "counts": {
                            "catalog_http_gets": 6,
                            "provider_completion_requests": 0,
                            "epicure_calls": 0,
                        },
                        "all_168_runtime_arguments_prevalidated": True,
                        "provider_substitution_performed": False,
                    }
                    attestation_path = study._write_artifact(
                        coordinator_root / "endpoint-attestations",
                        f"reasoning-effort-successor-block-{block['block_ordinal']:02d}-attestations",
                        attestation_document,
                    )
                    attest_ref = study._file_ref(repo_root, attestation_path)
                    _inject(failure_injector, "after_attestation_before_reservation")
                    if reservation is None:
                        accounting = _accounting(
                            plan=plan,
                            repo_root=repo_root,
                            coordinator_ledger=coordinator_ledger,
                            endpoint_roots=endpoint_roots,
                        )
                        if (
                            accounting["new_block_admission_allowed"] is not True
                            or accounting["next_block_id"] != block_id
                        ):
                            raise AdmissionDenied(
                                "successor budget changed before whole-block reservation"
                            )
                        reservation = _append_ledger(
                            coordinator_ledger,
                            role="coordinator",
                            event={
                                "event_type": "family_block_reservation_created",
                                "study_plan_sha256": plan["artifact_sha256"],
                                "admission_block_id": block_id,
                                "block_ordinal": block["block_ordinal"],
                                "wave_ids": block["wave_ids"],
                                "task_ids": block["task_ids"],
                                "task_families": block["task_families"],
                                "work_item_ids": block["work_item_ids"],
                                "reserved_usd": block["worst_case_reserve_usd"],
                                "total_exposure_before_usd": accounting[
                                    "current_total_exposure_usd"
                                ],
                                "projected_with_block_usd": accounting[
                                    "next_block_projected_total_usd"
                                ],
                                "endpoint_attestation": attest_ref,
                                "reservation_unit": (
                                    "one_complete_four_task_family_balanced_block"
                                ),
                                "pair_reservations_created": 0,
                                "replay_permitted": False,
                            },
                        )
                        new_block_admitted = True

                    try:
                        _inject(failure_injector, "after_reservation_before_first_item_start")
                    except BaseException as error:
                        outcomes.append(
                            _durable_block_failure(
                                plan=plan,
                                block=block,
                                reservation=reservation,
                                coordinator_root=coordinator_root,
                                coordinator_ledger=coordinator_ledger,
                                repo_root=repo_root,
                                stage="after_reservation_before_first_item_start",
                                error=error,
                            )
                        )
                    else:
                        attestation_by_endpoint = {
                            str(record["endpoint_id"]): record for record in attestations
                        }
                        for item_id in remaining:
                            state = _coordinator_state(
                                plan,
                                _load_ledger(coordinator_ledger, role="coordinator"),
                            )
                            if state["incidents"].get(block_id):
                                break
                            item = item_map[item_id]
                            endpoint_id = str(item["route_coordinate"]["endpoint_id"])
                            endpoint_root = endpoint_roots[endpoint_id]
                            endpoint_root.mkdir(parents=True, exist_ok=True)
                            source_root = endpoint_root / "source"
                            source_root.mkdir(parents=True, exist_ok=True)
                            endpoint_ledger = endpoint_root / "ledger.jsonl"
                            endpoint_state = _endpoint_state(
                                _load_ledger(endpoint_ledger, role="endpoint")
                            )
                            if item_id in endpoint_state["started"]:
                                raise FullStudyExecutionError(
                                    "successor item reached a replay loop"
                                )
                            _inject(failure_injector, "before_item_start", item_id)
                            _append_ledger(
                                endpoint_ledger,
                                role="endpoint",
                                event={
                                    "event_type": "item_execution_started",
                                    "study_plan_sha256": plan["artifact_sha256"],
                                    "admission_block_id": block_id,
                                    "task_wave_id": _item_wave_id(plan, item_id),
                                    "work_item_id": item_id,
                                    "run_id": item["run_id"],
                                    "endpoint_id": endpoint_id,
                                    "variant_id": item["route_coordinate"]["variant_id"],
                                    "block_reservation_entry_sha256": reservation[
                                        "entry_sha256"
                                    ],
                                    "raw_endpoint_execution_contract_sha256": (
                                        attestation_by_endpoint[endpoint_id][
                                            "raw_execution_contract_sha256"
                                        ]
                                    ),
                                    "replay_permitted": False,
                                },
                            )
                            policy, template = prepared[item_id]
                            args = copy.copy(template)
                            args.expected_endpoint_execution_sha256 = (
                                attestation_by_endpoint[endpoint_id][
                                    "raw_execution_contract_sha256"
                                ]
                            )
                            invocation_error: BaseException | None = None
                            try:
                                _inject(
                                    failure_injector,
                                    "after_item_start_before_provider_request",
                                    item_id,
                                )
                                provider_pair_invocations += 1
                                await _invoke_live_pair(
                                    args=args,
                                    policy=policy,
                                    endpoint=attestation_by_endpoint[endpoint_id][
                                        "raw_execution_contract"
                                    ],
                                )
                                _inject(
                                    failure_injector,
                                    "after_provider_invocation_before_classification",
                                    item_id,
                                )
                            except BaseException as error:
                                invocation_error = error
                            outcome = _classify_started_item(
                                plan=plan,
                                block=block,
                                item=item,
                                reservation=reservation,
                                repo_root=repo_root,
                                coordinator_root=coordinator_root,
                                coordinator_ledger=coordinator_ledger,
                                endpoint_root=endpoint_root,
                                endpoint_ledger=endpoint_ledger,
                                invocation_error=invocation_error,
                            )
                            outcomes.append(outcome)
                            if outcome["decision"] in {
                                "request_started_no_source_stop",
                                "durable_pipeline_incident_stop",
                            }:
                                break

                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                if (
                    reservation is not None
                    and not state["incidents"].get(block["admission_block_id"])
                    and all(
                        item_id in state["terminals"]
                        for item_id in block["work_item_ids"]
                    )
                ):
                    _terminalize_block(
                        plan=plan,
                        block=block,
                        coordinator_ledger=coordinator_ledger,
                    )

            final_entries = _load_ledger(coordinator_ledger, role="coordinator")
            final_state = _coordinator_state(plan, final_entries)
            final_accounting = _accounting(
                plan=plan,
                repo_root=repo_root,
                coordinator_ledger=coordinator_ledger,
                endpoint_roots=endpoint_roots,
            )

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "single_import_safe_atomic_family_block_execution_receipt",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "admission_block_id": block["admission_block_id"] if block else None,
        "block_ordinal": block["block_ordinal"] if block else None,
        "new_block_admitted": new_block_admitted,
        "all_runtime_items_prevalidated": len(prepared),
        "provider_pair_invocations": provider_pair_invocations,
        "outcomes": outcomes,
        "endpoint_attestation": attest_ref,
        "final_budget": final_accounting,
        "coordinator_ledger": {
            "path": (
                f"{str(plan['execution_roots']['coordinator']).rstrip('/')}"
                "/ledger.jsonl"
            ),
            "entry_count": len(final_entries),
            "head_entry_sha256": final_entries[-1]["entry_sha256"] if final_entries else None,
            "file_sha256": (
                study._file_sha256(coordinator_ledger)
                if coordinator_ledger.exists()
                else hashlib.sha256(b"").hexdigest()
            ),
        },
        "completed_family_blocks": len(final_state["completed"]),
        "completed_task_waves": len(final_state["completed"]) * 4,
        "active_block_id": final_state["active_block_id"],
        "same_identifier_replay_permitted": False,
        "provider_substitution_performed": False,
        "rank_eligible": False,
    }
    path = study._write_artifact(
        coordinator_root / "receipts",
        "reasoning-effort-successor-family-block-execution-receipt",
        receipt,
    )
    return {"path": str(path), "document": study._regular_json(path)}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--bound-preflight", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--max-new-family-blocks", type=int, default=1)
    parser.add_argument("--api-base", default="")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    from .config import get_settings

    args = _parser().parse_args(argv)
    if args.confirm != study.CONFIRMATION:
        raise SystemExit("exact successor one-family-block confirmation token is required")
    if args.max_new_family_blocks != 1:
        raise SystemExit("--max-new-family-blocks must be exactly 1")
    repo_root = args.repo_root.resolve()
    plan = study._regular_json(args.plan)
    human_protocol = study._regular_json(args.human_protocol)
    bound_preflight = study._regular_json(args.bound_preflight)
    settings = get_settings()
    result = asyncio.run(
        execute_one_block(
            plan=plan,
            human_protocol=human_protocol,
            bound_preflight=bound_preflight,
            repo_root=repo_root,
            api_base=args.api_base or settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
