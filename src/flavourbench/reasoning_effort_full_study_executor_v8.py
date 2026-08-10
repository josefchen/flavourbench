"""Execute one independently authorised V8 reasoning-effort block.

V8 rejects stale endpoint-incident disposition on every replay path and uses a
coordinator-only no-delivery transition for normal failures before a durable
``item_execution_started`` event.  It is inert without the exact V8
confirmation and a different independent reviewer's content-addressed GO.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_full_study_executor_v7 as v7
from . import reasoning_effort_full_study_v8 as study

frontier = v7.frontier
local = v7.local

RECEIPT_SCHEMA = "flavourbench-reasoning-effort-family-block-receipt-v8"
ATTESTATION_SCHEMA = "flavourbench-reasoning-effort-block-attestations-v8"

FullStudyExecutionError = v7.FullStudyExecutionError
SimulatedCrash = v7.SimulatedCrash
MissingCanonicalSource = v7.MissingCanonicalSource
ExecutionAdapters = v7.ExecutionAdapters
BoundRuntime = v7.BoundRuntime
CanonicalDisposition = v7.CanonicalDisposition
FailureInjector = v7.FailureInjector

_load_ledger = v7._load_ledger
_append_ledger = v7._append_ledger
_ledger_lock = v7._ledger_lock
_item_map = v7._item_map
_block_map = v7._block_map
_item_wave_id = v7._item_wave_id
_roots = v7._roots
_policy = v7._policy
_endpoint_state = v7._endpoint_state
_journal_evidence = v7._journal_evidence
_terminalize_block = v7._terminalize_block
_inject = v7._inject
_error_record = v7._error_record
_live_args = v7._live_args
_validate_live_args = v7._validate_live_args
prepare_all_runtime_items = v7.prepare_all_runtime_items
_require_live_environment_before_reservation = v7._require_live_environment_before_reservation
_invoke_live_pair = v7._invoke_live_pair
_verified_source_index = v7._verified_source_index
_canonical_source_for_item = v7._canonical_source_for_item
_coordinator_state = v7._coordinator_state
_verify_global_anchor = v7._verify_global_anchor
_global_accounting_locked = v7._global_accounting_locked
_terminal_common = v7._terminal_common
_coordinator_terminal_from_endpoint = v7._coordinator_terminal_from_endpoint
_append_local_terminal_idempotent = v7._append_local_terminal_idempotent
_validate_endpoint_incident_identity = v7._validate_endpoint_incident_identity
_safe_journal_evidence = v7._safe_journal_evidence
_outcome_from_terminal = v7._outcome_from_terminal
_LEDGER_PROTECTED = v7._LEDGER_PROTECTED


async def _attest_all_endpoints(
    *, plan: Mapping[str, Any], api_base: str, api_key: str
) -> list[dict[str, Any]]:
    """Read current contracts without applying V7's retired exact-capacity gate."""

    records = await asyncio.gather(
        *(
            local._attest_endpoint(
                api_base=api_base, api_key=api_key, endpoint_id=endpoint_id
            )
            for endpoint_id in study.ENDPOINTS
        )
    )
    _validated_attestations(plan=plan, attestations=records)
    return sorted(records, key=lambda value: str(value["endpoint_id"]))


def _validated_attestations(
    *, plan: Mapping[str, Any], attestations: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any], str]]:
    by_endpoint: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], str]] = {}
    for wrapper in attestations:
        endpoint_id = str(wrapper.get("endpoint_id") or "")
        raw = wrapper.get("raw_execution_contract")
        raw_sha = str(wrapper.get("raw_execution_contract_sha256") or "")
        if (
            endpoint_id in by_endpoint
            or endpoint_id not in study.ENDPOINTS
            or not isinstance(raw, Mapping)
            or not re.fullmatch(r"[0-9a-f]{64}", raw_sha)
            or study._sha256(raw) != raw_sha
        ):
            raise FullStudyExecutionError("endpoint attestation hash binding differs")
        model = plan["models"][endpoint_id]
        observed_semantic = study.semantic_endpoint_contract(raw)
        if (
            wrapper.get("requested_model_id") != model["model_id"]
            or wrapper.get("requested_provider_endpoint") != model["provider_endpoint"]
            or (wrapper.get("model") or {}).get("canonical_slug")
            != model["canonical_model_slug"]
            or wrapper.get("semantic_execution_contract") != observed_semantic
            or wrapper.get("semantic_execution_contract_sha256")
            != study._sha256(observed_semantic)
        ):
            raise FullStudyExecutionError("endpoint identity or semantic derivation differs")
        try:
            study.validate_monotone_capacity_contract(
                frozen=model["semantic_execution_contract"],
                observed=observed_semantic,
            )
        except study.FullStudyError as error:
            raise FullStudyExecutionError(
                f"{endpoint_id} violates V8 monotone capacity attestation: {error}"
            ) from error
        by_endpoint[endpoint_id] = (dict(wrapper), dict(raw), raw_sha)
    if set(by_endpoint) != set(study.ENDPOINTS):
        raise FullStudyExecutionError("attestations do not cover exactly three endpoints")
    return by_endpoint


def _bind_block_runtime_after_attestation(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    attestations: Sequence[Mapping[str, Any]],
    prepared_all: Mapping[str, tuple[Any, argparse.Namespace]],
    repo_root: Path,
    source_root: Path,
) -> dict[str, BoundRuntime]:
    by_endpoint = _validated_attestations(plan=plan, attestations=attestations)
    items = _item_map(plan)
    rebound: dict[str, BoundRuntime] = {}
    for item_id in block["work_item_ids"]:
        item = items[item_id]
        policy = prepared_all[item_id][0]
        endpoint_id = str(item["route_coordinate"]["endpoint_id"])
        wrapper, raw, raw_sha = by_endpoint[endpoint_id]
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source_root=source_root,
            raw_endpoint_sha256=raw_sha,
        )
        args.expected_execution_policy_sha256 = policy.sha256
        _validate_live_args(
            plan=plan,
            item=item,
            args=args,
            policy=policy,
            source_root=source_root,
        )
        if (
            policy.max_output_tokens != plan["common_protocol"]["max_output_tokens"]
            or policy.max_intermediate_tokens
            != plan["common_protocol"]["max_intermediate_tokens"]
        ):
            raise FullStudyExecutionError("provider capacity drift changed request caps")
        rebound[item_id] = BoundRuntime(
            policy=policy,
            args=args,
            endpoint_wrapper=wrapper,
            raw_execution_contract=raw,
            raw_execution_contract_sha256=raw_sha,
        )
    if list(rebound) != list(block["work_item_ids"]):
        raise FullStudyExecutionError("full block runtime binding order differs")
    return rebound


def _global_ledger_path(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(plan["execution_roots"]["canonical_global_reservation_ledger"])


def _canonical_source_root(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(plan["execution_roots"]["canonical_global_source"])


def _canonical_reservation_identity(
    *, plan: Mapping[str, Any], block: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    coordinate = item["route_coordinate"]
    return {
        "event_type": "reservation_created",
        "runner_run_id": item["run_id"],
        "manifest_sha256": item["manifest"]["semantic_sha256"],
        "model_id": coordinate["model_id"],
        "provider_tag": coordinate["provider_endpoint"],
        "reserved_usd": study._decimal_text(Decimal(item["worst_case_reserve_usd"])),
        "campaign_id": study.STUDY_ID,
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "work_item_id": item["work_item_id"],
        "reservation_role": "reasoning_effort_work_item_pair",
        "response_arms": 2,
        "replay_permitted": False,
    }


def _campaign_global_reservations(
    *, plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    items = _item_map(plan)
    blocks = _block_map(plan)
    item_to_block = {
        item_id: block_id
        for block_id, block in blocks.items()
        for item_id in block["work_item_ids"]
    }
    found: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if raw.get("campaign_id") != study.STUDY_ID:
            continue
        if raw.get("event_type") != "reservation_created":
            continue
        item_id = str(raw.get("work_item_id") or "")
        if (
            raw.get("study_plan_sha256") != plan["artifact_sha256"]
            or item_id not in items
            or item_id in found
        ):
            raise FullStudyExecutionError("forged or duplicate canonical reservation")
        block_id = item_to_block[item_id]
        expected = _canonical_reservation_identity(
            plan=plan, block=blocks[block_id], item=items[item_id]
        )
        if any(raw.get(key) != value for key, value in expected.items()):
            raise FullStudyExecutionError("canonical reservation identity differs")
        found[item_id] = dict(raw)
    return found


def _ensure_canonical_reservations(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    repo_root: Path,
    global_ledger: Path,
    source_root: Path | None,
    failure_injector: FailureInjector | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accounting = _global_accounting_locked(
        plan=plan,
        repo_root=repo_root,
        global_ledger=global_ledger,
        source_root=source_root,
    )
    found = _campaign_global_reservations(plan=plan, entries=accounting["entries"])
    items = _item_map(plan)
    missing = [item_id for item_id in block["work_item_ids"] if item_id not in found]
    missing_reserve = study._exact_sum(
        [Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in missing]
    )
    projected = study._exact_add(Decimal(accounting["current_total_exposure_usd"]), missing_reserve)
    foreign_incidents = [
        row for row in accounting["active_incidents"] if row.get("campaign_id") != study.STUDY_ID
    ]
    if (
        foreign_incidents
        or projected > study.ADMISSION_CEILING_USD
        or projected > study.HARD_CAP_USD
    ):
        raise frontier.AdmissionDenied("canonical shared-ledger admission is blocked")
    for item_id in missing:
        _inject(failure_injector, "before_global_reservation", item_id)
        entry = frontier.append_ledger_event(
            global_ledger,
            _canonical_reservation_identity(plan=plan, block=block, item=items[item_id]),
        )
        found[item_id] = entry
        _inject(failure_injector, "after_global_reservation", item_id)
    refreshed = _global_accounting_locked(
        plan=plan,
        repo_root=repo_root,
        global_ledger=global_ledger,
        source_root=source_root,
    )
    found = _campaign_global_reservations(plan=plan, entries=refreshed["entries"])
    ordered = [found[item_id] for item_id in block["work_item_ids"]]
    if len(ordered) != 28 or len({row["entry_sha256"] for row in ordered}) != 28:
        raise FullStudyExecutionError("canonical block reservations are incomplete")
    return ordered, refreshed


def _verify_local_global_binding(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    global_entries: Sequence[Mapping[str, Any]],
) -> None:
    found = _campaign_global_reservations(plan=plan, entries=global_entries)
    mapping = local_reservation.get("canonical_reservation_entry_sha256_by_work_item")
    if not isinstance(mapping, Mapping):
        raise FullStudyExecutionError("local reservation has no global mapping")
    expected = {item_id: found[item_id]["entry_sha256"] for item_id in block["work_item_ids"]}
    if dict(mapping) != expected or local_reservation.get(
        "canonical_reservation_entry_sha256s"
    ) != list(expected.values()):
        raise FullStudyExecutionError("local/global reservation binding differs")


def _canonical_disposition(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    global_ledger: Path,
) -> CanonicalDisposition:
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    expected = _canonical_reservation_identity(plan=plan, block=block, item=item)
    if any(canonical_reservation.get(key) != value for key, value in expected.items()):
        raise FullStudyExecutionError("canonical reservation identity differs at recovery")
    reservations = [
        row
        for row in entries
        if row.get("entry_sha256") == canonical_reservation.get("entry_sha256")
    ]
    if len(reservations) != 1:
        raise FullStudyExecutionError("canonical reservation is absent or ambiguous")
    artifacts = [
        row
        for row in entries
        if row.get("event_type") == "artifact_recorded"
        and row.get("reservation_entry_sha256") == canonical_reservation.get("entry_sha256")
    ]
    if len(artifacts) > 1:
        raise FullStudyExecutionError("canonical artifact finalization is ambiguous")
    if artifacts:
        event = artifacts[0]
        if (
            event.get("campaign_id") != study.STUDY_ID
            or event.get("study_plan_sha256") != plan["artifact_sha256"]
            or event.get("admission_block_id") != block["admission_block_id"]
            or event.get("work_item_id") != item["work_item_id"]
        ):
            raise FullStudyExecutionError("canonical artifact finalization identity differs")
        return CanonicalDisposition("artifact_recorded", reservations[0], event)
    active = frontier.active_ledger_reservations(entries)
    if canonical_reservation["entry_sha256"] not in active:
        raise FullStudyExecutionError("canonical reservation disposition is unexplained")
    return CanonicalDisposition("active_reservation", reservations[0], None)


def _source_terminal_payload(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    source_record: tuple[Path, dict[str, Any], str],
    repo_root: Path,
    endpoint_root: Path,
) -> dict[str, Any]:
    path, artifact, digest = source_record
    verified, verified_digest = frontier._verify_live_artifact(path)
    coordinate = item["route_coordinate"]
    if (
        verified_digest != digest
        or verified != artifact
        or artifact.get("run_id") != item["run_id"]
        or artifact.get("dataset_work_item_id") != item["work_item_id"]
        or artifact.get("requested_model_id") != coordinate["model_id"]
        or artifact.get("requested_provider") != coordinate["provider_endpoint"]
        or artifact.get("candidate_manifest_sha256") != item["manifest"]["semantic_sha256"]
    ):
        raise FullStudyExecutionError("canonical source identity differs")
    pair = study.pair_audit(plan=plan, item=item, source_path=path, repo_root=repo_root)
    accounting = pair.get("accounting") or {}
    if accounting.get("reconciled") is not True:
        raise FullStudyExecutionError("source generation cost is not fully reconciled")
    audit_path = study._write_artifact(
        endpoint_root / "audits",
        f"reasoning-effort-v8-pair-audit-{item['work_item_id'][:12]}",
        pair,
    )
    usable = pair.get("decision") == "passed_all_predicates"
    return {
        "disposition": "source_usable" if usable else "source_reliability_failure",
        "source_path": study._relative(repo_root, path),
        "source_artifact_sha256": digest,
        "pair_audit_path": study._relative(repo_root, audit_path),
        "pair_audit_sha256": study._regular_json(audit_path)["artifact_sha256"],
        "actual_cost_usd": study._decimal_text(Decimal(str(accounting["actual_cost_usd"]))),
        "audit_failures": list(pair.get("failures") or []),
    }


def _record_canonical_artifact(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    source_record: tuple[Path, dict[str, Any], str],
    global_ledger: Path,
) -> dict[str, Any]:
    path, artifact, digest = source_record
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    matches = [
        row
        for row in entries
        if row.get("event_type") == "artifact_recorded"
        and row.get("reservation_entry_sha256") == canonical_reservation["entry_sha256"]
    ]
    expected = {
        "event_type": "artifact_recorded",
        "runner_run_id": item["run_id"],
        "reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "manifest_sha256": item["manifest"]["semantic_sha256"],
        "model_id": item["route_coordinate"]["model_id"],
        "provider_tag": item["route_coordinate"]["provider_endpoint"],
        "artifact_filename": path.name,
        "artifact_sha256": digest,
        "campaign_id": study.STUDY_ID,
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "work_item_id": item["work_item_id"],
    }
    if matches:
        if len(matches) != 1 or any(
            matches[0].get(key) != value for key, value in expected.items()
        ):
            raise FullStudyExecutionError("canonical artifact finalization differs")
        return dict(matches[0])
    if artifact.get("artifact_sha256") != digest:
        raise FullStudyExecutionError("canonical source digest differs")
    return frontier.append_ledger_event(global_ledger, expected)


def _verify_endpoint_incident_current_disposition(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_incident: Mapping[str, Any],
    global_ledger: Path,
) -> CanonicalDisposition:
    """Require exact agreement between an incident and the current ledger."""

    _validate_endpoint_incident_identity(
        incident=endpoint_incident,
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
    )
    disposition = _canonical_disposition(
        plan=plan,
        block=block,
        item=item,
        canonical_reservation=canonical_reservation,
        global_ledger=global_ledger,
    )
    expected_artifact_event = (
        disposition.artifact_event.get("entry_sha256")
        if disposition.artifact_event is not None
        else None
    )
    expected_retained = (
        Decimal(item["worst_case_reserve_usd"]) if disposition.retained else Decimal(0)
    )
    try:
        observed_retained = Decimal(
            str(endpoint_incident.get("work_item_reserve_retained_usd") or "0")
        )
    except Exception as error:
        raise FullStudyExecutionError("endpoint incident retained reserve is malformed") from error
    if (
        endpoint_incident.get("canonical_reservation_status") != disposition.status
        or endpoint_incident.get("canonical_reservation_retained") is not disposition.retained
        or endpoint_incident.get("canonical_artifact_record_entry_sha256")
        != expected_artifact_event
        or observed_retained != expected_retained
    ):
        raise FullStudyExecutionError(
            "stale endpoint incident canonical disposition differs from global ledger"
        )
    return disposition


def _replay_endpoint_incident_exactly(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_incident: Mapping[str, Any],
    coordinator_ledger: Path,
    global_ledger: Path,
) -> dict[str, Any]:
    _verify_endpoint_incident_current_disposition(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
        endpoint_incident=endpoint_incident,
        global_ledger=global_ledger,
    )
    payload = {
        key: value for key, value in endpoint_incident.items() if key not in _LEDGER_PROTECTED
    }
    expected = {
        **payload,
        "endpoint_incident_entry_sha256": endpoint_incident["entry_sha256"],
    }
    state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
    matches = [
        row
        for row in state["incidents"].get(block["admission_block_id"], [])
        if row.get("work_item_id") == item["work_item_id"]
    ]
    if matches:
        if len(matches) != 1 or any(
            matches[0].get(key) != value for key, value in expected.items()
        ):
            raise FullStudyExecutionError("coordinator incident replay differs")
        return dict(matches[0])
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={"event_type": "family_block_execution_incident", **expected},
    )


def _append_prestart_no_delivery_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    error: BaseException,
) -> dict[str, Any]:
    """Record a valid coordinator-only transition before durable delivery."""

    item_id = str(item["work_item_id"])
    endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
    if (
        item_id in endpoint_state["started"]
        or item_id in endpoint_state["terminals"]
        or item_id in endpoint_state["incidents"]
    ):
        raise FullStudyExecutionError(
            "pre-start no-delivery transition conflicts with endpoint state"
        )
    disposition = _canonical_disposition(
        plan=plan,
        block=block,
        item=item,
        canonical_reservation=canonical_reservation,
        global_ledger=global_ledger,
    )
    common = {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, item_id),
        "work_item_id": item_id,
        "block_reservation_entry_sha256": local_reservation["entry_sha256"],
        "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "incident": "durable_pre_start_no_delivery",
        "delivery_evidence": {
            "item_execution_started_durable": False,
            "endpoint_event_count": 0,
            "provider_request_started": False,
        },
        **_error_record(error),
        "canonical_reservation_status": disposition.status,
        "canonical_artifact_record_entry_sha256": (
            disposition.artifact_event.get("entry_sha256")
            if disposition.artifact_event is not None
            else None
        ),
        "work_item_reserve_retained_usd": (
            item["worst_case_reserve_usd"] if disposition.retained else "0"
        ),
        "canonical_reservation_retained": disposition.retained,
        "endpoint_incident_appended": False,
        "replay_permitted": False,
    }
    state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
    matches = [
        row
        for row in state["incidents"].get(block["admission_block_id"], [])
        if row.get("work_item_id") == item_id
    ]
    if matches:
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in common.items()):
            raise FullStudyExecutionError("pre-start no-delivery replay differs")
        return dict(matches[0])
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={"event_type": "family_block_execution_incident", **common},
    )


def _append_incident_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    evidence: Mapping[str, Any],
    error: BaseException,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
    endpoint_terminal = endpoint_state["terminals"].get(item_id)
    if endpoint_terminal is not None:
        return _coordinator_terminal_from_endpoint(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_terminal=endpoint_terminal,
            coordinator_ledger=coordinator_ledger,
        )
    existing = endpoint_state["incidents"].get(item_id)
    if existing is not None:
        return _replay_endpoint_incident_exactly(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_incident=existing,
            coordinator_ledger=coordinator_ledger,
            global_ledger=global_ledger,
        )
    if item_id not in endpoint_state["started"]:
        raise FullStudyExecutionError(
            "post-start endpoint incident requires durable item_execution_started"
        )
    disposition = _canonical_disposition(
        plan=plan,
        block=block,
        item=item,
        canonical_reservation=canonical_reservation,
        global_ledger=global_ledger,
    )
    common = {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, item_id),
        "work_item_id": item_id,
        "block_reservation_entry_sha256": local_reservation["entry_sha256"],
        "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "incident": "durable_post_start_without_finalizable_canonical_source",
        "journal_evidence": dict(evidence),
        **_error_record(error),
        "canonical_reservation_status": disposition.status,
        "canonical_artifact_record_entry_sha256": (
            disposition.artifact_event.get("entry_sha256")
            if disposition.artifact_event is not None
            else None
        ),
        "work_item_reserve_retained_usd": (
            item["worst_case_reserve_usd"] if disposition.retained else "0"
        ),
        "canonical_reservation_retained": disposition.retained,
        "replay_permitted": False,
    }
    endpoint_incident = _append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={"event_type": "uncertain_execution_incident", **common},
    )
    _inject(
        failure_injector,
        "after_endpoint_incident_before_coordinator_incident",
        item_id,
    )
    return _replay_endpoint_incident_exactly(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical_reservation,
        endpoint_incident=endpoint_incident,
        coordinator_ledger=coordinator_ledger,
        global_ledger=global_ledger,
    )


async def _classification_fence(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    prepared_block: Mapping[str, BoundRuntime],
    repo_root: Path,
    source_root: Path,
    endpoint_root: Path,
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    adapters: ExecutionAdapters,
    failure_injector: FailureInjector | None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    current_ids = frozenset(_item_map(plan))
    classifier = adapters.classify_source or _source_terminal_payload
    try:
        endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
        endpoint_terminal = endpoint_state["terminals"].get(item_id)
        if endpoint_terminal is not None:
            recovered = _coordinator_terminal_from_endpoint(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_terminal=endpoint_terminal,
                coordinator_ledger=coordinator_ledger,
            )
            return _outcome_from_terminal(recovered)
        endpoint_incident = endpoint_state["incidents"].get(item_id)
        if endpoint_incident is not None:
            incident = _replay_endpoint_incident_exactly(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_incident=endpoint_incident,
                coordinator_ledger=coordinator_ledger,
                global_ledger=global_ledger,
            )
            return {
                "work_item_id": item_id,
                "decision": "durable_incident_reservation_derived",
                "incident_entry_sha256": incident["entry_sha256"],
            }

        runtime = prepared_block[item_id]
        if item_id not in endpoint_state["started"]:
            _inject(failure_injector, "before_item_start", item_id)
            _append_ledger(
                endpoint_ledger,
                role="endpoint",
                event={
                    "event_type": "item_execution_started",
                    "study_plan_sha256": plan["artifact_sha256"],
                    "admission_block_id": block["admission_block_id"],
                    "task_wave_id": _item_wave_id(plan, item_id),
                    "work_item_id": item_id,
                    "run_id": item["run_id"],
                    "endpoint_id": item["route_coordinate"]["endpoint_id"],
                    "variant_id": item["route_coordinate"]["variant_id"],
                    "block_reservation_entry_sha256": local_reservation["entry_sha256"],
                    "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
                    "raw_endpoint_execution_sha256": runtime.raw_execution_contract_sha256,
                    "replay_permitted": False,
                },
            )
            _inject(failure_injector, "after_item_start", item_id)

        _inject(failure_injector, "before_source_lookup", item_id)
        source = _canonical_source_for_item(source_root, item_id, current_work_item_ids=current_ids)
        if source is None:
            evidence = _safe_journal_evidence(source_root, item_id)
            if evidence["request_started_count"]:
                raise MissingCanonicalSource(
                    "provider request may have started without a canonical source"
                )
            _inject(failure_injector, "before_provider_invocation", item_id)
            if adapters.invoke_pair is None:
                await _invoke_live_pair(
                    args=runtime.args,
                    policy=runtime.policy,
                    raw_endpoint=runtime.raw_execution_contract,
                )
            else:
                await adapters.invoke_pair(
                    args=runtime.args,
                    policy=runtime.policy,
                    raw_endpoint=runtime.raw_execution_contract,
                )
            _inject(
                failure_injector,
                "after_provider_invocation_before_source_lookup",
                item_id,
            )
            source = _canonical_source_for_item(
                source_root, item_id, current_work_item_ids=current_ids
            )
        if source is None:
            raise MissingCanonicalSource("pair invocation produced no canonical source")
        _inject(failure_injector, "before_source_classification", item_id)
        payload = classifier(
            plan=plan,
            item=item,
            source_record=source,
            repo_root=repo_root,
            endpoint_root=endpoint_root,
        )
        _inject(failure_injector, "after_source_classification", item_id)
        _inject(failure_injector, "before_global_artifact_finalization", item_id)
        artifact_event = _record_canonical_artifact(
            plan=plan,
            block=block,
            item=item,
            canonical_reservation=canonical_reservation,
            source_record=source,
            global_ledger=global_ledger,
        )
        _inject(failure_injector, "after_global_artifact_finalization", item_id)
        _inject(failure_injector, "before_local_terminal", item_id)
        terminal = _append_local_terminal_idempotent(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            canonical_artifact_event=artifact_event,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            payload=payload,
            failure_injector=failure_injector,
        )
        _inject(failure_injector, "after_coordinator_terminal", item_id)
        return _outcome_from_terminal(terminal)
    except SimulatedCrash:
        raise
    except Exception as error:
        endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
        endpoint_terminal = endpoint_state["terminals"].get(item_id)
        if endpoint_terminal is not None:
            recovered = _coordinator_terminal_from_endpoint(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_terminal=endpoint_terminal,
                coordinator_ledger=coordinator_ledger,
            )
            return _outcome_from_terminal(recovered)
        existing_incident = endpoint_state["incidents"].get(item_id)
        if existing_incident is not None:
            incident = _replay_endpoint_incident_exactly(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_incident=existing_incident,
                coordinator_ledger=coordinator_ledger,
                global_ledger=global_ledger,
            )
            return {
                "work_item_id": item_id,
                "decision": "durable_incident_reservation_derived",
                "incident_entry_sha256": incident["entry_sha256"],
            }
        if item_id not in endpoint_state["started"]:
            prestart = _append_prestart_no_delivery_idempotent(
                plan=plan,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=canonical_reservation,
                endpoint_ledger=endpoint_ledger,
                coordinator_ledger=coordinator_ledger,
                global_ledger=global_ledger,
                error=error,
            )
            return {
                "work_item_id": item_id,
                "decision": "durable_pre_start_no_delivery",
                "incident_entry_sha256": prestart["entry_sha256"],
            }

        disposition = _canonical_disposition(
            plan=plan,
            block=block,
            item=item,
            canonical_reservation=canonical_reservation,
            global_ledger=global_ledger,
        )
        if disposition.artifact_event is not None:
            try:
                source = _canonical_source_for_item(
                    source_root, item_id, current_work_item_ids=current_ids
                )
                if source is not None:
                    payload = classifier(
                        plan=plan,
                        item=item,
                        source_record=source,
                        repo_root=repo_root,
                        endpoint_root=endpoint_root,
                    )
                    terminal = _append_local_terminal_idempotent(
                        plan=plan,
                        block=block,
                        item=item,
                        local_reservation=local_reservation,
                        canonical_reservation=canonical_reservation,
                        canonical_artifact_event=disposition.artifact_event,
                        endpoint_ledger=endpoint_ledger,
                        coordinator_ledger=coordinator_ledger,
                        payload=payload,
                        failure_injector=None,
                    )
                    return _outcome_from_terminal(terminal)
            except SimulatedCrash:
                raise
            except Exception:
                pass
        incident = _append_incident_idempotent(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            global_ledger=global_ledger,
            evidence=_safe_journal_evidence(source_root, item_id),
            error=error,
            failure_injector=failure_injector,
        )
        if incident.get("event_type") == "family_block_item_terminalized":
            return _outcome_from_terminal(incident)
        return {
            "work_item_id": item_id,
            "decision": "durable_incident_reservation_derived",
            "incident_entry_sha256": incident["entry_sha256"],
        }


def _receipt_prefix(block: Mapping[str, Any]) -> str:
    return f"reasoning-effort-v8-block-{int(block['block_ordinal']):02d}-receipt"


def _receipt_document(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    block: Mapping[str, Any],
    state: Mapping[str, Any],
    global_ledger: Path,
) -> dict[str, Any]:
    block_id = str(block["admission_block_id"])
    block_terminal = state["completed"].get(block_id)
    if block_terminal is None:
        raise FullStudyExecutionError("cannot write receipt for nonterminal block")
    terminals = [state["terminals"].get(item_id) for item_id in block["work_item_ids"]]
    if any(value is None for value in terminals):
        raise FullStudyExecutionError("receipt terminal set is incomplete")
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    reservations = _campaign_global_reservations(plan=plan, entries=entries)
    lifecycle: list[dict[str, Any]] = []
    for item_id, terminal in zip(block["work_item_ids"], terminals, strict=True):
        assert terminal is not None
        reservation = reservations[item_id]
        artifact_matches = [
            row
            for row in entries
            if row.get("event_type") == "artifact_recorded"
            and row.get("reservation_entry_sha256") == reservation["entry_sha256"]
        ]
        if len(artifact_matches) != 1 or artifact_matches[0].get("entry_sha256") != terminal.get(
            "canonical_artifact_record_entry_sha256"
        ):
            raise FullStudyExecutionError("receipt canonical lifecycle differs")
        lifecycle.append(
            {
                "work_item_id": item_id,
                "canonical_reservation_entry_sha256": reservation["entry_sha256"],
                "canonical_artifact_record_entry_sha256": artifact_matches[0]["entry_sha256"],
                "source_artifact_sha256": artifact_matches[0]["artifact_sha256"],
                "endpoint_terminal_entry_sha256": terminal["endpoint_terminal_entry_sha256"],
                "coordinator_terminal_entry_sha256": terminal["entry_sha256"],
            }
        )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "recoverable_shared_ledger_family_block_execution",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "bound_preflight_sha256": bound_preflight["artifact_sha256"],
        "governance_go_sha256": governance_go["artifact_sha256"],
        "admission_block_id": block_id,
        "block_terminal_entry_sha256": block_terminal["entry_sha256"],
        "canonical_reservation_count": 28,
        "outcomes": [_outcome_from_terminal(value) for value in terminals if value],
        "canonical_lifecycle": lifecycle,
        "block_terminal": True,
        "durable_incident": False,
        "provider_substitution_performed": False,
        "rank_eligible": False,
    }


def _ensure_terminal_receipt(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    block: Mapping[str, Any],
    state: Mapping[str, Any],
    coordinator_root: Path,
    global_ledger: Path,
) -> tuple[Path, dict[str, Any]]:
    document = _receipt_document(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
        block=block,
        state=state,
        global_ledger=global_ledger,
    )
    expected_sha = study._sha256(document)
    existing = sorted((coordinator_root / "receipts").glob(f"{_receipt_prefix(block)}-*.json"))
    if existing:
        if len(existing) != 1:
            raise FullStudyExecutionError("terminal block receipt is ambiguous")
        observed = study._regular_json(existing[0])
        body = {key: value for key, value in observed.items() if key != "artifact_sha256"}
        if observed.get("artifact_sha256") != expected_sha or body != document:
            raise FullStudyExecutionError("terminal block receipt differs")
        return existing[0], observed
    path = study._write_artifact(coordinator_root / "receipts", _receipt_prefix(block), document)
    return path, study._regular_json(path)


async def execute_one_block(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    governance_go: Mapping[str, Any],
    repo_root: Path,
    api_base: str,
    api_key: str,
    failure_injector: FailureInjector | None = None,
    adapters: ExecutionAdapters | None = None,
) -> dict[str, Any]:
    study.validate_plan(plan, repo_root=repo_root)
    study.verify_bound_preflight(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
    )
    study.verify_governance_go(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
    )
    boundary = adapters or ExecutionAdapters()
    if boundary.roots is None:
        coordinator_root, endpoint_roots = _roots(plan, repo_root)
    else:
        coordinator_root, raw_endpoint_roots = boundary.roots
        endpoint_roots = {str(key): Path(value) for key, value in raw_endpoint_roots.items()}
        if set(endpoint_roots) != set(study.ENDPOINTS):
            raise FullStudyExecutionError("adapter roots do not cover all endpoints")
    coordinator_ledger = coordinator_root / "ledger.jsonl"
    global_ledger = boundary.global_ledger_path or _global_ledger_path(plan, repo_root)
    source_root = boundary.source_root or _canonical_source_root(plan, repo_root)
    blocks = _block_map(plan)
    items = _item_map(plan)

    preliminary_state = _coordinator_state(
        plan, _load_ledger(coordinator_ledger, role="coordinator")
    )
    authorized_id = str(governance_go.get("authorized_admission_block_id") or "")
    if authorized_id in preliminary_state["completed"]:
        path, document = _ensure_terminal_receipt(
            plan=plan,
            human_protocol=human_protocol,
            bound_preflight=bound_preflight,
            governance_go=governance_go,
            block=blocks[authorized_id],
            state=preliminary_state,
            coordinator_root=coordinator_root,
            global_ledger=global_ledger,
        )
        return {
            "decision": "block_terminal_receipt_recovered",
            "document": document,
            "receipt_path": str(path),
            "outcomes": document["outcomes"],
        }
    if preliminary_state["active_block_id"]:
        target_id = str(preliminary_state["active_block_id"])
    else:
        completed = len(preliminary_state["completed"])
        if completed == 6:
            return {"decision": "all_blocks_terminal", "outcomes": []}
        target_id = str(plan["block_execution_order"][completed])
    block = blocks[target_id]
    if authorized_id != target_id:
        raise FullStudyExecutionError("independent GO does not authorize the target block")
    if preliminary_state["incidents"].get(target_id):
        return {
            "decision": "blocked_by_durable_incident",
            "admission_block_id": target_id,
            "outcomes": [],
        }

    prepared_all = prepare_all_runtime_items(
        plan=plan, repo_root=repo_root, source_root=source_root
    )
    _inject(failure_injector, "after_all_runtime_validation_before_side_effect")
    require_environment = (
        boundary.require_live_environment or _require_live_environment_before_reservation
    )
    require_environment()
    _verified_source_index(source_root, current_work_item_ids=frozenset(items))

    _inject(failure_injector, "before_endpoint_attestations")
    attest = boundary.attest_all or _attest_all_endpoints
    attestations = await attest(plan=plan, api_base=api_base, api_key=api_key)
    _inject(failure_injector, "after_endpoint_attestations")
    prepared_block = _bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=attestations,
        prepared_all=prepared_all,
        repo_root=repo_root,
        source_root=source_root,
    )
    attestation_document = {
        "schema_version": ATTESTATION_SCHEMA,
        "record_role": "raw_contract_monotone_capacity_bound_prestart_attestation",
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": target_id,
        "records": list(attestations),
        "record_sha256s": [study._sha256(record) for record in attestations],
        "raw_execution_contract_sha256_by_endpoint": {
            endpoint_id: runtime.raw_execution_contract_sha256
            for endpoint_id in study.ENDPOINTS
            for runtime in [
                next(
                    value
                    for item_id, value in prepared_block.items()
                    if items[item_id]["route_coordinate"]["endpoint_id"] == endpoint_id
                )
            ]
        },
        "capacity_attestation_by_endpoint": {
            endpoint_id: study.validate_monotone_capacity_contract(
                frozen=plan["models"][endpoint_id]["semantic_execution_contract"],
                observed=study.semantic_endpoint_contract(
                    next(
                        runtime.raw_execution_contract
                        for item_id, runtime in prepared_block.items()
                        if items[item_id]["route_coordinate"]["endpoint_id"] == endpoint_id
                    )
                ),
            )
            for endpoint_id in study.ENDPOINTS
        },
        "frozen_request_caps": {
            "max_intermediate_tokens": plan["common_protocol"]["max_intermediate_tokens"],
            "max_output_tokens": plan["common_protocol"]["max_output_tokens"],
        },
        "bound_runtime_work_item_ids": list(prepared_block),
        "bound_runtime_count": 28,
        "catalog_http_gets": sum(
            int(record.get("catalog_http_gets") or 0) for record in attestations
        ),
        "provider_completion_requests": 0,
        "epicure_calls": 0,
    }
    attestation_path = study._write_artifact(
        coordinator_root / "endpoint-attestations",
        f"reasoning-effort-v8-block-{block['block_ordinal']:02d}-attestations",
        attestation_document,
    )
    attestation_ref = study._file_ref(repo_root, attestation_path)
    _inject(failure_injector, "after_attestation_binding_before_global_lock")

    outcomes: list[dict[str, Any]] = []
    receipt_path: Path | None = None
    receipt_document: dict[str, Any] | None = None
    with frontier._exclusive_runner_lock(global_ledger):
        _inject(failure_injector, "after_global_lock_before_reservations")
        with _ledger_lock(coordinator_ledger):
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            if state["active_block_id"] not in {None, target_id}:
                raise FullStudyExecutionError("target block changed during attestation")
            existing_local = state["reservations"].get(target_id)
            if existing_local is not None:
                _verify_local_global_binding(
                    plan=plan,
                    block=block,
                    local_reservation=existing_local,
                    global_entries=frontier.load_ledger(global_ledger),
                )
            canonical_reservations, accounting = _ensure_canonical_reservations(
                plan=plan,
                block=block,
                repo_root=repo_root,
                global_ledger=global_ledger,
                source_root=source_root,
                failure_injector=failure_injector,
            )
            canonical_by_item = dict(
                zip(block["work_item_ids"], canonical_reservations, strict=True)
            )
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            local_reservation = state["reservations"].get(target_id)
            if local_reservation is None:
                _inject(failure_injector, "before_local_block_reservation")
                local_reservation = _append_ledger(
                    coordinator_ledger,
                    role="coordinator",
                    event={
                        "event_type": "family_block_reservation_created",
                        "study_plan_sha256": plan["artifact_sha256"],
                        "admission_block_id": target_id,
                        "block_ordinal": block["block_ordinal"],
                        "wave_ids": block["wave_ids"],
                        "task_ids": block["task_ids"],
                        "task_families": block["task_families"],
                        "work_item_ids": block["work_item_ids"],
                        "reserved_usd": block["worst_case_reserve_usd"],
                        "canonical_reservation_entry_sha256_by_work_item": {
                            item_id: canonical_by_item[item_id]["entry_sha256"]
                            for item_id in block["work_item_ids"]
                        },
                        "canonical_reservation_entry_sha256s": [
                            canonical_by_item[item_id]["entry_sha256"]
                            for item_id in block["work_item_ids"]
                        ],
                        "endpoint_attestation": attestation_ref,
                        "global_accounting_at_admission": {
                            key: value
                            for key, value in accounting.items()
                            if key not in {"entries", "active"}
                        },
                        "replay_permitted": False,
                    },
                )
                _inject(failure_injector, "after_local_block_reservation")
            _verify_local_global_binding(
                plan=plan,
                block=block,
                local_reservation=local_reservation,
                global_entries=frontier.load_ledger(global_ledger),
            )

            for item_id in block["work_item_ids"]:
                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                if item_id in state["terminals"]:
                    continue
                if state["incidents"].get(target_id):
                    break
                item = items[item_id]
                endpoint_id = str(item["route_coordinate"]["endpoint_id"])
                endpoint_root = endpoint_roots[endpoint_id]
                outcome = await _classification_fence(
                    plan=plan,
                    block=block,
                    item=item,
                    local_reservation=local_reservation,
                    canonical_reservation=canonical_by_item[item_id],
                    prepared_block=prepared_block,
                    repo_root=repo_root,
                    source_root=source_root,
                    endpoint_root=endpoint_root,
                    endpoint_ledger=endpoint_root / "ledger.jsonl",
                    coordinator_ledger=coordinator_ledger,
                    global_ledger=global_ledger,
                    adapters=boundary,
                    failure_injector=failure_injector,
                )
                outcomes.append(outcome)
                if outcome["decision"] in {
                    "durable_incident_reservation_derived",
                    "durable_pre_start_no_delivery",
                }:
                    break
            state = _coordinator_state(plan, _load_ledger(coordinator_ledger, role="coordinator"))
            if not state["incidents"].get(target_id) and all(
                item_id in state["terminals"] for item_id in block["work_item_ids"]
            ):
                _inject(failure_injector, "before_block_terminal")
                _terminalize_block(plan=plan, block=block, coordinator_ledger=coordinator_ledger)
                _inject(failure_injector, "after_block_terminal")
            final_state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            if target_id in final_state["completed"]:
                receipt_path, receipt_document = _ensure_terminal_receipt(
                    plan=plan,
                    human_protocol=human_protocol,
                    bound_preflight=bound_preflight,
                    governance_go=governance_go,
                    block=block,
                    state=final_state,
                    coordinator_root=coordinator_root,
                    global_ledger=global_ledger,
                )

    if receipt_document is not None and receipt_path is not None:
        return {
            "decision": "block_terminal",
            "document": receipt_document,
            "receipt_path": str(receipt_path),
            "outcomes": receipt_document["outcomes"],
        }
    return {
        "decision": "durable_incident_stop",
        "admission_block_id": target_id,
        "outcomes": outcomes,
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--bound-preflight", type=Path, required=True)
    parser.add_argument("--governance-go", type=Path, required=True)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--confirm")
    arguments = parser.parse_args(argv)
    if arguments.confirm != study.CONFIRMATION:
        raise SystemExit("exact V8 independently reviewed one-block confirmation is required")
    repo_root = arguments.repo_root.resolve()
    result = asyncio.run(
        execute_one_block(
            plan=study._regular_json(arguments.plan.resolve()),
            human_protocol=study._regular_json(arguments.human_protocol.resolve()),
            bound_preflight=study._regular_json(arguments.bound_preflight.resolve()),
            governance_go=study._regular_json(arguments.governance_go.resolve()),
            repo_root=repo_root,
            api_base=arguments.api_base,
            api_key=arguments.api_key,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
