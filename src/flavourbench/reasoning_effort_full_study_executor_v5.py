"""Execute one V5 reasoning-effort block with shared-ledger crash recovery.

This module is inert unless the exact V5 confirmation and an independently
content-addressed one-block GO are supplied.  It never releases a canonical
reservation without a canonical live-smoke artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import frontier_contract_runner as frontier
from . import reasoning_effort_full_study_executor_v1 as local
from . import reasoning_effort_full_study_v5 as study

RECEIPT_SCHEMA = "flavourbench-reasoning-effort-family-block-receipt-v5"
ATTESTATION_SCHEMA = "flavourbench-reasoning-effort-block-attestations-v5"

FullStudyExecutionError = local.FullStudyExecutionError
_load_ledger = local._load_ledger
_append_ledger = local._append_ledger
_ledger_lock = local._ledger_lock
_item_map = local._item_map
_block_map = local._block_map
_item_wave_id = local._item_wave_id
_roots = local._roots
_manifest = local._manifest
_policy = local._policy
_endpoint_state = local._endpoint_state
_journal_evidence = local._journal_evidence
_terminalize_block = local._terminalize_block

FailureInjector = Callable[[str, str | None], None]


class SimulatedCrash(BaseException):
    """Test-only process-crash cut; deliberately bypasses Exception handlers."""


class MissingCanonicalSource(FullStudyExecutionError):
    """A durable item start has no safely finalizable canonical source."""


def _global_ledger_path(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(
        plan["execution_roots"]["canonical_global_reservation_ledger"]
    )


def _canonical_source_root(plan: Mapping[str, Any], repo_root: Path) -> Path:
    return repo_root / str(plan["execution_roots"]["canonical_global_source"])


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


def _live_args(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    repo_root: Path,
    source_root: Path,
    raw_endpoint_sha256: str,
) -> argparse.Namespace:
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
        output_dir=str(source_root),
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
    expected_source = str(
        Path(plan["execution_roots"]["canonical_global_source"])
    )
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
        or not str(args.output_dir).endswith(expected_source)
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
    """Resolve every import, manifest, policy and argument without side effects."""

    source_root = _canonical_source_root(plan, repo_root)
    prepared: dict[str, tuple[Any, argparse.Namespace]] = {}
    for item_id, item in sorted(_item_map(plan).items()):
        policy = _policy(plan, item, repo_root)
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source_root=source_root,
            raw_endpoint_sha256="0" * 64,
        )
        _validate_live_args(plan=plan, item=item, args=args, policy=policy)
        prepared[item_id] = (policy, args)
    if len(prepared) != 168:
        raise FullStudyExecutionError("runtime preparation did not cover 168 items")
    return prepared


async def _attest_all_endpoints(
    *, plan: Mapping[str, Any], api_base: str, api_key: str
) -> list[dict[str, Any]]:
    return await local._attest_all_endpoints(
        plan=plan, api_base=api_base, api_key=api_key
    )


def _bind_block_runtime_after_attestation(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    attestations: Sequence[Mapping[str, Any]],
    prepared_all: Mapping[str, tuple[Any, argparse.Namespace]],
    repo_root: Path,
) -> dict[str, tuple[Any, argparse.Namespace]]:
    by_endpoint: dict[str, Mapping[str, Any]] = {}
    for record in attestations:
        endpoint_id = str(record.get("endpoint_id") or "")
        raw = record.get("raw_execution_contract")
        raw_sha = str(record.get("raw_execution_contract_sha256") or "")
        if (
            endpoint_id in by_endpoint
            or endpoint_id not in study.ENDPOINTS
            or not isinstance(raw, Mapping)
            or study._sha256(raw) != raw_sha
        ):
            raise FullStudyExecutionError("endpoint attestation hash binding differs")
        by_endpoint[endpoint_id] = record
    if set(by_endpoint) != set(study.ENDPOINTS):
        raise FullStudyExecutionError("attestations do not cover exactly three endpoints")
    source_root = Path(next(iter(prepared_all.values()))[1].output_dir)
    items = _item_map(plan)
    rebound: dict[str, tuple[Any, argparse.Namespace]] = {}
    for item_id in block["work_item_ids"]:
        item = items[item_id]
        policy = prepared_all[item_id][0]
        endpoint_id = str(item["route_coordinate"]["endpoint_id"])
        raw_sha = str(by_endpoint[endpoint_id]["raw_execution_contract_sha256"])
        args = _live_args(
            plan=plan,
            item=item,
            repo_root=repo_root,
            source_root=source_root,
            raw_endpoint_sha256=raw_sha,
        )
        # _live_args only uses repo_root for immutable manifests/policies.  Reuse
        # the already validated policy and restore the exact policy digest here.
        args.expected_execution_policy_sha256 = policy.sha256
        _validate_live_args(plan=plan, item=item, args=args, policy=policy)
        rebound[item_id] = (policy, args)
    if list(rebound) != list(block["work_item_ids"]):
        raise FullStudyExecutionError("full block runtime binding order differs")
    return rebound


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
            raise AdmissionDenied("live authority changed after reservation")
        await live_smoke(args)


def _coordinator_state(
    plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    state = local._coordinator_state(plan, entries)
    blocks = _block_map(plan)
    for block_id, reservation in state["reservations"].items():
        mapping = reservation.get("canonical_reservation_entry_sha256_by_work_item")
        shas = reservation.get("canonical_reservation_entry_sha256s")
        if (
            not isinstance(mapping, Mapping)
            or set(map(str, mapping)) != set(blocks[block_id]["work_item_ids"])
            or shas != [mapping[item_id] for item_id in blocks[block_id]["work_item_ids"]]
            or len(set(map(str, shas))) != 28
        ):
            raise FullStudyExecutionError("local block lacks exact canonical reservations")
    for terminal in state["terminals"].values():
        if not all(
            isinstance(terminal.get(field), str) and len(terminal[field]) == 64
            for field in (
                "canonical_reservation_entry_sha256",
                "canonical_artifact_record_entry_sha256",
                "source_artifact_sha256",
            )
        ):
            raise FullStudyExecutionError("local terminal lacks canonical finalization binding")
    return state


def _verify_global_anchor(
    *, plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> None:
    anchor = plan["canonical_global_ledger_anchor"]
    sequence = int(anchor["sequence"])
    if (
        len(entries) < sequence
        or entries[sequence - 1].get("entry_sha256") != anchor["head_entry_sha256"]
    ):
        raise FullStudyExecutionError("canonical global-ledger prefix differs")


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


def _global_accounting_locked(
    *, plan: Mapping[str, Any], repo_root: Path, global_ledger: Path
) -> dict[str, Any]:
    entries = frontier.load_ledger(global_ledger)
    _verify_global_anchor(plan=plan, entries=entries)
    artifacts_root = repo_root / "flavourbench/artifacts"
    scan = frontier.scan_live_smoke_artifacts(
        artifacts_root / "live-smoke",
        corrections_directory=artifacts_root / "corrections",
    )
    frontier.validate_ledger_artifact_links(
        entries,
        scan,
        reconciliation_directory=artifacts_root / "frontier-contract/reconciliations",
    )
    artifact_by_sha = {row.artifact_sha256: row for row in scan.artifacts}
    anchor_sequence = int(plan["canonical_global_ledger_anchor"]["sequence"])
    finalized_suffix = [
        row
        for row in entries[anchor_sequence:]
        if row.get("event_type") == "artifact_recorded"
    ]
    suffix_exposure = Decimal(0)
    seen: set[str] = set()
    for event in finalized_suffix:
        digest = str(event.get("artifact_sha256") or "")
        artifact = artifact_by_sha.get(digest)
        if artifact is None or digest in seen:
            raise FullStudyExecutionError("global suffix artifact exposure is ambiguous")
        seen.add(digest)
        suffix_exposure += artifact.exposure_usd
    active = frontier.active_ledger_reservations(entries)
    active_total = sum(active.values(), Decimal(0))
    current = study.CURRENT_EXPOSURE_USD + suffix_exposure + active_total
    active_incidents = [
        dict(row)
        for row in entries[anchor_sequence:]
        if row.get("event_type") == "execution_incident"
        and row.get("reservation_entry_sha256") in active
    ]
    return {
        "entries": entries,
        "active": active,
        "baseline_exposure_usd": study._decimal_text(study.CURRENT_EXPOSURE_USD),
        "post_anchor_finalized_exposure_usd": study._decimal_text(suffix_exposure),
        "canonical_active_reservation_usd": study._decimal_text(active_total),
        "current_total_exposure_usd": study._decimal_text(current),
        "active_incidents": active_incidents,
    }


def _ensure_canonical_reservations(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    repo_root: Path,
    global_ledger: Path,
    failure_injector: FailureInjector | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accounting = _global_accounting_locked(
        plan=plan, repo_root=repo_root, global_ledger=global_ledger
    )
    entries = accounting["entries"]
    found = _campaign_global_reservations(plan=plan, entries=entries)
    items = _item_map(plan)
    missing = [item_id for item_id in block["work_item_ids"] if item_id not in found]
    missing_reserve = sum(
        (Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in missing),
        Decimal(0),
    )
    projected = Decimal(accounting["current_total_exposure_usd"]) + missing_reserve
    foreign_incidents = [
        row
        for row in accounting["active_incidents"]
        if row.get("campaign_id") != study.STUDY_ID
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
            _canonical_reservation_identity(
                plan=plan, block=block, item=items[item_id]
            ),
        )
        found[item_id] = entry
        _inject(failure_injector, "after_global_reservation", item_id)
    refreshed = _global_accounting_locked(
        plan=plan, repo_root=repo_root, global_ledger=global_ledger
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
    expected = {
        item_id: found[item_id]["entry_sha256"] for item_id in block["work_item_ids"]
    }
    if dict(mapping) != expected or local_reservation.get(
        "canonical_reservation_entry_sha256s"
    ) != list(expected.values()):
        raise FullStudyExecutionError("local/global reservation binding differs")


def _canonical_source_for_item(source_root: Path, work_item_id: str):
    from .reasoning_effort_route_gate_v5 import _source_map

    return _source_map(source_root).get(work_item_id)


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
        or path.parent.resolve()
        != _canonical_source_root(plan, repo_root).resolve()
        or artifact.get("run_id") != item["run_id"]
        or artifact.get("dataset_work_item_id") != item["work_item_id"]
        or artifact.get("requested_model_id") != coordinate["model_id"]
        or artifact.get("requested_provider") != coordinate["provider_endpoint"]
        or artifact.get("candidate_manifest_sha256")
        != item["manifest"]["semantic_sha256"]
    ):
        raise FullStudyExecutionError("canonical source identity differs")
    pair = study.pair_audit(plan=plan, item=item, source_path=path, repo_root=repo_root)
    accounting = pair.get("accounting") or {}
    if accounting.get("reconciled") is not True:
        raise FullStudyExecutionError("source generation cost is not fully reconciled")
    audit_path = study._write_artifact(
        endpoint_root / "audits",
        f"reasoning-effort-v5-pair-audit-{item['work_item_id'][:12]}",
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
    event = frontier.append_ledger_event(global_ledger, expected)
    return event


def _append_local_terminal_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    canonical_artifact_event: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    payload: Mapping[str, Any],
    failure_injector: FailureInjector | None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    common = {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, item_id),
        "work_item_id": item_id,
        "block_reservation_entry_sha256": local_reservation["entry_sha256"],
        "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
        "canonical_artifact_record_entry_sha256": canonical_artifact_event["entry_sha256"],
        **dict(payload),
        "replay_permitted": False,
        "rank_eligible": False,
    }
    endpoint_entries = _load_ledger(endpoint_ledger, role="endpoint")
    endpoint_state = _endpoint_state(endpoint_entries)
    endpoint_terminal = endpoint_state["terminals"].get(item_id)
    if endpoint_terminal is None:
        endpoint_terminal = _append_ledger(
            endpoint_ledger,
            role="endpoint",
            event={"event_type": "source_terminalized", **common},
        )
    else:
        if any(endpoint_terminal.get(key) != value for key, value in common.items()):
            raise FullStudyExecutionError("endpoint terminal recovery binding differs")
    _inject(failure_injector, "after_endpoint_terminal_before_coordinator_terminal", item_id)
    coordinator_entries = _load_ledger(coordinator_ledger, role="coordinator")
    coordinator_state = _coordinator_state(plan, coordinator_entries)
    coordinator_terminal = coordinator_state["terminals"].get(item_id)
    coordinator_common = {
        **common,
        "endpoint_id": item["route_coordinate"]["endpoint_id"],
        "variant_id": item["route_coordinate"]["variant_id"],
        "endpoint_terminal_entry_sha256": endpoint_terminal["entry_sha256"],
    }
    if coordinator_terminal is None:
        coordinator_terminal = _append_ledger(
            coordinator_ledger,
            role="coordinator",
            event={"event_type": "family_block_item_terminalized", **coordinator_common},
        )
    elif any(coordinator_terminal.get(key) != value for key, value in coordinator_common.items()):
        raise FullStudyExecutionError("coordinator terminal recovery binding differs")
    return dict(coordinator_terminal)


def _append_incident_idempotent(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    evidence: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
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
        "work_item_reserve_retained_usd": item["worst_case_reserve_usd"],
        "canonical_reservation_retained": True,
        "replay_permitted": False,
    }
    endpoint_state = _endpoint_state(_load_ledger(endpoint_ledger, role="endpoint"))
    endpoint_incident = endpoint_state["incidents"].get(item_id)
    if endpoint_incident is None:
        endpoint_incident = _append_ledger(
            endpoint_ledger,
            role="endpoint",
            event={"event_type": "uncertain_execution_incident", **common},
        )
    coordinator_state = _coordinator_state(
        plan, _load_ledger(coordinator_ledger, role="coordinator")
    )
    existing = coordinator_state["incidents"].get(block["admission_block_id"], [])
    for row in existing:
        if row.get("work_item_id") == item_id:
            return dict(row)
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_execution_incident",
            **common,
            "endpoint_incident_entry_sha256": endpoint_incident["entry_sha256"],
        },
    )


async def _process_started_item(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    canonical_reservation: Mapping[str, Any],
    policy: Any,
    args: argparse.Namespace,
    endpoint_attestation: Mapping[str, Any],
    repo_root: Path,
    source_root: Path,
    endpoint_root: Path,
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    global_ledger: Path,
    failure_injector: FailureInjector | None,
) -> dict[str, Any]:
    item_id = str(item["work_item_id"])
    try:
        _inject(failure_injector, "before_source_lookup", item_id)
        source = _canonical_source_for_item(source_root, item_id)
        if source is None:
            evidence = _journal_evidence(source_root, item_id)
            if evidence["request_started_count"]:
                raise MissingCanonicalSource(
                    "provider request may have started without a canonical source"
                )
            _inject(failure_injector, "before_provider_invocation", item_id)
            await _invoke_live_pair(
                args=args, policy=policy, endpoint=endpoint_attestation
            )
            _inject(failure_injector, "after_provider_invocation_before_source_lookup", item_id)
            source = _canonical_source_for_item(source_root, item_id)
        if source is None:
            raise MissingCanonicalSource("pair invocation produced no canonical source")
        _inject(failure_injector, "before_source_classification", item_id)
        payload = _source_terminal_payload(
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
        return {
            "work_item_id": item_id,
            "decision": payload["disposition"],
            "terminal_entry_sha256": terminal["entry_sha256"],
        }
    except SimulatedCrash:
        raise
    except Exception as error:
        try:
            evidence = _journal_evidence(source_root, item_id)
        except Exception as evidence_error:
            evidence = {
                "journal_count": None,
                "request_started_count": None,
                "journals": [],
                "evidence_error": _error_record(evidence_error),
            }
        incident = _append_incident_idempotent(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical_reservation,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            evidence=evidence,
            error=error,
        )
        return {
            "work_item_id": item_id,
            "decision": "durable_incident_reservation_retained",
            "incident_entry_sha256": incident["entry_sha256"],
        }


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
    prepared_all = prepare_all_runtime_items(plan=plan, repo_root=repo_root)
    _inject(failure_injector, "after_all_runtime_validation_before_side_effect")
    _require_live_environment_before_reservation()

    coordinator_root, endpoint_roots = _roots(plan, repo_root)
    coordinator_ledger = coordinator_root / "ledger.jsonl"
    global_ledger = _global_ledger_path(plan, repo_root)
    source_root = _canonical_source_root(plan, repo_root)
    block_map = _block_map(plan)
    item_map = _item_map(plan)

    preliminary_state = _coordinator_state(
        plan, _load_ledger(coordinator_ledger, role="coordinator")
    )
    if preliminary_state["active_block_id"]:
        target_id = str(preliminary_state["active_block_id"])
    else:
        completed = len(preliminary_state["completed"])
        if completed == 6:
            return {"decision": "all_blocks_terminal", "outcomes": []}
        target_id = str(plan["block_execution_order"][completed])
    block = block_map[target_id]
    if governance_go.get("authorized_admission_block_id") != target_id:
        raise FullStudyExecutionError("independent GO does not authorize the target block")
    if preliminary_state["incidents"].get(target_id):
        return {
            "decision": "blocked_by_durable_incident",
            "admission_block_id": target_id,
            "outcomes": [],
        }

    _inject(failure_injector, "before_endpoint_attestations")
    attestations = await _attest_all_endpoints(
        plan=plan, api_base=api_base, api_key=api_key
    )
    _inject(failure_injector, "after_endpoint_attestations")
    prepared_block = _bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=attestations,
        prepared_all=prepared_all,
        repo_root=repo_root,
    )
    attestation_document = {
        "schema_version": ATTESTATION_SCHEMA,
        "record_role": "all_endpoint_hash_bound_full_block_runtime_attestation",
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": target_id,
        "records": list(attestations),
        "record_sha256s": [study._sha256(record) for record in attestations],
        "bound_runtime_work_item_ids": list(prepared_block),
        "bound_runtime_count": 28,
        "catalog_http_gets": 6,
        "provider_completion_requests": 0,
        "epicure_calls": 0,
    }
    attestation_path = study._write_artifact(
        coordinator_root / "endpoint-attestations",
        f"reasoning-effort-v5-block-{block['block_ordinal']:02d}-attestations",
        attestation_document,
    )
    attestation_ref = study._file_ref(repo_root, attestation_path)
    _inject(failure_injector, "after_attestation_binding_before_global_lock")

    outcomes: list[dict[str, Any]] = []
    with frontier._exclusive_runner_lock(global_ledger):
        _inject(failure_injector, "after_global_lock_before_reservations")
        with _ledger_lock(coordinator_ledger):
            state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            if state["active_block_id"] not in {None, target_id}:
                raise FullStudyExecutionError("target block changed during attestation")
            existing_local_reservation = state["reservations"].get(target_id)
            if existing_local_reservation is not None:
                _verify_local_global_binding(
                    plan=plan,
                    block=block,
                    local_reservation=existing_local_reservation,
                    global_entries=frontier.load_ledger(global_ledger),
                )
            canonical_reservations, accounting = _ensure_canonical_reservations(
                plan=plan,
                block=block,
                repo_root=repo_root,
                global_ledger=global_ledger,
                failure_injector=failure_injector,
            )
            canonical_by_item = dict(
                zip(block["work_item_ids"], canonical_reservations, strict=True)
            )
            state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
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

            attestation_by_endpoint = {
                str(record["endpoint_id"]): record for record in attestations
            }
            for item_id in block["work_item_ids"]:
                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                if item_id in state["terminals"]:
                    continue
                if state["incidents"].get(target_id):
                    break
                item = item_map[item_id]
                endpoint_id = str(item["route_coordinate"]["endpoint_id"])
                endpoint_root = endpoint_roots[endpoint_id]
                endpoint_ledger = endpoint_root / "ledger.jsonl"
                endpoint_state = _endpoint_state(
                    _load_ledger(endpoint_ledger, role="endpoint")
                )
                if item_id in endpoint_state["incidents"]:
                    _append_incident_idempotent(
                        plan=plan,
                        block=block,
                        item=item,
                        local_reservation=local_reservation,
                        canonical_reservation=canonical_by_item[item_id],
                        endpoint_ledger=endpoint_ledger,
                        coordinator_ledger=coordinator_ledger,
                        evidence={},
                        error=MissingCanonicalSource("recovered endpoint incident"),
                    )
                    break
                if item_id not in endpoint_state["started"]:
                    _inject(failure_injector, "before_item_start", item_id)
                    _append_ledger(
                        endpoint_ledger,
                        role="endpoint",
                        event={
                            "event_type": "item_execution_started",
                            "study_plan_sha256": plan["artifact_sha256"],
                            "admission_block_id": target_id,
                            "task_wave_id": _item_wave_id(plan, item_id),
                            "work_item_id": item_id,
                            "run_id": item["run_id"],
                            "endpoint_id": endpoint_id,
                            "variant_id": item["route_coordinate"]["variant_id"],
                            "block_reservation_entry_sha256": local_reservation[
                                "entry_sha256"
                            ],
                            "canonical_reservation_entry_sha256": canonical_by_item[
                                item_id
                            ]["entry_sha256"],
                            "raw_endpoint_execution_sha256": attestation_by_endpoint[
                                endpoint_id
                            ]["raw_execution_contract_sha256"],
                            "replay_permitted": False,
                        },
                    )
                    _inject(failure_injector, "after_item_start", item_id)
                policy, args = prepared_block[item_id]
                outcome = await _process_started_item(
                    plan=plan,
                    block=block,
                    item=item,
                    local_reservation=local_reservation,
                    canonical_reservation=canonical_by_item[item_id],
                    policy=policy,
                    args=args,
                    endpoint_attestation=attestation_by_endpoint[endpoint_id],
                    repo_root=repo_root,
                    source_root=source_root,
                    endpoint_root=endpoint_root,
                    endpoint_ledger=endpoint_ledger,
                    coordinator_ledger=coordinator_ledger,
                    global_ledger=global_ledger,
                    failure_injector=failure_injector,
                )
                outcomes.append(outcome)
                if outcome["decision"] == "durable_incident_reservation_retained":
                    break
            state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            if (
                not state["incidents"].get(target_id)
                and all(item_id in state["terminals"] for item_id in block["work_item_ids"])
            ):
                _inject(failure_injector, "before_block_terminal")
                _terminalize_block(
                    plan=plan, block=block, coordinator_ledger=coordinator_ledger
                )
                _inject(failure_injector, "after_block_terminal")
            final_state = _coordinator_state(
                plan, _load_ledger(coordinator_ledger, role="coordinator")
            )
            final_accounting = _global_accounting_locked(
                plan=plan, repo_root=repo_root, global_ledger=global_ledger
            )

    document = {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "crash_safe_shared_ledger_family_block_execution",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "bound_preflight_sha256": bound_preflight["artifact_sha256"],
        "governance_go_sha256": governance_go["artifact_sha256"],
        "admission_block_id": target_id,
        "canonical_reservation_count": 28,
        "outcomes": outcomes,
        "block_terminal": target_id in final_state["completed"],
        "durable_incident": bool(final_state["incidents"].get(target_id)),
        "global_accounting": {
            key: value
            for key, value in final_accounting.items()
            if key not in {"entries", "active"}
        },
        "provider_substitution_performed": False,
        "rank_eligible": False,
    }
    receipt_path = study._write_artifact(
        coordinator_root / "receipts",
        f"reasoning-effort-v5-block-{block['block_ordinal']:02d}-receipt",
        document,
    )
    return {
        "decision": (
            "block_terminal" if document["block_terminal"] else "durable_incident_stop"
        ),
        "document": study._regular_json(receipt_path),
        "receipt_path": str(receipt_path),
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
        raise SystemExit("exact V5 crash-safe one-family-block confirmation is required")
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
