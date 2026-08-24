"""Execute at most one frozen four-task reasoning-effort family block.

The admission unit is one task from every culinary family, not one model cell.
A block atomically reserves 28 matched Epicure pairs, holds the shared frontier
lock until every cell has a terminal disposition, and never replays an
identifier. Completed blocks are charged from reconciled immutable sources; an
active block is charged at its full reservation so partial sources are never
counted twice.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from . import reasoning_effort_full_study_v1 as study
from . import reasoning_effort_route_gate_v5 as v5

LEDGER_SCHEMA = "flavourbench-reasoning-effort-family-block-ledger-v2"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-family-block-execution-receipt-v2"
ATTESTATION_SCHEMA = "flavourbench-reasoning-effort-family-block-attestations-v2"
COORDINATOR_EVENTS = {
    "family_block_reservation_created",
    "family_block_item_terminalized",
    "family_block_execution_incident",
    "family_block_terminalized",
}
ENDPOINT_EVENTS = {
    "item_execution_started",
    "source_terminalized",
    "pre_generation_failure_terminalized",
    "uncertain_execution_incident",
}
SOURCE_DISPOSITIONS = {"source_usable", "source_reliability_failure"}
TERMINAL_DISPOSITIONS = SOURCE_DISPOSITIONS | {"pre_generation_failure_zero_cost"}


class FullStudyExecutionError(RuntimeError):
    """Execution would violate a frozen identity, source, or budget rule."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ledger_digest(entry: Mapping[str, Any]) -> str:
    body = dict(entry)
    body.pop("entry_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _contains_secret_key(value: object) -> bool:
    forbidden = {
        "api_key",
        "authorization",
        "cloudflare_ai_gateway_token",
        "environment",
        "mcp_token",
        "password",
        "secret",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_secret_key(item) for item in value)
    return False


def _load_ledger(path: Path, *, role: str) -> list[dict[str, Any]]:
    if role not in {"coordinator", "endpoint"}:
        raise FullStudyExecutionError(f"unsupported ledger role: {role}")
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise FullStudyExecutionError(f"ledger is not a regular file: {path}")
    allowed = COORDINATOR_EVENTS if role == "coordinator" else ENDPOINT_EVENTS
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise FullStudyExecutionError(f"blank {role} ledger line {number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise FullStudyExecutionError(f"invalid {role} ledger JSON line {number}") from error
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != LEDGER_SCHEMA
            or entry.get("ledger_role") != role
            or entry.get("sequence") != number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("event_type") not in allowed
            or entry.get("entry_sha256") != _ledger_digest(entry)
            or _contains_secret_key(entry)
        ):
            raise FullStudyExecutionError(f"{role} ledger integrity failed at line {number}")
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    return entries


def _append_ledger(path: Path, *, role: str, event: Mapping[str, Any]) -> dict[str, Any]:
    protected = {
        "schema_version",
        "ledger_role",
        "sequence",
        "recorded_at",
        "previous_entry_sha256",
        "entry_sha256",
    }
    if protected & set(event) or _contains_secret_key(event):
        raise FullStudyExecutionError("ledger event is protected or secret-bearing")
    entries = _load_ledger(path, role=role)
    entry = {
        "schema_version": LEDGER_SCHEMA,
        "ledger_role": role,
        "sequence": len(entries) + 1,
        "recorded_at": _utc_now(),
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        **dict(event),
    }
    allowed = COORDINATOR_EVENTS if role == "coordinator" else ENDPOINT_EVENTS
    if entry.get("event_type") not in allowed:
        raise FullStudyExecutionError("unsupported family-block ledger event")
    entry["entry_sha256"] = _ledger_digest(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    line = _canonical(entry) + b"\n"
    try:
        if os.write(descriptor, line) != len(line):
            raise OSError("short family-block ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


@contextmanager
def _ledger_lock(path: Path) -> Iterable[None]:
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _item_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {str(item["work_item_id"]): dict(item) for item in plan["work_items"]}
    if len(values) != 168:
        raise FullStudyExecutionError("study must contain 168 unique work items")
    return values


def _wave_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {str(wave["wave_id"]): dict(wave) for wave in plan["task_waves"]}
    if len(values) != 24:
        raise FullStudyExecutionError("study must contain 24 unique task waves")
    return values


def _block_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    values = {str(block["admission_block_id"]): dict(block) for block in plan["admission_blocks"]}
    if len(values) != 6:
        raise FullStudyExecutionError("study must contain six family-balanced blocks")
    return values


def _item_wave_id(plan: Mapping[str, Any], work_item_id: str) -> str:
    matches = [
        str(wave["wave_id"]) for wave in plan["task_waves"] if work_item_id in wave["work_item_ids"]
    ]
    if len(matches) != 1:
        raise FullStudyExecutionError("work item does not belong to exactly one task wave")
    return matches[0]


def _roots(plan: Mapping[str, Any], repo_root: Path) -> tuple[Path, dict[str, Path]]:
    coordinator = repo_root / str(plan["execution_roots"]["coordinator"])
    endpoints = {
        endpoint: repo_root / str(relative)
        for endpoint, relative in plan["execution_roots"]["endpoints"].items()
    }
    if set(endpoints) != set(study.ENDPOINTS):
        raise FullStudyExecutionError("execution roots do not cover exactly three endpoints")
    return coordinator, endpoints


def _manifest(plan: Mapping[str, Any], item: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    reference = item["manifest"]
    path = repo_root / str(reference["path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != reference["bytes"]
        or study._file_sha256(path) != reference["file_sha256"]
    ):
        raise FullStudyExecutionError("work-item manifest physical identity differs")
    document = study._regular_json(path)
    if (
        not study.verify_manifest_content_address(document)
        or document["content_address"]["digest"] != reference["semantic_sha256"]
    ):
        raise FullStudyExecutionError("work-item manifest content address differs")
    return document


def _prospective_policy_from_manifest(manifest: Mapping[str, Any]):
    """Parse a prospective manifest literally without changing v4 reconstruction."""
    from .response_envelope_route_v4 import _policy_from_manifest

    execution_policy = ((manifest.get("run_design") or {}).get("execution_policy") or {})
    policy = replace(
        _policy_from_manifest(manifest),
        pair_arm_scheduling=str(execution_policy["pair_arm_scheduling"]),
    )
    policy.validate()
    return policy


def _policy(plan: Mapping[str, Any], item: Mapping[str, Any], repo_root: Path):
    policy = _prospective_policy_from_manifest(_manifest(plan, item, repo_root))
    coordinate = item["route_coordinate"]
    checks = {
        "max_tool_calls_per_round": 13,
        "max_tool_calls_total": 13,
        "max_tool_rounds": 3,
        "max_output_tokens": 8192,
        "max_intermediate_tokens": 8192,
        "max_provider_attempts": 2,
        "evidence_protocol": "matched_evidence_v2",
        "final_response_mode": "plain_text",
        "pair_arm_scheduling": "concurrent",
    }
    if (
        any(getattr(policy, key) != value for key, value in checks.items())
        or policy.sha256 != coordinate["execution_policy_sha256"]
        or policy.intermediate_reasoning_effort != coordinate["intermediate_reasoning_effort"]
        or policy.final_reasoning_effort != coordinate["final_reasoning_effort"]
    ):
        raise FullStudyExecutionError("runtime policy differs from frozen common protocol")
    return policy


async def _attest_endpoint(*, api_base: str, api_key: str, endpoint_id: str) -> dict[str, Any]:
    if not api_key:
        raise FullStudyExecutionError("OpenRouter API key is absent")
    fixed = study.ENDPOINTS[endpoint_id]
    async with httpx.AsyncClient(
        base_url=api_base.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    ) as client:
        model, endpoint, dates = await v5._catalog_endpoint(
            client, fixed["model_id"], fixed["provider_endpoint"]
        )
    raw = v5.raw_endpoint_contract(endpoint)
    semantic = v5.semantic_endpoint_contract(raw)
    return {
        "endpoint_id": endpoint_id,
        "requested_model_id": fixed["model_id"],
        "requested_provider_endpoint": fixed["provider_endpoint"],
        "model": model,
        "raw_execution_contract": raw,
        "raw_execution_contract_sha256": study._sha256(raw),
        "semantic_execution_contract": semantic,
        "semantic_execution_contract_sha256": study._sha256(semantic),
        "response_dates": dates,
        "catalog_http_gets": 2,
        "provider_completion_requests": 0,
        "epicure_calls": 0,
    }


async def _attest_all_endpoints(
    *, plan: Mapping[str, Any], api_base: str, api_key: str
) -> list[dict[str, Any]]:
    records = await asyncio.gather(
        *(
            _attest_endpoint(api_base=api_base, api_key=api_key, endpoint_id=endpoint_id)
            for endpoint_id in study.ENDPOINTS
        )
    )
    for record in records:
        endpoint_id = str(record["endpoint_id"])
        frozen = plan["models"][endpoint_id]
        fixed = study.ENDPOINTS[endpoint_id]
        if (
            record["requested_model_id"] != fixed["model_id"]
            or record["requested_provider_endpoint"] != fixed["provider_endpoint"]
            or record["model"].get("canonical_slug") != fixed["canonical_model_slug"]
            or record["semantic_execution_contract_sha256"]
            != frozen["semantic_execution_contract_sha256"]
            or record["raw_execution_contract"].get("provider_name")
            != fixed["actual_provider_name"]
            or record["raw_execution_contract"].get("tag") != fixed["provider_endpoint"]
        ):
            raise FullStudyExecutionError(
                f"{endpoint_id} catalog identity or semantic contract drifted"
            )
    return sorted(records, key=lambda value: str(value["endpoint_id"]))


def _coordinator_state(
    plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    blocks = _block_map(plan)
    items = _item_map(plan)
    reservations: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    completed: dict[str, Mapping[str, Any]] = {}
    incidents: dict[str, list[Mapping[str, Any]]] = {}
    reservation_by_sha: dict[str, str] = {}
    for entry in entries:
        event = str(entry["event_type"])
        block_id = str(entry.get("admission_block_id") or "")
        if block_id not in blocks:
            raise FullStudyExecutionError("coordinator ledger names an unknown block")
        if entry.get("study_plan_sha256") != plan["artifact_sha256"]:
            raise FullStudyExecutionError("coordinator ledger plan binding differs")
        if event == "family_block_reservation_created":
            if block_id in reservations or block_id in completed:
                raise FullStudyExecutionError("family block was reserved more than once")
            block = blocks[block_id]
            if entry.get("work_item_ids") != block["work_item_ids"] or Decimal(
                str(entry.get("reserved_usd"))
            ) != Decimal(block["worst_case_reserve_usd"]):
                raise FullStudyExecutionError("block reservation differs from frozen block")
            reservations[block_id] = entry
            reservation_by_sha[str(entry["entry_sha256"])] = block_id
        elif event == "family_block_item_terminalized":
            item_id = str(entry.get("work_item_id") or "")
            reservation_sha = str(entry.get("block_reservation_entry_sha256") or "")
            if (
                item_id not in items
                or item_id in terminals
                or reservation_by_sha.get(reservation_sha) != block_id
                or item_id not in blocks[block_id]["work_item_ids"]
                or entry.get("disposition") not in TERMINAL_DISPOSITIONS
            ):
                raise FullStudyExecutionError("malformed or duplicate item terminal event")
            terminals[item_id] = entry
        elif event == "family_block_execution_incident":
            reservation_sha = str(entry.get("block_reservation_entry_sha256") or "")
            if reservation_by_sha.get(reservation_sha) != block_id:
                raise FullStudyExecutionError("incident lacks its block reservation")
            incidents.setdefault(block_id, []).append(entry)
        elif event == "family_block_terminalized":
            if block_id in completed or block_id not in reservations:
                raise FullStudyExecutionError("block terminalized without one reservation")
            expected = set(blocks[block_id]["work_item_ids"])
            observed = {
                item_id
                for item_id, terminal in terminals.items()
                if terminal.get("admission_block_id") == block_id
            }
            if observed != expected or incidents.get(block_id):
                raise FullStudyExecutionError("block terminalized incompletely or after incident")
            completed[block_id] = entry
    order = list(plan["block_execution_order"])
    reserved_order = [block_id for block_id in order if block_id in reservations]
    if reserved_order != order[: len(reserved_order)]:
        raise FullStudyExecutionError("block reservations are not a frozen-order prefix")
    completed_order = [block_id for block_id in order if block_id in completed]
    if completed_order != order[: len(completed_order)]:
        raise FullStudyExecutionError("completed blocks are not a frozen-order prefix")
    active = [block_id for block_id in reservations if block_id not in completed]
    if len(active) > 1:
        raise FullStudyExecutionError("more than one family block is active")
    if active and active[0] != order[len(completed_order)]:
        raise FullStudyExecutionError("active block is not the next frozen block")
    return {
        "reservations": reservations,
        "terminals": terminals,
        "completed": completed,
        "incidents": incidents,
        "active_block_id": active[0] if active else None,
    }


def _endpoint_state(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    started: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    incidents: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        item_id = str(entry.get("work_item_id") or "")
        event = entry.get("event_type")
        if not item_id:
            raise FullStudyExecutionError("endpoint ledger event lacks work-item identity")
        if event == "item_execution_started":
            if item_id in started:
                raise FullStudyExecutionError("work item was started more than once")
            started[item_id] = entry
        elif event in {
            "source_terminalized",
            "pre_generation_failure_terminalized",
        }:
            if item_id not in started or item_id in terminals or item_id in incidents:
                raise FullStudyExecutionError("endpoint terminal event order is invalid")
            terminals[item_id] = entry
        elif event == "uncertain_execution_incident":
            if item_id not in started or item_id in terminals or item_id in incidents:
                raise FullStudyExecutionError("endpoint incident event order is invalid")
            incidents[item_id] = entry
    return {"started": started, "terminals": terminals, "incidents": incidents}


def _journal_evidence(source_root: Path, work_item_id: str) -> dict[str, Any]:
    from .run_journal import load_run_journal, scan_recovery_journals

    states = scan_recovery_journals(source_root, dataset_work_item_id=work_item_id)
    request_started = 0
    descriptors: list[dict[str, Any]] = []
    for state in states:
        entries = load_run_journal(state.path)
        request_started += sum(
            entry.get("event_type") == "provider_attempt"
            and isinstance(entry.get("payload"), Mapping)
            and entry["payload"].get("event_type") == "request_started"
            for entry in entries
        )
        descriptors.append(
            {
                "filename": state.path.name,
                "sha256": state.journal_sha256,
                "head_entry_sha256": state.head_entry_sha256,
                "entry_count": state.entry_count,
                "finalized": state.finalized,
                "generation_ids": list(state.generation_ids),
                "unreconciled_generation_ids": list(state.unreconciled_generation_ids),
                "uncertain_attempt_ids": list(state.uncertain_attempt_ids),
            }
        )
    return {
        "journal_count": len(states),
        "request_started_count": request_started,
        "journals": descriptors,
    }


def _source_for_item(endpoint_root: Path, work_item_id: str):
    return v5._source_map(endpoint_root / "source").get(work_item_id)


def _source_terminal_payload(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    source_record: tuple[Path, dict[str, Any], str],
    repo_root: Path,
    endpoint_root: Path,
) -> dict[str, Any]:
    path, _, digest = source_record
    pair = study.pair_audit(plan=plan, item=item, source_path=path, repo_root=repo_root)
    accounting = pair.get("accounting") or {}
    if accounting.get("reconciled") is not True:
        raise FullStudyExecutionError("source generation cost is not fully reconciled")
    audit_path = study._write_artifact(
        endpoint_root / "audits",
        f"reasoning-effort-task-wave-pair-audit-{item['work_item_id'][:12]}",
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


def _global_state(repo_root: Path) -> tuple[Decimal, list[dict[str, Any]]]:
    from .frontier_coverage_repair_executor import _global_ledger_state

    artifacts = repo_root / "flavourbench/artifacts"
    active, blockers = _global_ledger_state(
        ledger_path=artifacts / "frontier-contract/ledger.jsonl",
        artifact_directory=artifacts / "live-smoke",
        corrections_directory=artifacts / "corrections",
        reconciliation_directory=artifacts / "frontier-contract/reconciliations",
    )
    return active, [dict(value) for value in blockers]


def _accounting(
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
    coordinator_ledger: Path,
    endpoint_roots: Mapping[str, Path],
) -> dict[str, Any]:
    receipt_ref = plan["source_artifacts"]["v6_rebased_receipt"]
    baseline_receipt = study._regular_json(repo_root / receipt_ref["path"])
    baseline_record = study.reconstruct_current_exposure(
        repo_root=repo_root, receipt=baseline_receipt
    )
    baseline = Decimal(baseline_record["current_total_exposure_usd"])
    entries = _load_ledger(coordinator_ledger, role="coordinator")
    state = _coordinator_state(plan, entries)
    items = _item_map(plan)
    blocks = _block_map(plan)
    completed_cost = Decimal(0)
    completed_sources = 0
    for block_id in state["completed"]:
        block_cost = Decimal(0)
        for item_id in blocks[block_id]["work_item_ids"]:
            terminal = state["terminals"][item_id]
            disposition = terminal["disposition"]
            endpoint = items[item_id]["route_coordinate"]["endpoint_id"]
            source = _source_for_item(endpoint_roots[endpoint], item_id)
            if disposition in SOURCE_DISPOSITIONS:
                if source is None:
                    raise FullStudyExecutionError("terminal source is absent")
                payload = _source_terminal_payload(
                    plan=plan,
                    item=items[item_id],
                    source_record=source,
                    repo_root=repo_root,
                    endpoint_root=endpoint_roots[endpoint],
                )
                if payload["source_artifact_sha256"] != terminal.get(
                    "source_artifact_sha256"
                ) or payload["actual_cost_usd"] != terminal.get("actual_cost_usd"):
                    raise FullStudyExecutionError("terminal source accounting differs")
                block_cost += Decimal(payload["actual_cost_usd"])
                completed_sources += 1
            elif source is not None:
                raise FullStudyExecutionError("zero-cost pre-generation item has a source")
        completed = state["completed"][block_id]
        if Decimal(str(completed.get("actual_cost_usd"))) != block_cost:
            raise FullStudyExecutionError("completed block cost does not rederive")
        completed_cost += block_cost
    active_block = state["active_block_id"]
    active_reserve = (
        Decimal(blocks[active_block]["worst_case_reserve_usd"]) if active_block else Decimal(0)
    )
    global_active, global_blockers = _global_state(repo_root)
    current = baseline + completed_cost + active_reserve + global_active
    order = list(plan["block_execution_order"])
    next_block_id = (
        order[len(state["completed"])] if not active_block and len(state["completed"]) < 6 else None
    )
    next_reserve = (
        Decimal(blocks[next_block_id]["worst_case_reserve_usd"]) if next_block_id else Decimal(0)
    )
    projected = current + next_reserve
    incident_blockers = [
        {
            "gate": "active_family_block_uncertain_delivery",
            "admission_block_id": block_id,
            "incident_entry_sha256": incident["entry_sha256"],
        }
        for block_id, incidents in state["incidents"].items()
        if block_id not in state["completed"]
        for incident in incidents
    ]
    blockers = [*global_blockers, *incident_blockers]
    return {
        "currency": "USD",
        "baseline": baseline_record,
        "completed_family_blocks": len(state["completed"]),
        "completed_task_waves": len(state["completed"]) * 4,
        "completed_source_pairs": completed_sources,
        "completed_block_actual_and_exposure_usd": study._decimal_text(completed_cost),
        "active_block_id": active_block,
        "active_block_full_reserve_usd": study._decimal_text(active_reserve),
        "global_active_reservation_usd": study._decimal_text(global_active),
        "current_total_exposure_usd": study._decimal_text(current),
        "next_block_id": next_block_id,
        "next_block_worst_case_reserve_usd": study._decimal_text(next_reserve),
        "next_block_projected_total_usd": study._decimal_text(projected),
        "admission_ceiling_usd": "85",
        "hard_cap_usd": "100",
        "blockers": blockers,
        "new_block_admission_allowed": bool(
            next_block_id
            and not active_block
            and not blockers
            and projected <= study.ADMISSION_CEILING_USD
            and projected <= study.HARD_CAP_USD
        ),
        "active_block_resume_allowed": bool(
            active_block and not blockers and current <= study.HARD_CAP_USD
        ),
        "active_block_sources_counted_inside_reserve_only": True,
    }


def _live_args(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    repo_root: Path,
    source: Path,
    raw_endpoint_sha256: str,
) -> argparse.Namespace:
    from .live_smoke import LIVE_SMOKE_CONFIRMATION

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


def _append_item_terminal(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    disposition = str(payload["disposition"])
    task_wave_id = _item_wave_id(plan, str(item["work_item_id"]))
    endpoint_event = _append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": (
                "source_terminalized"
                if disposition in SOURCE_DISPOSITIONS
                else "pre_generation_failure_terminalized"
            ),
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": task_wave_id,
            "work_item_id": item["work_item_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            **dict(payload),
            "replay_permitted": False,
            "rank_eligible": False,
        },
    )
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_item_terminalized",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": task_wave_id,
            "work_item_id": item["work_item_id"],
            "endpoint_id": item["route_coordinate"]["endpoint_id"],
            "variant_id": item["route_coordinate"]["variant_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            "endpoint_terminal_entry_sha256": endpoint_event["entry_sha256"],
            **dict(payload),
            "replay_permitted": False,
            "rank_eligible": False,
        },
    )


def _append_uncertain_incident(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    coordinator_ledger: Path,
    evidence: Mapping[str, Any],
    error: BaseException | None,
) -> dict[str, Any]:
    error_text = str(error) if error is not None else "recovered active execution state"
    common = {
        "study_plan_sha256": plan["artifact_sha256"],
        "admission_block_id": block["admission_block_id"],
        "task_wave_id": _item_wave_id(plan, str(item["work_item_id"])),
        "work_item_id": item["work_item_id"],
        "block_reservation_entry_sha256": reservation["entry_sha256"],
        "incident": "provider_request_started_without_reconciled_source",
        "journal_evidence": dict(evidence),
        "error_type": type(error).__name__ if error is not None else "RecoveredState",
        "error_sha256": hashlib.sha256(error_text.encode()).hexdigest(),
        "full_family_block_reserve_retained_usd": block["worst_case_reserve_usd"],
        "replay_permitted": False,
    }
    endpoint = _append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={"event_type": "uncertain_execution_incident", **common},
    )
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_execution_incident",
            **common,
            "endpoint_incident_entry_sha256": endpoint["entry_sha256"],
        },
    )


def _terminalize_block(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    coordinator_ledger: Path,
) -> dict[str, Any]:
    entries = _load_ledger(coordinator_ledger, role="coordinator")
    state = _coordinator_state(plan, entries)
    block_id = str(block["admission_block_id"])
    if block_id in state["completed"]:
        return dict(state["completed"][block_id])
    if state["incidents"].get(block_id):
        raise FullStudyExecutionError("uncertain family block cannot be terminalized")
    terminals = [state["terminals"].get(item_id) for item_id in block["work_item_ids"]]
    if any(value is None for value in terminals):
        raise FullStudyExecutionError("cannot terminalize an incomplete family block")
    actual = sum(
        Decimal(str(value.get("actual_cost_usd") or "0"))
        for value in terminals
        if value is not None
    )
    return _append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_terminalized",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block_id,
            "block_ordinal": block["block_ordinal"],
            "wave_ids": block["wave_ids"],
            "task_ids": block["task_ids"],
            "task_families": block["task_families"],
            "block_reservation_entry_sha256": state["reservations"][block_id]["entry_sha256"],
            "item_terminal_entry_sha256s": [value["entry_sha256"] for value in terminals],
            "terminal_pairs": 28,
            "source_pairs": sum(value["disposition"] in SOURCE_DISPOSITIONS for value in terminals),
            "pre_generation_failure_pairs": sum(
                value["disposition"] == "pre_generation_failure_zero_cost" for value in terminals
            ),
            "actual_cost_usd": study._decimal_text(actual),
            "conservative_exposure_usd": study._decimal_text(actual),
            "whole_family_block_reservation_released": True,
            "replay_permitted": False,
        },
    )


async def execute_one_block(
    *,
    plan: Mapping[str, Any],
    human_protocol: Mapping[str, Any],
    bound_preflight: Mapping[str, Any],
    repo_root: Path,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    from .config import get_settings
    from .frontier_contract_runner import AdmissionDenied, _exclusive_runner_lock
    from .live_smoke import live_smoke
    from .reasoning_effort_route_gate_v4 import (
        _policy_environment,
        _require_live_environment_before_reservation,
    )

    study.validate_plan(plan, repo_root=repo_root)
    study.verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    study.verify_bound_preflight(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
    )
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
                outcomes.append(
                    {
                        "decision": "blocked_active_block_uncertain_delivery",
                        "admission_block_id": active_id,
                    }
                )
                block = block_map[active_id]
                reservation = state["reservations"][active_id]
            elif active_id:
                if accounting["active_block_resume_allowed"] is not True:
                    raise AdmissionDenied("active family block is not budget-safe to resume")
                block = block_map[active_id]
                reservation = state["reservations"][active_id]
            else:
                if accounting["new_block_admission_allowed"] is not True:
                    outcomes.append({"decision": "no_new_block_admitted", "budget": accounting})
                    block = None
                    reservation = None
                else:
                    block = block_map[str(accounting["next_block_id"])]
                    reservation = None

            if block is not None and not state["incidents"].get(block["admission_block_id"]):
                # Recover durable state before any network operation.  Started
                # items are never replayed: a reconciled source is finalized;
                # no request-start is a zero-cost terminal failure; any request
                # start without a source retains the full block reservation.
                if reservation is not None:
                    for item_id in block["work_item_ids"]:
                        state = _coordinator_state(
                            plan, _load_ledger(coordinator_ledger, role="coordinator")
                        )
                        if item_id in state["terminals"]:
                            continue
                        item = item_map[item_id]
                        endpoint_id = item["route_coordinate"]["endpoint_id"]
                        endpoint_root = endpoint_roots[endpoint_id]
                        endpoint_ledger = endpoint_root / "ledger.jsonl"
                        endpoint_state = _endpoint_state(
                            _load_ledger(endpoint_ledger, role="endpoint")
                        )
                        source = _source_for_item(endpoint_root, item_id)
                        if source is not None:
                            try:
                                payload = _source_terminal_payload(
                                    plan=plan,
                                    item=item,
                                    source_record=source,
                                    repo_root=repo_root,
                                    endpoint_root=endpoint_root,
                                )
                            except FullStudyExecutionError as error:
                                evidence = _journal_evidence(endpoint_root / "source", item_id)
                                _append_uncertain_incident(
                                    plan=plan,
                                    block=block,
                                    item=item,
                                    reservation=reservation,
                                    endpoint_ledger=endpoint_ledger,
                                    coordinator_ledger=coordinator_ledger,
                                    evidence=evidence,
                                    error=error,
                                )
                                outcomes.append(
                                    {
                                        "work_item_id": item_id,
                                        "decision": "unreconciled_source_stop",
                                    }
                                )
                                break
                            if item_id not in endpoint_state["started"]:
                                raise FullStudyExecutionError("source exists without a start event")
                            _append_item_terminal(
                                plan=plan,
                                block=block,
                                item=item,
                                reservation=reservation,
                                endpoint_ledger=endpoint_ledger,
                                coordinator_ledger=coordinator_ledger,
                                payload=payload,
                            )
                            outcomes.append(
                                {"work_item_id": item_id, "decision": "recovered_source_no_replay"}
                            )
                            continue
                        if item_id in endpoint_state["incidents"]:
                            raise FullStudyExecutionError(
                                "endpoint incident lacks coordinator incident"
                            )
                        if item_id in endpoint_state["started"]:
                            evidence = _journal_evidence(endpoint_root / "source", item_id)
                            if evidence["request_started_count"]:
                                _append_uncertain_incident(
                                    plan=plan,
                                    block=block,
                                    item=item,
                                    reservation=reservation,
                                    endpoint_ledger=endpoint_ledger,
                                    coordinator_ledger=coordinator_ledger,
                                    evidence=evidence,
                                    error=None,
                                )
                                outcomes.append(
                                    {
                                        "work_item_id": item_id,
                                        "decision": "request_started_no_source_stop",
                                    }
                                )
                                break
                            payload = {
                                "disposition": "pre_generation_failure_zero_cost",
                                "actual_cost_usd": "0",
                                "journal_evidence": evidence,
                                "error_type": "RecoveredPreGenerationFailure",
                                "error_sha256": hashlib.sha256(
                                    b"recovered started item with no provider request"
                                ).hexdigest(),
                            }
                            _append_item_terminal(
                                plan=plan,
                                block=block,
                                item=item,
                                reservation=reservation,
                                endpoint_ledger=endpoint_ledger,
                                coordinator_ledger=coordinator_ledger,
                                payload=payload,
                            )
                            outcomes.append(
                                {
                                    "work_item_id": item_id,
                                    "decision": "recovered_pre_generation_failure",
                                }
                            )

                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                block_id = str(block["admission_block_id"])
                if not state["incidents"].get(block_id):
                    remaining = [
                        item_id
                        for item_id in block["work_item_ids"]
                        if item_id not in state["terminals"]
                    ]
                    if remaining:
                        attestations = await _attest_all_endpoints(
                            plan=plan, api_base=api_base, api_key=api_key
                        )
                        attestation_document = {
                            "schema_version": ATTESTATION_SCHEMA,
                            "record_role": "all_endpoint_pre_block_zero_generation_attestation",
                            "study_plan_sha256": plan["artifact_sha256"],
                            "admission_block_id": block_id,
                            "wave_ids": block["wave_ids"],
                            "records": attestations,
                            "counts": {
                                "catalog_http_gets": 6,
                                "provider_completion_requests": 0,
                                "epicure_calls": 0,
                            },
                            "all_semantic_contracts_match": True,
                            "provider_substitution_performed": False,
                        }
                        attestation_path = study._write_artifact(
                            coordinator_root / "endpoint-attestations",
                            f"reasoning-effort-family-block-{block['block_ordinal']:02d}-attestations",
                            attestation_document,
                        )
                        attest_ref = study._file_ref(repo_root, attestation_path)
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
                                    "budget or schedule changed before whole-block reservation"
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
                        attestation_by_endpoint = {
                            value["endpoint_id"]: value for value in attestations
                        }
                        for item_id in remaining:
                            state = _coordinator_state(
                                plan,
                                _load_ledger(coordinator_ledger, role="coordinator"),
                            )
                            if state["incidents"].get(block_id):
                                break
                            item = item_map[item_id]
                            endpoint_id = item["route_coordinate"]["endpoint_id"]
                            endpoint_root = endpoint_roots[endpoint_id]
                            endpoint_root.mkdir(parents=True, exist_ok=True)
                            source_root = endpoint_root / "source"
                            source_root.mkdir(parents=True, exist_ok=True)
                            endpoint_ledger = endpoint_root / "ledger.jsonl"
                            endpoint_state = _endpoint_state(
                                _load_ledger(endpoint_ledger, role="endpoint")
                            )
                            if item_id in endpoint_state["started"]:
                                raise FullStudyExecutionError("started item reached replay loop")
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
                                    "block_reservation_entry_sha256": reservation["entry_sha256"],
                                    "raw_endpoint_execution_contract_sha256": (
                                        attestation_by_endpoint[endpoint_id][
                                            "raw_execution_contract_sha256"
                                        ]
                                    ),
                                    "replay_permitted": False,
                                },
                            )
                            policy = _policy(plan, item, repo_root)
                            args = _live_args(
                                plan=plan,
                                item=item,
                                repo_root=repo_root,
                                source=source_root,
                                raw_endpoint_sha256=attestation_by_endpoint[endpoint_id][
                                    "raw_execution_contract_sha256"
                                ],
                            )
                            error: BaseException | None = None
                            try:
                                with _policy_environment(
                                    policy=policy,
                                    endpoint=attestation_by_endpoint[endpoint_id][
                                        "raw_execution_contract"
                                    ],
                                ):
                                    settings = get_settings()
                                    if (
                                        settings.execution_mode != "live"
                                        or not settings.live_authorized
                                    ):
                                        raise AdmissionDenied(
                                            "live authority changed after block reservation"
                                        )
                                    provider_pair_invocations += 1
                                    await live_smoke(args)
                            except Exception as caught:
                                error = caught
                            source = _source_for_item(endpoint_root, item_id)
                            if source is not None:
                                try:
                                    payload = _source_terminal_payload(
                                        plan=plan,
                                        item=item,
                                        source_record=source,
                                        repo_root=repo_root,
                                        endpoint_root=endpoint_root,
                                    )
                                except FullStudyExecutionError as audit_error:
                                    evidence = _journal_evidence(source_root, item_id)
                                    _append_uncertain_incident(
                                        plan=plan,
                                        block=block,
                                        item=item,
                                        reservation=reservation,
                                        endpoint_ledger=endpoint_ledger,
                                        coordinator_ledger=coordinator_ledger,
                                        evidence=evidence,
                                        error=audit_error,
                                    )
                                    outcomes.append(
                                        {
                                            "work_item_id": item_id,
                                            "decision": "unreconciled_source_stop",
                                        }
                                    )
                                    break
                                _append_item_terminal(
                                    plan=plan,
                                    block=block,
                                    item=item,
                                    reservation=reservation,
                                    endpoint_ledger=endpoint_ledger,
                                    coordinator_ledger=coordinator_ledger,
                                    payload=payload,
                                )
                                outcomes.append(
                                    {
                                        "work_item_id": item_id,
                                        "decision": payload["disposition"],
                                        "error_type": type(error).__name__ if error else None,
                                    }
                                )
                                continue
                            evidence = _journal_evidence(source_root, item_id)
                            if evidence["request_started_count"]:
                                _append_uncertain_incident(
                                    plan=plan,
                                    block=block,
                                    item=item,
                                    reservation=reservation,
                                    endpoint_ledger=endpoint_ledger,
                                    coordinator_ledger=coordinator_ledger,
                                    evidence=evidence,
                                    error=error,
                                )
                                outcomes.append(
                                    {
                                        "work_item_id": item_id,
                                        "decision": "request_started_no_source_stop",
                                    }
                                )
                                break
                            text = str(error) if error is not None else "no source after invocation"
                            payload = {
                                "disposition": "pre_generation_failure_zero_cost",
                                "actual_cost_usd": "0",
                                "journal_evidence": evidence,
                                "error_type": type(error).__name__ if error else "MissingSource",
                                "error_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            }
                            _append_item_terminal(
                                plan=plan,
                                block=block,
                                item=item,
                                reservation=reservation,
                                endpoint_ledger=endpoint_ledger,
                                coordinator_ledger=coordinator_ledger,
                                payload=payload,
                            )
                            outcomes.append(
                                {
                                    "work_item_id": item_id,
                                    "decision": "pre_generation_failure_zero_cost",
                                }
                            )

                state = _coordinator_state(
                    plan, _load_ledger(coordinator_ledger, role="coordinator")
                )
                if not state["incidents"].get(block_id) and all(
                    item_id in state["terminals"] for item_id in block["work_item_ids"]
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
        "record_role": "single_atomic_family_block_execution_receipt",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "admission_block_id": block["admission_block_id"] if block is not None else None,
        "block_ordinal": block["block_ordinal"] if block is not None else None,
        "wave_ids": block["wave_ids"] if block is not None else [],
        "new_block_admitted": new_block_admitted,
        "provider_pair_invocations": provider_pair_invocations,
        "outcomes": outcomes,
        "endpoint_attestation": attest_ref,
        "final_budget": final_accounting,
        "coordinator_ledger": {
            "path": study._relative(repo_root, coordinator_ledger),
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
        "reasoning-effort-family-block-execution-receipt",
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
        raise SystemExit("exact one-family-block confirmation token is required")
    if args.max_new_family_blocks != 1:
        raise SystemExit("--max-new-family-blocks must be exactly 1")
    repo_root = args.repo_root.resolve()
    plan = study._regular_json(args.plan)
    human_protocol = study._regular_json(args.human_protocol)
    bound_preflight = study._regular_json(args.bound_preflight)
    study.validate_plan(plan, repo_root=repo_root)
    study.verify_human_protocol_binding(plan=plan, human_protocol=human_protocol)
    study.verify_bound_preflight(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
    )
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
