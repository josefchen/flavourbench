"""Close the zero-call reasoning block stopped by a missing live-smoke symbol.

This recovery is deliberately specific to plan ``99b8f7…`` and its first
family-balanced block.  It performs no network operation, verifies the exact
post-crash state, terminalizes all 28 scheduled pairs as benchmark-pipeline
pre-generation failures, releases the atomic reservation, and emits a
content-addressed receipt.  Incident-affected identifiers are never replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import live_smoke
from . import reasoning_effort_full_study_executor_v1 as executor
from . import reasoning_effort_full_study_v1 as study
from . import reasoning_effort_pregen_recovery_v1 as prior
from .frontier_contract_runner import _exclusive_runner_lock

PLAN_SHA256 = "99b8f70ae81aa3a7b7e79a45bb4253cb58d26306f90ab5b9c4f09a6938f1a301"
HUMAN_PROTOCOL_SHA256 = "cd2a234f617158304a5eb4efed1c6e34198cd857f2de124b10dee09fdec370a8"
BOUND_PREFLIGHT_SHA256 = "58d509d8c9c4276ad9c497789652a9ba55a50320846b7dda5eb72853bfe25910"
BLOCK_ID = "c66c637dd057e13c18f36894e1a7b15093760058245306e8df1188cc4f725c7d"
RESERVATION_ENTRY_SHA256 = "98a63bb6682ac744f25957c7924b36580cd91b0193802cf8b1558305cc15abca"
FIRST_WORK_ITEM_ID = "2730b87bb45b420ce01388152c2734d07ea32b08723844f0f8f47519704a5916"
FIRST_START_ENTRY_SHA256 = "f0c21bd5c10ea747e2f8fa1287ad631b67e01573c6638a653db590c2a0d60fc2"
ATTESTATION_SEMANTIC_SHA256 = "c289e868e2cef9f7b1f4c7462a63cc54b5c1944cb25c982e21080f922dd2e131"
ATTESTATION_FILE_SHA256 = "26c52e32b72d31172d9b95243872068277f0f635f134f24ded3f5da1525b0d32"
INITIAL_COORDINATOR_LEDGER_SHA256 = (
    "59feb7d0989fdae0c92c0ee218780265e20a53c52a64243032c507876e163d1c"
)
INITIAL_GEMINI_LEDGER_SHA256 = "743fe02d1ce9108a3185c38a14441342d7fec368ded42c11dfecffda32daeb97"
GLOBAL_LEDGER_SHA256 = "82dea23af42c2b26b6c489beb35ac5f09e560fae10a0fb7df6875bd44851b29f"
EXECUTOR_FILE_SHA256 = "aa13f05ccaf7b85c91f3918f158fd76f242c7780674d82f8cd872fcb656d6ad4"
LIVE_SMOKE_FILE_SHA256 = "2cfdb38052df96082e74ad603e06d7d49701416a4cbc7c5b0fae7dafab84a42c"

CONTRACT_SCHEMA = "flavourbench-reasoning-effort-import-recovery-contract-v1"
INCIDENT_SCHEMA = "flavourbench-reasoning-effort-import-incident-v1"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-import-recovery-receipt-v1"
CONFIRMATION = "CLOSE_99B8_IMPORT_MISMATCH_WITH_ZERO_CALLS"


class RecoveryError(RuntimeError):
    """The live state cannot be closed under this narrow recovery contract."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"expected JSON object: {path}")
    return value


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("artifact_sha256", None)
    document["artifact_sha256"] = study._sha256(document)
    return document


def _physical_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RecoveryError(f"not a regular recovery input: {resolved}")
    return {
        "path": str(resolved.relative_to(repo_root.resolve())),
        "bytes": resolved.stat().st_size,
        "file_sha256": study._file_sha256(resolved),
    }


def _bindings(
    *,
    plan_path: Path,
    human_protocol_path: Path,
    bound_preflight_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load(plan_path)
    human = _load(human_protocol_path)
    bound = _load(bound_preflight_path)
    study.validate_plan(plan, repo_root=repo_root)
    study.verify_human_protocol_binding(plan=plan, human_protocol=human)
    study.verify_bound_preflight(plan=plan, human_protocol=human, bound_preflight=bound)
    if (
        plan.get("artifact_sha256") != PLAN_SHA256
        or human.get("artifact_sha256") != HUMAN_PROTOCOL_SHA256
        or bound.get("artifact_sha256") != BOUND_PREFLIGHT_SHA256
    ):
        raise RecoveryError("recovery bindings differ from the incident chain")
    return plan, human, bound


def _execution_files(plan: Mapping[str, Any], repo_root: Path) -> list[Path]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    return sorted(
        (
            path.resolve()
            for root in [coordinator, *endpoints.values()]
            if root.exists()
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".lock")
        ),
        key=str,
    )


def _attestation_path(plan: Mapping[str, Any], repo_root: Path) -> Path:
    coordinator, _ = executor._roots(plan, repo_root)
    return coordinator / "endpoint-attestations" / (
        "reasoning-effort-family-block-01-attestations-"
        f"{ATTESTATION_SEMANTIC_SHA256}.json"
    )


def _verify_attestation(plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path = _attestation_path(plan, repo_root)
    document = _load(path)
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("artifact_sha256") != ATTESTATION_SEMANTIC_SHA256
        or study._sha256(body) != ATTESTATION_SEMANTIC_SHA256
        or study._file_sha256(path) != ATTESTATION_FILE_SHA256
        or document.get("study_plan_sha256") != PLAN_SHA256
        or document.get("admission_block_id") != BLOCK_ID
        or document.get("counts")
        != {"catalog_http_gets": 6, "provider_completion_requests": 0, "epicure_calls": 0}
        or document.get("all_semantic_contracts_match") is not True
        or document.get("provider_substitution_performed") is not False
    ):
        raise RecoveryError("catalog-only endpoint attestation differs")
    return document


def _verify_defect(repo_root: Path) -> dict[str, Any]:
    executor_path = Path(executor.__file__).resolve()
    smoke_path = Path(live_smoke.__file__).resolve()
    if (
        study._file_sha256(executor_path) != EXECUTOR_FILE_SHA256
        or study._file_sha256(smoke_path) != LIVE_SMOKE_FILE_SHA256
    ):
        raise RecoveryError("incident source files differ")
    executor_text = executor_path.read_text(encoding="utf-8")
    if "from .live_smoke import LIVE_SMOKE_CONFIRMATION" not in executor_text:
        raise RecoveryError("the exact missing-symbol import is absent")
    if hasattr(live_smoke, "LIVE_SMOKE_CONFIRMATION"):
        raise RecoveryError("the missing live-smoke symbol unexpectedly exists")
    if getattr(live_smoke, "CONFIRMATION", None) != "UNRANKED_REAL_SMOKE":
        raise RecoveryError("the actual live-smoke confirmation differs")
    return {
        "executor": _physical_ref(repo_root, executor_path),
        "live_smoke": _physical_ref(repo_root, smoke_path),
        "missing_symbol": "LIVE_SMOKE_CONFIRMATION",
        "available_symbol": "CONFIRMATION",
        "available_value_sha256": hashlib.sha256(b"UNRANKED_REAL_SMOKE").hexdigest(),
        "failure_boundary": "before_live_argument_construction_and_before_provider_invocation",
    }


def _verify_original_state(plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    expected = {
        (coordinator / "ledger.jsonl").resolve(): INITIAL_COORDINATOR_LEDGER_SHA256,
        (endpoints["gemini"] / "ledger.jsonl").resolve(): INITIAL_GEMINI_LEDGER_SHA256,
        _attestation_path(plan, repo_root).resolve(): ATTESTATION_FILE_SHA256,
    }
    observed = _execution_files(plan, repo_root)
    if observed != sorted(expected, key=str):
        raise RecoveryError("execution-root inventory differs from the frozen crash state")
    for path, digest in expected.items():
        if study._file_sha256(path) != digest:
            raise RecoveryError(f"incident-state file differs: {path}")

    global_ledger = repo_root / "flavourbench/artifacts/frontier-contract/ledger.jsonl"
    if study._file_sha256(global_ledger) != GLOBAL_LEDGER_SHA256:
        raise RecoveryError("global paid-call ledger changed after the incident")
    coordinator_entries = executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")
    state = executor._coordinator_state(plan, coordinator_entries)
    if (
        len(coordinator_entries) != 1
        or state["active_block_id"] != BLOCK_ID
        or set(state["reservations"]) != {BLOCK_ID}
        or state["reservations"][BLOCK_ID]["entry_sha256"] != RESERVATION_ENTRY_SHA256
        or state["terminals"]
        or state["incidents"]
        or state["completed"]
    ):
        raise RecoveryError("coordinator crash state differs")
    for endpoint_id, endpoint_root in endpoints.items():
        entries = executor._load_ledger(endpoint_root / "ledger.jsonl", role="endpoint")
        endpoint_state = executor._endpoint_state(entries)
        expected_started = {FIRST_WORK_ITEM_ID} if endpoint_id == "gemini" else set()
        if (
            set(endpoint_state["started"]) != expected_started
            or endpoint_state["terminals"]
            or endpoint_state["incidents"]
        ):
            raise RecoveryError(f"{endpoint_id} crash state differs")
    for endpoint_root in endpoints.values():
        source_root = endpoint_root / "source"
        if source_root.exists() and any(source_root.rglob("*")):
            raise RecoveryError("source or journal evidence exists in an endpoint root")
    return {"coordinator": coordinator, "endpoints": endpoints, "global_ledger": global_ledger}


def build_incident(
    *,
    plan_path: Path,
    human_protocol_path: Path,
    bound_preflight_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    plan, _, _ = _bindings(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    )
    roots = _verify_original_state(plan, repo_root)
    attestation = _verify_attestation(plan, repo_root)
    defect = _verify_defect(repo_root)
    return _with_hash(
        {
            "schema_version": INCIDENT_SCHEMA,
            "record_role": "benchmark_pipeline_pre_generation_import_incident",
            "incident_code": "missing_live_smoke_confirmation_symbol",
            "failure_owner": "benchmark_pipeline",
            "study_plan_sha256": PLAN_SHA256,
            "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
            "bound_preflight_sha256": BOUND_PREFLIGHT_SHA256,
            "admission_block_id": BLOCK_ID,
            "block_reservation_entry_sha256": RESERVATION_ENTRY_SHA256,
            "recovery_source": _physical_ref(repo_root, Path(__file__)),
            "defect": defect,
            "evidence": {
                "coordinator_ledger_sha256": INITIAL_COORDINATOR_LEDGER_SHA256,
                "gemini_ledger_sha256": INITIAL_GEMINI_LEDGER_SHA256,
                "global_paid_call_ledger": _physical_ref(repo_root, roots["global_ledger"]),
                "first_started_work_item_id": FIRST_WORK_ITEM_ID,
                "provider_completion_requests": 0,
                "epicure_calls": 0,
                "generation_ids": [],
                "source_artifacts": 0,
                "run_journals": 0,
                "catalog_attestation": study._file_ref(
                    repo_root, _attestation_path(plan, repo_root)
                ),
                "catalog_http_gets": attestation["counts"]["catalog_http_gets"],
            },
            "impact": {
                "scheduled_pairs": 28,
                "generated_pairs": 0,
                "provider_completion_requests": 0,
                "epicure_calls": 0,
                "actual_cost_usd": "0",
                "model_reliability_eligible": False,
                "preference_eligible": False,
                "rank_eligible": False,
            },
        }
    )


def build_contract(
    *,
    plan_path: Path,
    human_protocol_path: Path,
    bound_preflight_path: Path,
    incident: Mapping[str, Any],
    incident_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    plan, _, _ = _bindings(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    )
    if not study._artifact_ok(incident, INCIDENT_SCHEMA):
        raise RecoveryError("incident artifact is invalid")
    if dict(incident) != build_incident(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    ):
        raise RecoveryError("incident does not reproduce from frozen evidence")
    block = executor._block_map(plan)[BLOCK_ID]
    return _with_hash(
        {
            "schema_version": CONTRACT_SCHEMA,
            "record_role": "zero_network_append_only_atomic_block_import_recovery",
            "target_plan_sha256": PLAN_SHA256,
            "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
            "bound_preflight_sha256": BOUND_PREFLIGHT_SHA256,
            "target_block_id": BLOCK_ID,
            "target_reservation_entry_sha256": RESERVATION_ENTRY_SHA256,
            "pipeline_incident": study._file_ref(repo_root, incident_path),
            "recovery_source": _physical_ref(repo_root, Path(__file__)),
            "action": {
                "scheduled_work_items_terminalized": len(block["work_item_ids"]),
                "disposition": "pre_generation_failure_zero_cost",
                "actual_cost_usd": "0",
                "whole_block_reservation_released": True,
                "new_provider_requests": 0,
                "new_epicure_calls": 0,
                "replay_permitted": False,
                "rank_eligible": False,
            },
            "confirmation": CONFIRMATION,
        }
    )


def freeze(
    *,
    plan_path: Path,
    human_protocol_path: Path,
    bound_preflight_path: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    incident = build_incident(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    )
    incident_path = study._write_artifact(
        output_dir, "reasoning-effort-import-pipeline-incident", incident
    )
    contract = build_contract(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        incident=incident,
        incident_path=incident_path,
        repo_root=repo_root,
    )
    contract_path = study._write_artifact(
        output_dir, "reasoning-effort-import-recovery-contract", contract
    )
    return {"incident": incident_path, "contract": contract_path}


def _terminal_payload(
    *, item_id: str, contract: Mapping[str, Any], incident_ref: Mapping[str, Any]
) -> dict[str, Any]:
    first = item_id == FIRST_WORK_ITEM_ID
    error_text = (
        "executor imported an absent LIVE_SMOKE_CONFIRMATION symbol"
        if first
        else "atomic block cancelled after a pre-generation executor import failure"
    )
    return {
        "disposition": "pre_generation_failure_zero_cost",
        "actual_cost_usd": "0",
        "journal_evidence": {"journal_count": 0, "request_started_count": 0, "journals": []},
        "error_type": (
            "MissingLiveSmokeConfirmationImport"
            if first
            else "AtomicBlockCancelledBeforeProviderRequest"
        ),
        "error_sha256": hashlib.sha256(error_text.encode()).hexdigest(),
        "recovery_contract_sha256": contract["artifact_sha256"],
        "pipeline_incident": dict(incident_ref),
        "failure_owner": "benchmark_pipeline",
        "failure_code": "missing_live_smoke_confirmation_symbol",
        "provider_completion_requests": 0,
        "provider_request_started_count": 0,
        "epicure_calls": 0,
        "generation_ids": [],
        "source_artifact_count": 0,
        "model_reliability_eligible": False,
        "preference_eligible": False,
    }


def _verify_contract(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    incident: Mapping[str, Any],
    incident_path: Path,
    repo_root: Path,
) -> None:
    if (
        not study._artifact_ok(contract, CONTRACT_SCHEMA)
        or not study._artifact_ok(incident, INCIDENT_SCHEMA)
        or _load(contract_path) != dict(contract)
        or _load(incident_path) != dict(incident)
        or contract.get("target_plan_sha256") != PLAN_SHA256
        or contract.get("confirmation") != CONFIRMATION
        or contract.get("pipeline_incident") != study._file_ref(repo_root, incident_path)
    ):
        raise RecoveryError("recovery contract is absent or invalid")
    source = contract.get("recovery_source") or {}
    source_path = repo_root / str(source.get("path") or "")
    if (
        not source_path.is_file()
        or study._file_sha256(source_path) != source.get("file_sha256")
        or source_path.stat().st_size != source.get("bytes")
    ):
        raise RecoveryError("recovery source differs from its frozen contract")


def _verify_prefix(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    incident_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    allowed = {
        _attestation_path(plan, repo_root).resolve(),
        (coordinator / "ledger.jsonl").resolve(),
        *((root / "ledger.jsonl").resolve() for root in endpoints.values()),
    }
    observed = set(_execution_files(plan, repo_root))
    if _attestation_path(plan, repo_root).resolve() not in observed or not observed <= allowed:
        raise RecoveryError("recovery prefix contains a source, journal, or unexpected file")

    coordinator_ledger = coordinator / "ledger.jsonl"
    entries = executor._load_ledger(coordinator_ledger, role="coordinator")
    state = executor._coordinator_state(plan, entries)
    if (
        set(state["reservations"]) != {BLOCK_ID}
        or state["reservations"][BLOCK_ID]["entry_sha256"] != RESERVATION_ENTRY_SHA256
        or state["incidents"]
    ):
        raise RecoveryError("recovery prefix changed the reservation or added an incident")
    block = executor._block_map(plan)[BLOCK_ID]
    ordered = list(block["work_item_ids"])
    coordinator_ids = [
        str(entry["work_item_id"])
        for entry in entries
        if entry["event_type"] == "family_block_item_terminalized"
    ]
    if coordinator_ids != ordered[: len(coordinator_ids)]:
        raise RecoveryError("coordinator terminals are not a frozen-order prefix")

    incident_ref = study._file_ref(repo_root, incident_path)
    started: set[str] = set()
    terminal_ids: set[str] = set()
    for endpoint_id, endpoint_root in endpoints.items():
        endpoint_entries = executor._load_ledger(endpoint_root / "ledger.jsonl", role="endpoint")
        endpoint_state = executor._endpoint_state(endpoint_entries)
        started.update(endpoint_state["started"])
        terminal_ids.update(endpoint_state["terminals"])
        if endpoint_state["incidents"]:
            raise RecoveryError(f"{endpoint_id} recovery prefix contains an incident")
        for item_id, start in endpoint_state["started"].items():
            if item_id not in ordered:
                raise RecoveryError("endpoint start names an unknown block item")
            if item_id == FIRST_WORK_ITEM_ID:
                if start.get("entry_sha256") != FIRST_START_ENTRY_SHA256:
                    raise RecoveryError("original Gemini start entry differs")
            elif any(
                start.get(key) is not expected
                for key, expected in {
                    "recovery_only": True,
                    "execution_attempted": False,
                    "provider_request_started": False,
                    "mcp_attempted": False,
                    "replay_permitted": False,
                }.items()
            ):
                raise RecoveryError("recovery-only start has invalid semantics")
        for item_id, terminal in endpoint_state["terminals"].items():
            expected = _terminal_payload(
                item_id=item_id, contract=contract, incident_ref=incident_ref
            )
            if any(terminal.get(key) != value for key, value in expected.items()):
                raise RecoveryError("endpoint recovery terminal has invalid semantics")

    coordinator_set = set(coordinator_ids)
    if not coordinator_set <= terminal_ids:
        raise RecoveryError("a coordinator terminal lacks its endpoint terminal")
    next_ids = {ordered[len(coordinator_ids)]} if len(coordinator_ids) < len(ordered) else set()
    if terminal_ids - coordinator_set not in (set(), next_ids):
        raise RecoveryError("endpoint-only terminal is not the next recovery item")
    for item_id in coordinator_ids:
        expected = _terminal_payload(
            item_id=item_id, contract=contract, incident_ref=incident_ref
        )
        if any(state["terminals"][item_id].get(key) != value for key, value in expected.items()):
            raise RecoveryError("coordinator recovery terminal has invalid semantics")
    unresolved = started - terminal_ids
    if unresolved not in (set(), next_ids):
        raise RecoveryError("more than one or an out-of-order start is open")
    if terminal_ids - coordinator_set and unresolved:
        raise RecoveryError("two append stages are simultaneously open")
    if BLOCK_ID in state["completed"] and len(coordinator_ids) != len(ordered):
        raise RecoveryError("block terminal precedes all item terminals")
    return {
        "coordinator": coordinator,
        "endpoints": endpoints,
        "state": state,
        "terminal_count": len(coordinator_ids),
        "completed": BLOCK_ID in state["completed"],
    }


def _recover_item(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    coordinator_ledger: Path,
    endpoint_root: Path,
    raw_endpoint_sha256: str,
    contract: Mapping[str, Any],
    incident_path: Path,
    repo_root: Path,
) -> None:
    item_id = str(item["work_item_id"])
    endpoint_ledger = endpoint_root / "ledger.jsonl"
    endpoint_state = executor._endpoint_state(
        executor._load_ledger(endpoint_ledger, role="endpoint")
    )
    if item_id not in endpoint_state["started"]:
        prior._append_recovery_start(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_ledger=endpoint_ledger,
            raw_endpoint_execution_contract_sha256=raw_endpoint_sha256,
        )
        _verify_prefix(
            plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root
        )
    evidence = executor._journal_evidence(endpoint_root / "source", item_id)
    if evidence != {"journal_count": 0, "request_started_count": 0, "journals": []}:
        raise RecoveryError("a scheduled item has request evidence")
    if executor._source_for_item(endpoint_root, item_id) is not None:
        raise RecoveryError("a scheduled item has a source artifact")
    incident_ref = study._file_ref(repo_root, incident_path)
    payload = _terminal_payload(item_id=item_id, contract=contract, incident_ref=incident_ref)
    endpoint_state = executor._endpoint_state(
        executor._load_ledger(endpoint_ledger, role="endpoint")
    )
    endpoint_terminal = endpoint_state["terminals"].get(item_id)
    if endpoint_terminal is None:
        endpoint_terminal = prior._append_endpoint_terminal(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_ledger=endpoint_ledger,
            payload=payload,
        )
        _verify_prefix(
            plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root
        )
    state = executor._coordinator_state(
        plan, executor._load_ledger(coordinator_ledger, role="coordinator")
    )
    if item_id not in state["terminals"]:
        prior._append_coordinator_terminal(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_terminal=endpoint_terminal,
            coordinator_ledger=coordinator_ledger,
            payload=payload,
        )
    _verify_prefix(plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root)


def recover(
    *,
    plan_path: Path,
    human_protocol_path: Path,
    bound_preflight_path: Path,
    contract_path: Path,
    incident_path: Path,
    repo_root: Path,
    output_dir: Path,
    confirmation: str,
) -> Path:
    if confirmation != CONFIRMATION:
        raise RecoveryError("exact zero-call recovery confirmation is required")
    plan, _, _ = _bindings(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    )
    contract = _load(contract_path)
    incident = _load(incident_path)
    _verify_contract(
        contract=contract,
        contract_path=contract_path,
        incident=incident,
        incident_path=incident_path,
        repo_root=repo_root,
    )
    attestation = _verify_attestation(plan, repo_root)
    _verify_defect(repo_root)
    coordinator, endpoints = executor._roots(plan, repo_root)
    coordinator_ledger = coordinator / "ledger.jsonl"
    global_ledger = repo_root / "flavourbench/artifacts/frontier-contract/ledger.jsonl"
    block = executor._block_map(plan)[BLOCK_ID]
    items = executor._item_map(plan)
    attestation_by_endpoint = {
        str(record["endpoint_id"]): record for record in attestation["records"]
    }
    incident_ref = study._file_ref(repo_root, incident_path)
    with _exclusive_runner_lock(global_ledger):
        with executor._ledger_lock(coordinator_ledger):
            prefix = _verify_prefix(
                plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root
            )
            reservation = prefix["state"]["reservations"][BLOCK_ID]
            for item_id in block["work_item_ids"]:
                state = executor._coordinator_state(
                    plan, executor._load_ledger(coordinator_ledger, role="coordinator")
                )
                if item_id in state["terminals"]:
                    continue
                item = items[item_id]
                endpoint_id = item["route_coordinate"]["endpoint_id"]
                _recover_item(
                    plan=plan,
                    block=block,
                    item=item,
                    reservation=reservation,
                    coordinator_ledger=coordinator_ledger,
                    endpoint_root=endpoints[endpoint_id],
                    raw_endpoint_sha256=attestation_by_endpoint[endpoint_id][
                        "raw_execution_contract_sha256"
                    ],
                    contract=contract,
                    incident_path=incident_path,
                    repo_root=repo_root,
                )
            prefix = _verify_prefix(
                plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root
            )
            terminal = (
                prefix["state"]["completed"].get(BLOCK_ID)
                or executor._terminalize_block(
                    plan=plan, block=block, coordinator_ledger=coordinator_ledger
                )
            )
            final = _verify_prefix(
                plan=plan, contract=contract, incident_path=incident_path, repo_root=repo_root
            )
            if (
                BLOCK_ID not in final["state"]["completed"]
                or terminal.get("actual_cost_usd") != "0"
                or terminal.get("pre_generation_failure_pairs") != 28
                or terminal.get("source_pairs") != 0
                or study._file_sha256(global_ledger) != GLOBAL_LEDGER_SHA256
            ):
                raise RecoveryError("zero-call family block did not close exactly")
            coordinator_entries = executor._load_ledger(
                coordinator_ledger, role="coordinator"
            )
            endpoint_entries = {
                key: executor._load_ledger(root / "ledger.jsonl", role="endpoint")
                for key, root in sorted(endpoints.items())
            }
            counts = {key: len(value) for key, value in endpoint_entries.items()}
            if len(coordinator_entries) != 30 or counts != {
                "deepseek": 16,
                "gemini": 24,
                "sonnet": 16,
            }:
                raise RecoveryError("final ledger row counts differ")
            receipt = _with_hash(
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "record_role": "append_only_zero_call_import_incident_recovery_receipt",
                    "recovery_contract_sha256": contract["artifact_sha256"],
                    "pipeline_incident": dict(incident_ref),
                    "study_plan_sha256": PLAN_SHA256,
                    "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
                    "bound_preflight_sha256": BOUND_PREFLIGHT_SHA256,
                    "admission_block_id": BLOCK_ID,
                    "block_terminal_entry_sha256": terminal["entry_sha256"],
                    "scheduled_pairs": 28,
                    "source_pairs": 0,
                    "pre_generation_failure_pairs": 28,
                    "provider_completion_requests": 0,
                    "epicure_calls": 0,
                    "actual_cost_usd": "0",
                    "reservation_released": True,
                    "replay_permitted": False,
                    "rank_eligible": False,
                    "global_paid_call_ledger_sha256": GLOBAL_LEDGER_SHA256,
                    "coordinator_ledger": {
                        "entries": len(coordinator_entries),
                        "head_entry_sha256": coordinator_entries[-1]["entry_sha256"],
                        "file_sha256": study._file_sha256(coordinator_ledger),
                    },
                    "endpoint_ledgers": {
                        key: {
                            "entries": len(value),
                            "head_entry_sha256": value[-1]["entry_sha256"],
                            "file_sha256": study._file_sha256(endpoints[key] / "ledger.jsonl"),
                        }
                        for key, value in endpoint_entries.items()
                    },
                }
            )
            receipt_path = study._write_artifact(
                output_dir, "reasoning-effort-import-recovery-receipt", receipt
            )
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--bound-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    apply = sub.add_parser("recover")
    apply.add_argument("--contract", type=Path, required=True)
    apply.add_argument("--incident", type=Path, required=True)
    apply.add_argument("--confirm", required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if args.command == "freeze":
        paths = freeze(
            plan_path=args.plan,
            human_protocol_path=args.human_protocol,
            bound_preflight_path=args.bound_preflight,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
        return
    receipt = recover(
        plan_path=args.plan,
        human_protocol_path=args.human_protocol,
        bound_preflight_path=args.bound_preflight,
        contract_path=args.contract,
        incident_path=args.incident,
        repo_root=repo_root,
        output_dir=output_dir,
        confirmation=args.confirm,
    )
    print(json.dumps({"receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    run()
