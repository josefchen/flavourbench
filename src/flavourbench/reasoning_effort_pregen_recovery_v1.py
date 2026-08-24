"""Close the zero-call reasoning block stopped by a frozen policy-parser defect.

This recovery is deliberately specific to plan ``03731c…`` and its first
family block.  It makes no network calls.  It verifies the exact pre-incident
ledgers and catalog-only attestation, terminalizes every scheduled pair as a
zero-cost pre-generation failure, releases the atomic reservation, and emits a
content-addressed receipt.  No work-item identifier becomes replayable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import reasoning_effort_full_study_executor_v1 as executor
from . import reasoning_effort_full_study_v1 as study
from .frontier_contract_runner import _exclusive_runner_lock
from .response_envelope_route_v4 import _policy_from_manifest

PLAN_SHA256 = "03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c"
HUMAN_PROTOCOL_SHA256 = "42fb1b5ea606034d4eb62eb813c957b87ffee44392e1c8f11322bf61fe7002ea"
BOUND_PREFLIGHT_SHA256 = "9c5cb664b5708fccfa49e20f8c362736786e705b06444ed7c59f5013181e8d8e"
BLOCK_ID = "988547d90ea09f47ab36dd9c4e24c3d75d659b1c21932528776e2ced20db3ca6"
RESERVATION_ENTRY_SHA256 = "eea726e623ff03d611ce1b39e9df8a2ffa570940dc49f5269820be729b255656"
FIRST_WORK_ITEM_ID = "2944bfd6a93c4d4055a5c7251033caf72087953923d5217a1fc1e04a9acbf221"
FIRST_START_ENTRY_SHA256 = "55b169fc276f8989fb5f7e4457e044177ba891aff9e8276f9a65369ddc8f80d7"
ATTESTATION_SEMANTIC_SHA256 = "f70ebdeb6b88d43855f33176ee06eaa7a6bb6cbd88a708e2c71b5527f68b73a8"
ATTESTATION_FILE_SHA256 = "f515c575ea20213d11f8eda07795b963b2403734bf7dc68700187e28bd38280c"
INITIAL_COORDINATOR_LEDGER_SHA256 = (
    "40fbc5137a6fab2b436d6a55b8d9918796acfbbb905681ab4a1026b541f05c79"
)
INITIAL_SONNET_LEDGER_SHA256 = "127c3700762059d419e82f00296616abe1243007d69d6b1c095f497dbfaa00f8"
CONTRACT_SCHEMA = "flavourbench-reasoning-effort-pregen-recovery-contract-v1"
INCIDENT_SCHEMA = "flavourbench-reasoning-effort-pipeline-incident-v1"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-pregen-recovery-receipt-v1"
CONFIRMATION = "CLOSE_03731_PREGEN_POLICY_MISMATCH_WITH_ZERO_CALLS"


class RecoveryError(RuntimeError):
    """The live state cannot be closed under this narrow recovery contract."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"expected JSON object: {path}")
    return value


def _with_artifact_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("artifact_sha256", None)
    document["artifact_sha256"] = study._sha256(document)
    return document


def _physical_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RecoveryError(f"recovery source is not a regular file: {resolved}")
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
        raise RecoveryError("recovery bindings differ from the exact incident chain")
    return plan, human, bound


def _expected_files(plan: Mapping[str, Any], repo_root: Path) -> dict[str, str]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    attestation = (
        coordinator
        / "endpoint-attestations"
        / (f"reasoning-effort-family-block-01-attestations-{ATTESTATION_SEMANTIC_SHA256}.json")
    )
    return {
        str((coordinator / "ledger.jsonl").resolve()): INITIAL_COORDINATOR_LEDGER_SHA256,
        str((endpoints["sonnet"] / "ledger.jsonl").resolve()): INITIAL_SONNET_LEDGER_SHA256,
        str(attestation.resolve()): ATTESTATION_FILE_SHA256,
    }


def _execution_files(plan: Mapping[str, Any], repo_root: Path) -> list[Path]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    roots = [coordinator, *endpoints.values()]
    return sorted(
        (
            path.resolve()
            for root in roots
            if root.exists()
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".lock")
        ),
        key=str,
    )


def _verify_original_state(plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    expected = _expected_files(plan, repo_root)
    observed = _execution_files(plan, repo_root)
    if [str(path) for path in observed] != sorted(expected):
        raise RecoveryError("execution-root file inventory differs from the frozen incident state")
    for path in observed:
        if study._file_sha256(path) != expected[str(path)]:
            raise RecoveryError(f"incident-state file differs: {path}")

    coordinator, endpoints = executor._roots(plan, repo_root)
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
        raise RecoveryError("coordinator incident state differs")
    for endpoint_id, endpoint_root in endpoints.items():
        entries = executor._load_ledger(endpoint_root / "ledger.jsonl", role="endpoint")
        endpoint_state = executor._endpoint_state(entries)
        expected_started = {FIRST_WORK_ITEM_ID} if endpoint_id == "sonnet" else set()
        if (
            set(endpoint_state["started"]) != expected_started
            or endpoint_state["terminals"]
            or endpoint_state["incidents"]
        ):
            raise RecoveryError(f"{endpoint_id} incident state differs")
    return {"coordinator": coordinator, "endpoints": endpoints}


def _attestation(plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    coordinator, _ = executor._roots(plan, repo_root)
    path = (
        coordinator
        / "endpoint-attestations"
        / (f"reasoning-effort-family-block-01-attestations-{ATTESTATION_SEMANTIC_SHA256}.json")
    )
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


def _policy_defect(plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    item = executor._item_map(plan)[FIRST_WORK_ITEM_ID]
    manifest = executor._manifest(plan, item, repo_root)
    frozen_document = (manifest.get("run_design") or {}).get("execution_policy") or {}
    frozen_scheduling = frozen_document.get("pair_arm_scheduling")
    parsed = _policy_from_manifest(manifest)
    if frozen_scheduling != "concurrent" or parsed.pair_arm_scheduling != "sequential":
        raise RecoveryError("the exact frozen concurrent/sequential parser mismatch was not found")
    if parsed.sha256 == item["route_coordinate"]["execution_policy_sha256"]:
        raise RecoveryError("parser mismatch no longer produces the recorded policy rejection")
    return {
        "work_item_id": FIRST_WORK_ITEM_ID,
        "frozen_pair_arm_scheduling": frozen_scheduling,
        "parsed_pair_arm_scheduling": parsed.pair_arm_scheduling,
        "frozen_execution_policy_sha256": item["route_coordinate"]["execution_policy_sha256"],
        "parsed_execution_policy_sha256": parsed.sha256,
    }


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
    attestation = _attestation(plan, repo_root)
    defect = _policy_defect(plan, repo_root)
    source = _physical_ref(repo_root, Path(__file__))
    incident = {
        "schema_version": INCIDENT_SCHEMA,
        "record_role": "benchmark_pipeline_pre_generation_incident",
        "incident_code": "pre_generation_policy_parser_mismatch",
        "failure_owner": "benchmark_pipeline",
        "study_plan_sha256": PLAN_SHA256,
        "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
        "bound_preflight_sha256": BOUND_PREFLIGHT_SHA256,
        "admission_block_id": BLOCK_ID,
        "block_reservation_entry_sha256": RESERVATION_ENTRY_SHA256,
        "recovery_source": source,
        "target_plan_source_closure_sha256": plan["source_code"]["closure_sha256"],
        "target_plan_environment_sha256": plan["source_code"]["execution_environment"][
            "environment_sha256"
        ],
        "evidence": {
            "coordinator_ledger_sha256": INITIAL_COORDINATOR_LEDGER_SHA256,
            "sonnet_ledger_sha256": INITIAL_SONNET_LEDGER_SHA256,
            "first_started_work_item_id": FIRST_WORK_ITEM_ID,
            "provider_completion_requests": 0,
            "epicure_calls": 0,
            "generation_ids": [],
            "source_artifacts": 0,
            "run_journals": 0,
            "catalog_attestation": study._file_ref(
                repo_root,
                roots["coordinator"]
                / "endpoint-attestations"
                / (
                    "reasoning-effort-family-block-01-attestations-"
                    f"{ATTESTATION_SEMANTIC_SHA256}.json"
                ),
            ),
            "catalog_http_gets": attestation["counts"]["catalog_http_gets"],
        },
        "defect": defect,
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
    return _with_artifact_hash(incident)


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
        raise RecoveryError("pipeline incident artifact is invalid")
    expected_incident = build_incident(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        repo_root=repo_root,
    )
    if dict(incident) != expected_incident:
        raise RecoveryError("pipeline incident does not reproduce from the frozen state")
    block = executor._block_map(plan)[BLOCK_ID]
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "record_role": "zero_network_append_only_atomic_block_recovery",
        "target_plan_sha256": PLAN_SHA256,
        "human_protocol_sha256": HUMAN_PROTOCOL_SHA256,
        "bound_preflight_sha256": BOUND_PREFLIGHT_SHA256,
        "target_block_id": BLOCK_ID,
        "target_reservation_entry_sha256": RESERVATION_ENTRY_SHA256,
        "pipeline_incident": study._file_ref(repo_root, incident_path),
        "recovery_source": _physical_ref(repo_root, Path(__file__)),
        "source_closure": {
            "target_plan_source_closure_sha256": plan["source_code"]["closure_sha256"],
            "target_plan_environment_sha256": plan["source_code"]["execution_environment"][
                "environment_sha256"
            ],
            "recovery_source_file_sha256": study._file_sha256(Path(__file__)),
        },
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
    return _with_artifact_hash(contract)


def _write(directory: Path, stem: str, document: Mapping[str, Any]) -> Path:
    return study._write_artifact(directory, stem, document)


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
    incident_path = _write(output_dir, "reasoning-effort-pregen-pipeline-incident", incident)
    contract = build_contract(
        plan_path=plan_path,
        human_protocol_path=human_protocol_path,
        bound_preflight_path=bound_preflight_path,
        incident=incident,
        incident_path=incident_path,
        repo_root=repo_root,
    )
    contract_path = _write(output_dir, "reasoning-effort-pregen-recovery-contract", contract)
    return {"incident": incident_path, "contract": contract_path}


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


def _terminal_payload(
    *, item_id: str, contract: Mapping[str, Any], incident_ref: Mapping[str, Any]
) -> dict[str, Any]:
    first = item_id == FIRST_WORK_ITEM_ID
    error_text = (
        "runtime parser forced sequential scheduling for a concurrent frozen policy"
        if first
        else "atomic block cancelled after pre-generation frozen-policy parser mismatch"
    )
    return {
        "disposition": "pre_generation_failure_zero_cost",
        "actual_cost_usd": "0",
        "journal_evidence": {
            "journal_count": 0,
            "request_started_count": 0,
            "journals": [],
        },
        "error_type": (
            "FrozenPolicyParserMismatch" if first else "AtomicBlockCancelledBeforeProviderRequest"
        ),
        "error_sha256": hashlib.sha256(error_text.encode()).hexdigest(),
        "recovery_contract_sha256": contract["artifact_sha256"],
        "pipeline_incident": dict(incident_ref),
        "failure_owner": "benchmark_pipeline",
        "failure_code": "pre_generation_policy_parser_mismatch",
        "provider_completion_requests": 0,
        "provider_request_started_count": 0,
        "epicure_calls": 0,
        "generation_ids": [],
        "source_artifact_count": 0,
        "model_reliability_eligible": False,
        "preference_eligible": False,
    }


def _append_recovery_start(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    raw_endpoint_execution_contract_sha256: str,
) -> dict[str, Any]:
    return executor._append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": "item_execution_started",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": executor._item_wave_id(plan, str(item["work_item_id"])),
            "work_item_id": item["work_item_id"],
            "run_id": item["run_id"],
            "endpoint_id": item["route_coordinate"]["endpoint_id"],
            "variant_id": item["route_coordinate"]["variant_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            "raw_endpoint_execution_contract_sha256": (raw_endpoint_execution_contract_sha256),
            "recovery_only": True,
            "execution_attempted": False,
            "provider_request_started": False,
            "mcp_attempted": False,
            "replay_permitted": False,
        },
    )


def _append_endpoint_terminal(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_ledger: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return executor._append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": "pre_generation_failure_terminalized",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": executor._item_wave_id(plan, str(item["work_item_id"])),
            "work_item_id": item["work_item_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            **dict(payload),
            "replay_permitted": False,
            "rank_eligible": False,
        },
    )


def _append_coordinator_terminal(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_terminal: Mapping[str, Any],
    coordinator_ledger: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return executor._append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_item_terminalized",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": executor._item_wave_id(plan, str(item["work_item_id"])),
            "work_item_id": item["work_item_id"],
            "endpoint_id": item["route_coordinate"]["endpoint_id"],
            "variant_id": item["route_coordinate"]["variant_id"],
            "block_reservation_entry_sha256": reservation["entry_sha256"],
            "endpoint_terminal_entry_sha256": endpoint_terminal["entry_sha256"],
            **dict(payload),
            "replay_permitted": False,
            "rank_eligible": False,
        },
    )


def _verify_recovery_prefix(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    incident_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    coordinator, endpoints = executor._roots(plan, repo_root)
    attestation_path = (
        coordinator
        / "endpoint-attestations"
        / (f"reasoning-effort-family-block-01-attestations-{ATTESTATION_SEMANTIC_SHA256}.json")
    ).resolve()
    allowed_files = {
        attestation_path,
        (coordinator / "ledger.jsonl").resolve(),
        *((root / "ledger.jsonl").resolve() for root in endpoints.values()),
    }
    observed = set(_execution_files(plan, repo_root))
    if attestation_path not in observed or not observed <= allowed_files:
        raise RecoveryError("recovery prefix contains an unexpected source or journal file")

    coordinator_ledger = coordinator / "ledger.jsonl"
    coordinator_entries = executor._load_ledger(coordinator_ledger, role="coordinator")
    state = executor._coordinator_state(plan, coordinator_entries)
    if (
        set(state["reservations"]) != {BLOCK_ID}
        or state["reservations"][BLOCK_ID]["entry_sha256"] != RESERVATION_ENTRY_SHA256
        or state["incidents"]
    ):
        raise RecoveryError("recovery prefix changed the reservation or added an incident event")
    block = executor._block_map(plan)[BLOCK_ID]
    ordered = list(block["work_item_ids"])
    coordinator_terminal_ids = [
        str(entry["work_item_id"])
        for entry in coordinator_entries
        if entry["event_type"] == "family_block_item_terminalized"
    ]
    if coordinator_terminal_ids != ordered[: len(coordinator_terminal_ids)]:
        raise RecoveryError("coordinator recovery terminals are not an exact frozen-order prefix")

    incident_ref = study._file_ref(repo_root, incident_path)
    endpoint_started: set[str] = set()
    endpoint_terminal_ids: set[str] = set()
    for endpoint_id, endpoint_root in endpoints.items():
        entries = executor._load_ledger(endpoint_root / "ledger.jsonl", role="endpoint")
        endpoint_state = executor._endpoint_state(entries)
        endpoint_started.update(endpoint_state["started"])
        endpoint_terminal_ids.update(endpoint_state["terminals"])
        if endpoint_state["incidents"]:
            raise RecoveryError(f"{endpoint_id} recovery prefix contains an incident")
        for item_id, start in endpoint_state["started"].items():
            if item_id not in ordered:
                raise RecoveryError("endpoint recovery start names an unknown block item")
            if item_id == FIRST_WORK_ITEM_ID:
                if (
                    start.get("entry_sha256") != FIRST_START_ENTRY_SHA256
                    or start.get("sequence") != 1
                    or start.get("previous_entry_sha256") is not None
                ):
                    raise RecoveryError("the original Sonnet start entry differs")
            elif (
                start.get("recovery_only") is not True
                or start.get("execution_attempted") is not False
                or start.get("provider_request_started") is not False
                or start.get("mcp_attempted") is not False
                or start.get("replay_permitted") is not False
            ):
                raise RecoveryError("a recovery-only endpoint start has invalid semantics")
        for item_id, terminal in endpoint_state["terminals"].items():
            expected_payload = _terminal_payload(
                item_id=item_id, contract=contract, incident_ref=incident_ref
            )
            if any(terminal.get(key) != value for key, value in expected_payload.items()) or (
                terminal.get("replay_permitted") is not False
                or terminal.get("rank_eligible") is not False
            ):
                raise RecoveryError("an endpoint recovery terminal has invalid semantics")
    coordinator_terminal_set = set(coordinator_terminal_ids)
    if not coordinator_terminal_set <= endpoint_terminal_ids:
        raise RecoveryError("a coordinator terminal lacks its endpoint terminal")
    endpoint_only = endpoint_terminal_ids - coordinator_terminal_set
    next_index = len(coordinator_terminal_ids)
    allowed_next = {ordered[next_index]} if next_index < len(ordered) else set()
    if endpoint_only not in (set(), allowed_next):
        raise RecoveryError("endpoint-only terminal is not the exact next recovery prefix")
    for item_id in coordinator_terminal_ids:
        terminal = state["terminals"][item_id]
        expected_payload = _terminal_payload(
            item_id=item_id, contract=contract, incident_ref=incident_ref
        )
        if any(terminal.get(key) != value for key, value in expected_payload.items()) or (
            terminal.get("replay_permitted") is not False
            or terminal.get("rank_eligible") is not False
        ):
            raise RecoveryError("a coordinator recovery terminal has invalid semantics")
    unresolved_starts = endpoint_started - endpoint_terminal_ids
    if unresolved_starts not in (
        set(),
        allowed_next,
    ):
        raise RecoveryError("recovery prefix has more than one or an out-of-order open start")
    if endpoint_only and unresolved_starts:
        raise RecoveryError("recovery prefix cannot contain two simultaneous open append steps")
    completed = BLOCK_ID in state["completed"]
    if completed and len(coordinator_terminal_ids) != len(ordered):
        raise RecoveryError("block terminal exists before all recovery terminals")
    return {
        "coordinator": coordinator,
        "endpoints": endpoints,
        "state": state,
        "terminal_count": len(coordinator_terminal_ids),
        "completed": completed,
    }


def _recover_item(
    *,
    plan: Mapping[str, Any],
    block: Mapping[str, Any],
    item: Mapping[str, Any],
    reservation: Mapping[str, Any],
    coordinator_ledger: Path,
    endpoint_root: Path,
    raw_endpoint_execution_contract_sha256: str,
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
        _append_recovery_start(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_ledger=endpoint_ledger,
            raw_endpoint_execution_contract_sha256=(raw_endpoint_execution_contract_sha256),
        )
        _verify_recovery_prefix(
            plan=plan,
            contract=contract,
            incident_path=incident_path,
            repo_root=repo_root,
        )
    endpoint_state = executor._endpoint_state(
        executor._load_ledger(endpoint_ledger, role="endpoint")
    )
    evidence = executor._journal_evidence(endpoint_root / "source", item_id)
    if (
        evidence
        != {
            "journal_count": 0,
            "request_started_count": 0,
            "journals": [],
        }
        or executor._source_for_item(endpoint_root, item_id) is not None
    ):
        raise RecoveryError("a scheduled item has provider-request or source evidence")
    incident_ref = study._file_ref(repo_root, incident_path)
    payload = _terminal_payload(item_id=item_id, contract=contract, incident_ref=incident_ref)
    if payload["journal_evidence"] != evidence:
        raise RecoveryError("zero-request journal evidence changed")
    endpoint_terminal = endpoint_state["terminals"].get(item_id)
    if endpoint_terminal is None:
        endpoint_terminal = _append_endpoint_terminal(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_ledger=endpoint_ledger,
            payload=payload,
        )
        _verify_recovery_prefix(
            plan=plan,
            contract=contract,
            incident_path=incident_path,
            repo_root=repo_root,
        )
    state = executor._coordinator_state(
        plan, executor._load_ledger(coordinator_ledger, role="coordinator")
    )
    if item_id not in state["terminals"]:
        _append_coordinator_terminal(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            endpoint_terminal=endpoint_terminal,
            coordinator_ledger=coordinator_ledger,
            payload=payload,
        )
    _verify_recovery_prefix(
        plan=plan,
        contract=contract,
        incident_path=incident_path,
        repo_root=repo_root,
    )


def _build_receipt(
    *,
    contract: Mapping[str, Any],
    incident_ref: Mapping[str, Any],
    terminal: Mapping[str, Any],
    coordinator_ledger: Path,
    endpoints: Mapping[str, Path],
) -> dict[str, Any]:
    coordinator_entries = executor._load_ledger(coordinator_ledger, role="coordinator")
    endpoint_entries = {
        endpoint_id: executor._load_ledger(root / "ledger.jsonl", role="endpoint")
        for endpoint_id, root in sorted(endpoints.items())
    }
    expected_endpoint_rows = {"deepseek": 16, "gemini": 24, "sonnet": 16}
    observed_endpoint_rows = {
        endpoint_id: len(entries) for endpoint_id, entries in endpoint_entries.items()
    }
    if len(coordinator_entries) != 30 or observed_endpoint_rows != expected_endpoint_rows:
        raise RecoveryError("final recovery ledger row counts differ")
    return _with_artifact_hash(
        {
            "schema_version": RECEIPT_SCHEMA,
            "record_role": "append_only_zero_call_atomic_block_recovery_receipt",
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
            "pre_recovery_ledgers": {
                "coordinator_file_sha256": INITIAL_COORDINATOR_LEDGER_SHA256,
                "coordinator_head_entry_sha256": RESERVATION_ENTRY_SHA256,
                "sonnet_file_sha256": INITIAL_SONNET_LEDGER_SHA256,
                "sonnet_head_entry_sha256": FIRST_START_ENTRY_SHA256,
                "deepseek_entries": 0,
                "gemini_entries": 0,
            },
            "coordinator_ledger": {
                "entries": len(coordinator_entries),
                "head_entry_sha256": coordinator_entries[-1]["entry_sha256"],
                "file_sha256": study._file_sha256(coordinator_ledger),
            },
            "endpoint_ledgers": {
                endpoint_id: {
                    "entries": len(entries),
                    "head_entry_sha256": entries[-1]["entry_sha256"],
                    "file_sha256": study._file_sha256(endpoints[endpoint_id] / "ledger.jsonl"),
                }
                for endpoint_id, entries in endpoint_entries.items()
            },
        }
    )


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
    attestation = _attestation(plan, repo_root)
    _policy_defect(plan, repo_root)
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
            prefix = _verify_recovery_prefix(
                plan=plan,
                contract=contract,
                incident_path=incident_path,
                repo_root=repo_root,
            )
            state = prefix["state"]
            reservation = state["reservations"][BLOCK_ID]
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
                    raw_endpoint_execution_contract_sha256=(
                        attestation_by_endpoint[endpoint_id]["raw_execution_contract_sha256"]
                    ),
                    contract=contract,
                    incident_path=incident_path,
                    repo_root=repo_root,
                )
            prefix = _verify_recovery_prefix(
                plan=plan,
                contract=contract,
                incident_path=incident_path,
                repo_root=repo_root,
            )
            if prefix["completed"]:
                terminal = prefix["state"]["completed"][BLOCK_ID]
            else:
                terminal = executor._terminalize_block(
                    plan=plan, block=block, coordinator_ledger=coordinator_ledger
                )
            final_prefix = _verify_recovery_prefix(
                plan=plan,
                contract=contract,
                incident_path=incident_path,
                repo_root=repo_root,
            )
            final_state = final_prefix["state"]
            if (
                BLOCK_ID not in final_state["completed"]
                or terminal.get("actual_cost_usd") != "0"
                or terminal.get("pre_generation_failure_pairs") != 28
                or terminal.get("source_pairs") != 0
            ):
                raise RecoveryError("zero-call family block did not terminalize exactly")
            receipt = _build_receipt(
                contract=contract,
                incident_ref=incident_ref,
                terminal=terminal,
                coordinator_ledger=coordinator_ledger,
                endpoints=endpoints,
            )
            receipt_path = _write(output_dir, "reasoning-effort-pregen-recovery-receipt", receipt)
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
    if args.command == "freeze":
        paths = freeze(
            plan_path=args.plan.resolve(),
            human_protocol_path=args.human_protocol.resolve(),
            bound_preflight_path=args.bound_preflight.resolve(),
            repo_root=repo_root,
            output_dir=args.output_dir.resolve(),
        )
        output: object = {key: str(value) for key, value in paths.items()}
    else:
        path = recover(
            plan_path=args.plan.resolve(),
            human_protocol_path=args.human_protocol.resolve(),
            bound_preflight_path=args.bound_preflight.resolve(),
            contract_path=args.contract.resolve(),
            incident_path=args.incident.resolve(),
            repo_root=repo_root,
            output_dir=args.output_dir.resolve(),
            confirmation=args.confirm,
        )
        output = str(path)
    print(json.dumps({"output": output}, indent=2))


if __name__ == "__main__":
    run()
