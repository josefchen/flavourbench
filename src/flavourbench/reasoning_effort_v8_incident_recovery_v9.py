"""Recover the V8 post-generation audit incident without replaying paid work.

The V9 recovery has two deliberately separate phases.  ``freeze`` and
``verify`` are offline and read-only with respect to every V8 execution file,
the canonical paid-call ledger, and the live-smoke source directory.  ``apply``
requires a separately content-addressed governance GO.  It may append exactly
one canonical ``artifact_recorded`` event for the already complete V8 source,
27 V2 no-delivery dispositions for the never-started reservations, and exactly
one terminal to a new V9 recovery ledger.  It never invokes a provider,
Epicure, catalog discovery, or a reservation API.

The 27 unused V8 identifiers are retired and may not be replayed.  A fresh V9
continuation is frozen under disjoint identifiers, but remains unreserved and
unauthorized until a later, separately governed continuation protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import frontier_contract_runner as frontier
from . import reasoning_effort_full_study_executor_v8 as v8_executor
from . import reasoning_effort_full_study_v1 as v1_study
from . import reasoning_effort_full_study_v8 as v8_study
from . import reasoning_effort_route_gate_v5 as route_v5
from . import reasoning_effort_source_closure_v9 as source_closure_v9
from .execution_policy import verify_policy_document
from .run_journal import recovery_state, scan_recovery_journals

INCIDENT_SCHEMA = "flavourbench-reasoning-effort-v8-live-incident-v9"
SOURCE_CLOSURE_ENVELOPE_SCHEMA = (
    "flavourbench-reasoning-effort-v8-incident-source-closure-envelope-v9"
)
RECOVERY_PLAN_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-plan-v9"
DRY_RUN_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-dry-run-v9"
OPERATOR_PROTOCOL_SCHEMA = (
    "flavourbench-reasoning-effort-v8-incident-recovery-operator-protocol-v9"
)
BUNDLE_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-bundle-v9"
HANDOFF_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-review-handoff-v9"
GOVERNANCE_GO_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-go-v9"
PAIR_AUDIT_SCHEMA = "flavourbench-reasoning-effort-v8-pair-audit-v9"
LOCAL_LEDGER_SCHEMA = "flavourbench-reasoning-effort-v8-recovery-ledger-v9"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-v8-incident-recovery-receipt-v9"

V8_STUDY_ID = v8_study.STUDY_ID
V8_PLAN_SHA256 = "8a167b860d28b7bda0d4b4e80a99167eaa47b893938ea85cb09fdd9a0e3c7cde"
V8_PLAN_FILE_SHA256 = "5396e1ae8507b2833631586de33230163ddb84303e71009b5f9b45bdcbd47c88"
V8_BLOCK_ID = "5fc0ce8adbe930c23d87a34eed712afb52a10a0d8813014127d85eb70ce5bb89"
V8_FIRST_WORK_ITEM_ID = (
    "feea54486923b0ec8f6efc718eab63ec1f510c50a5385fa3b85c853a349dd64e"
)
V8_FIRST_RUN_ID = "98a70a43-45a2-5eef-8fa9-c86b10226302"
V8_FIRST_RESERVATION_SHA256 = (
    "dd2bc97f5123ae7967ed683f43c3c4f2f58c0181c507cc8da865b8001f4c68b6"
)
V8_LOCAL_RESERVATION_SHA256 = (
    "c9641e5400370b17ddb75603818592124b6b228fd7d9e14cc1d90b08f689c6f0"
)
V8_ENDPOINT_START_SHA256 = (
    "f1e5b6df7497aa5feb20214c09749b28b325b98cd7d59686844d358ac11f5afb"
)
V8_ENDPOINT_INCIDENT_SHA256 = (
    "e0aca756014b1ebea831fa2c72eeb6c01703a6326f1def741bf73aee79b21a44"
)
V8_COORDINATOR_INCIDENT_SHA256 = (
    "00ea6a02cd48ff915f840d470b2781c54308e5216e5e9e084576a82bc41ef7f0"
)
V8_ERROR_SHA256 = "e9aeee31145d5309e96bc7e606c0f20956e6ec7934e1148a579f8a1f3167760b"
V8_ERROR_TEXT = (
    "module 'flavourbench.reasoning_effort_route_gate_v5' has no attribute "
    "'_verify_live_artifact'"
)
V8_GLOBAL_PREFIX_SEQUENCE = 57
V8_GLOBAL_PREFIX_HEAD_SHA256 = (
    "abdcaf9bdd10c83f96ad9584c188fc4e98e6dff13e5797916145916112279c22"
)
V8_GLOBAL_PREFIX_FILE_SHA256 = (
    "1766889aef6acb9413c511430b942b920e2e12ab1e439c8f281e3f374dcbcd68"
)

V8_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v8-monotone-capacity-attestation"
)
V9_ROOT = (
    "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-v8-live-incident-recovery-v9"
)
V8_PLAN_PATH = (
    f"{V8_ROOT}/plan/reasoning-effort-plan-v8-{V8_PLAN_SHA256}.json"
)
V8_COORDINATOR_LEDGER = f"{V8_ROOT}/runs/coordinator/ledger.jsonl"
V8_SONNET_LEDGER = f"{V8_ROOT}/runs/sonnet/ledger.jsonl"
V8_ATTESTATION_PATH = (
    f"{V8_ROOT}/runs/coordinator/endpoint-attestations/"
    "reasoning-effort-v8-block-01-attestations-"
    "0e0fb886e08083e0b30c9662754358a4fad79ec1985f2a0b5a1996213d6d7b72.json"
)
GLOBAL_LEDGER_PATH = "flavourbench/artifacts/frontier-contract/ledger.jsonl"
SOURCE_ROOT = "flavourbench/artifacts/live-smoke"
SOURCE_FILENAME = "20260808T183143Z-adf7471c974f.json"
SOURCE_ARTIFACT_SHA256 = (
    "adf7471c974fa97d3a27dca9f8763ae4ff4397c95b79838b699991cab4a88dd0"
)
SOURCE_FILE_SHA256 = "033dd72372a747a3f41140519b1659964e5934490838b8b76d6a0f763fdddd24"
JOURNAL_FILENAME = (
    "flavourbench-live-smoke-journal-"
    "567e5ddf0e64bfaf94c8f8417866d7d898770e558ecdadf0d348ce52278d9715.jsonl"
)
JOURNAL_SHA256 = "567e5ddf0e64bfaf94c8f8417866d7d898770e558ecdadf0d348ce52278d9715"
JOURNAL_HEAD_SHA256 = "0a8a4828ba0529375c60fb6ebed0214c536594b230751560a2e4aef825a3caa6"
ACTUAL_COST_MICROS = 100_547
ACTUAL_COST_USD = "0.100547"
GENERATION_IDS = (
    "gen-1786213904-IuTrJkUdK6qhnPMb1xxN",
    "gen-1786213904-NePTvLZiVOca8ntOvaOM",
    "gen-1786213916-w9ZLmWYgY2s4utWNn3Y0",
    "gen-1786213917-fQ3bxFBIOqiP4CC8Q9Uj",
    "gen-1786213921-vKUIx15rc0cjqXJl30Kd",
    "gen-1786213922-OaJvtu0LshYDu65WkBs6",
    "gen-1786213926-IFyfrLFMVFmMu0DDwkT3",
)
LEGACY_SEQUENTIAL_POLICY_SHA256 = (
    "98ff7b353bc4a3ba74c066225530f9828e1c6af600dd922ab5c4559f037d25b6"
)
FROZEN_CONCURRENT_POLICY_SHA256 = (
    "7992f74248bb67429e2f4a7da543dc99312a7116d7670ef99a86c58aeaa6902a"
)
V9_FREEZE_NONCE = "reasoning-effort-v8-live-incident-recovery-v9-2026-08-08"
V9_NAMESPACE = uuid.UUID("41b789d6-a5a5-4fd5-80cb-7ff89246982a")
V9_STUDY_ID = "frontier-reasoning-effort-v9-post-incident-fresh-continuation"
UNUSED_RESERVATION_USD = "16.43574656000000000000000000102"
POST_RECOVERY_EXPOSURE_USD = "48.11999382666666666666666666"
BASELINE_EXPOSURE_USD = "48.01944682666666666666666666"
CONFIRMATION = "APPLY_V9_ONE_SOURCE_IMPORT_ZERO_CALLS_NO_NEW_RESERVATIONS"

_verify_live_artifact = frontier._verify_live_artifact


class RecoveryError(RuntimeError):
    """The frozen V8 incident or requested V9 transition differs."""


class SimulatedCrash(RuntimeError):
    """A test-only crash at a named durable boundary."""


@dataclass(frozen=True)
class ForensicState:
    plan: dict[str, Any]
    block: dict[str, Any]
    items: dict[str, dict[str, Any]]
    reservations: dict[str, dict[str, Any]]
    source: dict[str, Any]
    source_path: Path
    source_digest: str
    pair_audit: dict[str, Any]
    continuation_items: tuple[dict[str, Any], ...]
    v8_tree_snapshot: tuple[dict[str, Any], ...]
    global_entries: tuple[dict[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: object) -> str:
    return v8_study._sha256(value)


def _file_sha256(path: Path) -> str:
    return v8_study._file_sha256(path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"expected a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(f"invalid JSON document: {path}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"expected a JSON object: {path}")
    return value


def _artifact_ok(document: Mapping[str, Any], schema: str) -> bool:
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return document.get("schema_version") == schema and document.get(
        "artifact_sha256"
    ) == _sha256(body)


def _with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    return {**body, "artifact_sha256": _sha256(body)}


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise RecoveryError(f"path is outside its root: {path}") from error


def _file_ref(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"file reference is not regular: {path}")
    return {
        "path": _relative(root, path),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def _artifact_ref(root: Path, path: Path) -> dict[str, Any]:
    document = _load_json(path)
    return {
        **_file_ref(root, path),
        "semantic_sha256": str(document.get("artifact_sha256") or ""),
    }


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    return v8_study._write_artifact(directory, prefix, payload)


def _tree_snapshot(root: Path, path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_dir():
        raise RecoveryError(f"snapshot root is not a regular directory: {path}")
    records: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            raise RecoveryError(f"snapshot contains a symlink: {candidate}")
        if candidate.is_file():
            records.append(_file_ref(root, candidate))
    return tuple(records)


def _verify_tree_snapshot(
    *, state_root: Path, relative_root: str, expected: Sequence[Mapping[str, Any]]
) -> None:
    observed = _tree_snapshot(state_root, state_root / relative_root)
    if list(observed) != [dict(record) for record in expected]:
        raise RecoveryError("V8 execution/artifact tree differs from the frozen snapshot")


def _verify_global_prefix(ledger_path: Path, *, exact_length: bool) -> list[dict[str, Any]]:
    entries = frontier.load_ledger(ledger_path)
    if len(entries) < V8_GLOBAL_PREFIX_SEQUENCE or (
        exact_length and len(entries) != V8_GLOBAL_PREFIX_SEQUENCE
    ):
        raise RecoveryError("canonical global ledger length differs from the V8 boundary")
    if entries[V8_GLOBAL_PREFIX_SEQUENCE - 1].get(
        "entry_sha256"
    ) != V8_GLOBAL_PREFIX_HEAD_SHA256:
        raise RecoveryError("canonical global ledger V8 prefix head differs")
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    prefix = b"".join(lines[:V8_GLOBAL_PREFIX_SEQUENCE])
    if hashlib.sha256(prefix).hexdigest() != V8_GLOBAL_PREFIX_FILE_SHA256:
        raise RecoveryError("canonical global ledger V8 byte prefix differs")
    return entries


def _manifest_document(state_root: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    reference = item.get("manifest") or {}
    path = state_root / str(reference.get("path") or "")
    document = _load_json(path)
    if (
        reference.get("semantic_sha256") != item["route_coordinate"]["manifest_sha256"]
        or document.get("content_address", {}).get("digest")
        != reference.get("semantic_sha256")
        or _file_sha256(path) != reference.get("file_sha256")
        or path.stat().st_size != reference.get("bytes")
        or not v8_study.verify_manifest_content_address(document)
    ):
        raise RecoveryError("first-item manifest does not verify")
    return document


def _concurrent_policy_proof(
    *, state_root: Path, item: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _manifest_document(state_root, item)
    design = manifest.get("run_design") or {}
    manifest_policy = design.get("execution_policy")
    source_policy = source.get("execution_policy")
    frozen_sha = str(item["route_coordinate"]["execution_policy_sha256"])
    if (
        frozen_sha != FROZEN_CONCURRENT_POLICY_SHA256
        or source.get("execution_policy_sha256") != frozen_sha
        or design.get("execution_policy_sha256") != frozen_sha
        or source_policy != manifest_policy
        or not verify_policy_document(source_policy)
        or not verify_policy_document(manifest_policy)
        or source_policy.get("content_address", {}).get("digest") != frozen_sha
        or manifest_policy.get("content_address", {}).get("digest") != frozen_sha
        or source_policy.get("pair_arm_scheduling") != "concurrent"
        or manifest_policy.get("pair_arm_scheduling") != "concurrent"
    ):
        raise RecoveryError("V1/V8 concurrent execution policy does not verify exactly")
    return {
        "decision": "exact_v1_v8_concurrent_policy_verified",
        "frozen_execution_policy_sha256": frozen_sha,
        "source_execution_policy_sha256": source["execution_policy_sha256"],
        "manifest_execution_policy_sha256": design["execution_policy_sha256"],
        "pair_arm_scheduling": "concurrent",
        "legacy_v4_derived_pair_arm_scheduling": "sequential",
        "legacy_v4_derived_execution_policy_sha256": LEGACY_SEQUENTIAL_POLICY_SHA256,
        "legacy_failure_removed_only_after_exact_policy_equality": (
            "variant_execution_policy_mismatch"
        ),
    }


def pair_audit_v9(
    *, plan: Mapping[str, Any], item: Mapping[str, Any], source_path: Path, state_root: Path
) -> dict[str, Any]:
    """Audit the source using the canonical verifier and the frozen V1/V8 policy."""

    source, digest = _verify_live_artifact(source_path)
    raw = route_v5.raw_endpoint_contract(source.get("endpoint_contract") or {})
    adapted = copy.deepcopy(dict(item))
    adapted["route_coordinate"]["endpoint_execution_contract_sha256"] = _sha256(raw)
    pair = route_v5._adapted_pair_audit(
        plan=v1_study._item_plan_view(plan, item),
        item=adapted,
        source_path=source_path,
        source=source,
        digest=digest,
        repo_root=state_root,
    )
    failures = [
        value
        for value in pair.get("failures") or []
        if value != "source_endpoint_semantic_contract_differs_from_v5_freeze"
    ]
    variant = str(item["route_coordinate"]["variant_id"])
    if variant == "explicit_low":
        failures = [
            value
            for value in failures
            if value != "explicit_low_reasoning_request_semantics_failed"
        ]
        starts = [
            event
            for event in source.get("provider_attempt_events") or []
            if isinstance(event, Mapping) and event.get("event_type") == "request_started"
        ]
        requests = [(event.get("metadata") or {}).get("request_contract") for event in starts]
        if not requests or any(
            not isinstance(request, Mapping)
            or request.get("reasoning_field_present") is not True
            or request.get("reasoning") != {"effort": "low", "exclude": True}
            for request in requests
        ):
            failures.append("explicit_low_reasoning_request_semantics_failed")

    observed_semantic = _sha256(route_v5.semantic_endpoint_contract(raw))
    expected_semantic = item["route_coordinate"]["semantic_execution_contract_sha256"]
    if observed_semantic != expected_semantic:
        failures.append("source_endpoint_semantic_contract_differs_from_task_wave_freeze")
    on = (source.get("results") or {}).get("epicure_on") or {}
    fanout = [
        int(record["tool_call_count"])
        for record in on.get("intermediate_outputs") or []
        if isinstance(record, Mapping) and record.get("tool_call_count") is not None
    ]
    if max(fanout or [0]) > 13:
        failures.append("round_fanout_exceeds_13")
    if len(on.get("tool_trace") or []) > 13:
        failures.append("total_tool_calls_exceed_13")

    policy_proof = _concurrent_policy_proof(
        state_root=state_root, item=item, source=source
    )
    if "variant_execution_policy_mismatch" in failures:
        failures.remove("variant_execution_policy_mismatch")
    frozen_contract = plan["models"][item["route_coordinate"]["endpoint_id"]][
        "semantic_execution_contract"
    ]
    capacity = v8_study.validate_monotone_capacity_contract(
        frozen=frozen_contract,
        observed=route_v5.semantic_endpoint_contract(raw),
    )
    pair.update(
        {
            "schema_version": PAIR_AUDIT_SCHEMA,
            "request_semantics": "reasoning_effort_explicit_low",
            "legacy_request_semantics_label": "reasoning_effort_explicit_high",
            "legacy_request_semantics_label_corrected": True,
            "execution_policy_v9": policy_proof,
            "capacity_attestation_v9": {
                **capacity,
                "frozen_contract_sha256": _sha256(frozen_contract),
                "observed_contract_sha256": observed_semantic,
            },
            "task_wave": {
                "observed_semantic_execution_contract_sha256": observed_semantic,
                "frozen_semantic_execution_contract_sha256": expected_semantic,
                "maximum_round_fanout": max(fanout or [0]),
                "executed_tool_calls_total": len(on.get("tool_trace") or []),
            },
        }
    )
    pair["failures"] = sorted(set(failures))
    pair["decision"] = "passed_all_predicates" if not pair["failures"] else "failed"
    if (
        pair["decision"] != "passed_all_predicates"
        or (pair.get("accounting") or {}).get("reconciled") is not True
        or (pair.get("accounting") or {}).get("actual_cost_micros")
        != ACTUAL_COST_MICROS
    ):
        raise RecoveryError(f"V9 pair audit did not pass exactly: {pair['failures']}")
    return pair


def _matching_sources(
    *, source_root: Path, item_ids: set[str], run_ids: set[str]
) -> list[tuple[Path, dict[str, Any], str]]:
    matches: list[tuple[Path, dict[str, Any], str]] = []
    if source_root.is_symlink() or not source_root.is_dir():
        raise RecoveryError("canonical source root is not a regular directory")
    for path in sorted(source_root.glob("*.json")):
        document = _load_json(path)
        item_match = str(document.get("dataset_work_item_id") or "") in item_ids
        run_match = str(document.get("run_id") or "") in run_ids
        if item_match != run_match:
            raise RecoveryError("source matches only one immutable V8 identity")
        if item_match:
            verified, digest = _verify_live_artifact(path)
            matches.append((path, verified, digest))
    return matches


def _verify_generation_evidence(
    *, source: Mapping[str, Any], journal_path: Path
) -> dict[str, Any]:
    state = recovery_state(journal_path)
    response_ids = sorted(
        str(event.get("generation_id") or "")
        for event in source.get("provider_attempt_events") or []
        if event.get("event_type") == "response_received" and event.get("generation_id")
    )
    reconciled = [
        event
        for event in source.get("provider_attempt_events") or []
        if event.get("event_type") == "accounting_reconciled"
    ]
    reconciled_ids = sorted(str(event.get("generation_id") or "") for event in reconciled)
    reconciled_cost = sum(
        int((event.get("metadata") or {}).get("cost_micros") or 0)
        for event in reconciled
    )
    expected = sorted(GENERATION_IDS)
    if (
        state.run_id != V8_FIRST_RUN_ID
        or state.journal_sha256 != JOURNAL_SHA256
        or state.head_entry_sha256 != JOURNAL_HEAD_SHA256
        or state.entry_count != 57
        or state.finalized is not True
        or list(state.generation_ids) != expected
        or state.unreconciled_generation_ids
        or state.uncertain_attempt_ids
        or response_ids != expected
        or reconciled_ids != expected
        or reconciled_cost != ACTUAL_COST_MICROS
        or (source.get("budget") or {}).get("actual_cost_micros") != ACTUAL_COST_MICROS
        or (source.get("budget") or {}).get("all_generation_costs_reconciled") is not True
    ):
        raise RecoveryError("source/journal generation accounting differs")
    return {
        "journal_sha256": state.journal_sha256,
        "journal_head_entry_sha256": state.head_entry_sha256,
        "journal_entries": state.entry_count,
        "journal_finalized": state.finalized,
        "generation_ids": expected,
        "generation_count": len(expected),
        "all_generation_costs_reconciled": True,
        "uncertain_attempt_ids": [],
        "unreconciled_generation_ids": [],
        "actual_cost_micros": ACTUAL_COST_MICROS,
        "actual_cost_usd": ACTUAL_COST_USD,
    }


def _verify_static_defect() -> dict[str, Any]:
    error_hash = hashlib.sha256(V8_ERROR_TEXT.encode()).hexdigest()
    v1_source = inspect.getsource(v1_study.pair_audit)
    if (
        error_hash != V8_ERROR_SHA256
        or "v5._verify_live_artifact(source_path)" not in v1_source
        or hasattr(route_v5, "_verify_live_artifact")
        or _verify_live_artifact is not frontier._verify_live_artifact
    ):
        raise RecoveryError("the exact V8 missing-verifier defect does not reproduce")
    return {
        "exception_type": "AttributeError",
        "error_sha256": error_hash,
        "missing_symbol": "reasoning_effort_route_gate_v5._verify_live_artifact",
        "failing_callsite": "reasoning_effort_full_study_v1.pair_audit",
        "canonical_successor_symbol": (
            "frontier_contract_runner._verify_live_artifact"
        ),
        "canonical_successor_alias_identity_verified": True,
    }


def verify_forensic_state(
    *,
    state_root: Path,
    exact_global_length: bool,
    expected_v8_snapshot: Sequence[Mapping[str, Any]] | None = None,
) -> ForensicState:
    state_root = state_root.resolve()
    plan_path = state_root / V8_PLAN_PATH
    plan = _load_json(plan_path)
    if (
        plan.get("artifact_sha256") != V8_PLAN_SHA256
        or _file_sha256(plan_path) != V8_PLAN_FILE_SHA256
        or _sha256({key: value for key, value in plan.items() if key != "artifact_sha256"})
        != V8_PLAN_SHA256
    ):
        raise RecoveryError("V8 plan differs")
    blocks = v8_executor._block_map(plan)
    items = v8_executor._item_map(plan)
    if V8_BLOCK_ID not in blocks or len(blocks[V8_BLOCK_ID]["work_item_ids"]) != 28:
        raise RecoveryError("V8 first block differs")
    block = blocks[V8_BLOCK_ID]
    if block["work_item_ids"][0] != V8_FIRST_WORK_ITEM_ID:
        raise RecoveryError("V8 first work item differs")

    snapshot = _tree_snapshot(state_root, state_root / V8_ROOT)
    if expected_v8_snapshot is not None:
        _verify_tree_snapshot(
            state_root=state_root,
            relative_root=V8_ROOT,
            expected=expected_v8_snapshot,
        )
    coordinator_entries = v8_executor._load_ledger(
        state_root / V8_COORDINATOR_LEDGER, role="coordinator"
    )
    sonnet_entries = v8_executor._load_ledger(
        state_root / V8_SONNET_LEDGER, role="endpoint"
    )
    if (
        [entry["entry_sha256"] for entry in coordinator_entries]
        != [V8_LOCAL_RESERVATION_SHA256, V8_COORDINATOR_INCIDENT_SHA256]
        or [entry["entry_sha256"] for entry in sonnet_entries]
        != [V8_ENDPOINT_START_SHA256, V8_ENDPOINT_INCIDENT_SHA256]
        or coordinator_entries[-1].get("error_sha256") != V8_ERROR_SHA256
        or sonnet_entries[-1].get("error_sha256") != V8_ERROR_SHA256
        or coordinator_entries[-1].get("canonical_reservation_retained") is not True
        or sonnet_entries[-1].get("canonical_reservation_retained") is not True
        or coordinator_entries[-1].get("canonical_artifact_record_entry_sha256") is not None
        or sonnet_entries[-1].get("canonical_artifact_record_entry_sha256") is not None
    ):
        raise RecoveryError("V8 local incident chain differs")
    for endpoint in ("deepseek", "gemini"):
        endpoint_ledger = state_root / V8_ROOT / "runs" / endpoint / "ledger.jsonl"
        if endpoint_ledger.exists():
            raise RecoveryError(f"V8 {endpoint} ledger unexpectedly exists")

    global_entries = _verify_global_prefix(
        state_root / GLOBAL_LEDGER_PATH, exact_length=exact_global_length
    )
    reservations = v8_executor._campaign_global_reservations(
        plan=plan, entries=global_entries
    )
    if list(reservations) != list(block["work_item_ids"]):
        raise RecoveryError("V8 canonical reservation order differs")
    if reservations[V8_FIRST_WORK_ITEM_ID]["entry_sha256"] != V8_FIRST_RESERVATION_SHA256:
        raise RecoveryError("V8 first canonical reservation differs")
    finalizations = [
        entry
        for entry in global_entries
        if entry.get("event_type")
        in {"artifact_recorded", "no_artifact_reconciliation_recorded"}
        and entry.get("reservation_entry_sha256")
        in {record["entry_sha256"] for record in reservations.values()}
    ]
    if exact_global_length and finalizations:
        raise RecoveryError("V8 reservations were already finalized before V9 freeze")
    if not exact_global_length:
        by_reservation: dict[str, list[dict[str, Any]]] = {}
        for entry in finalizations:
            by_reservation.setdefault(
                str(entry.get("reservation_entry_sha256") or ""), []
            ).append(entry)
        if any(len(values) > 1 for values in by_reservation.values()):
            raise RecoveryError("V8 canonical reservation disposition is ambiguous")
        first_finalizations = by_reservation.get(V8_FIRST_RESERVATION_SHA256, [])
        if first_finalizations and first_finalizations[0].get("event_type") != "artifact_recorded":
            raise RecoveryError("V8 completed source reservation has a non-artifact disposition")
        for item_id in block["work_item_ids"][1:]:
            reservation_id = reservations[item_id]["entry_sha256"]
            values = by_reservation.get(reservation_id, [])
            if values and values[0].get("event_type") != "no_artifact_reconciliation_recorded":
                raise RecoveryError("a never-started V8 reservation has a non-release disposition")

    item_ids = set(block["work_item_ids"])
    run_ids = {str(items[item_id]["run_id"]) for item_id in block["work_item_ids"]}
    matches = _matching_sources(
        source_root=state_root / SOURCE_ROOT, item_ids=item_ids, run_ids=run_ids
    )
    if len(matches) != 1:
        raise RecoveryError("V8 source inventory is not exactly one completed pair")
    source_path, source, source_digest = matches[0]
    first_item = items[V8_FIRST_WORK_ITEM_ID]
    if (
        source_path.name != SOURCE_FILENAME
        or source_digest != SOURCE_ARTIFACT_SHA256
        or _file_sha256(source_path) != SOURCE_FILE_SHA256
        or source.get("status") != "complete"
        or source.get("run_id") != V8_FIRST_RUN_ID
        or source.get("dataset_work_item_id") != V8_FIRST_WORK_ITEM_ID
        or source.get("requested_model_id")
        != first_item["route_coordinate"]["model_id"]
        or source.get("requested_provider")
        != first_item["route_coordinate"]["provider_endpoint"]
        or source.get("candidate_manifest_sha256")
        != first_item["manifest"]["semantic_sha256"]
    ):
        raise RecoveryError("V8 first source identity differs")
    journal_path = state_root / SOURCE_ROOT / JOURNAL_FILENAME
    _verify_generation_evidence(source=source, journal_path=journal_path)

    continuation: list[dict[str, Any]] = []
    for ordinal, item_id in enumerate(block["work_item_ids"][1:], start=1):
        item = items[item_id]
        journal_states = scan_recovery_journals(
            state_root / SOURCE_ROOT, dataset_work_item_id=item_id
        )
        endpoint_root = state_root / V8_ROOT / "runs" / str(
            item["route_coordinate"]["endpoint_id"]
        )
        endpoint_events = (
            v8_executor._load_ledger(endpoint_root / "ledger.jsonl", role="endpoint")
            if (endpoint_root / "ledger.jsonl").exists()
            else []
        )
        related_events = [
            event for event in endpoint_events if event.get("work_item_id") == item_id
        ]
        if journal_states or related_events:
            raise RecoveryError("a continuation item has V8 delivery evidence")
        continuation.append(
            {
                "ordinal": ordinal,
                "work_item_id": item_id,
                "run_id": item["run_id"],
                "endpoint_id": item["route_coordinate"]["endpoint_id"],
                "task_id": item["route_coordinate"]["task_id"],
                "task_family": item["route_coordinate"]["task_family"],
                "variant_id": item["route_coordinate"]["variant_id"],
                "canonical_reservation_entry_sha256": reservations[item_id]["entry_sha256"],
                "reserved_usd": reservations[item_id]["reserved_usd"],
                "delivery_evidence": {
                    "item_execution_started_events": 0,
                    "provider_request_journals": 0,
                    "source_artifacts": 0,
                    "canonical_finalizations_at_v8_stop": 0,
                },
                "v8_identifier_disposition": (
                    "release_after_content_addressed_v2_no_delivery_proof"
                ),
                "same_identifier_replay_permitted": False,
                "fresh_reservation_required_for_any_future_delivery": True,
            }
        )
    if len(continuation) != 27:
        raise RecoveryError("unused V8 reservation inventory is not exactly 27 items")

    pair = pair_audit_v9(
        plan=plan,
        item=first_item,
        source_path=source_path,
        state_root=state_root,
    )
    _verify_static_defect()
    return ForensicState(
        plan=plan,
        block=block,
        items=items,
        reservations=reservations,
        source=source,
        source_path=source_path,
        source_digest=source_digest,
        pair_audit=pair,
        continuation_items=tuple(continuation),
        v8_tree_snapshot=snapshot,
        global_entries=tuple(global_entries),
    )


def build_incident(*, state_root: Path, forensic: ForensicState) -> dict[str, Any]:
    generation = _verify_generation_evidence(
        source=forensic.source,
        journal_path=state_root / SOURCE_ROOT / JOURNAL_FILENAME,
    )
    payload = {
        "schema_version": INCIDENT_SCHEMA,
        "record_role": "append_only_forensic_record_of_v8_post_generation_audit_incident",
        "observed_on": "2026-08-08",
        "v8_study_id": V8_STUDY_ID,
        "v8_study_plan_sha256": V8_PLAN_SHA256,
        "v8_admission_block_id": V8_BLOCK_ID,
        "v8_local_block_reservation_entry_sha256": V8_LOCAL_RESERVATION_SHA256,
        "canonical_global_ledger_prefix": {
            "path": GLOBAL_LEDGER_PATH,
            "sequence": V8_GLOBAL_PREFIX_SEQUENCE,
            "head_entry_sha256": V8_GLOBAL_PREFIX_HEAD_SHA256,
            "file_sha256": V8_GLOBAL_PREFIX_FILE_SHA256,
        },
        "exact_failure": _verify_static_defect(),
        "local_incident_chain": {
            "endpoint_start_entry_sha256": V8_ENDPOINT_START_SHA256,
            "endpoint_incident_entry_sha256": V8_ENDPOINT_INCIDENT_SHA256,
            "coordinator_incident_entry_sha256": V8_COORDINATOR_INCIDENT_SHA256,
            "canonical_reservation_status_recorded": "active_reservation",
            "canonical_reservation_retained": True,
        },
        "completed_source": {
            **_file_ref(state_root, forensic.source_path),
            "artifact_sha256": forensic.source_digest,
            "status": forensic.source["status"],
            "work_item_id": V8_FIRST_WORK_ITEM_ID,
            "run_id": V8_FIRST_RUN_ID,
            "canonical_reservation_entry_sha256": V8_FIRST_RESERVATION_SHA256,
            "condition_count": 2,
            "synthetic_arms": 0,
            "actual_cost_micros": ACTUAL_COST_MICROS,
            "actual_cost_usd": ACTUAL_COST_USD,
        },
        "journal_and_generation_evidence": generation,
        "corrected_offline_pair_audit": forensic.pair_audit,
        "legacy_audit_defects": {
            "missing_verifier_symbol": True,
            "obsolete_sequential_policy_predicate": True,
            "legacy_sequential_policy_sha256": LEGACY_SEQUENTIAL_POLICY_SHA256,
            "frozen_v1_v8_concurrent_policy_sha256": FROZEN_CONCURRENT_POLICY_SHA256,
            "request_semantics_label_reported": "reasoning_effort_explicit_high",
            "request_semantics_label_corrected": "reasoning_effort_explicit_low",
            "low_reasoning_request_payloads_verified": True,
        },
        "untouched_reservations": {
            "count": len(forensic.continuation_items),
            "provider_requests": 0,
            "epicure_calls": 0,
            "source_artifacts": 0,
            "canonical_finalizations": 0,
            "items_sha256": _sha256(list(forensic.continuation_items)),
        },
        "v8_tree_snapshot": list(forensic.v8_tree_snapshot),
        "official": False,
        "rank_eligible": False,
    }
    return _with_hash(payload)


def build_source_closure_envelope(*, code_root: Path) -> dict[str, Any]:
    closure = source_closure_v9.build_source_closure(repo_root=code_root)
    return _with_hash(
        {
            "schema_version": SOURCE_CLOSURE_ENVELOPE_SCHEMA,
            "record_role": "v9_recovery_transitive_source_and_environment_closure",
            "source_closure": closure,
            "provider_or_epicure_calls": 0,
            "secrets_recorded": False,
        }
    )


def _no_delivery_proof_record(
    *, incident: Mapping[str, Any], forensic: ForensicState, item_record: Mapping[str, Any]
) -> dict[str, Any]:
    item_id = str(item_record["work_item_id"])
    reservation = forensic.reservations[item_id]
    tree_sha = _sha256(list(forensic.v8_tree_snapshot))
    identity_scan = {
        "work_item_id": item_id,
        "run_id": item_record["run_id"],
        "endpoint_id": item_record["endpoint_id"],
        "source_artifact_matches": 0,
        "journal_matches": 0,
        "endpoint_ledger_event_matches": 0,
        "canonical_prefix_finalization_matches": 0,
    }
    return {
        "schema_version": frontier.NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION,
        "record_role": "never_started_v8_work_item_no_delivery_reconciliation",
        "v8_incident_sha256": incident["artifact_sha256"],
        "reservation": {
            "ledger_entry_sha256": reservation["entry_sha256"],
            "runner_run_id": reservation["runner_run_id"],
            "model_id": reservation["model_id"],
            "provider_tag": reservation["provider_tag"],
            "manifest_sha256": reservation["manifest_sha256"],
            "study_plan_sha256": reservation["study_plan_sha256"],
            "admission_block_id": reservation["admission_block_id"],
            "work_item_id": reservation["work_item_id"],
            "reserved_usd": reservation["reserved_usd"],
        },
        "no_delivery_evidence": {
            "item_execution_started_events": 0,
            "provider_request_journals": 0,
            "provider_request_started_events": 0,
            "provider_response_received_events": 0,
            "source_artifacts": 0,
            "generation_ids": [],
            "mcp_trace_events": 0,
            "canonical_finalizations_before_reconciliation": 0,
            "evidence_snapshot": {
                "v8_tree_unchanged": True,
                "canonical_source_inventory_verified": True,
                "journal_inventory_verified": True,
                "v8_tree_snapshot_sha256": tree_sha,
                "target_identity_scan_sha256": _sha256(identity_scan),
            },
        },
        "conclusion": {
            "delivery_attempted": False,
            "provider_generation_request_reached": False,
            "provider_generation_cost_usd": "0",
            "epicure_called": False,
            "reservation_release_authorized": True,
            "same_identifier_replay_permitted": False,
            "disposition": "release_never_started_no_delivery_reservation",
        },
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "official": False,
        "rank_eligible": False,
    }


def freeze_no_delivery_proofs(
    *,
    state_root: Path,
    output_dir: Path,
    incident: Mapping[str, Any],
    forensic: ForensicState,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for item_record in forensic.continuation_items:
        record = _no_delivery_proof_record(
            incident=incident,
            forensic=forensic,
            item_record=item_record,
        )
        path = frontier.write_no_artifact_reconciliation(record, output_dir)
        loaded = _load_json(path)
        verified = frontier.validate_no_artifact_reconciliation_v2(
            path,
            ledger_entries=forensic.global_entries,
        )
        references.append(
            {
                **_file_ref(state_root, path),
                "content_sha256": verified.artifact_sha256,
                "reservation_entry_sha256": verified.reservation_entry_sha256,
                "work_item_id": verified.work_item_id,
                "record": loaded,
            }
        )
    if len(references) != 27 or len(
        {record["content_sha256"] for record in references}
    ) != 27:
        raise RecoveryError("V9 no-delivery proof set is incomplete or duplicated")
    return references


def _fresh_continuation_items(forensic: ForensicState) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    old_identifiers = {
        str(value)
        for item_id in forensic.block["work_item_ids"]
        for value in (
            item_id,
            forensic.items[item_id]["run_id"],
            *forensic.items[item_id]["arm_ids"],
            *[
                slot["attempt_id"]
                for slot in forensic.items[item_id].get("attempt_slots") or []
            ],
        )
    }
    for old_record in forensic.continuation_items:
        old = forensic.items[str(old_record["work_item_id"])]
        coordinate = {
            **copy.deepcopy(old["route_coordinate"]),
            "schema_version": "flavourbench-reasoning-effort-fresh-continuation-coordinate-v9",
            "freeze_nonce": V9_FREEZE_NONCE,
            "superseded_v8_work_item_id": old["work_item_id"],
            "superseded_v8_run_id": old["run_id"],
            "v8_incident_affected_identifier_reused": False,
        }
        route_cell_id = _sha256(coordinate)
        run_id = str(uuid.uuid5(V9_NAMESPACE, f"{V9_FREEZE_NONCE}:{route_cell_id}:run"))
        work_identity = {
            "schema_version": "flavourbench-reasoning-effort-fresh-work-item-v9",
            "freeze_nonce": V9_FREEZE_NONCE,
            "route_cell_id": route_cell_id,
            "run_id": run_id,
            "old_work_item_id": old["work_item_id"],
        }
        work_item_id = _sha256(work_identity)
        arm_ids = [f"{run_id}:epicure_off", f"{run_id}:epicure_on"]
        attempts: list[dict[str, Any]] = []
        for old_slot in old.get("attempt_slots") or []:
            condition = str(old_slot["arm_id"]).rsplit(":", 1)[-1]
            arm_id = arm_ids[0] if condition == "epicure_off" else arm_ids[1]
            attempt_identity = {
                "work_item_id": work_item_id,
                "arm_id": arm_id,
                "phase": old_slot["phase"],
                "attempt_index": old_slot["attempt_index"],
            }
            attempts.append(
                {
                    "arm_id": arm_id,
                    "phase": old_slot["phase"],
                    "attempt_index": old_slot["attempt_index"],
                    "attempt_id": str(
                        uuid.uuid5(
                            V9_NAMESPACE,
                            f"{V9_FREEZE_NONCE}:{_sha256(attempt_identity)}:attempt",
                        )
                    ),
                }
            )
        record = {
            "ordinal": old_record["ordinal"],
            "route_cell_id": route_cell_id,
            "work_item_id": work_item_id,
            "run_id": run_id,
            "arm_ids": arm_ids,
            "attempt_slots": attempts,
            "route_coordinate": coordinate,
            "task": copy.deepcopy(old["task"]),
            "manifest": copy.deepcopy(old["manifest"]),
            "worst_case_reserve_usd": old["worst_case_reserve_usd"],
            "supersedes_without_identifier_reuse": {
                "v8_work_item_id": old["work_item_id"],
                "v8_run_id": old["run_id"],
                "v8_canonical_reservation_entry_sha256": old_record[
                    "canonical_reservation_entry_sha256"
                ],
                "v8_no_delivery_proof_required": True,
            },
            "reservation_status": "not_created",
            "live_execution_authorized": False,
        }
        new_identifiers = {
            work_item_id,
            run_id,
            *arm_ids,
            *[slot["attempt_id"] for slot in attempts],
        }
        if old_identifiers.intersection(new_identifiers):
            raise RecoveryError("fresh V9 continuation reuses a V8 identifier")
        records.append(record)
    if len(records) != 27:
        raise RecoveryError("fresh V9 continuation does not contain 27 items")
    all_new = [
        value
        for record in records
        for value in (
            record["work_item_id"],
            record["run_id"],
            *record["arm_ids"],
            *[slot["attempt_id"] for slot in record["attempt_slots"]],
        )
    ]
    if len(all_new) != len(set(all_new)):
        raise RecoveryError("fresh V9 continuation identifiers collide")
    return records


def build_recovery_plan(
    *,
    incident: Mapping[str, Any],
    closure_envelope: Mapping[str, Any],
    forensic: ForensicState,
    no_delivery_proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    proof_refs = [
        {
            key: value
            for key, value in proof.items()
            if key != "record"
        }
        for proof in no_delivery_proofs
    ]
    fresh_items = _fresh_continuation_items(forensic)
    fresh_reserve = v8_study._exact_sum(
        [Decimal(item["worst_case_reserve_usd"]) for item in fresh_items]
    )
    retired_reserve = v8_study._exact_sum(
        [
            Decimal(
                forensic.reservations[str(item["work_item_id"])]["reserved_usd"]
            )
            for item in forensic.continuation_items
        ]
    )
    if (
        v8_study._decimal_text(fresh_reserve) != UNUSED_RESERVATION_USD
        or fresh_reserve != retired_reserve
    ):
        raise RecoveryError("fresh V9 continuation reserve differs from retired V8 reserve")
    payload = {
        "schema_version": RECOVERY_PLAN_SCHEMA,
        "record_role": (
            "append_only_v8_incident_recovery_and_fresh_unreserved_continuation_plan"
        ),
        "v8_study_plan_sha256": V8_PLAN_SHA256,
        "v8_incident_sha256": incident["artifact_sha256"],
        "source_closure_envelope_sha256": closure_envelope["artifact_sha256"],
        "source_closure_sha256": closure_envelope["source_closure"]["closure_sha256"],
        "recovery_action": {
            "canonical_global_events_exact": 28,
            "canonical_artifact_recorded_events": 1,
            "canonical_no_delivery_reconciliation_events": 27,
            "completed_work_item_id": V8_FIRST_WORK_ITEM_ID,
            "completed_run_id": V8_FIRST_RUN_ID,
            "completed_reservation_entry_sha256": V8_FIRST_RESERVATION_SHA256,
            "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
            "actual_cost_usd": ACTUAL_COST_USD,
            "new_provider_requests": 0,
            "new_epicure_calls": 0,
            "catalog_requests": 0,
            "new_reservations": 0,
            "v8_local_files_mutated": 0,
            "canonical_source_files_mutated": 0,
            "v9_local_terminal_events": 1,
            "released_never_started_reservation_usd": UNUSED_RESERVATION_USD,
            "post_recovery_total_exposure_usd": POST_RECOVERY_EXPOSURE_USD,
        },
        "continuation": {
            "decision": "retire_v8_identifiers_and_freeze_fresh_v9_continuation",
            "retired_v8_items": list(forensic.continuation_items),
            "no_delivery_proofs": proof_refs,
            "fresh_v9_items": fresh_items,
            "item_count": len(fresh_items),
            "canonical_active_v8_reservation_count_after_recovery": 0,
            "future_new_reservations_required": 27,
            "future_worst_case_reserve_usd": UNUSED_RESERVATION_USD,
            "fresh_study_id": V9_STUDY_ID,
            "fresh_freeze_nonce": V9_FREEZE_NONCE,
            "automatic_continuation_authorized": False,
            "requires_separate_independent_go": True,
            "requires_successor_continuation_executor": True,
            "v8_identifier_replay_permitted": False,
            "v8_identifiers_reused": False,
        },
        "governance_boundary": {
            "this_plan_authorizes_no_live_action_without_a_separate_go": True,
            "apply_go_schema": GOVERNANCE_GO_SCHEMA,
            "continuation_is_out_of_scope_for_apply": True,
            "rank_eligible": False,
            "official": False,
        },
        "confirmation": CONFIRMATION,
    }
    return _with_hash(payload)


def _expected_canonical_event(
    *, recovery_plan: Mapping[str, Any], forensic: ForensicState, pair_audit_sha256: str
) -> dict[str, Any]:
    item = forensic.items[V8_FIRST_WORK_ITEM_ID]
    return {
        "event_type": "artifact_recorded",
        "runner_run_id": V8_FIRST_RUN_ID,
        "reservation_entry_sha256": V8_FIRST_RESERVATION_SHA256,
        "manifest_sha256": item["manifest"]["semantic_sha256"],
        "model_id": item["route_coordinate"]["model_id"],
        "provider_tag": item["route_coordinate"]["provider_endpoint"],
        "artifact_filename": SOURCE_FILENAME,
        "artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "artifact_status": "complete",
        "artifact_exposure_usd": ACTUAL_COST_USD,
        "postflight_issues": [],
        "campaign_id": V8_STUDY_ID,
        "study_plan_sha256": V8_PLAN_SHA256,
        "admission_block_id": V8_BLOCK_ID,
        "work_item_id": V8_FIRST_WORK_ITEM_ID,
        "recovery_plan_sha256": recovery_plan["artifact_sha256"],
        "recovery_pair_audit_sha256": pair_audit_sha256,
        "recovery_disposition": "completed_source_imported_without_replay",
        "new_provider_requests": 0,
        "new_epicure_calls": 0,
        "new_reservations": 0,
    }


def _expected_release_event(
    *,
    recovery_plan: Mapping[str, Any],
    proof_reference: Mapping[str, Any],
    proof: frontier.NoArtifactReconciliationV2,
    forensic: ForensicState,
) -> dict[str, Any]:
    reservation = next(
        record
        for record in forensic.reservations.values()
        if record["entry_sha256"] == proof.reservation_entry_sha256
    )
    return {
        "event_type": "no_artifact_reconciliation_recorded",
        "runner_run_id": reservation["runner_run_id"],
        "reservation_entry_sha256": reservation["entry_sha256"],
        "model_id": reservation["model_id"],
        "provider_tag": reservation["provider_tag"],
        "manifest_sha256": reservation["manifest_sha256"],
        "study_plan_sha256": reservation["study_plan_sha256"],
        "admission_block_id": reservation["admission_block_id"],
        "work_item_id": reservation["work_item_id"],
        "reconciliation_filename": Path(str(proof_reference["path"])).name,
        "reconciliation_sha256": proof.artifact_sha256,
        "reconciliation_schema_version": (
            frontier.NO_ARTIFACT_RECONCILIATION_V2_SCHEMA_VERSION
        ),
        "released_exposure_usd": reservation["reserved_usd"],
        "provider_generation_cost_usd": "0",
        "decision": "release_never_started_no_delivery_reservation_v2",
        "recovery_plan_sha256": recovery_plan["artifact_sha256"],
        "campaign_id": V8_STUDY_ID,
        "new_provider_requests": 0,
        "new_epicure_calls": 0,
        "new_reservations": 0,
    }


def build_dry_run(
    *, incident: Mapping[str, Any], recovery_plan: Mapping[str, Any], forensic: ForensicState
) -> dict[str, Any]:
    pair_sha = _sha256(forensic.pair_audit)
    expected_event = _expected_canonical_event(
        recovery_plan=recovery_plan,
        forensic=forensic,
        pair_audit_sha256=pair_sha,
    )
    payload = {
        "schema_version": DRY_RUN_SCHEMA,
        "record_role": "offline_zero_side_effect_recovery_transition_preview",
        "v8_incident_sha256": incident["artifact_sha256"],
        "recovery_plan_sha256": recovery_plan["artifact_sha256"],
        "precondition": {
            "global_ledger_entries": len(forensic.global_entries),
            "active_v8_reservations": 28,
            "completed_source_pairs": 1,
            "never_started_pairs": 27,
        },
        "proposed_canonical_event": expected_event,
        "proposed_canonical_event_payload_sha256": _sha256(expected_event),
        "proposed_no_delivery_reconciliations": {
            "count": len(recovery_plan["continuation"]["no_delivery_proofs"]),
            "proof_sha256s": [
                reference["content_sha256"]
                for reference in recovery_plan["continuation"]["no_delivery_proofs"]
            ],
            "released_reservation_usd": UNUSED_RESERVATION_USD,
        },
        "projected_postcondition": {
            "canonical_artifact_events_appended": 1,
            "canonical_no_delivery_reconciliation_events_appended": 27,
            "canonical_events_appended": 28,
            "active_v8_reservations": 0,
            "released_never_started_reservation_usd": UNUSED_RESERVATION_USD,
            "post_recovery_total_exposure_usd": POST_RECOVERY_EXPOSURE_USD,
            "v9_local_terminal_events": 1,
            "provider_requests": 0,
            "epicure_calls": 0,
            "catalog_requests": 0,
            "new_reservations": 0,
            "fresh_v9_continuation_items_frozen_but_not_reserved": 27,
            "v8_files_changed": 0,
            "source_files_changed": 0,
        },
        "ledger_writes_performed": 0,
        "provider_or_epicure_calls_performed": 0,
        "apply_authorized": False,
    }
    return _with_hash(payload)


def build_operator_protocol(
    *,
    incident_ref: Mapping[str, Any],
    closure_ref: Mapping[str, Any],
    plan_ref: Mapping[str, Any],
    dry_run_ref: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": OPERATOR_PROTOCOL_SCHEMA,
        "record_role": "exact_independent_review_and_zero_call_apply_protocol",
        "inputs": {
            "incident": dict(incident_ref),
            "source_closure": dict(closure_ref),
            "recovery_plan": dict(plan_ref),
            "dry_run": dict(dry_run_ref),
        },
        "independent_review_checks": [
            "verify all four content addresses and physical hashes",
            "confirm the canonical verifier alias identity and exact AttributeError hash",
            "confirm the source and finalized journal contain the same seven generation IDs",
            "confirm all seven generation costs reconcile to 100547 micros",
            "confirm the V1/V8 concurrent policy equals source and manifest byte-semantically",
            "confirm only the obsolete V4 sequential predicate and wrong label are corrected",
            "confirm 27 unused V8 items have no start, journal, source, or finalization",
            "confirm 27 V2 proofs release exactly 16.43574656000000000000000000102 USD",
            "confirm fresh V9 continuation identities do not intersect any V8 identity",
            "confirm apply appends exactly 28 dispositions and makes no live call",
        ],
        "go_requirements": {
            "schema_version": GOVERNANCE_GO_SCHEMA,
            "decision": "go_for_one_source_import_and_27_no_delivery_releases",
            "maximum_canonical_ledger_events": 28,
            "new_provider_requests": 0,
            "new_epicure_calls": 0,
            "catalog_requests": 0,
            "new_reservations": 0,
            "continuation_authorized": False,
            "reviewer_is_executor": False,
            "reviewer_is_v9_builder": False,
        },
        "operator_steps": [
            "Run the verify command against the frozen bundle; this is read-only.",
            "Have an independent reviewer issue a content-addressed GO matching this protocol.",
            "Stop every retired V8 executor before apply; do not rerun V8.",
            "Run apply once with the exact confirmation and the independent GO.",
            "Rerun apply only for crash recovery; it must return the same 28 dispositions.",
            "Verify all 27 V8 no-delivery releases and the fresh, unreserved V9 identities.",
            "Obtain a separate GO before any continuation delivery.",
        ],
        "forbidden_actions": [
            "provider request",
            "Epicure request",
            "catalog request",
            "new reservation",
            "V8 local-ledger mutation",
            "source or journal mutation",
            "completed work-item replay",
            "automatic continuation",
        ],
    }
    return _with_hash(payload)


def build_review_handoff(
    *, bundle_ref: Mapping[str, Any], recovery_plan: Mapping[str, Any]
) -> dict[str, Any]:
    bundle_path = str(bundle_ref["path"])
    module = "flavourbench.reasoning_effort_v8_incident_recovery_v9"
    verify_command = (
        "PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python -m "
        f"{module} --state-root . --code-root . verify --bundle {bundle_path}"
    )
    apply_command = (
        "PYTHONPATH=flavourbench/src flavourbench/.venv/bin/python -m "
        f"{module} --state-root . --code-root . apply --bundle {bundle_path} "
        "--governance-go <content-addressed-independent-go.json> "
        f"--output-dir {V9_ROOT}/runs/recovery --confirm {CONFIRMATION}"
    )
    go_fixed_fields = {
        "schema_version": GOVERNANCE_GO_SCHEMA,
        "record_role": "independent_v9_zero_call_recovery_go",
        "decision": "go_for_one_source_import_and_27_no_delivery_releases",
        "recovery_plan_sha256": recovery_plan["artifact_sha256"],
        "reviewed_bundle_sha256": bundle_ref["semantic_sha256"],
        "reviewed_incident_sha256": recovery_plan["v8_incident_sha256"],
        "reviewed_source_closure_sha256": recovery_plan["source_closure_sha256"],
        "reviewed_no_delivery_proof_count": 27,
        "reviewed_fresh_identifier_count": 27,
        "maximum_canonical_ledger_events": 28,
        "new_provider_requests": 0,
        "new_epicure_calls": 0,
        "catalog_requests": 0,
        "new_reservations": 0,
        "continuation_authorized": False,
        "reviewer_is_executor": False,
        "reviewer_is_v9_builder": False,
        "independent_technical_review_completed": True,
    }
    return _with_hash(
        {
            "schema_version": HANDOFF_SCHEMA,
            "record_role": "independent_review_handoff_without_go_authority",
            "bundle": dict(bundle_ref),
            "recovery_plan_sha256": recovery_plan["artifact_sha256"],
            "read_only_verify_command": verify_command,
            "apply_command_after_valid_go_only": apply_command,
            "go_fixed_fields": go_fixed_fields,
            "go_reviewer_fields_required": {
                "reviewer_identity_commitment_sha256": (
                    "64 lowercase hexadecimal characters; no raw identity"
                ),
                "reviewed_at": "UTC ISO-8601 timestamp",
                "reviewer_role": "independent technical reviewer",
                "artifact_sha256": (
                    "sha256 of canonical JSON excluding artifact_sha256"
                ),
            },
            "go_filename_rule": "v9-recovery-independent-go-<artifact_sha256>.json",
            "apply_performed": False,
            "provider_or_epicure_calls": 0,
            "canonical_ledger_writes": 0,
        }
    )


def freeze(
    *, state_root: Path, code_root: Path, output_dir: Path
) -> dict[str, Path]:
    before_global = (state_root / GLOBAL_LEDGER_PATH).read_bytes()
    before_v8 = _tree_snapshot(state_root, state_root / V8_ROOT)
    before_source = (state_root / SOURCE_ROOT / SOURCE_FILENAME).read_bytes()
    before_journal = (state_root / SOURCE_ROOT / JOURNAL_FILENAME).read_bytes()
    forensic = verify_forensic_state(
        state_root=state_root,
        exact_global_length=True,
        expected_v8_snapshot=None,
    )
    incident = build_incident(state_root=state_root, forensic=forensic)
    incident_path = _write_artifact(output_dir, "v8-live-audit-incident", incident)
    closure = build_source_closure_envelope(code_root=code_root)
    closure_path = _write_artifact(output_dir, "v9-recovery-source-closure", closure)
    no_delivery_proofs = freeze_no_delivery_proofs(
        state_root=state_root,
        output_dir=output_dir / "reconciliations",
        incident=incident,
        forensic=forensic,
    )
    plan = build_recovery_plan(
        incident=incident,
        closure_envelope=closure,
        forensic=forensic,
        no_delivery_proofs=no_delivery_proofs,
    )
    plan_path = _write_artifact(output_dir, "v9-recovery-plan", plan)
    dry_run = build_dry_run(
        incident=incident, recovery_plan=plan, forensic=forensic
    )
    dry_run_path = _write_artifact(output_dir, "v9-recovery-dry-run", dry_run)
    protocol = build_operator_protocol(
        incident_ref=_artifact_ref(state_root, incident_path),
        closure_ref=_artifact_ref(state_root, closure_path),
        plan_ref=_artifact_ref(state_root, plan_path),
        dry_run_ref=_artifact_ref(state_root, dry_run_path),
    )
    protocol_path = _write_artifact(output_dir, "v9-recovery-operator-protocol", protocol)
    bundle = _with_hash(
        {
            "schema_version": BUNDLE_SCHEMA,
            "record_role": "offline_v9_incident_recovery_review_bundle",
            "incident": _artifact_ref(state_root, incident_path),
            "source_closure": _artifact_ref(state_root, closure_path),
            "recovery_plan": _artifact_ref(state_root, plan_path),
            "dry_run": _artifact_ref(state_root, dry_run_path),
            "operator_protocol": _artifact_ref(state_root, protocol_path),
            "frozen_with_provider_or_epicure_calls": 0,
            "frozen_with_ledger_writes": 0,
            "apply_go_present": False,
        }
    )
    bundle_path = _write_artifact(output_dir, "v9-recovery-review-bundle", bundle)
    handoff = build_review_handoff(
        bundle_ref=_artifact_ref(state_root, bundle_path),
        recovery_plan=plan,
    )
    handoff_path = _write_artifact(
        output_dir, "v9-independent-review-handoff", handoff
    )
    if (
        (state_root / GLOBAL_LEDGER_PATH).read_bytes() != before_global
        or _tree_snapshot(state_root, state_root / V8_ROOT) != before_v8
        or (state_root / SOURCE_ROOT / SOURCE_FILENAME).read_bytes() != before_source
        or (state_root / SOURCE_ROOT / JOURNAL_FILENAME).read_bytes() != before_journal
    ):
        raise RecoveryError("offline freeze changed a protected V8/global/source input")
    return {
        "incident": incident_path,
        "source_closure": closure_path,
        "recovery_plan": plan_path,
        "dry_run": dry_run_path,
        "operator_protocol": protocol_path,
        "bundle": bundle_path,
        "review_handoff": handoff_path,
    }


def _load_frozen_artifact(path: Path, schema: str) -> dict[str, Any]:
    document = _load_json(path)
    if not _artifact_ok(document, schema):
        raise RecoveryError(f"frozen artifact does not verify: {path}")
    if path.name.rsplit("-", 1)[-1] != f"{document['artifact_sha256']}.json":
        raise RecoveryError(f"frozen artifact filename is not content-addressed: {path}")
    return document


def verify_bundle(
    *, bundle_path: Path, state_root: Path, code_root: Path
) -> dict[str, Any]:
    bundle = _load_frozen_artifact(bundle_path, BUNDLE_SCHEMA)
    schemas = {
        "incident": INCIDENT_SCHEMA,
        "source_closure": SOURCE_CLOSURE_ENVELOPE_SCHEMA,
        "recovery_plan": RECOVERY_PLAN_SCHEMA,
        "dry_run": DRY_RUN_SCHEMA,
        "operator_protocol": OPERATOR_PROTOCOL_SCHEMA,
    }
    loaded: dict[str, dict[str, Any]] = {}
    for key, schema in schemas.items():
        reference = bundle.get(key) or {}
        path = state_root / str(reference.get("path") or "")
        document = _load_frozen_artifact(path, schema)
        if (
            _artifact_ref(state_root, path) != reference
            or document.get("artifact_sha256") != reference.get("semantic_sha256")
        ):
            raise RecoveryError(f"bundle reference differs: {key}")
        loaded[key] = document
    closure = loaded["source_closure"].get("source_closure")
    if not isinstance(closure, Mapping):
        raise RecoveryError("source-closure envelope is malformed")
    source_closure_v9.verify_source_closure(expected=closure, repo_root=code_root)
    plan = loaded["recovery_plan"]
    incident = loaded["incident"]
    if (
        plan.get("v8_incident_sha256") != incident.get("artifact_sha256")
        or plan.get("source_closure_envelope_sha256")
        != loaded["source_closure"].get("artifact_sha256")
        or loaded["dry_run"].get("recovery_plan_sha256")
        != plan.get("artifact_sha256")
    ):
        raise RecoveryError("V9 bundle cross-binding differs")
    proof_refs = plan.get("continuation", {}).get("no_delivery_proofs") or []
    if len(proof_refs) != 27:
        raise RecoveryError("V9 bundle does not bind exactly 27 no-delivery proofs")
    verified_proofs = []
    for reference in proof_refs:
        path = state_root / str(reference.get("path") or "")
        if _file_ref(state_root, path) != {
            key: reference[key] for key in ("path", "bytes", "file_sha256")
        }:
            raise RecoveryError("V9 no-delivery proof physical reference differs")
        verified = frontier.validate_no_artifact_reconciliation_v2(
            path,
            ledger_entries=frontier.load_ledger(state_root / GLOBAL_LEDGER_PATH),
        )
        if (
            verified.artifact_sha256 != reference.get("content_sha256")
            or verified.reservation_entry_sha256
            != reference.get("reservation_entry_sha256")
            or verified.work_item_id != reference.get("work_item_id")
        ):
            raise RecoveryError("V9 no-delivery proof semantic reference differs")
        verified_proofs.append(verified)
    released = v8_study._exact_sum(
        [
            Decimal(
                next(
                    entry["reserved_usd"]
                    for entry in frontier.load_ledger(state_root / GLOBAL_LEDGER_PATH)
                    if entry.get("entry_sha256") == proof.reservation_entry_sha256
                )
            )
            for proof in verified_proofs
        ]
    )
    if v8_study._decimal_text(released) != UNUSED_RESERVATION_USD:
        raise RecoveryError("V9 no-delivery proof reserve sum differs")
    forensic = verify_forensic_state(
        state_root=state_root,
        exact_global_length=True,
        expected_v8_snapshot=incident["v8_tree_snapshot"],
    )
    rederived_incident = build_incident(state_root=state_root, forensic=forensic)
    rederived_plan = build_recovery_plan(
        incident=incident,
        closure_envelope=loaded["source_closure"],
        forensic=forensic,
        no_delivery_proofs=[
            {
                **reference,
                "record": _load_json(state_root / str(reference["path"])),
            }
            for reference in plan["continuation"]["no_delivery_proofs"]
        ],
    )
    rederived_dry = build_dry_run(
        incident=incident, recovery_plan=plan, forensic=forensic
    )
    if (
        rederived_incident != incident
        or rederived_plan != plan
        or rederived_dry != loaded["dry_run"]
    ):
        raise RecoveryError("V9 offline bundle does not deterministically rederive")
    return {
        "decision": "offline_v9_bundle_verified_no_apply_authority",
        "bundle_sha256": bundle["artifact_sha256"],
        "recovery_plan_sha256": plan["artifact_sha256"],
        "completed_source_pairs": 1,
        "continuation_pairs": len(forensic.continuation_items),
        "provider_or_epicure_calls": 0,
        "ledger_writes": 0,
    }


def _verify_go(
    *,
    governance_go: Mapping[str, Any],
    recovery_plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    expected = {
        "record_role": "independent_v9_zero_call_recovery_go",
        "decision": "go_for_one_source_import_and_27_no_delivery_releases",
        "recovery_plan_sha256": recovery_plan["artifact_sha256"],
        "reviewed_bundle_sha256": bundle["artifact_sha256"],
        "reviewed_incident_sha256": recovery_plan["v8_incident_sha256"],
        "reviewed_source_closure_sha256": recovery_plan["source_closure_sha256"],
        "reviewed_no_delivery_proof_count": 27,
        "reviewed_fresh_identifier_count": 27,
        "maximum_canonical_ledger_events": 28,
        "new_provider_requests": 0,
        "new_epicure_calls": 0,
        "catalog_requests": 0,
        "new_reservations": 0,
        "continuation_authorized": False,
        "reviewer_is_executor": False,
        "reviewer_is_v9_builder": False,
        "independent_technical_review_completed": True,
    }
    commitment = str(governance_go.get("reviewer_identity_commitment_sha256") or "")
    reviewed_at = str(governance_go.get("reviewed_at") or "")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        reviewed_at_is_utc = (
            reviewed_at.endswith("Z")
            and parsed_reviewed_at.utcoffset() is not None
            and parsed_reviewed_at.utcoffset().total_seconds() == 0
        )
    except ValueError:
        reviewed_at_is_utc = False
    if (
        not _artifact_ok(governance_go, GOVERNANCE_GO_SCHEMA)
        or any(governance_go.get(key) != value for key, value in expected.items())
        or len(commitment) != 64
        or any(character not in "0123456789abcdef" for character in commitment)
        or governance_go.get("reviewer_role") != "independent technical reviewer"
        or not reviewed_at_is_utc
    ):
        raise RecoveryError("exact independent V9 source-import GO is absent or invalid")


def _local_ledger_digest(entry: Mapping[str, Any]) -> str:
    body = dict(entry)
    body.pop("entry_sha256", None)
    return _sha256(body)


def _load_local_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RecoveryError("V9 recovery ledger is not a regular file")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise RecoveryError("V9 recovery ledger contains a blank line")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryError("V9 recovery ledger contains invalid JSON") from error
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != LOCAL_LEDGER_SCHEMA
            or entry.get("sequence") != number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("event_type") != "incident_recovery_terminalized"
            or entry.get("entry_sha256") != _local_ledger_digest(entry)
        ):
            raise RecoveryError("V9 recovery ledger chain differs")
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    if len(entries) > 1:
        raise RecoveryError("V9 recovery ledger contains more than one terminal")
    return entries


def _append_local_terminal(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    entries = _load_local_ledger(path)
    if entries:
        protected = {
            "schema_version",
            "sequence",
            "recorded_at",
            "previous_entry_sha256",
            "entry_sha256",
            "event_type",
        }
        if any(
            entries[0].get(key) != value
            for key, value in payload.items()
            if key not in protected
        ):
            raise RecoveryError("existing V9 local terminal differs")
        return entries[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": LOCAL_LEDGER_SCHEMA,
        "sequence": 1,
        "recorded_at": _utc_now(),
        "previous_entry_sha256": None,
        "event_type": "incident_recovery_terminalized",
        **dict(payload),
    }
    entry["entry_sha256"] = _local_ledger_digest(entry)
    rendered = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        data = (rendered + "\n").encode("utf-8")
        if os.write(descriptor, data) != len(data):
            raise OSError("short V9 local-ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


def _inject(injector: Callable[[str], None] | None, point: str) -> None:
    if injector is not None:
        injector(point)


def apply_recovery(
    *,
    state_root: Path,
    code_root: Path,
    bundle_path: Path,
    governance_go_path: Path,
    output_dir: Path,
    confirmation: str,
    failure_injector: Callable[[str], None] | None = None,
) -> Path:
    if confirmation != CONFIRMATION:
        raise RecoveryError("exact V9 zero-call source-import confirmation is required")
    bundle = _load_frozen_artifact(bundle_path, BUNDLE_SCHEMA)
    references = {
        key: state_root / str((bundle.get(key) or {}).get("path") or "")
        for key in ("incident", "source_closure", "recovery_plan")
    }
    incident = _load_frozen_artifact(references["incident"], INCIDENT_SCHEMA)
    closure_envelope = _load_frozen_artifact(
        references["source_closure"], SOURCE_CLOSURE_ENVELOPE_SCHEMA
    )
    recovery_plan = _load_frozen_artifact(
        references["recovery_plan"], RECOVERY_PLAN_SCHEMA
    )
    governance_go = _load_json(governance_go_path)
    if governance_go_path.name != (
        "v9-recovery-independent-go-"
        f"{governance_go.get('artifact_sha256')}.json"
    ):
        raise RecoveryError("independent V9 GO filename is not content-addressed")
    _verify_go(
        governance_go=governance_go,
        recovery_plan=recovery_plan,
        bundle=bundle,
    )
    closure = closure_envelope.get("source_closure")
    if not isinstance(closure, Mapping):
        raise RecoveryError("V9 source closure is malformed")
    source_closure_v9.verify_source_closure(expected=closure, repo_root=code_root)
    forensic = verify_forensic_state(
        state_root=state_root,
        exact_global_length=False,
        expected_v8_snapshot=incident["v8_tree_snapshot"],
    )
    pair_audit_document = _with_hash(forensic.pair_audit)
    pair_audit_path = _write_artifact(
        output_dir / "audits", "v9-recovered-v8-pair-audit", pair_audit_document
    )
    expected_event = _expected_canonical_event(
        recovery_plan=recovery_plan,
        forensic=forensic,
        pair_audit_sha256=pair_audit_document["artifact_sha256"],
    )
    global_ledger = state_root / GLOBAL_LEDGER_PATH
    canonical_reconciliation_root = (
        state_root / "flavourbench/artifacts/frontier-contract/reconciliations"
    )
    proof_inputs: list[tuple[Mapping[str, Any], Path]] = []
    for reference in recovery_plan["continuation"]["no_delivery_proofs"]:
        frozen_path = state_root / str(reference["path"])
        record = _load_json(frozen_path)
        if (
            _file_ref(state_root, frozen_path)
            != {key: reference[key] for key in ("path", "bytes", "file_sha256")}
            or (record.get("content_address") or {}).get("digest")
            != reference["content_sha256"]
        ):
            raise RecoveryError("frozen V9 no-delivery proof differs before apply")
        frontier.validate_no_artifact_reconciliation_v2(
            frozen_path,
            ledger_entries=frontier.load_ledger(global_ledger),
        )
        unhashed = dict(record)
        unhashed.pop("content_address", None)
        canonical_path = frontier.write_no_artifact_reconciliation(
            unhashed, canonical_reconciliation_root
        )
        if canonical_path.name != frozen_path.name or _file_sha256(
            canonical_path
        ) != _file_sha256(frozen_path):
            raise RecoveryError("canonical no-delivery proof copy differs")
        proof_inputs.append((reference, canonical_path))
    if len(proof_inputs) != 27:
        raise RecoveryError("apply did not bind 27 no-delivery proofs")
    local_ledger = output_dir / "ledger.jsonl"
    _inject(failure_injector, "before_global_lock")
    with frontier._exclusive_runner_lock(global_ledger):
        with frontier._exclusive_runner_lock(local_ledger):
            _inject(failure_injector, "after_locks_before_revalidation")
            current = verify_forensic_state(
                state_root=state_root,
                exact_global_length=False,
                expected_v8_snapshot=incident["v8_tree_snapshot"],
            )
            entries = list(current.global_entries)
            matches = [
                entry
                for entry in entries
                if entry.get("event_type") == "artifact_recorded"
                and entry.get("reservation_entry_sha256") == V8_FIRST_RESERVATION_SHA256
            ]
            if matches:
                if len(matches) != 1 or any(
                    matches[0].get(key) != value for key, value in expected_event.items()
                ):
                    raise RecoveryError("existing canonical V9 source import differs")
                canonical_event = matches[0]
            else:
                _inject(failure_injector, "before_canonical_artifact_append")
                canonical_event = frontier.append_ledger_event(global_ledger, expected_event)
                _inject(failure_injector, "after_canonical_artifact_append")
            final_entries = frontier.load_ledger(global_ledger)
            matching_after = [
                entry
                for entry in final_entries
                if entry.get("event_type") == "artifact_recorded"
                and entry.get("reservation_entry_sha256") == V8_FIRST_RESERVATION_SHA256
            ]
            if matching_after != [canonical_event]:
                raise RecoveryError("canonical source import is not exactly once")
            release_events: list[dict[str, Any]] = []
            for ordinal, (reference, proof_path) in enumerate(proof_inputs, start=1):
                live_entries = frontier.load_ledger(global_ledger)
                proof = frontier.validate_no_artifact_reconciliation_v2(
                    proof_path,
                    ledger_entries=live_entries,
                )
                expected_release = _expected_release_event(
                    recovery_plan=recovery_plan,
                    proof_reference=reference,
                    proof=proof,
                    forensic=current,
                )
                dispositions = [
                    entry
                    for entry in live_entries
                    if entry.get("event_type")
                    in {"artifact_recorded", "no_artifact_reconciliation_recorded"}
                    and entry.get("reservation_entry_sha256")
                    == proof.reservation_entry_sha256
                ]
                if dispositions:
                    if len(dispositions) != 1 or any(
                        dispositions[0].get(key) != value
                        for key, value in expected_release.items()
                    ):
                        raise RecoveryError("existing V9 no-delivery release differs")
                    release = dispositions[0]
                else:
                    _inject(
                        failure_injector,
                        f"before_no_delivery_reconciliation_append_{ordinal:02d}",
                    )
                    release = frontier.append_ledger_event(global_ledger, expected_release)
                    _inject(
                        failure_injector,
                        f"after_no_delivery_reconciliation_append_{ordinal:02d}",
                    )
                release_events.append(release)
            final_entries = frontier.load_ledger(global_ledger)
            frontier.validate_ledger_artifact_links(
                final_entries,
                frontier.scan_live_smoke_artifacts(state_root / SOURCE_ROOT),
                reconciliation_directory=canonical_reconciliation_root,
            )
            active = frontier.active_ledger_reservations(final_entries)
            v8_reservation_ids = {
                record["entry_sha256"] for record in current.reservations.values()
            }
            if set(active).intersection(v8_reservation_ids):
                raise RecoveryError("a V8 reservation remains active after V9 recovery")
            released = v8_study._exact_sum(
                [Decimal(event["released_exposure_usd"]) for event in release_events]
            )
            if (
                len(release_events) != 27
                or len({event["entry_sha256"] for event in release_events}) != 27
                or v8_study._decimal_text(released) != UNUSED_RESERVATION_USD
            ):
                raise RecoveryError("V9 no-delivery release set or sum differs")
            local_payload = {
                "recovery_plan_sha256": recovery_plan["artifact_sha256"],
                "governance_go_sha256": governance_go["artifact_sha256"],
                "v8_study_plan_sha256": V8_PLAN_SHA256,
                "v8_admission_block_id": V8_BLOCK_ID,
                "work_item_id": V8_FIRST_WORK_ITEM_ID,
                "run_id": V8_FIRST_RUN_ID,
                "canonical_reservation_entry_sha256": V8_FIRST_RESERVATION_SHA256,
                "canonical_artifact_record_entry_sha256": canonical_event["entry_sha256"],
                "canonical_no_delivery_reconciliation_entry_sha256s": [
                    event["entry_sha256"] for event in release_events
                ],
                "no_delivery_reconciliation_sha256s": [
                    reference["content_sha256"]
                    for reference in recovery_plan["continuation"]["no_delivery_proofs"]
                ],
                "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
                "pair_audit": _artifact_ref(state_root, pair_audit_path),
                "disposition": "completed_source_imported_without_replay",
                "actual_cost_usd": ACTUAL_COST_USD,
                "provider_requests_during_recovery": 0,
                "epicure_calls_during_recovery": 0,
                "catalog_requests_during_recovery": 0,
                "new_reservations": 0,
                "released_never_started_reservation_count": 27,
                "released_never_started_reservation_usd": UNUSED_RESERVATION_USD,
                "active_v8_reservations": 0,
                "fresh_v9_continuation_items": 27,
                "fresh_v9_reservations_created": 0,
                "continuation_authorized": False,
                "rank_eligible": False,
            }
            _inject(failure_injector, "before_local_terminal_append")
            terminal = _append_local_terminal(local_ledger, local_payload)
            _inject(failure_injector, "after_local_terminal_append")
            _verify_tree_snapshot(
                state_root=state_root,
                relative_root=V8_ROOT,
                expected=incident["v8_tree_snapshot"],
            )
            if (
                _file_sha256(state_root / SOURCE_ROOT / SOURCE_FILENAME)
                != SOURCE_FILE_SHA256
                or _file_sha256(state_root / SOURCE_ROOT / JOURNAL_FILENAME)
                != JOURNAL_SHA256
            ):
                raise RecoveryError("canonical source or journal changed during recovery")
            receipt = _with_hash(
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "record_role": "append_only_zero_call_v8_incident_recovery_receipt",
                    "recovery_plan_sha256": recovery_plan["artifact_sha256"],
                    "governance_go_sha256": governance_go["artifact_sha256"],
                    "canonical_artifact_record_entry_sha256": canonical_event[
                        "entry_sha256"
                    ],
                    "local_terminal_entry_sha256": terminal["entry_sha256"],
                    "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
                    "pair_audit": _artifact_ref(state_root, pair_audit_path),
                    "actual_cost_usd": ACTUAL_COST_USD,
                    "canonical_artifact_events_for_completed_item": 1,
                    "canonical_no_delivery_reconciliation_events": 27,
                    "canonical_no_delivery_reconciliation_entry_sha256s": [
                        event["entry_sha256"] for event in release_events
                    ],
                    "released_never_started_reservation_usd": UNUSED_RESERVATION_USD,
                    "post_recovery_total_exposure_usd": POST_RECOVERY_EXPOSURE_USD,
                    "v8_active_reservations": 0,
                    "fresh_v9_continuation_items": 27,
                    "fresh_v9_reservations_created": 0,
                    "provider_requests_during_recovery": 0,
                    "epicure_calls_during_recovery": 0,
                    "catalog_requests_during_recovery": 0,
                    "new_reservations": 0,
                    "v8_local_files_mutated": 0,
                    "source_files_mutated": 0,
                    "continuation_authorized": False,
                    "rank_eligible": False,
                }
            )
            _inject(failure_injector, "before_receipt_write")
            receipt_path = _write_artifact(
                output_dir / "receipts", "v9-v8-incident-recovery-receipt", receipt
            )
            _inject(failure_injector, "after_receipt_write")
    return receipt_path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=_default_repo_root())
    parser.add_argument("--code-root", type=Path, default=_default_repo_root())
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, default=Path(V9_ROOT) / "offline")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--bundle", type=Path, required=True)
    apply_parser.add_argument("--governance-go", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path, default=Path(V9_ROOT) / "runs/recovery")
    apply_parser.add_argument("--confirm", required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    state_root = args.state_root.resolve()
    code_root = args.code_root.resolve()
    if args.command == "freeze":
        paths = freeze(
            state_root=state_root,
            code_root=code_root,
            output_dir=_resolve(state_root, args.output_dir),
        )
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
        return
    if args.command == "verify":
        result = verify_bundle(
            bundle_path=_resolve(state_root, args.bundle),
            state_root=state_root,
            code_root=code_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    receipt = apply_recovery(
        state_root=state_root,
        code_root=code_root,
        bundle_path=_resolve(state_root, args.bundle),
        governance_go_path=_resolve(state_root, args.governance_go),
        output_dir=_resolve(state_root, args.output_dir),
        confirmation=args.confirm,
    )
    print(json.dumps({"receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    run()
