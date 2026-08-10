"""Fresh Sonnet-only repair for the closed reasoning-effort v5 route gate.

V5 is never replayed.  This module reconstructs its Sonnet failure, widens the
client-side tool fan-out acceptance cap from 6/12 to 13/13, proves that the
widening was non-binding for the already accepted DeepSeek and Gemini route
cells, and freezes two fresh Sonnet identifiers.  Thirteen is the complete
Epicure catalog size and therefore permits at most one full catalog sweep.

Planning, auditing, manifest construction, and preflight make no provider or
MCP calls.  Paid execution remains separately confirmation-gated.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_route_gate_v5 as v5

FAILURE_AUDIT_SCHEMA = "flavourbench-reasoning-effort-sonnet-v5-failure-audit-v1"
BRIDGE_AUDIT_SCHEMA = "flavourbench-reasoning-effort-fanout-bridge-audit-v1"
ROUTE_PLAN_SCHEMA = "flavourbench-reasoning-effort-sonnet-route-gate-plan-v6"
EXECUTION_PLAN_SCHEMA = "flavourbench-reasoning-effort-sonnet-execution-plan-v6"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-sonnet-execution-receipt-v6"
AUDIT_SCHEMA = "flavourbench-reasoning-effort-sonnet-route-audit-v6"
CLOSURE_SCHEMA = "flavourbench-reasoning-effort-sonnet-route-closure-v6"
AGGREGATE_AUDIT_SCHEMA = "flavourbench-reasoning-effort-route-gate-audit-v6"
AGGREGATE_CLOSURE_SCHEMA = "flavourbench-reasoning-effort-route-gate-closure-v6"

V5_PLAN_SHA = "0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c"
V5_SONNET_RECEIPT_SHA = "6b54b77c744016dd17714b25f7f0e2795600fb204d02e37f11165451a35de7a6"
V5_SONNET_AUDIT_SHA = "4c5e4a6fb796f9791fbf5e1889d3a09fd52fa3a1e67f0e50a0dbb6daeba49feb"
V5_SONNET_CLOSURE_SHA = "99c194969edabe33ebfb942c1bf053515c871c953c65b8b8372400c4b245f068"
V5_GEMINI_RECEIPT_SHA = "157e3aaeb8faf02830c927ddbe035dcb7414900cf900c59bd23db57bf918b803"
V5_GEMINI_AUDIT_SHA = "63da19f18b9c2f3104d6ef775969cc1a0c8750ef5bcacf03cf9f6bfdd0223f23"
V5_GEMINI_CLOSURE_SHA = "44ba45a5c967744ffb9d9b107511a3104c4b0dfd6d708c8fdfddcf28c5ce0c04"
V5_AGGREGATE_AUDIT_SHA = "30271cb2108274271700be203d0eb3c7efde53875ca927c5425021ba27c32a35"
V5_AGGREGATE_CLOSURE_SHA = "e6ce615dbe15c29ae8066990371f7512b532eea2689ddb83fd5917b059d8859b"
V5_SONNET_SOURCE_SHA = "da4a0c1a79c0f46fbe15b000cbafb9003c7e02e7e12a766f1c21598da5cab2b8"
V5_SONNET_JOURNAL_SHA = "e9ce2279f6809603a5d898727c8bf6aaca1b2f0b690fa3d5d3d76c04c60e20e1"
V5_ENDPOINT_SNAPSHOT_SHA = (
    "ce46706dd7c2cb0605c3dd5abc34f36714f09a6074e155b18298393f14a38262"
)
V5_SONNET_DEFAULT_WORK_ID = (
    "aba09cd620977253262a747e706b94468ffe35cc60341ccdb37d7788d3f17144"
)
V5_SONNET_HIGH_WORK_ID = "2b2162532859e80a09548ab808052854eb35766ef6465afc24111fbd69bb3b09"
V5_ERROR = "ProviderError: provider tool-call fan-out (9) exceeded the per-round cap (6)"
V5_ERROR_SHA = "fb1dc9d8ef84ec08a83e22fa9f74700732f7981c473f54d32884b6e81d0cb007"

FREEZE_NONCE = "effort-v6-2026-08-03-sonnet-fanout-13-fresh-identifiers"
NAMESPACE = uuid.UUID("a813d8d6-04b4-5fc2-bfe5-64e24853f420")
CONFIRMATION = "RUN_EXACT_REASONING_EFFORT_V6_SONNET_2_PAIRS"
MAX_TOOL_CALLS_PER_ROUND = 13
MAX_TOOL_CALLS_TOTAL = 13
CONDITIONS = ("epicure_off", "epicure_on")


class RouteGateV6Error(RuntimeError):
    """A frozen v6 input or source-reconstruction predicate failed."""


def _artifact(path: Path, digest: str, schema: str | None = None) -> dict[str, Any]:
    document = v5._regular_json(path)
    if document.get("artifact_sha256") != digest:
        raise RouteGateV6Error(f"unexpected artifact identity: {path}")
    if schema is not None and not v5._artifact_verifies(document, schema):
        raise RouteGateV6Error(f"artifact schema/content address failed: {path}")
    return document


def _source_reference(repo_root: Path, path: Path, semantic: str) -> dict[str, Any]:
    return {
        "path": v5._relative(repo_root, path),
        "bytes": path.stat().st_size,
        "file_sha256": v5._file_sha256(path),
        "semantic_sha256": semantic,
    }


def _manifest_reference(repo_root: Path, path: Path, digest: str) -> dict[str, Any]:
    return _source_reference(repo_root, path, digest)


def build_failure_audit(
    *,
    v5_plan_path: Path,
    receipt_path: Path,
    audit_path: Path,
    closure_path: Path,
    aggregate_audit_path: Path,
    aggregate_closure_path: Path,
    ledger_path: Path,
    source_path: Path,
    journal_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    from .frontier_contract_runner import _verify_live_artifact
    from .real_dataset_runner import load_dataset_ledger

    plan = _artifact(v5_plan_path, V5_PLAN_SHA, v5.ROUTE_PLAN_SCHEMA)
    receipt = _artifact(receipt_path, V5_SONNET_RECEIPT_SHA, v5.ENDPOINT_RECEIPT_SCHEMA)
    audit = _artifact(audit_path, V5_SONNET_AUDIT_SHA, v5.ENDPOINT_AUDIT_SCHEMA)
    closure = _artifact(closure_path, V5_SONNET_CLOSURE_SHA, v5.ENDPOINT_CLOSURE_SCHEMA)
    aggregate_audit = _artifact(
        aggregate_audit_path, V5_AGGREGATE_AUDIT_SHA, v5.AGGREGATE_AUDIT_SCHEMA
    )
    aggregate_closure = _artifact(
        aggregate_closure_path,
        V5_AGGREGATE_CLOSURE_SHA,
        v5.AGGREGATE_CLOSURE_SCHEMA,
    )
    source, source_digest = _verify_live_artifact(source_path)
    if source_digest != V5_SONNET_SOURCE_SHA:
        raise RouteGateV6Error("Sonnet v5 source digest differs")
    v5._hash_chain_jsonl(journal_path)
    ledger = load_dataset_ledger(ledger_path)
    if v5._file_sha256(journal_path) != V5_SONNET_JOURNAL_SHA:
        raise RouteGateV6Error("Sonnet v5 journal digest differs")
    if source.get("run_journal", {}).get("sha256") != V5_SONNET_JOURNAL_SHA:
        raise RouteGateV6Error("source does not bind the exact finalized journal")
    if source.get("errors") != {"epicure_on": V5_ERROR}:
        raise RouteGateV6Error("Sonnet v5 error is not the exact fan-out failure")
    if hashlib.sha256(V5_ERROR.encode()).hexdigest() != V5_ERROR_SHA:
        raise RouteGateV6Error("Sonnet v5 error preimage does not match")
    policy = source.get("execution_policy") or {}
    limits = policy.get("limits") or {}
    events = [
        item
        for item in source.get("provider_attempt_events") or []
        if isinstance(item, Mapping)
    ]
    counts = Counter(str(item.get("event_type") or "") for item in events)
    responses = [item for item in events if item.get("event_type") == "response_received"]
    starts = [item for item in events if item.get("event_type") == "request_started"]
    reconciled = [
        item for item in events if item.get("event_type") == "accounting_reconciled"
    ]
    if (
        limits.get("max_tool_calls_per_round") != 6
        or limits.get("max_tool_calls_total") != 12
        or len(starts) != len(responses)
        or len(responses) != len(reconciled)
        or len(responses) != 5
        or counts.get("mcp_session_started") != 1
        or counts.get("mcp_session_attested") != 1
        or counts.get("mcp_call_started", 0) != 0
        or source.get("mcp_trace_events") != []
    ):
        raise RouteGateV6Error("Sonnet request/accounting/MCP boundary differs")
    response_ids = {str(item.get("generation_id") or "") for item in responses}
    metadata = [
        item.get("metadata")
        for item in reconciled
        if isinstance(item.get("metadata"), Mapping)
    ]
    metadata_ids = {str(item.get("generation_id") or "") for item in metadata}
    total_micros = sum(int(item.get("cost_micros") or 0) for item in metadata)
    if (
        response_ids != metadata_ids
        or len(response_ids) != 5
        or total_micros != 61_742
        or source.get("budget", {}).get("actual_cost_micros") != 61_742
        or source.get("budget", {}).get("all_generation_costs_reconciled") is not True
    ):
        raise RouteGateV6Error("all five Sonnet generations do not reconcile exactly")
    finalizations = [
        item
        for item in ledger
        if item.get("event_type") == "source_artifact_recorded"
        and item.get("work_item_id") == V5_SONNET_DEFAULT_WORK_ID
    ]
    if (
        len(finalizations) != 1
        or finalizations[0].get("route_gate_pair_passed") is not False
        or finalizations[0].get("actual_cost_usd") != "0.033475"
    ):
        raise RouteGateV6Error("v5 Sonnet ledger finalization differs")
    closed = closure.get("closed_identifiers") or {}
    if (
        set(closed.get("work_item_ids") or [])
        != {V5_SONNET_DEFAULT_WORK_ID, V5_SONNET_HIGH_WORK_ID}
        or closed.get("replay_permitted") is not False
        or closure.get("decision", {}).get("endpoint_qualified") is not False
        or aggregate_closure.get("decision", {}).get("route_gate_qualified") is not False
    ):
        raise RouteGateV6Error("v5 Sonnet/aggregate closures are not fail-closed")
    if audit.get("decision") != "failed_one_or_more_predicates" or aggregate_audit.get(
        "failures"
    ) != ["sonnet_did_not_pass"]:
        raise RouteGateV6Error("v5 source and aggregate audit decisions differ")
    if receipt.get("final_budget", {}).get("v5_actual_cost_usd") != "0.061742":
        raise RouteGateV6Error("v5 receipt does not retain complete source cost")
    return {
        "schema_version": FAILURE_AUDIT_SCHEMA,
        "record_role": "source_reconstructed_closed_sonnet_v5_fanout_failure",
        "v5_bindings": {
            "route_plan_sha256": plan["artifact_sha256"],
            "receipt_sha256": receipt["artifact_sha256"],
            "endpoint_audit_sha256": audit["artifact_sha256"],
            "endpoint_closure_sha256": closure["artifact_sha256"],
            "aggregate_audit_sha256": aggregate_audit["artifact_sha256"],
            "aggregate_closure_sha256": aggregate_closure["artifact_sha256"],
            "ledger_sha256": v5._file_sha256(ledger_path),
            "source_sha256": source_digest,
            "journal_sha256": v5._file_sha256(journal_path),
        },
        "root_cause": {
            "exception": V5_ERROR,
            "exception_sha256": V5_ERROR_SHA,
            "returned_tool_calls": 9,
            "frozen_per_round_cap": 6,
            "frozen_total_cap": 12,
            "failure_stage": "after_first_epicure_tool_selection_before_any_mcp_call",
            "provider_or_endpoint_failure": False,
            "epicure_service_failure": False,
            "client_protocol_limit_triggered": True,
        },
        "request_and_tool_boundary": {
            "provider_requests_started": len(starts),
            "accepted_chat_completions": len(responses),
            "generation_cost_records_reconciled": len(reconciled),
            "mcp_sessions_started": counts.get("mcp_session_started", 0),
            "mcp_sessions_attested": counts.get("mcp_session_attested", 0),
            "mcp_calls_started": counts.get("mcp_call_started", 0),
            "mcp_calls_completed": counts.get("mcp_call_completed", 0),
        },
        "accounting": {
            "complete_generation_cost_micros": total_micros,
            "complete_generation_cost_usd": "0.061742",
            "v5_ledger_partial_result_cost_usd": "0.033475",
            "ledger_understatement_usd": "0.028267",
            "complete_cost_retained_by_v5_final_budget": True,
        },
        "closure": {
            "v5_default_and_high_identifiers_closed": True,
            "v5_replay_permitted": False,
            "v5_aggregate_gate_qualified": False,
            "fresh_identifiers_required": True,
        },
        "claim_boundary": {
            "v5_sonnet_pair_usable": False,
            "v5_sonnet_quality_observations": 0,
            "official": False,
            "rank_eligible": False,
        },
    }


def _fanout_projection(source: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    on = (source.get("results") or {}).get("epicure_on") or {}
    outputs = [
        item
        for item in on.get("intermediate_outputs") or []
        if isinstance(item, Mapping) and item.get("tool_call_count") is not None
    ]
    counts = [int(item["tool_call_count"]) for item in outputs]
    starts = [
        item
        for item in source.get("provider_attempt_events") or []
        if isinstance(item, Mapping) and item.get("event_type") == "request_started"
    ]
    request_contracts = [
        (item.get("metadata") or {}).get("request_contract") or {} for item in starts
    ]
    forbidden = {
        key
        for contract in request_contracts
        for key in contract
        if "tool_calls_per_round" in str(key) or "tool_calls_total" in str(key)
    }
    return {
        "work_item_id": source.get("dataset_work_item_id"),
        "run_id": source.get("run_id"),
        "model_id": source.get("requested_model_id"),
        "variant_id": (
            "provider_default"
            if source.get("frozen_generation_contract", {}).get(
                "intermediate_reasoning_effort"
            )
            is None
            else "explicit_high"
        ),
        "source_path": str(source_path),
        "source_artifact_sha256": source.get("artifact_sha256"),
        "observed_tool_calls_by_selection_turn": counts,
        "maximum_observed_round_fanout": max(counts or [0]),
        "executed_tool_calls_total": len(on.get("tool_trace") or []),
        "client_fanout_caps_present_in_provider_request_contract": bool(forbidden),
        "request_contract_forbidden_keys": sorted(forbidden),
    }


def build_bridge_audit(
    *,
    v4_route_plan_path: Path,
    v4_closure_path: Path,
    v4_source_directory: Path,
    v5_plan_path: Path,
    gemini_audit_path: Path,
    gemini_closure_path: Path,
    gemini_source_directory: Path,
    repo_root: Path,
) -> dict[str, Any]:
    from .frontier_contract_runner import _verify_live_artifact
    from .reasoning_effort_route_gate_v4 import _audit_pair_source, _source_map

    v4_plan = _artifact(v4_route_plan_path, v5.V4_ROUTE_PLAN_SHA256)
    v4_closure = _artifact(v4_closure_path, v5.V4_CLOSURE_SHA256)
    v5_plan = _artifact(v5_plan_path, V5_PLAN_SHA, v5.ROUTE_PLAN_SCHEMA)
    gemini_audit = _artifact(
        gemini_audit_path, V5_GEMINI_AUDIT_SHA, v5.ENDPOINT_AUDIT_SCHEMA
    )
    gemini_closure = _artifact(
        gemini_closure_path, V5_GEMINI_CLOSURE_SHA, v5.ENDPOINT_CLOSURE_SCHEMA
    )
    if (
        gemini_audit.get("decision") != "passed_all_predicates"
        or gemini_closure.get("decision", {}).get("endpoint_qualified") is not True
        or gemini_closure.get("closed_identifiers", {}).get("replay_permitted") is not False
    ):
        raise RouteGateV6Error("Gemini v5 endpoint is not passed and permanently closed")
    projections: list[dict[str, Any]] = []
    v4_sources = _source_map(v4_source_directory)
    for item in v4_plan["work_items"][:2]:
        path, source, digest = v4_sources[item["work_item_id"]]
        pair = _audit_pair_source(
            route_plan=v4_plan,
            work_item=item,
            source_path=path,
            source=source,
            source_digest=digest,
            repo_root=repo_root,
        )
        if pair.get("decision") != "passed_all_predicates":
            raise RouteGateV6Error("preserved DeepSeek source no longer passes")
        projection = _fanout_projection(source, path)
        projection["source_path"] = v5._relative(repo_root, path)
        projections.append(projection)
    gemini_items = {
        item["work_item_id"]: item
        for item in v5_plan["work_items"]
        if item["endpoint_id"] == "gemini"
    }
    for path in sorted(gemini_source_directory.glob("*.json")):
        source, digest = _verify_live_artifact(path)
        item = gemini_items.get(source.get("dataset_work_item_id"))
        if item is None:
            raise RouteGateV6Error("unknown Gemini source in bridge directory")
        pair = v5._adapted_pair_audit(
            plan=v5_plan,
            item=item,
            source_path=path,
            source=source,
            digest=digest,
            repo_root=repo_root,
        )
        if pair.get("decision") != "passed_all_predicates":
            raise RouteGateV6Error("preserved Gemini source no longer passes")
        projection = _fanout_projection(source, path)
        projection["source_path"] = v5._relative(repo_root, path)
        projections.append(projection)
    if len(projections) != 4:
        raise RouteGateV6Error("bridge requires two DeepSeek and two Gemini sources")
    if any(
        item["maximum_observed_round_fanout"] > 6
        or item["executed_tool_calls_total"] > 12
        or item["client_fanout_caps_present_in_provider_request_contract"]
        for item in projections
    ):
        raise RouteGateV6Error("13/13 widening is not observationally non-binding")
    prior_closed = set(v4_closure.get("closed_identifiers", {}).get("work_item_ids") or [])
    if not {item["work_item_id"] for item in projections[:2]} <= prior_closed:
        raise RouteGateV6Error("DeepSeek bridge identifiers are not closed by v4")
    return {
        "schema_version": BRIDGE_AUDIT_SCHEMA,
        "record_role": "monotone_client_fanout_cap_widening_bridge",
        "source_route_bindings": {
            "v4_route_plan_sha256": v4_plan["artifact_sha256"],
            "v4_closure_sha256": v4_closure["artifact_sha256"],
            "v5_route_plan_sha256": v5_plan["artifact_sha256"],
            "v5_gemini_audit_sha256": gemini_audit["artifact_sha256"],
            "v5_gemini_closure_sha256": gemini_closure["artifact_sha256"],
        },
        "protocol_change": {
            "field": "client_side_tool_fanout_acceptance",
            "old_max_tool_calls_per_round": 6,
            "old_max_tool_calls_total": 12,
            "new_max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
            "new_max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
            "provider_request_payload_changed": False,
            "epicure_tool_catalog_changed": False,
            "tool_result_byte_caps_changed": False,
            "model_output_or_tool_trajectory_changed_for_bridged_sources": False,
        },
        "source_projections": projections,
        "decision": "four_prior_cells_observationally_invariant_under_v6_caps",
        "scope": {
            "route_compatibility_bridge_only": True,
            "quality_observations": 0,
            "retroactive_result_relabeling": False,
            "official": False,
            "rank_eligible": False,
        },
    }


def v6_policy(base_manifest: Mapping[str, Any], *, variant: str | None = None):
    from .response_envelope_route_v4 import _policy_from_manifest

    base = _policy_from_manifest(base_manifest)
    policy = replace(
        base,
        max_tool_calls_per_round=MAX_TOOL_CALLS_PER_ROUND,
        max_tool_calls_total=MAX_TOOL_CALLS_TOTAL,
        intermediate_reasoning_effort=("high" if variant == "explicit_high" else None)
        if variant is not None
        else base.intermediate_reasoning_effort,
        final_reasoning_effort=("high" if variant == "explicit_high" else None)
        if variant is not None
        else base.final_reasoning_effort,
    )
    policy.validate()
    return policy


def build_v6_manifest(
    *,
    base_manifest_path: Path,
    failure_audit: Mapping[str, Any],
    bridge_audit: Mapping[str, Any],
) -> dict[str, Any]:
    from .frontier_manifest import verify_manifest_content_address
    from .reasoning_effort_sensitivity_v4 import _manifest

    base = _manifest(base_manifest_path)
    policy = v6_policy(base)
    sonnet = [
        copy.deepcopy(item)
        for item in base.get("models") or []
        if (item.get("model") or {}).get("id") == "anthropic/claude-sonnet-5"
    ]
    if len(sonnet) != 1:
        raise RouteGateV6Error("base manifest lacks one Sonnet 5 route")
    payload = copy.deepcopy(base)
    payload.pop("content_address", None)
    payload["schema_version"] = "flavourbench-reasoning-effort-route-manifest-v6"
    payload["manifest_role"] = "sonnet_reasoning_effort_v6_route_gate_only"
    payload["status"] = "frozen_not_executed"
    payload["models"] = sonnet
    run_design = payload["run_design"]
    run_design["assignments_per_model"] = 1
    run_design["selected_task_count"] = 1
    run_design["expected_pairs"] = 2
    run_design["expected_arms"] = 4
    run_design["execution_policy"] = policy.document()
    run_design["execution_policy_sha256"] = policy.sha256
    run_design["generation_protocol"]["tool_fanout_acceptance"] = {
        "catalog_tool_count": 13,
        "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
        "max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
        "interpretation": "at_most_one_complete_catalog_sweep_per_arm",
        "client_side_only": True,
    }
    payload["budget"] = {
        "currency": "USD",
        "pair_worst_case_usd": "1.148724",
        "two_pair_worst_case_usd": "2.297448",
        "generation_spend_authorized_by_manifest": False,
    }
    payload["selection"] = {
        "model": "anthropic/claude-sonnet-5",
        "provider": "anthropic",
        "task_id": "fb-s0-substitution-003",
        "variants": ["provider_default", "explicit_high"],
        "quality_observations_used": 0,
    }
    payload["source"] = {
        **dict(payload.get("source") or {}),
        "base_manifest_sha256": base["content_address"]["digest"],
        "v5_failure_audit_sha256": failure_audit["artifact_sha256"],
        "fanout_bridge_audit_sha256": bridge_audit["artifact_sha256"],
    }
    payload["governance"] = {
        **dict(payload.get("governance") or {}),
        "manifest_class": "diagnostic_sonnet_route_gate_v6",
        "official": False,
        "rank_eligible": False,
        "v5_replay_permitted": False,
        "post_gate_protocol_fix_before_confirmatory_collection": True,
        "bridge_scope": "non_binding_route_compatibility_only",
    }
    digest = v5._sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(payload):
        raise RouteGateV6Error("derived v6 manifest content address failed")
    return payload


def _write_manifest(directory: Path, manifest: Mapping[str, Any]) -> Path:
    from .frontier_manifest import verify_manifest_content_address

    if not verify_manifest_content_address(manifest):
        raise RouteGateV6Error("refusing an invalid v6 manifest")
    digest = str(manifest["content_address"]["digest"])
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"flavourbench-reasoning-effort-v6-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RouteGateV6Error("content-addressed v6 manifest conflict")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _attempt_slots(run_id: str, route_cell_id: str) -> list[dict[str, Any]]:
    """Preallocate every external-call attempt under the fresh v6 namespace."""

    coordinates: list[tuple[str, str, int]] = []
    off = f"{run_id}:epicure_off"
    on = f"{run_id}:epicure_on"
    for phase in ("planning", "evidence_decision", "final"):
        coordinates.extend((off, phase, attempt) for attempt in (0, 1))
    for phase in ("planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"):
        coordinates.extend((on, phase, attempt) for attempt in (0, 1))
    coordinates.append((on, "mcp_session", 0))
    for round_index in range(3):
        for call_index in range(MAX_TOOL_CALLS_PER_ROUND):
            coordinates.append((on, f"mcp_tool_{round_index}_{call_index}", 0))
    return [
        {
            "arm_id": arm_id,
            "phase": phase,
            "attempt_index": attempt_index,
            "attempt_id": str(
                uuid.uuid5(
                    NAMESPACE,
                    f"{FREEZE_NONCE}:{route_cell_id}:{arm_id}:{phase}:{attempt_index}",
                )
            ),
        }
        for arm_id, phase, attempt_index in coordinates
    ]


def _closed_identifier_union(*closures: Mapping[str, Any]) -> dict[str, set[str]]:
    keys = (
        "route_cell_ids",
        "work_item_ids",
        "run_ids",
        "arm_ids",
        "attempt_ids",
        "generation_ids",
        "request_key_sha256s",
    )
    result = {key: set() for key in keys}
    for closure in closures:
        closed = closure.get("closed_identifiers") or {}
        if closed.get("replay_permitted") is not False:
            raise RouteGateV6Error("every predecessor closure must prohibit replay")
        for key in keys:
            result[key].update(str(value) for value in closed.get(key) or [] if value)
    return result


def _assert_embedded_artifact(
    document: Mapping[str, Any], schema: str, *, label: str
) -> None:
    if not v5._artifact_verifies(document, schema):
        raise RouteGateV6Error(f"{label} content address or schema failed")


def _source_code_binding(repo_root: Path) -> dict[str, Any]:
    module = Path(__file__).resolve()
    v5_module = Path(v5.__file__).resolve()
    return {
        "executor": {
            "path": v5._relative(repo_root, module),
            "bytes": module.stat().st_size,
            "sha256": v5._file_sha256(module),
        },
        "frozen_v5_generation_adapter": {
            "path": v5._relative(repo_root, v5_module),
            "bytes": v5_module.stat().st_size,
            "sha256": v5._file_sha256(v5_module),
        },
    }


def build_route_plan(
    *,
    v5_plan_path: Path,
    v4_closure_path: Path,
    v5_sonnet_closure_path: Path,
    v5_gemini_closure_path: Path,
    v5_aggregate_closure_path: Path,
    endpoint_snapshot_path: Path,
    failure_audit_path: Path,
    bridge_audit_path: Path,
    v6_manifest_path: Path,
    v5_gemini_receipt_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Freeze two fresh Sonnet pairs without making any external call."""

    from .frontier_manifest import verify_manifest_content_address

    predecessor = _artifact(v5_plan_path, V5_PLAN_SHA, v5.ROUTE_PLAN_SCHEMA)
    v4_closure = _artifact(v4_closure_path, v5.V4_CLOSURE_SHA256)
    sonnet_closure = _artifact(
        v5_sonnet_closure_path, V5_SONNET_CLOSURE_SHA, v5.ENDPOINT_CLOSURE_SCHEMA
    )
    gemini_closure = _artifact(
        v5_gemini_closure_path, V5_GEMINI_CLOSURE_SHA, v5.ENDPOINT_CLOSURE_SCHEMA
    )
    aggregate_closure = _artifact(
        v5_aggregate_closure_path,
        V5_AGGREGATE_CLOSURE_SHA,
        v5.AGGREGATE_CLOSURE_SCHEMA,
    )
    snapshot = _artifact(
        endpoint_snapshot_path, V5_ENDPOINT_SNAPSHOT_SHA, v5.SNAPSHOT_SCHEMA
    )
    failure = v5._regular_json(failure_audit_path)
    bridge = v5._regular_json(bridge_audit_path)
    _assert_embedded_artifact(failure, FAILURE_AUDIT_SCHEMA, label="failure audit")
    _assert_embedded_artifact(bridge, BRIDGE_AUDIT_SCHEMA, label="bridge audit")
    manifest = v5._regular_json(v6_manifest_path)
    if not verify_manifest_content_address(manifest):
        raise RouteGateV6Error("v6 manifest content address failed")
    receipt = _artifact(
        v5_gemini_receipt_path, V5_GEMINI_RECEIPT_SHA, v5.ENDPOINT_RECEIPT_SCHEMA
    )
    if (
        failure.get("closure", {}).get("v5_replay_permitted") is not False
        or failure.get("root_cause", {}).get("client_protocol_limit_triggered") is not True
        or bridge.get("decision")
        != "four_prior_cells_observationally_invariant_under_v6_caps"
        or aggregate_closure.get("decision", {}).get("replay_permitted") is not False
    ):
        raise RouteGateV6Error("v6 predecessor/bridge admission predicates failed")
    baseline = str(receipt.get("final_budget", {}).get("current_total_exposure_usd"))
    if baseline != "47.32616982666666666666666666":
        raise RouteGateV6Error("v6 budget baseline is not the final exact Gemini receipt")

    model = next(
        (
            copy.deepcopy(item)
            for item in predecessor.get("models") or []
            if item.get("endpoint_id") == "sonnet"
        ),
        None,
    )
    if model is None:
        raise RouteGateV6Error("v5 predecessor lacks the Sonnet endpoint")
    snapshot_model = v5._endpoint_record(snapshot, "sonnet")
    if (
        snapshot_model.get("raw_execution_contract_sha256")
        != model["snapshot_raw_execution_contract_sha256"]
        or snapshot_model.get("semantic_execution_contract_sha256")
        != model["semantic_execution_contract_sha256"]
    ):
        raise RouteGateV6Error("Sonnet v5 endpoint snapshot no longer matches predecessor")

    old_sonnet = {
        item["route_coordinate"]["variant_id"]: item
        for item in predecessor["work_items"]
        if item["endpoint_id"] == "sonnet"
    }
    variants = copy.deepcopy(predecessor["variants"])
    manifest_digest = manifest["content_address"]["digest"]
    items: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = variant["variant_id"]
        policy = v6_policy(manifest, variant=variant_id)
        coordinate = {
            "schema_version": "flavourbench-reasoning-effort-route-coordinate-v6",
            "freeze_nonce": FREEZE_NONCE,
            "predecessor_route_plan_sha256": predecessor["artifact_sha256"],
            "v6_manifest_sha256": manifest_digest,
            "endpoint_snapshot_sha256": snapshot["artifact_sha256"],
            "endpoint_id": "sonnet",
            "model_id": model["model_id"],
            "canonical_model_slug": model["canonical_model_slug"],
            "provider_endpoint": model["provider_endpoint"],
            "actual_provider_name": model["actual_provider_name"],
            "snapshot_raw_execution_contract_sha256": model[
                "snapshot_raw_execution_contract_sha256"
            ],
            "endpoint_execution_contract_sha256": model[
                "snapshot_raw_execution_contract_sha256"
            ],
            "semantic_execution_contract_sha256": model[
                "semantic_execution_contract_sha256"
            ],
            "provider_controls": model["provider_controls"],
            "task_id": predecessor["task"]["task_id"],
            "prompt_sha256": predecessor["task"]["prompt_sha256"],
            "variant_id": variant_id,
            "intermediate_reasoning_effort": variant["intermediate_reasoning_effort"],
            "final_reasoning_effort": variant["final_reasoning_effort"],
            "execution_policy_sha256": policy.sha256,
            "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
            "max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
            "epicure_bundle_sha256": predecessor["epicure"]["bundle_sha256"],
            "epicure_application_sha256": predecessor["epicure"][
                "application_sha256"
            ],
            "epicure_tool_schema_sha256": predecessor["epicure"][
                "tool_schema_sha256"
            ],
        }
        route_cell_id = v5._sha256(coordinate)
        work_item_id = v5._sha256(
            {
                "route_cell_id": route_cell_id,
                "role": "effort-v6-sonnet-fanout-recovery-gate",
            }
        )
        run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
        reserve = old_sonnet[variant_id]["worst_case_reserve_usd"]
        if reserve != "1.148724":
            raise RouteGateV6Error("Sonnet predecessor reserve differs from frozen envelope")
        items.append(
            {
                "endpoint_id": "sonnet",
                "route_coordinate": coordinate,
                "route_cell_id": route_cell_id,
                "work_item_id": work_item_id,
                "run_id": run_id,
                "arm_ids": [f"{run_id}:{condition}" for condition in CONDITIONS],
                "attempt_slots": _attempt_slots(run_id, route_cell_id),
                "worst_case_reserve_usd": reserve,
                "diagnostic_outputs_reused": False,
            }
        )

    prior = _closed_identifier_union(v4_closure, sonnet_closure, gemini_closure)
    fresh = {
        "route_cell_ids": [item["route_cell_id"] for item in items],
        "work_item_ids": [item["work_item_id"] for item in items],
        "run_ids": [item["run_id"] for item in items],
        "arm_ids": [value for item in items for value in item["arm_ids"]],
        "attempt_ids": [
            slot["attempt_id"] for item in items for slot in item["attempt_slots"]
        ],
    }
    for key, values in fresh.items():
        if len(values) != len(set(values)) or set(values) & prior[key]:
            raise RouteGateV6Error(f"fresh v6 {key} overlap sibling or predecessor")

    reserve = sum(Decimal(item["worst_case_reserve_usd"]) for item in items)
    projected = Decimal(baseline) + reserve
    if reserve != Decimal("2.297448") or projected != Decimal(
        "49.62361782666666666666666666"
    ):
        raise RouteGateV6Error("v6 two-pair budget projection differs")
    sources = {
        "v5_route_plan": _source_reference(
            repo_root, v5_plan_path, predecessor["artifact_sha256"]
        ),
        "v4_closure": _source_reference(
            repo_root, v4_closure_path, v4_closure["artifact_sha256"]
        ),
        "v5_sonnet_closure": _source_reference(
            repo_root, v5_sonnet_closure_path, sonnet_closure["artifact_sha256"]
        ),
        "v5_gemini_closure": _source_reference(
            repo_root, v5_gemini_closure_path, gemini_closure["artifact_sha256"]
        ),
        "v5_aggregate_closure": _source_reference(
            repo_root, v5_aggregate_closure_path, aggregate_closure["artifact_sha256"]
        ),
        "endpoint_snapshot": _source_reference(
            repo_root, endpoint_snapshot_path, snapshot["artifact_sha256"]
        ),
        "failure_audit": _source_reference(
            repo_root, failure_audit_path, failure["artifact_sha256"]
        ),
        "fanout_bridge_audit": _source_reference(
            repo_root, bridge_audit_path, bridge["artifact_sha256"]
        ),
        "manifest_v29": _manifest_reference(repo_root, v6_manifest_path, manifest_digest),
        "v6_manifest": _manifest_reference(repo_root, v6_manifest_path, manifest_digest),
        "v5_gemini_receipt": _source_reference(
            repo_root, v5_gemini_receipt_path, receipt["artifact_sha256"]
        ),
    }
    return {
        "schema_version": ROUTE_PLAN_SCHEMA,
        "record_role": "fresh_sonnet_only_fanout_recovery_route_gate",
        "freeze_nonce": FREEZE_NONCE,
        "task": predecessor["task"],
        "epicure": predecessor["epicure"],
        "variants": variants,
        "models": [model],
        "work_items": items,
        "execution_order": [item["work_item_id"] for item in items],
        "confirmation": CONFIRMATION,
        "budget": {
            "currency": "USD",
            "exact_v5_baseline_receipt_sha256": receipt["artifact_sha256"],
            "current_total_exposure_usd": baseline,
            "new_two_pair_worst_case_usd": v5._decimal_text(reserve),
            "projected_total_exposure_usd": v5._decimal_text(projected),
            "admission_ceiling_usd": "85",
            "hard_cap_usd": "100",
            "reserve_reuse_justification": (
                "provider_round_count_and_65536-byte cumulative tool-result cap unchanged"
            ),
        },
        "source_artifacts": sources,
        "source_code": _source_code_binding(repo_root),
        "counts": {
            "fresh_pairs": 2,
            "fresh_response_arms": 4,
            "effort_variants": 2,
            "models": 1,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "protocol_recovery": {
            "old_max_tool_calls_per_round": 6,
            "old_max_tool_calls_total": 12,
            "new_max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
            "new_max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
            "epicure_catalog_tool_count": 13,
            "at_most_one_complete_catalog_sweep": True,
            "provider_request_payload_changed": False,
            "provider_route_changed": False,
            "epicure_bundle_changed": False,
        },
        "acceptance": {
            "both_pairs_required": True,
            "each_arm_substantive": True,
            "epicure_off_zero_tool_calls": True,
            "epicure_on_successful_real_tool_call": True,
            "provider_default_reasoning_field_absent": True,
            "explicit_high_reasoning_effort_high": True,
            "all_generation_costs_reconciled": True,
            "stop_and_close_suffix_on_first_failure": True,
            "old_identifiers_replay_permitted": False,
            "new_identifiers_replay_permitted": False,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "route_compatibility_only": True,
            "quality_effect_estimable": False,
            "official": False,
            "rank_eligible": False,
            "enters_sensitivity_fit": False,
        },
    }


def validate_route_plan(plan: Mapping[str, Any], *, repo_root: Path) -> None:
    """Reopen all frozen inputs and rederive v6 identities and policy hashes."""

    from .frontier_manifest import verify_manifest_content_address
    from .reasoning_effort_route_gate_v4 import _variant_policy

    if not v5._artifact_verifies(plan, ROUTE_PLAN_SCHEMA):
        raise RouteGateV6Error("v6 route plan content address or schema failed")
    for label, record in (plan.get("source_code") or {}).items():
        path = repo_root / str(record.get("path") or "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or v5._file_sha256(path) != record.get("sha256")
        ):
            raise RouteGateV6Error(f"frozen source code differs: {label}")
    source_documents: dict[str, dict[str, Any]] = {}
    for label, record in (plan.get("source_artifacts") or {}).items():
        path = repo_root / str(record.get("path") or "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or v5._file_sha256(path) != record.get("file_sha256")
        ):
            raise RouteGateV6Error(f"frozen source artifact differs: {label}")
        document = v5._regular_json(path)
        if label in {"manifest_v29", "v6_manifest"}:
            if (
                not verify_manifest_content_address(document)
                or document.get("content_address", {}).get("digest")
                != record.get("semantic_sha256")
            ):
                raise RouteGateV6Error(f"manifest source binding differs: {label}")
        elif document.get("artifact_sha256") != record.get("semantic_sha256"):
            raise RouteGateV6Error(f"semantic source binding differs: {label}")
        source_documents[label] = document
    manifest = source_documents["v6_manifest"]
    policy_limits = manifest["run_design"]["execution_policy"]["limits"]
    if (
        policy_limits.get("max_tool_calls_per_round") != MAX_TOOL_CALLS_PER_ROUND
        or policy_limits.get("max_tool_calls_total") != MAX_TOOL_CALLS_TOTAL
    ):
        raise RouteGateV6Error("v6 manifest does not bind the 13/13 policy")
    if (
        plan.get("counts")
        != {
            "fresh_pairs": 2,
            "fresh_response_arms": 4,
            "effort_variants": 2,
            "models": 1,
            "synthetic_arms": 0,
            "quality_observations": 0,
        }
        or plan.get("confirmation") != CONFIRMATION
        or plan.get("claim_boundary", {}).get("quality_effect_estimable") is not False
    ):
        raise RouteGateV6Error("v6 count, confirmation, or claim boundary differs")
    items = plan.get("work_items") or []
    if len(items) != 2 or plan.get("execution_order") != [
        item["work_item_id"] for item in items
    ]:
        raise RouteGateV6Error("v6 requires exactly two ordered work items")
    for item in items:
        coordinate = item["route_coordinate"]
        route_cell_id = v5._sha256(coordinate)
        work_item_id = v5._sha256(
            {
                "route_cell_id": route_cell_id,
                "role": "effort-v6-sonnet-fanout-recovery-gate",
            }
        )
        run_id = str(uuid.uuid5(NAMESPACE, f"{route_cell_id}:{work_item_id}"))
        if (
            item.get("route_cell_id") != route_cell_id
            or item.get("work_item_id") != work_item_id
            or item.get("run_id") != run_id
            or item.get("arm_ids")
            != [f"{run_id}:{condition}" for condition in CONDITIONS]
            or item.get("attempt_slots") != _attempt_slots(run_id, route_cell_id)
            or item.get("diagnostic_outputs_reused") is not False
        ):
            raise RouteGateV6Error("v6 work-item identifiers do not rederive")
        policy = _variant_policy(plan, item, repo_root)
        if (
            policy.max_tool_calls_per_round != MAX_TOOL_CALLS_PER_ROUND
            or policy.max_tool_calls_total != MAX_TOOL_CALLS_TOTAL
            or policy.sha256 != coordinate.get("execution_policy_sha256")
        ):
            raise RouteGateV6Error("v6 variant policy does not rederive")
    closures = [
        source_documents[name]
        for name in (
            "v4_closure",
            "v5_sonnet_closure",
            "v5_gemini_closure",
        )
    ]
    prior = _closed_identifier_union(*closures)
    observed = {
        "route_cell_ids": {item["route_cell_id"] for item in items},
        "work_item_ids": {item["work_item_id"] for item in items},
        "run_ids": {item["run_id"] for item in items},
        "arm_ids": {value for item in items for value in item["arm_ids"]},
        "attempt_ids": {
            slot["attempt_id"] for item in items for slot in item["attempt_slots"]
        },
    }
    if any(values & prior[key] for key, values in observed.items()):
        raise RouteGateV6Error("v6 identifiers overlap a closed predecessor")
    reserve = sum(Decimal(item["worst_case_reserve_usd"]) for item in items)
    budget = plan.get("budget") or {}
    if (
        reserve != Decimal("2.297448")
        or Decimal(budget.get("current_total_exposure_usd"))
        != Decimal("47.32616982666666666666666666")
        or Decimal(budget.get("projected_total_exposure_usd"))
        != Decimal("49.62361782666666666666666666")
    ):
        raise RouteGateV6Error("v6 route budget does not rederive")


def _work_items(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = [dict(item) for item in plan.get("work_items") or []]
    if len(items) != 2 or [item["work_item_id"] for item in items] != plan.get(
        "execution_order"
    ):
        raise RouteGateV6Error("v6 execution order is not exactly two Sonnet items")
    return items


def _source_map(directory: Path) -> dict[str, tuple[Path, dict[str, Any], str]]:
    return v5._source_map(directory)


def _pair_audit(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    source_path: Path,
    source: Mapping[str, Any],
    digest: str,
    repo_root: Path,
) -> dict[str, Any]:
    pair = v5._adapted_pair_audit(
        plan=plan,
        item=item,
        source_path=source_path,
        source=source,
        digest=digest,
        repo_root=repo_root,
    )
    pair["endpoint_contract_v6"] = pair.pop("endpoint_contract_v5")
    pair["endpoint_contract_v6"]["frozen_semantic_execution_contract_sha256"] = (
        pair["endpoint_contract_v6"].pop(
            "frozen_semantic_execution_contract_sha256"
        )
    )
    pair["failures"] = [
        (
            "source_endpoint_semantic_contract_differs_from_v6_freeze"
            if value == "source_endpoint_semantic_contract_differs_from_v5_freeze"
            else value
        )
        for value in pair.get("failures") or []
    ]
    on = (source.get("results") or {}).get("epicure_on") or {}
    fanouts = [
        int(output["tool_call_count"])
        for output in on.get("intermediate_outputs") or []
        if isinstance(output, Mapping) and output.get("tool_call_count") is not None
    ]
    if max(fanouts or [0]) > MAX_TOOL_CALLS_PER_ROUND:
        pair["failures"].append("observed_round_fanout_exceeds_v6_cap")
    if len(on.get("tool_trace") or []) > MAX_TOOL_CALLS_TOTAL:
        pair["failures"].append("observed_total_tool_calls_exceeds_v6_cap")
    pair["fanout_audit"] = {
        "observed_tool_calls_by_selection_turn": fanouts,
        "maximum_observed_round_fanout": max(fanouts or [0]),
        "executed_tool_calls_total": len(on.get("tool_trace") or []),
        "frozen_max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
        "frozen_max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
    }
    pair["failures"] = sorted(set(pair["failures"]))
    pair["decision"] = "passed_all_predicates" if not pair["failures"] else "failed"
    return pair


def _v6_accounting(
    *,
    plan: Mapping[str, Any],
    root: Path,
    baseline_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    from .frontier_coverage_repair_executor import SupplementalRun, _run_accounting
    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    if (
        baseline_receipt.get("artifact_sha256") != V5_GEMINI_RECEIPT_SHA
        or baseline_receipt.get("final_budget", {}).get("current_total_exposure_usd")
        != plan.get("budget", {}).get("current_total_exposure_usd")
    ):
        raise RouteGateV6Error("v6 accounting lacks the exact final v5 receipt baseline")
    accounting = _run_accounting(
        SupplementalRun(
            source_directory=root / "source", ledger_path=root / "ledger.jsonl"
        ),
        label="reasoning_effort_route_gate_v6_sonnet",
    )
    baseline = Decimal(
        str(baseline_receipt["final_budget"]["current_total_exposure_usd"])
    )
    current = (
        baseline + accounting.exposure_usd + accounting.orphan_reservation_usd
    )
    reservations, _ = dataset_ledger_state(load_dataset_ledger(root / "ledger.jsonl"))
    outstanding = sum(
        (
            Decimal(str(item["worst_case_reserve_usd"]))
            for item in _work_items(plan)
            if item["work_item_id"] not in reservations
        ),
        Decimal(0),
    )
    projected = current + outstanding
    blockers = [dict(item) for item in accounting.blockers]
    return {
        "currency": "USD",
        "v5_budget_baseline_receipt_sha256": V5_GEMINI_RECEIPT_SHA,
        "v5_total_exposure_usd": v5._decimal_text(baseline),
        "v6_actual_cost_usd": v5._decimal_text(accounting.actual_cost_usd),
        "v6_source_exposure_usd": v5._decimal_text(accounting.exposure_usd),
        "v6_orphan_reservation_usd": v5._decimal_text(
            accounting.orphan_reservation_usd
        ),
        "current_total_exposure_usd": v5._decimal_text(current),
        "outstanding_v6_worst_case_usd": v5._decimal_text(outstanding),
        "projected_total_exposure_usd": v5._decimal_text(projected),
        "admission_ceiling_usd": "85",
        "hard_cap_usd": "100",
        "blockers": blockers,
        "admission_allowed": (
            not blockers and projected <= Decimal("85") and projected <= Decimal("100")
        ),
    }


def build_execution_plan(
    *,
    plan: Mapping[str, Any],
    root: Path,
    baseline_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a zero-network dry run; this function cannot call a provider or MCP."""

    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    budget = _v6_accounting(
        plan=plan, root=root, baseline_receipt=baseline_receipt
    )
    reservations, finalizations = dataset_ledger_state(
        load_dataset_ledger(root / "ledger.jsonl")
    )
    sources = _source_map(root / "source")
    decisions: list[dict[str, Any]] = []
    stopped = False
    for item in _work_items(plan):
        work_item_id = item["work_item_id"]
        if stopped:
            decision = "closed_unattempted_suffix_after_predecessor_failure"
        elif work_item_id in finalizations:
            passed = finalizations[work_item_id].get("route_gate_pair_passed") is True
            decision = "skip_finalized_pass" if passed else "stop_finalized_failure"
            stopped = not passed
        elif work_item_id in reservations and work_item_id in sources:
            decision = "recover_source_without_provider_call"
        elif work_item_id in reservations:
            decision = "closed_reserved_without_source_no_replay"
            stopped = True
        elif budget["admission_allowed"]:
            decision = "fresh_catalog_attestation_then_single_pair_reservation"
        else:
            decision = "blocked_before_catalog_or_provider_call"
            stopped = True
        decisions.append(
            {
                "work_item_id": work_item_id,
                "variant_id": item["route_coordinate"]["variant_id"],
                "worst_case_reserve_usd": item["worst_case_reserve_usd"],
                "decision": decision,
            }
        )
    return {
        "schema_version": EXECUTION_PLAN_SCHEMA,
        "record_role": "zero_call_fresh_sonnet_v6_execution_plan",
        "route_plan_sha256": plan["artifact_sha256"],
        "status": "admissible_dry_run" if budget["admission_allowed"] else "blocked_dry_run",
        "budget": budget,
        "decisions": decisions,
        "execution": {
            "confirmation": CONFIRMATION,
            "catalog_gets_made_by_plan": 0,
            "provider_completion_requests_made_by_plan": 0,
            "epicure_calls_made_by_plan": 0,
            "v4_or_v5_identifier_replay_permitted": False,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def _journal_for_run(source_directory: Path, run_id: str) -> Path | None:
    return v5._journal_for_run(source_directory, run_id)


async def execute(
    *,
    plan: Mapping[str, Any],
    root: Path,
    baseline_receipt: Mapping[str, Any],
    repo_root: Path,
    global_budget_lock_path: Path,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    """Execute the two fresh cells sequentially and close on the first failure."""

    from .config import get_settings
    from .frontier_contract_runner import AdmissionDenied, _exclusive_runner_lock
    from .live_smoke import live_smoke
    from .real_dataset_runner import (
        _dataset_ledger_lock,
        append_dataset_ledger_event,
        dataset_ledger_state,
        load_dataset_ledger,
    )
    from .reasoning_effort_route_gate_v4 import (
        _live_args,
        _policy_environment,
        _require_live_environment_before_reservation,
        _variant_policy,
    )

    validate_route_plan(plan, repo_root=repo_root)
    _require_live_environment_before_reservation()
    source_directory = root / "source"
    ledger_path = root / "ledger.jsonl"
    source_directory.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    attestations: list[dict[str, Any]] = []
    new_invocations = 0
    with _exclusive_runner_lock(global_budget_lock_path):
        with _dataset_ledger_lock(ledger_path):
            for item in _work_items(plan):
                entries = load_dataset_ledger(ledger_path)
                reservations, finalizations = dataset_ledger_state(entries)
                sources = _source_map(source_directory)
                work_item_id = item["work_item_id"]
                if work_item_id in finalizations:
                    passed = finalizations[work_item_id].get("route_gate_pair_passed") is True
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "skip_finalized_pass"
                            if passed
                            else "stop_finalized_failure",
                        }
                    )
                    if not passed:
                        break
                    continue
                if work_item_id in reservations:
                    record = sources.get(work_item_id)
                    if record is None:
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "stop_reserved_without_source_no_replay",
                            }
                        )
                        break
                    path, source, digest = record
                    pair = _pair_audit(
                        plan=plan,
                        item=item,
                        source_path=path,
                        source=source,
                        digest=digest,
                        repo_root=repo_root,
                    )
                    final = append_dataset_ledger_event(
                        ledger_path,
                        {
                            "event_type": "source_artifact_recorded",
                            "runner_run_id": "reasoning-effort-v6-sonnet",
                            "work_item_id": work_item_id,
                            "reservation_entry_sha256": reservations[work_item_id][
                                "entry_sha256"
                            ],
                            "source_artifact_sha256": digest,
                            "source_path": v5._relative(repo_root, path),
                            "route_gate_pair_passed": pair["decision"]
                            == "passed_all_predicates",
                            "pair_audit_sha256": v5._sha256(pair),
                            "actual_cost_usd": pair["accounting"]["actual_cost_usd"],
                            "quality_observations": 0,
                            "rank_eligible": False,
                        },
                    )
                    passed = pair["decision"] == "passed_all_predicates"
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "recovered_source_without_provider_call",
                            "passed": passed,
                            "ledger_entry_sha256": final["entry_sha256"],
                        }
                    )
                    if not passed:
                        break
                    continue
                if work_item_id in sources:
                    raise RouteGateV6Error("source exists without a prior reservation")

                attestation = await v5._one_endpoint_snapshot(
                    api_base=api_base, api_key=api_key, endpoint_id="sonnet"
                )
                attestation["schema_version"] = (
                    "flavourbench-reasoning-effort-endpoint-admission-attestation-v6"
                )
                attestation["record_role"] = (
                    "pre_reservation_zero_generation_sonnet_v6_endpoint_attestation"
                )
                attestation_path = v5._write_artifact(
                    root / "endpoint-attestations",
                    "sonnet-v6-pre-admission-"
                    + item["route_coordinate"]["variant_id"],
                    attestation,
                )
                attestation_document = v5._regular_json(attestation_path)
                attestations.append(
                    {
                        "work_item_id": work_item_id,
                        "path": v5._relative(repo_root, attestation_path),
                        "artifact_sha256": attestation_document["artifact_sha256"],
                    }
                )
                frozen_semantic = item["route_coordinate"][
                    "semantic_execution_contract_sha256"
                ]
                if attestation["semantic_execution_contract_sha256"] != frozen_semantic:
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "stop_semantic_endpoint_drift_before_reservation",
                            "frozen_semantic_sha256": frozen_semantic,
                            "observed_semantic_sha256": attestation[
                                "semantic_execution_contract_sha256"
                            ],
                        }
                    )
                    break
                budget = _v6_accounting(
                    plan=plan, root=root, baseline_receipt=baseline_receipt
                )
                if budget["admission_allowed"] is not True:
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "stop_budget_before_reservation",
                            "budget": budget,
                        }
                    )
                    break
                coordinate = item["route_coordinate"]
                reservation = append_dataset_ledger_event(
                    ledger_path,
                    {
                        "event_type": "reservation_created",
                        "runner_run_id": "reasoning-effort-v6-sonnet",
                        "work_item_id": work_item_id,
                        "route_plan_sha256": plan["artifact_sha256"],
                        "route_cell_id": item["route_cell_id"],
                        "run_id": item["run_id"],
                        "arm_ids": item["arm_ids"],
                        "model_id": coordinate["model_id"],
                        "canonical_model_slug": coordinate["canonical_model_slug"],
                        "provider_endpoint": coordinate["provider_endpoint"],
                        "actual_provider_name": coordinate["actual_provider_name"],
                        "endpoint_snapshot_sha256": coordinate[
                            "endpoint_snapshot_sha256"
                        ],
                        "raw_endpoint_execution_contract_sha256": attestation[
                            "raw_execution_contract_sha256"
                        ],
                        "semantic_endpoint_execution_contract_sha256": frozen_semantic,
                        "endpoint_attestation_sha256": attestation_document[
                            "artifact_sha256"
                        ],
                        "execution_policy_sha256": coordinate[
                            "execution_policy_sha256"
                        ],
                        "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
                        "max_tool_calls_total": MAX_TOOL_CALLS_TOTAL,
                        "variant_id": coordinate["variant_id"],
                        "intermediate_reasoning_effort": coordinate[
                            "intermediate_reasoning_effort"
                        ],
                        "final_reasoning_effort": coordinate[
                            "final_reasoning_effort"
                        ],
                        "conditions": list(CONDITIONS),
                        "reserved_usd": item["worst_case_reserve_usd"],
                        "total_exposure_before_usd": budget[
                            "current_total_exposure_usd"
                        ],
                        "projected_all_remaining_usd": budget[
                            "projected_total_exposure_usd"
                        ],
                        "replay_permitted": False,
                        "quality_observations": 0,
                        "rank_eligible": False,
                    },
                )
                policy = _variant_policy(plan, item, repo_root)
                if (
                    policy.max_tool_calls_per_round != MAX_TOOL_CALLS_PER_ROUND
                    or policy.max_tool_calls_total != MAX_TOOL_CALLS_TOTAL
                ):
                    raise RouteGateV6Error("runtime policy lost the frozen 13/13 caps")
                args = _live_args(
                    route_plan=plan,
                    work_item=item,
                    repo_root=repo_root,
                    source_directory=source_directory,
                )
                args.expected_endpoint_execution_sha256 = attestation[
                    "raw_execution_contract_sha256"
                ]
                try:
                    with _policy_environment(
                        policy=policy,
                        endpoint=attestation["raw_execution_contract"],
                    ):
                        settings = get_settings()
                        if settings.execution_mode != "live" or not settings.live_authorized:
                            raise AdmissionDenied("live authority changed after reservation")
                        new_invocations += 1
                        summary = await live_smoke(args)
                except Exception as error:
                    journal_path = _journal_for_run(source_directory, item["run_id"])
                    descriptor: dict[str, Any] | None = None
                    if journal_path is not None:
                        entries = v5._hash_chain_jsonl(journal_path)
                        descriptor = {
                            "path": v5._relative(repo_root, journal_path),
                            "sha256": v5._file_sha256(journal_path),
                            "entry_count": len(entries),
                            "event_types": [entry["event_type"] for entry in entries],
                            "provider_attempt_events": sum(
                                entry["event_type"] == "provider_attempt"
                                for entry in entries
                            ),
                            "mcp_trace_events": sum(
                                entry["event_type"] == "mcp_trace" for entry in entries
                            ),
                        }
                    incident = append_dataset_ledger_event(
                        ledger_path,
                        {
                            "event_type": "execution_incident",
                            "runner_run_id": "reasoning-effort-v6-sonnet",
                            "work_item_id": work_item_id,
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "endpoint_attestation_sha256": attestation_document[
                                "artifact_sha256"
                            ],
                            "incident": "reservation_retained_v6_closed_no_replay",
                            "error_type": type(error).__name__,
                            "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                            "journal": descriptor,
                            "replay_permitted": False,
                        },
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": "execution_incident_v6_closed_no_replay",
                            "incident_entry_sha256": incident["entry_sha256"],
                        }
                    )
                    break
                artifact_path = Path(str((summary or {}).get("artifact") or ""))
                if (
                    not artifact_path.is_file()
                    or artifact_path.resolve().parent != source_directory.resolve()
                ):
                    raise RouteGateV6Error(
                        "live smoke returned no source inside the v6 directory"
                    )
                path, source, digest = _source_map(source_directory)[work_item_id]
                pair = _pair_audit(
                    plan=plan,
                    item=item,
                    source_path=path,
                    source=source,
                    digest=digest,
                    repo_root=repo_root,
                )
                passed = pair["decision"] == "passed_all_predicates"
                final = append_dataset_ledger_event(
                    ledger_path,
                    {
                        "event_type": "source_artifact_recorded",
                        "runner_run_id": "reasoning-effort-v6-sonnet",
                        "work_item_id": work_item_id,
                        "reservation_entry_sha256": reservation["entry_sha256"],
                        "source_artifact_sha256": digest,
                        "source_path": v5._relative(repo_root, artifact_path),
                        "route_gate_pair_passed": passed,
                        "pair_audit_sha256": v5._sha256(pair),
                        "actual_cost_usd": pair["accounting"]["actual_cost_usd"],
                        "quality_observations": 0,
                        "rank_eligible": False,
                    },
                )
                outcomes.append(
                    {
                        "work_item_id": work_item_id,
                        "decision": "source_finalized_pass"
                        if passed
                        else "source_finalized_failure",
                        "source_artifact_sha256": digest,
                        "ledger_entry_sha256": final["entry_sha256"],
                        "failures": pair["failures"],
                    }
                )
                if not passed:
                    break
    final_budget = _v6_accounting(
        plan=plan, root=root, baseline_receipt=baseline_receipt
    )
    entries = load_dataset_ledger(ledger_path)
    _, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    passed = len(finalizations) == 2 and all(
        entry.get("route_gate_pair_passed") is True
        for entry in finalizations.values()
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "fresh_sonnet_v6_two_pair_execution_receipt",
        "route_plan_sha256": plan["artifact_sha256"],
        "status": "two_pair_sources_available" if passed else "failed_or_incomplete_closed",
        "new_pair_invocations_this_command": new_invocations,
        "total_source_pairs": len(sources),
        "total_finalized_pairs": len(finalizations),
        "source_artifacts": [
            {
                "work_item_id": work_item_id,
                "path": v5._relative(repo_root, path),
                "artifact_sha256": digest,
            }
            for work_item_id, (path, _, digest) in sorted(sources.items())
        ],
        "endpoint_attestations": attestations,
        "ledger": {
            "path": v5._relative(repo_root, ledger_path),
            "sha256": v5._file_sha256(ledger_path)
            if ledger_path.exists()
            else hashlib.sha256(b"").hexdigest(),
            "entry_count": len(entries),
            "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
        "outcomes": outcomes,
        "final_budget": final_budget,
        "uncertain_delivery_replayed": False,
        "v4_or_v5_identifier_replayed": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    receipt_path = v5._write_artifact(
        root / "receipts", "reasoning-effort-v6-sonnet-receipt", receipt
    )
    return {"path": str(receipt_path), "document": v5._regular_json(receipt_path)}


def build_audit(
    *,
    plan: Mapping[str, Any],
    receipt_path: Path,
    root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reconstruct both v6 cells from source, journals, and the append-only ledger."""

    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    receipt = v5._regular_json(receipt_path)
    ledger_path = root / "ledger.jsonl"
    source_directory = root / "source"
    failures: list[str] = []
    if not v5._artifact_verifies(receipt, RECEIPT_SCHEMA):
        failures.append("receipt_content_address_or_schema_failed")
    if (
        receipt.get("route_plan_sha256") != plan.get("artifact_sha256")
        or receipt.get("uncertain_delivery_replayed") is not False
        or receipt.get("v4_or_v5_identifier_replayed") is not False
    ):
        failures.append("receipt_route_or_no_replay_binding_failed")
    entries = load_dataset_ledger(ledger_path)
    reservations, finalizations = dataset_ledger_state(entries)
    sources = _source_map(source_directory)
    known = {item["work_item_id"] for item in _work_items(plan)}
    if (set(reservations) | set(finalizations) | set(sources)) - known:
        failures.append("unknown_work_item_in_v6_evidence")
    pair_audits: list[dict[str, Any]] = []
    incident_audits: list[dict[str, Any]] = []
    all_attempts: set[str] = set()
    all_generations: set[str] = set()
    all_request_keys: set[str] = set()
    for item in _work_items(plan):
        work_item_id = item["work_item_id"]
        record = sources.get(work_item_id)
        finalization = finalizations.get(work_item_id)
        if record is not None:
            path, source, digest = record
            pair = _pair_audit(
                plan=plan,
                item=item,
                source_path=path,
                source=source,
                digest=digest,
                repo_root=repo_root,
            )
            pair_audits.append(pair)
            if (
                finalization is None
                or finalization.get("source_artifact_sha256") != digest
                or finalization.get("route_gate_pair_passed")
                is not (pair["decision"] == "passed_all_predicates")
            ):
                failures.append(f"ledger_finalization_mismatch:{work_item_id}")
            for name, target in (
                ("attempt_ids", all_attempts),
                ("generation_ids", all_generations),
                ("request_key_sha256s", all_request_keys),
            ):
                observed = set(pair["identifiers"][name])
                if target & observed:
                    failures.append(f"cross_pair_{name}_overlap")
                target.update(observed)
            if pair["decision"] != "passed_all_predicates":
                failures.append(f"pair_predicate_failure:{work_item_id}")
            continue
        incidents = [
            entry
            for entry in entries
            if entry.get("event_type") == "execution_incident"
            and entry.get("work_item_id") == work_item_id
        ]
        if incidents:
            incident = incidents[-1]
            journal = incident.get("journal")
            reconstructed = False
            provider_attempts: int | None = None
            mcp_traces: int | None = None
            if isinstance(journal, Mapping):
                journal_path = repo_root / str(journal.get("path") or "")
                try:
                    journal_entries = v5._hash_chain_jsonl(journal_path)
                    reconstructed = (
                        v5._file_sha256(journal_path) == journal.get("sha256")
                        and len(journal_entries) == journal.get("entry_count")
                        and [entry["event_type"] for entry in journal_entries]
                        == journal.get("event_types")
                    )
                    provider_attempts = sum(
                        entry["event_type"] == "provider_attempt"
                        for entry in journal_entries
                    )
                    mcp_traces = sum(
                        entry["event_type"] == "mcp_trace"
                        for entry in journal_entries
                    )
                except (OSError, ValueError, TypeError, v5.RouteGateV5Error):
                    reconstructed = False
            incident_audits.append(
                {
                    "work_item_id": work_item_id,
                    "incident_entry_sha256": incident.get("entry_sha256"),
                    "journal_reconstructed": reconstructed,
                    "provider_attempt_events": provider_attempts,
                    "mcp_trace_events": mcp_traces,
                    "pre_generation_failure": reconstructed and provider_attempts == 0,
                    "reservation_retained": work_item_id in reservations,
                    "replay_permitted": False,
                }
            )
        failures.append(f"missing_complete_source:{work_item_id}")
    if len(pair_audits) != 2:
        failures.append("both_v6_sonnet_pairs_required")
    planned_attempts = {
        slot["attempt_id"]
        for item in _work_items(plan)
        for slot in item["attempt_slots"]
    }
    if not all_attempts <= planned_attempts:
        failures.append("attempt_outside_frozen_v6_pool")
    unique = sorted(set(failures))
    passed = not unique
    actual_micros = sum(
        int(pair["accounting"]["actual_cost_micros"]) for pair in pair_audits
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "record_role": "source_reconstructed_fresh_sonnet_v6_route_audit",
        "route_plan_sha256": plan["artifact_sha256"],
        "receipt": {
            "path": v5._relative(repo_root, receipt_path),
            "artifact_sha256": receipt.get("artifact_sha256"),
        },
        "ledger": {
            "path": v5._relative(repo_root, ledger_path),
            "sha256": v5._file_sha256(ledger_path)
            if ledger_path.exists()
            else hashlib.sha256(b"").hexdigest(),
            "entry_count": len(entries),
            "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique,
        "pair_audits": pair_audits,
        "incident_audits": incident_audits,
        "identifier_audit": {
            "planned_attempt_ids": sorted(planned_attempts),
            "observed_attempt_ids": sorted(all_attempts),
            "observed_generation_ids": sorted(all_generations),
            "observed_request_key_sha256s": sorted(all_request_keys),
        },
        "counts": {
            "required_pairs": 2,
            "source_verified_pairs": len(pair_audits),
            "usable_pairs": 2 if passed else 0,
            "usable_arms": 4 if passed else 0,
            "synthetic_arms": 0,
            "quality_observations": 0,
        },
        "accounting": {
            "actual_cost_micros": actual_micros,
            "actual_cost_usd": v5._decimal_text(
                Decimal(actual_micros) / Decimal(1_000_000)
            ),
            "all_source_generation_costs_reconciled": all(
                pair["accounting"]["reconciled"] is True for pair in pair_audits
            ),
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_closure(
    *, plan: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not v5._artifact_verifies(audit, AUDIT_SCHEMA):
        raise RouteGateV6Error("v6 audit does not verify")
    identifiers = audit.get("identifier_audit") or {}
    items = _work_items(plan)
    return {
        "schema_version": CLOSURE_SCHEMA,
        "record_role": "permanent_fresh_sonnet_v6_route_closure",
        "route_plan_sha256": plan["artifact_sha256"],
        "audit_sha256": audit["artifact_sha256"],
        "closed_identifiers": {
            "route_cell_ids": sorted(item["route_cell_id"] for item in items),
            "work_item_ids": sorted(item["work_item_id"] for item in items),
            "run_ids": sorted(item["run_id"] for item in items),
            "arm_ids": sorted(value for item in items for value in item["arm_ids"]),
            "attempt_ids": sorted(identifiers.get("planned_attempt_ids") or []),
            "used_attempt_ids": sorted(identifiers.get("observed_attempt_ids") or []),
            "generation_ids": sorted(identifiers.get("observed_generation_ids") or []),
            "request_key_sha256s": sorted(
                identifiers.get("observed_request_key_sha256s") or []
            ),
            "replay_permitted": False,
        },
        "decision": {
            "sonnet_v6_qualified": audit.get("decision") == "passed_all_predicates",
            "identifiers_permanently_closed": True,
            "route_gate_diagnostic_only": True,
            "full_study_execution_authorized": False,
        },
        "cost": audit.get("accounting"),
        "claim_boundary": audit.get("claim_boundary"),
    }


def build_aggregate_audit(
    *,
    plan: Mapping[str, Any],
    bridge_audit_path: Path,
    sonnet_audit_path: Path,
    sonnet_closure_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    bridge = v5._regular_json(bridge_audit_path)
    sonnet = v5._regular_json(sonnet_audit_path)
    closure = v5._regular_json(sonnet_closure_path)
    failures: list[str] = []
    if not v5._artifact_verifies(bridge, BRIDGE_AUDIT_SCHEMA):
        failures.append("fanout_bridge_audit_does_not_verify")
    if not v5._artifact_verifies(sonnet, AUDIT_SCHEMA):
        failures.append("sonnet_v6_audit_does_not_verify")
    if not v5._artifact_verifies(closure, CLOSURE_SCHEMA):
        failures.append("sonnet_v6_closure_does_not_verify")
    if (
        bridge.get("decision")
        != "four_prior_cells_observationally_invariant_under_v6_caps"
    ):
        failures.append("prior_route_cells_not_bridgeable")
    if (
        sonnet.get("route_plan_sha256") != plan.get("artifact_sha256")
        or sonnet.get("decision") != "passed_all_predicates"
        or closure.get("audit_sha256") != sonnet.get("artifact_sha256")
        or closure.get("closed_identifiers", {}).get("replay_permitted") is not False
    ):
        failures.append("sonnet_v6_did_not_pass_and_close")
    unique = sorted(set(failures))
    passed = not unique
    return {
        "schema_version": AGGREGATE_AUDIT_SCHEMA,
        "record_role": "six_cell_endpoint_specific_route_compatibility_audit_v6",
        "route_plan_sha256": plan["artifact_sha256"],
        "inputs": {
            "fanout_bridge_audit": {
                "path": v5._relative(repo_root, bridge_audit_path),
                "artifact_sha256": bridge.get("artifact_sha256"),
            },
            "sonnet_v6_audit": {
                "path": v5._relative(repo_root, sonnet_audit_path),
                "artifact_sha256": sonnet.get("artifact_sha256"),
            },
            "sonnet_v6_closure": {
                "path": v5._relative(repo_root, sonnet_closure_path),
                "artifact_sha256": closure.get("artifact_sha256"),
            },
        },
        "decision": "passed_all_predicates" if passed else "failed_one_or_more_predicates",
        "failures": unique,
        "counts": {
            "preserved_deepseek_route_cells": 2,
            "preserved_gemini_route_cells": 2,
            "fresh_sonnet_route_cells": 2 if passed else 0,
            "route_compatibility_cells": 6 if passed else 0,
            "quality_observations": 0,
        },
        "endpoint_specific_sensitivity": {
            "estimable_from_route_gate_outputs": False,
            "reason": "one diagnostic task has no blinded quality judgments",
            "future_reporting": (
                "report model-specific paired effects with separate intervals; do not "
                "pool disconnected endpoint graphs"
            ),
            "gemini_and_deepseek_may_support_future_endpoint_specific_effects": True,
            "aggregate_cross_model_effect_supported_now": False,
        },
        "study_admission": {
            "fresh_zero_call_confirmatory_preflight_permitted": passed,
            "full_study_execution_performed": False,
            "separate_authorization_and_freeze_required": True,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def build_aggregate_closure(
    *, plan: Mapping[str, Any], aggregate_audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not v5._artifact_verifies(aggregate_audit, AGGREGATE_AUDIT_SCHEMA):
        raise RouteGateV6Error("v6 aggregate audit does not verify")
    return {
        "schema_version": AGGREGATE_CLOSURE_SCHEMA,
        "record_role": "permanent_reasoning_effort_route_gate_v6_aggregate_closure",
        "route_plan_sha256": plan["artifact_sha256"],
        "aggregate_audit_sha256": aggregate_audit["artifact_sha256"],
        "decision": {
            "route_gate_qualified": aggregate_audit.get("decision")
            == "passed_all_predicates",
            "route_gate_only": True,
            "quality_sensitivity_result_available": False,
            "full_study_execution_performed": False,
            "replay_permitted": False,
        },
        "claim_boundary": aggregate_audit.get("claim_boundary"),
    }


def _default_paths(repo_root: Path) -> dict[str, Path]:
    project = repo_root / "flavourbench"
    current = project / "artifacts/season1/current-quality-run"
    v4_root = current / "reasoning-effort-sensitivity-v4"
    v5_root = current / "reasoning-effort-sensitivity-v5"
    v6_root = current / "reasoning-effort-sensitivity-v6"
    return {
        "project": project,
        "v6_root": v6_root,
        "v4_route": v4_root
        / (
            "reasoning-effort-v4-route-gate-plan-"
            "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json"
        ),
        "v4_closure": v4_root
        / "route-gate/reasoning-effort-v4-route-gate-closure-"
        "807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87.json",
        "v4_source": v4_root / "route-gate/source",
        "v5_plan": v5_root
        / "route-gate/reasoning-effort-v5-route-gate-plan-"
        f"{V5_PLAN_SHA}.json",
        "snapshot": v5_root
        / "endpoint-snapshot/reasoning-effort-v5-endpoint-snapshot-"
        f"{V5_ENDPOINT_SNAPSHOT_SHA}.json",
        "sonnet_receipt": v5_root
        / "sonnet/receipts/reasoning-effort-v5-sonnet-receipt-"
        f"{V5_SONNET_RECEIPT_SHA}.json",
        "sonnet_audit": v5_root
        / "sonnet/audits/reasoning-effort-v5-sonnet-audit-"
        f"{V5_SONNET_AUDIT_SHA}.json",
        "sonnet_closure": v5_root
        / "sonnet/closures/reasoning-effort-v5-sonnet-closure-"
        f"{V5_SONNET_CLOSURE_SHA}.json",
        "sonnet_ledger": v5_root / "sonnet/ledger.jsonl",
        "sonnet_source": v5_root
        / "sonnet/source"
        / f"20260803T194558Z-{V5_SONNET_SOURCE_SHA[:12]}.json",
        "sonnet_journal": v5_root
        / "sonnet/source/flavourbench-live-smoke-journal-"
        f"{V5_SONNET_JOURNAL_SHA}.jsonl",
        "gemini_receipt": v5_root
        / "gemini/receipts/reasoning-effort-v5-gemini-receipt-"
        f"{V5_GEMINI_RECEIPT_SHA}.json",
        "gemini_audit": v5_root
        / "gemini/audits/reasoning-effort-v5-gemini-audit-"
        f"{V5_GEMINI_AUDIT_SHA}.json",
        "gemini_closure": v5_root
        / "gemini/closures/reasoning-effort-v5-gemini-closure-"
        f"{V5_GEMINI_CLOSURE_SHA}.json",
        "gemini_source": v5_root / "gemini/source",
        "aggregate_audit": v5_root
        / "aggregate/reasoning-effort-v5-aggregate-audit-"
        f"{V5_AGGREGATE_AUDIT_SHA}.json",
        "aggregate_closure": v5_root
        / "aggregate/reasoning-effort-v5-aggregate-closure-"
        f"{V5_AGGREGATE_CLOSURE_SHA}.json",
        "base_manifest": current
        / "manifest-v29-high-resource/flavourbench-routed-unranked-"
        "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json",
        "global_budget_lock": project / "artifacts/frontier-contract/ledger.jsonl",
    }


def prepare(*, repo_root: Path) -> dict[str, Any]:
    """Materialize the complete zero-call v6 recovery packet."""

    paths = _default_paths(repo_root)
    root = paths["v6_root"]
    failure_payload = build_failure_audit(
        v5_plan_path=paths["v5_plan"],
        receipt_path=paths["sonnet_receipt"],
        audit_path=paths["sonnet_audit"],
        closure_path=paths["sonnet_closure"],
        aggregate_audit_path=paths["aggregate_audit"],
        aggregate_closure_path=paths["aggregate_closure"],
        ledger_path=paths["sonnet_ledger"],
        source_path=paths["sonnet_source"],
        journal_path=paths["sonnet_journal"],
        repo_root=repo_root,
    )
    failure_path = v5._write_artifact(
        root / "failure-audit", "reasoning-effort-v5-sonnet-failure-audit", failure_payload
    )
    bridge_payload = build_bridge_audit(
        v4_route_plan_path=paths["v4_route"],
        v4_closure_path=paths["v4_closure"],
        v4_source_directory=paths["v4_source"],
        v5_plan_path=paths["v5_plan"],
        gemini_audit_path=paths["gemini_audit"],
        gemini_closure_path=paths["gemini_closure"],
        gemini_source_directory=paths["gemini_source"],
        repo_root=repo_root,
    )
    bridge_path = v5._write_artifact(
        root / "bridge", "reasoning-effort-v6-fanout-bridge-audit", bridge_payload
    )
    failure = v5._regular_json(failure_path)
    bridge = v5._regular_json(bridge_path)
    manifest = build_v6_manifest(
        base_manifest_path=paths["base_manifest"],
        failure_audit=failure,
        bridge_audit=bridge,
    )
    manifest_path = _write_manifest(root / "manifest", manifest)
    route_payload = build_route_plan(
        v5_plan_path=paths["v5_plan"],
        v4_closure_path=paths["v4_closure"],
        v5_sonnet_closure_path=paths["sonnet_closure"],
        v5_gemini_closure_path=paths["gemini_closure"],
        v5_aggregate_closure_path=paths["aggregate_closure"],
        endpoint_snapshot_path=paths["snapshot"],
        failure_audit_path=failure_path,
        bridge_audit_path=bridge_path,
        v6_manifest_path=manifest_path,
        v5_gemini_receipt_path=paths["gemini_receipt"],
        repo_root=repo_root,
    )
    route_path = v5._write_artifact(
        root / "route-gate", "reasoning-effort-v6-sonnet-route-gate-plan", route_payload
    )
    route = v5._regular_json(route_path)
    validate_route_plan(route, repo_root=repo_root)
    baseline = _artifact(
        paths["gemini_receipt"], V5_GEMINI_RECEIPT_SHA, v5.ENDPOINT_RECEIPT_SCHEMA
    )
    execution_payload = build_execution_plan(
        plan=route, root=root / "sonnet", baseline_receipt=baseline
    )
    execution_path = v5._write_artifact(
        root / "route-gate",
        "reasoning-effort-v6-sonnet-execution-plan",
        execution_payload,
    )
    return {
        "status": "frozen_zero_call_recovery_packet",
        "provider_completion_requests": 0,
        "epicure_calls": 0,
        "failure_audit": str(failure_path.resolve()),
        "fanout_bridge_audit": str(bridge_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "route_plan": str(route_path.resolve()),
        "execution_plan": str(execution_path.resolve()),
        "confirmation": CONFIRMATION,
    }


def _parser() -> argparse.ArgumentParser:
    default_repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--route-plan", type=Path, required=True)
    execute_parser.add_argument("--root", type=Path)
    execute_parser.add_argument("--baseline-receipt", type=Path)
    execute_parser.add_argument("--global-budget-lock", type=Path)
    execute_parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    execute_parser.add_argument("--confirm", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--route-plan", type=Path, required=True)
    audit_parser.add_argument("--receipt", type=Path, required=True)
    audit_parser.add_argument("--root", type=Path)

    close_parser = subparsers.add_parser("close")
    close_parser.add_argument("--route-plan", type=Path, required=True)
    close_parser.add_argument("--audit", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate-audit")
    aggregate_parser.add_argument("--route-plan", type=Path, required=True)
    aggregate_parser.add_argument("--bridge-audit", type=Path, required=True)
    aggregate_parser.add_argument("--sonnet-audit", type=Path, required=True)
    aggregate_parser.add_argument("--sonnet-closure", type=Path, required=True)

    aggregate_close = subparsers.add_parser("aggregate-close")
    aggregate_close.add_argument("--route-plan", type=Path, required=True)
    aggregate_close.add_argument("--aggregate-audit", type=Path, required=True)
    return parser


def _print(document: Mapping[str, Any]) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    defaults = _default_paths(repo_root)
    if arguments.command == "prepare":
        _print(prepare(repo_root=repo_root))
        return
    route = v5._regular_json(arguments.route_plan)
    validate_route_plan(route, repo_root=repo_root)
    root = (
        getattr(arguments, "root", None) or defaults["v6_root"] / "sonnet"
    ).resolve()
    if arguments.command == "execute":
        if arguments.confirm != CONFIRMATION:
            raise RouteGateV6Error(f"execution requires --confirm {CONFIRMATION}")
        baseline_path = arguments.baseline_receipt or defaults["gemini_receipt"]
        baseline = _artifact(
            baseline_path, V5_GEMINI_RECEIPT_SHA, v5.ENDPOINT_RECEIPT_SCHEMA
        )
        result = asyncio.run(
            execute(
                plan=route,
                root=root,
                baseline_receipt=baseline,
                repo_root=repo_root,
                global_budget_lock_path=(
                    arguments.global_budget_lock or defaults["global_budget_lock"]
                ),
                api_base=arguments.api_base,
                api_key=v5._api_key(),
            )
        )
        _print(
            {
                "output": str(Path(result["path"]).resolve()),
                "artifact_sha256": result["document"]["artifact_sha256"],
                "status": result["document"]["status"],
            }
        )
        return
    if arguments.command == "audit":
        payload = build_audit(
            plan=route,
            receipt_path=arguments.receipt,
            root=root,
            repo_root=repo_root,
        )
        path = v5._write_artifact(
            defaults["v6_root"] / "sonnet/audits",
            "reasoning-effort-v6-sonnet-audit",
            payload,
        )
        _print({"output": str(path.resolve()), **v5._regular_json(path)})
        return
    if arguments.command == "close":
        audit = v5._regular_json(arguments.audit)
        payload = build_closure(plan=route, audit=audit)
        path = v5._write_artifact(
            defaults["v6_root"] / "sonnet/closures",
            "reasoning-effort-v6-sonnet-closure",
            payload,
        )
        _print({"output": str(path.resolve()), **v5._regular_json(path)})
        return
    if arguments.command == "aggregate-audit":
        payload = build_aggregate_audit(
            plan=route,
            bridge_audit_path=arguments.bridge_audit,
            sonnet_audit_path=arguments.sonnet_audit,
            sonnet_closure_path=arguments.sonnet_closure,
            repo_root=repo_root,
        )
        path = v5._write_artifact(
            defaults["v6_root"] / "aggregate",
            "reasoning-effort-v6-aggregate-audit",
            payload,
        )
        _print({"output": str(path.resolve()), **v5._regular_json(path)})
        return
    if arguments.command == "aggregate-close":
        audit = v5._regular_json(arguments.aggregate_audit)
        payload = build_aggregate_closure(plan=route, aggregate_audit=audit)
        path = v5._write_artifact(
            defaults["v6_root"] / "aggregate",
            "reasoning-effort-v6-aggregate-closure",
            payload,
        )
        _print({"output": str(path.resolve()), **v5._regular_json(path)})


if __name__ == "__main__":
    run()
