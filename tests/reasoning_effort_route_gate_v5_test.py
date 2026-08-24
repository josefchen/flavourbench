from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

import flavourbench.reasoning_effort_route_gate_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CURRENT = ROOT / "artifacts/season1/current-quality-run"
V4 = CURRENT / "reasoning-effort-sensitivity-v4"
V5 = CURRENT / "reasoning-effort-sensitivity-v5"
V4_ROUTE = V4 / (
    "reasoning-effort-v4-route-gate-plan-"
    "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json"
)
V4_RECEIPT = V4 / "route-gate" / (
    "reasoning-effort-v4-route-gate-execution-receipt-"
    "172f4a08003656371de69c0907975f83761597338b159031b16052417d575852.json"
)
V4_AUDIT = V4 / "route-gate" / (
    "reasoning-effort-v4-route-gate-audit-"
    "c90617d7b6a8cab918bf0f50f7190f8ad8f49badb5ce036c7c9fa716d7d9a959.json"
)
V4_CLOSURE = V4 / "route-gate" / (
    "reasoning-effort-v4-route-gate-closure-"
    "807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87.json"
)
V4_LEDGER = V4 / "route-gate/ledger.jsonl"
V4_SOURCE = V4 / "route-gate/source"
V4_JOURNAL = V4_SOURCE / (
    ".flavourbench-live-smoke-journal-"
    "19125098-99b0-58af-b87b-a6260a9c5bd3.inprogress.jsonl"
)
SNAPSHOT = V5 / "endpoint-snapshot" / (
    "reasoning-effort-v5-endpoint-snapshot-"
    "ce46706dd7c2cb0605c3dd5abc34f36714f09a6074e155b18298393f14a38262.json"
)
INCIDENT = V5 / "v4-incident" / (
    "reasoning-effort-v4-pre-request-audit-"
    "691bc2b19a36b49984907f15fe0890577b4b25aa7251b6edf6ac1d6960694915.json"
)
FROZEN_PLAN = V5 / "route-gate" / (
    "reasoning-effort-v5-route-gate-plan-"
    "0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c.json"
)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    value = json.loads(SNAPSHOT.read_text())
    assert v5._artifact_verifies(value, v5.SNAPSHOT_SCHEMA)
    return value


@pytest.fixture(scope="module")
def plan() -> dict:
    value = v5.build_route_plan(
        v4_route_plan_path=V4_ROUTE,
        v4_receipt_path=V4_RECEIPT,
        v4_audit_path=V4_AUDIT,
        v4_closure_path=V4_CLOSURE,
        v4_incident_audit_path=INCIDENT,
        endpoint_snapshot_path=SNAPSHOT,
        v4_source_directory=V4_SOURCE,
        repo_root=REPO_ROOT,
    )
    return {**value, "artifact_sha256": v5._sha256(value)}


def test_raw_drift_is_real_but_below_frozen_semantic_quantum(snapshot: dict) -> None:
    manifest = json.loads(
        (
            CURRENT
            / "manifest-v29-high-resource"
            / "flavourbench-routed-unranked-"
            "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json"
        ).read_text()
    )
    frozen = v5.raw_endpoint_contract(
        next(
            item["endpoint"]
            for item in manifest["models"]
            if item["model"]["id"] == "google/gemini-3.6-flash"
        )
    )
    observed = next(
        item["raw_execution_contract"]
        for item in snapshot["records"]
        if item["endpoint_id"] == "gemini"
    )
    old = Decimal(frozen["pricing"]["input_cache_write"])
    new = Decimal(observed["pricing"]["input_cache_write"])
    assert old != new
    assert abs(old - new) * Decimal(1_000_000) == Decimal("3E-17")
    assert v5._sha256(frozen) == v5.V4_GEMINI_RAW_SHA256
    assert v5._sha256(observed) == (
        "a6eb94ab6d58f19a73175150f2f8aa4ef618b731098a744848b50459e1292b22"
    )
    assert v5.semantic_endpoint_contract(frozen) == v5.semantic_endpoint_contract(observed)


def test_material_price_or_capability_change_fails_semantic_contract(snapshot: dict) -> None:
    observed = copy.deepcopy(
        next(
            item["raw_execution_contract"]
            for item in snapshot["records"]
            if item["endpoint_id"] == "gemini"
        )
    )
    baseline = v5.semantic_endpoint_contract(observed)
    observed["pricing"]["completion"] = "0.00000376"
    assert v5.semantic_endpoint_contract(observed) != baseline
    observed["pricing"]["completion"] = "0.00000375"
    observed["supported_parameters"].remove("reasoning_effort")
    assert v5.semantic_endpoint_contract(observed) != baseline


def test_v4_incident_reconstructs_zero_generation_boundary() -> None:
    audit = v5.build_v4_incident_audit(
        v4_route_plan_path=V4_ROUTE,
        v4_receipt_path=V4_RECEIPT,
        v4_audit_path=V4_AUDIT,
        v4_closure_path=V4_CLOSURE,
        v4_ledger_path=V4_LEDGER,
        v4_journal_path=V4_JOURNAL,
        v4_source_directory=V4_SOURCE,
        endpoint_snapshot_path=SNAPSHOT,
        repo_root=REPO_ROOT,
    )
    assert audit["incident"]["exception_sha256"] == v5.V4_ERROR_SHA256
    assert audit["request_boundary"]["provider_completion_requests"] == 0
    assert audit["request_boundary"]["mcp_tool_calls"] == 0
    assert audit["accounting"]["reserved_usd_retained_as_conservative_exposure"] == (
        "0.6765315"
    )
    assert audit["closure"]["old_identifiers_replay_permitted"] is False


def test_v5_preserves_two_deepseek_sources_and_freezes_four_fresh_pairs(plan: dict) -> None:
    assert plan["counts"] == {
        "intended_pairs": 6,
        "accepted_prior_pairs": 2,
        "fresh_pairs": 4,
        "fresh_response_arms": 8,
        "models": 3,
        "effort_variants": 2,
        "synthetic_arms": 0,
        "quality_observations": 0,
    }
    assert all(item["accepted_without_replay"] for item in plan["accepted_prior_pairs"])
    assert plan["budget"]["new_four_pair_worst_case_usd"] == "3.650511"
    assert {item["endpoint_id"] for item in plan["work_items"]} == {"gemini", "sonnet"}


def test_frozen_plan_reopens_every_bound_source() -> None:
    frozen = json.loads(FROZEN_PLAN.read_text())
    v5.validate_route_plan(frozen, repo_root=REPO_ROOT)
    assert frozen["artifact_sha256"] == (
        "0481ecd9c8260967275e18a72d4ed265352d35ca2254f554ba55053bc61bb71c"
    )


def test_fresh_ids_are_disjoint_from_every_closed_v4_id(plan: dict) -> None:
    closure = json.loads(V4_CLOSURE.read_text())
    prior = v5._closed_v4_identifiers(closure)
    observed = {
        "route_cell_ids": {item["route_cell_id"] for item in plan["work_items"]},
        "work_item_ids": {item["work_item_id"] for item in plan["work_items"]},
        "run_ids": {item["run_id"] for item in plan["work_items"]},
        "arm_ids": {value for item in plan["work_items"] for value in item["arm_ids"]},
        "attempt_ids": {
            slot["attempt_id"]
            for item in plan["work_items"]
            for slot in item["attempt_slots"]
        },
    }
    for key, values in observed.items():
        assert not values & prior[key]


def test_endpoint_plans_are_independent_and_budget_is_conservative(
    plan: dict, tmp_path: Path
) -> None:
    receipt = json.loads(V4_RECEIPT.read_text())
    gemini = v5.build_endpoint_execution_plan(
        plan=plan,
        endpoint_id="gemini",
        endpoint_root=tmp_path / "gemini",
        peer_endpoint_root=tmp_path / "sonnet",
        v4_receipt=receipt,
    )
    sonnet = v5.build_endpoint_execution_plan(
        plan=plan,
        endpoint_id="sonnet",
        endpoint_root=tmp_path / "sonnet",
        peer_endpoint_root=tmp_path / "gemini",
        v4_receipt=receipt,
    )
    assert gemini["status"] == sonnet["status"] == "admissible_dry_run"
    assert gemini["budget"]["current_total_exposure_usd"] == (
        "47.19792482666666666666666666"
    )
    assert gemini["budget"]["projected_total_exposure_usd"] == (
        "50.84843582666666666666666666"
    )
    assert gemini["execution"]["provider_completion_requests_made_by_plan"] == 0
    assert sonnet["execution"]["provider_completion_requests_made_by_plan"] == 0
    assert gemini["execution"]["confirmation"] != sonnet["execution"]["confirmation"]


def test_endpoint_closure_never_closes_peer_identifiers(plan: dict) -> None:
    gemini_items = v5._endpoint_items(plan, "gemini")
    sonnet_items = v5._endpoint_items(plan, "sonnet")
    audit_payload = {
        "schema_version": v5.ENDPOINT_AUDIT_SCHEMA,
        "record_role": "source_reconstructed_endpoint_isolated_reasoning_effort_audit",
        "route_plan_sha256": plan["artifact_sha256"],
        "endpoint_id": "gemini",
        "decision": "failed_one_or_more_predicates",
        "identifier_audit": {
            "planned_attempt_ids": [
                slot["attempt_id"] for item in gemini_items for slot in item["attempt_slots"]
            ],
            "observed_attempt_ids": [],
            "observed_generation_ids": [],
            "observed_request_key_sha256s": [],
        },
        "incident_audits": [],
        "accounting": {},
        "claim_boundary": plan["claim_boundary"],
    }
    audit = {**audit_payload, "artifact_sha256": v5._sha256(audit_payload)}
    closure = v5.build_endpoint_closure(plan=plan, endpoint_id="gemini", audit=audit)
    closed = set(closure["closed_identifiers"]["work_item_ids"])
    assert closed == {item["work_item_id"] for item in gemini_items}
    assert not closed & {item["work_item_id"] for item in sonnet_items}
    assert closure["decision"]["other_endpoint_execution_blocked"] is False
