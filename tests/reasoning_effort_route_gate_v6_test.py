from __future__ import annotations

import json
from pathlib import Path

import flavourbench.reasoning_effort_route_gate_v5 as v5
import flavourbench.reasoning_effort_route_gate_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CURRENT = ROOT / "artifacts/season1/current-quality-run"
V4 = CURRENT / "reasoning-effort-sensitivity-v4"
V5 = CURRENT / "reasoning-effort-sensitivity-v5"
V6 = CURRENT / "reasoning-effort-sensitivity-v6"

V4_ROUTE = V4 / (
    "reasoning-effort-v4-route-gate-plan-"
    "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json"
)
V4_CLOSURE = V4 / "route-gate" / (
    "reasoning-effort-v4-route-gate-closure-"
    "807aa054e7f0aaaa770630adae7696bba8fc24251d7ed2b08082b46a0edfde87.json"
)
V4_SOURCE = V4 / "route-gate/source"
V5_PLAN = V5 / "route-gate" / (
    "reasoning-effort-v5-route-gate-plan-"
    f"{v6.V5_PLAN_SHA}.json"
)
SNAPSHOT = V5 / "endpoint-snapshot" / (
    "reasoning-effort-v5-endpoint-snapshot-"
    f"{v6.V5_ENDPOINT_SNAPSHOT_SHA}.json"
)
SONNET_RECEIPT = V5 / "sonnet/receipts" / (
    "reasoning-effort-v5-sonnet-receipt-"
    f"{v6.V5_SONNET_RECEIPT_SHA}.json"
)
SONNET_AUDIT = V5 / "sonnet/audits" / (
    "reasoning-effort-v5-sonnet-audit-"
    f"{v6.V5_SONNET_AUDIT_SHA}.json"
)
SONNET_CLOSURE = V5 / "sonnet/closures" / (
    "reasoning-effort-v5-sonnet-closure-"
    f"{v6.V5_SONNET_CLOSURE_SHA}.json"
)
SONNET_SOURCE = V5 / "sonnet/source" / (
    f"20260803T194558Z-{v6.V5_SONNET_SOURCE_SHA[:12]}.json"
)
SONNET_JOURNAL = V5 / "sonnet/source" / (
    "flavourbench-live-smoke-journal-"
    f"{v6.V5_SONNET_JOURNAL_SHA}.jsonl"
)
GEMINI_RECEIPT = V5 / "gemini/receipts" / (
    "reasoning-effort-v5-gemini-receipt-"
    f"{v6.V5_GEMINI_RECEIPT_SHA}.json"
)
GEMINI_AUDIT = V5 / "gemini/audits" / (
    "reasoning-effort-v5-gemini-audit-"
    f"{v6.V5_GEMINI_AUDIT_SHA}.json"
)
GEMINI_CLOSURE = V5 / "gemini/closures" / (
    "reasoning-effort-v5-gemini-closure-"
    f"{v6.V5_GEMINI_CLOSURE_SHA}.json"
)
AGGREGATE_AUDIT = V5 / "aggregate" / (
    "reasoning-effort-v5-aggregate-audit-"
    f"{v6.V5_AGGREGATE_AUDIT_SHA}.json"
)
AGGREGATE_CLOSURE = V5 / "aggregate" / (
    "reasoning-effort-v5-aggregate-closure-"
    f"{v6.V5_AGGREGATE_CLOSURE_SHA}.json"
)
FAILURE = V6 / "failure-audit" / (
    "reasoning-effort-v5-sonnet-failure-audit-"
    "f308f5b5fc57ce6f1d9c52b0e0f21f653843b96fd3ae295c21fe6ba7a3320c34.json"
)
BRIDGE = V6 / "bridge" / (
    "reasoning-effort-v6-fanout-bridge-audit-"
    "9d389ac19fff5a57d801c3ee076f38276793413c260b47cf135601ab441a81f4.json"
)
MANIFEST = V6 / "manifest" / (
    "flavourbench-reasoning-effort-v6-"
    "052a214a4d1358ca80aba3612949c3aa1177907924a0f0c62df588424301eef1.json"
)
PLAN = V6 / "route-gate" / (
    "reasoning-effort-v6-sonnet-route-gate-plan-"
    "905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c.json"
)
PREFLIGHT = V6 / "route-gate" / (
    "reasoning-effort-v6-sonnet-execution-plan-"
    "4091db95f115d79aa454821aa0700941284dd1a8795e787c5db7d2a405121d54.json"
)
BASE_MANIFEST = CURRENT / "manifest-v29-high-resource" / (
    "flavourbench-routed-unranked-"
    "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_v5_failure_reconstructs_provider_client_and_mcp_boundary() -> None:
    audit = v6.build_failure_audit(
        v5_plan_path=V5_PLAN,
        receipt_path=SONNET_RECEIPT,
        audit_path=SONNET_AUDIT,
        closure_path=SONNET_CLOSURE,
        aggregate_audit_path=AGGREGATE_AUDIT,
        aggregate_closure_path=AGGREGATE_CLOSURE,
        ledger_path=V5 / "sonnet/ledger.jsonl",
        source_path=SONNET_SOURCE,
        journal_path=SONNET_JOURNAL,
        repo_root=REPO_ROOT,
    )
    root = audit["root_cause"]
    assert root["exception_sha256"] == v6.V5_ERROR_SHA
    assert root["returned_tool_calls"] == 9
    assert root["frozen_per_round_cap"] == 6
    assert root["client_protocol_limit_triggered"] is True
    assert root["provider_or_endpoint_failure"] is False
    assert root["epicure_service_failure"] is False
    boundary = audit["request_and_tool_boundary"]
    assert boundary["provider_requests_started"] == 5
    assert boundary["accepted_chat_completions"] == 5
    assert boundary["generation_cost_records_reconciled"] == 5
    assert boundary["mcp_calls_started"] == boundary["mcp_calls_completed"] == 0


def test_complete_source_cost_exposes_the_v5_ledger_understatement() -> None:
    audit = _json(FAILURE)
    assert v5._artifact_verifies(audit, v6.FAILURE_AUDIT_SCHEMA)
    assert audit["accounting"] == {
        "complete_generation_cost_micros": 61742,
        "complete_generation_cost_usd": "0.061742",
        "v5_ledger_partial_result_cost_usd": "0.033475",
        "ledger_understatement_usd": "0.028267",
        "complete_cost_retained_by_v5_final_budget": True,
    }
    assert audit["closure"]["v5_replay_permitted"] is False


def test_bridge_is_observationally_non_binding_for_four_closed_cells() -> None:
    rebuilt = v6.build_bridge_audit(
        v4_route_plan_path=V4_ROUTE,
        v4_closure_path=V4_CLOSURE,
        v4_source_directory=V4_SOURCE,
        v5_plan_path=V5_PLAN,
        gemini_audit_path=GEMINI_AUDIT,
        gemini_closure_path=GEMINI_CLOSURE,
        gemini_source_directory=V5 / "gemini/source",
        repo_root=REPO_ROOT,
    )
    projections = rebuilt["source_projections"]
    assert [item["maximum_observed_round_fanout"] for item in projections] == [
        3,
        4,
        1,
        4,
    ]
    assert [item["executed_tool_calls_total"] for item in projections] == [3, 4, 1, 4]
    assert not any(
        item["client_fanout_caps_present_in_provider_request_contract"]
        for item in projections
    )
    assert rebuilt["scope"]["quality_observations"] == 0


def test_manifest_and_route_plan_freeze_the_material_13_by_13_policy() -> None:
    manifest = _json(MANIFEST)
    limits = manifest["run_design"]["execution_policy"]["limits"]
    assert limits["max_tool_calls_per_round"] == 13
    assert limits["max_tool_calls_total"] == 13
    assert limits["max_cumulative_tool_result_bytes"] == 65536
    plan = _json(PLAN)
    v6.validate_route_plan(plan, repo_root=REPO_ROOT)
    assert all(len(item["attempt_slots"]) == 56 for item in plan["work_items"])
    assert {
        item["route_coordinate"]["execution_policy_sha256"]
        for item in plan["work_items"]
    } == {
        "6bd6e2110979c03aae185c5e17d5ea323519bddc424ee6306519545b1a29af93",
        "2bd94f24787d2ba02ddb3d2a70a558d675408637a7c3799e9ad970c7e16ada2d",
    }


def test_derived_manifest_and_route_plan_reproduce_the_frozen_hashes() -> None:
    failure = _json(FAILURE)
    bridge = _json(BRIDGE)
    manifest = v6.build_v6_manifest(
        base_manifest_path=BASE_MANIFEST,
        failure_audit=failure,
        bridge_audit=bridge,
    )
    assert manifest["content_address"]["digest"] == (
        "052a214a4d1358ca80aba3612949c3aa1177907924a0f0c62df588424301eef1"
    )
    route = v6.build_route_plan(
        v5_plan_path=V5_PLAN,
        v4_closure_path=V4_CLOSURE,
        v5_sonnet_closure_path=SONNET_CLOSURE,
        v5_gemini_closure_path=GEMINI_CLOSURE,
        v5_aggregate_closure_path=AGGREGATE_CLOSURE,
        endpoint_snapshot_path=SNAPSHOT,
        failure_audit_path=FAILURE,
        bridge_audit_path=BRIDGE,
        v6_manifest_path=MANIFEST,
        v5_gemini_receipt_path=GEMINI_RECEIPT,
        repo_root=REPO_ROOT,
    )
    assert v5._sha256(route) == (
        "905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c"
    )


def test_fresh_ids_do_not_overlap_any_v4_or_v5_endpoint_closure() -> None:
    plan = _json(PLAN)
    prior = v6._closed_identifier_union(
        _json(V4_CLOSURE), _json(SONNET_CLOSURE), _json(GEMINI_CLOSURE)
    )
    observed = {
        "route_cell_ids": {item["route_cell_id"] for item in plan["work_items"]},
        "work_item_ids": {item["work_item_id"] for item in plan["work_items"]},
        "run_ids": {item["run_id"] for item in plan["work_items"]},
        "arm_ids": {
            arm for item in plan["work_items"] for arm in item["arm_ids"]
        },
        "attempt_ids": {
            slot["attempt_id"]
            for item in plan["work_items"]
            for slot in item["attempt_slots"]
        },
    }
    assert all(not values & prior[key] for key, values in observed.items())


def test_historical_preflight_made_no_calls_and_used_exact_freeze_budget() -> None:
    preflight = _json(PREFLIGHT)
    assert v5._artifact_verifies(preflight, v6.EXECUTION_PLAN_SCHEMA)
    assert preflight["execution"]["catalog_gets_made_by_plan"] == 0
    assert preflight["execution"]["provider_completion_requests_made_by_plan"] == 0
    assert preflight["execution"]["epicure_calls_made_by_plan"] == 0
    assert preflight["budget"]["current_total_exposure_usd"] == (
        "47.32616982666666666666666666"
    )
    assert preflight["budget"]["projected_total_exposure_usd"] == (
        "49.62361782666666666666666666"
    )


def test_aggregate_cannot_pass_without_two_source_verified_v6_pairs(
    tmp_path: Path,
) -> None:
    plan = _json(PLAN)
    receipt_path = v5._write_artifact(
        tmp_path / "receipts",
        "empty-v6-receipt",
        {
            "schema_version": v6.RECEIPT_SCHEMA,
            "route_plan_sha256": plan["artifact_sha256"],
            "uncertain_delivery_replayed": False,
            "v4_or_v5_identifier_replayed": False,
        },
    )
    audit_payload = v6.build_audit(
        plan=plan,
        receipt_path=receipt_path,
        root=tmp_path,
        repo_root=tmp_path,
    )
    assert audit_payload["decision"] == "failed_one_or_more_predicates"
    assert audit_payload["counts"]["usable_pairs"] == 0
    audit_path = v5._write_artifact(tmp_path / "audits", "v6-audit", audit_payload)
    audit = _json(audit_path)
    closure_payload = v6.build_closure(plan=plan, audit=audit)
    closure_path = v5._write_artifact(
        tmp_path / "closures", "v6-closure", closure_payload
    )
    bridge = _json(BRIDGE)
    bridge.pop("artifact_sha256")
    bridge_path = v5._write_artifact(tmp_path / "bridge", "bridge", bridge)
    aggregate = v6.build_aggregate_audit(
        plan=plan,
        bridge_audit_path=bridge_path,
        sonnet_audit_path=audit_path,
        sonnet_closure_path=closure_path,
        repo_root=tmp_path,
    )
    assert aggregate["decision"] == "failed_one_or_more_predicates"
    assert aggregate["endpoint_specific_sensitivity"][
        "estimable_from_route_gate_outputs"
    ] is False
    assert aggregate["study_admission"][
        "fresh_zero_call_confirmatory_preflight_permitted"
    ] is False
