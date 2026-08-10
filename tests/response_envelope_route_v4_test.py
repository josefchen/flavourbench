from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from flavourbench.frontier_coverage_repair_executor import (
    SupplementalRun,
    _budget_and_plan,
    build_materialization,
)
from flavourbench.provider import OpenRouterProvider, ProviderError
from flavourbench.response_envelope_route_v4 import (
    AUDIT_SCHEMA_VERSION,
    CANONICAL_MODEL_SLUG,
    EXPECTED_PROVIDER_CONTROLS,
    MODEL_ID,
    PROVIDER_NAME,
    PROVIDER_TAG,
    RECEIPT_SCHEMA_VERSION,
    _sha256,
    _write_artifact,
    build_catalog_snapshot,
    build_v4_audit,
    build_v4_closure,
    build_v4_plan,
    verify_v4_plan,
    verify_v4_route_acceptance_paths,
)
from flavourbench.run_journal import RunJournal

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
MANIFEST = next((CURRENT / "manifest-v29-high-resource").glob("*.json"))
V3_ROOT = CURRENT / "reasoning-effort-sensitivity-v3-route-validation"
V3_PLAN = V3_ROOT / (
    "reasoning-effort-v3-route-validation-plan-"
    "be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json"
)
V3_AUDIT = V3_ROOT / "final-be2f9d19/audits" / (
    "reasoning-effort-v3-route-validation-audit-"
    "aa66b52d784d813251f7506bbff3eff287f6a94c206fe0550b081ad34a37fb78.json"
)
V3_CLOSURE = V3_ROOT / (
    "reasoning-effort-v3-route-closure-"
    "290713a8758e9dcabd8567ed086425390537a121385a7ed6c956845d8d3ca1fb.json"
)
ROUTE_REGISTRY = ROOT / "artifacts/frontier-refresh/2026-08-01" / (
    "current-route-registry/aggregate/current-route-registry-"
    "b300d460ec3d93dbfdaea64e0809abf858fa9efb570d0bddeac28566b6cdf010.json"
)
CATALOG_AUDIT = CURRENT / "catalog-audit-v2" / (
    "current-model-catalog-audit-"
    "9a4507f9c83da65e3e2fe1fd03e147c36ef216dd490a7067bcd79440f1d28947.json"
)
TASKS = CURRENT / "reasoning-effort-sensitivity-v1/tasks" / (
    "reasoning-sensitivity-task-dossier-"
    "ceafb0eaeb047eee1bcdd506ca2442e8532f1afd8cb35f5f57dcfa7424af997e.json"
)
EPICURE = ROOT.parent / "paper/flavourbench/provenance/epicure-runtime-provenance-attestation.json"
COVERAGE_PLAN = ROOT.parent / (
    "paper/flavourbench/provenance/current-frontier-coverage-execution-plan.json"
)
ARENA = CURRENT / "frontier-model-arena-review-pool-quarantine-v1" / (
    "frontier-model-arena-review-pool-"
    "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json"
)
COVERAGE_SCHEDULE = CURRENT / "frontier-coverage-repair-v1" / (
    "frontier-coverage-repair-"
    "45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json"
)
VALIDATED_TASKS = ROOT / "artifacts/season1/task-validity/development-v2" / (
    "development-task-validity-v2-"
    "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
)
ROUTES = (
    MANIFEST,
    next((CURRENT / "manifest-v42-high-resource-cohere-direct").glob("*.json")),
)
BUDGET_AUDIT = CURRENT / "frontier-budget-audits" / (
    "frontier-global-budget-"
    "5c37a79515a4a0a69bdc0a6d19f21a68f7f2cc24b697088c96878a1eb5da528c.json"
)
SENSITIVITY_ROOT = CURRENT / "reasoning-effort-sensitivity-v1/runs"
SUPPLEMENTAL = tuple(
    SupplementalRun(
        source_directory=SENSITIVITY_ROOT / variant / "source",
        ledger_path=SENSITIVITY_ROOT / variant / "ledger.jsonl",
    )
    for variant in ("explicit_low", "provider_default", "explicit_high")
)
V4_EVIDENCE_ROOT = CURRENT / "response-envelope-route-v4"
V4_REAL_PLAN = V4_EVIDENCE_ROOT / (
    "response-envelope-route-v4-plan-"
    "a3ef7434064415c93ab78fe818339e0466b100bee01e10e67cbdf1e4d848a4d6.json"
)
V4_REAL_AUDIT = V4_EVIDENCE_ROOT / (
    "response-envelope-route-v4-audit-"
    "70fb6f9389885059f0ddf9bb6868ffe846ebcd48df67644a34075b9043dd32c3.json"
)
V4_REAL_CLOSURE = V4_EVIDENCE_ROOT / (
    "response-envelope-route-v4-closure-"
    "dfb54062b304b31c52f69a9698d6ffeda39f38f7bdf749d60fc9554f0d15078c.json"
)


@pytest.fixture()
def governed_root() -> Path:
    path = ROOT / "artifacts" / f"v4-test-{uuid.uuid4()}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _plan(governed_root: Path) -> tuple[dict, Path]:
    manifest = json.loads(MANIFEST.read_text())
    record = next(item for item in manifest["models"] if item["model"]["id"] == MODEL_ID)
    catalog = build_catalog_snapshot(
        model_document={"data": record["model"]},
        endpoint_document={"data": {"endpoints": [record["endpoint"]]}},
        observed_at="2026-08-03T00:00:00Z",
    )
    catalog_path = _write_artifact(governed_root, "catalog", catalog)
    payload = build_v4_plan(
        repo_root=ROOT.parent,
        v3_plan_path=V3_PLAN,
        v3_audit_path=V3_AUDIT,
        v3_closure_path=V3_CLOSURE,
        route_registry_path=ROUTE_REGISTRY,
        catalog_audit_path=CATALOG_AUDIT,
        fresh_catalog_path=catalog_path,
        high_resource_manifest_path=MANIFEST,
        task_dossier_path=TASKS,
        epicure_attestation_path=EPICURE,
        coverage_execution_plan_path=COVERAGE_PLAN,
        freeze_nonce="11111111-1111-4111-8111-111111111111",
    )
    path = _write_artifact(governed_root, "plan", payload)
    return json.loads(path.read_text()), path


def _request_contract(*, tool: bool = False) -> dict:
    return {
        "model": MODEL_ID,
        "provider": EXPECTED_PROVIDER_CONTROLS,
        "reasoning": {"effort": "low", "exclude": True},
        "reasoning_field_present": True,
        "response_format_sha256": None,
        "response_format_present": False,
        "tool_choice": "required" if tool else None,
        "tools": (
            [{"name": "find_pairings", "parameters_sha256": "1" * 64}] if tool else []
        ),
        "tools_present": tool,
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 20260715,
        "message_count": 2,
        "messages_sha256": "2" * 64,
    }


def _source_and_receipt(
    governed_root: Path, plan: dict
) -> tuple[Path, Path, Path, Path]:
    source_dir = governed_root / "source"
    source_dir.mkdir()
    run_id = plan["work"]["run_id"]
    slot_map = {
        (item["arm_id"], item["phase"], item["attempt_index"]): item["attempt_id"]
        for item in plan["work"]["attempt_slots"]
    }
    provider_events: list[dict] = []
    journal = RunJournal.create(source_dir, run_id=run_id, metadata={"fixture": True})
    generations: dict[str, list[str]] = {"epicure_off": [], "epicure_on": []}
    phases = {
        "epicure_off": ["planning", "evidence_decision", "final"],
        "epicure_on": ["planning", "tool_round_0", "final"],
    }
    for condition in ("epicure_off", "epicure_on"):
        arm_id = f"{run_id}:{condition}"
        for index, phase in enumerate(phases[condition]):
            attempt_id = slot_map[(arm_id, phase, 0)]
            generation_id = f"gen-v4-{condition}-{index}"
            generations[condition].append(generation_id)
            request = _request_contract(tool=phase == "tool_round_0")
            start = {
                "attempt_id": attempt_id,
                "arm_id": arm_id,
                "request_key_sha256": _sha256({"arm": arm_id, "phase": phase}),
                "phase": phase,
                "attempt_index": 0,
                "event_type": "request_started",
                "generation_id": "",
                "http_status": None,
                "error_type": "",
                "payload_sha256": _sha256({"phase": phase}),
                "metadata": {
                    "request_contract": request,
                    "request_contract_sha256": _sha256(request),
                },
            }
            received = {
                **start,
                "event_type": "response_received",
                "generation_id": generation_id,
                "http_status": 200,
                "metadata": {
                    "response_model": CANONICAL_MODEL_SLUG,
                    "finish_reason": "stop",
                    "openrouter_cache_status": "",
                    "cloudflare_cache_status": "MISS",
                    "response_envelope": {
                        "classification": "chat_completions",
                        "accepted_chat_completion": True,
                    },
                },
            }
            for event in (start, received):
                provider_events.append(event)
                journal.append("provider_attempt", event)
    on_arm = f"{run_id}:epicure_on"
    session_id = slot_map[(on_arm, "mcp_session", 0)]
    mcp_session_started = {
        "attempt_id": session_id,
        "arm_id": on_arm,
        "request_key_sha256": _sha256("mcp-session"),
        "phase": "mcp_session",
        "attempt_index": 0,
        "event_type": "mcp_session_started",
        "generation_id": "",
        "http_status": None,
        "error_type": "",
        "payload_sha256": "3" * 64,
        "metadata": {},
    }
    mcp_session_attested = {
        **mcp_session_started,
        "phase": "mcp_attestation",
        "event_type": "mcp_session_attested",
    }
    mcp_attempt_id = slot_map[(on_arm, "mcp_tool_0_0", 0)]
    mcp_started = {
        **mcp_session_started,
        "attempt_id": mcp_attempt_id,
        "phase": "mcp_tool_0_0",
        "event_type": "mcp_call_started",
        "request_key_sha256": _sha256("mcp-call"),
    }
    result_sha256 = hashlib.sha256(b"Real Epicure fixture evidence").hexdigest()
    mcp_completed = {
        **mcp_started,
        "event_type": "mcp_call_completed",
        "payload_sha256": result_sha256,
    }
    for event in (
        mcp_session_started,
        mcp_session_attested,
        mcp_started,
        mcp_completed,
    ):
        provider_events.append(event)
        journal.append("provider_attempt", event)
    mcp_trace = {
        "arm_id": on_arm,
        "round_index": 0,
        "name": "find_pairings",
        "arguments": {"ingredients": ["XO sauce"]},
        "result": "Real Epicure fixture evidence",
        "result_sha256": result_sha256,
        "latency_ms": 10,
        "is_error": False,
    }
    journal.append("mcp_trace", mcp_trace)
    descriptor = journal.finalize({"status": "generation_complete"})
    answer = (
        "Build the substitute around fermented savouriness, seafood depth, and restrained heat. "
        "Cook miso gently with anchovy paste, then add a little fish sauce and crisp aromatics; "
        "taste before adjusting because every pantry product has a different salt concentration."
    )
    results = {}
    for condition in ("epicure_off", "epicure_on"):
        metadata = [
            {
                "generation_id": generation_id,
                "model": CANONICAL_MODEL_SLUG,
                "provider": PROVIDER_NAME,
                "cost_micros": 1,
                "reconciled": True,
            }
            for generation_id in generations[condition]
        ]
        results[condition] = {
            "answer_markdown": answer,
            "actual_model_id": CANONICAL_MODEL_SLUG,
            "actual_provider": PROVIDER_NAME,
            "generation_ids": generations[condition],
            "generation_metadata": metadata,
            "cost_reconciled": True,
            "finish_reason": "stop",
            "final_response_mode": "plain_text",
            "intermediate_outputs": [],
            "tool_trace": [
                {
                    "round_index": 0,
                    "name": "find_pairings",
                    "arguments": {"ingredients": ["XO sauce"]},
                    "result": "Real Epicure fixture evidence",
                    "result_sha256": result_sha256,
                    "latency_ms": 10,
                    "is_error": False,
                }
            ]
            if condition == "epicure_on"
            else [],
        }
    source = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": run_id,
        "status": "complete",
        "requested_conditions": ["epicure_off", "epicure_on"],
        "requested_model_id": MODEL_ID,
        "requested_provider": PROVIDER_TAG,
        "model_contract": {"canonical_slug": CANONICAL_MODEL_SLUG},
        "endpoint_contract": {"provider_name": PROVIDER_NAME},
        "endpoint_execution_contract_sha256": plan["route"][
            "endpoint_execution_contract_sha256"
        ],
        "provider_routing_controls": EXPECTED_PROVIDER_CONTROLS,
        "provider_routing_controls_sha256": _sha256(EXPECTED_PROVIDER_CONTROLS),
        "execution_policy_sha256": plan["execution_policy_sha256"],
        "epicure": {
            "release_id": plan["epicure"]["release_id"],
            "bundle_sha256": plan["epicure"]["bundle_sha256"],
            "application_sha256": plan["epicure"]["application_sha256"],
        },
        "epicure_tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
        "errors": {},
        "results": results,
        "provider_attempt_events": provider_events,
        "mcp_trace_events": [mcp_trace],
        "budget": {"all_generation_costs_reconciled": True, "actual_cost_micros": 6},
        "incomplete_generation_metadata": [],
        "run_journal": descriptor.payload(),
    }
    source["artifact_sha256"] = _sha256(source)
    source_path = source_dir / f"source-{source['artifact_sha256']}.json"
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "record_role": "single_invocation_v4_execution_receipt",
        "v4_plan_sha256": plan["artifact_sha256"],
        "work_item_id": plan["work"]["work_item_id"],
        "run_id": run_id,
        "invocation_count": 1,
        "status": "source_artifact_available",
        "live_artifact": {
            "path": str(source_path.relative_to(ROOT.parent)),
            "sha256": source["artifact_sha256"],
        },
        "summary": {"status": "complete"},
        "error": None,
        "retry_outside_frozen_provider_phases": False,
        "second_route_attempted": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    receipt_path = _write_artifact(governed_root, "receipt", receipt)
    plan_path = next(governed_root.glob("plan-*.json"))
    audit = build_v4_audit(
        plan_path=plan_path,
        receipt_path=receipt_path,
        repo_root=ROOT.parent,
    )
    assert audit["schema_version"] == AUDIT_SCHEMA_VERSION
    audit_path = _write_artifact(governed_root, "audit", audit)
    audit_document = json.loads(audit_path.read_text())
    closure = build_v4_closure(
        plan=plan,
        audit=audit_document,
        receipt_path=receipt_path,
        repo_root=ROOT.parent,
    )
    closure_path = _write_artifact(governed_root, "closure", closure)
    return source_path, receipt_path, audit_path, closure_path


def test_v4_plan_freezes_fresh_non_replayed_identifiers(governed_root: Path) -> None:
    plan, _ = _plan(governed_root)
    assert verify_v4_plan(plan, repo_root=ROOT.parent, require_current_sources=True)
    assert plan["work"]["matched_pairs"] == 1
    assert plan["work"]["response_arms"] == 2
    assert len(plan["work"]["attempt_slots"]) == 35
    assert plan["budget"]["reserved_worst_case_usd"] == "0.05"
    assert plan["budget"]["admitted"] is True
    assert plan["claim_boundary"]["quality_observations"] == 0
    assert plan["claim_boundary"]["synthetic_or_model_judge_evidence"] is False


def test_closed_real_v4_evidence_reconstructs_from_immutable_sources() -> None:
    assert verify_v4_route_acceptance_paths(
        plan_path=V4_REAL_PLAN,
        audit_path=V4_REAL_AUDIT,
        closure_path=V4_REAL_CLOSURE,
        repo_root=ROOT.parent,
    )
    audit = json.loads(V4_REAL_AUDIT.read_text())
    correction = audit["verification_source"]
    assert correction["generation_source_files_unchanged"] is True
    assert correction["provider_calls_made_by_correction"] is False
    assert correction["mcp_calls_made_by_correction"] is False
    assert audit["counts"]["provider_requests"] == 7
    assert audit["counts"]["successful_epicure_tool_calls"] == 5
    assert audit["accounting"]["actual_cost_usd"] == "0.002899"


def test_attempt_id_factory_rejects_missing_or_duplicate_slots(monkeypatch) -> None:
    monkeypatch.setattr(
        "flavourbench.provider.get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "openrouter_api_key": "",
                "openrouter_http_referer": "https://example.test",
                "openrouter_title": "test",
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "openrouter_accounting_base_url": "https://openrouter.ai/api/v1",
                "openrouter_timeout_seconds": 10,
                "cloudflare_ai_gateway_token": "",
            },
        )(),
    )
    ids = iter(["frozen-one", "frozen-one"])
    provider = OpenRouterProvider(attempt_id_factory=lambda _arm, _phase, _index: next(ids))
    assert provider._new_attempt_id("arm", "final", 0) == "frozen-one"  # noqa: SLF001
    with pytest.raises(ProviderError, match="reused"):
        provider._new_attempt_id("arm", "final", 1)  # noqa: SLF001


def test_source_reconstruction_passes_and_tampering_fails(governed_root: Path) -> None:
    plan, plan_path = _plan(governed_root)
    source_path, _, audit_path, closure_path = _source_and_receipt(governed_root, plan)
    assert verify_v4_route_acceptance_paths(
        plan_path=plan_path,
        audit_path=audit_path,
        closure_path=closure_path,
        repo_root=ROOT.parent,
    )
    source = json.loads(source_path.read_text())
    source["provider_routing_controls"]["allow_fallbacks"] = True
    source["artifact_sha256"] = _sha256(
        {key: value for key, value in source.items() if key != "artifact_sha256"}
    )
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    assert not verify_v4_route_acceptance_paths(
        plan_path=plan_path,
        audit_path=audit_path,
        closure_path=closure_path,
        repo_root=ROOT.parent,
    )


def test_hash_only_fabricated_pass_cannot_unlock_v4(governed_root: Path) -> None:
    plan, plan_path = _plan(governed_root)
    _, _, audit_path, closure_path = _source_and_receipt(governed_root, plan)
    audit = json.loads(audit_path.read_text())
    audit["counts"]["provider_requests"] += 1
    audit["artifact_sha256"] = _sha256(
        {key: value for key, value in audit.items() if key != "artifact_sha256"}
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    assert not verify_v4_route_acceptance_paths(
        plan_path=plan_path,
        audit_path=audit_path,
        closure_path=closure_path,
        repo_root=ROOT.parent,
    )


def test_v4_pass_materializes_executable_zero_call_coverage_plan(
    governed_root: Path,
) -> None:
    plan, plan_path = _plan(governed_root)
    _, _, audit_path, closure_path = _source_and_receipt(governed_root, plan)
    materialization = build_materialization(
        schedule_path=COVERAGE_SCHEDULE,
        arena_path=ARENA,
        task_validity_path=VALIDATED_TASKS,
        route_manifest_paths=ROUTES,
    )
    with patch("flavourbench.frontier_coverage_repair_executor.subprocess.run") as run_mock:
        _, budget, coverage_plan = _budget_and_plan(
            materialization,
            budget_audit_path=BUDGET_AUDIT,
            project_root=ROOT,
            supplemental_runs=SUPPLEMENTAL,
            source_directory=governed_root / "coverage-source",
            corrections_directory=governed_root / "coverage-corrections",
            response_directory=governed_root / "coverage-responses",
            ledger_path=governed_root / "coverage-ledger.jsonl",
            global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
            global_artifact_directory=ROOT / "artifacts/live-smoke",
            global_corrections_directory=ROOT / "artifacts/corrections",
            global_reconciliation_directory=(
                ROOT / "artifacts/frontier-contract/reconciliations"
            ),
            cap_usd=Decimal("100"),
            admission_fraction=Decimal("0.85"),
            response_envelope_route_plan_path=plan_path,
            response_envelope_route_audit_path=audit_path,
            response_envelope_route_closure_path=closure_path,
        )
    run_mock.assert_not_called()
    assert budget.admission_allowed is True
    assert coverage_plan["status"] == "admissible_dry_run"
    assert coverage_plan["provider_calls_made_by_plan"] == 0
    assert coverage_plan["epicure_calls_made_by_plan"] == 0
    assert coverage_plan["response_envelope_route_gate"]["status"] == (
        "passed_all_predicates_v4_source_reconstructed"
    )
