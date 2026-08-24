from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from flavourbench.frontier_contract_runner import AdmissionDenied
from flavourbench.reasoning_effort_route_gate_v4 import (
    CLOSURE_SCHEMA,
    EXECUTION_RECEIPT_SCHEMA,
    EXPECTED_DIGESTS,
    RouteGateError,
    _artifact_document_verifies,
    _expected_provider_controls,
    _request_reasoning_predicate,
    _require_live_environment_before_reservation,
    _sha256,
    _validate_route_plan_shape,
    _write_artifact,
    build_closure,
    build_execution_plan,
    build_route_audit,
    load_authoritative_inputs,
    verify_closure,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CURRENT = ROOT / "artifacts/season1/current-quality-run/reasoning-effort-sensitivity-v4"
HISTORY = CURRENT / (
    "reasoning-effort-v4-history-audit-"
    "308ac12ebdf375d83337d55a98a0c5aef055f6cb9b26d74795bf09d14b80b386.json"
)
BASELINE = CURRENT / (
    "reasoning-effort-v4-low-baseline-audit-"
    "1fce54a13e2f844ae7a5d6b2d6f97eee4a8f37d58520d026c25ebe31cb2970e6.json"
)
ROUTE_PLAN = CURRENT / (
    "reasoning-effort-v4-route-gate-plan-"
    "2ff31d457f7fb1cdfcb9f5e46ae8c47827a47bbaf4c8f15fd526f1ddf16bf352.json"
)
STUDY_PLAN = CURRENT / (
    "reasoning-effort-v4-study-plan-"
    "733977cc3eac48316244adcf9beb726824505173b9fe52140cb664ad35d348c0.json"
)
RUNNER_ASSETS = CURRENT / (
    "reasoning-effort-v4-runner-assets-"
    "f4516e382422add2a0a68b17857e7b724090e6b49542158cc2927b6cb8be6ebf.json"
)
PREFLIGHT = CURRENT / (
    "reasoning-effort-v4-preflight-"
    "a7396f64a4db08dc1eef8425b59eb61f21836bdc5a8c572f12748f6ee3e239f7.json"
)


@pytest.fixture(scope="module")
def inputs() -> dict:
    return load_authoritative_inputs(
        repo_root=REPO_ROOT,
        history_path=HISTORY,
        baseline_path=BASELINE,
        route_plan_path=ROUTE_PLAN,
        study_plan_path=STUDY_PLAN,
        runner_assets_path=RUNNER_ASSETS,
        preflight_path=PREFLIGHT,
    )


def test_exact_authoritative_package_rederives(inputs: dict) -> None:
    assert inputs["route_plan"]["artifact_sha256"] == EXPECTED_DIGESTS["route_plan"]
    assert inputs["study_plan"]["artifact_sha256"] == EXPECTED_DIGESTS["study_plan"]
    assert inputs["runner_assets"]["artifact_sha256"] == EXPECTED_DIGESTS[
        "runner_assets"
    ]
    assert inputs["preflight"]["artifact_sha256"] == EXPECTED_DIGESTS["preflight"]


def test_self_rehashed_route_plan_is_not_authoritative(inputs: dict) -> None:
    forged = json.loads(json.dumps(inputs["route_plan"]))
    forged["budget"]["route_gate_worst_case_usd"] = "0"
    forged["artifact_sha256"] = _sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(RouteGateError, match="authoritative"):
        _validate_route_plan_shape(forged, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("contract", "variant", "expected"),
    [
        ({"reasoning_field_present": False, "reasoning": None}, "provider_default", True),
        ({"reasoning_field_present": True, "reasoning": None}, "provider_default", False),
        (
            {
                "reasoning_field_present": True,
                "reasoning": {"effort": "high", "exclude": True},
            },
            "explicit_high",
            True,
        ),
        (
            {
                "reasoning_field_present": False,
                "reasoning": {"effort": "high", "exclude": True},
            },
            "explicit_high",
            False,
        ),
        (
            {
                "reasoning_field_present": True,
                "reasoning": {"effort": "low", "exclude": True},
            },
            "explicit_high",
            False,
        ),
    ],
)
def test_reasoning_semantics_are_presence_sensitive(
    contract: dict, variant: str, expected: bool
) -> None:
    assert _request_reasoning_predicate(contract, variant) is expected


def test_provider_controls_are_exact_route_and_price_bound(inputs: dict) -> None:
    route_plan = inputs["route_plan"]
    by_model = {}
    for work_item in route_plan["work_items"]:
        controls = _expected_provider_controls(route_plan, work_item, REPO_ROOT)
        model_id = work_item["route_coordinate"]["model_id"]
        by_model[model_id] = controls
        assert controls["allow_fallbacks"] is False
        assert controls["require_parameters"] is True
        assert controls["data_collection"] == "deny"
        assert controls["only"] == [work_item["route_coordinate"]["provider_endpoint"]]
        assert "zdr" not in controls
    assert by_model["deepseek/deepseek-v4-flash-0731"]["max_price"] == {
        "prompt": 0.09,
        "completion": 0.18,
    }
    assert by_model["google/gemini-3.6-flash"]["max_price"] == {
        "prompt": 0.75,
        "completion": 3.75,
    }
    assert by_model["anthropic/claude-sonnet-5"]["max_price"] == {
        "prompt": 2.0,
        "completion": 10.0,
    }


def test_zero_call_plan_never_invokes_live_smoke(inputs: dict, tmp_path: Path) -> None:
    budget = {
        "admission_allowed": True,
        "projected_total_exposure_usd": "60",
        "blockers": [],
    }
    with patch(
        "flavourbench.reasoning_effort_route_gate_v4.live_smoke"
    ) as live_mock:
        plan = build_execution_plan(
            route_plan=inputs["route_plan"],
            budget=budget,
            ledger_path=tmp_path / "ledger.jsonl",
            source_directory=tmp_path / "source",
        )
    live_mock.assert_not_called()
    assert plan["status"] == "admissible_dry_run"
    assert plan["counts"]["existing_sources"] == 0
    assert plan["counts"]["quality_observations"] == 0
    assert all(
        item["decision"] == "admit_one_pair_after_exact_reservation"
        for item in plan["decisions"]
    )


def test_hash_only_receipt_cannot_create_a_pass(inputs: dict, tmp_path: Path) -> None:
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "record_role": "six_pair_sequential_reasoning_effort_route_gate_receipt",
        "route_plan_sha256": inputs["route_plan"]["artifact_sha256"],
        "status": "six_pair_sources_available",
        "new_pair_invocations_this_command": 6,
        "total_source_pairs": 6,
        "total_finalized_pairs": 6,
        "source_artifacts": [],
        "ledger": {
            "path": "invented",
            "sha256": "0" * 64,
            "entry_count": 12,
            "head_entry_sha256": "1" * 64,
        },
        "outcomes": [],
        "final_budget": {},
        "retry_outside_prefrozen_provider_phases": False,
        "uncertain_delivery_replayed": False,
        "failed_suffix_reopened": False,
        "quality_observations": 0,
        "rank_eligible": False,
    }
    repo_tmp = ROOT / "artifacts" / f"route-gate-test-{tmp_path.name}"
    repo_tmp.mkdir(parents=True)
    try:
        receipt_path = _write_artifact(repo_tmp, "receipt", receipt)
        receipt_document = json.loads(receipt_path.read_text())
        audit = build_route_audit(
            route_plan=inputs["route_plan"],
            receipt=receipt_document,
            receipt_path=receipt_path,
            ledger_path=repo_tmp / "ledger.jsonl",
            source_directory=repo_tmp / "source",
            repo_root=REPO_ROOT,
        )
        assert audit["decision"] == "failed_one_or_more_predicates"
        assert audit["counts"]["usable_pairs"] == 0
        assert "all_six_pair_sources_are_required" in audit["failures"]
    finally:
        for path in sorted(repo_tmp.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        repo_tmp.rmdir()


def test_failed_audit_closes_every_identifier_without_unlocking_study(
    inputs: dict, tmp_path: Path
) -> None:
    receipt_path = ROOT / "artifacts" / f"route-gate-test-{tmp_path.name}.json"
    receipt_path.write_text("{}\n")
    audit_payload = {
        "schema_version": "flavourbench-reasoning-effort-route-gate-audit-v1",
        "record_role": "source_reconstructed_reasoning_effort_default_high_route_gate",
        "route_plan_sha256": inputs["route_plan"]["artifact_sha256"],
        "execution_receipt": {
            "path": str(receipt_path.relative_to(REPO_ROOT)),
            "artifact_sha256": "2" * 64,
        },
        "ledger": {},
        "decision": "failed_one_or_more_predicates",
        "failures": ["test_failure"],
        "pair_audits": [],
        "counts": {},
        "identifier_audit": {
            "planned_attempt_ids": sorted(
                slot["attempt_id"]
                for item in inputs["route_plan"]["work_items"]
                for slot in item["attempt_slots"]
            ),
            "observed_attempt_ids": [],
            "observed_generation_ids": [],
            "observed_request_key_sha256s": [],
        },
        "accounting": {"actual_cost_usd": "0"},
        "study_admission": {
            "authorized": False,
            "scope": "materialize_a_fresh_zero_call_full_study_preflight_only",
            "full_48_pair_study_executed": False,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "quality_observations": 0,
            "official": False,
            "rank_eligible": False,
            "enters_sensitivity_fit": False,
        },
    }
    try:
        audit_path = _write_artifact(tmp_path, "audit", audit_payload)
        audit = json.loads(audit_path.read_text())
        closure_payload = build_closure(
            route_plan=inputs["route_plan"],
            audit=audit,
            receipt_path=receipt_path,
            repo_root=REPO_ROOT,
        )
        closure_path = _write_artifact(tmp_path, "closure", closure_payload)
        closure = json.loads(closure_path.read_text())
        assert _artifact_document_verifies(closure, CLOSURE_SCHEMA)
        assert verify_closure(closure, route_plan=inputs["route_plan"], audit=audit)
        assert closure["closed_identifiers"]["replay_permitted"] is False
        assert closure["decision"]["route_gate_qualified"] is False
        assert closure["decision"]["full_study_zero_call_preflight_permitted"] is False
        assert len(closure["closed_identifiers"]["unused_attempt_ids"]) == len(
            closure["closed_identifiers"]["attempt_ids"]
        )
    finally:
        receipt_path.unlink(missing_ok=True)


def test_environment_gate_runs_before_any_reservation(monkeypatch) -> None:
    settings = type(
        "Settings",
        (),
        {
            "execution_mode": "mock",
            "live_authorized": False,
            "openrouter_api_key": "",
            "mcp_token": "",
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "cloudflare_ai_gateway_token": "",
        },
    )()
    monkeypatch.setattr(
        "flavourbench.reasoning_effort_route_gate_v4.get_settings", lambda: settings
    )
    with pytest.raises(AdmissionDenied, match="pre-reservation"):
        _require_live_environment_before_reservation()
