from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.reasoning_effort_route_recovery import (
    RouteRecoveryError,
    _sha256,
    _write_artifact,
    build_v2_closed_identifiers,
    build_v3_route_closure,
    build_v3_route_plan,
    build_v3_route_validation_audit,
    materialize_v3_route_assets,
    verify_v3_route_closure,
    verify_v3_route_plan,
    verify_v3_route_validation_pass_audit,
)

ROOT = Path(__file__).parents[2]
V2_ROOT = (
    ROOT
    / "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-sensitivity-v2-route-validation"
)
V2_PLAN = (
    V2_ROOT
    / "reasoning-effort-v2-route-validation-plan-"
    "65b64747cbbecf116e3756f69bdbc7c0ccaf1a99a446d3352faa79b432e14e0f.json"
)
V2_FINAL = V2_ROOT / "final-65b64747"
V2_AUDIT = (
    V2_FINAL
    / "audits/reasoning-effort-v2-route-validation-audit-"
    "481303eefacc872701d6a09aa9baeefe887027f655ccdc114ab881c8a16ff821.json"
)
V2_ASSETS = (
    V2_FINAL
    / "reasoning-effort-v2-route-runner-assets-"
    "67ae96e905c36a1a6cdb2c208190bc6f9456dc5f4da9832cbcdc352e3f6ed805.json"
)
PROVIDER_SOURCE = ROOT / "flavourbench/src/flavourbench/provider.py"
V3_ROOT = (
    ROOT
    / "flavourbench/artifacts/season1/current-quality-run/"
    "reasoning-effort-sensitivity-v3-route-validation"
)
V3_PLAN = (
    V3_ROOT
    / "reasoning-effort-v3-route-validation-plan-"
    "be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json"
)
V3_ASSETS = (
    V3_ROOT
    / "final-be2f9d19/reasoning-effort-v3-route-runner-assets-"
    "aa2d631e73355d03f4f68709981e7d6995922158d9b057cac5bed29ad02a1844.json"
)
V3_PRIOR_AUDIT = (
    V3_ROOT
    / "final-be2f9d19/audits/reasoning-effort-v3-route-validation-audit-"
    "78f8209bdde679dafaa5cea45fe541fddd0cd2e78795c6e2a486ae6b7f2d8455.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_recovery_closes_v2_and_materializes_only_fresh_no_call_assets(
    tmp_path: Path,
) -> None:
    v2_plan = _load(V2_PLAN)
    v2_audit = _load(V2_AUDIT)
    v2_assets = _load(V2_ASSETS)

    closure = build_v2_closed_identifiers(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_runner_assets=v2_assets,
    )
    assert closure["counts"] == {
        "work_item_ids": 3,
        "attempt_ids": 5,
        "generation_ids": 4,
        "source_artifacts": 1,
    }
    assert closure["closure"]["replay_permitted"] is False
    closure_path = _write_artifact(tmp_path, "v2-closure", closure)
    written_closure = _load(closure_path)

    plan = build_v3_route_plan(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_closed_identifiers=written_closure,
        provider_source_path=PROVIDER_SOURCE,
    )
    plan_path = _write_artifact(tmp_path, "v3-plan", plan)
    written_plan = _load(plan_path)
    assert verify_v3_route_plan(written_plan)
    assert written_plan["preflight"]["decision"] == (
        "ready_to_materialize_v3_route_validation_only"
    )
    assert written_plan["budget"]["projected_post_route_exposure_usd"] == (
        "52.53410099333333333333333333"
    )
    v2_work_ids = set(closure["work_item_ids"])
    v3_work_ids = {
        item["work_item_id"] for item in written_plan["route_validation"]["work_items"]
    }
    assert len(v3_work_ids) == 3
    assert not v2_work_ids & v3_work_ids

    assets = materialize_v3_route_assets(
        plan=written_plan,
        plan_path=plan_path,
        v2_runner_assets=v2_assets,
        output_dir=tmp_path / "assets",
    )
    assert assets["provider_calls_made"] is False
    assert assets["epicure_calls_made"] is False
    assert assets["execution_order"] == [
        "explicit_low",
        "provider_default",
        "explicit_high",
    ]
    for variant in assets["variants"]:
        assert "--execute" not in variant["dry_run_command"]
        assert variant["live_command"] == [
            *variant["dry_run_command"],
            "--execute",
            "--confirm",
            "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
        ]
        manifest = _load(Path(variant["manifest"]))
        assert manifest["governance"]["v3_route_validation_only"] is True
        assert manifest["governance"]["rank_eligible"] is False
        assert manifest["run_design"]["route_validation_override"][
            "work_item_id"
        ] == variant["fresh_work_item_id"]


def test_v3_recovery_rejects_a_nonfailed_v2_audit() -> None:
    v2_plan = _load(V2_PLAN)
    v2_audit = _load(V2_AUDIT)
    v2_assets = _load(V2_ASSETS)
    v2_audit["decision"] = "passed_all_predicates"

    with pytest.raises(RouteRecoveryError, match="v2 audit does not verify"):
        build_v2_closed_identifiers(
            v2_plan=v2_plan,
            v2_audit=v2_audit,
            v2_runner_assets=v2_assets,
        )


def test_v3_source_derived_audit_is_not_executed_before_live_records(
    tmp_path: Path,
) -> None:
    v2_plan = _load(V2_PLAN)
    v2_audit = _load(V2_AUDIT)
    v2_assets = _load(V2_ASSETS)
    closure = build_v2_closed_identifiers(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_runner_assets=v2_assets,
    )
    closure_path = _write_artifact(tmp_path, "closure", closure)
    plan = build_v3_route_plan(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_closed_identifiers=_load(closure_path),
        provider_source_path=PROVIDER_SOURCE,
    )
    plan_path = _write_artifact(
        tmp_path, "reasoning-effort-v3-route-validation-plan", plan
    )
    written_plan = _load(plan_path)
    assets = materialize_v3_route_assets(
        plan=written_plan,
        plan_path=plan_path,
        v2_runner_assets=v2_assets,
        output_dir=tmp_path / "assets",
    )
    assets_path = _write_artifact(tmp_path, "assets", assets)
    receipt = build_v3_route_validation_audit(
        plan=written_plan,
        runner_assets=_load(assets_path),
    )

    assert receipt["decision"] == "not_executed"
    assert receipt["counts"]["attempted_pairs"] == 0
    assert receipt["full_sensitivity_admission"]["authorized"] is False
    assert not verify_v3_route_validation_pass_audit(
        {**receipt, "artifact_sha256": _sha256(receipt)}, written_plan
    )


def test_v3_strict_pass_verifier_enforces_closed_ids_retry_accounting_and_cost(
    tmp_path: Path,
) -> None:
    v2_plan = _load(V2_PLAN)
    v2_audit = _load(V2_AUDIT)
    closure = build_v2_closed_identifiers(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_runner_assets=_load(V2_ASSETS),
    )
    closure_path = _write_artifact(tmp_path, "closure", closure)
    plan_payload = build_v3_route_plan(
        v2_plan=v2_plan,
        v2_audit=v2_audit,
        v2_closed_identifiers=_load(closure_path),
        provider_source_path=PROVIDER_SOURCE,
    )
    plan_path = _write_artifact(tmp_path, "plan", plan_payload)
    plan = _load(plan_path)
    work_items = plan["route_validation"]["work_items"]
    closed = plan["closed_identifiers"]
    predicates = [
        {
            "predicate_id": item["predicate_id"],
            "status": "passed",
            "passed": True,
            "failures": [],
            "evidence_sha256": [plan["artifact_sha256"]],
        }
        for item in plan["acceptance_gate"]["predicates"]
    ]
    source_artifacts = [
        {
            "work_item_id": item["work_item_id"],
            "variant_id": item["variant_id"],
            "source_artifact_sha256": f"{index + 1:064x}",
            "summary_artifact_sha256": f"{index + 11:064x}",
            "response_artifact_sha256s": [
                f"{index + 21:064x}",
                f"{index + 31:064x}",
            ],
            "immutable": True,
        }
        for index, item in enumerate(work_items)
    ]
    attempts = [f"v3-attempt-{index}" for index in range(9)]
    generations = [f"v3-generation-{index}" for index in range(9)]
    baseline = plan["budget"]["post_v2_conservative_exposure_usd"]
    post_route = str(Decimal(baseline) + Decimal("0.3"))
    payload = {
        "schema_version": "flavourbench-reasoning-effort-v3-route-validation-audit-v1",
        "record_role": "source_derived_fail_closed_v3_route_gate_receipt",
        "v3_route_plan_sha256": plan["artifact_sha256"],
        "runner_assets_sha256": "a" * 64,
        "route_cell_id": plan["route_validation"]["route_cell_id"],
        "decision": "passed_all_predicates",
        "source_artifacts": source_artifacts,
        "variant_audits": [],
        "counts": {
            "attempted_pairs": 3,
            "usable_pairs": 3,
            "intended_arms": 6,
            "usable_arms": 6,
            "provider_requests": 9,
            "provider_responses": 9,
            "retryable_error_envelope_rejections": 0,
            "retried_error_envelope_rejections": 0,
            "terminal_error_envelope_rejections": 0,
            "unsafe_provider_rejections": 0,
            "successful_epicure_tool_calls": 3,
            "epicure_off_tool_calls": 0,
            "synthetic_arms": 0,
            "identity_mismatches": 0,
            "unreconciled_generations": 0,
            "non_chat_generation_responses": 0,
            "truncated_or_invalid_arms": 0,
        },
        "response_envelope_audit": {
            "contract_sha256": plan["safe_response_envelope_contract"][
                "contract_sha256"
            ],
            "provider_source_sha256": plan["source"]["provider_source_sha256"],
            "all_generation_responses_chat_completions": True,
            "retryable_rejections_excluded_from_generation_accounting": True,
            "safe_retryable_error_envelope_rejections": 0,
            "retried_error_envelope_rejections": 0,
            "terminal_error_envelope_rejections": 0,
            "unsafe_provider_rejections": 0,
        },
        "identity_audit": {
            "model_id": plan["route_validation"]["model_id"],
            "provider_endpoint": work_items[0]["provider_endpoint"],
            "actual_provider_name": work_items[0]["actual_provider_name"],
            "runtime_id": plan["epicure"]["runtime_id"],
            "bundle_sha256": plan["epicure"]["bundle_sha256"],
            "application_sha256": plan["epicure"]["application_sha256"],
            "tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
            "lineage_inventory_sha256": plan["epicure"]["lineage_inventory_sha256"],
        },
        "identifier_freshness_audit": {
            "closed_v1_identifiers_sha256": closed["v1"]["inventory_sha256"],
            "closed_v2_identifiers_sha256": closed["v2"]["inventory_sha256"],
            "closed_work_item_ids": sorted(
                set(closed["v1"]["work_item_ids"] + closed["v2"]["work_item_ids"])
            ),
            "closed_attempt_ids": sorted(
                set(closed["v1"]["attempt_ids"] + closed["v2"]["attempt_ids"])
            ),
            "closed_generation_ids": sorted(
                set(closed["v1"]["generation_ids"] + closed["v2"]["generation_ids"])
            ),
            "v3_work_item_ids": sorted(item["work_item_id"] for item in work_items),
            "v3_attempt_ids": attempts,
            "v3_generation_ids": generations,
            "work_item_overlap": [],
            "attempt_id_overlap": [],
            "generation_id_overlap": [],
            "duplicate_attempt_ids": [],
            "duplicate_generation_ids": [],
            "missing_generation_id_responses": 0,
            "all_identifiers_fresh": True,
        },
        "accounting_audit": {
            "identified_generation_cost_usd": "0.3",
            "conservative_retained_exposure_usd": "0.3",
            "post_route_conservative_exposure_usd": post_route,
            "admission_ceiling_usd": plan["budget"]["admission_ceiling_usd"],
            "accepted_generation_accounting_complete": True,
            "full_route_accounting_complete": True,
            "rejected_error_envelope_cost_lookups": 0,
        },
        "predicate_results": predicates,
        "full_sensitivity_admission": {
            "authorized": True,
            "scope": "materialize_prespecified_36_pair_72_arm_study",
            "route_validation_outputs_reused": False,
        },
        "quality_fit_eligible": False,
        "official": False,
        "rank_eligible": False,
    }
    receipt = {**payload, "artifact_sha256": _sha256(payload)}
    assert verify_v3_route_validation_pass_audit(receipt, plan)

    receipt["accounting_audit"]["rejected_error_envelope_cost_lookups"] = 1
    receipt["artifact_sha256"] = _sha256(
        {key: value for key, value in receipt.items() if key != "artifact_sha256"}
    )
    assert not verify_v3_route_validation_pass_audit(receipt, plan)


def test_v3_audit_treats_exhausted_allowlisted_rejections_as_safe_terminal_failures() -> None:
    receipt = build_v3_route_validation_audit(
        plan=_load(V3_PLAN),
        runner_assets=_load(V3_ASSETS),
    )

    assert receipt["decision"] == "failed_one_or_more_predicates"
    assert receipt["counts"]["provider_requests"] == 4
    assert receipt["counts"]["provider_responses"] == 0
    assert receipt["counts"]["retryable_error_envelope_rejections"] == 4
    assert receipt["counts"]["retried_error_envelope_rejections"] == 2
    assert receipt["counts"]["terminal_error_envelope_rejections"] == 2
    assert receipt["counts"]["unsafe_provider_rejections"] == 0
    retry_predicate = next(
        item
        for item in receipt["predicate_results"]
        if item["predicate_id"] == "retryable_error_envelope_safety"
    )
    assert retry_predicate["failures"] == []
    assert receipt["accounting_audit"][
        "accepted_generation_accounting_complete"
    ] is True
    assert receipt["accounting_audit"]["full_route_accounting_complete"] is False
    assert receipt["accounting_audit"]["rejected_error_envelope_cost_lookups"] == 0


def test_v3_closure_blocks_v4_after_both_arms_exhaust_route_retries(
    tmp_path: Path,
) -> None:
    plan = _load(V3_PLAN)
    assets = _load(V3_ASSETS)
    corrected_payload = build_v3_route_validation_audit(
        plan=plan,
        runner_assets=assets,
    )
    corrected_path = _write_artifact(tmp_path, "corrected-audit", corrected_payload)
    closure_payload = build_v3_route_closure(
        plan=plan,
        runner_assets=assets,
        prior_audit=_load(V3_PRIOR_AUDIT),
        corrected_audit=_load(corrected_path),
    )
    closure_path = _write_artifact(tmp_path, "v3-closure", closure_payload)
    closure = _load(closure_path)

    assert verify_v3_route_closure(closure)
    assert closure["closed_identifiers"]["replay_permitted"] is False
    assert len(closure["closed_identifiers"]["attempt_ids"]) == 4
    assert closure["closed_identifiers"]["generation_ids"] == []
    assert closure["observed_route_failure"]["accepted_generations"] == 0
    assert closure["decision"]["v4_materialized"] is False
    assert closure["decision"]["v4_authorized"] is False
