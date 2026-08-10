from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.reasoning_effort_sensitivity import (
    DEFAULT_MODEL_IDS,
    SensitivityProtocolError,
    _sha256,
    build_plan,
    build_preflight_receipt,
    build_v2_route_validation_plan,
    verify_plan,
)


def _write_artifact(path: Path, payload: dict) -> Path:
    digest = _sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    destination = path / f"fixture-{digest}.json"
    destination.write_text(json.dumps(document), encoding="utf-8")
    return destination


def _content_manifest(path: Path, payload: dict) -> Path:
    digest = _sha256(payload)
    document = {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
    destination = path / f"manifest-{digest}.json"
    destination.write_text(json.dumps(document), encoding="utf-8")
    return destination


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    policy = {
        "schema_version": "flavourbench-real-execution-policy-v9",
        "decoding": {"temperature": 1.0, "top_p": 0.95, "seed": 1},
        "limits": {
            "max_output_tokens": 8192,
            "max_intermediate_tokens": 8192,
            "max_tool_rounds": 3,
            "max_tool_result_bytes": 16384,
            "max_cumulative_tool_result_bytes": 65536,
            "max_tool_calls_per_round": 6,
            "max_tool_calls_total": 12,
            "max_provider_attempts": 2,
            "tool_argument_repair_turns": 1,
            "required_tool_contract_max_intermediate_tokens": 2048,
        },
        "cost_forecast": {
            "approximate_non_user_prompt_bytes": 2000,
            "conservative_bytes_per_token": 3,
            "tool_catalog_bytes_bound": 24000,
        },
        "pair_arm_scheduling": "concurrent",
        "evidence_protocol": "matched_evidence_v2",
        "final_response_mode": "plain_text",
        "matched_planning": True,
        "required_tool_contract_protocol": "direct_tool_first_v1",
        "epicure_on_tool_required": True,
        "reasoning": {
            "intermediate_effort": "low",
            "final_effort": "low",
            "exclude_from_provider_response": True,
        },
    }
    entries = []
    defaults = {
        DEFAULT_MODEL_IDS[0]: "medium",
        DEFAULT_MODEL_IDS[1]: "high",
        DEFAULT_MODEL_IDS[2]: "high",
    }
    for index, model_id in enumerate(DEFAULT_MODEL_IDS, start=1):
        entry = {
            "slot": {"slot_id": f"slot-{index}"},
            "model": {
                "id": model_id,
                "canonical_slug": f"canonical-{index}",
                "name": model_id,
                "reasoning": {
                    "supported_efforts": ["low", "medium", "high"],
                    "default_effort": defaults[model_id],
                    "default_enabled": True,
                    "mandatory": False,
                },
            },
            "endpoint": {
                "tag": f"provider-{index}",
                "provider_name": f"provider-{index}",
                "supported_parameters": ["reasoning", "reasoning_effort", "tools"],
            },
            "execution_route": {"selected_backend": "openrouter"},
            "endpoint_document_sha256": f"{index}" * 64,
            "backend_contract_sha256": "unfrozen",
            "forecast": {"pairs": 4, "model_block_worst_case_usd": "2"},
        }
        entries.append(entry)
    manifest = {
        "schema_version": "flavourbench-routed-candidate-manifest-v1",
        "status": "unranked_candidate",
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "models": entries,
        "run_design": {
            "execution_policy_sha256": "e" * 64,
            "execution_policy": policy,
            "generation_protocol": {
                "intermediate_reasoning_effort": "low",
                "final_reasoning_effort": "low",
            },
        },
    }
    manifest_path = _content_manifest(tmp_path, manifest)

    tasks = []
    anchors = []
    for index, family in enumerate(
        ("substitution", "composition", "cookability", "evidence"), start=1
    ):
        prompt = f"Human-authored {family} prompt {index}"
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        task_id = (
            "fb-s0-substitution-003" if family == "substitution" else f"task-{family}"
        )
        tasks.append(
            {
                "task_id": task_id,
                "family": family,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
                "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                "rank_eligible": False,
                "confirmatory_eligible": False,
                "source_url": "https://example.test/task",
                "source_license": "CC BY-SA 4.0",
                "task_specific_criterion_status": "pending_independent_human_authoring",
                "surface_dependency_screen": {"status": "pass", "failure_reasons": []},
            }
        )
        anchors.append(
            {
                "task_id": task_id,
                "family": family,
                "prompt_sha256": prompt_sha,
                "suspect": False,
            }
        )
    validity_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "flavourbench-development-task-validity-v2",
            "counts": {"synthetic_tasks": 0},
            "claim_boundary": {
                "official": False,
                "rank_eligible": False,
                "supports_official_leaderboard": False,
                "synthetic_tasks": 0,
            },
            "tasks": tasks,
        },
    )
    schedule_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "fixture-coverage-schedule",
            "anchors": anchors,
            "quarantined_task_ids": ["not-selected"],
            "synthetic_tasks": 0,
            "official": False,
        },
    )
    lineage_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "epicure-recovered-runtime-inventory-v2",
            "runtime_id": "epicure-mcp-1790-r1+fixture",
            "bundle": {"sha256": "b" * 64},
            "application": {
                "sha256": "a" * 64,
                "git": {
                    "dirty_files": [
                        {
                            "git_status": " M",
                            "path": "Dockerfile",
                            "bytes": 1,
                            "sha256": "d" * 64,
                        }
                    ]
                },
            },
            "tool_contract": {"semantic_sha256": "c" * 64},
            "release_gates": {"clean_signed_application_release": False},
            "runtime_attestation": {"matches_recovered_checkout": True},
            "rank_eligible": False,
            "redistributable": False,
        },
    )
    budget_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "flavourbench-frontier-global-budget-audit-v1",
            "currency": "USD",
            "synthetic_sources": 0,
            "hard_cap_respected": True,
            "current_total_exposure_usd": "40",
            "admission_ceiling_usd": "85",
            "hard_cap_usd": "100",
        },
    )
    return {
        "manifest": manifest_path,
        "tasks": validity_path,
        "schedule": schedule_path,
        "lineage": lineage_path,
        "budget": budget_path,
    }


def test_plan_freezes_real_balanced_three_effort_design_and_separates_officialization(
    tmp_path: Path,
) -> None:
    paths = _fixtures(tmp_path)
    payload = build_plan(
        base_manifest_path=paths["manifest"],
        lineage_inventory_path=paths["lineage"],
        coverage_schedule_path=paths["schedule"],
        task_validity_path=paths["tasks"],
        budget_audit_path=paths["budget"],
    )
    path = _write_artifact(tmp_path, payload)
    plan = json.loads(path.read_text(encoding="utf-8"))

    assert verify_plan(plan)
    assert plan["base_evidence"]["verified_explicit_low_configuration"][
        "final_reasoning_effort"
    ] == "low"
    assert len(plan["task_design"]["anchors"]) == 4
    assert plan["task_design"]["synthetic_tasks"] == 0
    assert plan["model_design"]["model_count"] == 3
    assert plan["execution"]["pairs"] == 36
    assert plan["execution"]["response_arms"] == 72
    assert plan["execution"]["synthetic_arms"] == 0
    assert plan["budget"]["study_worst_case_usd"] == "18"
    assert plan["preflight"]["decision"] == "eligible_for_live_route_smoke"
    assert plan["preflight"]["collection_blockers"] == []
    assert {item["gate"] for item in plan["preflight"]["officialization_blockers"]} == {
        "epicure_rank_eligible_release",
        "epicure_redistributable_release",
        "epicure_immutable_application_release",
    }
    receipt = build_preflight_receipt(plan)
    assert receipt["decision"] == "live_smoke_required"
    assert receipt["provider_calls_made"] == 0
    assert receipt["epicure_calls_made"] == 0


def test_plan_rejects_quarantined_anchor(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    schedule = json.loads(paths["schedule"].read_text(encoding="utf-8"))
    schedule.pop("artifact_sha256")
    schedule["quarantined_task_ids"] = [schedule["anchors"][0]["task_id"]]
    paths["schedule"] = _write_artifact(tmp_path, schedule)

    with pytest.raises(SensitivityProtocolError, match="non-suspect"):
        build_plan(
            base_manifest_path=paths["manifest"],
            lineage_inventory_path=paths["lineage"],
            coverage_schedule_path=paths["schedule"],
            task_validity_path=paths["tasks"],
            budget_audit_path=paths["budget"],
        )


def test_plan_blocks_shared_budget_without_dropping_work_items(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    budget = json.loads(paths["budget"].read_text(encoding="utf-8"))
    budget.pop("artifact_sha256")
    budget["current_total_exposure_usd"] = "80"
    paths["budget"] = _write_artifact(tmp_path, budget)
    plan = build_plan(
        base_manifest_path=paths["manifest"],
        lineage_inventory_path=paths["lineage"],
        coverage_schedule_path=paths["schedule"],
        task_validity_path=paths["tasks"],
        budget_audit_path=paths["budget"],
    )

    assert len(plan["execution"]["work_items"]) == 36
    assert "shared_budget_admission" in {
        blocker["gate"] for blocker in plan["preflight"]["collection_blockers"]
    }


def test_lineage_release_cannot_be_promoted_by_mutating_unhashed_copy(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    original = json.loads(paths["lineage"].read_text(encoding="utf-8"))
    forged = copy.deepcopy(original)
    forged["rank_eligible"] = True
    forged["redistributable"] = True
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(SensitivityProtocolError, match="does not verify"):
        build_plan(
            base_manifest_path=paths["manifest"],
            lineage_inventory_path=forged_path,
            coverage_schedule_path=paths["schedule"],
            task_validity_path=paths["tasks"],
            budget_audit_path=paths["budget"],
        )


def test_v2_route_validation_freezes_fresh_diagnostic_ids_without_calls(
    tmp_path: Path,
) -> None:
    paths = _fixtures(tmp_path)
    v1_payload = build_plan(
        base_manifest_path=paths["manifest"],
        lineage_inventory_path=paths["lineage"],
        coverage_schedule_path=paths["schedule"],
        task_validity_path=paths["tasks"],
        budget_audit_path=paths["budget"],
    )
    v1_path = _write_artifact(tmp_path, v1_payload)
    v1_plan = json.loads(v1_path.read_text(encoding="utf-8"))
    audit_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "flavourbench-reasoning-effort-smoke-audit-v1",
            "plan_sha256": v1_plan["artifact_sha256"],
            "decision": "blocked_before_full_study_due_smoke_failures",
            "counts": {"usable_pairs": 0},
            "variants": [
                {
                    "variant_id": variant_id,
                    "conservative_retained_exposure_usd": "1.5",
                }
                for variant_id in ("explicit_low", "provider_default", "explicit_high")
            ],
            "budget": {"post_smoke_conservative_exposure_usd": "44.5"},
        },
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    base_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    closed_ids_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "flavourbench-reasoning-effort-v1-closed-identifiers-v1",
            "v1_smoke_audit_sha256": audit["artifact_sha256"],
            "work_item_ids": sorted(
                item["work_item_id"]
                for item in v1_plan["execution"]["work_items"]
                if item["model_id"] == "openai/gpt-5.6-sol-pro"
                and item["task_id"] == "fb-s0-substitution-003"
            ),
            "attempt_ids": ["v1-attempt-1"],
            "generation_ids": ["v1-generation-1"],
        },
    )
    closed_ids = json.loads(closed_ids_path.read_text(encoding="utf-8"))

    old_lineage = json.loads(paths["lineage"].read_text(encoding="utf-8"))
    old_lineage.pop("artifact_sha256")
    old_lineage["supersedes_parser_defective_inventory_sha256"] = v1_plan["epicure"][
        "lineage_inventory_sha256"
    ]
    corrected_lineage_path = _write_artifact(tmp_path, old_lineage)
    corrected_lineage = json.loads(corrected_lineage_path.read_text(encoding="utf-8"))
    lineage_correction_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "epicure-recovered-runtime-inventory-correction-v1",
            "parser_defective_inventory_sha256": v1_plan["epicure"][
                "lineage_inventory_sha256"
            ],
            "authoritative_inventory_sha256": corrected_lineage["artifact_sha256"],
            "correction": {"other_inventory_fields_changed": 0},
        },
    )
    lineage_correction = json.loads(
        lineage_correction_path.read_text(encoding="utf-8")
    )

    provider_source = tmp_path / "provider.py"
    provider_source.write_text(
        "\n".join(
            (
                "openrouter_error_envelope",
                "gateway_api_envelope",
                "responses_api_schema_mismatch",
                "unknown_non_chat_completion_envelope",
                "response_envelope",
            )
        ),
        encoding="utf-8",
    )
    payload = build_v2_route_validation_plan(
        v1_plan=v1_plan,
        v1_smoke_audit=audit,
        v1_closed_identifiers=closed_ids,
        base_manifest=base_manifest,
        lineage_inventory=corrected_lineage,
        lineage_correction=lineage_correction,
        provider_source_path=provider_source,
    )

    work_items = payload["route_validation"]["work_items"]
    v1_ids = {
        item["work_item_id"]
        for item in v1_plan["execution"]["work_items"]
        if item["model_id"] == "openai/gpt-5.6-sol-pro"
        and item["task_id"] == "fb-s0-substitution-003"
    }
    assert len(work_items) == 3
    assert len({item["work_item_id"] for item in work_items}) == 3
    assert {item["work_item_id"] for item in work_items}.isdisjoint(v1_ids)
    assert payload["route_validation"]["response_arms"] == 6
    assert payload["route_validation"]["diagnostic_outputs_enter_quality_fit"] is False
    assert payload["full_sensitivity"]["status"] == (
        "blocked_pending_v2_route_validation"
    )
    assert payload["preflight"] == {
        "decision": "ready_to_materialize_v2_route_validation_only",
        "collection_blockers": [],
        "provider_calls_made": False,
        "epicure_calls_made": False,
    }
    assert payload["budget"]["projected_post_route_exposure_usd"] == "49"
    assert payload["epicure"]["lineage_inventory_sha256"] == corrected_lineage[
        "artifact_sha256"
    ]


def test_v2_route_validation_rejects_parser_defective_v1_lineage(
    tmp_path: Path,
) -> None:
    paths = _fixtures(tmp_path)
    v1_path = _write_artifact(
        tmp_path,
        build_plan(
            base_manifest_path=paths["manifest"],
            lineage_inventory_path=paths["lineage"],
            coverage_schedule_path=paths["schedule"],
            task_validity_path=paths["tasks"],
            budget_audit_path=paths["budget"],
        ),
    )
    v1_plan = json.loads(v1_path.read_text(encoding="utf-8"))
    audit_path = _write_artifact(
        tmp_path,
        {
            "plan_sha256": v1_plan["artifact_sha256"],
            "decision": "blocked_before_full_study_due_smoke_failures",
            "counts": {"usable_pairs": 0},
        },
    )
    provider_source = tmp_path / "provider.py"
    provider_source.write_text(
        "openrouter_error_envelope gateway_api_envelope "
        "responses_api_schema_mismatch unknown_non_chat_completion_envelope "
        "response_envelope",
        encoding="utf-8",
    )
    closed_ids_path = _write_artifact(
        tmp_path,
        {
            "schema_version": "flavourbench-reasoning-effort-v1-closed-identifiers-v1",
            "v1_smoke_audit_sha256": json.loads(
                audit_path.read_text(encoding="utf-8")
            )["artifact_sha256"],
            "work_item_ids": sorted(
                item["work_item_id"]
                for item in v1_plan["execution"]["work_items"]
                if item["model_id"] == "openai/gpt-5.6-sol-pro"
                and item["task_id"] == "fb-s0-substitution-003"
            ),
            "attempt_ids": ["v1-attempt-1"],
            "generation_ids": ["v1-generation-1"],
        },
    )

    with pytest.raises(SensitivityProtocolError, match="parser-defective"):
        lineage = json.loads(paths["lineage"].read_text(encoding="utf-8"))
        correction_path = _write_artifact(
            tmp_path,
            {
                "schema_version": "epicure-recovered-runtime-inventory-correction-v1",
                "parser_defective_inventory_sha256": lineage["artifact_sha256"],
                "authoritative_inventory_sha256": lineage["artifact_sha256"],
                "correction": {"other_inventory_fields_changed": 0},
            },
        )
        build_v2_route_validation_plan(
            v1_plan=v1_plan,
            v1_smoke_audit=json.loads(audit_path.read_text(encoding="utf-8")),
            v1_closed_identifiers=json.loads(
                closed_ids_path.read_text(encoding="utf-8")
            ),
            base_manifest=json.loads(paths["manifest"].read_text(encoding="utf-8")),
            lineage_inventory=lineage,
            lineage_correction=json.loads(
                correction_path.read_text(encoding="utf-8")
            ),
            provider_source_path=provider_source,
        )
