from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.expert_calibration import TASK_SCOPE_QUARANTINE, TASK_SCOPE_REVIEW_SHA256
from flavourbench.season1_readiness import (
    DEFAULT_METHOD_VALIDATION,
    HUMAN_QA_SCHEMA_VERSION,
    Season1ReadinessError,
    build_report,
    canonical_bytes,
    latest_human_report,
    valid_research_archive,
)

ROOT = Path(__file__).resolve().parents[1]


def _hashed(payload: dict) -> dict:
    return {**payload, "artifact_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest()}


def _human_payload() -> dict:
    reviewed_tasks = sorted(TASK_SCOPE_QUARANTINE)[:7]
    return {
        "schema_version": HUMAN_QA_SCHEMA_VERSION,
        "observed_at": "2026-07-31T15:50:30.015367+00:00",
        "source_pool": {
            "historical_review_session_pool_sha256": "a" * 64,
            "candidate_artifact": {"artifact_sha256": "a" * 64},
        },
        "scope_audit": {
            "general_track_quarantine_tasks_reviewed": 7,
            "governed_quarantine_tasks": 17,
            "task_public_ids": reviewed_tasks,
            "governance_review": {
                "schema_version": "flavourbench-specialist-scope-review-v1",
                "artifact_sha256": TASK_SCOPE_REVIEW_SHA256,
                "quarantined_task_count": 17,
            },
        },
        "review_progress": {
            "completed_presentations": 32,
            "unique_primary_judgments": 29,
        },
        "completion_audit": {
            "non_normal_response_arms": 4,
            "replacement_candidate": {"artifact_sha256": "b" * 64},
        },
        "claim_boundary": {
            "evidence_status": "restricted_operational_qa",
            "paper_use": False,
            "research_use": False,
            "rank_eligible": False,
            "leaderboard_use": False,
        },
    }


def _inputs() -> dict:
    method_validation = json.loads(DEFAULT_METHOD_VALIDATION.read_text(encoding="utf-8"))
    panel = _hashed(
        {
            "schema_version": "flavourbench-current-route-registry-v1",
            "status": "current_registry_with_one_failed_exact_route",
            "counts": {
                "contract_failed": 1,
                "contract_passed": 15,
                "models": 16,
                "quality_observations": 0,
                "rankable_comparisons": 0,
                "real_provider_generations_in_passed_receipts": 30,
                "real_epicure_calls_in_passed_receipts": 15,
            },
            "rank_eligible": False,
        }
    )
    parity = _hashed(
        {
            "schema_version": "parity-v1",
            "selected_contract_evidence": {
                "routes": 8,
                "all_normal_stop": True,
                "all_strict_structured_output": True,
                "reconciled_cost_usd": "0.2",
            },
            "total_qualification_exposure_interval_usd": {"lower": "0.2", "upper": "6.2"},
            "claim_boundary": {"rank_eligible": False},
        }
    )
    human = _hashed(_human_payload())
    design = _hashed(
        {
            "schema_version": "flavourbench-season1-study-design-v5",
            "status": "prospective-design-superseding-v4-before-scored-collection",
            "target_population": {"language": "English"},
            "task_bank": {
                "total": 240,
                "minimum_surface_diagnostic_coverage": 96,
                "admission": {
                    "blind_prompt_only_solutions_per_task": 2,
                    "independent_reconciliations_per_task": 2,
                    "independent_adjudications_per_task": 1,
                    "minimum_human_validity_records": 720,
                    "validator_calibration_minimum_precision": 0.95,
                    "validator_calibration_minimum_recall": 0.90,
                    "independent_validator_contract_reviews_per_task": 1,
                    "independent_contamination_audit_reviews_per_task": 1,
                },
                "contamination": {
                    "methods": ["exact", "fuzzy", "ngram", "semantic", "web"],
                    "labeled_detection_calibration": {
                        "minimum_cases": 150,
                        "minimum_overall_precision": 0.95,
                        "minimum_overall_recall": 0.90,
                        "minimum_paraphrase_recall": 0.85,
                        "required_before_scored_collection": True,
                    },
                },
            },
            "primary_controlled_collection": {
                "model_arena": {"total_battles": 3200, "endpoint_appearances": 6400},
                "epicure_uplift": {"total_pairs": 3200},
                "total_model_response_arms": 12800,
            },
            "analysis": {"no_composite_score": True},
            "claim_boundary": {
                "synthetic_observations": (
                    "prohibited-from-all-scored-and-supplemental-empirical-evidence"
                )
            },
            "validity_and_robustness": {
                "post_collection_item_audit": {
                    "minimum_random_tasks": 60,
                    "all_anomaly_flagged_tasks": True,
                    "minimum_independent_auditors_per_task": 2,
                    "release_requires_zero_unresolved_material_defects": True,
                },
                "generation_reliability_panel": {
                    "task_count": 20,
                    "endpoint_count": 16,
                    "independent_generations_per_cell": 3,
                    "total_panel_arms": 1920,
                    "retries_are_not_repetitions": True,
                },
                "prompt_sensitivity_audit": {
                    "total_response_arms": 480,
                    "ranking_use": "development-only-non-ranking-audit",
                },
                "practical_cookability_execution": {
                    "task_count": 24,
                    "total_kitchen_executions": 48,
                },
                "total_planned_real_model_response_arms_including_robustness": 14560,
            },
        }
    )
    database = {
        "scope": {
            "season_id": "season-1-id",
            "season_slug": "season-1",
            "all_release_counts_season_scoped": True,
        },
        "counts": {
            "official_epicure_releases": 0,
            "official_battles": 0,
            "rank_eligible_battles": 0,
            "votes": 29,
            "valid_public_preferences": 0,
            "valid_expert_judgments": 0,
        },
        "season_1": {
            "id": "season-1-id",
            "status": "draft",
            "official": False,
            "manifest_sha256": "unfrozen",
            "prompt_registry_sha256": "unfrozen",
            "tool_registry_sha256": "unfrozen",
            "analysis_plan_sha256": "unfrozen",
            "protocol_bundle_sha256": "unfrozen",
        },
        "task_bank": {
            "season1_eligible": 0,
            "scored": 0,
            "development": 0,
            "private_reserve": 0,
            "calibration_only": 32,
            "synthetic": 0,
            "independent_task_approvals": 0,
            "legacy_task_candidate_reviews": 0,
            "blind_prompt_only_validity_records": 0,
            "independent_reconciliation_records": 0,
            "independent_adjudication_records": 0,
            "task_review_evidence_store_available": False,
            "construct_blueprint_verified": False,
            "construct_blueprint_sha256": None,
            "criterion_pack_tasks": 0,
            "contamination_replay_tasks": 0,
            "contamination_replay_bundle_count": 0,
            "surface_diagnostic_tasks": 0,
            "required_surface_diagnostic_tasks": 96,
            "validator_calibration_verified": False,
            "validator_calibration_artifact_sha256": None,
            "contamination_calibration_verified": False,
            "contamination_calibration_artifact_sha256": None,
            "contamination_calibration_case_count": 0,
            "contamination_calibration_precision_milli": 0,
            "contamination_calibration_recall_milli": 0,
            "contamination_calibration_paraphrase_recall_milli": 0,
        },
        "provider_budgets": {},
        "independent_reviewers_by_family": {
            "substitution": 0,
            "composition": 0,
            "cookability": 0,
            "evidence": 0,
        },
        "research_consent": {"consented_battles": 0},
        "fixture_rows": {"catalog_models": 0, "season_models": 0, "battles": 0},
        "leaderboard_snapshots": [],
        "research_release_archives": [],
    }
    services = {
        name: {"status": "running", "health": "healthy"} for name in ("db", "api", "worker", "mcp")
    }
    return {
        "panel": panel,
        "parity": parity,
        "human": human,
        "study_design": design,
        "method_validation": method_validation,
        "epicure_release": {
            "release_id": "epicure-mcp-1790-r1",
            "status": "release_candidate_blocked",
            "bundle": {"sha256": "b" * 64},
            "rights": {"status": "pending"},
            "rank_eligible": False,
        },
        "database": database,
        "services": services,
        "consent": {"active": False, "status": "not active"},
        "sources": [],
    }


def _robustness_evidence(study_design_sha256: str) -> dict[str, dict]:
    common = {
        "status": "complete",
        "study_design_artifact_sha256": study_design_sha256,
        "synthetic_observations": 0,
    }
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()
    families = ("composition", "cookability", "evidence", "substitution")
    reliability_tasks = [f"task-{index:02d}" for index in range(20)]
    reliability_task_families = {
        task_id: families[index // 5] for index, task_id in enumerate(reliability_tasks)
    }
    post_records = []
    for index in range(60):
        defect = index < 2
        post_records.append(
            {
                "task_id": f"audit-task-{index:03d}",
                "task_content_sha256": digest(f"task-content-{index}"),
                "selection_reasons": ["random", "anomaly"] if index < 7 else ["random"],
                "auditor_commitments_sha256": [
                    digest(f"auditor-a-{index}"),
                    digest(f"auditor-b-{index}"),
                ],
                "material_defect": defect,
                "resolution_status": (
                    "retired_and_snapshots_recomputed" if defect else "no_material_defect"
                ),
                **(
                    {
                        "challenge_artifact_sha256": digest(f"challenge-{index}"),
                        "retirement_event_sha256": digest(f"retirement-{index}"),
                        "snapshot_recomputation_artifact_sha256": digest(
                            f"recomputation-{index}"
                        ),
                    }
                    if defect
                    else {}
                ),
            }
        )
    reliability_records = []
    for task_id in reliability_tasks:
        for endpoint_index in range(16):
            for condition in ("epicure_off", "epicure_on"):
                cell = f"{task_id}-endpoint-{endpoint_index:02d}-{condition}"
                reliability_records.append(
                    {
                        "task_id": task_id,
                        "endpoint_id": f"endpoint-{endpoint_index:02d}",
                        "condition": condition,
                        "arm_ids": [f"arm-{cell}-{run}" for run in range(3)],
                        "provider_retry_attempt_ids": [],
                    }
                )
    prompt_records = []
    prompt_variants = [digest(f"prompt-variant-{index}") for index in range(3)]
    for task_id in reliability_tasks:
        for endpoint_index in range(8):
            for variant_index, variant_sha256 in enumerate(prompt_variants):
                prompt_records.append(
                    {
                        "task_id": task_id,
                        "endpoint_id": f"prompt-endpoint-{endpoint_index:02d}",
                        "prompt_variant_sha256": variant_sha256,
                        "arm_id": (
                            f"prompt-arm-{task_id}-{endpoint_index:02d}-{variant_index}"
                        ),
                        "rank_eligible": False,
                    }
                )
    kitchen_records = []
    for task_index in range(24):
        for cook_index in range(2):
            kitchen_records.append(
                {
                    "execution_id": f"kitchen-{task_index:02d}-{cook_index}",
                    "task_id": f"cook-task-{task_index:02d}",
                    "output_id": f"cook-output-{task_index:02d}",
                    "cook_commitment_sha256": digest(f"cook-{task_index}-{cook_index}"),
                    "model_and_condition_blinded": True,
                    "completed": True,
                    "elapsed_seconds": 1800 + task_index,
                    "instruction_deviations": [],
                    "yield_recorded": True,
                    "blinded_acceptability": 4,
                }
            )
    return {
        "post_collection_item_audit": _hashed(
            {
                **common,
                "schema_version": "flavourbench-season1-post-collection-item-audit-v1",
                "counts": {
                    "population_tasks": 240,
                    "random_tasks_audited": 60,
                    "anomaly_flagged_tasks": 7,
                    "anomaly_flagged_tasks_audited": 7,
                    "minimum_independent_auditors_per_task": 2,
                    "confirmed_material_defects": 2,
                    "retired_material_defects": 2,
                    "unresolved_material_defects": 0,
                    "unique_tasks_audited": 60,
                },
                "task_records": post_records,
                "sampling_seed_committed_before_model_results": True,
                "original_task_roles_excluded": True,
                "affected_snapshots_recomputed": True,
            }
        ),
        "generation_reliability_panel": _hashed(
            {
                **common,
                "schema_version": "flavourbench-season1-generation-reliability-panel-v1",
                "counts": {
                    "tasks": 20,
                    "endpoints": 16,
                    "conditions": 2,
                    "independent_generations_per_cell": 3,
                    "response_arms": 1920,
                    "incremental_response_arms": 1280,
                },
                "retries_counted_as_repetitions": False,
                "ranking_use": "reported-separately-not-pooled",
                "task_families": reliability_task_families,
                "cell_records": reliability_records,
                "metrics_artifact_sha256": digest("reliability-metrics"),
            }
        ),
        "prompt_sensitivity_audit": _hashed(
            {
                **common,
                "schema_version": "flavourbench-season1-prompt-sensitivity-audit-v1",
                "split": "development",
                "counts": {
                    "tasks": 20,
                    "endpoints": 8,
                    "prompt_variants": 3,
                    "response_arms": 480,
                },
                "selection_after_results": False,
                "rank_eligible": False,
                "task_families": reliability_task_families,
                "arm_records": prompt_records,
                "metrics_artifact_sha256": digest("prompt-metrics"),
            }
        ),
        "practical_cookability_execution": _hashed(
            {
                **common,
                "schema_version": "flavourbench-season1-practical-cookability-execution-v1",
                "counts": {
                    "tasks": 24,
                    "outputs": 24,
                    "independent_cooks_per_output": 2,
                    "kitchen_executions": 48,
                },
                "model_and_condition_blinded": True,
                "ranking_use": "construct-validity-only-not-pooled",
                "execution_records": kitchen_records,
                "output_selection_artifact_sha256": digest("output-selection"),
                "rubric_association_artifact_sha256": digest("rubric-association"),
            }
        ),
    }


def test_readiness_passes_real_contracts_without_inventing_quality_evidence() -> None:
    report = build_report(**_inputs())

    assert report["release_gates"]["prospective_study_design"]["status"] == "pass"
    assert report["release_gates"]["exact_model_contracts"]["status"] == "partial_contract_only"
    assert report["candidate_panel"]["quality_observations"] == 0
    assert report["schema_version"] == "flavourbench-season1-readiness-audit-v13"
    withdrawal_gate = report["release_gates"]["task_candidate_withdrawal_integrity"]
    assert withdrawal_gate["status"] == "pass"
    assert withdrawal_gate["observed"]["withdrawn_candidates_imported"] == 0
    assert report["release_gates"]["statistical_method_validation"]["status"] == "pass"
    assert report["restricted_human_review_qa"]["human_evidence_eligible"] is False
    assert report["restricted_human_review_qa"]["preference_aggregates_republished"] is False
    assert report["restricted_human_review_qa"]["governed_quarantine_tasks"] == 17
    assert report["restricted_human_review_qa"]["reviewed_quarantine_tasks"] == 7
    assert "anonymous_follow_up" not in report
    assert report["decision"]["closed_generation_admission_ready"] is False
    assert report["decision"]["quality_leaderboard_release_ready"] is False
    assert report["release_gates"]["signed_research_archive"]["status"] == "blocked"
    assert report["release_gates"]["post_collection_item_audit"]["status"] == "not_started"
    assert report["release_gates"]["generation_reliability_panel"]["status"] == "not_started"
    assert report["release_gates"]["prompt_sensitivity_audit"]["status"] == "not_started"
    assert report["release_gates"]["practical_cookability_execution"]["status"] == "not_started"


def test_restricted_real_evidence_is_exposed_without_promoting_results() -> None:
    inputs = _inputs()
    paths = {
        "real_human_pilot": ROOT
        / "artifacts/season1/human-review/real-human-pilot-v1/"
        "real-human-pilot-quality-"
        "933b35a81b96a364b805bed5f39c29fd8f79fb6a29868d5d6fd7c152d2557430.json",
        "real_pilot_validators": ROOT
        / "artifacts/season1/validators/real-pilot-v1/"
        "real-arm-validator-audit-"
        "5ff552da8c8ad7a7bcee277bd28f2dd18b4d41f1c3c9425e0707bec41184a40d.json",
        "surface_clean_tasks": ROOT
        / "artifacts/season1/task-validity/development-v2/"
        "development-task-validity-v2-"
        "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846.json",
        "current_model_manifest": ROOT
        / "artifacts/season1/current-quality-run/manifest-v13-evidence-boundary/"
        "flavourbench-openrouter-unranked-"
        "12f411f86c67af5555036851713290bcaf04e1d725bada5af937839753e7db54.json",
        "current_catalog_audit": ROOT
        / "artifacts/season1/current-quality-run/catalog-audit-v2/"
        "current-model-catalog-audit-"
        "9a4507f9c83da65e3e2fe1fd03e147c36ef216dd490a7067bcd79440f1d28947.json",
    }
    inputs.update(
        {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    )

    report = build_report(**inputs)

    assert report["restricted_real_human_pilot"]["primary_human_judgments"] == 29
    assert report["restricted_real_human_pilot"]["paper_use"] is False
    validator_observed = report["real_pilot_deterministic_validation"]["observed"]
    assert validator_observed["real_response_arms"] == 192
    assert validator_observed["evidence_claim_boundary_warnings"] == 9
    assert report["surface_clean_real_human_development_pool"]["counts"][
        "surface_dependency_quarantined"
    ] == 15
    prospective = report["prospective_current_model_manifest"]
    assert prospective["model_count"] == 14
    assert prospective["evidence_protocol"] == "matched_evidence_v2"
    assert prospective["provider_calls_made"] is False
    catalog_audit = report["current_model_catalog_audit"]
    assert catalog_audit["counts"]["freshness_contract_passed"] == 14
    assert catalog_audit["counts"]["quality_observations"] == 0
    assert report["release_gates"]["prospective_roster_catalog_freshness"]["status"] == (
        "pass_catalog_only"
    )
    assert report["decision"]["quality_leaderboard_release_ready"] is False
    assert report["decision"]["evidence_inventory"] == {
        "official_prospective_quality_observations": 0,
        "restricted_real_human_pilot_judgments": 29,
        "restricted_real_model_arms_with_validator_receipts": 192,
        "development_task_validation_packet_tasks": 0,
        "development_independent_blind_validity_records": 0,
        "development_independently_validated_tasks": 0,
        "synthetic_observations": 0,
    }


def test_current_catalog_audit_must_bind_manifest_and_remain_tamper_evident() -> None:
    inputs = _inputs()
    manifest_path = ROOT / (
        "artifacts/season1/current-quality-run/manifest-v13-evidence-boundary/"
        "flavourbench-openrouter-unranked-"
        "12f411f86c67af5555036851713290bcaf04e1d725bada5af937839753e7db54.json"
    )
    audit_path = ROOT / (
        "artifacts/season1/current-quality-run/catalog-audit-v2/"
        "current-model-catalog-audit-"
        "9a4507f9c83da65e3e2fe1fd03e147c36ef216dd490a7067bcd79440f1d28947.json"
    )
    inputs["current_model_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs["current_catalog_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))

    inputs["current_catalog_audit"]["counts"]["quality_observations"] = 1
    with pytest.raises(Season1ReadinessError, match="catalog audit failed closed"):
        build_report(**inputs)

    inputs["current_catalog_audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(inputs["current_catalog_audit"])
    payload.pop("artifact_sha256")
    payload["source_manifest_sha256"] = "0" * 64
    inputs["current_catalog_audit"] = _hashed(payload)
    with pytest.raises(Season1ReadinessError, match="does not bind"):
        build_report(**inputs)


def test_task_bank_and_lineage_remain_hard_release_gates() -> None:
    inputs = _inputs()
    report = build_report(**inputs)

    assert report["release_gates"]["confirmatory_task_bank"]["observed"]["total"] == 0
    validity = report["release_gates"]["three_stage_task_validity"]
    assert validity["required"]["blind_prompt_only_records"] == 480
    assert validity["required"]["reconciled_source_validations"] == 480
    assert validity["required"]["independent_adjudications"] == 240
    assert validity["required"]["completed_human_validity_records"] == 720
    assert report["release_gates"]["epicure_release"]["status"] == "blocked"
    assert report["release_gates"]["construct_blueprint"]["status"] == "blocked"
    assert report["release_gates"]["contamination_replay"]["status"] == "blocked"
    assert report["release_gates"]["validator_calibration"]["status"] == "blocked"
    contamination = report["release_gates"]["contamination_detection_calibration"]
    assert contamination["status"] == "blocked"
    assert contamination["required"] == {
        "cases": 150,
        "precision": 0.95,
        "recall": 0.9,
        "paraphrase_recall": 0.85,
    }
    assert report["release_gates"]["minimum_result_sample"]["required_snapshot_cells"] == [
        "epicure_uplift:expert_independent",
        "epicure_uplift:public",
        "model_arena:expert_independent",
        "model_arena:public",
    ]


def test_robustness_evidence_is_content_addressed_and_fail_closed() -> None:
    inputs = _inputs()
    design_sha256 = inputs["study_design"]["artifact_sha256"]
    inputs["robustness_evidence"] = _robustness_evidence(design_sha256)

    report = build_report(**inputs)

    for gate in (
        "post_collection_item_audit",
        "generation_reliability_panel",
        "prompt_sensitivity_audit",
        "practical_cookability_execution",
    ):
        assert report["release_gates"][gate]["status"] == "pass"

    tampered = _robustness_evidence(design_sha256)
    tampered["generation_reliability_panel"]["counts"]["response_arms"] = 1919
    inputs["robustness_evidence"] = tampered
    report = build_report(**inputs)
    assert report["release_gates"]["generation_reliability_panel"]["status"] == "not_started"
    assert report["decision"]["quality_leaderboard_release_ready"] is False

    malformed = _robustness_evidence(design_sha256)
    malformed["practical_cookability_execution"]["counts"]["kitchen_executions"] = None
    inputs["robustness_evidence"] = malformed
    report = build_report(**inputs)
    assert report["release_gates"]["practical_cookability_execution"]["status"] == "not_started"

    underfilled = _robustness_evidence(design_sha256)
    prompt_payload = dict(underfilled["prompt_sensitivity_audit"])
    prompt_payload.pop("artifact_sha256")
    prompt_payload["arm_records"] = prompt_payload["arm_records"][:-1]
    underfilled["prompt_sensitivity_audit"] = _hashed(prompt_payload)
    inputs["robustness_evidence"] = underfilled
    report = build_report(**inputs)
    assert report["release_gates"]["prompt_sensitivity_audit"]["status"] == "not_started"


def test_method_validation_is_reproduced_and_tampering_blocks_collection() -> None:
    inputs = _inputs()
    inputs["method_validation"]["scenarios"][0]["uplift"]["interval_coverage"] = 1.0

    report = build_report(**inputs)

    assert report["release_gates"]["statistical_method_validation"]["status"] == "blocked"
    assert report["decision"]["closed_generation_admission_ready"] is False


def test_readiness_rejects_unscoped_or_cross_season_control_plane_counts() -> None:
    inputs = _inputs()
    inputs["database"].pop("scope")

    with pytest.raises(Season1ReadinessError, match="scoped to Season 1"):
        build_report(**inputs)

    inputs = _inputs()
    inputs["database"]["scope"]["season_id"] = "another-season-id"
    inputs["database"]["counts"].update(
        {
            "official_battles": 4000,
            "rank_eligible_battles": 4000,
            "valid_public_preferences": 4000,
            "valid_expert_judgments": 1920,
        }
    )

    with pytest.raises(Season1ReadinessError, match="identity does not match"):
        build_report(**inputs)


def test_raw_vote_counts_cannot_satisfy_the_snapshot_bound_result_gate() -> None:
    inputs = _inputs()
    counts = inputs["database"]["counts"]
    counts.update(
        {
            "official_battles": 4000,
            "rank_eligible_battles": 4000,
            "valid_public_preferences": 4000,
            "valid_expert_judgments": 1919,
        }
    )

    report = build_report(**inputs)

    floor = report["release_gates"]["minimum_result_sample"]
    assert floor["status"] == "not_started"
    assert floor["accepted_snapshot_cells"] == []
    assert floor["source"] == "content-addressed canonical analysis snapshots"


def test_readiness_rejects_human_review_evidence_without_fail_closed_boundary() -> None:
    inputs = _inputs()
    human = _human_payload()
    human["claim_boundary"]["paper_use"] = True
    inputs["human"] = _hashed(human)

    with pytest.raises(Season1ReadinessError, match="fail-closed"):
        build_report(**inputs)


def test_readiness_rejects_superseded_human_review_schema() -> None:
    inputs = _inputs()
    human = _human_payload()
    human["schema_version"] = "flavourbench-human-review-operational-qa-v2"
    inputs["human"] = _hashed(human)

    with pytest.raises(Season1ReadinessError, match="fail-closed"):
        build_report(**inputs)


def test_latest_human_report_uses_only_valid_v3_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "operational-qa"
    output_dir.mkdir()
    old_payload = _human_payload()
    old_payload["schema_version"] = "flavourbench-human-review-operational-qa-v2"
    old = _hashed(old_payload)
    current = _hashed(_human_payload())
    old_path = output_dir / f"restricted-operational-qa-{old['artifact_sha256']}.json"
    current_path = output_dir / f"restricted-operational-qa-{current['artifact_sha256']}.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    assert latest_human_report(tmp_path) == current_path


def test_research_archive_requires_exact_snapshots_and_verified_bytes() -> None:
    snapshot_ids = ["snapshot-a", "snapshot-b", "snapshot-c", "snapshot-d"]
    snapshot_set_sha256 = hashlib.sha256(
        json.dumps({"snapshot_ids": snapshot_ids}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "snapshot_set_sha256": snapshot_set_sha256,
        "requirements_lock_sha256": "b" * 64,
        "build_image_digest": f"sha256:{'c' * 64}",
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    archive = {
        "id": "archive-a",
        "season_id": "season-1-id",
        "archive_class": "internal_official",
        "schema_version": "flavourbench-research-release-v1",
        "snapshot_ids_json": snapshot_ids,
        "snapshot_set_sha256": snapshot_set_sha256,
        "manifest_json": manifest,
        "manifest_sha256": manifest_sha256,
        "archive_sha256": "a" * 64,
        "requirements_lock_sha256": "b" * 64,
        "build_image_digest": f"sha256:{'c' * 64}",
        "signature_algorithm": "Ed25519",
        "size_bytes": 10,
        "member_count": 2,
        "source_date_epoch": 0,
        "verification": {
            "metadata_signature_valid": True,
            "archive_file_verified": True,
            "inventory_valid": True,
            "reproducible_metadata_valid": True,
            "verified_manifest_sha256": manifest_sha256,
        },
    }
    assert valid_research_archive(
        archive,
        season_id="season-1-id",
        snapshot_ids=snapshot_ids,
    )
    robustness_digests = {"generation_reliability_panel": "d" * 64}
    assert not valid_research_archive(
        archive,
        season_id="season-1-id",
        snapshot_ids=snapshot_ids,
        robustness_evidence_sha256=robustness_digests,
    )
    manifest["robustness_evidence_sha256"] = robustness_digests
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    archive["manifest_sha256"] = manifest_sha256
    archive["verification"]["verified_manifest_sha256"] = manifest_sha256
    assert valid_research_archive(
        archive,
        season_id="season-1-id",
        snapshot_ids=snapshot_ids,
        robustness_evidence_sha256=robustness_digests,
    )
    archive["verification"]["archive_file_verified"] = False
    assert not valid_research_archive(
        archive,
        season_id="season-1-id",
        snapshot_ids=snapshot_ids,
    )
