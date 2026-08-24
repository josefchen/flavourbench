from __future__ import annotations

import copy

import pytest

from flavourbench.expert_calibration import TASK_SCOPE_QUARANTINE, TASK_SCOPE_REVIEW_SHA256
from flavourbench.real_human_pilot import RealHumanPilotError, build_pilot_report


def _rubric(left: int, right: int) -> dict:
    dimensions = (
        "task_completion",
        "constraint_compliance",
        "coherence",
        "sensory_promise",
        "cookability",
        "clarity",
        "originality",
        "evidence_use",
        "calibration",
    )
    return {
        "left": {dimension: left for dimension in dimensions},
        "right": {dimension: right for dimension in dimensions},
        "review_metadata": {},
    }


def _snapshot() -> dict:
    historical = "c" * 64
    base = {
        "created_at": "2026-07-31T10:00:00+00:00",
        "duration_ms": 60_000,
        "answer_review_duration_ms": 40_000,
        "speed_flag": False,
        "task_validity": "valid",
        "left_status": "complete",
        "right_status": "complete",
        "left_finish_reason": "stop",
        "right_finish_reason": "stop",
        "battle_manifest_sha256": historical,
        "left_failure_tags": [],
        "right_failure_tags": [],
        "primary_rubric": None,
        "primary_choice": None,
        "mode": "primary",
        "track": "epicure_uplift",
    }
    scope_task = sorted(TASK_SCOPE_QUARANTINE)[0]
    records = [
        {
            **base,
            "battle_id": "one",
            "task_public_id": "general-one",
            "category": "substitution",
            "choice": "left",
            "left_condition": "epicure_on",
            "right_condition": "epicure_off",
            "model_ids": ["model/a", "model/a"],
            "rubric": _rubric(5, 3),
        },
        {
            **base,
            "battle_id": "two",
            "task_public_id": scope_task,
            "category": "composition",
            "choice": "tie",
            "left_condition": "epicure_off",
            "right_condition": "epicure_on",
            "model_ids": ["model/b", "model/b"],
            "rubric": _rubric(4, 4),
        },
        {
            **base,
            "battle_id": "three",
            "task_public_id": "general-three",
            "category": "cookability",
            "choice": "left",
            "left_condition": "epicure_off",
            "right_condition": "epicure_on",
            "left_finish_reason": "length",
            "model_ids": ["model/a", "model/a"],
            "rubric": _rubric(4, 2),
        },
    ]
    return {
        "schema_revision": "0022",
        "reviewer": {
            "cohort": "expert_independent",
            "qualification_verified": False,
            "identity_collection_prohibited": True,
            "consent_reference_present": True,
            "consent_document_sha256": "e" * 64,
            "independent_expert_validation_claim": False,
        },
        "session": {
            "opened_at": "2026-07-31T09:00:00+00:00",
            "target_presentations": 4,
            "protocol_version": "v4",
            "protocol_sha256": "a" * 64,
            "admission_pathway": "anonymous_external_rater",
            "source_pool_sha256": historical,
        },
        "source_pool": {
            "candidate_pack_sha256": historical,
            "battle_count": 32,
            "source_arm_count": 64,
            "synthetic_arm_count": 0,
            "rank_eligible_battle_count": 0,
            "data_stratum": "development",
            "run_class": "pilot",
        },
        "reviewer_self_report": {
            "recognized_reliability_repeats": 0,
            "reported_potential_safety_hazards": 0,
        },
        "restriction": {"evidence_status": "restricted_operational_qa"},
        "consent_evidence": {
            "status": "inactive_or_unapproved",
            "collection_permitted": False,
        },
        "candidate_evidence": {
            "real_provider_calls": 113,
            "real_epicure_calls": 80,
            "successful_real_epicure_calls": 73,
        },
        "candidate_source": {
            "path": "candidate-v1/candidate.json",
            "file_sha256": "d" * 64,
            "artifact_sha256": historical,
            "schema_version": "flavourbench-expert-calibration-candidate-v1",
        },
        "replacement_candidate": {
            "artifact_sha256": "b" * 64,
            "schema_version": "flavourbench-expert-calibration-candidate-v11",
            "observed": {
                "candidate_pairs": 32,
                "source_arms": 64,
                "real_provider_calls": 110,
                "real_epicure_calls": 89,
                "synthetic_arms": 0,
            },
        },
        "scope_governance": {
            "artifact_sha256": TASK_SCOPE_REVIEW_SHA256,
            "schema_version": "flavourbench-specialist-scope-review-v1",
            "quarantined_task_count": len(TASK_SCOPE_QUARANTINE),
        },
        "retained_provider_generation_ids": 77,
        "task_state": {
            "total": 32,
            "calibration": 32,
            "synthetic": 0,
            "source_classes": {"paid_real_legacy_pilot_quarantined_from_season1": 32},
        },
        "records": records,
    }


def test_report_exposes_real_bounded_signal_without_promoting_it() -> None:
    report = build_pilot_report(_snapshot())

    assert report["real_data_inventory"] == {
        "candidate_pairs": 32,
        "paid_source_arms": 64,
        "real_provider_calls": 113,
        "real_epicure_calls": 80,
        "synthetic_arms": 0,
        "primary_human_judgments": 3,
        "finish_clean_primary_judgments": 2,
        "distinct_models": 2,
        "distinct_task_families": 3,
    }
    validity = report["task_validity_diagnostic"]
    assert validity["rater_valid"] == 3
    assert validity["later_governance_scope_disagreements"] == 1
    assert validity["post_scope_agreement_rate"] == pytest.approx(2 / 3, abs=0.0001)
    preference = report["finish_clean_uplift"]["overall"]["preference"]
    assert preference["epicure_wins"] == 1
    assert preference["ties"] == 1
    assert preference["epicure_losses"] == 0
    assert report["claim_boundary"]["paper_use"] is False
    assert report["claim_boundary"]["official_leaderboard_use"] is False
    assert report["coverage"]["model_quality_ranking_permitted"] is False


def test_report_normalizes_rubrics_to_epicure_condition() -> None:
    report = build_pilot_report(_snapshot())
    deltas = report["finish_clean_uplift"]["overall"]["rubric_deltas"]

    assert deltas["task_completion"]["paired_scores"] == 2
    assert deltas["task_completion"]["mean_on_minus_off"] == 1.0
    assert set(report["finish_clean_uplift"]["by_exact_model"]) == {"model/a", "model/b"}


def test_report_rejects_cross_model_uplift_pair() -> None:
    snapshot = copy.deepcopy(_snapshot())
    snapshot["records"][0]["model_ids"] = ["model/a", "model/b"]

    with pytest.raises(RealHumanPilotError, match="one exact model identity"):
        build_pilot_report(snapshot)


def test_report_rejects_model_arena_record_in_uplift_session() -> None:
    snapshot = copy.deepcopy(_snapshot())
    snapshot["records"][0]["track"] = "model_arena"

    with pytest.raises(RealHumanPilotError, match="only uplift pairs"):
        build_pilot_report(snapshot)
