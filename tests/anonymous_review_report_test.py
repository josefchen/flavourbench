from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.anonymous_review_report import (
    CONTROL_PLANE_SQL,
    AnonymousReviewReportError,
    attach_candidate_evidence,
    attach_replacement_candidate_evidence,
    build_report,
    canonical_bytes,
)
from flavourbench.expert_calibration import TASK_SCOPE_QUARANTINE, TASK_SCOPE_REVIEW_SHA256

HISTORICAL_POOL_SHA256 = "c" * 64
LIVE_HISTORICAL_POOL_SHA256 = "12a5558def32bacaf5e34bd81592253af517ada559985de48e4ae6398ef7cd57"
LIVE_REPLACEMENT_POOL_SHA256 = "94e917b6c202eb49953f3a8c22f897301eaa7ffba47116b83c915d17a6850b69"


def _rubric(score: int) -> dict:
    dimensions = {
        "task_completion": score,
        "constraint_compliance": score,
        "coherence": score,
        "sensory_promise": score,
        "cookability": score,
        "clarity": score,
        "originality": score,
        "evidence_use": score,
        "calibration": score,
    }
    return {"left": dimensions, "right": dimensions, "review_metadata": {}}


def _snapshot() -> dict:
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
        "battle_manifest_sha256": HISTORICAL_POOL_SHA256,
        "left_failure_tags": [],
        "right_failure_tags": [],
        "model_ids": ["model/a"],
        "rubric": _rubric(4),
        "primary_rubric": None,
        "primary_choice": None,
    }
    records = [
        {
            **base,
            "battle_id": "one",
            "task_public_id": "task-one",
            "mode": "primary",
            "track": "epicure_uplift",
            "category": "substitution",
            "choice": "left",
            "left_condition": "epicure_on",
            "right_condition": "epicure_off",
        },
        {
            **base,
            "battle_id": "two",
            "task_public_id": "task-two",
            "mode": "primary",
            "track": "epicure_uplift",
            "category": "composition",
            "choice": "tie",
            "left_condition": "epicure_off",
            "right_condition": "epicure_on",
        },
        {
            **base,
            "battle_id": "three",
            "task_public_id": "task-three",
            "mode": "primary",
            "track": "epicure_uplift",
            "category": "cookability",
            "choice": "right",
            "left_condition": "epicure_on",
            "right_condition": "epicure_off",
        },
        {
            **base,
            "battle_id": "one",
            "task_public_id": "task-one",
            "mode": "reliability_repeat",
            "track": "epicure_uplift",
            "category": "substitution",
            "choice": "left",
            "primary_choice": "left",
            "primary_rubric": _rubric(5),
            "left_condition": "epicure_on",
            "right_condition": "epicure_off",
        },
    ]
    return {
        "schema_revision": "0012",
        "reviewer": {
            "cohort": "expert_independent",
            "qualification_verified": False,
            "identity_collection_prohibited": True,
            "consent_reference_present": True,
            "independent_expert_validation_claim": False,
        },
        "session": {
            "opened_at": "2026-07-31T09:00:00+00:00",
            "target_presentations": 5,
            "protocol_version": "v4",
            "protocol_sha256": "a" * 64,
            "admission_pathway": "anonymous_external_rater_v1",
            "source_pool_sha256": HISTORICAL_POOL_SHA256,
        },
        "source_pool": {
            "candidate_pack_sha256": HISTORICAL_POOL_SHA256,
            "battle_count": 32,
            "source_arm_count": 64,
            "provider_call_count": 113,
            "tool_call_count": 80,
            "synthetic_arm_count": 0,
            "rank_eligible_battle_count": 0,
            "data_stratum": "development",
            "run_class": "pilot",
        },
        "reviewer_self_report": {
            "recognized_reliability_repeats": 1,
            "reported_potential_safety_hazards": 0,
        },
        "restriction": {"evidence_status": "restricted_operational_qa"},
        "consent_evidence": {
            "status": "inactive_or_unapproved",
            "collection_permitted": False,
        },
        "replacement_candidate": {
            "artifact_sha256": "b" * 64,
            "schema_version": "candidate-v4",
            "observed": {
                "candidate_pairs": 32,
                "source_arms": 64,
                "real_provider_calls": 64,
                "real_epicure_calls": 32,
                "synthetic_arms": 0,
            },
        },
        "candidate_evidence": {
            "real_provider_calls": 113,
            "real_epicure_calls": 80,
            "successful_real_epicure_calls": 73,
        },
        "candidate_source": {
            "path": "artifacts/expert-calibration/candidate-v1/candidate.json",
            "file_sha256": "d" * 64,
            "artifact_sha256": HISTORICAL_POOL_SHA256,
            "schema_version": "flavourbench-expert-calibration-candidate-v1",
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


def test_build_report_preserves_claim_boundary_and_uplift_orientation() -> None:
    report = build_report(_snapshot())

    assert report["review_progress"]["completed_presentations"] == 4
    assert report["review_progress"]["unique_primary_judgments"] == 3
    assert report["restricted_diagnostic_preferences"]["all_primary_condition_normalized"] == {
        "n": 3,
        "epicure_wins": 1,
        "ties": 1,
        "epicure_losses": 1,
        "both_bad": 0,
        "effective_n_excluding_both_bad": 3,
        "tie_adjusted_epicure_preference_share": 0.5,
    }
    assert report["repeat_check"]["observed_exact_agreement"] == 1.0
    assert report["repeat_check"]["dimension_comparisons"] == 18
    assert report["repeat_check"]["reliability_interpretable"] is False
    assert report["claim_boundary"]["rank_eligible"] is False
    assert report["claim_boundary"]["independent_expert_validation"] is False
    assert report["claim_boundary"]["paper_use"] is False


def test_report_is_identity_minimized() -> None:
    report = build_report(_snapshot())
    serialized = str(report)

    assert "battle_id" not in serialized
    assert "reviewer_id" not in serialized
    assert "model/a" not in serialized
    assert report["source_pool"]["synthetic_arms"] == 0
    assert report["source_pool"]["provider_calls"] == 113
    assert report["source_pool"]["retained_provider_generation_ids"] == 77


def test_control_plane_query_binds_pool_and_aggregates_to_historical_session() -> None:
    assert "restriction.event_type = 'expert_review_batch_restricted'" in CONTROL_PLANE_SQL
    assert "= 'restricted_operational_qa'" in CONTROL_PLANE_SQL
    assert (
        "event.entity_id = review_session.payload ->> 'anonymous_external_pool_sha256'"
        in CONTROL_PLANE_SQL
    )
    assert "'source_pool_sha256', payload ->> 'anonymous_external_pool_sha256'" in CONTROL_PLANE_SQL
    assert "battles.season_id = pool_season.season_id" in CONTROL_PLANE_SQL
    assert "tasks.season_id = pool_season.season_id" in CONTROL_PLANE_SQL


def test_live_snapshot_attachments_keep_historical_candidate_separate_from_v11() -> None:
    project_root = Path(__file__).resolve().parents[1]
    candidate_root = project_root / "artifacts" / "expert-calibration"
    snapshot = _snapshot()
    snapshot["session"]["source_pool_sha256"] = LIVE_HISTORICAL_POOL_SHA256
    snapshot["source_pool"]["candidate_pack_sha256"] = LIVE_HISTORICAL_POOL_SHA256
    for record in snapshot["records"]:
        record["battle_manifest_sha256"] = LIVE_HISTORICAL_POOL_SHA256
    snapshot.pop("candidate_evidence")
    snapshot.pop("candidate_source")
    snapshot.pop("replacement_candidate")
    snapshot.pop("scope_governance")

    attach_candidate_evidence(snapshot, candidate_root)
    attach_replacement_candidate_evidence(snapshot, candidate_root / "candidate-v11")
    report = build_report(snapshot)

    assert snapshot["candidate_source"]["artifact_sha256"] == LIVE_HISTORICAL_POOL_SHA256
    assert (
        snapshot["candidate_source"]["schema_version"]
        == "flavourbench-expert-calibration-candidate-v1"
    )
    assert (
        report["source_pool"]["historical_review_session_pool_sha256"]
        == LIVE_HISTORICAL_POOL_SHA256
    )
    assert (
        report["completion_audit"]["replacement_candidate"]["artifact_sha256"]
        == LIVE_REPLACEMENT_POOL_SHA256
    )
    assert report["scope_audit"]["governed_quarantine_tasks"] == 17
    assert LIVE_REPLACEMENT_POOL_SHA256 not in json.dumps(report["source_pool"])


def test_candidate_attachment_rejects_latest_pool_misattributed_to_historical_session() -> None:
    snapshot = _snapshot()
    snapshot["source_pool"]["candidate_pack_sha256"] = LIVE_REPLACEMENT_POOL_SHA256

    with pytest.raises(
        AnonymousReviewReportError,
        match="does not match the historical review session",
    ):
        attach_candidate_evidence(snapshot)


def test_report_rejects_review_record_from_replacement_pool() -> None:
    snapshot = _snapshot()
    snapshot["records"][0]["battle_manifest_sha256"] = LIVE_REPLACEMENT_POOL_SHA256

    with pytest.raises(AnonymousReviewReportError, match="historical source pool"):
        build_report(snapshot)


def test_candidate_attachment_verifies_versioned_schema(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate-v2"
    candidate_dir.mkdir()
    payload = {
        "schema_version": "flavourbench-expert-calibration-candidate-v11",
        "observed": {"candidate_pairs": 32},
    }
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    path = candidate_dir / f"candidate-pack-{digest}.json"
    path.write_text(
        json.dumps({**payload, "artifact_sha256": digest}, sort_keys=True),
        encoding="utf-8",
    )
    snapshot = _snapshot()
    snapshot["session"]["source_pool_sha256"] = digest
    snapshot["source_pool"]["candidate_pack_sha256"] = digest

    with pytest.raises(AnonymousReviewReportError, match="digest or schema is invalid"):
        attach_candidate_evidence(snapshot, tmp_path)
