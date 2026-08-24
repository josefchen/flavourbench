from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from flavourbench.anonymous_pool_rotation import (
    REQUIRED_FRONTIER_FAMILY_COUNTS,
    REQUIRED_FRONTIER_SOURCE_CLASS,
    ROTATION_CORRECTION_EVENT_TYPE,
    ROTATION_EVENT_TYPE,
    rotate_anonymous_pool,
)
from flavourbench.anonymous_reviewer_control import (
    anonymous_pool_reconsented,
    append_pool_reconsent,
)
from flavourbench.models import Base, ExpertReviewer, RunEvent

ROOT = Path(__file__).resolve().parents[1]
REPLACEMENT_SHA256 = "94e917b6c202eb49953f3a8c22f897301eaa7ffba47116b83c915d17a6850b69"
REPLACEMENT_PATH = (
    ROOT
    / "artifacts"
    / "expert-calibration"
    / "candidate-v11"
    / f"candidate-pack-{REPLACEMENT_SHA256}.json"
)
REQUIRED_FRONTIER_SHA256 = (
    "f4daaef029dfc46d739be479d601938eb75ee73d957b73bd5607762dc6a8e9b2"
)
REQUIRED_FRONTIER_PATH = (
    ROOT
    / "artifacts"
    / "season1"
    / "current-quality-run"
    / "pilot-v24-required-epicure"
    / "review-pool"
    / f"required-frontier-review-pool-{REQUIRED_FRONTIER_SHA256}.json"
)
PRIOR_SHA256 = "2" * 64


def _reviewer(*, pool_sha256: str, activation_sha256: str | None) -> ExpertReviewer:
    profile = {
        "admission_pathway": "anonymous_external_rater",
        "anonymous_external_admission_status": "active",
        "anonymous_external_pool_sha256": pool_sha256,
        "consent_document_sha256": "1" * 64,
        "identity_collection_prohibited": True,
        "independence_basis": "reviewer_self_attestation",
        "qualification_basis": "reviewer_self_attestation_unverified",
        "independent_expert_validation_claim": False,
    }
    if activation_sha256 is not None:
        profile["anonymous_external_pool_activation_sha256"] = activation_sha256
    return ExpertReviewer(
        id="anonymous-rotation-reviewer",
        reviewer_code="fbr-anon-rotation-fixture",
        invitation_sha256="3" * 64,
        qualification_json=["substitution", "composition", "cookability", "evidence"],
        qualification_verified=False,
        cohort="expert_independent",
        profile_json=profile,
        batch_reveal_only=True,
        active=True,
    )


def test_rotation_without_a_pool_matched_session_records_honest_no_session_state() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        reviewer = _reviewer(pool_sha256=PRIOR_SHA256, activation_sha256="4" * 64)
        session.add_all(
            [
                reviewer,
                RunEvent(
                    entity_type="author_evaluator_pool",
                    entity_id=REPLACEMENT_SHA256,
                    event_type="author_evaluator_pool_imported",
                    payload_json={"candidate_pack_sha256": REPLACEMENT_SHA256},
                ),
                RunEvent(
                    entity_type="expert_review_session",
                    entity_id="unrelated-session",
                    event_type="expert_review_session_opened",
                    payload_json={
                        "reviewer_id": reviewer.id,
                        "anonymous_external_pool_sha256": "5" * 64,
                    },
                ),
                RunEvent(
                    entity_type="expert_review_session",
                    entity_id="unrelated-session",
                    event_type="expert_review_batch_restricted",
                    payload_json={"research_use": False},
                ),
            ]
        )
        session.flush()
        _, appended = append_pool_reconsent(session, reviewer)
        assert appended is True
        assert anonymous_pool_reconsented(session, reviewer) is True
        session.commit()

        result = rotate_anonymous_pool(session, candidate_path=REPLACEMENT_PATH)
        assert result["idempotent"] is False
        assert result["reviewEnabled"] is False
        rotation = session.get(RunEvent, result["eventId"])
        assert rotation is not None
        assert rotation.payload_json["prior_review_session_id"] is None
        assert rotation.payload_json["prior_review_session_binding"] == "none_for_prior_pool"
        assert rotation.payload_json["prior_batch_status"] == "no_review_session_for_prior_pool"
        session.refresh(reviewer)
        assert reviewer.profile_json["anonymous_external_pool_sha256"] == REPLACEMENT_SHA256
        assert reviewer.profile_json["prior_restricted_review_session_id"] is None
        assert anonymous_pool_reconsented(session, reviewer) is False

        repeated = rotate_anonymous_pool(session, candidate_path=REPLACEMENT_PATH)
        assert repeated["idempotent"] is True
        assert repeated["correctionEventId"] is None
        assert (
            session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.event_type == ROTATION_EVENT_TYPE)
            )
            == 1
        )


def test_legacy_misbound_rotation_gets_one_append_only_correction() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        reviewer = _reviewer(pool_sha256=REPLACEMENT_SHA256, activation_sha256=None)
        original_payload = {
            "prior_pool_sha256": PRIOR_SHA256,
            "replacement_pool_sha256": REPLACEMENT_SHA256,
            "prior_review_session_id": "session-for-different-pool",
            "prior_batch_status": "restricted_operational_qa",
            "review_enabled": False,
            "raw_prior_records_preserved": True,
        }
        rotation = RunEvent(
            id="legacy-misbound-rotation",
            entity_type="expert_reviewer",
            entity_id=reviewer.id,
            event_type=ROTATION_EVENT_TYPE,
            payload_json=deepcopy(original_payload),
        )
        session.add_all(
            [
                reviewer,
                rotation,
                RunEvent(
                    entity_type="expert_review_session",
                    entity_id="session-for-different-pool",
                    event_type="expert_review_session_opened",
                    payload_json={
                        "reviewer_id": reviewer.id,
                        "anonymous_external_pool_sha256": "6" * 64,
                    },
                ),
                RunEvent(
                    entity_type="expert_review_session",
                    entity_id="session-for-different-pool",
                    event_type="expert_review_batch_restricted",
                    payload_json={"research_use": False},
                ),
            ]
        )
        session.commit()

        repaired = rotate_anonymous_pool(session, candidate_path=REPLACEMENT_PATH)
        assert repaired["idempotent"] is True
        assert repaired["correctionAppended"] is True
        correction = session.get(RunEvent, repaired["correctionEventId"])
        assert correction is not None
        assert correction.event_type == ROTATION_CORRECTION_EVENT_TYPE
        assert correction.payload_json["supersedes_rotation_event_id"] == rotation.id
        assert correction.payload_json["corrected_prior_review_session_id"] is None
        assert (
            correction.payload_json["corrected_prior_session_binding"]
            == "none_for_prior_pool"
        )
        assert "prior_review_session_pool_mismatch" in correction.payload_json[
            "correction_reasons"
        ]
        assert "missing_pool_activation_epoch" in correction.payload_json[
            "correction_reasons"
        ]
        session.refresh(rotation)
        assert rotation.payload_json == original_payload

        repeated = rotate_anonymous_pool(session, candidate_path=REPLACEMENT_PATH)
        assert repeated["correctionAppended"] is False
        assert repeated["correctionEventId"] == correction.id
        assert (
            session.scalar(
                select(func.count())
                .select_from(RunEvent)
                .where(RunEvent.event_type == ROTATION_CORRECTION_EVENT_TYPE)
            )
            == 1
        )


def test_rotation_accepts_required_frontier_pool_and_keeps_observed_family_cells() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        reviewer = _reviewer(pool_sha256=PRIOR_SHA256, activation_sha256="4" * 64)
        session.add_all(
            [
                reviewer,
                RunEvent(
                    entity_type="current_frontier_review_pool",
                    entity_id=REQUIRED_FRONTIER_SHA256,
                    event_type="current_frontier_review_pool_imported",
                    payload_json={
                        "review_pool_sha256": REQUIRED_FRONTIER_SHA256,
                        "counts": {
                            "tasks": 4,
                            "battles": 43,
                            "arms": 86,
                            "toolCalls": 182,
                        },
                        "synthetic_arm_count": 0,
                        "rank_eligible_battle_count": 0,
                    },
                ),
            ]
        )
        session.commit()

        result = rotate_anonymous_pool(session, candidate_path=REQUIRED_FRONTIER_PATH)
        assert result["idempotent"] is False
        assert result["replacementPoolSha256"] == REQUIRED_FRONTIER_SHA256
        assert result["reviewEnabled"] is False

        session.refresh(reviewer)
        profile = reviewer.profile_json
        assert profile["anonymous_external_pool_sha256"] == REQUIRED_FRONTIER_SHA256
        assert profile["anonymous_external_primary_judgments"] == 43
        assert profile["anonymous_external_primary_by_family"] == (
            REQUIRED_FRONTIER_FAMILY_COUNTS
        )
        assert profile["anonymous_external_reliability_repeats"] == 5
        assert profile["anonymous_external_target_judgments"] == 48
        assert profile["calibration_candidate"]["source_class"] == (
            REQUIRED_FRONTIER_SOURCE_CLASS
        )
        assert profile["calibration_candidate"]["synthetic_arms"] == 0
        assert anonymous_pool_reconsented(session, reviewer) is False

        repeated = rotate_anonymous_pool(session, candidate_path=REQUIRED_FRONTIER_PATH)
        assert repeated["idempotent"] is True
        assert repeated["eventId"] == result["eventId"]
        assert repeated["reviewEnabled"] is False
