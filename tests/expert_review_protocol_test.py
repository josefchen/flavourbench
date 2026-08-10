from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from flavourbench.config import get_settings
from flavourbench.consent_documents import ConsentDocumentResolution
from flavourbench.database import SessionLocal
from flavourbench.engine import run_worker_once
from flavourbench.expert_review import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    REQUIRED_ACKNOWLEDGEMENTS,
    RUBRIC_DIMENSIONS,
    author_evaluator_workload_cell_targets,
    canonical_sha256,
    normalize_choice,
    normalize_rubric,
    protocol_payload,
    reliability_summary,
    validate_acknowledgements,
    workload_cell_targets,
)
from flavourbench.main import (
    _expert_consent_document_active,
    _record_expert_task_scope_events,
    _review_fatigue_status,
    _reviewer_claim_boundary,
    app,
)
from flavourbench.models import Battle, ExpertReviewer, Incident, RunEvent, Vote
from flavourbench.schemas import ExpertReviewCreate, ExpertTaskAssessmentCreate, VoteCreate
from flavourbench.service_ranking import _task_validity_admissible

SERVICE_HEADERS = {
    "X-FlavourBench-Service-Token": "test-service-token",
    "X-FlavourBench-Pseudonym": "9" * 64,
}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}
ACTIVE_EXPERT_CONSENT_SHA256 = get_settings().active_expert_consent_sha256s[0]


def _arm_scores(score: int = 4) -> dict[str, int]:
    return {dimension: score for dimension in RUBRIC_DIMENSIONS}


def _review_body(*, task_validity: str = "valid") -> dict[str, object]:
    return {
        "choice": "left",
        "reasonTags": [],
        "rubric": {
            "rubric_version": PROTOCOL_VERSION,
            "left": _arm_scores(4),
            "right": _arm_scores(3),
            "review_metadata": {
                "confidence": 4,
                "task_validity": task_validity,
                "task_issue_tags": [] if task_validity == "valid" else ["missing_context"],
                "task_note": "",
                "answerability": "answerable",
                "family_fit": "in_family",
                "scope_eligibility": "general_track",
                "specialist_domains": [],
                "practical_check": "reasoned_only",
                "comparative_rationale": (
                    "Answer A gives more specific culinary reasoning and practical execution cues."
                ),
            },
        },
    }


def _specialist_safety_review_body() -> dict[str, object]:
    body = _review_body()
    metadata = body["rubric"]["review_metadata"]
    metadata["scope_eligibility"] = "specialist_track"
    metadata["specialist_domains"] = ["food_safety"]
    metadata["left_failure_tags"] = ["safety_hazard"]
    return body


def _calibration_candidate_body() -> dict[str, object]:
    return {
        "candidate_pack_sha256": "8" * 64,
        "identity_commitment_sha256": "9" * 64,
        "source_class": "paid_real_legacy_pilot_quarantined_from_season1",
        "candidate_pack_reference": "artifact://expert-calibration/candidate-v1",
        "candidate_pairs": 32,
        "candidate_pairs_by_family": {
            "substitution": 8,
            "composition": 8,
            "cookability": 8,
            "evidence": 8,
        },
        "source_arms": 64,
        "real_provider_calls": 113,
        "real_epicure_calls": 80,
        "successful_real_epicure_calls": 73,
        "synthetic_arms": 0,
        "rank_eligible": False,
        "status": "candidate_pending_independent_gold_adjudication",
    }


def test_protocol_is_content_addressed_and_requires_exact_acknowledgements() -> None:
    payload = protocol_payload()
    assert payload["protocolVersion"] == PROTOCOL_VERSION
    assert len(payload["rubricScale"]["dimensions"]) == 9
    assert len(PROTOCOL_SHA256) == 64
    assert "unsafe_or_impractical" not in payload["responseFailureTags"]
    assert "invented_evidence" not in payload["responseFailureTags"]
    assert validate_acknowledgements(sorted(REQUIRED_ACKNOWLEDGEMENTS)) == sorted(
        REQUIRED_ACKNOWLEDGEMENTS
    )
    with pytest.raises(ValueError, match="missing"):
        validate_acknowledgements(sorted(REQUIRED_ACKNOWLEDGEMENTS - {"conflict_disclosed"}))


@pytest.mark.parametrize("legacy_tag", ("unsafe_or_impractical", "invented_evidence"))
def test_legacy_combined_failure_tags_are_rejected_prospectively(legacy_tag: str) -> None:
    public_payload = {"choice": "left", "reasonTags": [legacy_tag]}
    with pytest.raises(ValueError, match="unsupported reason tag"):
        VoteCreate.model_validate(public_payload)

    expert_payload = _review_body()
    expert_payload["reasonTags"] = [legacy_tag]
    with pytest.raises(ValueError, match="unsupported reason tag"):
        ExpertReviewCreate.model_validate(expert_payload)

    arm_payload = _review_body()
    arm_payload["rubric"]["review_metadata"]["left_failure_tags"] = [legacy_tag]
    with pytest.raises(ValueError, match="unsupported response failure tag"):
        ExpertReviewCreate.model_validate(arm_payload)


def test_legacy_expert_routes_are_retired() -> None:
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "legacy-route-retirement-fixture",
                "qualified_families": ["cookability"],
                "qualification_reference": "verified-practice",
                "qualification_verified": True,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": "no-conflict",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0.9,
                "compensation_reference": "unpaid-test",
            },
        )
        assert invited.status_code == 200, invited.text
        expert_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {invited.json()['invitation']}",
        }
        assignment = client.get("/v1/expert/assignments/next", headers=expert_headers)
        assert assignment.status_code == 410

        vote = client.post(
            "/v1/expert/battles/00000000-0000-0000-0000-000000000000/votes",
            json=_review_body(),
            headers={**expert_headers, "Idempotency-Key": "retired-route-vote"},
        )
        assert vote.status_code == 410


def test_public_scope_admission_requires_a_verified_independent_expert() -> None:
    battle = Battle(
        id="scope-admission-public-battle",
        data_stratum="public_freeform",
        controlled_run_id=None,
    )
    reviewer = ExpertReviewer(
        id="scope-admission-independent-reviewer",
        reviewer_code="scope-admission-independent-reviewer",
        invitation_sha256="a" * 64,
        qualification_json=["cookability"],
        qualification_verified=True,
        cohort="expert_independent",
        profile_json={
            "affiliation_class": "independent_external",
            "calibration_candidate": {
                "candidate_pack_sha256": "8" * 64,
                "candidate_pairs": 32,
                "source_arms": 64,
                "real_provider_calls": 113,
                "successful_real_epicure_calls": 73,
                "synthetic_arms": 0,
                "rank_eligible": False,
            },
            "qualification_reference": "verified-practice",
            "conflict_disclosure_reference": "no-conflict",
            "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
            "training_material_sha256": "2" * 64,
            "calibration_set_sha256": "3" * 64,
            "calibration_item_count": 20,
            "calibration_gold_adjudicator_count": 2,
            "calibration_accuracy": 0.9,
            "admission_decision_reference": "scope-admission-decision",
            "admission_decision_sha256": "4" * 64,
        },
    )
    with TestClient(app):
        with SessionLocal() as session:
            evidence_fields = (
                "qualification_reference",
                "conflict_disclosure_reference",
                "consent_document_sha256",
                "training_material_sha256",
                "calibration_set_sha256",
                "calibration_item_count",
                "calibration_gold_adjudicator_count",
                "calibration_accuracy",
                "admission_decision_reference",
                "admission_decision_sha256",
            )
            candidate = reviewer.profile_json["calibration_candidate"]
            session.add(reviewer)
            session.add_all(
                [
                    RunEvent(
                        entity_type="expert_reviewer",
                        entity_id=reviewer.id,
                        event_type="expert_calibration_candidate_registered",
                        payload_json={
                            "reviewer_code": reviewer.reviewer_code,
                            "cohort": reviewer.cohort,
                            "candidate": candidate,
                            "candidate_record_sha256": canonical_sha256(candidate),
                        },
                    ),
                RunEvent(
                    entity_type="expert_reviewer",
                    entity_id=reviewer.id,
                    event_type="expert_reviewer_admitted",
                    payload_json={
                        "cohort": reviewer.cohort,
                        "qualified_families": reviewer.qualification_json,
                        "affiliation_class": "independent_external",
                        "admission_protocol_version": "expert-admission-v2",
                        "consent_active_at_admission": True,
                        "calibration_candidate": candidate,
                        "calibration_candidate_record_sha256": canonical_sha256(
                            candidate
                        ),
                        "evidence": {
                            field: reviewer.profile_json[field]
                            for field in evidence_fields
                        },
                    },
                ),
                ]
            )
            session.flush()
            _record_expert_task_scope_events(
                session,
                battle=battle,
                reviewer=reviewer,
                review_session_id="scope-admission-session",
                review_assignment_id="scope-admission-assignment",
                assessment={
                    "scope_eligibility": "general_track",
                    "specialist_domains": [],
                    "general_track_eligible": True,
                },
                assessment_sha256="b" * 64,
                presentation_sha256="e" * 64,
            )
            session.flush()
            event = session.scalar(
                select(RunEvent).where(
                    RunEvent.entity_type == "battle",
                    RunEvent.entity_id == battle.id,
                    RunEvent.event_type == "battle_general_track_scope_admitted",
                )
            )
            assert event is not None
            assert event.payload_json["general_track_eligible"] is True
            assert event.payload_json["admission_basis"].startswith("sealed_qualified")
            session.rollback()

            specialist_battle = Battle(
                id="scope-quarantine-task-battle",
                task_id="scope-quarantine-task",
                data_stratum="controlled",
                controlled_run_id=None,
            )
            affiliated = ExpertReviewer(
                id="scope-quarantine-affiliated-reviewer",
                reviewer_code="scope-quarantine-affiliated-reviewer",
                invitation_sha256="c" * 64,
                qualification_json=["cookability"],
                qualification_verified=True,
                cohort="expert_product_affiliated",
                profile_json={"affiliation_class": "product_affiliated"},
            )
            _record_expert_task_scope_events(
                session,
                battle=specialist_battle,
                reviewer=affiliated,
                review_session_id="scope-quarantine-session",
                review_assignment_id="scope-quarantine-assignment",
                assessment={
                    "scope_eligibility": "specialist_track",
                    "specialist_domains": ["food_safety"],
                    "general_track_eligible": False,
                },
                assessment_sha256="d" * 64,
                presentation_sha256="f" * 64,
            )
            session.flush()
            task_event = session.scalar(
                select(RunEvent).where(
                    RunEvent.entity_type == "task",
                    RunEvent.entity_id == specialist_battle.task_id,
                    RunEvent.event_type == "task_general_track_scope_quarantined",
                )
            )
            battle_event = session.scalar(
                select(RunEvent).where(
                    RunEvent.entity_type == "battle",
                    RunEvent.entity_id == specialist_battle.id,
                    RunEvent.event_type == "battle_ranking_restricted",
                )
            )
            assert task_event is not None
            assert task_event.payload_json["general_track_eligible"] is False
            assert battle_event is not None
            assert battle_event.payload_json["operational_use"] is False
            session.rollback()


def test_public_display_and_voting_reject_non_normal_final_completion() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/v1/battles",
            headers=SERVICE_HEADERS,
            json={
                "prompt": "Design a practical leek and white bean supper with clear doneness cues.",
                "category": "cookability",
                "clientNonce": "non-normal-finish-public-test",
            },
        )
        assert created.status_code == 202, created.text
        while asyncio.run(run_worker_once("non-normal-finish-public-worker")):
            pass
        battle_id = created.json()["battleId"]
        with SessionLocal() as session:
            session.execute(
                text(
                    "UPDATE response_arms SET finish_reason = 'length' "
                    "WHERE id = (SELECT left_arm_id FROM battles WHERE id = :battle_id)"
                ),
                {"battle_id": battle_id},
            )
            session.commit()

        displayed = client.get(f"/v1/battles/{battle_id}", headers=SERVICE_HEADERS)
        assert displayed.status_code == 200, displayed.text
        assert displayed.json()["status"] == "failed"
        assert displayed.json()["answers"] == []

        vote = client.post(
            f"/v1/battles/{battle_id}/votes",
            headers={**SERVICE_HEADERS, "Idempotency-Key": "non-normal-finish-vote"},
            json={"choice": "left", "reasonTags": []},
        )
        assert vote.status_code == 409
        assert "normal final completion" in vote.json()["detail"]


def test_task_scope_is_separate_and_general_track_eligibility_is_derived() -> None:
    general = ExpertTaskAssessmentCreate.model_validate(
        {
            "task_validity": "valid",
            "task_issue_tags": [],
            "task_note": "",
            "answerability": "answerable",
            "family_fit": "in_family",
            "scope_eligibility": "general_track",
            "specialist_domains": [],
        }
    )
    assert general.general_track_eligible is True

    specialist = ExpertTaskAssessmentCreate.model_validate(
        {
            "task_validity": "valid",
            "task_issue_tags": [],
            "task_note": "Safety dominates the comparison.",
            "answerability": "answerable",
            "family_fit": "in_family",
            "scope_eligibility": "specialist_track",
            "specialist_domains": ["food_safety"],
        }
    )
    assert specialist.general_track_eligible is False

    with pytest.raises(ValueError, match="specialist domain"):
        ExpertTaskAssessmentCreate.model_validate(
            {
                "task_validity": "valid",
                "task_issue_tags": [],
                "task_note": "",
                "answerability": "answerable",
                "family_fit": "in_family",
                "scope_eligibility": "specialist_track",
                "specialist_domains": [],
            }
        )


def test_affiliated_workload_preserves_preregistered_track_and_family_quotas() -> None:
    targets = workload_cell_targets(1080)
    assert targets["primary_judgments"] == 960
    assert targets["reliability_repeats"] == 120
    assert set(targets["primary"]["model_arena"].values()) == {80}
    assert set(targets["primary"]["epicure_uplift"].values()) == {160}
    assert set(targets["reliability"]["model_arena"].values()) == {10}
    assert set(targets["reliability"]["epicure_uplift"].values()) == {20}


def test_author_evaluator_workload_uses_real_uplift_pairs_and_concealed_repeats() -> None:
    targets = author_evaluator_workload_cell_targets(32)
    assert targets["primary_judgments"] == 32
    assert targets["reliability_repeats"] == 4
    assert targets["total_presentations"] == 36
    assert set(targets["primary"]["model_arena"].values()) == {0}
    assert set(targets["primary"]["epicure_uplift"].values()) == {8}
    assert set(targets["reliability"]["epicure_uplift"].values()) == {1}


def test_swapped_reliability_presentations_normalize_to_canonical_sides() -> None:
    side_map = {"left": "right", "right": "left"}
    rubric = {
        "rubric_version": PROTOCOL_VERSION,
        "review_metadata": {
            "confidence": 5,
            "left_failure_tags": ["generic"],
            "right_failure_tags": ["ignored_constraint"],
        },
        "left": _arm_scores(2),
        "right": _arm_scores(5),
    }
    normalized = normalize_rubric(rubric, side_map)
    assert normalize_choice("left", side_map) == "right"
    assert normalize_choice("tie", side_map) == "tie"
    assert normalized["left"]["task_completion"] == 5
    assert normalized["right"]["task_completion"] == 2
    assert normalized["review_metadata"]["left_failure_tags"] == ["ignored_constraint"]
    assert normalized["review_metadata"]["right_failure_tags"] == ["generic"]


def test_reliability_summary_reports_preference_and_dimension_consistency() -> None:
    primary = {
        "battle-1": {
            "choice": "left",
            "rubric": {"left": _arm_scores(4), "right": _arm_scores(3)},
        }
    }
    repeats = [
        {
            "battle_id": "battle-1",
            "normalized_choice": "left",
            "normalized_rubric": {"left": _arm_scores(4), "right": _arm_scores(2)},
        }
    ]
    summary = reliability_summary(primary, repeats)
    assert summary["exactPreferenceAgreement"] == 1.0
    assert summary["preferenceAgreementInterval95"] is not None
    assert summary["dimensionComparisons"] == 18
    assert summary["meanAbsoluteDimensionDifference"] == 0.5
    assert summary["provisional"] is True


def test_expert_ranking_requires_v2_pre_response_task_validity() -> None:
    public_vote = Vote(cohort="public", rubric_json={})
    valid_expert_vote = Vote(
        cohort="expert_product_affiliated",
        rubric_json={
            "rubric_version": PROTOCOL_VERSION,
            "review_metadata": {
                "task_validity": "minor_issue",
                "general_track_eligible": True,
            },
        },
    )
    invalid_expert_vote = Vote(
        cohort="expert_product_affiliated",
        rubric_json={
            "rubric_version": PROTOCOL_VERSION,
            "review_metadata": {
                "task_validity": "invalid",
                "general_track_eligible": False,
            },
        },
    )
    specialist_expert_vote = Vote(
        cohort="expert_product_affiliated",
        rubric_json={
            "rubric_version": PROTOCOL_VERSION,
            "review_metadata": {
                "task_validity": "valid",
                "general_track_eligible": False,
            },
        },
    )
    legacy_expert_vote = Vote(
        cohort="expert_product_affiliated",
        rubric_json={"left": _arm_scores(), "right": _arm_scores()},
    )
    assert _task_validity_admissible(public_vote) is True
    assert _task_validity_admissible(valid_expert_vote) is True
    assert _task_validity_admissible(invalid_expert_vote) is False
    assert _task_validity_admissible(specialist_expert_vote) is False
    assert _task_validity_admissible(legacy_expert_vote) is False


def test_server_enforces_rolling_daily_and_continuous_block_limits() -> None:
    now = datetime(2026, 7, 29, 12, tzinfo=UTC)
    submitted = {
        str(index): RunEvent(created_at=now - timedelta(minutes=31 - index)) for index in range(32)
    }
    fatigue = _review_fatigue_status(submitted, now=now)
    assert fatigue["presentationsLast24Hours"] == 32
    assert fatigue["dailyLimitReached"] is True
    assert fatigue["dailyLimitResetsAt"] == "2026-07-30T11:29:00+00:00"
    assert fatigue["secondsUntilDailyReset"] == 84_540
    assert fatigue["breakRequired"] is False

    long_block = {
        str(index): RunEvent(created_at=now - timedelta(minutes=60 - index * 10))
        for index in range(7)
    }
    fatigue = _review_fatigue_status(long_block, now=now)
    assert fatigue["breakRequired"] is True


def test_pending_affiliated_expert_can_inspect_onboarding_but_not_review() -> None:
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "josef-affiliated-pending-fixture",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "self-reported-eight-year-practice",
                "qualification_verified": False,
                "affiliation_class": "product_affiliated",
                "conflict_disclosure_reference": "kaikaku-epicure-developer",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0,
                "compensation_reference": "unpaid-self-evaluation",
            },
        )
        assert invited.status_code == 200, invited.text
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invited.json()['invitation']}",
        }
        candidate = client.put(
            (f"/v1/admin/experts/{invited.json()['reviewerId']}/calibration-candidate"),
            headers=ADMIN_HEADERS,
            json=_calibration_candidate_body(),
        )
        assert candidate.status_code == 200, candidate.text
        assert candidate.json()["idempotent"] is False

        onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["admission"]["status"] == "pending"
        assert onboarding.json()["admission"]["reviewEnabled"] is False
        assert onboarding.json()["reviewer"]["cohort"] == "expert_product_affiliated"
        assert onboarding.json()["calibration"]["candidatePairs"] == 32
        assert onboarding.json()["calibration"]["realProviderCalls"] == 113
        assert onboarding.json()["calibration"]["realEpicureCalls"] == 80
        assert onboarding.json()["calibration"]["syntheticArms"] == 0

        protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert protocol.status_code == 403
        assert protocol.json()["detail"] == "expert onboarding and calibration are incomplete"


def test_pending_expert_requires_evidence_bound_admission_before_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "josef-affiliated-admission-fixture",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "pending-qualification",
                "qualification_verified": False,
                "affiliation_class": "product_affiliated",
                "conflict_disclosure_reference": "pending-conflict-disclosure",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0,
                "compensation_reference": "unpaid-self-evaluation",
            },
        )
        assert invited.status_code == 200, invited.text
        reviewer_id = invited.json()["reviewerId"]
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invited.json()['invitation']}",
        }
        candidate = client.put(
            f"/v1/admin/experts/{reviewer_id}/calibration-candidate",
            headers=ADMIN_HEADERS,
            json=_calibration_candidate_body(),
        )
        assert candidate.status_code == 200, candidate.text
        admission = {
            "qualification_reference": "verified-eight-year-practice",
            "conflict_disclosure_reference": "kaikaku-epicure-developer",
            "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
            "training_material_sha256": "5" * 64,
            "calibration_set_sha256": "6" * 64,
            "calibration_item_count": 20,
            "calibration_gold_adjudicator_count": 2,
            "calibration_accuracy": 0.85,
            "admission_decision_reference": "signed-admission-decision-001",
        }

        admitted = client.put(
            f"/v1/admin/experts/{reviewer_id}/admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert admitted.status_code == 200, admitted.text
        assert admitted.json()["admissionStatus"] == "active"
        assert admitted.json()["idempotent"] is False

        onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["admission"]["reviewEnabled"] is True
        protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert protocol.status_code == 200, protocol.text

        with monkeypatch.context() as scoped_patch:
            scoped_patch.setattr(
                "flavourbench.main.resolve_expert_consent_document",
                lambda _digest: ConsentDocumentResolution(
                    ACTIVE_EXPERT_CONSENT_SHA256,
                    "document_missing",
                    None,
                ),
            )
            blocked_onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
            assert blocked_onboarding.status_code == 200, blocked_onboarding.text
            assert blocked_onboarding.json()["admission"]["reviewEnabled"] is False
            assert blocked_onboarding.json()["admission"]["blockers"] == [
                "active expert-consent document"
            ]
            blocked_protocol = client.get("/v1/expert/protocol", headers=expert_headers)
            assert blocked_protocol.status_code == 403, blocked_protocol.text
            assert "bound consent document is active" in blocked_protocol.json()["detail"]

        repeated = client.put(
            f"/v1/admin/experts/{reviewer_id}/admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["idempotent"] is True

        changed = {
            **admission,
            "calibration_set_sha256": "7" * 64,
        }
        conflict = client.put(
            f"/v1/admin/experts/{reviewer_id}/admission",
            headers=ADMIN_HEADERS,
            json=changed,
        )
        assert conflict.status_code == 409, conflict.text


def test_author_evaluator_admission_removes_external_gold_gate_without_claiming_independence() -> (
    None
):
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "josef-author-evaluator-fixture",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "self-reported-eight-year-practice",
                "qualification_verified": False,
                "affiliation_class": "product_affiliated",
                "conflict_disclosure_reference": "benchmark-author-and-epicure-developer",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0,
                "compensation_reference": "unpaid-author-evaluation",
            },
        )
        assert invited.status_code == 200, invited.text
        reviewer_id = invited.json()["reviewerId"]
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invited.json()['invitation']}",
        }
        candidate = client.put(
            f"/v1/admin/experts/{reviewer_id}/calibration-candidate",
            headers=ADMIN_HEADERS,
            json=_calibration_candidate_body(),
        )
        assert candidate.status_code == 200, candidate.text
        admission = {
            "qualification_reference": "Josef Chen; eight years culinary practice",
            "conflict_disclosure_reference": (
                "Josef Chen develops Epicure and authors the FlavourBench manuscript"
            ),
            "candidate_pack_sha256": "8" * 64,
            "primary_judgments": 32,
            "admission_decision_reference": "author-evaluator-pathway-2026-07-30",
            "independent_validation_claim": False,
        }
        admitted = client.put(
            f"/v1/admin/experts/{reviewer_id}/author-evaluator-admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert admitted.status_code == 200, admitted.text
        assert admitted.json()["admissionPathway"] == "author_evaluator"
        assert admitted.json()["targetJudgments"] == 36
        assert admitted.json()["independentValidationClaim"] is False
        assert admitted.json()["idempotent"] is False

        onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["admission"] == {
            "status": "active",
            "reviewEnabled": True,
            "blockers": [],
            "pathway": "author_evaluator",
            "targetJudgments": 36,
        }
        assert onboarding.json()["calibration"]["requiredForAdmission"] is False
        assert onboarding.json()["calibration"]["independentGoldBallots"] == 0
        assert onboarding.json()["reviewer"]["calibrationAccuracy"] is None
        assert "not independent expert validation" in onboarding.json()["claimBoundary"]

        protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert protocol.status_code == 200, protocol.text
        assert protocol.json()["reviewer"]["admissionPathway"] == "author_evaluator"
        assert protocol.json()["reviewer"]["targetJudgments"] == 36

        opened = client.post(
            "/v1/expert/sessions",
            headers=expert_headers,
            json={
                "protocolSha256": PROTOCOL_SHA256,
                "controlledRunId": None,
                "targetJudgments": 36,
                "acknowledgements": sorted(REQUIRED_ACKNOWLEDGEMENTS),
            },
        )
        assert opened.status_code == 201, opened.text
        wrong_target = client.post(
            "/v1/expert/sessions",
            headers=expert_headers,
            json={
                "protocolSha256": PROTOCOL_SHA256,
                "controlledRunId": None,
                "targetJudgments": 40,
                "acknowledgements": sorted(REQUIRED_ACKNOWLEDGEMENTS),
            },
        )
        assert wrong_target.status_code == 409, wrong_target.text

        repeated = client.put(
            f"/v1/admin/experts/{reviewer_id}/author-evaluator-admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["idempotent"] is True

        invalid_claim = client.put(
            f"/v1/admin/experts/{reviewer_id}/author-evaluator-admission",
            headers=ADMIN_HEADERS,
            json={**admission, "independent_validation_claim": True},
        )
        assert invalid_claim.status_code == 422, invalid_claim.text


def test_anonymous_external_rater_is_identity_minimized_and_not_mislabelled_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "fbr-anon-external-fixture",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "pending anonymous self-attestation",
                "qualification_verified": False,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": (
                    "pending anonymous independence self-attestation"
                ),
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0,
                "compensation_reference": "unpaid anonymous external review",
            },
        )
        assert invited.status_code == 200, invited.text
        reviewer_id = invited.json()["reviewerId"]
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invited.json()['invitation']}",
        }
        candidate = client.put(
            f"/v1/admin/experts/{reviewer_id}/calibration-candidate",
            headers=ADMIN_HEADERS,
            json=_calibration_candidate_body(),
        )
        assert candidate.status_code == 200, candidate.text
        admission = {
            "candidate_pack_sha256": "8" * 64,
            "primary_judgments": 32,
            "admission_decision_reference": ("anonymous-external-rater-pathway-2026-07-31"),
            "identity_collection_prohibited": True,
            "independence_self_attestation_required": True,
            "qualification_self_attestation_required": True,
        }
        admitted = client.put(
            f"/v1/admin/experts/{reviewer_id}/anonymous-external-admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert admitted.status_code == 200, admitted.text
        assert admitted.json()["admissionPathway"] == "anonymous_external_rater"
        assert admitted.json()["qualificationVerified"] is False
        assert admitted.json()["targetJudgments"] == 36
        assert admitted.json()["independentExpertValidationClaim"] is False

        onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["admission"] == {
            "status": "pending",
            "reviewEnabled": False,
            "blockers": ["pool-specific anonymous reviewer re-consent"],
            "pathway": "anonymous_external_rater",
            "targetJudgments": None,
        }
        reconsent_contract = onboarding.json()["reconsent"]
        assert reconsent_contract["required"] is True
        assert reconsent_contract["recordedForCurrentPool"] is False
        assert reconsent_contract["consentDocumentStatus"] == "active"
        assert isinstance(reconsent_contract["consentDocumentText"], str)
        assert hashlib.sha256(
            reconsent_contract["consentDocumentText"].encode("utf-8")
        ).hexdigest() == reconsent_contract["consentDocumentSha256"]

        premature_protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert premature_protocol.status_code == 403
        assert "re-consent" in premature_protocol.json()["detail"]

        wrong_reconsent = client.post(
            "/v1/expert/anonymous-external-reconsent",
            headers=expert_headers,
            json={
                "candidate_pack_sha256": reconsent_contract["candidatePackSha256"],
                "pool_activation_sha256": "f" * 64,
                "consent_document_sha256": reconsent_contract["consentDocumentSha256"],
                "consent_statement_sha256": reconsent_contract["statementSha256"],
                "voluntary_participation_accepted": True,
                "pool_specific_consent_accepted": True,
            },
        )
        assert wrong_reconsent.status_code == 409

        reconsent_body = {
            "candidate_pack_sha256": reconsent_contract["candidatePackSha256"],
            "pool_activation_sha256": reconsent_contract["poolActivationSha256"],
            "consent_document_sha256": reconsent_contract["consentDocumentSha256"],
            "consent_statement_sha256": reconsent_contract["statementSha256"],
            "voluntary_participation_accepted": True,
            "pool_specific_consent_accepted": True,
        }
        for unresolved_status in ("document_missing", "document_hash_mismatch"):
            with monkeypatch.context() as scoped_patch:
                scoped_patch.setattr(
                    "flavourbench.main.resolve_expert_consent_document",
                    lambda _digest, status=unresolved_status: ConsentDocumentResolution(
                        reconsent_contract["consentDocumentSha256"],
                        status,
                        None,
                    ),
                )
                blocked_reconsent = client.post(
                    "/v1/expert/anonymous-external-reconsent",
                    headers=expert_headers,
                    json=reconsent_body,
                )
            assert blocked_reconsent.status_code == 409
            assert "consent document is not active" in blocked_reconsent.json()["detail"]

        reconsented = client.post(
            "/v1/expert/anonymous-external-reconsent",
            headers=expert_headers,
            json=reconsent_body,
        )
        assert reconsented.status_code == 200, reconsented.text
        assert reconsented.json()["reviewEnabled"] is True
        assert reconsented.json()["idempotent"] is False
        repeated_reconsent = client.post(
            "/v1/expert/anonymous-external-reconsent",
            headers=expert_headers,
            json=reconsent_body,
        )
        assert repeated_reconsent.status_code == 200, repeated_reconsent.text
        assert repeated_reconsent.json()["idempotent"] is True

        onboarding = client.get("/v1/expert/onboarding", headers=expert_headers)
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["admission"] == {
            "status": "active",
            "reviewEnabled": True,
            "blockers": [],
            "pathway": "anonymous_external_rater",
            "targetJudgments": 36,
        }
        assert onboarding.json()["reviewer"]["qualificationVerified"] is False
        assert (
            onboarding.json()["reviewer"]["qualificationBasis"]
            == "reviewer_self_attestation_unverified"
        )
        assert onboarding.json()["calibration"]["requiredForAdmission"] is False
        assert "pseudonymous external rater" in onboarding.json()["claimBoundary"]
        assert "not verified expert consensus" in onboarding.json()["claimBoundary"]

        protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert protocol.status_code == 200, protocol.text
        assert protocol.json()["reviewer"]["admissionPathway"] == "anonymous_external_rater"
        assert (
            "not an author or developer"
            in protocol.json()["acknowledgementStatements"]["conflict_disclosed"]
        )
        assert (
            "self-attestation"
            in protocol.json()["acknowledgementStatements"]["culinary_competence"]
        )

        opened = client.post(
            "/v1/expert/sessions",
            headers=expert_headers,
            json={
                "protocolSha256": PROTOCOL_SHA256,
                "controlledRunId": None,
                "targetJudgments": 36,
                "acknowledgements": sorted(REQUIRED_ACKNOWLEDGEMENTS),
            },
        )
        assert opened.status_code == 201, opened.text
        status = client.get(
            f"/v1/expert/sessions/{opened.json()['reviewSessionId']}/status",
            headers=expert_headers,
        )
        assert status.status_code == 200, status.text
        assert status.json()["qualityGate"]["independentExternalRater"] is False
        assert status.json()["qualityGate"]["selfAttestedExternalRater"] is True
        assert status.json()["qualityGate"]["verifiedIndependentExternalRater"] is False
        assert status.json()["qualityGate"]["independentExpertValidation"] is False

        repeated = client.put(
            f"/v1/admin/experts/{reviewer_id}/anonymous-external-admission",
            headers=ADMIN_HEADERS,
            json=admission,
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["idempotent"] is True


def test_anonymous_external_rater_rejects_unregistered_consent_document() -> None:
    reviewer = ExpertReviewer(
        reviewer_code="fbr-anon-inactive-consent",
        invitation_sha256="2" * 64,
        qualification_json=["substitution"],
        qualification_verified=False,
        cohort="expert_independent",
        profile_json={"consent_document_sha256": "2" * 64},
        batch_reveal_only=True,
    )
    assert _expert_consent_document_active(reviewer) is False


def test_pending_anonymous_rater_claim_never_implies_verified_qualification() -> None:
    reviewer = ExpertReviewer(
        reviewer_code="fbr-anon-pending-consent",
        invitation_sha256="3" * 64,
        qualification_json=["substitution"],
        qualification_verified=False,
        cohort="expert_independent",
        profile_json={
            "consent_document_sha256": "2" * 64,
            "admission_pathway": "anonymous_external_rater",
            "anonymous_external_admission_status": "active",
            "anonymous_external_pool_sha256": "4" * 64,
            "identity_collection_prohibited": True,
            "independence_basis": "reviewer_self_attestation",
            "qualification_basis": "reviewer_self_attestation_unverified",
            "independent_expert_validation_claim": False,
        },
        batch_reveal_only=True,
    )
    claim = _reviewer_claim_boundary(reviewer)
    assert "pseudonymous external rater" in claim
    assert "self-attested" in claim
    assert "qualification-verified" not in claim


def test_affiliated_expert_v2_session_records_blinded_review_and_status() -> None:
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "josef-affiliated-v2-fixture",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "verified-eight-year-practice",
                "qualification_verified": True,
                "affiliation_class": "product_affiliated",
                "conflict_disclosure_reference": "kaikaku-epicure-developer",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0.9,
                "compensation_reference": "unpaid-self-evaluation",
            },
        )
        assert invited.status_code == 200, invited.text
        reviewer_id = invited.json()["reviewerId"]
        invitation = invited.json()["invitation"]
        expert_headers = {
            "X-FlavourBench-Service-Token": "test-service-token",
            "Authorization": f"Bearer {invitation}",
        }
        candidate = client.put(
            f"/v1/admin/experts/{reviewer_id}/calibration-candidate",
            headers=ADMIN_HEADERS,
            json=_calibration_candidate_body(),
        )
        assert candidate.status_code == 200, candidate.text
        admitted = client.put(
            f"/v1/admin/experts/{reviewer_id}/admission",
            headers=ADMIN_HEADERS,
            json={
                "qualification_reference": "verified-eight-year-practice",
                "conflict_disclosure_reference": "kaikaku-epicure-developer",
                "consent_document_sha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_item_count": 20,
                "calibration_gold_adjudicator_count": 2,
                "calibration_accuracy": 0.9,
                "admission_decision_reference": "affiliated-v2-governed-admission",
            },
        )
        assert admitted.status_code == 200, admitted.text

        created = client.post(
            "/v1/battles",
            headers=SERVICE_HEADERS,
            json={
                "prompt": (
                    "Design a vegetarian cauliflower main with black garlic and clear "
                    "doneness cues."
                ),
                "category": "cookability",
                "clientNonce": "affiliated-v2-review-battle",
            },
        )
        assert created.status_code == 202, created.text
        while asyncio.run(run_worker_once("affiliated-v2-review-worker")):
            pass
        battle_id = created.json()["battleId"]

        protocol = client.get("/v1/expert/protocol", headers=expert_headers)
        assert protocol.status_code == 200, protocol.text
        assert protocol.json()["protocolSha256"] == PROTOCOL_SHA256
        assert protocol.json()["reviewer"]["cohort"] == "expert_product_affiliated"

        opened = client.post(
            "/v1/expert/sessions",
            headers=expert_headers,
            json={
                "protocolSha256": PROTOCOL_SHA256,
                "controlledRunId": None,
                "targetJudgments": 40,
                "acknowledgements": sorted(REQUIRED_ACKNOWLEDGEMENTS),
            },
        )
        assert opened.status_code == 201, opened.text
        review_session_id = opened.json()["reviewSessionId"]

        claimed = client.post(
            f"/v1/expert/sessions/{review_session_id}/assignments/next",
            headers=expert_headers,
            json={},
        )
        assert claimed.status_code == 200, claimed.text
        assignment = claimed.json()["assignment"]
        assert assignment is not None
        assert assignment["stage"] == "task_assessment"
        assert assignment["answers"] == []
        assert assignment["reveal"] is None
        assert "modelId" not in claimed.text
        assert "condition" not in claimed.text

        review_assignment_id = assignment["reviewAssignmentId"]
        assessment_headers = {
            **expert_headers,
            "Idempotency-Key": "affiliated-v2-task-assessment-001",
        }
        assessed = client.post(
            f"/v1/expert/review-assignments/{review_assignment_id}/task-assessment",
            headers=assessment_headers,
            json={
                "task_validity": "valid",
                "task_issue_tags": [],
                "task_note": "",
                "answerability": "answerable",
                "family_fit": "in_family",
                "scope_eligibility": "specialist_track",
                "specialist_domains": ["food_safety"],
            },
        )
        assert assessed.status_code == 200, assessed.text
        assert assessed.json()["assignment"]["stage"] == "response_review"
        assert len(assessed.json()["assignment"]["answers"]) == 2
        assert "modelId" not in assessed.text
        assert "condition" not in assessed.text
        with SessionLocal() as session:
            battle_scope_events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type == "battle",
                    RunEvent.entity_id == battle_id,
                    RunEvent.event_type == "battle_ranking_restricted",
                )
            ).all()
            assert len(battle_scope_events) == 1
            assert battle_scope_events[0].payload_json["operational_use"] is False

        assessed_again = client.post(
            f"/v1/expert/review-assignments/{review_assignment_id}/task-assessment",
            headers=assessment_headers,
            json={
                "task_validity": "valid",
                "task_issue_tags": [],
                "task_note": "",
                "answerability": "answerable",
                "family_fit": "in_family",
                "scope_eligibility": "specialist_track",
                "specialist_domains": ["food_safety"],
            },
        )
        assert assessed_again.status_code == 200, assessed_again.text
        with SessionLocal() as session:
            assert (
                len(
                    session.scalars(
                        select(RunEvent).where(
                            RunEvent.entity_type == "battle",
                            RunEvent.entity_id == battle_id,
                            RunEvent.event_type == "battle_ranking_restricted",
                        )
                    ).all()
                )
                == 1
            )

        review_headers = {**expert_headers, "Idempotency-Key": "affiliated-v2-review-001"}
        recorded = client.post(
            f"/v1/expert/review-assignments/{review_assignment_id}",
            headers=review_headers,
            json=_specialist_safety_review_body(),
        )
        assert recorded.status_code == 200, recorded.text
        assert recorded.json()["recorded"] is True
        assert recorded.json()["reveal"] is None

        repeated = client.post(
            f"/v1/expert/review-assignments/{review_assignment_id}",
            headers=review_headers,
            json=_specialist_safety_review_body(),
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["voteId"] == recorded.json()["voteId"]
        with SessionLocal() as session:
            safety_events = session.scalars(
                select(RunEvent).where(
                    RunEvent.event_type == "reviewer_reported_potential_safety_hazard",
                    RunEvent.payload_json["battle_id"].as_string() == battle_id,
                )
            ).all()
            incidents = session.scalars(
                select(Incident).where(
                    Incident.battle_id == battle_id,
                    Incident.code == "reviewer_reported_potential_safety_hazard",
                )
            ).all()
            assert len(safety_events) == 1
            assert safety_events[0].payload_json["reported_tags"] == ["safety_hazard"]
            assert (
                safety_events[0].payload_json["preference_exclusion_requested"] is False
            )
            assert len(incidents) == 1

        status = client.get(
            f"/v1/expert/sessions/{review_session_id}/status",
            headers=expert_headers,
        )
        assert status.status_code == 200, status.text
        assert status.json()["primaryJudgments"] == 1
        assert status.json()["qualityGate"]["cellQuotaReached"] is False
        assert status.json()["qualityGate"]["modelExposureReached"] is False
        assert status.json()["qualityGate"]["comparisonGraphConnected"] is False
        assert status.json()["qualityGate"]["independentExpertValidation"] is False
        assert "product-affiliated" in status.json()["claimBoundary"]

        exported = client.get(
            f"/v1/admin/expert-sessions/{review_session_id}/export",
            headers=ADMIN_HEADERS,
        )
        assert exported.status_code == 200, exported.text
        export_payload = exported.json()
        export_sha256 = export_payload.pop("exportSha256")
        assert export_sha256 == canonical_sha256(export_payload)
        assert len(export_payload["records"]) == 1
        assert export_payload["records"][0]["normalizedChoice"] == "left"
        assert all(
            "answerMarkdown" not in arm and len(arm["answerMarkdownSha256"]) == 64
            for arm in export_payload["records"][0]["arms"]
        )
