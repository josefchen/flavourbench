from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate

from flavourbench.task_validation_campaign import (
    ZERO_SHA256,
    TaskValidationCampaignError,
    build_campaign_packet,
    build_quality_report,
    build_readiness_decision,
    derive_candidate_state,
    load_inputs,
    make_ledger_event,
    public_event_view,
    verify_campaign_packet,
    verify_event_chain,
)

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = ROOT / "artifacts/season1/prospective-task-acquisition-v1"
BUNDLE = ACQUISITION / (
    "public-human-task-candidates-"
    "b13ab30bfb391e57a24c81d0398dc98e408d88e5a0bf21c4e758bf9271724cc3.json"
)
ASSIGNMENT = ACQUISITION / (
    "public-human-task-review-assignment-"
    "631932c0560ec417e47ff4c3ea94814ca9c944253252d3f4adcee8bd595221f9.json"
)
RECEIPT = ACQUISITION / (
    "public-api-acquisition-receipt-"
    "847a95f7159ba778281fd5c20f0489a75f4655fc08a0de8075a0cba950259045.json"
)
EVENT_SCHEMA = ROOT / "contracts/season1/task-validation-ledger-event-v1.schema.json"


def _inputs() -> tuple[dict, dict, dict]:
    return load_inputs(bundle_path=BUNDLE, assignment_path=ASSIGNMENT, receipt_path=RECEIPT)


def _packet() -> dict:
    bundle, assignment, receipt = _inputs()
    return build_campaign_packet(
        bundle=bundle,
        assignment=assignment,
        receipt=receipt,
        physical_hashes={
            "candidate_bundle": "a" * 64,
            "review_assignment": "b" * 64,
            "acquisition_receipt": "c" * 64,
        },
    )


def _event(
    *,
    campaign: str,
    sequence: int,
    event_type: str,
    person: str,
    payload: dict,
    previous: str,
) -> dict:
    return make_ledger_event(
        campaign_sha256=campaign,
        sequence=sequence,
        event_id=f"event-{sequence}",
        event_type=event_type,  # type: ignore[arg-type]
        reviewer_pseudonym=f"reviewer-{person[0]}",
        person_commitment_sha256=person,
        reviewer_admission_receipt_sha256=str(sequence) * 64,
        payload=payload,
        previous_event_sha256=previous,
    )


def _valid_ballot(candidate_id: str, family: str = "composition") -> dict:
    return {
        "candidate_id": candidate_id,
        "decision": "valid",
        "checks": {
            "construct_fit": True,
            "context_complete": True,
            "coherent_question": True,
            "general_track_scope": True,
            "answer_leakage_absent": True,
            "discrimination_value": True,
        },
        "family": family,
        "construct_cell_id": "bridge_ingredient_reasoning",
        "difficulty_tier": "integrative",
        "success_criteria": ["Explains the functional bridge between the ingredients."],
        "permitted_variations": ["Accepts another workable bridge with a clear rationale."],
        "disqualifying_errors": ["Proposes a combination that violates the stated constraint."],
        "objective_checks": [],
        "source_metadata_seen": False,
        "other_ballot_seen": False,
        "model_outputs_seen": False,
    }


def test_campaign_freezes_120_target_with_60_candidate_reserve_and_no_fake_evidence() -> None:
    packet = _packet()
    verify_campaign_packet(packet)
    assert packet["target"] == {
        "validated_tasks": 120,
        "validated_tasks_per_family": 30,
        "candidate_slate": 180,
        "candidate_slate_per_scheduling_family": 45,
        "reserve_candidates": 60,
        "activation": (
            "fixed schedule order; stop only when human-confirmed quotas reach 30 in each "
            "family or the 180-candidate slate is exhausted"
        ),
        "effect_based_or_model_result_based_stopping": False,
    }
    assert packet["observations"] == {
        "human_ballots": 0,
        "adjudications": 0,
        "batch_audits": 0,
        "model_calls": 0,
        "epicure_calls": 0,
        "synthetic_tasks": 0,
    }
    assert packet["validation_protocol"]["unanimous_candidate_adjudication"] is False
    assert packet["minimum_human_workload"]["best_case_human_actions"] == 482
    assert packet["minimum_human_workload"][
        "minimum_distinct_people_with_full_cross_task_reuse"
    ] == 5
    assert len(packet["candidate_schedule"]) == 180
    assert len({row["candidate_id"] for row in packet["candidate_schedule"]}) == 180


def test_quality_report_marks_real_source_risks_as_review_triggers_not_ground_truth() -> None:
    bundle, assignment, receipt = _inputs()
    report = build_quality_report(
        bundle=bundle,
        assignment=assignment,
        receipt=receipt,
        physical_hashes={},
    )
    assert report["checks"]["synthetic_tasks"] == 0
    assert report["checks"]["model_calls"] == 0
    assert report["checks"]["source_answer_payloads_requested"] == 0
    assert report["checks"]["unique_candidate_ids"] == 180
    assert report["checks"]["unique_attributed_authors"] == 152
    assert report["checks"]["maximum_candidates_per_author"] == 4
    assert report["manual_review_triggers"]["direct_url_present"]["count"] == 6
    assert report["manual_review_triggers"]["visual_or_video_reference"]["count"] == 4
    assert report["license_field_anomalies"]["question_api_license_null_count"] == 1
    assert report["assessment"]["task_validity"] == "not_yet_established"
    assert report["assessment"]["automated_trigger_interpretation"] == (
        "review_priority_not_human_ground_truth"
    )
    readiness = build_readiness_decision(campaign=_packet(), quality_report=report)
    decisions = {row["gate"]: row["decision"] for row in readiness["gates"]}
    assert decisions["source_acquisition_and_provenance"] == "go"
    assert decisions["live_ballot_collection"] == "no_go"
    assert decisions["official_task_bank"] == "no_go"
    assert decisions["contamination_free_claim"] == "permanent_no_go"
    assert readiness["claim_boundary"]["human_release_authority_exercised"] is False


def test_two_independent_consensus_ballots_need_no_adjudicator() -> None:
    campaign = _packet()["artifact_sha256"]
    candidate = "candidate-one"
    events: list[dict] = []
    prior = ZERO_SHA256
    for sequence, person in enumerate(("a" * 64, "b" * 64), start=1):
        event = _event(
            campaign=campaign,
            sequence=sequence,
            event_type="blind_ballot",
            person=person,
            payload=_valid_ballot(candidate),
            previous=prior,
        )
        events.append(event)
        prior = event["event_sha256"]
    provisional = derive_candidate_state(candidate_id=candidate, events=events)
    assert provisional["status"] == "awaiting_criterion_pack_confirmations"
    pack_sha256 = provisional["criterion_pack_sha256"]
    for sequence, person in enumerate(("a" * 64, "b" * 64), start=3):
        event = _event(
            campaign=campaign,
            sequence=sequence,
            event_type="criterion_pack_confirmation",
            person=person,
            payload={
                "candidate_id": candidate,
                "criterion_pack_sha256": pack_sha256,
                "accepted": True,
            },
            previous=prior,
        )
        events.append(event)
        prior = event["event_sha256"]
    verified = verify_event_chain(events, campaign_sha256=campaign)
    state = derive_candidate_state(candidate_id=candidate, events=verified)
    assert state["status"] == "validated_consensus"
    assert state["adjudication_required"] is False


def test_disagreement_requires_a_distinct_third_person() -> None:
    campaign = _packet()["artifact_sha256"]
    candidate = "candidate-two"
    first = _event(
        campaign=campaign,
        sequence=1,
        event_type="blind_ballot",
        person="a" * 64,
        payload=_valid_ballot(candidate, "composition"),
        previous=ZERO_SHA256,
    )
    second = _event(
        campaign=campaign,
        sequence=2,
        event_type="blind_ballot",
        person="b" * 64,
        payload=_valid_ballot(candidate, "evidence"),
        previous=first["event_sha256"],
    )
    assert derive_candidate_state(candidate_id=candidate, events=[first, second]) == {
        "status": "awaiting_adjudication",
        "adjudication_required": True,
    }
    invalid_adjudication = _event(
        campaign=campaign,
        sequence=3,
        event_type="adjudication",
        person="a" * 64,
        payload={
            "candidate_id": candidate,
            "decision": "reject",
            "model_outputs_seen": False,
        },
        previous=second["event_sha256"],
    )
    with pytest.raises(TaskValidationCampaignError, match="validator adjudicated"):
        derive_candidate_state(
            candidate_id=candidate,
            events=[first, second, invalid_adjudication],
        )


def test_chain_tampering_and_same_person_double_ballots_fail_closed() -> None:
    campaign = _packet()["artifact_sha256"]
    first = _event(
        campaign=campaign,
        sequence=1,
        event_type="blind_ballot",
        person="a" * 64,
        payload=_valid_ballot("candidate-three"),
        previous=ZERO_SHA256,
    )
    tampered = json.loads(json.dumps(first))
    tampered["payload"]["decision"] = "exclude"
    with pytest.raises(TaskValidationCampaignError, match="digest differs"):
        verify_event_chain([tampered], campaign_sha256=campaign)

    second = _event(
        campaign=campaign,
        sequence=2,
        event_type="blind_ballot",
        person="a" * 64,
        payload=_valid_ballot("candidate-three"),
        previous=first["event_sha256"],
    )
    with pytest.raises(TaskValidationCampaignError, match="person uniqueness"):
        derive_candidate_state(candidate_id="candidate-three", events=[first, second])


def test_public_event_view_removes_private_identity_linkage() -> None:
    event = _event(
        campaign=_packet()["artifact_sha256"],
        sequence=1,
        event_type="blind_ballot",
        person="a" * 64,
        payload=_valid_ballot("candidate-four"),
        previous=ZERO_SHA256,
    )
    public = public_event_view(event)
    validate(instance=event, schema=json.loads(EVENT_SCHEMA.read_text(encoding="utf-8")))
    assert public["reviewer_pseudonym"] == "reviewer-a"
    assert "person_commitment_sha256" not in public
    assert "reviewer_admission_receipt_sha256" not in public
