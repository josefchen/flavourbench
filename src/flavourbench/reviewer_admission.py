"""Evidence-bound admission checks shared by review and ranking paths."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .consent_documents import resolve_expert_consent_document
from .models import ExpertReviewer, RunEvent


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _utc(value: datetime) -> datetime:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).astimezone(UTC)


def _candidate_evidence_valid(candidate: object, evidence: object) -> bool:
    if not isinstance(candidate, dict) or not isinstance(evidence, dict):
        return False
    calibration_accuracy = evidence.get("calibration_accuracy")
    calibration_items = evidence.get("calibration_item_count")
    gold_adjudicators = evidence.get("calibration_gold_adjudicator_count")
    return bool(
        candidate.get("synthetic_arms") == 0
        and candidate.get("rank_eligible") is False
        and isinstance(candidate.get("candidate_pack_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", candidate["candidate_pack_sha256"])
        and isinstance(candidate.get("candidate_pairs"), int)
        and candidate["candidate_pairs"] >= 20
        and isinstance(candidate.get("source_arms"), int)
        and candidate["source_arms"] >= candidate["candidate_pairs"] * 2
        and isinstance(candidate.get("real_provider_calls"), int)
        and candidate["real_provider_calls"] >= candidate["source_arms"]
        and isinstance(candidate.get("successful_real_epicure_calls"), int)
        and candidate["successful_real_epicure_calls"] >= candidate["candidate_pairs"]
        and isinstance(calibration_accuracy, (int, float))
        and calibration_accuracy >= 0.8
        and isinstance(calibration_items, int)
        and 20 <= calibration_items <= candidate["candidate_pairs"]
        and isinstance(gold_adjudicators, int)
        and gold_adjudicators >= 2
    )


def calibrated_expert_admission_event(
    session: Session,
    reviewer: ExpertReviewer,
) -> RunEvent | None:
    """Return the verified ordinary-expert admission event, or fail closed."""

    profile = reviewer.profile_json
    candidate = profile.get("calibration_candidate")
    consent_sha256 = profile.get("consent_document_sha256")
    if not (
        reviewer.qualification_verified
        and reviewer.active
        and reviewer.revoked_at is None
        and isinstance(consent_sha256, str)
        and resolve_expert_consent_document(consent_sha256).status == "active"
        and _candidate_evidence_valid(candidate, profile)
        and all(
            isinstance(profile.get(field), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(profile[field]))
            for field in (
                "consent_document_sha256",
                "training_material_sha256",
                "calibration_set_sha256",
                "admission_decision_sha256",
            )
        )
    ):
        return None
    event = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer.id,
            RunEvent.event_type == "expert_reviewer_admitted",
        )
        .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
    )
    if event is None:
        return None
    evidence = event.payload_json.get("evidence")
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
    if not (
        event.payload_json.get("cohort") == reviewer.cohort
        and event.payload_json.get("qualified_families") == reviewer.qualification_json
        and event.payload_json.get("affiliation_class")
        == reviewer.profile_json.get("affiliation_class")
        and event.payload_json.get("admission_protocol_version") == "expert-admission-v2"
        and event.payload_json.get("consent_active_at_admission") is True
        and event.payload_json.get("calibration_candidate") == candidate
        and event.payload_json.get("calibration_candidate_record_sha256")
        == _canonical_sha256(candidate)
        and isinstance(evidence, dict)
        and all(evidence.get(field) == profile.get(field) for field in evidence_fields)
    ):
        return None
    return event


def historical_expert_admission_event(
    session: Session,
    *,
    reviewer_id: str,
    event_id: str,
    as_of: datetime,
) -> RunEvent | None:
    """Verify one immutable admission at a historical cutoff.

    This path deliberately ignores the current consent registry and current reviewer
    activation flags. Those govern live access. A frozen ranking instead follows the
    admission, candidate-registration, and revocation events that existed by ``as_of``.
    """

    event = session.get(RunEvent, event_id)
    if not (
        event is not None
        and event.entity_type == "expert_reviewer"
        and event.entity_id == reviewer_id
        and event.event_type == "expert_reviewer_admitted"
        and _utc(event.created_at) <= _utc(as_of)
    ):
        return None
    payload = event.payload_json if isinstance(event.payload_json, dict) else {}
    evidence = payload.get("evidence")
    candidate = payload.get("calibration_candidate")
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
    if not (
        payload.get("admission_protocol_version") == "expert-admission-v2"
        and payload.get("consent_active_at_admission") is True
        and payload.get("affiliation_class")
        in {"independent_external", "product_affiliated", "provider_affiliated"}
        and isinstance(payload.get("cohort"), str)
        and isinstance(payload.get("qualified_families"), list)
        and isinstance(evidence, dict)
        and _candidate_evidence_valid(candidate, evidence)
        and payload.get("calibration_candidate_record_sha256") == _canonical_sha256(candidate)
        and all(isinstance(evidence.get(field), (str, int, float)) for field in evidence_fields)
        and all(
            isinstance(evidence.get(field), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(evidence[field]))
            for field in (
                "consent_document_sha256",
                "training_material_sha256",
                "calibration_set_sha256",
                "admission_decision_sha256",
            )
        )
    ):
        return None
    registration = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer_id,
            RunEvent.event_type == "expert_calibration_candidate_registered",
            RunEvent.created_at <= event.created_at,
        )
        .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
    )
    registration_payload = (
        registration.payload_json
        if registration is not None and isinstance(registration.payload_json, dict)
        else {}
    )
    if not (
        registration is not None
        and registration_payload.get("candidate") == candidate
        and registration_payload.get("candidate_record_sha256") == _canonical_sha256(candidate)
    ):
        return None
    invalidating = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer_id,
            RunEvent.event_type.in_(
                {
                    "expert_reviewer_revoked",
                    "expert_reviewer_admission_superseded",
                }
            ),
            RunEvent.created_at <= as_of,
            RunEvent.payload_json["admission_event_id"].as_string() == event.id,
        )
    )
    return None if invalidating is not None else event


def calibrated_expert_admission_active(
    session: Session,
    reviewer: ExpertReviewer,
) -> bool:
    return calibrated_expert_admission_event(session, reviewer) is not None
