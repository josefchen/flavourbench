"""Fail-closed activation evidence for the anonymous external-review pathway."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .expert_review import canonical_sha256
from .models import ExpertReviewer, RunEvent

RECONSENT_SCHEMA_VERSION = "flavourbench-anonymous-external-pool-reconsent-v1"
RECONSENT_EVENT_TYPE = "expert_anonymous_external_pool_reconsented"
RECONSENT_STATEMENT = (
    "I voluntarily consent to review the response pool identified by its SHA-256 "
    "digest under the active consent document identified by its SHA-256 digest. "
    "I understand that this acceptance applies only to that pool activation, that "
    "my identity is not collected, and that I may stop reviewing at any time."
)
RECONSENT_STATEMENT_SHA256 = hashlib.sha256(RECONSENT_STATEMENT.encode()).hexdigest()
CONTROL_NAMESPACE = uuid.UUID("d7ba3136-f29c-4c5b-91e1-6a323847028d")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def reviewer_control_lock(session: Session, reviewer_id: str) -> ExpertReviewer | None:
    """Serialize admission, rotation, and re-consent for one reviewer."""

    return session.scalar(
        select(ExpertReviewer).where(ExpertReviewer.id == reviewer_id).with_for_update()
    )


def admission_activation_sha256(
    *, reviewer_id: str, pool_sha256: str, admission_decision_sha256: str
) -> str:
    return canonical_sha256(
        {
            "kind": "anonymous_external_initial_pool_activation",
            "reviewer_id": reviewer_id,
            "pool_sha256": pool_sha256,
            "admission_decision_sha256": admission_decision_sha256,
        }
    )


def rotation_activation_sha256(
    *,
    reviewer_id: str,
    prior_activation_sha256: str,
    prior_pool_sha256: str,
    replacement_pool_sha256: str,
    rotation_event_id: str,
) -> str:
    return canonical_sha256(
        {
            "kind": "anonymous_external_rotated_pool_activation",
            "reviewer_id": reviewer_id,
            "prior_activation_sha256": prior_activation_sha256,
            "prior_pool_sha256": prior_pool_sha256,
            "replacement_pool_sha256": replacement_pool_sha256,
            "rotation_event_id": rotation_event_id,
        }
    )


def repaired_activation_sha256(
    *, reviewer_id: str, pool_sha256: str, rotation_event_id: str
) -> str:
    """Create a fresh activation epoch when repairing a legacy rotation."""

    return canonical_sha256(
        {
            "kind": "anonymous_external_legacy_rotation_activation_repair",
            "reviewer_id": reviewer_id,
            "pool_sha256": pool_sha256,
            "rotation_event_id": rotation_event_id,
        }
    )


def _expected_reconsent_payload(reviewer: ExpertReviewer) -> dict[str, Any] | None:
    profile = reviewer.profile_json
    pool_sha256 = profile.get("anonymous_external_pool_sha256")
    activation_sha256 = profile.get("anonymous_external_pool_activation_sha256")
    consent_sha256 = profile.get("consent_document_sha256")
    if not all(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value)
        for value in (pool_sha256, activation_sha256, consent_sha256)
    ):
        return None
    payload = {
        "schema_version": RECONSENT_SCHEMA_VERSION,
        "pool_sha256": pool_sha256,
        "pool_activation_sha256": activation_sha256,
        "consent_document_sha256": consent_sha256,
        "consent_statement_sha256": RECONSENT_STATEMENT_SHA256,
        "voluntary_participation_accepted": True,
        "pool_specific_consent_accepted": True,
        "identity_collection_prohibited": True,
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def reconsent_event_id(reviewer: ExpertReviewer) -> str | None:
    payload = _expected_reconsent_payload(reviewer)
    if payload is None:
        return None
    return str(
        uuid.uuid5(
            CONTROL_NAMESPACE,
            (
                f"reconsent:{reviewer.id}:{payload['pool_sha256']}:"
                f"{payload['pool_activation_sha256']}:"
                f"{payload['consent_document_sha256']}"
            ),
        )
    )


def anonymous_pool_reconsented(session: Session, reviewer: ExpertReviewer) -> bool:
    """Require exact immutable evidence for the reviewer's current pool epoch."""

    expected = _expected_reconsent_payload(reviewer)
    event_id = reconsent_event_id(reviewer)
    if expected is None or event_id is None:
        return False
    event = session.get(RunEvent, event_id)
    return bool(
        event is not None
        and event.entity_type == "expert_reviewer"
        and event.entity_id == reviewer.id
        and event.event_type == RECONSENT_EVENT_TYPE
        and event.payload_json == expected
    )


def append_pool_reconsent(session: Session, reviewer: ExpertReviewer) -> tuple[RunEvent, bool]:
    """Append one deterministic re-consent event for the current pool epoch."""

    payload = _expected_reconsent_payload(reviewer)
    event_id = reconsent_event_id(reviewer)
    if payload is None or event_id is None:
        raise ValueError("anonymous reviewer pool activation evidence is incomplete")
    existing = session.get(RunEvent, event_id)
    if existing is not None:
        if not (
            existing.entity_type == "expert_reviewer"
            and existing.entity_id == reviewer.id
            and existing.event_type == RECONSENT_EVENT_TYPE
            and existing.payload_json == payload
        ):
            raise ValueError("anonymous reviewer re-consent event has drifted")
        return existing, False
    event = RunEvent(
        id=event_id,
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type=RECONSENT_EVENT_TYPE,
        payload_json=payload,
    )
    session.add(event)
    session.flush()
    return event, True
