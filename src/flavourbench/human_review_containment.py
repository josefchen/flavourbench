"""Append-only containment for an inadmissible human-review QA batch."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import SessionLocal
from .engine import is_complete_finish_reason
from .expert_calibration import TASK_SCOPE_QUARANTINE, TASK_SCOPE_REVIEW_SHA256
from .expert_review import canonical_sha256
from .models import (
    Battle,
    ExpertReviewer,
    Incident,
    ResponseArm,
    RunEvent,
    Task,
)

INACTIVE_CONSENT_SHA256 = "2e28f1b6ea9521e4e9e17503686a44ee38454bd2409ddf336cf2c567f3cae34d"


class HumanReviewContainmentError(RuntimeError):
    """The review batch could not be contained without weakening its audit trail."""


def _latest_anonymous_reviewer(session: Session, *, for_update: bool = False) -> ExpertReviewer:
    statement = (
        select(ExpertReviewer)
        .where(
            ExpertReviewer.active.is_(True),
            ExpertReviewer.cohort == "expert_independent",
            ExpertReviewer.profile_json["admission_pathway"].as_string()
            == "anonymous_external_rater",
        )
        .order_by(ExpertReviewer.created_at.desc())
    )
    if for_update:
        statement = statement.with_for_update()
    reviewer = session.scalar(statement)
    if reviewer is None:
        raise HumanReviewContainmentError("anonymous external reviewer record is unavailable")
    return reviewer


def _latest_review_session(session: Session, reviewer_id: str) -> RunEvent:
    event = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.event_type == "expert_review_session_opened",
            RunEvent.payload_json["reviewer_id"].as_string() == reviewer_id,
        )
        .order_by(RunEvent.created_at.desc())
    )
    if event is None:
        raise HumanReviewContainmentError("anonymous review session is unavailable")
    return event


def _latest_review_session_for_pool(
    session: Session, reviewer_id: str, pool_sha256: str
) -> RunEvent | None:
    """Return only a session whose immutable opening event names the exact pool."""

    return session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.event_type == "expert_review_session_opened",
            RunEvent.payload_json["reviewer_id"].as_string() == reviewer_id,
            RunEvent.payload_json["anonymous_external_pool_sha256"].as_string() == pool_sha256,
        )
        .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
    )


def _append_event_once(
    session: Session,
    *,
    entity_type: str,
    entity_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> bool:
    evidence_sha256 = canonical_sha256(payload)
    existing = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == entity_type,
            RunEvent.entity_id == entity_id,
            RunEvent.event_type == event_type,
            RunEvent.payload_json["evidence_sha256"].as_string() == evidence_sha256,
        )
    )
    if existing is not None:
        return False
    session.add(
        RunEvent(
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event_type,
            payload_json={**payload, "evidence_sha256": evidence_sha256},
        )
    )
    return True


def _append_incident_once(
    session: Session,
    *,
    severity: str,
    code: str,
    detail: str,
    battle_id: str | None = None,
) -> bool:
    existing = session.scalar(
        select(Incident).where(
            Incident.code == code,
            Incident.battle_id == battle_id,
            Incident.detail == detail,
        )
    )
    if existing is not None:
        return False
    session.add(
        Incident(
            severity=severity,
            code=code,
            detail=detail,
            battle_id=battle_id,
        )
    )
    return True


def _parse_safety_arms(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        task_id, separator, side = value.partition(":")
        if not separator or side not in {"left", "right"} or not task_id:
            raise HumanReviewContainmentError(
                "reported safety arms must use TASK_PUBLIC_ID:left or TASK_PUBLIC_ID:right"
            )
        parsed[task_id] = side
    return parsed


def contain_review_batch(
    session: Session,
    *,
    recognized_repeats: int,
    reported_truncations_minimum: int,
    reported_safety_arms: dict[str, str],
) -> dict[str, Any]:
    reviewer = _latest_anonymous_reviewer(session)
    review_session = _latest_review_session(session, reviewer.id)
    review_session_id = review_session.entity_id
    consent_sha256 = str(reviewer.profile_json.get("consent_document_sha256") or "")
    if consent_sha256 != INACTIVE_CONSENT_SHA256:
        raise HumanReviewContainmentError(
            "reviewer consent hash no longer matches the contained incident"
        )

    submitted = list(
        session.scalars(
            select(RunEvent).where(
                RunEvent.entity_type == "expert_review_assignment",
                RunEvent.event_type == "expert_review_assignment_submitted",
                RunEvent.payload_json["review_session_id"].as_string() == review_session_id,
            )
        ).all()
    )
    primary = [event for event in submitted if event.payload_json.get("mode") == "primary"]
    repeats = [
        event for event in submitted if event.payload_json.get("mode") == "reliability_repeat"
    ]
    if recognized_repeats > len(repeats):
        raise HumanReviewContainmentError("recognized repeat count exceeds stored repeats")

    battle_ids = {
        str(event.payload_json.get("battle_id"))
        for event in primary
        if event.payload_json.get("battle_id")
    }
    battles = {
        battle.id: battle
        for battle in session.scalars(select(Battle).where(Battle.id.in_(battle_ids))).all()
    }
    arms = list(
        session.scalars(select(ResponseArm).where(ResponseArm.battle_id.in_(battle_ids))).all()
    )
    non_normal_arms = [
        arm
        for arm in arms
        if arm.status != "complete" or not is_complete_finish_reason(arm.finish_reason)
    ]
    non_normal_battles = {arm.battle_id for arm in non_normal_arms}
    if len(non_normal_arms) < reported_truncations_minimum:
        raise HumanReviewContainmentError(
            "stored finish reasons do not support the reported truncation minimum"
        )

    tasks = {
        task.id: task
        for task in session.scalars(
            select(Task).where(Task.id.in_({battle.task_id for battle in battles.values()}))
        ).all()
    }
    reviewed_tasks_by_public_id = {task.public_id: task for task in tasks.values()}
    reviewed_scope_quarantine = sorted(
        set(reviewed_tasks_by_public_id).intersection(TASK_SCOPE_QUARANTINE)
    )

    safety_rows: list[tuple[Task, Battle, ResponseArm]] = []
    for task_public_id, side in reported_safety_arms.items():
        task = reviewed_tasks_by_public_id.get(task_public_id)
        if task is None:
            raise HumanReviewContainmentError(
                f"reported safety task was not reviewed: {task_public_id}"
            )
        battle = next(
            (row for row in battles.values() if row.task_id == task.id),
            None,
        )
        if battle is None:
            raise HumanReviewContainmentError(
                f"reported safety task has no reviewed battle: {task_public_id}"
            )
        arm = next(
            (row for row in arms if row.battle_id == battle.id and row.side == side),
            None,
        )
        if arm is None:
            raise HumanReviewContainmentError(
                f"reported safety arm is unavailable: {task_public_id}:{side}"
            )
        safety_rows.append((task, battle, arm))

    self_report = {
        "source": "direct_reviewer_self_report",
        "recognized_reliability_repeats": recognized_repeats,
        "repeat_scores_deliberately_mirrored": recognized_repeats > 0,
        "reported_truncated_responses_minimum": reported_truncations_minimum,
        "reported_potential_safety_hazards": len(safety_rows),
        "reported_evidence_misuse_as_decisive_signal": True,
        "interpretation": "self_reported_operational_quality_signal_not_adjudicated_result",
    }
    appended_events = int(
        _append_event_once(
            session,
            entity_type="expert_review_session",
            entity_id=review_session_id,
            event_type="expert_review_batch_self_reported_quality_signal",
            payload=self_report,
        )
    )

    restriction = {
        "evidence_status": "restricted_operational_qa",
        "consent_document_sha256": consent_sha256,
        "consent_document_status": "inactive_draft",
        "completed_presentations": len(submitted),
        "primary_judgments": len(primary),
        "reliability_repeats": len(repeats),
        "recognized_reliability_repeats": recognized_repeats,
        "non_normal_response_arms": len(non_normal_arms),
        "affected_primary_pairs": len(non_normal_battles),
        "reviewed_scope_quarantine_tasks": reviewed_scope_quarantine,
        "reported_potential_safety_hazards": len(safety_rows),
        "research_use": False,
        "paper_use": False,
        "ranking_use": False,
        "leaderboard_use": False,
        "raw_records_preserved": True,
        "remediation": (
            "institutional determination, active consent, re-consent if reuse is allowed, "
            "finish-clean recollection, and qualification-matched specialist adjudication"
        ),
    }
    appended_events += int(
        _append_event_once(
            session,
            entity_type="expert_review_session",
            entity_id=review_session_id,
            event_type="expert_review_batch_restricted",
            payload=restriction,
        )
    )

    appended_incidents = int(
        _append_incident_once(
            session,
            severity="critical",
            code="inactive_expert_consent_review_batch",
            detail=(
                "Human-review records were collected under a document marked inactive. "
                "The batch is restricted to operational QA and excluded from research, paper, "
                "ranking, and leaderboard use."
            ),
        )
    )

    for battle_id in sorted(non_normal_battles):
        finish_reasons = sorted(
            str(arm.finish_reason or "missing")
            for arm in non_normal_arms
            if arm.battle_id == battle_id
        )
        appended_incidents += int(
            _append_incident_once(
                session,
                severity="high",
                code="review_pair_contains_non_normal_final_response",
                detail=(
                    "A human-reviewed pair contained a final response with a non-normal finish "
                    f"state: {', '.join(finish_reasons)}. The pair is excluded."
                ),
                battle_id=battle_id,
            )
        )
        for arm in (row for row in non_normal_arms if row.battle_id == battle_id):
            appended_events += int(
                _append_event_once(
                    session,
                    entity_type="response_arm",
                    entity_id=arm.id,
                    event_type="response_arm_non_normal_completion_detected",
                    payload={
                        "battle_id": battle_id,
                        "finish_reason": arm.finish_reason,
                        "ranking_use": False,
                        "review_use": False,
                    },
                )
            )

    for task_public_id in reviewed_scope_quarantine:
        task = reviewed_tasks_by_public_id[task_public_id]
        appended_events += int(
            _append_event_once(
                session,
                entity_type="task",
                entity_id=task.id,
                event_type="task_general_track_scope_quarantined",
                payload={
                    "task_public_id": task_public_id,
                    "scope": "specialist_review_required",
                    "general_track_eligible": False,
                    "task_scope_review_sha256": TASK_SCOPE_REVIEW_SHA256,
                },
            )
        )

    for task, battle, arm in safety_rows:
        appended_events += int(
            _append_event_once(
                session,
                entity_type="response_arm",
                entity_id=arm.id,
                event_type="reviewer_reported_potential_safety_hazard",
                payload={
                    "battle_id": battle.id,
                    "task_public_id": task.public_id,
                    "condition": arm.condition,
                    "status": "pending_qualified_food_safety_adjudication",
                    "verified_safety_error": False,
                    "ranking_use": False,
                },
            )
        )
        appended_incidents += int(
            _append_incident_once(
                session,
                severity="critical",
                code="reviewer_reported_potential_safety_hazard",
                detail=(
                    "A reviewer reported a potentially unsafe dosing recommendation. "
                    "The report is unverified pending qualified food-safety adjudication, and "
                    "the comparison is excluded."
                ),
                battle_id=battle.id,
            )
        )

    session.commit()
    return {
        "reviewSessionId": review_session_id,
        "evidenceStatus": "restricted_operational_qa",
        "completedPresentations": len(submitted),
        "primaryJudgments": len(primary),
        "reliabilityRepeats": len(repeats),
        "nonNormalResponseArms": len(non_normal_arms),
        "affectedPrimaryPairs": len(non_normal_battles),
        "scopeQuarantineTasksReviewed": len(reviewed_scope_quarantine),
        "reportedPotentialSafetyHazards": len(safety_rows),
        "eventsAppended": appended_events,
        "incidentsAppended": appended_incidents,
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognized-repeats", type=int, required=True)
    parser.add_argument("--reported-truncations-minimum", type=int, required=True)
    parser.add_argument("--reported-safety-arm", action="append", default=[])
    args = parser.parse_args()
    with SessionLocal() as session:
        result = contain_review_batch(
            session,
            recognized_repeats=args.recognized_repeats,
            reported_truncations_minimum=args.reported_truncations_minimum,
            reported_safety_arms=_parse_safety_arms(args.reported_safety_arm),
        )
    print(result)


if __name__ == "__main__":
    run()
