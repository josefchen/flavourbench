from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Battle, RunEvent, Task

TASK_LIFECYCLE_SCHEMA_VERSION = "flavourbench-task-lifecycle-v1"
TASK_FIRST_USE_SCHEMA_VERSION = "flavourbench-task-first-use-v1"


class TaskLifecycleError(ValueError):
    """A confirmatory task has an incomplete or contradictory event history."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise TaskLifecycleError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskLifecycleError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise TaskLifecycleError(f"{field} must carry an explicit UTC offset")
    return parsed.astimezone(UTC)


def task_lifecycle_seal_sha256(
    *,
    task_public_id: str,
    task_revision: int,
    candidate_record_sha256: str,
    task_record_sha256: str,
    task_evidence_root_sha256: str,
    authored_at: datetime,
    sealed_at: datetime,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": TASK_LIFECYCLE_SCHEMA_VERSION,
            "task_public_id": task_public_id,
            "task_revision": task_revision,
            "candidate_record_sha256": candidate_record_sha256,
            "task_record_sha256": task_record_sha256,
            "task_evidence_root_sha256": task_evidence_root_sha256,
            "authored_at": _as_utc(authored_at).isoformat(),
            "sealed_at": _as_utc(sealed_at).isoformat(),
        }
    )


@dataclass(frozen=True)
class VerifiedTaskLifecycle:
    authored_at: datetime
    sealed_at: datetime
    first_used_at: datetime | None
    released_at: datetime | None
    retired_at: datetime | None
    lifecycle_seal_sha256: str


def verify_task_lifecycle(session: Session, task: Task) -> VerifiedTaskLifecycle:
    provenance = task.provenance_json
    if not isinstance(provenance, dict) or provenance.get("confirmatory_eligible") is not True:
        raise TaskLifecycleError("task is not a confirmatory task")
    events = list(
        session.scalars(
            select(RunEvent)
            .where(RunEvent.entity_type == "task", RunEvent.entity_id == task.id)
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )
    by_type: dict[str, list[RunEvent]] = {}
    for event in events:
        by_type.setdefault(event.event_type, []).append(event)
    authorship = by_type.get("confirmatory_task_authorship_recorded", [])
    seals = by_type.get("confirmatory_task_sealed", [])
    first_uses = by_type.get("confirmatory_task_first_used", [])
    releases = by_type.get("confirmatory_task_released", [])
    retirements = by_type.get("confirmatory_task_retired", [])
    if len(authorship) != 1 or len(seals) != 1:
        raise TaskLifecycleError("task requires exactly one authorship and one seal event")
    if any(len(rows) > 1 for rows in (first_uses, releases, retirements)):
        raise TaskLifecycleError("task lifecycle contains a duplicate terminal event")

    authored_payload = authorship[0].payload_json
    seal_payload = seals[0].payload_json
    authored_at = _parse_utc(authored_payload.get("authored_at"), field="authored_at")
    sealed_at = _parse_utc(seal_payload.get("sealed_at"), field="sealed_at")
    if authored_at > sealed_at or _as_utc(task.created_at) != sealed_at:
        raise TaskLifecycleError("task was sealed before authorship or at a different instant")
    expected_seal = task_lifecycle_seal_sha256(
        task_public_id=task.public_id,
        task_revision=task.revision,
        candidate_record_sha256=str(provenance.get("candidate_record_sha256", "")),
        task_record_sha256=str(provenance.get("task_record_sha256", "")),
        task_evidence_root_sha256=str(provenance.get("task_evidence_root_sha256", "")),
        authored_at=authored_at,
        sealed_at=sealed_at,
    )
    if (
        authored_payload.get("public_id") != task.public_id
        or authored_payload.get("revision") != task.revision
        or authored_payload.get("candidate_record_sha256")
        != provenance.get("candidate_record_sha256")
        or seal_payload.get("public_id") != task.public_id
        or seal_payload.get("revision") != task.revision
        or seal_payload.get("prompt_sha256") != task.prompt_sha256
        or seal_payload.get("task_record_sha256") != provenance.get("task_record_sha256")
        or seal_payload.get("task_evidence_root_sha256")
        != provenance.get("task_evidence_root_sha256")
        or seal_payload.get("lifecycle_seal_sha256") != expected_seal
        or provenance.get("task_lifecycle_seal_sha256") != expected_seal
        or provenance.get("authored_at") != authored_at.isoformat()
        or provenance.get("sealed_at") != sealed_at.isoformat()
    ):
        raise TaskLifecycleError("task lifecycle seal does not reproduce")

    def optional_time(rows: list[RunEvent], field: str) -> datetime | None:
        return _parse_utc(rows[0].payload_json.get(field), field=field) if rows else None

    first_used_at = optional_time(first_uses, "first_used_at")
    released_at = optional_time(releases, "released_at")
    retired_at = optional_time(retirements, "retired_at")
    for value in (first_used_at, released_at, retired_at):
        if value is not None and value < sealed_at:
            raise TaskLifecycleError("task lifecycle transition predates its seal")
    if released_at is not None and retired_at is not None and retired_at < released_at:
        raise TaskLifecycleError("task retirement predates public release")
    if first_uses:
        payload = first_uses[0].payload_json
        expected_first_use = _canonical_sha256(
            {
                "schema_version": TASK_FIRST_USE_SCHEMA_VERSION,
                "task_id": task.id,
                "task_public_id": task.public_id,
                "task_revision": task.revision,
                "task_lifecycle_seal_sha256": expected_seal,
                "battle_id": payload.get("battle_id"),
                "season_id": task.season_id,
                "prompt_sha256": task.prompt_sha256,
                "first_used_at": first_used_at.isoformat() if first_used_at else None,
            }
        )
        if (
            payload.get("task_first_use_sha256") != expected_first_use
            or payload.get("task_lifecycle_seal_sha256") != expected_seal
            or payload.get("task_public_id") != task.public_id
            or payload.get("task_revision") != task.revision
            or payload.get("season_id") != task.season_id
            or payload.get("prompt_sha256") != task.prompt_sha256
        ):
            raise TaskLifecycleError("task first-use event does not reproduce")
    return VerifiedTaskLifecycle(
        authored_at=authored_at,
        sealed_at=sealed_at,
        first_used_at=first_used_at,
        released_at=released_at,
        retired_at=retired_at,
        lifecycle_seal_sha256=expected_seal,
    )


def record_task_first_use(session: Session, *, task: Task, battle: Battle) -> RunEvent:
    """Atomically record the first task exposure, or return its existing event."""

    locked_task = session.scalar(select(Task).where(Task.id == task.id).with_for_update())
    if locked_task is None:
        raise TaskLifecycleError("task disappeared before first use")
    lifecycle = verify_task_lifecycle(session, locked_task)
    existing = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "task",
            RunEvent.entity_id == task.id,
            RunEvent.event_type == "confirmatory_task_first_used",
        )
    )
    if existing is not None:
        return existing
    if lifecycle.retired_at is not None:
        raise TaskLifecycleError("retired task cannot enter a new battle")
    first_used_at = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema_version": TASK_FIRST_USE_SCHEMA_VERSION,
        "task_id": task.id,
        "task_public_id": task.public_id,
        "task_revision": task.revision,
        "task_lifecycle_seal_sha256": lifecycle.lifecycle_seal_sha256,
        "battle_id": battle.id,
        "season_id": task.season_id,
        "prompt_sha256": task.prompt_sha256,
        "first_used_at": first_used_at.isoformat(),
    }
    payload["task_first_use_sha256"] = _canonical_sha256(payload)
    event = RunEvent(
        entity_type="task",
        entity_id=task.id,
        event_type="confirmatory_task_first_used",
        payload_json=payload,
        created_at=first_used_at,
    )
    session.add(event)
    session.flush()
    return event


def task_lifecycle_matches(
    session: Session,
    task: Task,
    *,
    require_first_use: bool = False,
) -> bool:
    try:
        lifecycle = verify_task_lifecycle(session, task)
    except TaskLifecycleError:
        return False
    return lifecycle.first_used_at is not None if require_first_use else True
