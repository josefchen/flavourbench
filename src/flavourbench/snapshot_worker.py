from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select

from .database import session_scope
from .models import Job


def _leased_job(
    session,
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
    *,
    lock: bool,
) -> Job | None:
    statement = select(Job).where(
        Job.id == job_id,
        Job.kind == "leaderboard_snapshot",
        Job.status == "running",
        Job.claimed_by == claimed_by,
        Job.attempts == claim_attempt,
    )
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement)


def process_leaderboard_snapshot_job(
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
) -> None:
    """Run the expensive frozen bootstrap outside the API request lifecycle."""

    with session_scope() as session:
        job = _leased_job(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=False,
        )
        if job is None:
            return
        request: dict[str, Any] = dict(job.payload_json)

    try:
        # Imported lazily so the worker does not create an engine/main import cycle.
        from .main import _create_leaderboard_snapshot

        with session_scope() as session:
            result = _create_leaderboard_snapshot(
                session,
                str(request["season"]),
                str(request["track"]),
                str(request["cohort"]),
                str(request["category"]),
                str(request["data_stratum"]),
                (
                    str(request["controlled_run_id"])
                    if request.get("controlled_run_id") is not None
                    else None
                ),
            )
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else f"{type(exc).__name__}: {exc}"
        with session_scope() as session:
            job = _leased_job(
                session,
                job_id,
                claimed_by,
                claim_attempt,
                lock=True,
            )
            if job is not None:
                job.status = "failed"
                job.last_error = str(detail)[:4_000]
                job.completed_at = datetime.now(UTC)
        return

    with session_scope() as session:
        job = _leased_job(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        if job is None:
            return
        job.payload_json = {
            **request,
            "snapshot_id": result["snapshotId"],
            "payload_sha256": result["payloadSha256"],
            "input_evidence_sha256": result["inputEvidenceSha256"],
            "evidence_cutoff_at": result["evidenceCutoffAt"],
        }
        job.status = "complete"
        job.last_error = None
        job.completed_at = datetime.now(UTC)
