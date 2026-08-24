from __future__ import annotations

from datetime import UTC, datetime

import flavourbench.main as main_module
from flavourbench.database import init_database, session_scope
from flavourbench.engine import claim_job
from flavourbench.models import Job
from flavourbench.snapshot_worker import process_leaderboard_snapshot_job


def _queue_snapshot_job() -> tuple[str, str, int]:
    init_database()
    with session_scope() as session:
        job = Job(
            kind="leaderboard_snapshot",
            battle_id=None,
            payload_json={
                "season": "season-1",
                "track": "model_arena",
                "cohort": "expert_independent",
                "category": "all",
                "data_stratum": "controlled",
                "controlled_run_id": "controlled-run-test",
                "analysis_request_sha256": "a" * 64,
            },
            status="queued",
            max_attempts=1,
            available_at=datetime.now(UTC),
        )
        session.add(job)
        session.flush()
        job_id = job.id
    with session_scope() as session:
        lease = claim_job(session, "snapshot-worker-test")
        assert lease is not None
    assert lease[0] == job_id
    return lease


def test_snapshot_worker_persists_content_addresses(monkeypatch) -> None:
    lease = _queue_snapshot_job()

    def create_snapshot(*args, **kwargs):
        assert args[1:] == (
            "season-1",
            "model_arena",
            "expert_independent",
            "all",
            "controlled",
            "controlled-run-test",
        )
        assert kwargs == {}
        return {
            "snapshotId": "snapshot-test",
            "payloadSha256": "b" * 64,
            "inputEvidenceSha256": "c" * 64,
            "evidenceCutoffAt": "2026-08-01T00:00:00Z",
        }

    monkeypatch.setattr(main_module, "_create_leaderboard_snapshot", create_snapshot)
    process_leaderboard_snapshot_job(*lease)

    with session_scope() as session:
        job = session.get(Job, lease[0])
        assert job is not None
        assert job.status == "complete"
        assert job.completed_at is not None
        assert job.last_error is None
        assert job.payload_json["snapshot_id"] == "snapshot-test"
        assert job.payload_json["payload_sha256"] == "b" * 64
        assert job.payload_json["input_evidence_sha256"] == "c" * 64


def test_snapshot_worker_fails_closed_without_requeue(monkeypatch) -> None:
    lease = _queue_snapshot_job()

    def fail_snapshot(*args, **kwargs):
        raise RuntimeError("bootstrap contract failed")

    monkeypatch.setattr(main_module, "_create_leaderboard_snapshot", fail_snapshot)
    process_leaderboard_snapshot_job(*lease)

    with session_scope() as session:
        job = session.get(Job, lease[0])
        assert job is not None
        assert job.status == "failed"
        assert job.completed_at is not None
        assert job.last_error == "RuntimeError: bootstrap contract failed"
