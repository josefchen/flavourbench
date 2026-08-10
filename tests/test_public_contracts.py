from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import flavourbench.arena as arena
from flavourbench.config import get_settings
from flavourbench.database import assert_production_fixture_free, session_scope
from flavourbench.engine import _record_known_zero_cost, reconcile_battle_cost, run_worker_once
from flavourbench.main import app
from flavourbench.models import (
    Battle,
    ExpertReviewer,
    Job,
    ResponseArm,
    RunEvent,
    Vote,
)

SERVICE = "test-service-token"
ADMIN = "test-admin-token"
PSEUDONYM = "e" * 64
HEADERS = {
    "X-FlavourBench-Service-Token": SERVICE,
    "X-FlavourBench-Pseudonym": PSEUDONYM,
}
ADMIN_HEADERS = {**HEADERS, "X-FlavourBench-Admin-Token": ADMIN}


def create_battle(
    client: TestClient,
    nonce: str,
    *,
    pseudonym: str = PSEUDONYM,
    category: str = "composition",
    prompt: str = "Compose a practical cauliflower main with black garlic and clear doneness cues.",
    consent: bool = False,
) -> str:
    response = client.post(
        "/v1/battles",
        headers={**HEADERS, "X-FlavourBench-Pseudonym": pseudonym},
        json={
            "prompt": prompt,
            "category": category,
            "researchConsent": consent,
            "clientNonce": nonce,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["battleId"]


def complete_jobs(worker: str) -> None:
    while asyncio.run(run_worker_once(worker)):
        pass


def fail_queued_battle(session, battle: Battle) -> None:
    arms = session.scalars(
        select(ResponseArm).where(ResponseArm.battle_id == battle.id).order_by(ResponseArm.side)
    ).all()
    job = session.scalar(select(Job).where(Job.battle_id == battle.id))
    assert len(arms) == 2 and job is not None and job.status == "queued"
    job.status = "running"
    session.flush()
    arm_completed_at = datetime.now(UTC) + timedelta(milliseconds=10)
    for arm in arms:
        arm.status = "failed"
        arm.error_code = "TestFailure"
        arm.error_detail = "Deliberate provider-free failure fixture."
        arm.completed_at = arm_completed_at
        _record_known_zero_cost(
            session,
            battle,
            arm,
            reason="public_failure_contract_fixture",
        )
    session.flush()
    battle.status = "failed"
    battle.completed_at = arm_completed_at + timedelta(milliseconds=1)
    job.status = "failed"
    job.completed_at = battle.completed_at
    session.flush()
    reconcile_battle_cost(session, battle)


def test_health_service_and_admin_token_boundaries() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ready",
            "executionMode": "mock",
            "database": "ready",
            "databaseDialect": "sqlite",
            "databaseRole": "sqlite",
            "schemaRevision": "unmanaged",
        }

        missing = client.get("/v1/models")
        wrong = client.get("/v1/models", headers={"X-FlavourBench-Service-Token": "wrong"})
        assert missing.status_code == wrong.status_code == 401
        assert missing.json()["detail"] == wrong.json()["detail"] == "invalid service token"
        assert client.get("/v1/models", headers=HEADERS).status_code == 200

        missing_admin = client.post(
            "/v1/admin/leaderboards/snapshot?season=missing",
            headers=HEADERS,
        )
        wrong_admin = client.post(
            "/v1/admin/leaderboards/snapshot?season=missing",
            headers={**HEADERS, "X-FlavourBench-Admin-Token": "wrong"},
        )
        valid_admin = client.post(
            "/v1/admin/leaderboards/snapshot?season=missing",
            headers=ADMIN_HEADERS,
        )
        assert missing_admin.status_code == wrong_admin.status_code == 401
        assert missing_admin.json()["detail"] == "invalid admin token"
        assert valid_admin.status_code == 404
        assert valid_admin.json()["detail"] == "season not found"


def test_production_fixture_guard_rejects_the_development_seed() -> None:
    with TestClient(app):
        with session_scope() as session:
            assert_production_fixture_free(session, environment="development")
            with pytest.raises(RuntimeError, match="legacy fixture records"):
                assert_production_fixture_free(session, environment="production")


def test_battle_schema_pseudonym_normalization_and_persisted_construction() -> None:
    with TestClient(app) as client:
        no_pseudonym = client.post(
            "/v1/battles",
            headers={"X-FlavourBench-Service-Token": SERVICE},
            json={
                "prompt": "This prompt is long enough to pass schema validation.",
                "category": "composition",
                "clientNonce": "public-contract-no-pseudonym",
            },
        )
        assert no_pseudonym.status_code == 401
        assert no_pseudonym.json()["detail"] == "request pseudonym is missing"

        for payload in [
            {"prompt": "short", "category": "composition", "clientNonce": "validnonce"},
            {
                "prompt": "This prompt has an unsupported category value.",
                "category": "unknown",
                "clientNonce": "validnonce2",
            },
            {
                "prompt": "This prompt has a nonce that is too short.",
                "category": "composition",
                "clientNonce": "tiny",
            },
        ]:
            assert client.post("/v1/battles", headers=HEADERS, json=payload).status_code == 422

        battle_id = create_battle(
            client,
            "public-contract-construction-0001",
            prompt="  Roast   cauliflower\nwith black garlic and give two doneness cues.  ",
            consent=True,
        )
        with session_scope() as session:
            battle = session.get(Battle, battle_id)
            assert battle is not None
            assert (
                battle.prompt == "Roast cauliflower with black garlic and give two doneness cues."
            )
            assert battle.prompt_sha256 == hashlib.sha256(battle.prompt.encode()).hexdigest()
            assert battle.run_class == "mock"
            assert battle.rank_eligible is False
            assert battle.data_stratum == "public_freeform"
            assert battle.task_id is None and battle.task_revision is None
            assert battle.manifest_sha256 == "unfrozen"
            assert battle.scheduler_version == "coverage-balanced-server-random-v1"
            assert len(battle.assignment_seed) == 64
            assert "/" in battle.track_assignment_probability
            assert "/" in battle.model_assignment_probability
            assert battle.side_assignment_probability == "1/2"
            assert battle.release_review_status == "pending"
            assert battle.retention_until is not None
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle_id)
            ).all()
            assert sorted(arm.side for arm in arms) == ["left", "right"]
            assert all(arm.status == "queued" for arm in arms)
            assert all(arm.decoding_json["structured_output"] is True for arm in arms)
            assert all(arm.decoding_json["max_tool_rounds"] == 8 for arm in arms)
            jobs = session.scalars(select(Job).where(Job.battle_id == battle_id)).all()
            assert len(jobs) == 1 and jobs[0].kind == "generate_battle"
            events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type == "battle",
                    RunEvent.entity_id == battle_id,
                    RunEvent.event_type == "battle_queued",
                )
            ).all()
            assert len(events) == 1


def test_battle_creation_idempotency_and_per_pseudonym_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = get_settings()
    limited = real.model_copy(update={"admission_max_battles": 1})
    monkeypatch.setattr(arena, "get_settings", lambda: limited)
    pseudonym = "f" * 64
    with TestClient(app) as client:
        first_id = create_battle(
            client,
            "public-contract-rate-0001",
            pseudonym=pseudonym,
        )
        repeated_id = create_battle(
            client,
            "public-contract-rate-0001",
            pseudonym=pseudonym,
        )
        assert repeated_id == first_id
        rejected = client.post(
            "/v1/battles",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": pseudonym},
            json={
                "prompt": "A second distinct prompt should reach the admission limit.",
                "category": "composition",
                "clientNonce": "public-contract-rate-0002",
            },
        )
        assert rejected.status_code == 429
        assert rejected.json()["detail"] == "battle admission limit reached"
        other_id = create_battle(
            client,
            "public-contract-rate-0002",
            pseudonym="0" * 64,
        )
        assert other_id != first_id

        with session_scope() as session:
            assert session.scalar(select(func.count(Job.id)).where(Job.battle_id == first_id)) == 1
            assert (
                session.scalar(
                    select(func.count(ResponseArm.id)).where(ResponseArm.battle_id == first_id)
                )
                == 2
            )


def test_requester_polling_status_and_failure_contract() -> None:
    owner = "1" * 64
    with TestClient(app) as client:
        battle_id = create_battle(
            client,
            "public-contract-polling-0001",
            pseudonym=owner,
        )
        queued = client.get(
            f"/v1/battles/{battle_id}",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": owner},
        )
        assert queued.status_code == 200
        assert queued.json()["answers"] == []
        assert queued.json()["reveal"] is None
        assert queued.json()["error"] is None

        for pseudonym in ["2" * 64, "not-a-valid-pseudonym"]:
            hidden = client.get(
                f"/v1/battles/{battle_id}",
                headers={**HEADERS, "X-FlavourBench-Pseudonym": pseudonym},
            )
            assert hidden.status_code in {401, 404}
        missing = client.get(
            "/v1/battles/not-a-battle",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": owner},
        )
        assert missing.status_code == 404 and missing.json()["detail"] == "battle not found"

        with session_scope() as session:
            battle = session.get(Battle, battle_id)
            assert battle
            fail_queued_battle(session, battle)
        failed = client.get(
            f"/v1/battles/{battle_id}",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": owner},
        )
        assert failed.json()["answers"] == []
        assert failed.json()["reveal"] is None
        assert failed.json()["error"] == (
            "One or both answers failed; this battle cannot be voted on."
        )


def test_vote_validation_idempotency_conflicts_and_reveal() -> None:
    rater = "3" * 64
    with TestClient(app) as client:
        queued_id = create_battle(
            client,
            "public-contract-vote-queued",
            pseudonym=rater,
        )
        queued_vote = client.post(
            f"/v1/battles/{queued_id}/votes",
            headers={
                **HEADERS,
                "X-FlavourBench-Pseudonym": rater,
                "Idempotency-Key": "queued-vote-key",
            },
            json={"choice": "left", "reasonTags": []},
        )
        assert queued_vote.status_code == 409
        assert queued_vote.json()["detail"] == "battle is not voteable"

        complete_jobs("public-contract-vote-worker")
        invalid_tag = client.post(
            f"/v1/battles/{queued_id}/votes",
            headers={
                **HEADERS,
                "X-FlavourBench-Pseudonym": rater,
                "Idempotency-Key": "invalid-tag-key",
            },
            json={"choice": "left", "reasonTags": ["not_allowed"]},
        )
        assert invalid_tag.status_code == 422
        for key in ["", "x" * 121]:
            headers = {**HEADERS, "X-FlavourBench-Pseudonym": rater}
            if key:
                headers["Idempotency-Key"] = key
            response = client.post(
                f"/v1/battles/{queued_id}/votes",
                headers=headers,
                json={"choice": "tie", "reasonTags": []},
            )
            assert response.status_code == 400

        key = "public-contract-vote-key"
        payload = {
            "choice": "both_bad",
            "reasonTags": ["generic", "generic", "unclear"],
        }
        first = client.post(
            f"/v1/battles/{queued_id}/votes",
            headers={
                **HEADERS,
                "X-FlavourBench-Pseudonym": rater,
                "Idempotency-Key": key,
            },
            json=payload,
        )
        assert first.status_code == 200, first.text
        assert [arm["side"] for arm in first.json()["reveal"]["arms"]] == ["left", "right"]
        repeated = client.post(
            f"/v1/battles/{queued_id}/votes",
            headers={
                **HEADERS,
                "X-FlavourBench-Pseudonym": rater,
                "Idempotency-Key": key,
            },
            json=payload,
        )
        assert repeated.status_code == 200
        assert repeated.json()["voteId"] == first.json()["voteId"]
        second_vote = client.post(
            f"/v1/battles/{queued_id}/votes",
            headers={
                **HEADERS,
                "X-FlavourBench-Pseudonym": rater,
                "Idempotency-Key": "another-public-vote-key",
            },
            json={"choice": "left", "reasonTags": []},
        )
        assert second_vote.status_code == 409
        assert second_vote.json()["detail"] == "this battle already has a vote"

        visible = client.get(
            f"/v1/battles/{queued_id}",
            headers={**HEADERS, "X-FlavourBench-Pseudonym": rater},
        )
        assert visible.json()["reveal"] == first.json()["reveal"]
        with session_scope() as session:
            vote = session.get(Vote, first.json()["voteId"])
            assert vote and vote.reason_tags_json == ["generic", "unclear"]


def test_models_leaderboards_and_snapshot_metadata() -> None:
    with TestClient(app) as client:
        models = client.get("/v1/models?season=season-0", headers=HEADERS)
        assert models.status_code == 200
        payload = models.json()
        assert payload["catalogCount"] == len(payload["models"])
        assert payload["catalogCount"] >= 12
        names = [item["name"] for item in payload["models"]]
        assert names == sorted(names)
        assert any(item["status"] == "season_eligible" for item in payload["models"])
        assert all(
            {
                "id",
                "canonicalSlug",
                "name",
                "family",
                "openWeight",
                "status",
                "supportsTools",
                "supportsStructuredOutput",
                "contextLength",
                "slotRole",
            }
            <= item.keys()
            for item in payload["models"]
        )
        unknown_season = client.get("/v1/models?season=does-not-exist", headers=HEADERS)
        assert unknown_season.status_code == 200
        assert all(item["status"] != "season_eligible" for item in unknown_season.json()["models"])

        leaderboard = client.get(
            "/v1/leaderboards?season=season-0&track=model_arena&rater_cohort=public&task_family=nonexistent",
            headers=HEADERS,
        )
        assert leaderboard.status_code == 200
        board = leaderboard.json()
        assert board["rows"] == []
        assert board["official"] is False
        assert board["sampleNotice"] == "No release-approved public leaderboard snapshot"
        assert board["snapshotId"] is None
        assert board["data_stratum"] == "public_freeform"
        assert board["budget"] == {
            "capMicros": 0,
            "usedMicros": 0,
            "reservedMicros": 0,
            "admissionThresholdBasisPoints": 8500,
            "drainThresholdBasisPoints": 9500,
            "hardStopBasisPoints": 10000,
        }
        assert client.get("/v1/leaderboards?season=missing", headers=HEADERS).status_code == 404
        assert client.get("/v1/leaderboards?track=invalid", headers=HEADERS).status_code == 422
        assert (
            client.get("/v1/leaderboards?rater_cohort=invalid", headers=HEADERS).status_code == 422
        )

        first = client.post(
            "/v1/admin/leaderboards/snapshot?season=season-0&track=model_arena&cohort=public&category=nonexistent",
            headers=ADMIN_HEADERS,
        )
        second = client.post(
            "/v1/admin/leaderboards/snapshot?season=season-0&track=model_arena&cohort=public&category=nonexistent",
            headers=ADMIN_HEADERS,
        )
        assert first.status_code == second.status_code == 409
        assert first.json()["detail"] == "snapshots require an active season"


def test_expert_invitation_authentication_storage_and_duplicate_code() -> None:
    reviewer_code = "public-contract-expert"
    invite_body = {
        "reviewer_code": reviewer_code,
        "qualified_families": ["substitution", "evidence"],
        "qualification_reference": "verified-culinary-practice-record",
        "qualification_verified": True,
        "affiliation_class": "independent_external",
        "conflict_disclosure_reference": "conflict-disclosure-none",
        "consent_document_sha256": "a" * 64,
        "training_material_sha256": "b" * 64,
        "calibration_set_sha256": "c" * 64,
        "calibration_accuracy": 0.9,
        "compensation_reference": "documented-volunteer-agreement",
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json=invite_body,
        )
        assert response.status_code == 200, response.text
        invitation = response.json()["invitation"]
        assert invitation
        assert "returned once" in response.json()["notice"]
        with session_scope() as session:
            reviewer = session.scalar(
                select(ExpertReviewer).where(ExpertReviewer.reviewer_code == reviewer_code)
            )
            assert reviewer is not None
            assert reviewer.invitation_sha256 == hashlib.sha256(invitation.encode()).hexdigest()
            assert invitation not in reviewer.invitation_sha256
            assert reviewer.qualification_verified is False
            assert reviewer.cohort == "expert_independent"
            assert reviewer.profile_json["calibration_accuracy"] is None
            assert reviewer.profile_json["claimed_calibration_accuracy"] == 0.9

        duplicate = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json=invite_body,
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"] == "reviewer code already exists"
        for body in [
            {"reviewer_code": "x", "qualified_families": ["evidence"]},
            {"reviewer_code": "invalid code", "qualified_families": ["evidence"]},
            {"reviewer_code": "no-families", "qualified_families": []},
            {
                **invite_body,
                "reviewer_code": "duplicate-families",
                "qualified_families": ["evidence", "evidence"],
            },
        ]:
            assert (
                client.post("/v1/admin/experts", headers=ADMIN_HEADERS, json=body).status_code
                == 422
            )

        for authorization in [None, "Bearer wrong"]:
            headers = {"X-FlavourBench-Service-Token": SERVICE}
            if authorization:
                headers["Authorization"] = authorization
            assignment = client.get("/v1/expert/assignments/next", headers=headers)
            assert assignment.status_code == 401
        valid = client.get(
            "/v1/expert/assignments/next",
            headers={
                "X-FlavourBench-Service-Token": SERVICE,
                "Authorization": f"Bearer {invitation}",
            },
        )
        assert valid.status_code == 410
        assert valid.json()["detail"] == (
            "legacy expert assignments are retired; use versioned review sessions"
        )


def test_consent_release_review_export_and_sanitization() -> None:
    pseudonym = "4" * 64
    with TestClient(app) as client:
        approved_id = create_battle(
            client,
            "public-contract-release-approved",
            pseudonym=pseudonym,
            category="evidence",
            prompt=(
                "Explain Epicure evidence to person@example.com and call +34 612 345 678 "
                "without claiming that similarity proves causation."
            ),
            consent=True,
        )
        unconsented_id = create_battle(
            client,
            "public-contract-release-no-consent",
            pseudonym="5" * 64,
            consent=False,
        )
        complete_jobs("public-contract-release-worker")
        approved = client.post(
            f"/v1/admin/battles/{approved_id}/release-review",
            headers=ADMIN_HEADERS,
            json={"status": "approved", "review_reference": "qa-release-review"},
        )
        assert approved.status_code == 200
        assert approved.json()["releaseReviewStatus"] == "approved"
        rejected = client.post(
            f"/v1/admin/battles/{unconsented_id}/release-review",
            headers=ADMIN_HEADERS,
            json={"status": "approved", "review_reference": "qa-release-review"},
        )
        assert rejected.status_code == 404
        invalid = client.post(
            f"/v1/admin/battles/{approved_id}/release-review",
            headers=ADMIN_HEADERS,
            json={"status": "pending", "review_reference": "qa-release-review"},
        )
        assert invalid.status_code == 422

        export = client.get("/v1/admin/research-export?season=season-0", headers=ADMIN_HEADERS)
        assert export.status_code == 200
        payload = export.json()
        record = next(item for item in payload["records"] if item["battleId"] == approved_id)
        assert "[EMAIL REDACTED]" in record["prompt"]
        assert "[PHONE REDACTED]" in record["prompt"]
        assert all(
            {"side", "modelId", "condition", "answer", "generationId"} <= arm.keys()
            for arm in record["arms"]
        )
        assert len(payload["exportSha256"]) == 64
        repeated = client.get("/v1/admin/research-export?season=season-0", headers=ADMIN_HEADERS)
        assert repeated.json()["exportSha256"] == payload["exportSha256"]
        missing = client.get("/v1/admin/research-export?season=missing", headers=ADMIN_HEADERS)
        assert missing.status_code == 404
