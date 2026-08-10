from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

import flavourbench.arena as arena
import flavourbench.main as main_module
from flavourbench.config import get_settings
from flavourbench.database import session_scope
from flavourbench.engine import reconcile_battle_cost
from flavourbench.main import app
from flavourbench.models import (
    Battle,
    ControlledRun,
    CostEvent,
    ExpertReviewer,
    Job,
    LeaderboardSnapshot,
    ResponseArm,
    Season,
    SeasonModel,
    Task,
)
from flavourbench.protocol_contract import build_protocol_bundle
from flavourbench.service_ranking import snapshot_hash

SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}


def _complete_controlled_battle(session, battle: Battle) -> None:
    arms = session.scalars(
        select(ResponseArm).where(ResponseArm.battle_id == battle.id).order_by(ResponseArm.side)
    ).all()
    assert len(arms) == 2
    job = session.scalar(select(Job).where(Job.battle_id == battle.id))
    assert job is not None and job.status == "queued"
    job.status = "running"
    session.flush()

    arm_completed_at = datetime.now(UTC) + timedelta(milliseconds=10)
    for arm in arms:
        generation_id = f"fixture-{battle.id}-{arm.side}"
        arm.status = "complete"
        arm.actual_provider_slug = arm.provider_slug
        arm.actual_model_id = arm.model_id
        arm.generation_id = generation_id
        arm.provider_generation_ids_json = [generation_id]
        arm.finish_reason = "stop"
        arm.answer_markdown = f"Anonymous answer {arm.side}"
        arm.output_json = {"answer_markdown": arm.answer_markdown}
        arm.cost_reconciled = True
        arm.cost_accounting_basis = "known_zero_no_provider_acceptance"
        arm.billing_reconciliation_status = "known_zero_no_provider_acceptance"
        arm.completed_at = arm_completed_at
        session.add(
            CostEvent(
                season_id=battle.season_id,
                battle_id=battle.id,
                arm_id=arm.id,
                kind="actual",
                amount_micros=0,
                provider=arm.provider_slug,
                generation_id=generation_id,
                accounting_json={"basis": "known_zero_no_provider_acceptance"},
            )
        )
    session.flush()
    battle.status = "complete"
    battle.completed_at = arm_completed_at + timedelta(milliseconds=1)
    job.status = "complete"
    job.completed_at = battle.completed_at
    session.flush()
    reconcile_battle_cost(session, battle)


def _create_run(client: TestClient, suffix: str) -> tuple[str, str, dict]:
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == "season-0"))
        assert season is not None
        task = session.scalar(
            select(Task).where(Task.season_id == season.id).order_by(Task.public_id)
        )
        slot = session.scalar(
            select(SeasonModel)
            .where(
                SeasonModel.season_id == season.id,
                SeasonModel.eligible.is_(True),
            )
            .order_by(SeasonModel.model_id)
        )
        assert task is not None and slot is not None
        task_public_id = task.public_id
        model_id = slot.model_id
    response = client.post(
        "/v1/admin/controlled-runs",
        headers=ADMIN_HEADERS,
        json={
            "season": "season-0",
            "organizationReference": f"organization:{suffix}",
            "protocolVersion": "flavourbench-controlled-run-v1",
            "raterPlanSha256": hashlib.sha256(f"rater:{suffix}".encode()).hexdigest(),
            "analysisPlanSha256": hashlib.sha256(f"analysis:{suffix}".encode()).hexdigest(),
            "submittedEndpointModelId": model_id,
            "submittedModelCardSha256": hashlib.sha256(f"model-card:{suffix}".encode()).hexdigest(),
            "dataPolicySha256": hashlib.sha256(f"data-policy:{suffix}".encode()).hexdigest(),
            "modelIds": [model_id],
            "taskSchedule": [
                {
                    "taskPublicId": task_public_id,
                    "track": "epicure_uplift",
                    "modelIds": [model_id],
                    "repetitionIndex": 1,
                }
            ],
            "budgetCapMicros": 5_000_000,
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["runId"], payload["accessToken"], payload


def test_controlled_run_token_scope_run_card_and_closed_admission() -> None:
    with TestClient(app) as client:
        run_id, token, created = _create_run(client, "contract")
        assert created["runCard"]["publication_default"] == "private"
        assert created["runCard"]["schema_version"] == "flavourbench-controlled-run-card-v7"
        assert created["runCard"]["signing"] == {
            "algorithm": "HMAC-SHA256",
            "key_id": "primary",
            "verification_scope": "FlavourBench service-held key",
        }
        assert created["runCard"]["cost_accounting_policy"] == {
            "controlled_run_used_basis": "endpoint_generation_receipts",
            "aggregate_invoice_variance_scope": "season_and_provider_account_only",
            "credits_restore_spend_authority": False,
        }
        canonical = json.dumps(created["runCard"], sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(canonical).hexdigest() == created["runCardSha256"]
        expected_signature = hmac.new(
            get_settings().run_card_signing_secret.encode(),
            created["runCardSha256"].encode(),
            hashlib.sha256,
        ).hexdigest()
        assert hmac.compare_digest(expected_signature, created["runCardSignature"])

        wrong = client.get(
            f"/v1/controlled/runs/{run_id}",
            headers={**SERVICE_HEADERS, "Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401
        overview = client.get(
            f"/v1/controlled/runs/{run_id}",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token}"},
        )
        assert overview.status_code == 200
        assert overview.json()["dataStratum"] == "controlled"
        assert overview.json()["budget"]["capMicros"] == 5_000_000
        assert overview.json()["budget"]["accountingBasis"] == ("endpoint_generation_receipts")
        assert overview.json()["budget"]["aggregateInvoiceVarianceScope"] == (
            "season_and_provider_account_only"
        )
        assert overview.json()["budget"]["creditsRestoreSpendAuthority"] is False

        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            task = session.scalar(
                select(Task).where(Task.season_id == run.season_id).order_by(Task.public_id)
            )
            assert run is not None and task is not None
            assert run.access_token_sha256 == hashlib.sha256(token.encode()).hexdigest()
            assert token not in json.dumps(run.run_card_json)
            task_id = task.public_id

        queued = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token}"},
            json={
                "taskPublicId": task_id,
                "expectedAssignmentOrdinal": 0,
                "clientNonce": "controlled-contract-001",
            },
        )
        assert queued.status_code == 202, queued.text
        with session_scope() as session:
            battle = session.get(Battle, queued.json()["battleId"])
            assert battle is not None
            assert battle.controlled_run_id == run_id
            assert battle.data_stratum == "controlled"
            assert battle.task_id is not None
            _complete_controlled_battle(session, battle)

        for action in ("collection_complete", "close"):
            transition = client.post(
                f"/v1/admin/controlled-runs/{run_id}/lifecycle",
                headers=ADMIN_HEADERS,
                json={
                    "action": action,
                    "authorizationReference": f"controlled-contract-{action}",
                },
            )
            assert transition.status_code == 200, transition.text

        closed = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token}"},
            json={
                "taskPublicId": task_id,
                "expectedAssignmentOrdinal": 0,
                "clientNonce": "controlled-contract-002",
            },
        )
        assert closed.status_code == 409
        assert closed.json()["detail"] == "controlled run is not accepting battles"


def test_private_snapshots_are_token_scoped_and_never_publicly_published() -> None:
    with TestClient(app) as client:
        run_a, token_a, _ = _create_run(client, "private-a")
        run_b, token_b, _ = _create_run(client, "private-b")
        with session_scope() as session:
            first = session.get(ControlledRun, run_a)
            second = session.get(ControlledRun, run_b)
            assert first is not None and second is not None
            payload_a = {
                "rows": [{"competitor_id": "private-a"}],
                "data_stratum": "controlled",
                "controlled_run_id": first.id,
            }
            payload_b = {
                "rows": [{"competitor_id": "private-b"}],
                "data_stratum": "controlled",
                "controlled_run_id": second.id,
            }
            evidence_a = {"scope": "private-a"}
            evidence_b = {"scope": "private-b"}
            snapshot_a = LeaderboardSnapshot(
                season_id=first.season_id,
                track="model_arena",
                cohort="expert_independent",
                category="all",
                data_stratum="controlled",
                controlled_run_id=first.id,
                publication_status="draft",
                input_sha256=snapshot_hash(payload_a),
                input_evidence_sha256=snapshot_hash(evidence_a),
                input_evidence_json=evidence_a,
                payload_sha256=snapshot_hash(payload_a),
                payload_json=payload_a,
            )
            snapshot_b = LeaderboardSnapshot(
                season_id=second.season_id,
                track="model_arena",
                cohort="expert_independent",
                category="all",
                data_stratum="controlled",
                controlled_run_id=second.id,
                publication_status="draft",
                input_sha256=snapshot_hash(payload_b),
                input_evidence_sha256=snapshot_hash(evidence_b),
                input_evidence_json=evidence_b,
                payload_sha256=snapshot_hash(payload_b),
                payload_json=payload_b,
            )
            session.add_all([snapshot_a, snapshot_b])
            session.flush()
            snapshot_a_id = snapshot_a.id

        own = client.get(
            f"/v1/controlled/runs/{run_a}/leaderboards",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token_a}"},
        )
        assert own.status_code == 409
        assert own.json()["detail"] == "controlled-run season is not active"
        cross_tenant = client.get(
            f"/v1/controlled/runs/{run_a}/leaderboards",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token_b}"},
        )
        assert cross_tenant.status_code == 401

        blocked_publish = client.post(
            f"/v1/admin/leaderboards/snapshots/{snapshot_a_id}/publish",
            headers=ADMIN_HEADERS,
            json={"publicationReference": "must-remain-private"},
        )
        assert blocked_publish.status_code == 409

        authorization = client.post(
            f"/v1/controlled/runs/{run_a}/release-authorization",
            headers=ADMIN_HEADERS,
            json={
                "authorized": True,
                "authorizationReference": "customer-release-approval-a",
            },
        )
        assert authorization.status_code == 200
        assert authorization.json()["releaseAuthorized"] is True
        still_blocked = client.post(
            f"/v1/admin/leaderboards/snapshots/{snapshot_a_id}/publish",
            headers=ADMIN_HEADERS,
            json={"publicationReference": "controlled-release-needs-separate-route"},
        )
        assert still_blocked.status_code == 409

        public = client.get(
            "/v1/leaderboards?season=season-0&track=model_arena&"
            "rater_cohort=expert_independent&task_family=all",
            headers=SERVICE_HEADERS,
        )
        assert public.status_code == 200
        assert public.json()["data_stratum"] == "public_freeform"
        assert public.json()["rows"] == []


def test_controlled_publication_is_current_and_revocation_withdraws_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope: dict[str, str | None] = {"run_id": None}

    def ranking_payload(*_args: object, **_kwargs: object) -> dict:
        return {
            "track": "model_arena",
            "cohort": "expert_independent",
            "cohort_label": "Expert_Independent",
            "category": "all",
            "data_stratum": "controlled",
            "controlled_run_id": scope["run_id"],
            "rows": [{"competitor_id": "customer/model", "battles": 40}],
            "method": "frozen controlled test method",
            "preference_observation_ids": [],
            "preference_observation_sha256": main_module._canonical_sha256(
                {"vote_ids": []}
            ),
            "manifest_sha256": "test",
            "accounting": {
                "response_arms": 80,
                "cost_reconciled_arms": 80,
                "cost_unreconciled_arms": 0,
                "billing_reconciliation_complete": True,
                "complete": True,
            },
        }

    def evidence_manifest(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema_version": "controlled-snapshot-test-v1",
            "controlled_run_id": scope["run_id"],
            "analysis_observations": {
                "eligible_judgment_ids": [],
                "preference_observation_ids": [],
                "preference_observation_sha256": main_module._canonical_sha256(
                    {"vote_ids": []}
                ),
            },
        }

    monkeypatch.setattr(main_module, "model_leaderboard", ranking_payload)
    monkeypatch.setattr(main_module, "_snapshot_evidence_manifest", evidence_manifest)

    with TestClient(app) as client:
        run_id, token, _ = _create_run(client, "published-controlled")
        scope["run_id"] = run_id
        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            assert run is not None
            season = session.get(Season, run.season_id)
            task = session.scalar(
                select(Task).where(Task.season_id == run.season_id).order_by(Task.public_id)
            )
            assert season is not None and task is not None
            prior_season_status = season.status
            task_public_id = task.public_id

        queued = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token}"},
            json={
                "taskPublicId": task_public_id,
                "expectedAssignmentOrdinal": 0,
                "clientNonce": "published-controlled-battle",
            },
        )
        assert queued.status_code == 202, queued.text
        with session_scope() as session:
            battle = session.get(Battle, queued.json()["battleId"])
            assert battle is not None
            _complete_controlled_battle(session, battle)
        for action in ("collection_complete", "close"):
            transition = client.post(
                f"/v1/admin/controlled-runs/{run_id}/lifecycle",
                headers=ADMIN_HEADERS,
                json={
                    "action": action,
                    "authorizationReference": f"published-controlled-{action}",
                },
            )
            assert transition.status_code == 200, transition.text
        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            assert run is not None
            season = session.get(Season, run.season_id)
            assert season is not None
            season.status = "active"

        try:
            draft = client.post(
                "/v1/admin/leaderboards/snapshot?season=season-0&track=model_arena&"
                "cohort=expert_independent&category=all&data_stratum=controlled&"
                f"controlled_run_id={run_id}",
                headers=ADMIN_HEADERS,
            )
            assert draft.status_code == 200, draft.text
            snapshot_id = draft.json()["snapshotId"]

            authorized = client.post(
                f"/v1/controlled/runs/{run_id}/release-authorization",
                headers=ADMIN_HEADERS,
                json={
                    "authorized": True,
                    "authorizationReference": "controlled-test-release",
                },
            )
            assert authorized.status_code == 200, authorized.text

            published = client.post(
                f"/v1/admin/controlled-runs/{run_id}/snapshots/{snapshot_id}/publish",
                headers=ADMIN_HEADERS,
                json={"publicationReference": "controlled-test-publication"},
            )
            assert published.status_code == 200, published.text

            visible = client.get(
                f"/v1/controlled/runs/{run_id}/leaderboards",
                headers={**SERVICE_HEADERS, "Authorization": f"Bearer {token}"},
            )
            assert visible.status_code == 200, visible.text
            assert visible.json()["snapshotId"] == snapshot_id

            evidence = client.get(
                f"/v1/controlled/runs/{run_id}/evidence?snapshot_id={snapshot_id}",
                headers={
                    **SERVICE_HEADERS,
                    "Authorization": f"Bearer {token}",
                },
            )
            assert evidence.status_code == 200, evidence.text
            evidence_body = evidence.json()
            envelope = evidence_body["envelope"]
            envelope_sha256 = main_module._canonical_sha256(envelope)
            assert envelope["snapshot"]["id"] == snapshot_id
            assert envelope["run_card"]["run_id"] == run_id
            assert (
                envelope["release_authorization"]["reference_sha256"]
                == hashlib.sha256(b"controlled-test-release").hexdigest()
            )
            assert (
                envelope["snapshot"]["publication_reference_sha256"]
                == hashlib.sha256(b"controlled-test-publication").hexdigest()
            )
            assert envelope["evidence_manifest"] == evidence_manifest()
            assert evidence_body["envelopeSha256"] == envelope_sha256
            expected_signature = hmac.new(
                get_settings().run_card_signing_secret.encode(),
                f"flavourbench-controlled-evidence-v1:{envelope_sha256}".encode(),
                hashlib.sha256,
            ).hexdigest()
            assert hmac.compare_digest(evidence_body["signature"], expected_signature)
            assert evidence_body["signingKeyId"] == "primary"
            assert evidence_body["verificationScope"] == "FlavourBench service-held key"

            revoked = client.post(
                f"/v1/controlled/runs/{run_id}/release-authorization",
                headers=ADMIN_HEADERS,
                json={
                    "authorized": False,
                    "authorizationReference": "controlled-test-revocation",
                },
            )
            assert revoked.status_code == 200, revoked.text
            assert revoked.json()["withdrawnSnapshotIds"] == [snapshot_id]

            blocked = client.get(
                f"/v1/controlled/runs/{run_id}/leaderboards",
                headers={
                    **SERVICE_HEADERS,
                    "Authorization": f"Bearer {token}",
                },
            )
            assert blocked.status_code == 409
            evidence_blocked = client.get(
                f"/v1/controlled/runs/{run_id}/evidence?snapshot_id={snapshot_id}",
                headers={
                    **SERVICE_HEADERS,
                    "Authorization": f"Bearer {token}",
                },
            )
            assert evidence_blocked.status_code == 409
            with session_scope() as session:
                snapshot = session.get(LeaderboardSnapshot, snapshot_id)
                assert snapshot is not None
                assert snapshot.publication_status == "withdrawn"
        finally:
            with session_scope() as session:
                run = session.get(ControlledRun, run_id)
                assert run is not None
                season = session.get(Season, run.season_id)
                assert season is not None
                season.status = prior_season_status


def test_controlled_run_budget_reservation_is_transactional_and_independent() -> None:
    season = Season(
        id="season-budget-test",
        slug="season-budget-test",
        name="Budget test",
        status="active",
        official=True,
        manifest_sha256="m" * 64,
        tool_registry_sha256="t" * 64,
        epicure_release_id="release",
        epicure_bundle_sha256="e" * 64,
        epicure_application_sha256="a" * 64,
        budget_cap_micros=10_000,
        budget_used_micros=0,
        budget_reserved_micros=0,
    )
    controlled_run = ControlledRun(
        season_id=season.id,
        organization_reference_sha256="1" * 64,
        access_token_sha256="2" * 64,
        status="active",
        protocol_version="budget-test-v1",
        rater_plan_sha256="3" * 64,
        analysis_plan_sha256="4" * 64,
        budget_cap_micros=1_000,
        budget_used_micros=700,
        budget_reserved_micros=0,
        run_card_json={"test": True},
        run_card_sha256="5" * 64,
        run_card_signature="6" * 64,
    )
    model = SeasonModel(
        season_id=season.id,
        model_id="budget/model",
        slot_role="test",
        worst_case_cost_micros=100,
    )
    original = arena.get_settings
    arena.get_settings = lambda: SimpleNamespace(execution_mode="live")  # type: ignore[assignment]
    try:
        reserved = arena._reserve_budget(None, season, [model], controlled_run)  # type: ignore[arg-type]
        assert reserved == 100
        assert season.budget_reserved_micros == 100
        assert controlled_run.budget_reserved_micros == 100
        controlled_run.budget_used_micros = 750
        controlled_run.budget_reserved_micros = 0
        with pytest.raises(HTTPException, match="controlled-run budget admission is closed"):
            arena._reserve_budget(None, season, [model], controlled_run)  # type: ignore[arg-type]
        assert controlled_run.budget_reserved_micros == 0
    finally:
        arena.get_settings = original  # type: ignore[assignment]


def test_public_leaderboard_reads_only_current_evidence_backed_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_slug = "publication-gate-season"
    material_revision = {"value": 1}

    def ranking_payload(*_args: object, **_kwargs: object) -> dict:
        return {
            "track": "model_arena",
            "cohort": "public",
            "cohort_label": "Public",
            "category": "all",
            "data_stratum": "public_freeform",
            "controlled_run_id": None,
            "rows": [{"competitor_id": "released/model", "battles": 100}],
            "method": "frozen test method",
            "preference_observation_ids": [],
            "preference_observation_sha256": main_module._canonical_sha256(
                {"vote_ids": []}
            ),
            "manifest_sha256": "9" * 64,
            "accounting": {
                "response_arms": 200,
                "cost_reconciled_arms": 200,
                "cost_unreconciled_arms": 0,
                "billing_reconciliation_complete": True,
                "complete": True,
            },
        }

    def evidence_manifest(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema_version": "snapshot-test-v1",
            "revision": material_revision["value"],
            "analysis_observations": {
                "eligible_judgment_ids": [],
                "preference_observation_ids": [],
                "preference_observation_sha256": main_module._canonical_sha256(
                    {"vote_ids": []}
                ),
            },
        }

    monkeypatch.setattr(main_module, "model_leaderboard", ranking_payload)
    monkeypatch.setattr(main_module, "_snapshot_evidence_manifest", evidence_manifest)
    with TestClient(app) as client:
        with session_scope() as session:
            season = Season(
                slug=season_slug,
                name="Publication gate",
                status="active",
                official=True,
                manifest_sha256="9" * 64,
                epicure_release_id="publication-gate-release",
                epicure_bundle_sha256="e" * 64,
                epicure_application_sha256="a" * 64,
                tool_registry_sha256="d" * 64,
                analysis_plan_sha256="c" * 64,
            )
            session.add(season)
            session.flush()
            protocol, protocol_sha256 = build_protocol_bundle(
                tool_registry_sha256=season.tool_registry_sha256,
                epicure_release_id=season.epicure_release_id,
                epicure_bundle_sha256=season.epicure_bundle_sha256,
                epicure_application_sha256=season.epicure_application_sha256,
                analysis_plan_sha256=season.analysis_plan_sha256,
            )
            season.protocol_bundle_json = protocol
            season.protocol_bundle_sha256 = protocol_sha256

        draft = client.post(
            f"/v1/admin/leaderboards/snapshot?season={season_slug}&track="
            "model_arena&cohort=public&category=all",
            headers=ADMIN_HEADERS,
        )
        assert draft.status_code == 200, draft.text
        snapshot_id = draft.json()["snapshotId"]

        before = client.get(
            f"/v1/leaderboards?season={season_slug}&track=model_arena&"
            "rater_cohort=public&task_family=all",
            headers=SERVICE_HEADERS,
        )
        assert before.status_code == 200
        assert before.json()["rows"] == []
        assert before.json()["snapshotId"] is None

        published = client.post(
            f"/v1/admin/leaderboards/snapshots/{snapshot_id}/publish",
            headers=ADMIN_HEADERS,
            json={"publicationReference": "publication-gate-approved"},
        )
        assert published.status_code == 200, published.text
        assert published.json()["publicationStatus"] == "published"

        after = client.get(
            f"/v1/leaderboards?season={season_slug}&track=model_arena&"
            "rater_cohort=public&task_family=all",
            headers=SERVICE_HEADERS,
        )
        assert after.status_code == 200
        assert after.json()["rows"] == [{"competitor_id": "released/model", "battles": 100}]
        assert after.json()["snapshotId"] == snapshot_id
        assert after.json()["official"] is True

        replacement = client.post(
            f"/v1/admin/leaderboards/snapshot?season={season_slug}&track="
            "model_arena&cohort=public&category=all",
            headers=ADMIN_HEADERS,
        )
        assert replacement.status_code == 200, replacement.text
        replacement_id = replacement.json()["snapshotId"]
        parallel_stale = client.post(
            f"/v1/admin/leaderboards/snapshot?season={season_slug}&track="
            "model_arena&cohort=public&category=all",
            headers=ADMIN_HEADERS,
        )
        assert parallel_stale.status_code == 200, parallel_stale.text
        parallel_stale_id = parallel_stale.json()["snapshotId"]
        replacement_publish = client.post(
            f"/v1/admin/leaderboards/snapshots/{replacement_id}/publish",
            headers=ADMIN_HEADERS,
            json={"publicationReference": "publication-gate-replacement"},
        )
        assert replacement_publish.status_code == 200, replacement_publish.text
        stale_publish = client.post(
            f"/v1/admin/leaderboards/snapshots/{parallel_stale_id}/publish",
            headers=ADMIN_HEADERS,
            json={"publicationReference": "stale-parallel-draft"},
        )
        assert stale_publish.status_code == 409
        assert "head changed" in stale_publish.json()["detail"]
        with session_scope() as session:
            predecessor = session.get(LeaderboardSnapshot, snapshot_id)
            assert predecessor is not None
            assert predecessor.publication_status == "withdrawn"
            stale_snapshot = session.get(LeaderboardSnapshot, parallel_stale_id)
            assert stale_snapshot is not None
            assert stale_snapshot.publication_status == "draft"

        material_revision["value"] = 2
        stale = client.get(
            f"/v1/leaderboards?season={season_slug}&track=model_arena&"
            "rater_cohort=public&task_family=all",
            headers=SERVICE_HEADERS,
        )
        assert stale.status_code == 503
        with session_scope() as session:
            withdrawn = session.get(LeaderboardSnapshot, replacement_id)
            assert withdrawn is not None
            assert withdrawn.publication_status == "withdrawn"

        no_fallback = client.get(
            f"/v1/leaderboards?season={season_slug}&track=model_arena&"
            "rater_cohort=public&task_family=all",
            headers=SERVICE_HEADERS,
        )
        assert no_fallback.status_code == 200
        assert no_fallback.json()["rows"] == []
        assert no_fallback.json()["snapshotId"] is None


def test_controlled_reviewer_authorization_cannot_reactivate_legacy_routes() -> None:
    with TestClient(app) as client:
        run_id, run_token, _ = _create_run(client, "expert-scope")
        invite = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "controlled-expert-scope",
                "qualified_families": [
                    "substitution",
                    "composition",
                    "cookability",
                    "evidence",
                ],
                "qualification_reference": "verified-eight-year-practice",
                "qualification_verified": True,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": "no-conflict-recorded",
                "consent_document_sha256": "1" * 64,
                "training_material_sha256": "2" * 64,
                "calibration_set_sha256": "3" * 64,
                "calibration_accuracy": 0.9,
                "compensation_reference": "documented-volunteer-review",
            },
        )
        assert invite.status_code == 200, invite.text
        invitation = invite.json()["invitation"]
        reviewer_id = invite.json()["reviewerId"]

        with session_scope() as session:
            run = session.get(ControlledRun, run_id)
            task = session.scalar(
                select(Task).where(Task.season_id == run.season_id).order_by(Task.public_id)
            )
            reviewer = session.get(ExpertReviewer, reviewer_id)
            assert run is not None and task is not None and reviewer is not None
            task_public_id = task.public_id

        queued = client.post(
            f"/v1/controlled/runs/{run_id}/battles",
            headers={**SERVICE_HEADERS, "Authorization": f"Bearer {run_token}"},
            json={
                "taskPublicId": task_public_id,
                "expectedAssignmentOrdinal": 0,
                "clientNonce": "expert-scope-battle",
            },
        )
        assert queued.status_code == 202, queued.text
        battle_id = queued.json()["battleId"]
        with session_scope() as session:
            battle = session.get(Battle, battle_id)
            assert battle is not None
            _complete_controlled_battle(session, battle)

        expert_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {invitation}",
        }
        generic = client.get("/v1/expert/assignments/next", headers=expert_headers)
        assert generic.status_code == 410

        unauthorized_assignment = client.get(
            f"/v1/expert/assignments/next?controlled_run_id={run_id}",
            headers=expert_headers,
        )
        assert unauthorized_assignment.status_code == 410
        unauthorized_vote = client.post(
            f"/v1/expert/battles/{battle_id}/votes",
            headers={**expert_headers, "Idempotency-Key": "expert-scope-denied"},
            json={"choice": "left", "reasonTags": [], "rubric": {}},
        )
        assert unauthorized_vote.status_code == 410

        authorized = client.put(
            f"/v1/admin/controlled-runs/{run_id}/reviewers/{reviewer_id}",
            headers=ADMIN_HEADERS,
            json={"authorizationReference": "rater-plan-assignment-001", "active": True},
        )
        assert authorized.status_code == 404, authorized.text
        assert authorized.json()["detail"] == "controlled run or reviewer not found"
        assignment = client.get(
            f"/v1/expert/assignments/next?controlled_run_id={run_id}",
            headers=expert_headers,
        )
        assert assignment.status_code == 410

        accepted_vote = client.post(
            f"/v1/expert/battles/{battle_id}/votes",
            headers={**expert_headers, "Idempotency-Key": "expert-scope-accepted"},
            json={"choice": "left", "reasonTags": [], "rubric": {}},
        )
        assert accepted_vote.status_code == 410
