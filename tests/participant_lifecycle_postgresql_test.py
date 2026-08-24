from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

import flavourbench.participant_lifecycle as participant_lifecycle
from flavourbench.config import Settings
from flavourbench.database import (
    EXPECTED_SCHEMA_REVISION,
    _assert_postgresql_participant_lifecycle_guards,
    _assert_postgresql_reviewer_task_validation_guards,
)
from flavourbench.models import (
    ExpertReviewer,
    ReviewerConsentAcceptance,
    ReviewerDeletionReceipt,
    ReviewerParticipationLifecycle,
    ReviewerWithdrawalReceipt,
    Season,
)
from flavourbench.participant_lifecycle import (
    ActiveHumanStudyBinding,
    ParticipantLifecycleError,
    accept_participant_consent,
    create_retention_schedule,
    enroll_participant_identity,
    execute_private_payload_deletion,
    issue_enrollment_offer,
    withdraw_participant,
)

POSTGRES_URL = os.environ.get("FLAVOURBENCH_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="FLAVOURBENCH_TEST_POSTGRES_URL was not provided",
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
CONSENT_SHA256 = "a" * 64
ACTIVATION_SHA256 = "b" * 64
RETENTION_SHA256 = "c" * 64
POSTGRESQL_CAPACITY_HELPER = ROOT / "tests" / "postgresql_candidate_capacity_helper.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _psycopg_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _settings() -> Settings:
    return Settings(
        reviewer_identity_hmac_secret="pg-participant-identity-hmac-secret-00000001",
        reviewer_credential_hmac_secret="pg-participant-credential-hmac-secret-000001",
        human_study_activation_manifest_sha256=ACTIVATION_SHA256,
    )


@pytest.fixture(scope="module")
def postgres_engine() -> Engine:
    source = make_url(POSTGRES_URL)
    if source.database is None or "test" not in source.database.lower():
        pytest.fail("FLAVOURBENCH_TEST_POSTGRES_URL must name a disposable test database")
    owner_url = source.set(drivername="postgresql+psycopg", database="postgres")
    owner_dsn = _psycopg_url(owner_url.render_as_string(hide_password=False))
    database_name = f"flavourbench_participant_{uuid.uuid4().hex[:12]}"

    with psycopg.connect(owner_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_user, rolsuper OR rolcreatedb "
                "FROM pg_catalog.pg_roles WHERE rolname = current_user"
            )
            if cursor.fetchone() != ("flavourbench_owner", True):
                pytest.fail(
                    "the participant PostgreSQL suite requires a disposable "
                    "flavourbench_owner URL with CREATEDB"
                )
            cursor.execute(
                sql.SQL("CREATE DATABASE {} OWNER flavourbench_owner").format(
                    sql.Identifier(database_name)
                )
            )

    test_url = source.set(
        drivername="postgresql+psycopg",
        database=database_name,
    ).render_as_string(hide_password=False)
    environment = os.environ.copy()
    environment.update(
        {
            "FLAVOURBENCH_DATABASE_URL": test_url,
            "FLAVOURBENCH_ENVIRONMENT": "test",
            "FLAVOURBENCH_SERVICE_ROLE": "migration",
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        }
    )
    engine: Engine | None = None
    try:
        migration = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if migration.returncode:
            pytest.fail(
                f"temporary PostgreSQL migration failed:\n{migration.stdout}\n{migration.stderr}"
            )
        engine = create_engine(test_url)
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        with psycopg.connect(owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                        sql.Identifier(database_name)
                    )
                )


@pytest.fixture
def governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        participant_lifecycle,
        "require_active_human_study",
        lambda **_kwargs: ActiveHumanStudyBinding(
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            retention_policy_sha256=RETENTION_SHA256,
            consent_text="# PostgreSQL participant test consent",
        ),
    )
    monkeypatch.setattr(
        participant_lifecycle,
        "require_original_accepted_manifest",
        lambda _acceptance, **_kwargs: None,
    )


def _season(session: Session, suffix: str) -> Season:
    season = Season(
        id=str(uuid.uuid4()),
        slug=f"pg-participant-{suffix}-{uuid.uuid4().hex[:8]}",
        name=f"PostgreSQL participant {suffix}",
        epicure_release_id="epicure-test",
    )
    session.add(season)
    session.flush()
    return season


def _enroll(
    session: Session,
    *,
    season: Season,
    suffix: str,
    settings: Settings,
) -> tuple[str, str, str]:
    token, _ = issue_enrollment_offer(
        session,
        season=season,
        consent_document_sha256=CONSENT_SHA256,
        now=NOW,
        settings=settings,
    )
    accepted = accept_participant_consent(
        session,
        enrollment_token=token,
        consent_document_sha256=CONSENT_SHA256,
        activation_manifest_sha256=ACTIVATION_SHA256,
        confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
        idempotency_key=f"pg-accept-{suffix}",
        now=NOW + timedelta(seconds=1),
        settings=settings,
    )
    identity = enroll_participant_identity(
        session,
        receipt_credential=accepted.receipt_credential,
        identity_issuer="https://identity.example.test",
        issuer_subject=f"pg-raw-subject-{suffix}",
        identity_evidence_sha256=_sha(f"pg-identity-{suffix}"),
        roles=["output_rater"],
        qualified_families=["cookability"],
        affiliation_class="independent_external",
        now=NOW + timedelta(seconds=2),
        settings=settings,
    )
    return accepted.receipt_credential, identity.reviewer.id, identity.lifecycle.id


def test_postgresql_guards_expiry_cross_scope_and_append_only(
    postgres_engine: Engine,
    governance: None,
) -> None:
    del governance
    settings = _settings()
    with postgres_engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            EXPECTED_SCHEMA_REVISION
        )
        _assert_postgresql_participant_lifecycle_guards(connection)

    with Session(postgres_engine) as session:
        expired_season = _season(session, "expired")
        expired_token, _ = issue_enrollment_offer(
            session,
            season=expired_season,
            consent_document_sha256=CONSENT_SHA256,
            ttl_seconds=300,
            now=NOW,
            settings=settings,
        )
        session.commit()
    with Session(postgres_engine) as session:
        with pytest.raises(ParticipantLifecycleError, match="expired"):
            accept_participant_consent(
                session,
                enrollment_token=expired_token,
                consent_document_sha256=CONSENT_SHA256,
                activation_manifest_sha256=ACTIVATION_SHA256,
                confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
                idempotency_key="pg-expired-acceptance",
                now=NOW + timedelta(seconds=301),
                settings=settings,
            )
        session.rollback()

    with Session(postgres_engine) as session:
        season = _season(session, "scope")
        receipt_a, reviewer_a, lifecycle_a = _enroll(
            session,
            season=season,
            suffix="scope-a",
            settings=settings,
        )
        receipt_b, _, _ = _enroll(
            session,
            season=season,
            suffix="scope-b",
            settings=settings,
        )
        schedule = create_retention_schedule(
            session,
            reviewer_id=reviewer_a,
            analysis_freeze_at=NOW,
            first_public_release_at=NOW,
            now=NOW,
            settings=settings,
        )
        withdrawal = withdraw_participant(
            session,
            receipt_credential=receipt_a,
            idempotency_key="pg-scope-withdrawal",
            reason_code="privacy_request",
            now=NOW + timedelta(minutes=1),
            settings=settings,
        )
        session.commit()
        withdrawal_id = withdrawal.id
        schedule_id = schedule.id
        direct_due = schedule.direct_payload_delete_due_at
        audit_until = schedule.pseudonymous_audit_retain_until

    invalid_receipt_sha256 = _sha("pg-invalid-redaction-receipt")
    with pytest.raises(DBAPIError, match="redaction is incomplete"):
        with postgres_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO reviewer_deletion_receipts "
                    "(id, lifecycle_id, retention_schedule_id, season_id, reviewer_id, "
                    "request_sha256, execution_basis, redacted_fields_json, "
                    "private_payload_before_sha256, audit_marker_sha256, "
                    "direct_payload_delete_due_at, pseudonymous_audit_retain_until, "
                    "prior_judgments_preserved, receipt_sha256, executed_at, created_at) "
                    "SELECT :id, lifecycle.id, :schedule_id, lifecycle.season_id, "
                    "lifecycle.reviewer_id, :request_sha256, 'participant_request', "
                    "CAST(:redacted_fields_json AS json), :private_payload_before_sha256, "
                    "lifecycle.audit_marker_sha256, :direct_due, :audit_until, true, "
                    ":receipt_sha256, :executed_at, :executed_at "
                    "FROM reviewer_participation_lifecycles AS lifecycle "
                    "WHERE lifecycle.id = :lifecycle_id"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "schedule_id": schedule_id,
                    "request_sha256": _sha("pg-invalid-redaction-request"),
                    "redacted_fields_json": json.dumps(
                        list(participant_lifecycle.REDACTED_PROFILE_FIELDS)
                    ),
                    "private_payload_before_sha256": _sha("pg-private-payload-before"),
                    "direct_due": direct_due,
                    "audit_until": audit_until,
                    "receipt_sha256": invalid_receipt_sha256,
                    "executed_at": NOW + timedelta(minutes=2),
                    "lifecycle_id": lifecycle_a,
                },
            )
            connection.execute(
                text(
                    "UPDATE expert_reviewers SET profile_json = CAST(:profile_json AS json), "
                    "qualification_json = CAST('[]' AS json), "
                    "qualification_verified = false, privacy_status = 'redacted', "
                    "privacy_redacted_at = :redacted_at, "
                    "privacy_redaction_receipt_sha256 = :receipt_sha256 "
                    "WHERE id = :reviewer_id"
                ),
                {
                    "profile_json": json.dumps(
                        {
                            "schema_version": "flavourbench-reviewer-redacted-profile-v1",
                            "privacy_status": "redacted",
                            "audit_marker_sha256": _sha("pg-wrong-audit-marker"),
                            "private_payload_before_sha256": _sha("pg-wrong-private-payload"),
                        }
                    ),
                    "redacted_at": NOW + timedelta(minutes=2),
                    "receipt_sha256": invalid_receipt_sha256,
                    "reviewer_id": reviewer_a,
                },
            )

    with Session(postgres_engine) as session:
        with pytest.raises(ParticipantLifecycleError, match="crossed reviewer scope"):
            execute_private_payload_deletion(
                session,
                reviewer_id=reviewer_a,
                idempotency_key="pg-cross-scope-deletion",
                execution_basis="participant_request",
                receipt_credential=receipt_b,
                now=NOW + timedelta(minutes=2),
                settings=settings,
            )
        session.rollback()
        lifecycle = session.scalar(
            select(ReviewerParticipationLifecycle).where(
                ReviewerParticipationLifecycle.reviewer_id == reviewer_a
            )
        )
        assert lifecycle is not None and lifecycle.status == "withdrawn"

    with (
        postgres_engine.begin() as connection,
        pytest.raises(DBAPIError, match="append-only"),
    ):
        connection.execute(
            text(
                "UPDATE reviewer_withdrawal_receipts "
                "SET reason_code = 'safety_concern' WHERE id = :receipt_id"
            ),
            {"receipt_id": withdrawal_id},
        )


def test_postgresql_concurrent_accept_withdraw_and_delete_are_singleton(
    postgres_engine: Engine,
    governance: None,
) -> None:
    del governance
    settings = _settings()
    with Session(postgres_engine) as session:
        season = _season(session, "race")
        season_id = season.id
        enrollment_token, _ = issue_enrollment_offer(
            session,
            season=season,
            consent_document_sha256=CONSENT_SHA256,
            now=NOW,
            settings=settings,
        )
        session.commit()

    acceptance_barrier = Barrier(2)

    def accept_once() -> tuple[str, str]:
        with Session(postgres_engine) as session:
            acceptance_barrier.wait(timeout=10)
            result = accept_participant_consent(
                session,
                enrollment_token=enrollment_token,
                consent_document_sha256=CONSENT_SHA256,
                activation_manifest_sha256=ACTIVATION_SHA256,
                confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
                idempotency_key="pg-concurrent-acceptance",
                now=NOW + timedelta(seconds=1),
                settings=settings,
            )
            session.commit()
            return result.acceptance.receipt_sha256, result.receipt_credential

    with ThreadPoolExecutor(max_workers=2) as executor:
        acceptance_outcomes = list(executor.map(lambda _index: accept_once(), range(2)))
    assert len({outcome[0] for outcome in acceptance_outcomes}) == 1
    assert len({outcome[1] for outcome in acceptance_outcomes}) == 1
    receipt_credential = acceptance_outcomes[0][1]

    with Session(postgres_engine) as session:
        assert (
            len(
                session.scalars(
                    select(ReviewerConsentAcceptance).where(
                        ReviewerConsentAcceptance.season_id == season_id
                    )
                ).all()
            )
            == 1
        )
        season = session.get(Season, season_id)
        assert season is not None
        identity = enroll_participant_identity(
            session,
            receipt_credential=receipt_credential,
            identity_issuer="https://identity.example.test",
            issuer_subject="pg-concurrent-raw-subject",
            identity_evidence_sha256=_sha("pg-concurrent-identity"),
            roles=["output_rater"],
            qualified_families=["cookability"],
            affiliation_class="independent_external",
            now=NOW + timedelta(seconds=2),
            settings=settings,
        )
        create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=NOW,
            first_public_release_at=NOW,
            now=NOW,
            settings=settings,
        )
        reviewer_id = identity.reviewer.id
        session.commit()

    withdrawal_barrier = Barrier(2)

    def withdraw_once() -> str:
        with Session(postgres_engine) as session:
            withdrawal_barrier.wait(timeout=10)
            receipt = withdraw_participant(
                session,
                receipt_credential=receipt_credential,
                idempotency_key="pg-concurrent-withdrawal",
                reason_code="voluntary_withdrawal",
                now=NOW + timedelta(hours=1),
                settings=settings,
            )
            session.commit()
            return receipt.receipt_sha256

    with ThreadPoolExecutor(max_workers=2) as executor:
        withdrawal_outcomes = list(executor.map(lambda _index: withdraw_once(), range(2)))
    assert len(set(withdrawal_outcomes)) == 1

    deletion_barrier = Barrier(2)

    def delete_once() -> str:
        with Session(postgres_engine) as session:
            deletion_barrier.wait(timeout=10)
            receipt = execute_private_payload_deletion(
                session,
                reviewer_id=reviewer_id,
                idempotency_key="pg-concurrent-deletion",
                execution_basis="participant_request",
                receipt_credential=receipt_credential,
                now=NOW + timedelta(hours=2),
                settings=settings,
            )
            session.commit()
            return receipt.receipt_sha256

    with ThreadPoolExecutor(max_workers=2) as executor:
        deletion_outcomes = list(executor.map(lambda _index: delete_once(), range(2)))
    assert len(set(deletion_outcomes)) == 1

    with Session(postgres_engine) as session:
        assert (
            session.scalar(
                select(ReviewerWithdrawalReceipt).where(
                    ReviewerWithdrawalReceipt.reviewer_id == reviewer_id
                )
            )
            is not None
        )
        assert (
            session.scalar(
                select(ReviewerDeletionReceipt).where(
                    ReviewerDeletionReceipt.reviewer_id == reviewer_id
                )
            )
            is not None
        )
        lifecycle = session.scalar(
            select(ReviewerParticipationLifecycle).where(
                ReviewerParticipationLifecycle.reviewer_id == reviewer_id
            )
        )
        reviewer = session.get(ExpertReviewer, reviewer_id)
        assert lifecycle is not None and lifecycle.status == "redacted"
        assert reviewer is not None and reviewer.privacy_status == "redacted"


def test_postgresql_candidate_capacity_is_cross_process_serialized(
    postgres_engine: Engine,
    tmp_path: Path,
) -> None:
    with postgres_engine.connect() as connection:
        _assert_postgresql_reviewer_task_validation_guards(connection)
    database_url = _psycopg_url(postgres_engine.url.render_as_string(hide_password=False))
    table_name = f"task_validation_capacity_probe_{uuid.uuid4().hex}"
    trigger_name = f"trg_{table_name}"
    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE public.{} (campaign_sha256 text NOT NULL, "
                    "candidate_id text, event_type text NOT NULL)"
                ).format(sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE TRIGGER {} BEFORE INSERT ON public.{} FOR EACH ROW "
                    "EXECUTE FUNCTION public."
                    "flavourbench_task_validation_candidate_capacity_v1()"
                ).format(sql.Identifier(trigger_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO public.{} "
                    "(campaign_sha256, candidate_id, event_type) "
                    "VALUES ('capacity-campaign', 'capacity-candidate', 'blind_ballot')"
                ).format(sql.Identifier(table_name))
            )

    barrier_directory = tmp_path / "postgresql-capacity-barrier"
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(POSTGRESQL_CAPACITY_HELPER),
                "--database-url",
                database_url,
                "--table",
                table_name,
                "--ordinal",
                str(ordinal),
                "--barrier-directory",
                str(barrier_directory),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for ordinal in (2, 3)
    ]
    outputs: list[str] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            outputs.append(stdout)
        assert sorted(
            str(json.loads(output.strip().splitlines()[-1])["status"]) for output in outputs
        ) == ["inserted", "sealed"]
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table_name))
                )
                assert cursor.fetchone() == (2,)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        with psycopg.connect(database_url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS public.{}").format(sql.Identifier(table_name))
                )
