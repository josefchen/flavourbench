from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

import flavourbench.participant_lifecycle as participant_lifecycle
from flavourbench.config import Settings
from flavourbench.database import (
    _PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256,
    _PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS,
    _REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256,
    EXPECTED_SCHEMA_REVISION,
)
from flavourbench.models import (
    ReviewerParticipationLifecycle,
    Season,
)
from flavourbench.participant_lifecycle import (
    ActiveHumanStudyBinding,
    accept_participant_consent,
    create_retention_schedule,
    enroll_participant_identity,
    execute_private_payload_deletion,
    issue_enrollment_offer,
    withdraw_participant,
)

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "alembic" / "versions" / "0035_participant_lifecycle_privacy.py"
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
CONSENT_SHA256 = "a" * 64
ACTIVATION_SHA256 = "b" * 64
RETENTION_SHA256 = "c" * 64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _upgrade_sqlite(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'participant-migration.sqlite3'}"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("alembic")), "upgrade", "head"],
        cwd=ROOT,
        env={
            **os.environ,
            "FLAVOURBENCH_DATABASE_URL": database_url,
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return create_engine(database_url)


def _settings() -> Settings:
    return Settings(
        reviewer_identity_hmac_secret="migration-identity-hmac-secret-00000000001",
        reviewer_credential_hmac_secret="migration-credential-hmac-secret-000000001",
        human_study_activation_manifest_sha256=ACTIVATION_SHA256,
    )


def _governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        participant_lifecycle,
        "require_active_human_study",
        lambda **_kwargs: ActiveHumanStudyBinding(
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            retention_policy_sha256=RETENTION_SHA256,
            consent_text="# Test consent",
        ),
    )
    monkeypatch.setattr(
        participant_lifecycle,
        "require_original_accepted_manifest",
        lambda _acceptance, **_kwargs: None,
    )


def test_0035_is_linear_irreversible_and_installs_both_dialect_guards(
    tmp_path: Path,
) -> None:
    engine = _upgrade_sqlite(tmp_path)
    source = MIGRATION.read_text(encoding="utf-8")
    assert EXPECTED_SCHEMA_REVISION == "0035_participant_lifecycle_privacy"
    assert 'revision = "0035_participant_lifecycle_privacy"' in source
    assert 'down_revision = "0034_task_validation_replay_binding"' in source
    assert "downgrade across participant consent, withdrawal, and privacy evidence" in source
    assert source.count("SET search_path = pg_catalog, public") == 11

    inspector = inspect(engine)
    assert {
        "reviewer_enrollment_offers",
        "reviewer_consent_acceptances",
        "reviewer_participation_lifecycles",
        "reviewer_withdrawal_receipts",
        "reviewer_retention_schedules",
        "reviewer_deletion_receipts",
    } <= set(inspector.get_table_names())
    expert_columns = {column["name"] for column in inspector.get_columns("expert_reviewers")}
    assert {
        "privacy_status",
        "privacy_redacted_at",
        "privacy_redaction_receipt_sha256",
    } <= expert_columns
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            EXPECTED_SCHEMA_REVISION
        )
        triggers = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).scalars()
        )
    assert {
        "trg_reviewer_consent_acceptances_guard_v1",
        "trg_reviewer_participation_lifecycles_insert_guard_v1",
        "trg_reviewer_withdrawal_receipts_guard_v1",
        "trg_reviewer_retention_schedules_guard_v1",
        "trg_reviewer_deletion_receipts_guard_v1",
        "trg_reviewer_enrollment_offers_lifecycle_v1",
        "trg_reviewer_participation_lifecycles_lifecycle_v1",
        "trg_reviewer_identity_bindings_participant_consent_v1",
        "trg_reviewer_qualification_evidence_participant_consent_v1",
        "trg_reviewer_access_credentials_participation_v1_insert",
        "trg_reviewer_access_credentials_participation_v1_update",
        "trg_controlled_run_reviewers_participation_v1_insert",
        "trg_controlled_run_reviewers_participation_v1_update",
        "trg_votes_participation_v1_insert",
        "trg_votes_participation_v1_update",
        "trg_task_validation_campaign_events_participation_v1",
        "trg_task_validation_audit_authorizations_participation_v1",
        "trg_task_validation_campaign_events_candidate_capacity_v1",
        "trg_expert_reviewers_privacy_lifecycle_v1",
    } <= triggers


def test_migrated_sqlite_enforces_receipt_ordering_deadlines_and_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _upgrade_sqlite(tmp_path)
    _governance(monkeypatch)
    settings = _settings()
    with Session(engine) as session:
        season = Season(
            id="migration-participant-season",
            slug="migration-participant-season",
            name="Migration participant season",
            epicure_release_id="epicure-test",
        )
        session.add(season)
        session.flush()
        enrollment_token, _ = issue_enrollment_offer(
            session,
            season=season,
            consent_document_sha256=CONSENT_SHA256,
            now=NOW,
            settings=settings,
        )
        accepted = accept_participant_consent(
            session,
            enrollment_token=enrollment_token,
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
            idempotency_key="migration-acceptance",
            now=NOW + timedelta(seconds=1),
            settings=settings,
        )
        identity = enroll_participant_identity(
            session,
            receipt_credential=accepted.receipt_credential,
            identity_issuer="https://identity.example.test",
            issuer_subject="migration-raw-subject",
            identity_evidence_sha256=_sha("migration-identity"),
            roles=["output_rater"],
            qualified_families=["cookability"],
            affiliation_class="independent_external",
            now=NOW + timedelta(seconds=2),
            settings=settings,
        )
        schedule = create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=datetime(2024, 2, 29, 12, 0, tzinfo=UTC),
            first_public_release_at=datetime(2024, 2, 29, 12, 0, tzinfo=UTC),
            now=NOW,
            settings=settings,
        )
        withdrawal = withdraw_participant(
            session,
            receipt_credential=accepted.receipt_credential,
            idempotency_key="migration-withdrawal",
            reason_code="privacy_request",
            now=NOW + timedelta(seconds=3),
            settings=settings,
        )
        session.commit()
        reviewer_id = identity.reviewer.id
        lifecycle_id = identity.lifecycle.id
        schedule_id = schedule.id

        invalid_receipt_sha256 = _sha("invalid-redaction-receipt")
        with pytest.raises(DatabaseError, match="one-way"):
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO reviewer_deletion_receipts "
                        "(id, lifecycle_id, retention_schedule_id, season_id, reviewer_id, "
                        "request_sha256, execution_basis, redacted_fields_json, "
                        "private_payload_before_sha256, audit_marker_sha256, "
                        "direct_payload_delete_due_at, pseudonymous_audit_retain_until, "
                        "prior_judgments_preserved, receipt_sha256, executed_at, created_at) "
                        "SELECT 'invalid-redaction-receipt', lifecycle.id, schedule.id, "
                        "lifecycle.season_id, lifecycle.reviewer_id, :request_sha256, "
                        "'participant_request', :redacted_fields_json, "
                        ":private_payload_before_sha256, lifecycle.audit_marker_sha256, "
                        "schedule.direct_payload_delete_due_at, "
                        "schedule.pseudonymous_audit_retain_until, 1, :receipt_sha256, "
                        ":executed_at, :executed_at "
                        "FROM reviewer_participation_lifecycles AS lifecycle "
                        "JOIN reviewer_retention_schedules AS schedule "
                        "ON schedule.lifecycle_id = lifecycle.id "
                        "WHERE lifecycle.id = :lifecycle_id AND schedule.id = :schedule_id"
                    ),
                    {
                        "lifecycle_id": lifecycle_id,
                        "schedule_id": schedule_id,
                        "request_sha256": _sha("invalid-redaction-request"),
                        "redacted_fields_json": json.dumps(
                            list(participant_lifecycle.REDACTED_PROFILE_FIELDS)
                        ),
                        "private_payload_before_sha256": _sha("private-payload-before"),
                        "receipt_sha256": invalid_receipt_sha256,
                        "executed_at": NOW + timedelta(seconds=4),
                    },
                )
                connection.execute(
                    text(
                        "UPDATE expert_reviewers SET profile_json = :profile_json, "
                        "qualification_json = '[]', qualification_verified = 0, "
                        "privacy_status = 'redacted', privacy_redacted_at = :redacted_at, "
                        "privacy_redaction_receipt_sha256 = :receipt_sha256 "
                        "WHERE id = :reviewer_id"
                    ),
                    {
                        "profile_json": json.dumps(
                            {
                                "schema_version": "flavourbench-reviewer-redacted-profile-v1",
                                "privacy_status": "redacted",
                                "audit_marker_sha256": _sha("wrong-audit-marker"),
                                "private_payload_before_sha256": _sha("wrong-private-payload"),
                            }
                        ),
                        "redacted_at": NOW + timedelta(seconds=4),
                        "receipt_sha256": invalid_receipt_sha256,
                        "reviewer_id": reviewer_id,
                    },
                )

        session.expire_all()
        deletion = execute_private_payload_deletion(
            session,
            reviewer_id=reviewer_id,
            idempotency_key="migration-scheduled-deletion",
            execution_basis="scheduled_retention",
            now=NOW + timedelta(seconds=4),
            settings=settings,
        )
        session.commit()

        assert schedule.direct_payload_delete_due_at.date() == datetime(2025, 2, 28).date()
        assert schedule.pseudonymous_audit_retain_until.date() == datetime(2029, 2, 28).date()
        assert withdrawal.prior_judgments_preserved
        assert deletion.prior_judgments_preserved
        assert session.scalar(select(ReviewerParticipationLifecycle)).status == "redacted"
        acceptance_id = accepted.acceptance.id
        deletion_id = deletion.id

    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        connection.execute(
            text("UPDATE reviewer_consent_acceptances SET receipt_sha256 = :digest WHERE id = :id"),
            {"digest": _sha("illegal-acceptance-mutation"), "id": acceptance_id},
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="one-way"),
    ):
        connection.execute(
            text("UPDATE expert_reviewers SET active = 1 WHERE id = :reviewer_id"),
            {"reviewer_id": reviewer_id},
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        connection.execute(
            text("DELETE FROM reviewer_deletion_receipts WHERE id = :id"),
            {"id": deletion_id},
        )


def test_migrated_sqlite_rejects_server_verified_identity_without_participant_consent(
    tmp_path: Path,
) -> None:
    engine = _upgrade_sqlite(tmp_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO seasons (id, slug, name, status, official, manifest_sha256, "
                "prompt_registry_sha256, tool_registry_sha256, epicure_release_id, "
                "epicure_bundle_sha256, epicure_application_sha256, analysis_plan_sha256, "
                "protocol_bundle_json, protocol_bundle_sha256, budget_cap_micros, "
                "budget_used_micros, budget_reserved_micros, created_at) "
                "VALUES ('legacy-season', 'legacy-season', 'Legacy season', 'draft', 0, "
                ":digest, :digest, :digest, 'epicure-test', :digest, :digest, :digest, '{}', "
                ":digest, 0, 0, 0, :created_at)"
            ),
            {"digest": _sha("legacy"), "created_at": NOW.isoformat()},
        )
        connection.execute(
            text(
                "INSERT INTO expert_reviewers (id, reviewer_code, invitation_sha256, "
                "qualification_json, qualification_verified, cohort, profile_json, "
                "batch_reveal_only, active, privacy_status, created_at) "
                "VALUES ('legacy-reviewer', 'legacy-reviewer', :digest, '[]', 0, "
                "'expert_independent', '{}', 1, 1, 'retained', :created_at)"
            ),
            {"digest": _sha("legacy-invitation"), "created_at": NOW.isoformat()},
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="participant-owned consent"),
    ):
        connection.execute(
            text(
                "INSERT INTO reviewer_identity_bindings (id, season_id, reviewer_id, "
                "person_commitment_sha256, identity_issuer_sha256, identity_evidence_sha256, "
                "hmac_key_id, verification_method, assurance_level, roles_json, created_at) "
                "VALUES ('legacy-binding', 'legacy-season', 'legacy-reviewer', :person, "
                ":issuer, :evidence, 'primary', 'season_hmac_issuer_subject_v1', "
                "'server_verified', '[\"output_rater\"]', :created_at)"
            ),
            {
                "person": _sha("legacy-person"),
                "issuer": _sha("legacy-issuer"),
                "evidence": _sha("legacy-evidence"),
                "created_at": NOW.isoformat(),
            },
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="participant-owned consent"),
    ):
        connection.execute(
            text(
                "INSERT INTO reviewer_identity_bindings (id, season_id, reviewer_id, "
                "person_commitment_sha256, identity_issuer_sha256, identity_evidence_sha256, "
                "hmac_key_id, verification_method, assurance_level, roles_json, created_at) "
                "VALUES ('legacy-unverified-binding', 'legacy-season', 'legacy-reviewer', "
                ":person, :issuer, :evidence, 'legacy', 'legacy_unverified', "
                "'legacy_unverified', '[\"output_rater\"]', :created_at)"
            ),
            {
                "person": _sha("legacy-unverified-person"),
                "issuer": _sha("legacy-unverified-issuer"),
                "evidence": _sha("legacy-unverified-evidence"),
                "created_at": NOW.isoformat(),
            },
        )


def test_migrated_sqlite_serializes_candidate_event_capacity(tmp_path: Path) -> None:
    engine = _upgrade_sqlite(tmp_path)
    raw_connection = engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        trigger_rows = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND tbl_name = 'task_validation_campaign_events'"
        ).fetchall()
        capacity_trigger = "trg_task_validation_campaign_events_candidate_capacity_v1"
        assert capacity_trigger in {str(row[0]) for row in trigger_rows}
        for (trigger_name,) in trigger_rows:
            if trigger_name != capacity_trigger:
                cursor.execute(f'DROP TRIGGER "{str(trigger_name).replace(chr(34), chr(34) * 2)}"')

        statement = (
            "INSERT INTO task_validation_campaign_events "
            "(id, season_id, campaign_sha256, sequence, event_id, event_type, "
            "candidate_id, reviewer_id, identity_binding_id, family_admission_id, "
            "audit_authorization_id, reviewer_pseudonym, person_commitment_sha256, "
            "reviewer_admission_receipt_sha256, payload_json, previous_event_sha256, "
            "event_sha256, created_at) VALUES (?, ?, ?, ?, ?, 'blind_ballot', ?, ?, ?, ?, "
            "NULL, ?, ?, ?, '{}', ?, ?, ?)"
        )
        campaign_sha256 = _sha("capacity-campaign")
        candidate_id = "capacity-candidate"

        def parameters(ordinal: int) -> tuple[object, ...]:
            return (
                f"capacity-event-{ordinal}",
                "capacity-season",
                campaign_sha256,
                ordinal,
                f"capacity-event-id-{ordinal}",
                candidate_id,
                f"capacity-reviewer-{ordinal}",
                f"capacity-binding-{ordinal}",
                f"capacity-admission-{ordinal}",
                f"capacity-pseudonym-{ordinal}",
                _sha(f"capacity-person-{ordinal}"),
                _sha(f"capacity-receipt-{ordinal}"),
                _sha(f"capacity-previous-{ordinal}"),
                _sha(f"capacity-event-sha-{ordinal}"),
                NOW.isoformat(),
            )

        cursor.execute(statement, parameters(1))
        cursor.execute(statement, parameters(2))
        with pytest.raises(
            sqlite3.IntegrityError,
            match="task-validation candidate event capacity is already sealed",
        ):
            cursor.execute(statement, parameters(3))
        raw_connection.rollback()
    finally:
        raw_connection.close()


def test_readiness_hashes_cover_every_postgresql_0035_function_and_trigger() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    postgresql_source = source.split("def _create_sqlite_guards", maxsplit=1)[0]
    observed = {
        name: hashlib.sha256(" ".join(body.split()).encode()).hexdigest()
        for name, body in re.findall(
            r"CREATE OR REPLACE FUNCTION public\."
            r"(flavourbench_[a-z0-9_]+)\(\).*?AS \$\$(.*?)\$\$",
            postgresql_source,
            flags=re.DOTALL,
        )
    }
    capacity_function = "flavourbench_task_validation_candidate_capacity_v1"
    assert (
        observed.pop(capacity_function)
        == _REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256[capacity_function]
    )
    assert observed == _PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256
    literal_triggers = set(re.findall(r"CREATE TRIGGER (trg_[a-z0-9_]+)", postgresql_source))
    assert {
        "trg_reviewer_consent_acceptances_guard_v1",
        "trg_reviewer_participation_lifecycles_insert_guard_v1",
        "trg_reviewer_withdrawal_receipts_guard_v1",
        "trg_reviewer_retention_schedules_guard_v1",
        "trg_reviewer_deletion_receipts_guard_v1",
        "trg_reviewer_enrollment_offers_lifecycle_v1",
        "trg_reviewer_participation_lifecycles_lifecycle_v1",
        "trg_expert_reviewers_privacy_lifecycle_v1",
        "trg_task_validation_campaign_events_candidate_capacity_v1",
    } <= literal_triggers
    assert len(_PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS) == 19
    assert "CREATE TRIGGER trg_{table}_append_only_v1" in postgresql_source
    assert "CREATE TRIGGER {trigger}" in postgresql_source
    assert {expected[1] for expected in _PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS.values()} == set(
        _PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256
    )
