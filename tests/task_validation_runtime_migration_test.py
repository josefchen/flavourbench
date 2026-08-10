from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DatabaseError

from flavourbench.task_validation_replay_binding import (
    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
    TASK_VALIDATION_RIGHTS_REQUIRED_IDS,
    TASK_VALIDATION_V1_REPLAY_SHA256,
    TASK_VALIDATION_V6_CAMPAIGN_SHA256,
    rights_audit_plan,
)


def _upgrade_fresh_database(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'task-validation-runtime.sqlite3'}"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("alembic")), "upgrade", "head"],
        cwd=project_root,
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


def test_task_validation_runtime_is_linear_and_append_only(tmp_path: Path) -> None:
    engine = _upgrade_fresh_database(tmp_path)
    inspector = inspect(engine)
    assert {
        "task_validation_audit_authorizations",
        "task_validation_campaign_events",
    } <= set(inspector.get_table_names())
    with engine.begin() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            "0035_participant_lifecycle_privacy"
        )
        trigger_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).scalars()
        )
        assert {
            "trg_task_validation_audit_authorizations_append_only_v1_update",
            "trg_task_validation_audit_authorizations_append_only_v1_delete",
            "trg_task_validation_campaign_events_append_only_v1_update",
            "trg_task_validation_campaign_events_append_only_v1_delete",
            "trg_task_validation_audit_authorizations_replay_v1",
            "trg_task_validation_campaign_events_replay_v1",
        } <= trigger_names
        index_names = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = "
                    "'task_validation_campaign_events'"
                )
            )
        }
        assert "uq_task_validation_campaign_events_audit_authorization_type" in index_names
        connection.execute(
            text(
                """
                INSERT INTO task_validation_campaign_events (
                    id, season_id, campaign_sha256, sequence, event_id, event_type,
                    candidate_id, reviewer_id, identity_binding_id, family_admission_id,
                    audit_authorization_id, reviewer_pseudonym,
                    person_commitment_sha256, reviewer_admission_receipt_sha256,
                    payload_json, previous_event_sha256, event_sha256, created_at
                ) VALUES (
                    :id, :season, :campaign, 1, :event_id, 'blind_ballot',
                    :candidate, :reviewer, :binding, :admission,
                    NULL, :pseudonym, :person_commitment, :receipt,
                    :payload, :previous, :event_sha, :created_at
                )
                """
            ),
            {
                "id": "event-1",
                "season": "season-1",
                "campaign": "a" * 64,
                "event_id": "submission-1",
                "candidate": "candidate-1",
                "reviewer": "reviewer-1",
                "binding": "binding-1",
                "admission": "admission-1",
                "pseudonym": "reviewer-pseudonym",
                "person_commitment": "b" * 64,
                "receipt": "c" * 64,
                "payload": "{}",
                "previous": "0" * 64,
                "event_sha": "d" * 64,
                "created_at": "2026-08-08T12:00:00+00:00",
            },
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="task-validation campaign evidence is append-only"),
    ):
        connection.execute(
            text(
                "UPDATE task_validation_campaign_events "
                "SET reviewer_pseudonym = 'changed' WHERE id = 'event-1'"
            )
        )
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="task-validation campaign evidence is append-only"),
    ):
        connection.execute(text("DELETE FROM task_validation_campaign_events WHERE id = 'event-1'"))


def test_postgresql_runtime_migration_keeps_0030_as_its_only_parent() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0031_task_validation_campaign_runtime.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0031_task_validation_campaign_runtime"' in migration
    assert 'down_revision = "0030_reviewer_identity_admission"' in migration
    assert "flavourbench_task_validation_append_only_v1" in migration
    assert "flavourbench_task_validation_event_guard_v1" in migration
    assert "task-validation campaign evidence is append-only" in migration
    assert "downgrade across reviewer identity" in migration
    assert "AS authorization" not in migration
    assert "AS audit_auth" in migration


def test_audit_singleton_migration_is_partial_and_irreversible() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0032_task_validation_audit_singleton.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0032_task_validation_audit_singleton"' in migration
    assert 'down_revision = "0031_task_validation_campaign_runtime"' in migration
    assert "audit_authorization_id IS NOT NULL" in migration
    assert "postgresql_where=predicate" in migration
    assert "sqlite_where=predicate" in migration
    assert "downgrade across sealed task-validation audit evidence" in migration


def test_reviewer_task_guard_hardening_is_linear_and_fail_closed() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0033_reviewer_task_guard_hardening.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0033_reviewer_task_guard_hardening"' in migration
    assert 'down_revision = "0032_task_validation_audit_singleton"' in migration
    assert migration.count("SET search_path = pg_catalog, public") == 6
    for qualified_table in (
        "public.reviewer_identity_bindings",
        "public.reviewer_qualification_evidence",
        "public.reviewer_calibration_ballots",
        "public.reviewer_calibration_sets",
        "public.reviewer_family_admissions",
        "public.expert_reviewers",
        "public.battles",
        "public.task_validation_audit_authorizations",
    ):
        assert qualified_table in migration
    assert "GRANT " not in migration
    assert "REVOKE " not in migration
    assert "downgrade across reviewer and task-validation guard hardening" in migration


def test_replay_binding_migration_is_linear_exact_and_irreversible() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0034_task_validation_replay_binding.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0034_task_validation_replay_binding"' in migration
    assert 'down_revision = "0033_reviewer_task_guard_hardening"' in migration
    assert TASK_VALIDATION_V6_CAMPAIGN_SHA256 in migration
    assert TASK_VALIDATION_V1_REPLAY_SHA256 in migration
    assert TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256 in migration
    assert all(method in migration for method in ("exact", "fuzzy", "ngram", "semantic", "web"))
    assert "contamination_campaign_coverage_verified" in migration
    assert "downgrade across task-validation automated replay bindings" in migration


def test_sqlite_replay_trigger_rejects_contamination_and_changed_plan_arrays(
    tmp_path: Path,
) -> None:
    engine = _upgrade_fresh_database(tmp_path)
    statement = text(
        """
        INSERT INTO task_validation_audit_authorizations (
            id, season_id, campaign_sha256, reviewer_id, identity_binding_id,
            audit_kind, cohort, qualification_evidence_sha256,
            conflict_evidence_sha256, automated_evidence_sha256,
            audit_plan_json, audit_plan_sha256, decision_reference_sha256,
            authorization_sha256, created_at
        ) VALUES (
            :id, 'season-1', :campaign, 'reviewer-1', 'binding-1',
            :audit_kind, 'expert_independent', :qualification, :conflict,
            :automated, :plan, :plan_sha, :decision, :authorization_sha,
            '2026-08-08T12:00:00+00:00'
        )
        """
    )
    base = {
        "campaign": TASK_VALIDATION_V6_CAMPAIGN_SHA256,
        "audit_kind": "rights",
        "qualification": "1" * 64,
        "conflict": "2" * 64,
        "automated": TASK_VALIDATION_V1_REPLAY_SHA256,
        "plan": json.dumps(rights_audit_plan()),
        "plan_sha": TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
        "decision": "3" * 64,
        "authorization_sha": "4" * 64,
    }
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="task-validation audit replay binding is inadmissible"),
    ):
        connection.execute(
            statement,
            {**base, "id": "contamination-auth", "audit_kind": "contamination"},
        )
    changed_plan = rights_audit_plan()
    changed_plan["sample_candidate_ids"] = list(reversed(changed_plan["sample_candidate_ids"]))
    with (
        engine.begin() as connection,
        pytest.raises(DatabaseError, match="task-validation audit replay binding is inadmissible"),
    ):
        connection.execute(
            statement,
            {**base, "id": "changed-plan-auth", "plan": json.dumps(changed_plan)},
        )


def test_sqlite_rejects_a_second_audit_for_one_authorization(tmp_path: Path) -> None:
    engine = _upgrade_fresh_database(tmp_path)
    plan = rights_audit_plan()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO task_validation_audit_authorizations (
                    id, season_id, campaign_sha256, reviewer_id, identity_binding_id,
                    audit_kind, cohort, qualification_evidence_sha256,
                    conflict_evidence_sha256, automated_evidence_sha256,
                    audit_plan_json, audit_plan_sha256, decision_reference_sha256,
                    authorization_sha256, created_at
                ) VALUES (
                    :id, :season, :campaign, :reviewer, :binding,
                    'rights', 'expert_independent', :qualification, :conflict,
                    :automated, :plan, :plan_sha, :decision, :authorization_sha,
                    :created_at
                )
                """
            ),
            {
                "id": "audit-authorization-1",
                "season": "season-1",
                "campaign": TASK_VALIDATION_V6_CAMPAIGN_SHA256,
                "reviewer": "reviewer-1",
                "binding": "binding-1",
                "qualification": "1" * 64,
                "conflict": "2" * 64,
                "automated": TASK_VALIDATION_V1_REPLAY_SHA256,
                "plan": json.dumps(plan),
                "plan_sha": TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
                "decision": "3" * 64,
                "authorization_sha": "4" * 64,
                "created_at": "2026-08-08T12:00:00+00:00",
            },
        )
    statement = text(
        """
        INSERT INTO task_validation_campaign_events (
            id, season_id, campaign_sha256, sequence, event_id, event_type,
            candidate_id, reviewer_id, identity_binding_id, family_admission_id,
            audit_authorization_id, reviewer_pseudonym,
            person_commitment_sha256, reviewer_admission_receipt_sha256,
            payload_json, previous_event_sha256, event_sha256, created_at
        ) VALUES (
            :id, :season, :campaign, :sequence, :event_id, 'rights_batch_audit',
            NULL, :reviewer, :binding, NULL,
            :authorization, :pseudonym,
            :person_commitment, :receipt,
            :payload, :previous, :event_sha, :created_at
        )
        """
    )
    common = {
        "season": "season-1",
        "campaign": TASK_VALIDATION_V6_CAMPAIGN_SHA256,
        "reviewer": "reviewer-1",
        "binding": "binding-1",
        "authorization": "audit-authorization-1",
        "pseudonym": "reviewer-pseudonym",
        "person_commitment": "b" * 64,
        "receipt": "4" * 64,
        "payload": json.dumps(
            {
                "audit_kind": "rights",
                "audit_plan_sha256": TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
                "automated_evidence_sha256": TASK_VALIDATION_V1_REPLAY_SHA256,
                "automated_evidence_verified": True,
                "rights_snapshot_integrity_verified": True,
                "local_prompt_risk_replay_verified": True,
                "contamination_campaign_coverage_verified": False,
                "reviewed_candidate_ids": list(TASK_VALIDATION_RIGHTS_REQUIRED_IDS),
            }
        ),
        "previous": "0" * 64,
        "created_at": "2026-08-08T12:00:00+00:00",
    }
    invalid_payload = json.loads(common["payload"])
    invalid_payload["reviewed_candidate_ids"] = list(reversed(TASK_VALIDATION_RIGHTS_REQUIRED_IDS))
    with (
        engine.begin() as connection,
        pytest.raises(
            DatabaseError,
            match="task-validation audit event replay binding is inadmissible",
        ),
    ):
        connection.execute(
            statement,
            {
                **common,
                "id": "invalid-audit-event",
                "sequence": 1,
                "event_id": "invalid-audit-submission",
                "payload": json.dumps(invalid_payload),
                "event_sha": "c" * 64,
            },
        )
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                **common,
                "id": "audit-event-1",
                "sequence": 1,
                "event_id": "audit-submission-1",
                "event_sha": "d" * 64,
            },
        )
        with pytest.raises(DatabaseError, match="UNIQUE constraint failed"):
            connection.execute(
                statement,
                {
                    **common,
                    "id": "audit-event-2",
                    "sequence": 2,
                    "event_id": "audit-submission-2",
                    "event_sha": "e" * 64,
                },
            )
