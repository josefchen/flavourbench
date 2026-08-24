"""Add the v6 public-source task-validation campaign runtime.

Revision ID: 0031_task_validation_campaign_runtime
Revises: 0030_reviewer_identity_admission
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0031_task_validation_campaign_runtime"
down_revision = "0030_reviewer_identity_admission"
branch_labels = None
depends_on = None


def _create_tables() -> None:
    op.create_table(
        "task_validation_audit_authorizations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("campaign_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("audit_kind", sa.String(length=32), nullable=False),
        sa.Column("cohort", sa.String(length=48), nullable=False),
        sa.Column("qualification_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("conflict_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("automated_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("audit_plan_json", sa.JSON(), nullable=False),
        sa.Column("audit_plan_sha256", sa.String(length=64), nullable=False),
        sa.Column("decision_reference_sha256", sa.String(length=64), nullable=False),
        sa.Column("authorization_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "audit_kind IN ('rights', 'contamination')",
            name="ck_task_validation_audit_authorizations_kind",
        ),
        sa.CheckConstraint(
            "cohort = 'expert_independent'",
            name="ck_task_validation_audit_authorizations_cohort",
        ),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "campaign_sha256",
            "audit_kind",
            name="uq_task_validation_audit_authorizations_campaign_kind",
        ),
        sa.UniqueConstraint(
            "campaign_sha256",
            "identity_binding_id",
            name="uq_task_validation_audit_authorizations_campaign_person",
        ),
        sa.UniqueConstraint(
            "authorization_sha256",
            name="uq_task_validation_audit_authorizations_digest",
        ),
    )
    for column in (
        "audit_kind",
        "authorization_sha256",
        "campaign_sha256",
        "identity_binding_id",
        "reviewer_id",
        "season_id",
    ):
        op.create_index(
            f"ix_task_validation_audit_authorizations_{column}",
            "task_validation_audit_authorizations",
            [column],
        )

    op.create_table(
        "task_validation_campaign_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("campaign_sha256", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.String(length=160), nullable=False),
        sa.Column("event_type", sa.String(length=48), nullable=False),
        sa.Column("candidate_id", sa.String(length=36), nullable=True),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("family_admission_id", sa.String(length=36), nullable=True),
        sa.Column("audit_authorization_id", sa.String(length=36), nullable=True),
        sa.Column("reviewer_pseudonym", sa.String(length=120), nullable=False),
        sa.Column("person_commitment_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewer_admission_receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("previous_event_sha256", sa.String(length=64), nullable=False),
        sa.Column("event_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('blind_ballot', 'criterion_pack_confirmation', "
            "'adjudication', 'rights_batch_audit', 'contamination_batch_audit')",
            name="ck_task_validation_campaign_events_type",
        ),
        sa.CheckConstraint(
            "sequence >= 1",
            name="ck_task_validation_campaign_events_sequence",
        ),
        sa.CheckConstraint(
            "((event_type IN ('blind_ballot', 'criterion_pack_confirmation', "
            "'adjudication') AND candidate_id IS NOT NULL "
            "AND family_admission_id IS NOT NULL AND audit_authorization_id IS NULL) OR "
            "(event_type IN ('rights_batch_audit', 'contamination_batch_audit') "
            "AND candidate_id IS NULL AND family_admission_id IS NULL "
            "AND audit_authorization_id IS NOT NULL))",
            name="ck_task_validation_campaign_events_authorization_shape",
        ),
        sa.ForeignKeyConstraint(
            ["audit_authorization_id"], ["task_validation_audit_authorizations.id"]
        ),
        sa.ForeignKeyConstraint(["family_admission_id"], ["reviewer_family_admissions.id"]),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "campaign_sha256",
            "candidate_id",
            "event_type",
            "identity_binding_id",
            name="uq_task_validation_campaign_events_candidate_person_type",
        ),
        sa.UniqueConstraint(
            "event_sha256",
            name="uq_task_validation_campaign_events_digest",
        ),
        sa.UniqueConstraint(
            "campaign_sha256",
            "event_id",
            name="uq_task_validation_campaign_events_event_id",
        ),
        sa.UniqueConstraint(
            "campaign_sha256",
            "sequence",
            name="uq_task_validation_campaign_events_sequence",
        ),
    )
    for column in (
        "audit_authorization_id",
        "campaign_sha256",
        "candidate_id",
        "event_sha256",
        "event_type",
        "family_admission_id",
        "identity_binding_id",
        "reviewer_id",
        "reviewer_pseudonym",
        "season_id",
    ):
        op.create_index(
            f"ix_task_validation_campaign_events_{column}",
            "task_validation_campaign_events",
            [column],
        )


def _create_postgresql_guards() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_task_validation_append_only_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'task-validation campaign evidence is append-only';
        END;
        $$
        """
    )
    for table in (
        "task_validation_audit_authorizations",
        "task_validation_campaign_events",
    ):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_append_only_v1
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION flavourbench_task_validation_append_only_v1()
            """
        )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_task_validation_event_guard_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE aligned_count integer;
        DECLARE expected_role text;
        DECLARE expected_kind text;
        BEGIN
          SELECT count(*) INTO aligned_count
          FROM reviewer_identity_bindings AS binding
          JOIN expert_reviewers AS reviewer ON reviewer.id = NEW.reviewer_id
          WHERE binding.id = NEW.identity_binding_id
            AND binding.season_id = NEW.season_id
            AND binding.reviewer_id = NEW.reviewer_id
            AND binding.person_commitment_sha256 = NEW.person_commitment_sha256
            AND binding.assurance_level = 'server_verified'
            AND reviewer.active IS TRUE;
          IF aligned_count <> 1 THEN
            RAISE EXCEPTION 'task-validation event identity is inadmissible';
          END IF;

          IF NEW.event_type IN (
              'blind_ballot', 'criterion_pack_confirmation', 'adjudication'
          ) THEN
            expected_role := CASE WHEN NEW.event_type = 'adjudication'
                                  THEN 'task_adjudicator' ELSE 'task_validator' END;
            SELECT count(*) INTO aligned_count
            FROM reviewer_family_admissions AS admission
            WHERE admission.id = NEW.family_admission_id
              AND admission.season_id = NEW.season_id
              AND admission.reviewer_id = NEW.reviewer_id
              AND admission.identity_binding_id = NEW.identity_binding_id
              AND admission.review_role = expected_role
              AND admission.cohort = 'expert_independent'
              AND admission.evidence_bundle_sha256 =
                  NEW.reviewer_admission_receipt_sha256
              AND COALESCE(NEW.created_at, now()) BETWEEN admission.valid_from
                                                      AND admission.valid_until;
          ELSE
            expected_kind := CASE WHEN NEW.event_type = 'rights_batch_audit'
                                  THEN 'rights' ELSE 'contamination' END;
            SELECT count(*) INTO aligned_count
            FROM task_validation_audit_authorizations AS audit_auth
            WHERE audit_auth.id = NEW.audit_authorization_id
              AND audit_auth.season_id = NEW.season_id
              AND audit_auth.reviewer_id = NEW.reviewer_id
              AND audit_auth.identity_binding_id = NEW.identity_binding_id
              AND audit_auth.audit_kind = expected_kind
              AND audit_auth.authorization_sha256 =
                  NEW.reviewer_admission_receipt_sha256;
          END IF;
          IF aligned_count <> 1 THEN
            RAISE EXCEPTION 'task-validation event authority is inadmissible';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_campaign_events_authority_v1
        BEFORE INSERT ON task_validation_campaign_events
        FOR EACH ROW EXECUTE FUNCTION flavourbench_task_validation_event_guard_v1()
        """
    )


def _create_sqlite_guards() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    for table in (
        "task_validation_audit_authorizations",
        "task_validation_campaign_events",
    ):
        for operation in ("update", "delete"):
            op.execute(
                f"""
                CREATE TRIGGER trg_{table}_append_only_v1_{operation}
                BEFORE {operation.upper()} ON {table} FOR EACH ROW
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'task-validation campaign evidence is append-only'
                    );
                END;
                """
            )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0031: {dialect}")
    _create_tables()
    _create_postgresql_guards()
    _create_sqlite_guards()


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across reviewer identity, task-validation ballots, and audit evidence "
        "is prohibited"
    )
