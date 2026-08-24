"""Add participant-owned consent, withdrawal, and privacy lifecycle.

Revision ID: 0035_participant_lifecycle_privacy
Revises: 0034_task_validation_replay_binding
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0035_participant_lifecycle_privacy"
down_revision = "0034_task_validation_replay_binding"
branch_labels = None
depends_on = None


def _add_reviewer_privacy_columns() -> None:
    op.add_column(
        "expert_reviewers",
        sa.Column(
            "privacy_status",
            sa.String(length=24),
            nullable=False,
            server_default="retained",
        ),
    )
    op.add_column(
        "expert_reviewers",
        sa.Column("privacy_redacted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "expert_reviewers",
        sa.Column(
            "privacy_redaction_receipt_sha256",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_expert_reviewers_privacy_status",
        "expert_reviewers",
        ["privacy_status"],
    )
    op.create_index(
        "ix_expert_reviewers_privacy_redaction_receipt_sha256",
        "expert_reviewers",
        ["privacy_redaction_receipt_sha256"],
    )
    if op.get_bind().dialect.name == "postgresql":
        op.create_check_constraint(
            "ck_expert_reviewers_privacy_status",
            "expert_reviewers",
            "(privacy_status = 'retained' AND privacy_redacted_at IS NULL "
            "AND privacy_redaction_receipt_sha256 IS NULL) OR "
            "(privacy_status = 'redacted' AND privacy_redacted_at IS NOT NULL "
            "AND privacy_redaction_receipt_sha256 IS NOT NULL)",
        )


def _create_tables() -> None:
    op.create_table(
        "reviewer_enrollment_offers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("credential_prefix", sa.String(length=32), nullable=False),
        sa.Column("secret_hmac_sha256", sa.String(length=64), nullable=False),
        sa.Column("hmac_key_id", sa.String(length=64), nullable=False),
        sa.Column("consent_document_sha256", sa.String(length=64), nullable=False),
        sa.Column("activation_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_request_sha256", sa.String(length=64), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'accepted', 'revoked')",
            name="ck_reviewer_enrollment_offers_status",
        ),
        sa.CheckConstraint(
            "expires_at > not_before",
            name="ck_reviewer_enrollment_offers_window",
        ),
        sa.CheckConstraint(
            "(status = 'active' AND accepted_at IS NULL AND revoked_at IS NULL "
            "AND accepted_request_sha256 IS NULL) OR "
            "(status = 'accepted' AND accepted_at IS NOT NULL AND revoked_at IS NULL "
            "AND accepted_request_sha256 IS NOT NULL) OR "
            "(status = 'revoked' AND accepted_at IS NULL AND revoked_at IS NOT NULL "
            "AND accepted_request_sha256 IS NULL)",
            name="ck_reviewer_enrollment_offers_terminal_shape",
        ),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("credential_prefix", name="uq_reviewer_enrollment_offers_prefix"),
        sa.UniqueConstraint("secret_hmac_sha256", name="uq_reviewer_enrollment_offers_secret_hmac"),
    )
    for column in (
        "activation_manifest_sha256",
        "consent_document_sha256",
        "credential_prefix",
        "expires_at",
        "season_id",
        "status",
    ):
        op.create_index(
            f"ix_reviewer_enrollment_offers_{column}",
            "reviewer_enrollment_offers",
            [column],
        )

    op.create_table(
        "reviewer_consent_acceptances",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("enrollment_offer_id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("consent_document_sha256", sa.String(length=64), nullable=False),
        sa.Column("activation_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("retention_policy_sha256", sa.String(length=64), nullable=False),
        sa.Column("acceptance_statement_sha256", sa.String(length=64), nullable=False),
        sa.Column("confirmation_set_sha256", sa.String(length=64), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("receipt_prefix", sa.String(length=32), nullable=False),
        sa.Column("receipt_secret_hmac_sha256", sa.String(length=64), nullable=False),
        sa.Column("hmac_key_id", sa.String(length=64), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["enrollment_offer_id"], ["reviewer_enrollment_offers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("enrollment_offer_id", name="uq_reviewer_consent_acceptances_offer"),
        sa.UniqueConstraint("request_sha256", name="uq_reviewer_consent_acceptances_request"),
        sa.UniqueConstraint("receipt_prefix", name="uq_reviewer_consent_acceptances_prefix"),
        sa.UniqueConstraint(
            "receipt_secret_hmac_sha256",
            name="uq_reviewer_consent_acceptances_receipt_secret",
        ),
        sa.UniqueConstraint("receipt_sha256", name="uq_reviewer_consent_acceptances_receipt"),
    )
    for column in (
        "activation_manifest_sha256",
        "consent_document_sha256",
        "enrollment_offer_id",
        "receipt_prefix",
        "receipt_sha256",
        "request_sha256",
        "season_id",
    ):
        op.create_index(
            f"ix_reviewer_consent_acceptances_{column}",
            "reviewer_consent_acceptances",
            [column],
        )

    op.create_table(
        "reviewer_participation_lifecycles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("consent_acceptance_id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("audit_marker_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("assignments_stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("withdrawal_receipt_sha256", sa.String(length=64), nullable=True),
        sa.Column("redacted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deletion_receipt_sha256", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'withdrawn', 'redacted')",
            name="ck_reviewer_participation_lifecycles_status",
        ),
        sa.CheckConstraint(
            "(status = 'active' AND withdrawn_at IS NULL AND assignments_stopped_at IS NULL "
            "AND withdrawal_receipt_sha256 IS NULL AND redacted_at IS NULL "
            "AND deletion_receipt_sha256 IS NULL) OR "
            "(status = 'withdrawn' AND withdrawn_at IS NOT NULL "
            "AND assignments_stopped_at IS NOT NULL AND withdrawal_receipt_sha256 IS NOT NULL "
            "AND redacted_at IS NULL AND deletion_receipt_sha256 IS NULL) OR "
            "(status = 'redacted' AND withdrawn_at IS NOT NULL "
            "AND assignments_stopped_at IS NOT NULL AND withdrawal_receipt_sha256 IS NOT NULL "
            "AND redacted_at IS NOT NULL AND deletion_receipt_sha256 IS NOT NULL)",
            name="ck_reviewer_participation_lifecycles_terminal_shape",
        ),
        sa.ForeignKeyConstraint(["consent_acceptance_id"], ["reviewer_consent_acceptances.id"]),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "consent_acceptance_id",
            name="uq_reviewer_participation_lifecycles_acceptance",
        ),
        sa.UniqueConstraint(
            "identity_binding_id",
            name="uq_reviewer_participation_lifecycles_binding",
        ),
        sa.UniqueConstraint(
            "season_id",
            "reviewer_id",
            name="uq_reviewer_participation_lifecycles_reviewer",
        ),
        sa.UniqueConstraint("audit_marker_sha256"),
    )
    for column in (
        "audit_marker_sha256",
        "consent_acceptance_id",
        "deletion_receipt_sha256",
        "identity_binding_id",
        "reviewer_id",
        "season_id",
        "status",
        "withdrawal_receipt_sha256",
    ):
        op.create_index(
            f"ix_reviewer_participation_lifecycles_{column}",
            "reviewer_participation_lifecycles",
            [column],
        )

    op.create_table(
        "reviewer_withdrawal_receipts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("lifecycle_id", sa.String(length=36), nullable=False),
        sa.Column("consent_acceptance_id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("reason_code", sa.String(length=32), nullable=False),
        sa.Column("credentials_revoked_count", sa.Integer(), nullable=False),
        sa.Column("assignments_stopped_count", sa.Integer(), nullable=False),
        sa.Column("prior_judgments_preserved", sa.Boolean(), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "reason_code IN ('voluntary_withdrawal', 'privacy_request', 'safety_concern')",
            name="ck_reviewer_withdrawal_receipts_reason",
        ),
        sa.CheckConstraint(
            "credentials_revoked_count >= 0 AND assignments_stopped_count >= 0",
            name="ck_reviewer_withdrawal_receipts_counts",
        ),
        sa.CheckConstraint(
            "prior_judgments_preserved = true",
            name="ck_reviewer_withdrawal_receipts_history",
        ),
        sa.ForeignKeyConstraint(["consent_acceptance_id"], ["reviewer_consent_acceptances.id"]),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["lifecycle_id"], ["reviewer_participation_lifecycles.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lifecycle_id", name="uq_reviewer_withdrawal_receipts_lifecycle"),
        sa.UniqueConstraint("request_sha256", name="uq_reviewer_withdrawal_receipts_request"),
        sa.UniqueConstraint("receipt_sha256", name="uq_reviewer_withdrawal_receipts_receipt"),
    )
    for column in (
        "consent_acceptance_id",
        "identity_binding_id",
        "lifecycle_id",
        "receipt_sha256",
        "request_sha256",
        "reviewer_id",
        "season_id",
    ):
        op.create_index(
            f"ix_reviewer_withdrawal_receipts_{column}",
            "reviewer_withdrawal_receipts",
            [column],
        )

    op.create_table(
        "reviewer_retention_schedules",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("lifecycle_id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("analysis_freeze_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("first_public_release_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("direct_payload_delete_due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pseudonymous_audit_retain_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_policy_sha256", sa.String(length=64), nullable=False),
        sa.Column("schedule_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "direct_payload_delete_due_at > analysis_freeze_at",
            name="ck_reviewer_retention_schedules_direct_deadline",
        ),
        sa.CheckConstraint(
            "pseudonymous_audit_retain_until > first_public_release_at",
            name="ck_reviewer_retention_schedules_audit_deadline",
        ),
        sa.ForeignKeyConstraint(["lifecycle_id"], ["reviewer_participation_lifecycles.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lifecycle_id", name="uq_reviewer_retention_schedules_lifecycle"),
        sa.UniqueConstraint("schedule_sha256", name="uq_reviewer_retention_schedules_digest"),
    )
    for column in (
        "direct_payload_delete_due_at",
        "lifecycle_id",
        "pseudonymous_audit_retain_until",
        "reviewer_id",
        "schedule_sha256",
        "season_id",
    ):
        op.create_index(
            f"ix_reviewer_retention_schedules_{column}",
            "reviewer_retention_schedules",
            [column],
        )

    op.create_table(
        "reviewer_deletion_receipts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("lifecycle_id", sa.String(length=36), nullable=False),
        sa.Column("retention_schedule_id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("execution_basis", sa.String(length=32), nullable=False),
        sa.Column("redacted_fields_json", sa.JSON(), nullable=False),
        sa.Column("private_payload_before_sha256", sa.String(length=64), nullable=False),
        sa.Column("audit_marker_sha256", sa.String(length=64), nullable=False),
        sa.Column("direct_payload_delete_due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pseudonymous_audit_retain_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prior_judgments_preserved", sa.Boolean(), nullable=False),
        sa.Column("receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "execution_basis IN ('scheduled_retention', 'participant_request')",
            name="ck_reviewer_deletion_receipts_basis",
        ),
        sa.CheckConstraint(
            "prior_judgments_preserved = true",
            name="ck_reviewer_deletion_receipts_history",
        ),
        sa.ForeignKeyConstraint(["lifecycle_id"], ["reviewer_participation_lifecycles.id"]),
        sa.ForeignKeyConstraint(["retention_schedule_id"], ["reviewer_retention_schedules.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lifecycle_id", name="uq_reviewer_deletion_receipts_lifecycle"),
        sa.UniqueConstraint("request_sha256", name="uq_reviewer_deletion_receipts_request"),
        sa.UniqueConstraint("receipt_sha256", name="uq_reviewer_deletion_receipts_receipt"),
    )
    for column in (
        "lifecycle_id",
        "receipt_sha256",
        "request_sha256",
        "retention_schedule_id",
        "reviewer_id",
        "season_id",
    ):
        op.create_index(
            f"ix_reviewer_deletion_receipts_{column}",
            "reviewer_deletion_receipts",
            [column],
        )


def _assert_task_validation_candidate_capacity() -> None:
    overfilled = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT campaign_sha256, candidate_id, event_type, count(*) AS event_count "
                "FROM task_validation_campaign_events "
                "WHERE candidate_id IS NOT NULL AND event_type IN "
                "('blind_ballot', 'criterion_pack_confirmation', 'adjudication') "
                "GROUP BY campaign_sha256, candidate_id, event_type "
                "HAVING (event_type IN ('blind_ballot', 'criterion_pack_confirmation') "
                "AND count(*) > 2) OR (event_type = 'adjudication' AND count(*) > 1) "
                "LIMIT 1"
            )
        )
        .first()
    )
    if overfilled is not None:
        raise RuntimeError(
            "0035 refuses a task-validation ledger with overfilled candidate event slots"
        )


def _create_postgresql_guards() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_task_validation_candidate_capacity_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        DECLARE event_count integer;
        DECLARE event_limit integer;
        BEGIN
          IF NEW.candidate_id IS NULL OR NEW.event_type NOT IN (
            'blind_ballot', 'criterion_pack_confirmation', 'adjudication'
          ) THEN
            RETURN NEW;
          END IF;
          event_limit := CASE WHEN NEW.event_type = 'adjudication' THEN 1 ELSE 2 END;
          PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
              NEW.campaign_sha256 || ':' || NEW.candidate_id || ':' || NEW.event_type,
              0
            )
          );
          EXECUTE pg_catalog.format(
            'SELECT count(*) FROM %s AS event '
            'WHERE event.campaign_sha256 = $1 '
            'AND event.candidate_id = $2 AND event.event_type = $3',
            TG_RELID::pg_catalog.regclass
          ) INTO event_count USING
            NEW.campaign_sha256, NEW.candidate_id, NEW.event_type;
          IF event_count >= event_limit THEN
            RAISE EXCEPTION 'task-validation candidate event capacity is already sealed';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_campaign_events_candidate_capacity_v1
        BEFORE INSERT ON public.task_validation_campaign_events
        FOR EACH ROW EXECUTE FUNCTION
          public.flavourbench_task_validation_candidate_capacity_v1()
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_participant_append_only_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          RAISE EXCEPTION 'participant lifecycle receipts and schedules are append-only';
        END;
        $$
        """
    )
    for table in (
        "reviewer_consent_acceptances",
        "reviewer_withdrawal_receipts",
        "reviewer_retention_schedules",
        "reviewer_deletion_receipts",
    ):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_append_only_v1
            BEFORE UPDATE OR DELETE ON public.{table}
            FOR EACH ROW EXECUTE FUNCTION public.flavourbench_participant_append_only_v1()
            """
        )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_consent_acceptance_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.retention_policy_sha256 !~ '^[0-9a-f]{64}$'
             OR NEW.receipt_secret_hmac_sha256 !~ '^[0-9a-f]{64}$'
             OR NEW.receipt_sha256 !~ '^[0-9a-f]{64}$'
             OR NOT EXISTS (
               SELECT 1 FROM public.reviewer_enrollment_offers AS offer
               WHERE offer.id = NEW.enrollment_offer_id
                 AND offer.season_id = NEW.season_id
                 AND offer.consent_document_sha256 = NEW.consent_document_sha256
                 AND offer.activation_manifest_sha256 = NEW.activation_manifest_sha256
                 AND offer.status = 'accepted'
                 AND offer.accepted_request_sha256 = NEW.request_sha256
                 AND offer.accepted_at = NEW.accepted_at
                 AND NEW.accepted_at BETWEEN offer.not_before AND offer.expires_at
             ) THEN
            RAISE EXCEPTION 'participant consent acceptance is not bound to one consumed offer';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_consent_acceptances_guard_v1
        BEFORE INSERT ON public.reviewer_consent_acceptances
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_consent_acceptance_guard_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_participation_insert_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.status <> 'active' OR NOT EXISTS (
            SELECT 1
            FROM public.reviewer_consent_acceptances AS acceptance
            JOIN public.reviewer_identity_bindings AS binding
              ON binding.id = NEW.identity_binding_id
            JOIN public.expert_reviewers AS reviewer
              ON reviewer.id = NEW.reviewer_id
            WHERE acceptance.id = NEW.consent_acceptance_id
              AND acceptance.season_id = NEW.season_id
              AND binding.season_id = NEW.season_id
              AND binding.reviewer_id = NEW.reviewer_id
              AND binding.assurance_level = 'server_verified'
              AND reviewer.active IS TRUE
              AND reviewer.privacy_status = 'retained'
              AND reviewer.profile_json->>'consent_acceptance_sha256' =
                  acceptance.receipt_sha256
              AND reviewer.profile_json->>'activation_manifest_sha256' =
                  acceptance.activation_manifest_sha256
          ) THEN
            RAISE EXCEPTION 'participant lifecycle consent and identity are misaligned';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_participation_lifecycles_insert_guard_v1
        BEFORE INSERT ON public.reviewer_participation_lifecycles
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_participation_insert_guard_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_withdrawal_receipt_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.prior_judgments_preserved IS NOT TRUE OR NOT EXISTS (
            SELECT 1
            FROM public.reviewer_participation_lifecycles AS lifecycle
            JOIN public.expert_reviewers AS reviewer ON reviewer.id = lifecycle.reviewer_id
            WHERE lifecycle.id = NEW.lifecycle_id
              AND lifecycle.consent_acceptance_id = NEW.consent_acceptance_id
              AND lifecycle.season_id = NEW.season_id
              AND lifecycle.reviewer_id = NEW.reviewer_id
              AND lifecycle.identity_binding_id = NEW.identity_binding_id
              AND lifecycle.status = 'active'
              AND reviewer.active IS FALSE
              AND reviewer.revoked_at IS NOT NULL
          ) OR EXISTS (
            SELECT 1 FROM public.reviewer_access_credentials AS credential
            WHERE credential.reviewer_id = NEW.reviewer_id
              AND credential.status = 'active'
          ) OR EXISTS (
            SELECT 1 FROM public.controlled_run_reviewers AS assignment
            WHERE assignment.reviewer_id = NEW.reviewer_id
              AND assignment.active IS TRUE
          ) THEN
            RAISE EXCEPTION 'participant withdrawal did not atomically stop forward authority';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_withdrawal_receipts_guard_v1
        BEFORE INSERT ON public.reviewer_withdrawal_receipts
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_withdrawal_receipt_guard_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_retention_schedule_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.direct_payload_delete_due_at <>
                 NEW.analysis_freeze_at + INTERVAL '12 months'
             OR NEW.pseudonymous_audit_retain_until <>
                 NEW.first_public_release_at + INTERVAL '60 months'
             OR NOT EXISTS (
               SELECT 1
               FROM public.reviewer_participation_lifecycles AS lifecycle
               JOIN public.reviewer_consent_acceptances AS acceptance
                 ON acceptance.id = lifecycle.consent_acceptance_id
               WHERE lifecycle.id = NEW.lifecycle_id
                 AND lifecycle.season_id = NEW.season_id
                 AND lifecycle.reviewer_id = NEW.reviewer_id
                 AND lifecycle.status IN ('active', 'withdrawn')
                 AND acceptance.retention_policy_sha256 = NEW.retention_policy_sha256
             ) THEN
            RAISE EXCEPTION 'reviewer retention schedule scope or deadlines are invalid';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_retention_schedules_guard_v1
        BEFORE INSERT ON public.reviewer_retention_schedules
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_retention_schedule_guard_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_deletion_receipt_guard_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF NEW.prior_judgments_preserved IS NOT TRUE
             OR NEW.redacted_fields_json::jsonb <>
                pg_catalog.jsonb_build_array(
                  'expert_reviewers.profile_json',
                  'expert_reviewers.qualification_json',
                  'expert_reviewers.qualification_verified'
                )
             OR (NEW.execution_basis = 'scheduled_retention'
                 AND CURRENT_TIMESTAMP < NEW.direct_payload_delete_due_at)
             OR NOT EXISTS (
               SELECT 1
               FROM public.reviewer_participation_lifecycles AS lifecycle
               JOIN public.reviewer_retention_schedules AS schedule
                 ON schedule.id = NEW.retention_schedule_id
               JOIN public.expert_reviewers AS reviewer
                 ON reviewer.id = lifecycle.reviewer_id
               WHERE lifecycle.id = NEW.lifecycle_id
                 AND lifecycle.season_id = NEW.season_id
                 AND lifecycle.reviewer_id = NEW.reviewer_id
                 AND lifecycle.audit_marker_sha256 = NEW.audit_marker_sha256
                 AND lifecycle.status = 'withdrawn'
                 AND schedule.lifecycle_id = lifecycle.id
                 AND schedule.direct_payload_delete_due_at =
                     NEW.direct_payload_delete_due_at
                 AND schedule.pseudonymous_audit_retain_until =
                     NEW.pseudonymous_audit_retain_until
                 AND reviewer.active IS FALSE
                 AND reviewer.privacy_status = 'retained'
             ) THEN
            RAISE EXCEPTION 'reviewer deletion receipt scope or deadline is invalid';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_deletion_receipts_guard_v1
        BEFORE INSERT ON public.reviewer_deletion_receipts
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_deletion_receipt_guard_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_enrollment_offer_lifecycle_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            RAISE EXCEPTION 'reviewer enrollment offers cannot be deleted';
          END IF;
          IF (NEW.id, NEW.season_id, NEW.credential_prefix, NEW.secret_hmac_sha256,
              NEW.hmac_key_id, NEW.consent_document_sha256,
              NEW.activation_manifest_sha256, NEW.not_before, NEW.expires_at,
              NEW.created_at)
             IS DISTINCT FROM
             (OLD.id, OLD.season_id, OLD.credential_prefix, OLD.secret_hmac_sha256,
              OLD.hmac_key_id, OLD.consent_document_sha256,
              OLD.activation_manifest_sha256, OLD.not_before, OLD.expires_at,
              OLD.created_at) THEN
            RAISE EXCEPTION 'reviewer enrollment offer contract is immutable';
          END IF;
          IF OLD.status <> 'active' OR NEW.status NOT IN ('accepted', 'revoked') THEN
            RAISE EXCEPTION 'terminal reviewer enrollment offer is immutable';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_enrollment_offers_lifecycle_v1
        BEFORE UPDATE OR DELETE ON public.reviewer_enrollment_offers
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_enrollment_offer_lifecycle_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_participation_lifecycle_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            RAISE EXCEPTION 'reviewer participation lifecycle cannot be deleted';
          END IF;
          IF (NEW.id, NEW.consent_acceptance_id, NEW.season_id, NEW.reviewer_id,
              NEW.identity_binding_id, NEW.audit_marker_sha256, NEW.created_at)
             IS DISTINCT FROM
             (OLD.id, OLD.consent_acceptance_id, OLD.season_id, OLD.reviewer_id,
              OLD.identity_binding_id, OLD.audit_marker_sha256, OLD.created_at) THEN
            RAISE EXCEPTION 'reviewer participation identity is immutable';
          END IF;
          IF OLD.status = 'active' AND NEW.status = 'withdrawn' THEN
            IF NOT EXISTS (
              SELECT 1 FROM public.reviewer_withdrawal_receipts AS receipt
              WHERE receipt.lifecycle_id = NEW.id
                AND receipt.receipt_sha256 = NEW.withdrawal_receipt_sha256
            ) THEN
              RAISE EXCEPTION 'withdrawal transition requires its append-only receipt';
            END IF;
          ELSIF OLD.status = 'withdrawn' AND NEW.status = 'redacted' THEN
            IF NOT EXISTS (
              SELECT 1 FROM public.reviewer_deletion_receipts AS receipt
              WHERE receipt.lifecycle_id = NEW.id
                AND receipt.receipt_sha256 = NEW.deletion_receipt_sha256
            ) THEN
              RAISE EXCEPTION 'redaction transition requires its append-only receipt';
            END IF;
          ELSE
            RAISE EXCEPTION 'reviewer participation lifecycle is monotone and terminal';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_participation_lifecycles_lifecycle_v1
        BEFORE UPDATE OR DELETE ON public.reviewer_participation_lifecycles
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_participation_lifecycle_v1()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_participant_forward_authority_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF TG_ARGV[0] = 'identity' THEN
            IF NOT EXISTS (
              SELECT 1
              FROM public.expert_reviewers AS reviewer
              JOIN public.reviewer_consent_acceptances AS acceptance
                ON acceptance.receipt_sha256 =
                   reviewer.profile_json->>'consent_acceptance_sha256'
              WHERE reviewer.id = NEW.reviewer_id
                AND acceptance.season_id = NEW.season_id
                AND acceptance.activation_manifest_sha256 =
                    reviewer.profile_json->>'activation_manifest_sha256'
                AND reviewer.active IS TRUE
                AND reviewer.privacy_status = 'retained'
            ) THEN
              RAISE EXCEPTION 'reviewer identity requires participant-owned consent';
            END IF;
          ELSIF TG_ARGV[0] = 'qualification' THEN
            IF NOT EXISTS (
              SELECT 1
              FROM public.reviewer_participation_lifecycles AS lifecycle
              JOIN public.reviewer_consent_acceptances AS acceptance
                ON acceptance.id = lifecycle.consent_acceptance_id
              WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
                AND lifecycle.reviewer_id = NEW.reviewer_id
                AND lifecycle.season_id = NEW.season_id
                AND lifecycle.status = 'active'
                AND acceptance.consent_document_sha256 = NEW.consent_document_sha256
            ) THEN
              RAISE EXCEPTION 'reviewer qualification requires active participant consent';
            END IF;
          ELSIF TG_ARGV[0] = 'credential' THEN
            IF NEW.status = 'active' THEN
              IF NOT EXISTS (
                SELECT 1 FROM public.reviewer_participation_lifecycles AS lifecycle
                WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
                  AND lifecycle.reviewer_id = NEW.reviewer_id
                  AND lifecycle.season_id = NEW.season_id
                  AND lifecycle.status = 'active'
              ) THEN
                RAISE EXCEPTION 'reviewer credential requires active participation';
              END IF;
            END IF;
          ELSIF TG_ARGV[0] = 'assignment' THEN
            IF NEW.active IS TRUE THEN
              IF NOT EXISTS (
                SELECT 1
                FROM public.controlled_runs AS controlled_run
                JOIN public.reviewer_participation_lifecycles AS lifecycle
                  ON lifecycle.season_id = controlled_run.season_id
                WHERE controlled_run.id = NEW.controlled_run_id
                  AND lifecycle.reviewer_id = NEW.reviewer_id
                  AND lifecycle.status = 'active'
              ) THEN
                RAISE EXCEPTION 'reviewer assignment requires active participation';
              END IF;
            END IF;
          ELSIF TG_ARGV[0] = 'vote' THEN
            IF NEW.provenance_status = 'expert_verified_v1' THEN
              IF NOT EXISTS (
                SELECT 1 FROM public.reviewer_participation_lifecycles AS lifecycle
                WHERE lifecycle.identity_binding_id = NEW.reviewer_identity_binding_id
                  AND lifecycle.reviewer_id = NEW.reviewer_id
                  AND lifecycle.status = 'active'
              ) THEN
                RAISE EXCEPTION 'reviewer judgment requires active participation';
              END IF;
            END IF;
          ELSIF TG_ARGV[0] IN ('task_event', 'task_audit_authorization') THEN
            IF EXISTS (
              SELECT 1 FROM public.expert_reviewers AS reviewer
              WHERE reviewer.id = NEW.reviewer_id
                AND reviewer.profile_json::jsonb ? 'consent_acceptance_sha256'
            ) AND NOT EXISTS (
              SELECT 1 FROM public.reviewer_participation_lifecycles AS lifecycle
              WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
                AND lifecycle.reviewer_id = NEW.reviewer_id
                AND lifecycle.season_id = NEW.season_id
                AND lifecycle.status = 'active'
            ) THEN
              RAISE EXCEPTION 'task-validation work requires active participation';
            END IF;
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    for trigger, table, operation, argument in (
        (
            "trg_reviewer_identity_bindings_participant_consent_v1",
            "reviewer_identity_bindings",
            "INSERT",
            "identity",
        ),
        (
            "trg_reviewer_qualification_evidence_participant_consent_v1",
            "reviewer_qualification_evidence",
            "INSERT",
            "qualification",
        ),
        (
            "trg_reviewer_access_credentials_participation_v1",
            "reviewer_access_credentials",
            "INSERT OR UPDATE",
            "credential",
        ),
        (
            "trg_controlled_run_reviewers_participation_v1",
            "controlled_run_reviewers",
            "INSERT OR UPDATE",
            "assignment",
        ),
        (
            "trg_votes_participation_v1",
            "votes",
            "INSERT OR UPDATE",
            "vote",
        ),
        (
            "trg_task_validation_campaign_events_participation_v1",
            "task_validation_campaign_events",
            "INSERT",
            "task_event",
        ),
        (
            "trg_task_validation_audit_authorizations_participation_v1",
            "task_validation_audit_authorizations",
            "INSERT",
            "task_audit_authorization",
        ),
    ):
        op.execute(
            f"""
            CREATE TRIGGER {trigger}
            BEFORE {operation} ON public.{table}
            FOR EACH ROW EXECUTE FUNCTION
              public.flavourbench_participant_forward_authority_v1('{argument}')
            """
        )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.flavourbench_reviewer_privacy_lifecycle_v1()
        RETURNS trigger LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
          IF OLD.privacy_status = 'redacted' AND
             (NEW.profile_json::text, NEW.qualification_json::text,
              NEW.qualification_verified, NEW.privacy_status, NEW.privacy_redacted_at,
              NEW.privacy_redaction_receipt_sha256, NEW.active, NEW.revoked_at)
             IS DISTINCT FROM
             (OLD.profile_json::text, OLD.qualification_json::text,
              OLD.qualification_verified, OLD.privacy_status, OLD.privacy_redacted_at,
              OLD.privacy_redaction_receipt_sha256, OLD.active, OLD.revoked_at) THEN
            RAISE EXCEPTION 'redacted reviewer private payload is terminal';
          END IF;
          IF OLD.privacy_status = 'retained' AND NEW.privacy_status = 'redacted' THEN
            IF NEW.active IS TRUE OR NEW.revoked_at IS NULL
               OR NEW.qualification_json::jsonb <> '[]'::jsonb
               OR NEW.qualification_verified IS TRUE
               OR NEW.privacy_redacted_at IS NULL
               OR NOT EXISTS (
                 SELECT 1 FROM public.reviewer_deletion_receipts AS receipt
                 WHERE receipt.reviewer_id = NEW.id
                   AND receipt.receipt_sha256 = NEW.privacy_redaction_receipt_sha256
                   AND NEW.profile_json::jsonb = pg_catalog.jsonb_build_object(
                     'schema_version', 'flavourbench-reviewer-redacted-profile-v1',
                     'privacy_status', 'redacted',
                     'audit_marker_sha256', receipt.audit_marker_sha256,
                     'private_payload_before_sha256',
                       receipt.private_payload_before_sha256
                   )
               ) THEN
              RAISE EXCEPTION 'reviewer private payload redaction is incomplete';
            END IF;
          ELSIF OLD.privacy_status <> NEW.privacy_status THEN
            RAISE EXCEPTION 'reviewer privacy lifecycle is one-way';
          END IF;
          IF OLD.active IS FALSE AND NEW.active IS TRUE AND EXISTS (
            SELECT 1 FROM public.reviewer_participation_lifecycles AS lifecycle
            WHERE lifecycle.reviewer_id = NEW.id
              AND lifecycle.status IN ('withdrawn', 'redacted')
          ) THEN
            RAISE EXCEPTION 'withdrawn reviewer cannot be reactivated';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_expert_reviewers_privacy_lifecycle_v1
        BEFORE UPDATE ON public.expert_reviewers
        FOR EACH ROW EXECUTE FUNCTION public.flavourbench_reviewer_privacy_lifecycle_v1()
        """
    )


def _create_sqlite_guards() -> None:
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_campaign_events_candidate_capacity_v1
        BEFORE INSERT ON task_validation_campaign_events FOR EACH ROW
        WHEN NEW.candidate_id IS NOT NULL AND (
          (
            NEW.event_type IN ('blind_ballot', 'criterion_pack_confirmation')
            AND (
              SELECT count(*) FROM task_validation_campaign_events AS event
              WHERE event.campaign_sha256 = NEW.campaign_sha256
                AND event.candidate_id = NEW.candidate_id
                AND event.event_type = NEW.event_type
            ) >= 2
          ) OR (
            NEW.event_type = 'adjudication'
            AND (
              SELECT count(*) FROM task_validation_campaign_events AS event
              WHERE event.campaign_sha256 = NEW.campaign_sha256
                AND event.candidate_id = NEW.candidate_id
                AND event.event_type = NEW.event_type
            ) >= 1
          )
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'task-validation candidate event capacity is already sealed'
          );
        END;
        """
    )
    for table in (
        "reviewer_consent_acceptances",
        "reviewer_withdrawal_receipts",
        "reviewer_retention_schedules",
        "reviewer_deletion_receipts",
    ):
        for operation in ("update", "delete"):
            op.execute(
                f"""
                CREATE TRIGGER trg_{table}_append_only_v1_{operation}
                BEFORE {operation.upper()} ON {table} FOR EACH ROW
                BEGIN
                  SELECT RAISE(
                    ABORT,
                    'participant lifecycle receipts and schedules are append-only'
                  );
                END;
                """
            )

    op.execute(
        """
        CREATE TRIGGER trg_reviewer_consent_acceptances_guard_v1
        BEFORE INSERT ON reviewer_consent_acceptances FOR EACH ROW
        WHEN NOT EXISTS (
          SELECT 1 FROM reviewer_enrollment_offers AS offer
          WHERE offer.id = NEW.enrollment_offer_id
            AND offer.season_id = NEW.season_id
            AND offer.consent_document_sha256 = NEW.consent_document_sha256
            AND offer.activation_manifest_sha256 = NEW.activation_manifest_sha256
            AND offer.status = 'accepted'
            AND offer.accepted_request_sha256 = NEW.request_sha256
            AND offer.accepted_at IS NEW.accepted_at
            AND julianday(NEW.accepted_at) BETWEEN julianday(offer.not_before)
                                                AND julianday(offer.expires_at)
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'participant consent acceptance is not bound to one consumed offer'
          );
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_participation_lifecycles_insert_guard_v1
        BEFORE INSERT ON reviewer_participation_lifecycles FOR EACH ROW
        WHEN NEW.status <> 'active' OR NOT EXISTS (
          SELECT 1
          FROM reviewer_consent_acceptances AS acceptance
          JOIN reviewer_identity_bindings AS binding
            ON binding.id = NEW.identity_binding_id
          JOIN expert_reviewers AS reviewer ON reviewer.id = NEW.reviewer_id
          WHERE acceptance.id = NEW.consent_acceptance_id
            AND acceptance.season_id = NEW.season_id
            AND binding.season_id = NEW.season_id
            AND binding.reviewer_id = NEW.reviewer_id
            AND binding.assurance_level = 'server_verified'
            AND reviewer.active IS 1
            AND reviewer.privacy_status = 'retained'
            AND json_extract(
                  reviewer.profile_json,
                  '$.consent_acceptance_sha256'
                ) = acceptance.receipt_sha256
            AND json_extract(
                  reviewer.profile_json,
                  '$.activation_manifest_sha256'
                ) = acceptance.activation_manifest_sha256
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'participant lifecycle consent and identity are misaligned'
          );
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_withdrawal_receipts_guard_v1
        BEFORE INSERT ON reviewer_withdrawal_receipts FOR EACH ROW
        WHEN NEW.prior_judgments_preserved IS NOT 1 OR NOT EXISTS (
          SELECT 1
          FROM reviewer_participation_lifecycles AS lifecycle
          JOIN expert_reviewers AS reviewer ON reviewer.id = lifecycle.reviewer_id
          WHERE lifecycle.id = NEW.lifecycle_id
            AND lifecycle.consent_acceptance_id = NEW.consent_acceptance_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.identity_binding_id = NEW.identity_binding_id
            AND lifecycle.status = 'active'
            AND reviewer.active IS 0
            AND reviewer.revoked_at IS NOT NULL
        ) OR EXISTS (
          SELECT 1 FROM reviewer_access_credentials AS credential
          WHERE credential.reviewer_id = NEW.reviewer_id
            AND credential.status = 'active'
        ) OR EXISTS (
          SELECT 1 FROM controlled_run_reviewers AS assignment
          WHERE assignment.reviewer_id = NEW.reviewer_id
            AND assignment.active IS 1
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'participant withdrawal did not atomically stop forward authority'
          );
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_retention_schedules_guard_v1
        BEFORE INSERT ON reviewer_retention_schedules FOR EACH ROW
        WHEN date(NEW.direct_payload_delete_due_at) IS NOT (
          CASE
            WHEN CAST(strftime('%d', NEW.analysis_freeze_at) AS INTEGER) >
                 CAST(strftime(
                   '%d',
                   date(NEW.analysis_freeze_at, 'start of month', '+13 months', '-1 day')
                 ) AS INTEGER)
              THEN date(
                NEW.analysis_freeze_at,
                'start of month',
                '+13 months',
                '-1 day'
              )
            ELSE date(
              NEW.analysis_freeze_at,
              'start of month',
              '+12 months',
              '+' || (CAST(strftime('%d', NEW.analysis_freeze_at) AS INTEGER) - 1) ||
                ' days'
            )
          END
        ) OR time(NEW.direct_payload_delete_due_at) IS NOT time(NEW.analysis_freeze_at)
          OR date(NEW.pseudonymous_audit_retain_until) IS NOT (
          CASE
            WHEN CAST(strftime('%d', NEW.first_public_release_at) AS INTEGER) >
                 CAST(strftime(
                   '%d',
                   date(
                     NEW.first_public_release_at,
                     'start of month',
                     '+61 months',
                     '-1 day'
                   )
                 ) AS INTEGER)
              THEN date(
                NEW.first_public_release_at,
                'start of month',
                '+61 months',
                '-1 day'
              )
            ELSE date(
              NEW.first_public_release_at,
              'start of month',
              '+60 months',
              '+' ||
                (CAST(strftime('%d', NEW.first_public_release_at) AS INTEGER) - 1) ||
                ' days'
            )
          END
        ) OR time(NEW.pseudonymous_audit_retain_until) IS NOT
             time(NEW.first_public_release_at)
          OR NOT EXISTS (
          SELECT 1
          FROM reviewer_participation_lifecycles AS lifecycle
          JOIN reviewer_consent_acceptances AS acceptance
            ON acceptance.id = lifecycle.consent_acceptance_id
          WHERE lifecycle.id = NEW.lifecycle_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.status IN ('active', 'withdrawn')
            AND acceptance.retention_policy_sha256 = NEW.retention_policy_sha256
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'reviewer retention schedule scope or deadlines are invalid'
          );
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_deletion_receipts_guard_v1
        BEFORE INSERT ON reviewer_deletion_receipts FOR EACH ROW
        WHEN NEW.prior_judgments_preserved IS NOT 1
          OR json(NEW.redacted_fields_json) <>
             json_array(
               'expert_reviewers.profile_json',
               'expert_reviewers.qualification_json',
               'expert_reviewers.qualification_verified'
             )
          OR (
            NEW.execution_basis = 'scheduled_retention'
            AND julianday('now') < julianday(NEW.direct_payload_delete_due_at)
          )
          OR NOT EXISTS (
          SELECT 1
          FROM reviewer_participation_lifecycles AS lifecycle
          JOIN reviewer_retention_schedules AS schedule
            ON schedule.id = NEW.retention_schedule_id
          JOIN expert_reviewers AS reviewer ON reviewer.id = lifecycle.reviewer_id
          WHERE lifecycle.id = NEW.lifecycle_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.audit_marker_sha256 = NEW.audit_marker_sha256
            AND lifecycle.status = 'withdrawn'
            AND schedule.lifecycle_id = lifecycle.id
            AND schedule.direct_payload_delete_due_at IS
                NEW.direct_payload_delete_due_at
            AND schedule.pseudonymous_audit_retain_until IS
                NEW.pseudonymous_audit_retain_until
            AND reviewer.active IS 0
            AND reviewer.privacy_status = 'retained'
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'reviewer deletion receipt scope or deadline is invalid'
          );
        END;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trg_reviewer_enrollment_offers_lifecycle_v1
        BEFORE UPDATE ON reviewer_enrollment_offers FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.season_id IS NOT NEW.season_id
          OR OLD.credential_prefix IS NOT NEW.credential_prefix
          OR OLD.secret_hmac_sha256 IS NOT NEW.secret_hmac_sha256
          OR OLD.hmac_key_id IS NOT NEW.hmac_key_id
          OR OLD.consent_document_sha256 IS NOT NEW.consent_document_sha256
          OR OLD.activation_manifest_sha256 IS NOT NEW.activation_manifest_sha256
          OR OLD.not_before IS NOT NEW.not_before
          OR OLD.expires_at IS NOT NEW.expires_at
          OR OLD.created_at IS NOT NEW.created_at
          OR OLD.status <> 'active'
          OR NEW.status NOT IN ('accepted', 'revoked')
        BEGIN
          SELECT RAISE(ABORT, 'reviewer enrollment offer lifecycle is immutable');
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_enrollment_offers_no_delete_v1
        BEFORE DELETE ON reviewer_enrollment_offers FOR EACH ROW
        BEGIN
          SELECT RAISE(ABORT, 'reviewer enrollment offers cannot be deleted');
        END;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trg_reviewer_participation_lifecycles_lifecycle_v1
        BEFORE UPDATE ON reviewer_participation_lifecycles FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.consent_acceptance_id IS NOT NEW.consent_acceptance_id
          OR OLD.season_id IS NOT NEW.season_id
          OR OLD.reviewer_id IS NOT NEW.reviewer_id
          OR OLD.identity_binding_id IS NOT NEW.identity_binding_id
          OR OLD.audit_marker_sha256 IS NOT NEW.audit_marker_sha256
          OR OLD.created_at IS NOT NEW.created_at
          OR NOT (
            (OLD.status = 'active' AND NEW.status = 'withdrawn' AND EXISTS (
              SELECT 1 FROM reviewer_withdrawal_receipts AS receipt
              WHERE receipt.lifecycle_id = NEW.id
                AND receipt.receipt_sha256 = NEW.withdrawal_receipt_sha256
            ))
            OR
            (OLD.status = 'withdrawn' AND NEW.status = 'redacted' AND EXISTS (
              SELECT 1 FROM reviewer_deletion_receipts AS receipt
              WHERE receipt.lifecycle_id = NEW.id
                AND receipt.receipt_sha256 = NEW.deletion_receipt_sha256
            ))
          )
        BEGIN
          SELECT RAISE(ABORT, 'reviewer participation lifecycle is monotone and terminal');
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_participation_lifecycles_no_delete_v1
        BEFORE DELETE ON reviewer_participation_lifecycles FOR EACH ROW
        BEGIN
          SELECT RAISE(ABORT, 'reviewer participation lifecycle cannot be deleted');
        END;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trg_reviewer_identity_bindings_participant_consent_v1
        BEFORE INSERT ON reviewer_identity_bindings FOR EACH ROW
        WHEN NOT EXISTS (
          SELECT 1
          FROM expert_reviewers AS reviewer
          JOIN reviewer_consent_acceptances AS acceptance
            ON acceptance.receipt_sha256 =
               json_extract(reviewer.profile_json, '$.consent_acceptance_sha256')
          WHERE reviewer.id = NEW.reviewer_id
            AND acceptance.season_id = NEW.season_id
            AND acceptance.activation_manifest_sha256 =
                json_extract(reviewer.profile_json, '$.activation_manifest_sha256')
            AND reviewer.active IS 1
            AND reviewer.privacy_status = 'retained'
        )
        BEGIN
          SELECT RAISE(ABORT, 'reviewer identity requires participant-owned consent');
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_qualification_evidence_participant_consent_v1
        BEFORE INSERT ON reviewer_qualification_evidence FOR EACH ROW
        WHEN NOT EXISTS (
          SELECT 1
          FROM reviewer_participation_lifecycles AS lifecycle
          JOIN reviewer_consent_acceptances AS acceptance
            ON acceptance.id = lifecycle.consent_acceptance_id
          WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.status = 'active'
            AND acceptance.consent_document_sha256 = NEW.consent_document_sha256
        )
        BEGIN
          SELECT RAISE(ABORT, 'reviewer qualification requires active participant consent');
        END;
        """
    )
    for operation in ("insert", "update"):
        op.execute(
            f"""
            CREATE TRIGGER trg_reviewer_access_credentials_participation_v1_{operation}
            BEFORE {operation.upper()} ON reviewer_access_credentials FOR EACH ROW
            WHEN NEW.status = 'active' AND NOT EXISTS (
              SELECT 1 FROM reviewer_participation_lifecycles AS lifecycle
              WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
                AND lifecycle.reviewer_id = NEW.reviewer_id
                AND lifecycle.season_id = NEW.season_id
                AND lifecycle.status = 'active'
            )
            BEGIN
              SELECT RAISE(ABORT, 'reviewer credential requires active participation');
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER trg_controlled_run_reviewers_participation_v1_{operation}
            BEFORE {operation.upper()} ON controlled_run_reviewers FOR EACH ROW
            WHEN NEW.active IS 1 AND NOT EXISTS (
              SELECT 1
              FROM controlled_runs AS controlled_run
              JOIN reviewer_participation_lifecycles AS lifecycle
                ON lifecycle.season_id = controlled_run.season_id
              WHERE controlled_run.id = NEW.controlled_run_id
                AND lifecycle.reviewer_id = NEW.reviewer_id
                AND lifecycle.status = 'active'
            )
            BEGIN
              SELECT RAISE(ABORT, 'reviewer assignment requires active participation');
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER trg_votes_participation_v1_{operation}
            BEFORE {operation.upper()} ON votes FOR EACH ROW
            WHEN NEW.provenance_status = 'expert_verified_v1' AND NOT EXISTS (
              SELECT 1 FROM reviewer_participation_lifecycles AS lifecycle
              WHERE lifecycle.identity_binding_id = NEW.reviewer_identity_binding_id
                AND lifecycle.reviewer_id = NEW.reviewer_id
                AND lifecycle.status = 'active'
            )
            BEGIN
              SELECT RAISE(ABORT, 'reviewer judgment requires active participation');
            END;
            """
        )

    op.execute(
        """
        CREATE TRIGGER trg_task_validation_campaign_events_participation_v1
        BEFORE INSERT ON task_validation_campaign_events FOR EACH ROW
        WHEN EXISTS (
          SELECT 1 FROM expert_reviewers AS reviewer
          WHERE reviewer.id = NEW.reviewer_id
            AND json_type(
                  reviewer.profile_json,
                  '$.consent_acceptance_sha256'
                ) = 'text'
        ) AND NOT EXISTS (
          SELECT 1 FROM reviewer_participation_lifecycles AS lifecycle
          WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.status = 'active'
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'task-validation work requires active participation'
          );
        END;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_task_validation_audit_authorizations_participation_v1
        BEFORE INSERT ON task_validation_audit_authorizations FOR EACH ROW
        WHEN EXISTS (
          SELECT 1 FROM expert_reviewers AS reviewer
          WHERE reviewer.id = NEW.reviewer_id
            AND json_type(
                  reviewer.profile_json,
                  '$.consent_acceptance_sha256'
                ) = 'text'
        ) AND NOT EXISTS (
          SELECT 1 FROM reviewer_participation_lifecycles AS lifecycle
          WHERE lifecycle.identity_binding_id = NEW.identity_binding_id
            AND lifecycle.reviewer_id = NEW.reviewer_id
            AND lifecycle.season_id = NEW.season_id
            AND lifecycle.status = 'active'
        )
        BEGIN
          SELECT RAISE(
            ABORT,
            'task-validation work requires active participation'
          );
        END;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trg_expert_reviewers_privacy_lifecycle_v1
        BEFORE UPDATE ON expert_reviewers FOR EACH ROW
        WHEN (
          OLD.privacy_status = 'redacted' AND (
            OLD.profile_json IS NOT NEW.profile_json
            OR OLD.qualification_json IS NOT NEW.qualification_json
            OR OLD.qualification_verified IS NOT NEW.qualification_verified
            OR OLD.privacy_status IS NOT NEW.privacy_status
            OR OLD.privacy_redacted_at IS NOT NEW.privacy_redacted_at
            OR OLD.privacy_redaction_receipt_sha256 IS NOT
               NEW.privacy_redaction_receipt_sha256
            OR OLD.active IS NOT NEW.active
            OR OLD.revoked_at IS NOT NEW.revoked_at
          )
        ) OR (
          OLD.privacy_status = 'retained' AND NEW.privacy_status = 'redacted' AND (
            NEW.active IS NOT 0
            OR NEW.revoked_at IS NULL
            OR json(NEW.qualification_json) <> json('[]')
            OR NEW.qualification_verified IS NOT 0
            OR NEW.privacy_redacted_at IS NULL
            OR (SELECT count(*) FROM json_each(NEW.profile_json)) <> 4
            OR json_extract(NEW.profile_json, '$.schema_version') IS NOT
               'flavourbench-reviewer-redacted-profile-v1'
            OR json_extract(NEW.profile_json, '$.privacy_status') IS NOT 'redacted'
            OR NOT EXISTS (
              SELECT 1 FROM reviewer_deletion_receipts AS receipt
              WHERE receipt.reviewer_id = NEW.id
                AND receipt.receipt_sha256 = NEW.privacy_redaction_receipt_sha256
                AND json_extract(NEW.profile_json, '$.audit_marker_sha256') IS
                    receipt.audit_marker_sha256
                AND json_extract(
                      NEW.profile_json,
                      '$.private_payload_before_sha256'
                    ) IS receipt.private_payload_before_sha256
            )
          )
        ) OR (
          OLD.privacy_status <> NEW.privacy_status AND NOT (
            OLD.privacy_status = 'retained' AND NEW.privacy_status = 'redacted'
          )
        ) OR (
          OLD.active IS 0 AND NEW.active IS 1 AND EXISTS (
            SELECT 1 FROM reviewer_participation_lifecycles AS lifecycle
            WHERE lifecycle.reviewer_id = NEW.id
              AND lifecycle.status IN ('withdrawn', 'redacted')
          )
        )
        BEGIN
          SELECT RAISE(ABORT, 'reviewer privacy lifecycle is one-way');
        END;
        """
    )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0035: {dialect}")
    _add_reviewer_privacy_columns()
    _create_tables()
    _assert_task_validation_candidate_capacity()
    if dialect == "postgresql":
        _create_postgresql_guards()
    else:
        _create_sqlite_guards()


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across participant consent, withdrawal, and privacy evidence is prohibited"
    )
