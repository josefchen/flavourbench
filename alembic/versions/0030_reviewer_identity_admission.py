"""Add season-scoped reviewer identity, admission, and vote provenance.

Revision ID: 0030_reviewer_identity_admission
Revises: 0029_qwencloud_direct_backend
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0030_reviewer_identity_admission"
down_revision = "0029_qwencloud_direct_backend"
branch_labels = None
depends_on = None


def _restore_sqlite_vote_guards() -> None:
    """Restore vote triggers dropped by SQLite batch table recreation.

    Alembic implements ``batch_alter_table`` on SQLite by copying and replacing
    the table. SQLite does not carry triggers across that replacement, so every
    pre-existing vote boundary has to be recreated explicitly after the
    upgrade copy operation. Downgrade is prohibited before any DDL runs.
    """

    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return

    for trigger in (
        "trg_vote_evidence_guard_insert",
        "trg_vote_evidence_guard_update",
        "trg_vote_evidence_guard_delete",
        "trg_vote_normal_finish_guard_insert",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    op.execute(
        """
        CREATE TRIGGER trg_vote_evidence_guard_insert
        BEFORE INSERT ON votes FOR EACH ROW
        WHEN NEW.choice NOT IN ('left', 'right', 'tie', 'both_bad')
          OR NEW.cohort NOT IN (
              'public',
              'expert_independent',
              'expert_product_affiliated',
              'expert_provider_affiliated'
          )
          OR NOT EXISTS (
              SELECT 1 FROM battles AS b
              JOIN response_arms AS left_arm
                ON left_arm.id = b.left_arm_id
               AND left_arm.battle_id = b.id
               AND left_arm.side = 'left'
               AND left_arm.status = 'complete'
              JOIN response_arms AS right_arm
                ON right_arm.id = b.right_arm_id
               AND right_arm.battle_id = b.id
               AND right_arm.side = 'right'
               AND right_arm.status = 'complete'
              WHERE b.id = NEW.battle_id
                AND b.status = 'complete'
                AND b.completed_at IS NOT NULL
                AND NEW.created_at >= b.completed_at
                AND b.left_arm_id IS NOT NULL
                AND b.right_arm_id IS NOT NULL
                AND b.left_arm_id <> b.right_arm_id
          )
        BEGIN
            SELECT RAISE(ABORT, 'vote does not follow a valid completed battle');
        END;
        """
    )
    for operation in ("update", "delete"):
        op.execute(
            f"""
            CREATE TRIGGER trg_vote_evidence_guard_{operation}
            BEFORE {operation.upper()} ON votes FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'votes are append-only');
            END;
            """
        )
    op.execute(
        """
        CREATE TRIGGER trg_vote_normal_finish_guard_insert
        BEFORE INSERT ON votes FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1
            FROM battles AS battle
            JOIN response_arms AS left_arm
              ON left_arm.id = battle.left_arm_id
             AND left_arm.battle_id = battle.id
             AND left_arm.side = 'left'
             AND left_arm.status = 'complete'
             AND lower(trim(COALESCE(left_arm.finish_reason, '')))
                 IN ('completed', 'end_turn', 'stop', 'stop_sequence')
            JOIN response_arms AS right_arm
              ON right_arm.id = battle.right_arm_id
             AND right_arm.battle_id = battle.id
             AND right_arm.side = 'right'
             AND right_arm.status = 'complete'
             AND lower(trim(COALESCE(right_arm.finish_reason, '')))
                 IN ('completed', 'end_turn', 'stop', 'stop_sequence')
            WHERE battle.id = NEW.battle_id
              AND battle.status = 'complete'
        )
        BEGIN
            SELECT RAISE(ABORT, 'vote requires two normally finished response arms');
        END;
        """
    )


def _create_reviewer_tables() -> None:
    op.create_table(
        "reviewer_identity_bindings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("person_commitment_sha256", sa.String(length=64), nullable=False),
        sa.Column("identity_issuer_sha256", sa.String(length=64), nullable=False),
        sa.Column("identity_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("hmac_key_id", sa.String(length=64), nullable=False),
        sa.Column("verification_method", sa.String(length=64), nullable=False),
        sa.Column("assurance_level", sa.String(length=32), nullable=False),
        sa.Column("roles_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "assurance_level IN ('server_verified', 'legacy_unverified')",
            name="ck_reviewer_identity_bindings_assurance",
        ),
        sa.CheckConstraint(
            "verification_method IN ('season_hmac_issuer_subject_v1', 'legacy_unverified')",
            name="ck_reviewer_identity_bindings_method",
        ),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "season_id",
            "person_commitment_sha256",
            name="uq_reviewer_identity_bindings_season_person",
        ),
        sa.UniqueConstraint(
            "season_id",
            "reviewer_id",
            name="uq_reviewer_identity_bindings_season_reviewer",
        ),
    )
    op.create_index(
        "ix_reviewer_identity_bindings_person_commitment_sha256",
        "reviewer_identity_bindings",
        ["person_commitment_sha256"],
    )
    op.create_index(
        "ix_reviewer_identity_bindings_reviewer_id",
        "reviewer_identity_bindings",
        ["reviewer_id"],
    )
    op.create_index(
        "ix_reviewer_identity_bindings_season_id",
        "reviewer_identity_bindings",
        ["season_id"],
    )

    op.create_table(
        "reviewer_access_credentials",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("credential_prefix", sa.String(length=32), nullable=False),
        sa.Column("secret_hmac_sha256", sa.String(length=64), nullable=False),
        sa.Column("hmac_key_id", sa.String(length=64), nullable=False),
        sa.Column("credential_kind", sa.String(length=32), nullable=False),
        sa.Column("scopes_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("maximum_uses", sa.Integer(), nullable=False),
        sa.Column("use_count", sa.Integer(), nullable=False),
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "credential_kind IN ('enrollment_once', 'review_session')",
            name="ck_reviewer_access_credentials_kind",
        ),
        sa.CheckConstraint(
            "(credential_kind = 'enrollment_once' AND maximum_uses = 1) OR "
            "credential_kind = 'review_session'",
            name="ck_reviewer_access_credentials_one_time_enrollment",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'consumed', 'revoked')",
            name="ck_reviewer_access_credentials_status",
        ),
        sa.CheckConstraint(
            "maximum_uses >= 1 AND maximum_uses <= 256 AND use_count >= 0 "
            "AND use_count <= maximum_uses",
            name="ck_reviewer_access_credentials_usage",
        ),
        sa.CheckConstraint(
            "expires_at > not_before",
            name="ck_reviewer_access_credentials_window",
        ),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "credential_prefix", name="uq_reviewer_access_credentials_prefix"
        ),
        sa.UniqueConstraint(
            "secret_hmac_sha256", name="uq_reviewer_access_credentials_secret"
        ),
    )
    for column in (
        "credential_prefix",
        "expires_at",
        "identity_binding_id",
        "reviewer_id",
        "season_id",
        "status",
    ):
        op.create_index(
            f"ix_reviewer_access_credentials_{column}",
            "reviewer_access_credentials",
            [column],
        )

    op.create_table(
        "reviewer_qualification_evidence",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=False),
        sa.Column("affiliation_class", sa.String(length=40), nullable=False),
        sa.Column("independence_verified", sa.Boolean(), nullable=False),
        sa.Column("conflict_cleared", sa.Boolean(), nullable=False),
        sa.Column("verification_status", sa.String(length=24), nullable=False),
        sa.Column("qualification_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("independence_evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("conflict_disclosure_sha256", sa.String(length=64), nullable=False),
        sa.Column("consent_document_sha256", sa.String(length=64), nullable=False),
        sa.Column("training_material_sha256", sa.String(length=64), nullable=False),
        sa.Column("verifier_principal_sha256", sa.String(length=64), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "affiliation_class IN ('independent_external', 'product_affiliated', "
            "'provider_affiliated')",
            name="ck_reviewer_qualification_evidence_affiliation",
        ),
        sa.CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_qualification_evidence_family",
        ),
        sa.CheckConstraint(
            "valid_until IS NULL OR valid_until > verified_at",
            name="ck_reviewer_qualification_evidence_window",
        ),
        sa.CheckConstraint(
            "verification_status = 'verified'",
            name="ck_reviewer_qualification_evidence_verified",
        ),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "identity_binding_id",
            "family",
            "qualification_evidence_sha256",
            name="uq_reviewer_qualification_evidence_binding_family_evidence",
        ),
    )
    for column in ("family", "identity_binding_id", "reviewer_id", "season_id", "valid_until"):
        op.create_index(
            f"ix_reviewer_qualification_evidence_{column}",
            "reviewer_qualification_evidence",
            [column],
        )

    op.create_table(
        "reviewer_calibration_sets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=False),
        sa.Column("calibration_set_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("scoring_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("real_source_arms", sa.Integer(), nullable=False),
        sa.Column("synthetic_arms", sa.Integer(), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_calibration_sets_family",
        ),
        sa.CheckConstraint(
            "item_count >= 1 AND real_source_arms >= item_count * 2 AND synthetic_arms = 0",
            name="ck_reviewer_calibration_sets_real_outputs",
        ),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "season_id",
            "family",
            "calibration_set_sha256",
            name="uq_reviewer_calibration_sets_season_family_hash",
        ),
    )
    for column in ("calibration_set_sha256", "family", "season_id"):
        op.create_index(
            f"ix_reviewer_calibration_sets_{column}",
            "reviewer_calibration_sets",
            [column],
        )

    op.create_table(
        "reviewer_calibration_ballots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("calibration_set_id", sa.String(length=36), nullable=False),
        sa.Column("ballot_sha256", sa.String(length=64), nullable=False),
        sa.Column("scoring_result_sha256", sa.String(length=64), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("correct_count", sa.Integer(), nullable=False),
        sa.Column("accuracy_milli", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "item_count >= 1 AND correct_count >= 0 AND correct_count <= item_count "
            "AND accuracy_milli >= 0 AND accuracy_milli <= 1000",
            name="ck_reviewer_calibration_ballots_score",
        ),
        sa.ForeignKeyConstraint(["calibration_set_id"], ["reviewer_calibration_sets.id"]),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ballot_sha256"),
        sa.UniqueConstraint(
            "calibration_set_id",
            "identity_binding_id",
            name="uq_reviewer_calibration_ballots_set_binding",
        ),
    )
    for column in ("calibration_set_id", "identity_binding_id", "reviewer_id", "season_id"):
        op.create_index(
            f"ix_reviewer_calibration_ballots_{column}",
            "reviewer_calibration_ballots",
            [column],
        )

    op.create_table(
        "reviewer_family_admissions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("season_id", sa.String(length=36), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("identity_binding_id", sa.String(length=36), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=False),
        sa.Column("review_role", sa.String(length=32), nullable=False),
        sa.Column("cohort", sa.String(length=48), nullable=False),
        sa.Column("qualification_evidence_id", sa.String(length=36), nullable=False),
        sa.Column("calibration_ballot_id", sa.String(length=36), nullable=True),
        sa.Column("admission_policy_json", sa.JSON(), nullable=False),
        sa.Column("admission_policy_sha256", sa.String(length=64), nullable=False),
        sa.Column("evidence_bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("decision_reference_sha256", sa.String(length=64), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "cohort IN ('expert_independent', 'expert_product_affiliated', "
            "'expert_provider_affiliated')",
            name="ck_reviewer_family_admissions_cohort",
        ),
        sa.CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_family_admissions_family",
        ),
        sa.CheckConstraint(
            "review_role IN ('task_author', 'task_validator', 'task_adjudicator', "
            "'output_rater')",
            name="ck_reviewer_family_admissions_role",
        ),
        sa.CheckConstraint(
            "valid_until > valid_from", name="ck_reviewer_family_admissions_window"
        ),
        sa.ForeignKeyConstraint(["calibration_ballot_id"], ["reviewer_calibration_ballots.id"]),
        sa.ForeignKeyConstraint(["identity_binding_id"], ["reviewer_identity_bindings.id"]),
        sa.ForeignKeyConstraint(
            ["qualification_evidence_id"], ["reviewer_qualification_evidence.id"]
        ),
        sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evidence_bundle_sha256"),
        sa.UniqueConstraint(
            "identity_binding_id",
            "family",
            "review_role",
            "admission_policy_sha256",
            name="uq_reviewer_family_admissions_binding_family_role_policy",
        ),
    )
    for column in (
        "calibration_ballot_id",
        "cohort",
        "family",
        "identity_binding_id",
        "qualification_evidence_id",
        "review_role",
        "reviewer_id",
        "season_id",
        "valid_until",
    ):
        op.create_index(
            f"ix_reviewer_family_admissions_{column}",
            "reviewer_family_admissions",
            [column],
        )


def _add_vote_provenance() -> None:
    with op.batch_alter_table("votes") as batch:
        batch.add_column(sa.Column("reviewer_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column("reviewer_identity_binding_id", sa.String(length=36), nullable=True)
        )
        batch.add_column(
            sa.Column("reviewer_family_admission_id", sa.String(length=36), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "provenance_status",
                sa.String(length=32),
                nullable=False,
                server_default="legacy_unverified",
            )
        )
        batch.add_column(sa.Column("provenance_sha256", sa.String(length=64), nullable=True))
        batch.create_foreign_key(
            "fk_votes_reviewer_id_expert_reviewers",
            "expert_reviewers",
            ["reviewer_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_votes_reviewer_identity_binding_id",
            "reviewer_identity_bindings",
            ["reviewer_identity_binding_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_votes_reviewer_family_admission_id",
            "reviewer_family_admissions",
            ["reviewer_family_admission_id"],
            ["id"],
        )
        batch.create_check_constraint(
            "ck_votes_provenance_status",
            "provenance_status IN ('legacy_unverified', 'public_pseudonymous', "
            "'expert_verified_v1')",
        )
        batch.create_check_constraint(
            "ck_votes_reviewer_provenance_shape",
            "(provenance_status = 'expert_verified_v1' AND reviewer_id IS NOT NULL "
            "AND reviewer_identity_binding_id IS NOT NULL "
            "AND reviewer_family_admission_id IS NOT NULL "
            "AND provenance_sha256 IS NOT NULL) OR "
            "(provenance_status <> 'expert_verified_v1' AND reviewer_id IS NULL "
            "AND reviewer_identity_binding_id IS NULL "
            "AND reviewer_family_admission_id IS NULL)",
        )
    for column in (
        "provenance_sha256",
        "provenance_status",
        "reviewer_family_admission_id",
        "reviewer_id",
        "reviewer_identity_binding_id",
    ):
        op.create_index(f"ix_votes_{column}", "votes", [column])
    op.create_index(
        "uq_votes_verified_person_battle",
        "votes",
        ["battle_id", "reviewer_identity_binding_id"],
        unique=True,
        postgresql_where=sa.text(
            "provenance_status = 'expert_verified_v1' "
            "AND reviewer_identity_binding_id IS NOT NULL"
        ),
        sqlite_where=sa.text(
            "provenance_status = 'expert_verified_v1' "
            "AND reviewer_identity_binding_id IS NOT NULL"
        ),
    )


def _create_postgresql_guards() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_reviewer_evidence_append_only_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'reviewer identity and admission evidence is append-only';
        END;
        $$
        """
    )
    for table in (
        "reviewer_identity_bindings",
        "reviewer_qualification_evidence",
        "reviewer_calibration_sets",
        "reviewer_calibration_ballots",
        "reviewer_family_admissions",
    ):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_append_only_v1
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION flavourbench_reviewer_evidence_append_only_v1()
            """
        )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_reviewer_credential_lifecycle_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_OP = 'DELETE' THEN
            RAISE EXCEPTION 'reviewer access credentials cannot be deleted';
          END IF;
          IF (NEW.id, NEW.season_id, NEW.reviewer_id, NEW.identity_binding_id,
              NEW.credential_prefix, NEW.secret_hmac_sha256, NEW.hmac_key_id,
              NEW.credential_kind, NEW.scopes_json::text, NEW.maximum_uses,
              NEW.not_before, NEW.expires_at, NEW.created_at)
             IS DISTINCT FROM
             (OLD.id, OLD.season_id, OLD.reviewer_id, OLD.identity_binding_id,
              OLD.credential_prefix, OLD.secret_hmac_sha256, OLD.hmac_key_id,
              OLD.credential_kind, OLD.scopes_json::text, OLD.maximum_uses,
              OLD.not_before, OLD.expires_at, OLD.created_at) THEN
            RAISE EXCEPTION 'reviewer credential contract is immutable';
          END IF;
          IF OLD.status IN ('consumed', 'revoked') THEN
            RAISE EXCEPTION 'terminal reviewer credential is immutable';
          END IF;
          IF NEW.use_count < OLD.use_count OR NEW.use_count > OLD.use_count + 1 THEN
            RAISE EXCEPTION 'reviewer credential use count is not monotone';
          END IF;
          IF NEW.status = 'active' AND
             (NEW.use_count >= NEW.maximum_uses OR NEW.consumed_at IS NOT NULL OR
              NEW.revoked_at IS NOT NULL) THEN
            RAISE EXCEPTION 'active reviewer credential has terminal state';
          ELSIF NEW.status = 'consumed' AND
                (NEW.use_count <> NEW.maximum_uses OR NEW.consumed_at IS NULL OR
                 NEW.revoked_at IS NOT NULL) THEN
            RAISE EXCEPTION 'consumed reviewer credential is incomplete';
          ELSIF NEW.status = 'revoked' AND
                (NEW.revoked_at IS NULL OR NEW.consumed_at IS NOT NULL) THEN
            RAISE EXCEPTION 'revoked reviewer credential is incomplete';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_access_credentials_lifecycle_v1
        BEFORE UPDATE OR DELETE ON reviewer_access_credentials
        FOR EACH ROW EXECUTE FUNCTION flavourbench_reviewer_credential_lifecycle_v1()
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_reviewer_family_admission_guard_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE aligned_count integer;
        DECLARE calibration_required boolean;
        DECLARE minimum_accuracy integer;
        BEGIN
          IF NEW.admission_policy_json->>'schema_version' <>
             'flavourbench-reviewer-admission-policy-v1' THEN
            RAISE EXCEPTION 'reviewer admission policy schema is invalid';
          END IF;
          calibration_required :=
            (NEW.admission_policy_json->>'requires_calibration')::boolean;
          minimum_accuracy :=
            (NEW.admission_policy_json->>'minimum_accuracy_milli')::integer;
          IF minimum_accuracy < 0 OR minimum_accuracy > 1000 THEN
            RAISE EXCEPTION 'reviewer admission accuracy threshold is invalid';
          END IF;
          SELECT count(*) INTO aligned_count
          FROM reviewer_identity_bindings AS binding
          JOIN reviewer_qualification_evidence AS qualification
            ON qualification.id = NEW.qualification_evidence_id
          JOIN expert_reviewers AS reviewer ON reviewer.id = NEW.reviewer_id
          WHERE binding.id = NEW.identity_binding_id
            AND binding.season_id = NEW.season_id
            AND binding.reviewer_id = NEW.reviewer_id
            AND binding.assurance_level = 'server_verified'
            AND binding.roles_json::jsonb ? NEW.review_role
            AND qualification.season_id = NEW.season_id
            AND qualification.reviewer_id = NEW.reviewer_id
            AND qualification.identity_binding_id = binding.id
            AND qualification.family = NEW.family
            AND qualification.verification_status = 'verified'
            AND NEW.valid_from >= qualification.verified_at
            AND (qualification.valid_until IS NULL OR
                 NEW.valid_until <= qualification.valid_until)
            AND ((qualification.affiliation_class = 'independent_external' AND
                  NEW.cohort = 'expert_independent' AND
                  qualification.independence_verified IS TRUE AND
                  qualification.conflict_cleared IS TRUE) OR
                 (qualification.affiliation_class = 'product_affiliated' AND
                  NEW.cohort = 'expert_product_affiliated') OR
                 (qualification.affiliation_class = 'provider_affiliated' AND
                  NEW.cohort = 'expert_provider_affiliated'))
            AND reviewer.active IS TRUE;
          IF aligned_count <> 1 THEN
            RAISE EXCEPTION 'reviewer admission identity or qualification is misaligned';
          END IF;
          IF calibration_required AND NEW.calibration_ballot_id IS NULL THEN
            RAISE EXCEPTION 'reviewer admission requires calibration';
          END IF;
          IF NEW.calibration_ballot_id IS NOT NULL THEN
            SELECT count(*) INTO aligned_count
            FROM reviewer_calibration_ballots AS ballot
            JOIN reviewer_calibration_sets AS calibration_set
              ON calibration_set.id = ballot.calibration_set_id
            WHERE ballot.id = NEW.calibration_ballot_id
              AND ballot.season_id = NEW.season_id
              AND ballot.reviewer_id = NEW.reviewer_id
              AND ballot.identity_binding_id = NEW.identity_binding_id
              AND ballot.passed IS TRUE
              AND ballot.accuracy_milli >= minimum_accuracy
              AND ballot.completed_at <= NEW.valid_from
              AND calibration_set.season_id = NEW.season_id
              AND calibration_set.family = NEW.family
              AND calibration_set.synthetic_arms = 0;
            IF aligned_count <> 1 THEN
              RAISE EXCEPTION 'reviewer admission calibration is misaligned';
            END IF;
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reviewer_family_admissions_guard_v1
        BEFORE INSERT ON reviewer_family_admissions
        FOR EACH ROW EXECUTE FUNCTION flavourbench_reviewer_family_admission_guard_v1()
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION flavourbench_verified_expert_vote_guard_v1()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE aligned_count integer;
        BEGIN
          IF NEW.provenance_status = 'expert_verified_v1' THEN
            IF NEW.cohort = 'public' THEN
              RAISE EXCEPTION 'verified reviewer provenance is restricted to expert cohorts';
            END IF;
            SELECT count(*) INTO aligned_count
            FROM battles AS battle
            JOIN reviewer_identity_bindings AS binding
              ON binding.id = NEW.reviewer_identity_binding_id
            JOIN reviewer_family_admissions AS admission
              ON admission.id = NEW.reviewer_family_admission_id
            JOIN expert_reviewers AS reviewer
              ON reviewer.id = NEW.reviewer_id
            WHERE battle.id = NEW.battle_id
              AND binding.season_id = battle.season_id
              AND binding.reviewer_id = NEW.reviewer_id
              AND binding.assurance_level = 'server_verified'
              AND admission.season_id = battle.season_id
              AND admission.reviewer_id = NEW.reviewer_id
              AND admission.identity_binding_id = binding.id
              AND admission.family = battle.category
              AND admission.review_role = 'output_rater'
              AND admission.cohort = NEW.cohort
              AND COALESCE(NEW.created_at, now()) BETWEEN admission.valid_from
                                                      AND admission.valid_until
              AND reviewer.active IS TRUE;
            IF aligned_count <> 1 THEN
              RAISE EXCEPTION 'verified expert vote is not backed by one active family admission';
            END IF;
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_votes_verified_expert_provenance_v1
        BEFORE INSERT OR UPDATE ON votes
        FOR EACH ROW EXECUTE FUNCTION flavourbench_verified_expert_vote_guard_v1()
        """
    )


def upgrade() -> None:
    _create_reviewer_tables()
    _add_vote_provenance()
    _restore_sqlite_vote_guards()
    _create_postgresql_guards()


def downgrade() -> None:
    # Reversing this revision would delete the identity and admission evidence
    # that gives expert votes their provenance.  It would also let a requested
    # multi-revision downgrade mutate this head before an older irreversible
    # authority revision rejects the operation.  Fail before any DDL so the
    # database remains at the content-addressed head.
    raise RuntimeError(
        "downgrade across research-release archive policy is prohibited; "
        "downgrade across reviewer identity and admission authority is prohibited"
    )
