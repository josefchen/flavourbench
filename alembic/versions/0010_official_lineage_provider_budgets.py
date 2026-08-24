"""Add fail-closed Epicure lineage and provider-scoped budget contracts.

Revision ID: 0010_lineage_budgets
Revises: 0009_commercial_run_integrity
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_lineage_budgets"
down_revision = "0009_commercial_run_integrity"
branch_labels = None
depends_on = None


def _columns(bind: sa.Connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def _indexes(bind: sa.Connection, table: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    json_object_default = (
        sa.text("'{}'::json") if bind.dialect.name == "postgresql" else sa.text("'{}'")
    )

    if "epicure_releases" not in tables:
        op.create_table(
            "epicure_releases",
            sa.Column("release_id", sa.String(length=160), primary_key=True),
            sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
            sa.Column("application_sha256", sa.String(length=64), nullable=False),
            sa.Column("public_release_uri", sa.Text(), nullable=False),
            sa.Column("release_artifact_sha256", sa.String(length=64), nullable=False),
            sa.Column("rights_clearance_sha256", sa.String(length=64), nullable=False),
            sa.Column("verification_report_sha256", sa.String(length=64), nullable=False),
            sa.Column("lineage_manifest_json", sa.JSON(), nullable=False),
            sa.Column("lineage_manifest_sha256", sa.String(length=64), nullable=False),
            sa.Column("public_release_match", sa.Boolean(), nullable=False),
            sa.Column("redistribution_rights_cleared", sa.Boolean(), nullable=False),
            sa.Column("reproducibility_verified", sa.Boolean(), nullable=False),
            sa.Column("official_eligible", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("lineage_manifest_sha256"),
        )
        op.create_index(
            "ix_epicure_releases_bundle_sha256",
            "epicure_releases",
            ["bundle_sha256"],
        )
        op.create_index(
            "ix_epicure_releases_application_sha256",
            "epicure_releases",
            ["application_sha256"],
        )
        op.create_index(
            "ix_epicure_releases_lineage_manifest_sha256",
            "epicure_releases",
            ["lineage_manifest_sha256"],
        )
        op.create_index(
            "ix_epicure_releases_official_eligible",
            "epicure_releases",
            ["official_eligible"],
        )

    if "season_provider_budgets" not in tables:
        op.create_table(
            "season_provider_budgets",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "season_id",
                sa.String(length=36),
                sa.ForeignKey("seasons.id"),
                nullable=False,
            ),
            sa.Column("execution_backend", sa.String(length=32), nullable=False),
            sa.Column("currency", sa.String(length=8), nullable=False),
            sa.Column("budget_cap_micros", sa.BigInteger(), nullable=False),
            sa.Column("budget_used_micros", sa.BigInteger(), nullable=False),
            sa.Column("budget_reserved_micros", sa.BigInteger(), nullable=False),
            sa.Column("account_scope_sha256", sa.String(length=64), nullable=False),
            sa.Column(
                "authorization_reference_sha256",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column("authorization_envelope_json", sa.JSON(), nullable=False),
            sa.Column(
                "authorization_envelope_sha256",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "execution_backend IN ('openrouter', 'bedrock', 'mock')",
                name="ck_season_provider_budgets_backend",
            ),
            sa.UniqueConstraint("season_id", "execution_backend"),
            sa.UniqueConstraint("authorization_envelope_sha256"),
        )
        op.create_index(
            "ix_season_provider_budgets_season_id",
            "season_provider_budgets",
            ["season_id"],
        )
        op.create_index(
            "ix_season_provider_budgets_execution_backend",
            "season_provider_budgets",
            ["execution_backend"],
        )

    if "provider_account_budgets" not in tables:
        op.create_table(
            "provider_account_budgets",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("execution_backend", sa.String(length=32), nullable=False),
            sa.Column("currency", sa.String(length=8), nullable=False),
            sa.Column("budget_cap_micros", sa.BigInteger(), nullable=False),
            sa.Column("budget_used_micros", sa.BigInteger(), nullable=False),
            sa.Column("budget_reserved_micros", sa.BigInteger(), nullable=False),
            sa.Column("account_scope_sha256", sa.String(length=64), nullable=False),
            sa.Column(
                "authorization_reference_sha256", sa.String(length=64), nullable=False
            ),
            sa.Column("authorization_envelope_json", sa.JSON(), nullable=False),
            sa.Column(
                "authorization_envelope_sha256", sa.String(length=64), nullable=False
            ),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "execution_backend IN ('openrouter', 'bedrock')",
                name="ck_provider_account_budgets_backend",
            ),
            sa.UniqueConstraint("execution_backend", "account_scope_sha256"),
            sa.UniqueConstraint("authorization_envelope_sha256"),
        )
        op.create_index(
            "ix_provider_account_budgets_execution_backend",
            "provider_account_budgets",
            ["execution_backend"],
        )
        op.create_index(
            "ix_provider_account_budgets_account_scope_sha256",
            "provider_account_budgets",
            ["account_scope_sha256"],
        )

    if "season_models" in tables:
        columns = _columns(bind, "season_models")
        if "execution_backend" not in columns:
            op.add_column(
                "season_models",
                sa.Column(
                    "execution_backend",
                    sa.String(length=32),
                    nullable=False,
                    server_default="openrouter",
                ),
            )
        if "backend_contract_json" not in columns:
            op.add_column(
                "season_models",
                sa.Column(
                    "backend_contract_json",
                    sa.JSON(),
                    nullable=False,
                    server_default=json_object_default,
                ),
            )
        if "backend_contract_sha256" not in columns:
            op.add_column(
                "season_models",
                sa.Column(
                    "backend_contract_sha256",
                    sa.String(length=64),
                    nullable=False,
                    server_default="unfrozen",
                ),
            )
        if "ix_season_models_execution_backend" not in _indexes(bind, "season_models"):
            op.create_index(
                "ix_season_models_execution_backend",
                "season_models",
                ["execution_backend"],
            )

    if "battles" in tables:
        columns = _columns(bind, "battles")
        if "provider_reservations_json" not in columns:
            op.add_column(
                "battles",
                sa.Column(
                    "provider_reservations_json",
                    sa.JSON(),
                    nullable=False,
                    server_default=json_object_default,
                ),
            )

    if "response_arms" in tables:
        columns = _columns(bind, "response_arms")
        if "execution_backend" not in columns:
            op.add_column(
                "response_arms",
                sa.Column(
                    "execution_backend",
                    sa.String(length=32),
                    nullable=False,
                    server_default="openrouter",
                ),
            )
        if "cost_accounting_basis" not in columns:
            op.add_column(
                "response_arms",
                sa.Column(
                    "cost_accounting_basis",
                    sa.String(length=80),
                    nullable=False,
                    server_default="unrecorded",
                ),
            )
        if "billing_reconciliation_status" not in columns:
            op.add_column(
                "response_arms",
                sa.Column(
                    "billing_reconciliation_status",
                    sa.String(length=80),
                    nullable=False,
                    server_default="unrecorded",
                ),
            )
        if "backend_response_schema_sha256" not in columns:
            op.add_column(
                "response_arms",
                sa.Column(
                    "backend_response_schema_sha256",
                    sa.String(length=64),
                    nullable=False,
                    server_default="unresolved",
                ),
            )
        if "backend_tool_schema_sha256" not in columns:
            op.add_column(
                "response_arms",
                sa.Column(
                    "backend_tool_schema_sha256",
                    sa.String(length=64),
                    nullable=False,
                    server_default="unresolved",
                ),
            )
        if "ix_response_arms_execution_backend" not in _indexes(bind, "response_arms"):
            op.create_index(
                "ix_response_arms_execution_backend",
                "response_arms",
                ["execution_backend"],
            )
        if (
            "ix_response_arms_billing_reconciliation_status"
            not in _indexes(bind, "response_arms")
        ):
            op.create_index(
                "ix_response_arms_billing_reconciliation_status",
                "response_arms",
                ["billing_reconciliation_status"],
            )

    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_epicure_release_append_only()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'Epicure release records are append-only';
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_epicure_release_append_only ON epicure_releases;
            CREATE TRIGGER trg_epicure_release_append_only
            BEFORE UPDATE OR DELETE ON epicure_releases
            FOR EACH ROW EXECUTE FUNCTION flavourbench_epicure_release_append_only();

            CREATE OR REPLACE FUNCTION flavourbench_provider_budget_contract_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                   OLD.currency IS DISTINCT FROM NEW.currency OR
                   OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros OR
                   OLD.account_scope_sha256 IS DISTINCT FROM NEW.account_scope_sha256 OR
                   OLD.authorization_reference_sha256 IS DISTINCT FROM
                       NEW.authorization_reference_sha256 OR
                   OLD.authorization_envelope_json IS DISTINCT FROM
                       NEW.authorization_envelope_json OR
                   OLD.authorization_envelope_sha256 IS DISTINCT FROM
                       NEW.authorization_envelope_sha256 OR
                   OLD.valid_until IS DISTINCT FROM NEW.valid_until THEN
                    RAISE EXCEPTION 'provider budget authorization is immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_provider_budget_contract_immutable
                ON season_provider_budgets;
            CREATE TRIGGER trg_provider_budget_contract_immutable
            BEFORE UPDATE ON season_provider_budgets
            FOR EACH ROW EXECUTE FUNCTION flavourbench_provider_budget_contract_immutable();

            CREATE OR REPLACE FUNCTION flavourbench_account_budget_contract_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                   OLD.currency IS DISTINCT FROM NEW.currency OR
                   OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros OR
                   OLD.account_scope_sha256 IS DISTINCT FROM NEW.account_scope_sha256 OR
                   OLD.authorization_reference_sha256 IS DISTINCT FROM
                       NEW.authorization_reference_sha256 OR
                   OLD.authorization_envelope_json IS DISTINCT FROM
                       NEW.authorization_envelope_json OR
                   OLD.authorization_envelope_sha256 IS DISTINCT FROM
                       NEW.authorization_envelope_sha256 OR
                   OLD.valid_until IS DISTINCT FROM NEW.valid_until THEN
                    RAISE EXCEPTION 'provider account authorization is immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_account_budget_contract_immutable
                ON provider_account_budgets;
            CREATE TRIGGER trg_account_budget_contract_immutable
            BEFORE UPDATE ON provider_account_budgets
            FOR EACH ROW EXECUTE FUNCTION flavourbench_account_budget_contract_immutable();

            CREATE OR REPLACE FUNCTION flavourbench_0010_execution_contract_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF TG_TABLE_NAME = 'season_models' AND
                   OLD.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved') AND
                   (OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                    OLD.backend_contract_json IS DISTINCT FROM NEW.backend_contract_json OR
                    OLD.backend_contract_sha256 IS DISTINCT FROM
                        NEW.backend_contract_sha256) THEN
                    RAISE EXCEPTION 'frozen season execution backend is immutable';
                ELSIF TG_TABLE_NAME = 'battles' AND
                   OLD.provider_reservations_json IS DISTINCT FROM
                       NEW.provider_reservations_json THEN
                    RAISE EXCEPTION 'battle provider reservation contract is immutable';
                ELSIF TG_TABLE_NAME = 'response_arms' AND
                   OLD.execution_backend IS DISTINCT FROM NEW.execution_backend THEN
                    RAISE EXCEPTION 'response-arm execution backend is immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_season_model_backend_immutable ON season_models;
            CREATE TRIGGER trg_season_model_backend_immutable
            BEFORE UPDATE ON season_models FOR EACH ROW
            EXECUTE FUNCTION flavourbench_0010_execution_contract_immutable();
            DROP TRIGGER IF EXISTS trg_battle_provider_reservation_immutable ON battles;
            CREATE TRIGGER trg_battle_provider_reservation_immutable
            BEFORE UPDATE ON battles FOR EACH ROW
            EXECUTE FUNCTION flavourbench_0010_execution_contract_immutable();
            DROP TRIGGER IF EXISTS trg_response_arm_backend_immutable ON response_arms;
            CREATE TRIGGER trg_response_arm_backend_immutable
            BEFORE UPDATE ON response_arms FOR EACH ROW
            EXECUTE FUNCTION flavourbench_0010_execution_contract_immutable();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            DROP TRIGGER IF EXISTS trg_response_arm_backend_immutable ON response_arms;
            DROP TRIGGER IF EXISTS trg_battle_provider_reservation_immutable ON battles;
            DROP TRIGGER IF EXISTS trg_season_model_backend_immutable ON season_models;
            DROP FUNCTION IF EXISTS flavourbench_0010_execution_contract_immutable();
            DROP TRIGGER IF EXISTS trg_provider_budget_contract_immutable
                ON season_provider_budgets;
            DROP FUNCTION IF EXISTS flavourbench_provider_budget_contract_immutable();
            DROP TRIGGER IF EXISTS trg_account_budget_contract_immutable
                ON provider_account_budgets;
            DROP FUNCTION IF EXISTS flavourbench_account_budget_contract_immutable();
            DROP TRIGGER IF EXISTS trg_epicure_release_append_only ON epicure_releases;
            DROP FUNCTION IF EXISTS flavourbench_epicure_release_append_only();
            """
        )
    if "response_arms" in tables:
        indexes = _indexes(bind, "response_arms")
        if "ix_response_arms_billing_reconciliation_status" in indexes:
            op.drop_index(
                "ix_response_arms_billing_reconciliation_status",
                table_name="response_arms",
            )
        if "ix_response_arms_execution_backend" in indexes:
            op.drop_index("ix_response_arms_execution_backend", table_name="response_arms")
        if "execution_backend" in _columns(bind, "response_arms"):
            op.drop_column("response_arms", "execution_backend")
        for name in (
            "backend_tool_schema_sha256",
            "backend_response_schema_sha256",
            "billing_reconciliation_status",
            "cost_accounting_basis",
        ):
            if name in _columns(bind, "response_arms"):
                op.drop_column("response_arms", name)
    if "battles" in tables and "provider_reservations_json" in _columns(bind, "battles"):
        op.drop_column("battles", "provider_reservations_json")
    if "season_models" in tables:
        indexes = _indexes(bind, "season_models")
        if "ix_season_models_execution_backend" in indexes:
            op.drop_index("ix_season_models_execution_backend", table_name="season_models")
        if "execution_backend" in _columns(bind, "season_models"):
            op.drop_column("season_models", "execution_backend")
        for name in ("backend_contract_sha256", "backend_contract_json"):
            if name in _columns(bind, "season_models"):
                op.drop_column("season_models", name)
    if "season_provider_budgets" in tables:
        op.drop_table("season_provider_budgets")
    if "provider_account_budgets" in tables:
        op.drop_table("provider_account_budgets")
    if "epicure_releases" in tables:
        op.drop_table("epicure_releases")
