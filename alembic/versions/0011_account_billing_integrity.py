"""Add non-resettable account authority and immutable AWS billing evidence.

Revision ID: 0011_account_billing
Revises: 0010_lineage_budgets
"""

from __future__ import annotations

import hashlib
import json

import sqlalchemy as sa

from alembic import op

revision = "0011_account_billing"
down_revision = "0010_lineage_budgets"
branch_labels = None
depends_on = None


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tables(bind: sa.Connection) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _columns(bind: sa.Connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def _indexes(bind: sa.Connection, table: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    json_default = (
        sa.text("'{}'::json")
        if bind.dialect.name == "postgresql"
        else sa.text("'{}'")
    )

    with op.batch_alter_table("season_provider_budgets") as batch:
        batch.add_column(
            sa.Column(
                "account_authorization_envelope_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unresolved",
            )
        )

    # Every 0010 authorization is retained as evidence. The prior schema did not
    # carry independently verifiable opening-balance or credential-binding
    # receipts, so all legacy ledgers fail closed in pending_verification until
    # an append-only authorization epoch activates the preserved ledger.
    with op.batch_alter_table("provider_account_budgets") as batch:
        batch.add_column(
            sa.Column(
                "status",
                sa.String(length=24),
                nullable=False,
                server_default="pending_verification",
            )
        )
        batch.add_column(
            sa.Column(
                "opening_used_micros",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "opening_reserved_micros",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "opening_balance_json",
                sa.JSON(),
                nullable=False,
                server_default=json_default,
            )
        )
        batch.add_column(
            sa.Column(
                "opening_balance_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unresolved",
            )
        )
        batch.add_column(
            sa.Column(
                "credential_binding_json",
                sa.JSON(),
                nullable=False,
                server_default=json_default,
            )
        )
        batch.add_column(
            sa.Column(
                "credential_binding_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unresolved",
            )
        )
        batch.add_column(
            sa.Column(
                "authorization_hmac_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unresolved",
            )
        )
        batch.add_column(
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.create_check_constraint(
            "ck_provider_account_budgets_status",
            "status IN ('pending_verification', 'active', 'revoked')",
        )
        batch.create_check_constraint(
            "ck_provider_account_budgets_nonnegative",
            "budget_cap_micros > 0 AND budget_used_micros >= 0 AND "
            "budget_reserved_micros >= 0 AND opening_used_micros >= 0 AND "
            "opening_reserved_micros >= 0",
        )
        batch.create_check_constraint(
            "ck_provider_account_budgets_revocation",
            "(status = 'revoked' AND revoked_at IS NOT NULL) OR "
            "(status != 'revoked' AND revoked_at IS NULL)",
        )

    legacy_table = sa.table(
        "provider_account_budgets",
        sa.column("id", sa.String()),
        sa.column("execution_backend", sa.String()),
        sa.column("account_scope_sha256", sa.String()),
        sa.column("budget_used_micros", sa.BigInteger()),
        sa.column("budget_reserved_micros", sa.BigInteger()),
        sa.column("opening_used_micros", sa.BigInteger()),
        sa.column("opening_reserved_micros", sa.BigInteger()),
        sa.column("opening_balance_json", sa.JSON()),
        sa.column("opening_balance_sha256", sa.String()),
        sa.column("credential_binding_json", sa.JSON()),
        sa.column("credential_binding_sha256", sa.String()),
    )
    rows = bind.execute(
        sa.select(
            legacy_table.c.id,
            legacy_table.c.execution_backend,
            legacy_table.c.account_scope_sha256,
            legacy_table.c.budget_used_micros,
            legacy_table.c.budget_reserved_micros,
        )
    ).mappings()
    for row in rows:
        opening = {
            "schema_version": "flavourbench-provider-opening-balance-legacy-0011",
            "legacy_authorization_id": row["id"],
            "execution_backend": row["execution_backend"],
            "account_scope_sha256": row["account_scope_sha256"],
            "governed_used_micros": int(row["budget_used_micros"]),
            "governed_reserved_micros": int(row["budget_reserved_micros"]),
            "migration_status": "revoked_requires_reprovision",
        }
        binding = {
            "schema_version": "flavourbench-provider-binding-legacy-0011",
            "legacy_authorization_id": row["id"],
            "migration_status": "unverified_revoked",
        }
        bind.execute(
            sa.update(legacy_table)
            .where(legacy_table.c.id == row["id"])
            .values(
                opening_used_micros=int(row["budget_used_micros"]),
                opening_reserved_micros=int(row["budget_reserved_micros"]),
                opening_balance_json=opening,
                opening_balance_sha256=_sha256(opening),
                credential_binding_json=binding,
                credential_binding_sha256=_sha256(binding),
            )
        )

    op.create_index(
        "ix_provider_account_budgets_status",
        "provider_account_budgets",
        ["status"],
    )
    op.create_index(
        "ux_provider_account_budgets_opening_balance_sha256",
        "provider_account_budgets",
        ["opening_balance_sha256"],
        unique=True,
    )

    op.create_table(
        "provider_account_authorizations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "provider_account_budget_id",
            sa.String(length=36),
            sa.ForeignKey("provider_account_budgets.id"),
            nullable=False,
        ),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("account_scope_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "supersedes_authorization_id",
            sa.String(length=36),
            sa.ForeignKey("provider_account_authorizations.id"),
            nullable=True,
            unique=True,
        ),
        sa.Column(
            "authorization_reference_sha256", sa.String(length=64), nullable=False
        ),
        sa.Column("exposure_attestation_json", sa.JSON(), nullable=False),
        sa.Column(
            "exposure_attestation_sha256", sa.String(length=64), nullable=False
        ),
        sa.Column("authorized_used_micros", sa.BigInteger(), nullable=False),
        sa.Column("authorized_reserved_micros", sa.BigInteger(), nullable=False),
        sa.Column("credential_binding_json", sa.JSON(), nullable=False),
        sa.Column("credential_binding_sha256", sa.String(length=64), nullable=False),
        sa.Column("authorization_envelope_json", sa.JSON(), nullable=False),
        sa.Column(
            "authorization_envelope_sha256",
            sa.String(length=64),
            nullable=False,
            unique=True,
        ),
        sa.Column("authorization_hmac_sha256", sa.String(length=64), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "execution_backend IN ('openrouter', 'bedrock')",
            name="ck_provider_account_authorizations_backend",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_provider_account_authorizations_status",
        ),
        sa.CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL) OR "
            "(status = 'revoked' AND revoked_at IS NOT NULL)",
            name="ck_provider_account_authorizations_revocation",
        ),
        sa.CheckConstraint(
            "authorized_used_micros >= 0 AND authorized_reserved_micros >= 0",
            name="ck_provider_account_authorizations_nonnegative",
        ),
    )
    for name, columns in (
        (
            "ix_provider_account_authorizations_provider_account_budget_id",
            ["provider_account_budget_id"],
        ),
        ("ix_provider_account_authorizations_execution_backend", ["execution_backend"]),
        (
            "ix_provider_account_authorizations_account_scope_sha256",
            ["account_scope_sha256"],
        ),
        ("ix_provider_account_authorizations_status", ["status"]),
    ):
        op.create_index(name, "provider_account_authorizations", columns)
    op.create_index(
        "uq_provider_account_authorizations_active_scope",
        "provider_account_authorizations",
        ["execution_backend", "account_scope_sha256"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "bedrock_billing_crosschecks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "season_id",
            sa.String(length=36),
            sa.ForeignKey("seasons.id"),
            nullable=False,
        ),
        sa.Column(
            "provider_account_budget_id",
            sa.String(length=36),
            sa.ForeignKey("provider_account_budgets.id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "supersedes_crosscheck_id",
            sa.String(length=36),
            sa.ForeignKey("bedrock_billing_crosschecks.id"),
            nullable=True,
            unique=True,
        ),
        sa.Column("source_kind", sa.String(length=48), nullable=False),
        sa.Column("source_artifact_uri", sa.Text(), nullable=False),
        sa.Column("source_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("statement_sha256", sa.String(length=64), nullable=False),
        sa.Column("coverage_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("coverage_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("arm_set_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "generation_request_map_sha256", sa.String(length=64), nullable=False
        ),
        sa.Column("rate_card_estimated_micros", sa.BigInteger(), nullable=False),
        sa.Column("billed_usage_micros", sa.BigInteger(), nullable=False),
        sa.Column("billing_difference_micros", sa.BigInteger(), nullable=False),
        sa.Column("ledger_delta_micros", sa.BigInteger(), nullable=False),
        sa.Column("tolerance_micros", sa.BigInteger(), nullable=False),
        sa.Column("credits_policy", sa.String(length=80), nullable=False),
        sa.Column(
            "authorization_reference_sha256", sa.String(length=64), nullable=False
        ),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('accepted', 'discrepant')",
            name="ck_bedrock_billing_crosschecks_status",
        ),
        sa.UniqueConstraint("evidence_sha256"),
        sa.UniqueConstraint("source_artifact_sha256", "statement_sha256"),
    )
    for name, columns in (
        ("ix_bedrock_billing_crosschecks_season_id", ["season_id"]),
        (
            "ix_bedrock_billing_crosschecks_provider_account_budget_id",
            ["provider_account_budget_id"],
        ),
        ("ix_bedrock_billing_crosschecks_status", ["status"]),
        (
            "ix_bedrock_billing_crosschecks_supersedes_crosscheck_id",
            ["supersedes_crosscheck_id"],
        ),
        (
            "ix_bedrock_billing_crosschecks_source_artifact_sha256",
            ["source_artifact_sha256"],
        ),
        ("ix_bedrock_billing_crosschecks_arm_set_sha256", ["arm_set_sha256"]),
    ):
        op.create_index(name, "bedrock_billing_crosschecks", columns)

    op.create_table(
        "bedrock_billing_crosscheck_arms",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "crosscheck_id",
            sa.String(length=36),
            sa.ForeignKey("bedrock_billing_crosschecks.id"),
            nullable=False,
        ),
        sa.Column(
            "arm_id",
            sa.String(length=36),
            sa.ForeignKey("response_arms.id"),
            nullable=False,
        ),
        sa.Column("generation_set_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("crosscheck_id", "arm_id"),
    )
    op.create_index(
        "ix_bedrock_billing_crosscheck_arms_crosscheck_id",
        "bedrock_billing_crosscheck_arms",
        ["crosscheck_id"],
    )
    op.create_index(
        "ix_bedrock_billing_crosscheck_arms_arm_id",
        "bedrock_billing_crosscheck_arms",
        ["arm_id"],
    )

    # Historical development fixtures prior to 0007 did not contain the full raw
    # evidence schema. Keep the forward path compatible without pretending that
    # an absent cost ledger can be indexed.
    if "cost_events" in _tables(bind):
        cost_event_indexes = _indexes(bind, "cost_events")
        if "uq_cost_events_battle_provider_reserve" not in cost_event_indexes:
            op.create_index(
                "uq_cost_events_battle_provider_reserve",
                "cost_events",
                ["battle_id", "provider"],
                unique=True,
                postgresql_where=sa.text("kind = 'provider_reserve'"),
                sqlite_where=sa.text("kind = 'provider_reserve'"),
            )
        if (
            "uq_cost_events_battle_provider_account_reserve"
            not in cost_event_indexes
        ):
            op.create_index(
                "uq_cost_events_battle_provider_account_reserve",
                "cost_events",
                ["battle_id", "provider"],
                unique=True,
                postgresql_where=sa.text("kind = 'provider_account_reserve'"),
                sqlite_where=sa.text("kind = 'provider_account_reserve'"),
            )

    if bind.dialect.name == "postgresql":
        op.execute(
            """
            DROP TRIGGER IF EXISTS trg_provider_budget_contract_immutable
                ON season_provider_budgets;
            CREATE OR REPLACE FUNCTION flavourbench_provider_budget_contract_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'provider budget authorizations cannot be deleted';
                END IF;
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                   OLD.currency IS DISTINCT FROM NEW.currency OR
                   OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros OR
                   OLD.account_scope_sha256 IS DISTINCT FROM NEW.account_scope_sha256 OR
                   OLD.authorization_reference_sha256 IS DISTINCT FROM
                       NEW.authorization_reference_sha256 OR
                   OLD.account_authorization_envelope_sha256 IS DISTINCT FROM
                       NEW.account_authorization_envelope_sha256 OR
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
            CREATE TRIGGER trg_provider_budget_contract_immutable
            BEFORE UPDATE OR DELETE ON season_provider_budgets
            FOR EACH ROW EXECUTE FUNCTION flavourbench_provider_budget_contract_immutable();

            DROP TRIGGER IF EXISTS trg_account_budget_contract_immutable
                ON provider_account_budgets;
            CREATE OR REPLACE FUNCTION flavourbench_account_budget_contract_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'provider account ledgers cannot be deleted';
                END IF;
                IF OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                   OLD.currency IS DISTINCT FROM NEW.currency OR
                   OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros OR
                   OLD.opening_used_micros IS DISTINCT FROM NEW.opening_used_micros OR
                   OLD.opening_reserved_micros IS DISTINCT FROM
                       NEW.opening_reserved_micros OR
                   OLD.account_scope_sha256 IS DISTINCT FROM NEW.account_scope_sha256 OR
                   OLD.authorization_reference_sha256 IS DISTINCT FROM
                       NEW.authorization_reference_sha256 OR
                   OLD.opening_balance_json IS DISTINCT FROM NEW.opening_balance_json OR
                   OLD.opening_balance_sha256 IS DISTINCT FROM
                       NEW.opening_balance_sha256 OR
                   OLD.credential_binding_json IS DISTINCT FROM
                       NEW.credential_binding_json OR
                   OLD.credential_binding_sha256 IS DISTINCT FROM
                       NEW.credential_binding_sha256 OR
                   OLD.authorization_envelope_json IS DISTINCT FROM
                       NEW.authorization_envelope_json OR
                   OLD.authorization_envelope_sha256 IS DISTINCT FROM
                       NEW.authorization_envelope_sha256 OR
                   OLD.authorization_hmac_sha256 IS DISTINCT FROM
                       NEW.authorization_hmac_sha256 OR
                   OLD.valid_until IS DISTINCT FROM NEW.valid_until THEN
                    RAISE EXCEPTION 'provider account ledger contract is immutable';
                END IF;
                IF OLD.status = 'revoked' AND
                   (NEW.status IS DISTINCT FROM OLD.status OR
                    NEW.revoked_at IS DISTINCT FROM OLD.revoked_at) THEN
                    RAISE EXCEPTION 'provider account ledger revocation is irreversible';
                ELSIF NEW.status IS DISTINCT FROM OLD.status OR
                      NEW.revoked_at IS DISTINCT FROM OLD.revoked_at THEN
                    IF NOT (
                        (OLD.status = 'pending_verification' AND
                         NEW.status = 'active' AND
                         OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL) OR
                        (OLD.status IN ('pending_verification', 'active') AND
                         NEW.status = 'revoked' AND
                         OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL)
                    ) THEN
                        RAISE EXCEPTION 'invalid provider account ledger status transition';
                    END IF;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER trg_account_budget_contract_immutable
            BEFORE UPDATE OR DELETE ON provider_account_budgets
            FOR EACH ROW EXECUTE FUNCTION flavourbench_account_budget_contract_immutable();

            CREATE OR REPLACE FUNCTION flavourbench_account_authorization_immutable()
            RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'provider account authorization epochs are append-only';
                END IF;
                IF OLD.provider_account_budget_id IS DISTINCT FROM
                       NEW.provider_account_budget_id OR
                   OLD.execution_backend IS DISTINCT FROM NEW.execution_backend OR
                   OLD.account_scope_sha256 IS DISTINCT FROM NEW.account_scope_sha256 OR
                   OLD.supersedes_authorization_id IS DISTINCT FROM
                       NEW.supersedes_authorization_id OR
                   OLD.authorization_reference_sha256 IS DISTINCT FROM
                       NEW.authorization_reference_sha256 OR
                   OLD.exposure_attestation_json IS DISTINCT FROM
                       NEW.exposure_attestation_json OR
                   OLD.exposure_attestation_sha256 IS DISTINCT FROM
                       NEW.exposure_attestation_sha256 OR
                   OLD.authorized_used_micros IS DISTINCT FROM
                       NEW.authorized_used_micros OR
                   OLD.authorized_reserved_micros IS DISTINCT FROM
                       NEW.authorized_reserved_micros OR
                   OLD.credential_binding_json IS DISTINCT FROM
                       NEW.credential_binding_json OR
                   OLD.credential_binding_sha256 IS DISTINCT FROM
                       NEW.credential_binding_sha256 OR
                   OLD.authorization_envelope_json IS DISTINCT FROM
                       NEW.authorization_envelope_json OR
                   OLD.authorization_envelope_sha256 IS DISTINCT FROM
                       NEW.authorization_envelope_sha256 OR
                   OLD.authorization_hmac_sha256 IS DISTINCT FROM
                       NEW.authorization_hmac_sha256 OR
                   OLD.valid_until IS DISTINCT FROM NEW.valid_until THEN
                    RAISE EXCEPTION 'provider account authorization epoch is immutable';
                END IF;
                IF OLD.status = 'active' AND NEW.status = 'revoked' AND
                   OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL THEN
                    RETURN NEW;
                END IF;
                IF NEW.status IS DISTINCT FROM OLD.status OR
                   NEW.revoked_at IS DISTINCT FROM OLD.revoked_at THEN
                    RAISE EXCEPTION 'provider account authorization revocation is invalid';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER trg_account_authorization_immutable
            BEFORE UPDATE OR DELETE ON provider_account_authorizations
            FOR EACH ROW EXECUTE FUNCTION
                flavourbench_account_authorization_immutable();

            CREATE OR REPLACE FUNCTION flavourbench_raw_record_append_only()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'FlavourBench raw evidence is append-only';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        for table in (
            "votes",
            "cost_events",
            "generation_attempts",
            "bedrock_billing_crosschecks",
            "bedrock_billing_crosscheck_arms",
        ):
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON {table}; "
                f"CREATE TRIGGER trg_{table}_append_only BEFORE UPDATE OR DELETE "
                f"ON {table} FOR EACH ROW EXECUTE FUNCTION "
                "flavourbench_raw_record_append_only();"
            )


def downgrade() -> None:
    bind = op.get_bind()
    tables = _tables(bind)
    if bind.dialect.name == "postgresql":
        for table in (
            "votes",
            "cost_events",
            "generation_attempts",
            "bedrock_billing_crosschecks",
            "bedrock_billing_crosscheck_arms",
        ):
            op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON {table}")
        op.execute("DROP FUNCTION IF EXISTS flavourbench_raw_record_append_only()")
        op.execute(
            "DROP TRIGGER IF EXISTS trg_account_authorization_immutable "
            "ON provider_account_authorizations"
        )
        op.execute(
            "DROP FUNCTION IF EXISTS flavourbench_account_authorization_immutable()"
        )
    if "cost_events" in tables:
        cost_event_indexes = _indexes(bind, "cost_events")
        if "uq_cost_events_battle_provider_account_reserve" in cost_event_indexes:
            op.drop_index(
                "uq_cost_events_battle_provider_account_reserve",
                table_name="cost_events",
            )
        if "uq_cost_events_battle_provider_reserve" in cost_event_indexes:
            op.drop_index(
                "uq_cost_events_battle_provider_reserve", table_name="cost_events"
            )
    if "bedrock_billing_crosscheck_arms" in tables:
        op.drop_table("bedrock_billing_crosscheck_arms")
    if "bedrock_billing_crosschecks" in tables:
        op.drop_table("bedrock_billing_crosschecks")
    if "provider_account_authorizations" in tables:
        op.drop_table("provider_account_authorizations")
    if "provider_account_budgets" in tables:
        account_indexes = _indexes(bind, "provider_account_budgets")
        if "ux_provider_account_budgets_opening_balance_sha256" in account_indexes:
            op.drop_index(
                "ux_provider_account_budgets_opening_balance_sha256",
                table_name="provider_account_budgets",
            )
        if "ix_provider_account_budgets_status" in account_indexes:
            op.drop_index(
                "ix_provider_account_budgets_status",
                table_name="provider_account_budgets",
            )
        account_columns = _columns(bind, "provider_account_budgets")
        with op.batch_alter_table("provider_account_budgets") as batch:
            for constraint in (
                "ck_provider_account_budgets_revocation",
                "ck_provider_account_budgets_nonnegative",
                "ck_provider_account_budgets_status",
            ):
                batch.drop_constraint(constraint, type_="check")
            for column in (
                "revoked_at",
                "authorization_hmac_sha256",
                "credential_binding_sha256",
                "credential_binding_json",
                "opening_balance_sha256",
                "opening_balance_json",
                "opening_reserved_micros",
                "opening_used_micros",
                "status",
            ):
                if column in account_columns:
                    batch.drop_column(column)
    if "season_provider_budgets" in tables and (
        "account_authorization_envelope_sha256"
        in _columns(bind, "season_provider_budgets")
    ):
        with op.batch_alter_table("season_provider_budgets") as batch:
            batch.drop_column("account_authorization_envelope_sha256")

    # Reconstitute the exact 0010 trigger bodies. 0011 replaced these functions
    # with definitions that reference columns removed by this downgrade.
    if bind.dialect.name == "postgresql":
        op.execute(
            """
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
            """
        )
