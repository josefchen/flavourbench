"""Freeze exact per-season model endpoint contracts.

Revision ID: 0004_frozen_endpoint_contracts
Revises: 0003_generation_attempts
Create Date: 2026-07-15
"""

import sqlalchemy as sa

from alembic import op

revision = "0004_frozen_endpoint_contracts"
down_revision = "0003_generation_attempts"
branch_labels = None
depends_on = None


CONTRACT_COLUMNS = (
    "model_id",
    "slot_role",
    "provider_slug",
    "expected_actual_model_id",
    "expected_actual_provider_slug",
    "supported_parameters_json",
    "decoding_json",
    "endpoint_max_completion_tokens",
    "endpoint_document_sha256",
    "endpoint_contract_sha256",
    "worst_case_cost_micros",
    "manifest_sha256",
)


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("season_models")}
    json_array_default = (
        sa.text("'[]'::json")
        if bind.dialect.name == "postgresql"
        else sa.text("'[]'")
    )
    json_object_default = (
        sa.text("'{}'::json")
        if bind.dialect.name == "postgresql"
        else sa.text("'{}'")
    )
    additions = {
        "expected_actual_model_id": sa.Column(
            "expected_actual_model_id",
            sa.String(length=240),
            nullable=False,
            server_default="unfrozen",
        ),
        "expected_actual_provider_slug": sa.Column(
            "expected_actual_provider_slug",
            sa.String(length=160),
            nullable=False,
            server_default="unfrozen",
        ),
        "supported_parameters_json": sa.Column(
            "supported_parameters_json",
            sa.JSON(),
            nullable=False,
            server_default=json_array_default,
        ),
        "decoding_json": sa.Column(
            "decoding_json",
            sa.JSON(),
            nullable=False,
            server_default=json_object_default,
        ),
        "endpoint_max_completion_tokens": sa.Column(
            "endpoint_max_completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        "endpoint_document_sha256": sa.Column(
            "endpoint_document_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="unfrozen",
        ),
        "endpoint_contract_sha256": sa.Column(
            "endpoint_contract_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="unfrozen",
        ),
    }
    for name, column in additions.items():
        if name not in columns:
            op.add_column("season_models", column)

    if bind.dialect.name == "postgresql":
        comparisons = " OR ".join(
            f"OLD.{column} IS DISTINCT FROM NEW.{column}" for column in CONTRACT_COLUMNS
        )
        op.execute(
            f"""
            CREATE FUNCTION flavourbench_prevent_endpoint_contract_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                   AND ({comparisons}) THEN
                    RAISE EXCEPTION
                        'frozen season endpoint contract is immutable; create a new manifest';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_season_models_endpoint_contract_immutable
            BEFORE UPDATE ON season_models
            FOR EACH ROW
            EXECUTE FUNCTION flavourbench_prevent_endpoint_contract_update();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "DROP TRIGGER IF EXISTS trg_season_models_endpoint_contract_immutable "
            "ON season_models"
        )
        op.execute("DROP FUNCTION IF EXISTS flavourbench_prevent_endpoint_contract_update()")
    columns = {column["name"] for column in sa.inspect(bind).get_columns("season_models")}
    with op.batch_alter_table("season_models") as batch:
        for name in (
            "endpoint_contract_sha256",
            "endpoint_document_sha256",
            "endpoint_max_completion_tokens",
            "decoding_json",
            "supported_parameters_json",
            "expected_actual_provider_slug",
            "expected_actual_model_id",
        ):
            if name in columns:
                batch.drop_column(name)
