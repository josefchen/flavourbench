"""Persist append-only provider-attempt lifecycle evidence.

Revision ID: 0003_generation_attempts
Revises: 0002_scientific_isolation
Create Date: 2026-07-15
"""

import sqlalchemy as sa

from alembic import op

revision = "0003_generation_attempts"
down_revision = "0002_scientific_isolation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "generation_attempts" not in inspector.get_table_names():
        op.create_table(
            "generation_attempts",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("attempt_id", sa.String(length=36), nullable=False),
            sa.Column("arm_id", sa.String(length=36), nullable=False),
            sa.Column("request_key_sha256", sa.String(length=64), nullable=False),
            sa.Column("phase", sa.String(length=80), nullable=False),
            sa.Column("attempt_index", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(length=80), nullable=False),
            sa.Column("generation_id", sa.String(length=160), nullable=True),
            sa.Column("http_status", sa.Integer(), nullable=True),
            sa.Column("error_type", sa.String(length=160), nullable=True),
            sa.Column("payload_sha256", sa.String(length=64), nullable=False),
            sa.Column("metadata_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["arm_id"], ["response_arms.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    existing_indexes = {
        index["name"] for index in sa.inspect(bind).get_indexes("generation_attempts")
    }
    indexes = {
        "ix_generation_attempts_attempt_id": ["attempt_id"],
        "ix_generation_attempts_arm_id": ["arm_id"],
        "ix_generation_attempts_phase": ["phase"],
        "ix_generation_attempts_event_type": ["event_type"],
        "ix_generation_attempts_generation_id": ["generation_id"],
        "ix_generation_attempts_arm_created": ["arm_id", "created_at"],
        "ix_generation_attempts_attempt": ["attempt_id", "created_at"],
    }
    for name, columns in indexes.items():
        if name not in existing_indexes:
            op.create_index(name, "generation_attempts", columns, unique=False)

    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_reject_generation_attempt_mutation()
            RETURNS trigger AS $$
            BEGIN
              RAISE EXCEPTION 'generation_attempts is append-only';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute("DROP TRIGGER IF EXISTS generation_attempts_append_only ON generation_attempts")
        op.execute(
            """
            CREATE TRIGGER generation_attempts_append_only
            BEFORE UPDATE OR DELETE ON generation_attempts
            FOR EACH ROW EXECUTE FUNCTION flavourbench_reject_generation_attempt_mutation()
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS generation_attempts_append_only ON generation_attempts")
        op.execute(
            "DROP FUNCTION IF EXISTS flavourbench_reject_generation_attempt_mutation()"
        )
    if "generation_attempts" in sa.inspect(bind).get_table_names():
        op.drop_table("generation_attempts")
