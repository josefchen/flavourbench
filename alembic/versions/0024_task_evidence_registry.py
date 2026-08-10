"""Add an append-only registry for task-bound confirmatory evidence.

Revision ID: 0024_task_evidence_registry
Revises: 0023_postgresql_finish_guard_release_fence
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0024_task_evidence_registry"
down_revision = "0023_postgresql_finish_guard_release_fence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "task_evidence_artifacts" in tables:
        raise RuntimeError(
            "task_evidence_artifacts already exists before revision 0024; "
            "refusing to bless an unverified schema"
        )
    op.create_table(
        "task_evidence_artifacts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(length=36),
            sa.ForeignKey("tasks.id"),
            nullable=False,
        ),
        sa.Column("evidence_type", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=96), nullable=False),
        sa.Column("revision_ordinal", sa.Integer(), nullable=False),
        sa.Column("artifact_json", sa.JSON(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("task_binding_sha256", sa.String(length=64), nullable=False),
        sa.Column("verification_receipt_json", sa.JSON(), nullable=False),
        sa.Column("verification_receipt_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "supersedes_artifact_id",
            sa.String(length=36),
            sa.ForeignKey("task_evidence_artifacts.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "evidence_type IN ('validator_contract', 'contamination_audit')",
            name="ck_task_evidence_artifacts_type",
        ),
        sa.CheckConstraint(
            "revision_ordinal > 0",
            name="ck_task_evidence_artifacts_positive_revision",
        ),
        sa.UniqueConstraint("artifact_sha256"),
        sa.UniqueConstraint("task_id", "evidence_type", "revision_ordinal"),
        sa.UniqueConstraint("supersedes_artifact_id"),
        sa.UniqueConstraint("task_binding_sha256"),
        sa.UniqueConstraint("verification_receipt_sha256"),
    )
    op.create_index(
        "ix_task_evidence_artifacts_task_id",
        "task_evidence_artifacts",
        ["task_id"],
    )
    op.create_index(
        "ix_task_evidence_artifacts_evidence_type",
        "task_evidence_artifacts",
        ["evidence_type"],
    )
    op.create_index(
        "ix_task_evidence_artifacts_artifact_sha256",
        "task_evidence_artifacts",
        ["artifact_sha256"],
    )
    op.create_index(
        "ix_task_evidence_artifacts_verification_receipt_sha256",
        "task_evidence_artifacts",
        ["verification_receipt_sha256"],
    )


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        raise RuntimeError(
            "PostgreSQL downgrade across the append-only task-evidence registry is prohibited"
        )
    if dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0024: {dialect}")
    op.drop_table("task_evidence_artifacts")
