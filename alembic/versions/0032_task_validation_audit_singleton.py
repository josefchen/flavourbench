"""Seal one task-validation batch-audit decision per authorization.

Revision ID: 0032_task_validation_audit_singleton
Revises: 0031_task_validation_campaign_runtime
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0032_task_validation_audit_singleton"
down_revision = "0031_task_validation_campaign_runtime"
branch_labels = None
depends_on = None


INDEX_NAME = "uq_task_validation_campaign_events_audit_authorization_type"


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0032: {dialect}")
    predicate = sa.text("audit_authorization_id IS NOT NULL")
    op.create_index(
        INDEX_NAME,
        "task_validation_campaign_events",
        ["campaign_sha256", "audit_authorization_id", "event_type"],
        unique=True,
        postgresql_where=predicate,
        sqlite_where=predicate,
    )


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across sealed task-validation audit evidence is prohibited"
    )
