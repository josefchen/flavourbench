"""Widen vote and snapshot cohorts for disclosed expert affiliations.

Revision ID: 0006_widen_rater_cohorts
Revises: 0005_expert_governance
Create Date: 2026-07-16
"""

import sqlalchemy as sa

from alembic import op

revision = "0006_widen_rater_cohorts"
down_revision = "0005_expert_governance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("votes") as batch:
        batch.alter_column(
            "cohort",
            existing_type=sa.String(length=24),
            type_=sa.String(length=48),
            existing_nullable=False,
        )
    with op.batch_alter_table("leaderboard_snapshots") as batch:
        batch.alter_column(
            "cohort",
            existing_type=sa.String(length=24),
            type_=sa.String(length=48),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("leaderboard_snapshots") as batch:
        batch.alter_column(
            "cohort",
            existing_type=sa.String(length=48),
            type_=sa.String(length=24),
            existing_nullable=False,
        )
    with op.batch_alter_table("votes") as batch:
        batch.alter_column(
            "cohort",
            existing_type=sa.String(length=48),
            type_=sa.String(length=24),
            existing_nullable=False,
        )
