"""Bind leaderboard snapshots to one benchmark population.

Revision ID: 0007_leaderboard_data_stratum
Revises: 0006_widen_rater_cohorts
"""

import sqlalchemy as sa

from alembic import op

revision = "0007_leaderboard_data_stratum"
down_revision = "0006_widen_rater_cohorts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {
        column["name"] for column in inspector.get_columns("leaderboard_snapshots")
    }
    if "data_stratum" not in columns:
        op.add_column(
            "leaderboard_snapshots",
            sa.Column(
                "data_stratum",
                sa.String(length=32),
                nullable=False,
                server_default="legacy_mixed",
            ),
        )

    indexes = {
        index["name"] for index in sa.inspect(bind).get_indexes("leaderboard_snapshots")
    }
    if "ix_leaderboard_snapshots_data_stratum" not in indexes:
        op.create_index(
            "ix_leaderboard_snapshots_data_stratum",
            "leaderboard_snapshots",
            ["data_stratum"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {
        index["name"] for index in inspector.get_indexes("leaderboard_snapshots")
    }
    if "ix_leaderboard_snapshots_data_stratum" in indexes:
        op.drop_index(
            "ix_leaderboard_snapshots_data_stratum",
            table_name="leaderboard_snapshots",
        )
    columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("leaderboard_snapshots")
    }
    if "data_stratum" in columns:
        op.drop_column("leaderboard_snapshots", "data_stratum")
