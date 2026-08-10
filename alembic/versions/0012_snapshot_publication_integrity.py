"""Enforce one published leaderboard snapshot per release scope.

Revision ID: 0012_snapshot_integrity
Revises: 0011_account_billing
"""

from __future__ import annotations

from collections import defaultdict

import sqlalchemy as sa

from alembic import op

revision = "0012_snapshot_integrity"
down_revision = "0011_account_billing"
branch_labels = None
depends_on = None


PUBLIC_INDEX = "uq_leaderboard_snapshots_one_published_public_scope"
CONTROLLED_INDEX = "uq_leaderboard_snapshots_one_published_controlled_scope"
STATUS_TRIGGER = "trg_leaderboard_snapshot_status_forward_only"
STATUS_FUNCTION = "flavourbench_leaderboard_snapshot_status_forward_only"


def _withdraw_duplicate_publications(bind: sa.Connection) -> None:
    snapshots = sa.table(
        "leaderboard_snapshots",
        sa.column("id", sa.String()),
        sa.column("season_id", sa.String()),
        sa.column("track", sa.String()),
        sa.column("cohort", sa.String()),
        sa.column("category", sa.String()),
        sa.column("data_stratum", sa.String()),
        sa.column("controlled_run_id", sa.String()),
        sa.column("publication_status", sa.String()),
        sa.column("published_at", sa.DateTime(timezone=True)),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    rows = bind.execute(
        sa.select(
            snapshots.c.id,
            snapshots.c.season_id,
            snapshots.c.track,
            snapshots.c.cohort,
            snapshots.c.category,
            snapshots.c.data_stratum,
            snapshots.c.controlled_run_id,
            snapshots.c.published_at,
            snapshots.c.created_at,
        )
        .where(snapshots.c.publication_status == "published")
        .order_by(
            snapshots.c.published_at.desc(),
            snapshots.c.created_at.desc(),
            snapshots.c.id.desc(),
        )
    ).mappings()
    grouped: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for row in rows:
        scope = (
            row["season_id"],
            row["controlled_run_id"],
            row["track"],
            row["cohort"],
            row["category"],
            row["data_stratum"],
        )
        grouped[scope].append(str(row["id"]))
    duplicate_ids = [
        snapshot_id for identifiers in grouped.values() for snapshot_id in identifiers[1:]
    ]
    if duplicate_ids:
        bind.execute(
            sa.update(snapshots)
            .where(snapshots.c.id.in_(duplicate_ids))
            .values(publication_status="withdrawn")
        )


def upgrade() -> None:
    bind = op.get_bind()
    _withdraw_duplicate_publications(bind)
    public_predicate = sa.text("publication_status = 'published' AND controlled_run_id IS NULL")
    controlled_predicate = sa.text(
        "publication_status = 'published' AND controlled_run_id IS NOT NULL"
    )
    op.create_index(
        PUBLIC_INDEX,
        "leaderboard_snapshots",
        ["season_id", "track", "cohort", "category", "data_stratum"],
        unique=True,
        postgresql_where=public_predicate,
        sqlite_where=public_predicate,
    )
    op.create_index(
        CONTROLLED_INDEX,
        "leaderboard_snapshots",
        ["controlled_run_id", "track", "cohort", "category", "data_stratum"],
        unique=True,
        postgresql_where=controlled_predicate,
        sqlite_where=controlled_predicate,
    )
    if bind.dialect.name == "postgresql":
        op.execute(
            f"""
            CREATE FUNCTION {STATUS_FUNCTION}() RETURNS trigger AS $$
            BEGIN
                IF NOT (
                    OLD.publication_status = NEW.publication_status OR
                    (OLD.publication_status = 'draft' AND
                     NEW.publication_status IN ('published', 'withdrawn')) OR
                    (OLD.publication_status = 'published' AND
                     NEW.publication_status = 'withdrawn')
                ) THEN
                    RAISE EXCEPTION 'leaderboard snapshot status cannot move backward';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {STATUS_TRIGGER}
            BEFORE UPDATE OF publication_status ON leaderboard_snapshots
            FOR EACH ROW EXECUTE FUNCTION {STATUS_FUNCTION}()
            """
        )
    elif bind.dialect.name == "sqlite":
        op.execute(
            f"""
            CREATE TRIGGER {STATUS_TRIGGER}
            BEFORE UPDATE OF publication_status ON leaderboard_snapshots
            FOR EACH ROW
            WHEN NOT (
                OLD.publication_status = NEW.publication_status OR
                (OLD.publication_status = 'draft' AND
                 NEW.publication_status IN ('published', 'withdrawn')) OR
                (OLD.publication_status = 'published' AND
                 NEW.publication_status = 'withdrawn')
            )
            BEGIN
                SELECT RAISE(ABORT, 'leaderboard snapshot status cannot move backward');
            END;
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {STATUS_TRIGGER} ON leaderboard_snapshots")
        op.execute(f"DROP FUNCTION IF EXISTS {STATUS_FUNCTION}()")
    elif bind.dialect.name == "sqlite":
        op.execute(f"DROP TRIGGER IF EXISTS {STATUS_TRIGGER}")
    op.drop_index(CONTROLLED_INDEX, table_name="leaderboard_snapshots")
    op.drop_index(PUBLIC_INDEX, table_name="leaderboard_snapshots")
