"""Add tenant-scoped controlled runs and publication-gated snapshots.

Revision ID: 0008_private_controlled_runs
Revises: 0007_leaderboard_data_stratum
"""

import sqlalchemy as sa

from alembic import op

revision = "0008_private_controlled_runs"
down_revision = "0007_leaderboard_data_stratum"
branch_labels = None
depends_on = None


IMMUTABLE_BATTLE_COLUMNS = (
    "season_id",
    "run_class",
    "rank_eligible",
    "data_stratum",
    "task_id",
    "task_revision",
    "controlled_run_id",
    "manifest_sha256",
    "scheduler_version",
    "assignment_seed",
    "track_assignment_probability",
    "model_assignment_probability",
    "side_assignment_probability",
    "track",
    "category",
    "prompt_sha256",
    "client_nonce_sha256",
)


def _index_names(bind: sa.Connection, table: str) -> set[str]:
    return {row["name"] for row in sa.inspect(bind).get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "controlled_runs" not in tables:
        op.create_table(
            "controlled_runs",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("season_id", sa.String(length=36), nullable=False),
            sa.Column("organization_reference_sha256", sa.String(length=64), nullable=False),
            sa.Column("access_token_sha256", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
            sa.Column("protocol_version", sa.String(length=80), nullable=False),
            sa.Column("rater_plan_sha256", sa.String(length=64), nullable=False),
            sa.Column("analysis_plan_sha256", sa.String(length=64), nullable=False),
            sa.Column("budget_cap_micros", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("budget_used_micros", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column(
                "budget_reserved_micros", sa.BigInteger(), nullable=False, server_default="0"
            ),
            sa.Column("run_card_json", sa.JSON(), nullable=False),
            sa.Column("run_card_sha256", sa.String(length=64), nullable=False),
            sa.Column("run_card_signature", sa.String(length=64), nullable=False),
            sa.Column(
                "release_authorized", sa.Boolean(), nullable=False, server_default=sa.false()
            ),
            sa.Column(
                "release_authorization_reference_sha256",
                sa.String(length=64),
                nullable=True,
            ),
            sa.Column("release_authorized_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "status IN ('active', 'collection_complete', 'closed', 'revoked')",
                name="ck_controlled_runs_status",
            ),
            sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("access_token_sha256"),
            sa.UniqueConstraint("run_card_sha256"),
        )

    controlled_indexes = _index_names(bind, "controlled_runs")
    controlled_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("controlled_runs")
    }
    for name in ("budget_used_micros", "budget_reserved_micros"):
        if name not in controlled_columns:
            op.add_column(
                "controlled_runs",
                sa.Column(name, sa.BigInteger(), nullable=False, server_default="0"),
            )
    for name, columns in {
        "ix_controlled_runs_season_id": ["season_id"],
        "ix_controlled_runs_organization_reference_sha256": [
            "organization_reference_sha256"
        ],
        "ix_controlled_runs_access_token_sha256": ["access_token_sha256"],
        "ix_controlled_runs_status": ["status"],
        "ix_controlled_runs_run_card_sha256": ["run_card_sha256"],
        "ix_controlled_runs_release_authorized": ["release_authorized"],
    }.items():
        if name not in controlled_indexes:
            op.create_index(name, "controlled_runs", columns)

    tables = set(sa.inspect(bind).get_table_names())
    if "controlled_run_reviewers" not in tables:
        op.create_table(
            "controlled_run_reviewers",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("controlled_run_id", sa.String(length=36), nullable=False),
            sa.Column("reviewer_id", sa.String(length=36), nullable=False),
            sa.Column(
                "authorization_reference_sha256", sa.String(length=64), nullable=False
            ),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["controlled_run_id"], ["controlled_runs.id"]),
            sa.ForeignKeyConstraint(["reviewer_id"], ["expert_reviewers.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("controlled_run_id", "reviewer_id"),
        )
    reviewer_indexes = _index_names(bind, "controlled_run_reviewers")
    for name, columns in {
        "ix_controlled_run_reviewers_controlled_run_id": ["controlled_run_id"],
        "ix_controlled_run_reviewers_reviewer_id": ["reviewer_id"],
        "ix_controlled_run_reviewers_active": ["active"],
    }.items():
        if name not in reviewer_indexes:
            op.create_index(name, "controlled_run_reviewers", columns)

    battle_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("battles")
    }
    if "controlled_run_id" not in battle_columns:
        op.add_column(
            "battles", sa.Column("controlled_run_id", sa.String(length=36), nullable=True)
        )
    battle_fks = sa.inspect(bind).get_foreign_keys("battles")
    has_battle_run_fk = any(
        row.get("constrained_columns") == ["controlled_run_id"]
        and row.get("referred_table") == "controlled_runs"
        for row in battle_fks
    )
    if not has_battle_run_fk:
        with op.batch_alter_table("battles") as batch:
            batch.create_foreign_key(
                "fk_battles_controlled_run_id_controlled_runs",
                "controlled_runs",
                ["controlled_run_id"],
                ["id"],
            )
    battle_indexes = _index_names(bind, "battles")
    if "ix_battles_controlled_run_id" not in battle_indexes:
        op.create_index("ix_battles_controlled_run_id", "battles", ["controlled_run_id"])
    if "ix_battles_controlled_run_rank_scope" not in battle_indexes:
        op.create_index(
            "ix_battles_controlled_run_rank_scope",
            "battles",
            ["controlled_run_id", "track", "status"],
        )

    snapshot_columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("leaderboard_snapshots")
    }
    additions = {
        "controlled_run_id": sa.Column(
            "controlled_run_id", sa.String(length=36), nullable=True
        ),
        "publication_status": sa.Column(
            "publication_status",
            sa.String(length=24),
            nullable=False,
            server_default="draft",
        ),
        "publication_reference_sha256": sa.Column(
            "publication_reference_sha256", sa.String(length=64), nullable=True
        ),
        "published_at": sa.Column(
            "published_at", sa.DateTime(timezone=True), nullable=True
        ),
    }
    for name, column in additions.items():
        if name not in snapshot_columns:
            op.add_column("leaderboard_snapshots", column)

    snapshot_inspector = sa.inspect(bind)
    snapshot_fks = snapshot_inspector.get_foreign_keys("leaderboard_snapshots")
    has_snapshot_run_fk = any(
        row.get("constrained_columns") == ["controlled_run_id"]
        and row.get("referred_table") == "controlled_runs"
        for row in snapshot_fks
    )
    snapshot_checks = {
        row.get("name")
        for row in snapshot_inspector.get_check_constraints("leaderboard_snapshots")
    }
    if (
        not has_snapshot_run_fk
        or "ck_leaderboard_snapshots_publication_status" not in snapshot_checks
    ):
        with op.batch_alter_table("leaderboard_snapshots") as batch:
            if not has_snapshot_run_fk:
                batch.create_foreign_key(
                    "fk_leaderboard_snapshots_controlled_run_id_controlled_runs",
                    "controlled_runs",
                    ["controlled_run_id"],
                    ["id"],
                )
            if "ck_leaderboard_snapshots_publication_status" not in snapshot_checks:
                batch.create_check_constraint(
                    "ck_leaderboard_snapshots_publication_status",
                    "publication_status IN ('draft', 'published', 'withdrawn')",
                )
    snapshot_indexes = _index_names(bind, "leaderboard_snapshots")
    for name, columns in {
        "ix_leaderboard_snapshots_controlled_run_id": ["controlled_run_id"],
        "ix_leaderboard_snapshots_publication_status": ["publication_status"],
    }.items():
        if name not in snapshot_indexes:
            op.create_index(name, "leaderboard_snapshots", columns)

    if bind.dialect.name == "postgresql":
        comparisons = " OR ".join(
            f"OLD.{column} IS DISTINCT FROM NEW.{column}"
            for column in IMMUTABLE_BATTLE_COLUMNS
        )
        op.execute(
            f"""
            CREATE OR REPLACE FUNCTION flavourbench_prevent_battle_provenance_update()
            RETURNS trigger AS $$
            BEGIN
                IF {comparisons} THEN
                    RAISE EXCEPTION
                        'battle scientific provenance is immutable; insert a superseding battle';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        legacy_columns = tuple(
            column
            for column in IMMUTABLE_BATTLE_COLUMNS
            if column != "controlled_run_id"
        )
        comparisons = " OR ".join(
            f"OLD.{column} IS DISTINCT FROM NEW.{column}"
            for column in legacy_columns
        )
        op.execute(
            f"""
            CREATE OR REPLACE FUNCTION flavourbench_prevent_battle_provenance_update()
            RETURNS trigger AS $$
            BEGIN
                IF {comparisons} THEN
                    RAISE EXCEPTION
                        'battle scientific provenance is immutable; insert a superseding battle';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    snapshot_indexes = _index_names(bind, "leaderboard_snapshots")
    for name in (
        "ix_leaderboard_snapshots_publication_status",
        "ix_leaderboard_snapshots_controlled_run_id",
    ):
        if name in snapshot_indexes:
            op.drop_index(name, table_name="leaderboard_snapshots")
    with op.batch_alter_table("leaderboard_snapshots") as batch:
        snapshot_fks = sa.inspect(bind).get_foreign_keys("leaderboard_snapshots")
        if any(
            row.get("constrained_columns") == ["controlled_run_id"]
            for row in snapshot_fks
        ):
            batch.drop_constraint(
                "fk_leaderboard_snapshots_controlled_run_id_controlled_runs",
                type_="foreignkey",
            )
        checks = {
            row.get("name")
            for row in sa.inspect(bind).get_check_constraints("leaderboard_snapshots")
        }
        if "ck_leaderboard_snapshots_publication_status" in checks:
            batch.drop_constraint(
                "ck_leaderboard_snapshots_publication_status", type_="check"
            )
        columns = {
            row["name"] for row in sa.inspect(bind).get_columns("leaderboard_snapshots")
        }
        for name in (
            "published_at",
            "publication_reference_sha256",
            "publication_status",
            "controlled_run_id",
        ):
            if name in columns:
                batch.drop_column(name)

    battle_indexes = _index_names(bind, "battles")
    for name in (
        "ix_battles_controlled_run_rank_scope",
        "ix_battles_controlled_run_id",
    ):
        if name in battle_indexes:
            op.drop_index(name, table_name="battles")
    with op.batch_alter_table("battles") as batch:
        battle_fks = sa.inspect(bind).get_foreign_keys("battles")
        if any(row.get("constrained_columns") == ["controlled_run_id"] for row in battle_fks):
            batch.drop_constraint(
                "fk_battles_controlled_run_id_controlled_runs", type_="foreignkey"
            )
        if "controlled_run_id" in {
            row["name"] for row in sa.inspect(bind).get_columns("battles")
        }:
            batch.drop_column("controlled_run_id")
    if "controlled_run_reviewers" in set(sa.inspect(bind).get_table_names()):
        op.drop_table("controlled_run_reviewers")
    if "controlled_runs" in set(sa.inspect(bind).get_table_names()):
        op.drop_table("controlled_runs")
