"""Add immutable scientific-isolation metadata to battles.

Revision ID: 0002_scientific_isolation
Revises: 0001_initial
Create Date: 2026-07-15
"""

import sqlalchemy as sa

from alembic import op

revision = "0002_scientific_isolation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


IMMUTABLE_COLUMNS = (
    "season_id",
    "run_class",
    "rank_eligible",
    "data_stratum",
    "task_id",
    "task_revision",
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


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    task_columns = {column["name"] for column in inspector.get_columns("tasks")}
    if "revision" not in task_columns:
        op.add_column(
            "tasks",
            sa.Column("revision", sa.Integer(), nullable=False, server_default=sa.text("1")),
        )

    battle_columns = {column["name"] for column in inspector.get_columns("battles")}
    additions = {
        "run_class": sa.Column(
            "run_class",
            sa.String(length=24),
            nullable=False,
            server_default="exploratory",
        ),
        "rank_eligible": sa.Column(
            "rank_eligible", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        "data_stratum": sa.Column(
            "data_stratum",
            sa.String(length=32),
            nullable=False,
            server_default="development",
        ),
        "task_id": sa.Column("task_id", sa.String(length=36), nullable=True),
        "task_revision": sa.Column("task_revision", sa.Integer(), nullable=True),
        "manifest_sha256": sa.Column(
            "manifest_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="unfrozen",
        ),
        "scheduler_version": sa.Column(
            "scheduler_version",
            sa.String(length=80),
            nullable=False,
            server_default="legacy-unversioned",
        ),
        "assignment_seed": sa.Column(
            "assignment_seed",
            sa.String(length=64),
            nullable=False,
            server_default="0" * 64,
        ),
        "track_assignment_probability": sa.Column(
            "track_assignment_probability",
            sa.String(length=48),
            nullable=False,
            server_default="unknown",
        ),
        "model_assignment_probability": sa.Column(
            "model_assignment_probability",
            sa.String(length=48),
            nullable=False,
            server_default="unknown",
        ),
        "side_assignment_probability": sa.Column(
            "side_assignment_probability",
            sa.String(length=48),
            nullable=False,
            server_default="unknown",
        ),
    }
    for name, column in additions.items():
        if name not in battle_columns:
            op.add_column("battles", column)

    inspector = sa.inspect(bind)
    foreign_key_rows = inspector.get_foreign_keys("battles")
    foreign_keys = {item.get("name") for item in foreign_key_rows}
    task_foreign_key_exists = any(
        item.get("constrained_columns") == ["task_id"]
        and item.get("referred_table") == "tasks"
        for item in foreign_key_rows
    )
    checks = {item.get("name") for item in inspector.get_check_constraints("battles")}
    indexes = {item.get("name") for item in inspector.get_indexes("battles")}

    with op.batch_alter_table("battles") as batch:
        if (
            "fk_battles_task_id_tasks" not in foreign_keys
            and not task_foreign_key_exists
        ):
            batch.create_foreign_key(
                "fk_battles_task_id_tasks", "tasks", ["task_id"], ["id"]
            )
        if "ck_battles_run_class" not in checks:
            batch.create_check_constraint(
                "ck_battles_run_class",
                "run_class IN ('mock', 'smoke', 'exploratory', 'pilot', 'official')",
            )
        if "ck_battles_data_stratum" not in checks:
            batch.create_check_constraint(
                "ck_battles_data_stratum",
                "data_stratum IN ('public_freeform', 'controlled', 'development', 'legacy')",
            )
        if "ck_battles_task_revision_pair" not in checks:
            batch.create_check_constraint(
                "ck_battles_task_revision_pair",
                "(task_id IS NULL AND task_revision IS NULL) OR "
                "(task_id IS NOT NULL AND task_revision IS NOT NULL)",
            )

    index_definitions = {
        "ix_battles_run_class": ["run_class"],
        "ix_battles_rank_eligible": ["rank_eligible"],
        "ix_battles_data_stratum": ["data_stratum"],
        "ix_battles_task_id": ["task_id"],
        "ix_battles_manifest_sha256": ["manifest_sha256"],
        "ix_battles_rank_scope": [
            "season_id",
            "rank_eligible",
            "run_class",
            "manifest_sha256",
            "track",
        ],
    }
    for name, columns in index_definitions.items():
        if name not in indexes:
            op.create_index(name, "battles", columns)

    if bind.dialect.name == "postgresql":
        comparisons = " OR ".join(
            f"OLD.{column} IS DISTINCT FROM NEW.{column}" for column in IMMUTABLE_COLUMNS
        )
        op.execute(
            f"""
            CREATE FUNCTION flavourbench_prevent_battle_provenance_update()
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
        op.execute(
            """
            CREATE TRIGGER trg_battles_scientific_provenance_immutable
            BEFORE UPDATE ON battles
            FOR EACH ROW
            EXECUTE FUNCTION flavourbench_prevent_battle_provenance_update();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "DROP TRIGGER IF EXISTS trg_battles_scientific_provenance_immutable ON battles"
        )
        op.execute("DROP FUNCTION IF EXISTS flavourbench_prevent_battle_provenance_update()")

    op.drop_index("ix_battles_rank_scope", table_name="battles")
    op.drop_index("ix_battles_manifest_sha256", table_name="battles")
    op.drop_index("ix_battles_task_id", table_name="battles")
    op.drop_index("ix_battles_data_stratum", table_name="battles")
    op.drop_index("ix_battles_rank_eligible", table_name="battles")
    op.drop_index("ix_battles_run_class", table_name="battles")
    with op.batch_alter_table("battles") as batch:
        batch.drop_constraint("ck_battles_task_revision_pair", type_="check")
        batch.drop_constraint("ck_battles_data_stratum", type_="check")
        batch.drop_constraint("ck_battles_run_class", type_="check")
        batch.drop_constraint("fk_battles_task_id_tasks", type_="foreignkey")
        for column_name in (
            "side_assignment_probability",
            "model_assignment_probability",
            "track_assignment_probability",
            "assignment_seed",
            "scheduler_version",
            "manifest_sha256",
            "task_revision",
            "task_id",
            "data_stratum",
            "rank_eligible",
            "run_class",
        ):
            batch.drop_column(column_name)
    with op.batch_alter_table("tasks") as batch:
        batch.drop_column("revision")
