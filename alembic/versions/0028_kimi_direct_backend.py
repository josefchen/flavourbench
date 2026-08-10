"""Add the governed direct-Kimi execution backend.

Revision ID: 0028_kimi_direct_backend
Revises: 0027_research_release_archives
"""

from __future__ import annotations

from alembic import op

revision = "0028_kimi_direct_backend"
down_revision = "0027_research_release_archives"
branch_labels = None
depends_on = None


CONSTRAINTS = (
    (
        "season_provider_budgets",
        "ck_season_provider_budgets_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', 'mock')",
    ),
    (
        "provider_account_budgets",
        "ck_provider_account_budgets_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct')",
    ),
    (
        "provider_account_authorizations",
        "ck_provider_account_authorizations_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct')",
    ),
)


def _restore_sqlite_budget_guards() -> None:
    """Restore triggers removed by SQLite's table-recreation migration path.

    Alembic implements check-constraint changes in SQLite by copying each table.
    SQLite does not carry triggers across that copy, so the two spend ledgers
    would otherwise lose the monotonicity guards installed by revision 0015.
    Keep these bodies byte-for-byte equivalent to that revision's SQLite guards.
    """

    for table in ("season_provider_budgets", "provider_account_budgets"):
        trigger = f"trg_governed_spend_monotonic_{table}"
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}_insert")
        op.execute(
            f"""
            CREATE TRIGGER {trigger}
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            WHEN NEW.budget_used_micros < 0
              OR NEW.budget_reserved_micros < 0
              OR NEW.budget_used_micros < OLD.budget_used_micros
            BEGIN
                SELECT RAISE(ABORT, 'governed spend cannot move backward');
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {trigger}_insert
            BEFORE INSERT ON {table}
            FOR EACH ROW
            WHEN NEW.budget_used_micros < 0 OR NEW.budget_reserved_micros < 0
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'governed spend and reservations must be nonnegative'
                );
            END;
            """
        )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0028: {dialect}")
    for table, name, expression in CONSTRAINTS:
        if dialect == "postgresql":
            op.drop_constraint(name, table, type_="check")
            op.create_check_constraint(name, table, expression)
        else:
            with op.batch_alter_table(table, recreate="always") as batch:
                batch.drop_constraint(name, type_="check")
                batch.create_check_constraint(name, expression)
    if dialect == "sqlite":
        _restore_sqlite_budget_guards()


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across research-release archive policy is prohibited; "
        "downgrade across provider-backend authority is prohibited"
    )
