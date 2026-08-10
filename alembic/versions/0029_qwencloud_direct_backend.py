"""Add the governed direct-QwenCloud execution backend.

Revision ID: 0029_qwencloud_direct_backend
Revises: 0028_kimi_direct_backend
"""

from __future__ import annotations

import re

import sqlalchemy as sa

from alembic import op

revision = "0029_qwencloud_direct_backend"
down_revision = "0028_kimi_direct_backend"
branch_labels = None
depends_on = None


CONSTRAINTS = (
    (
        "season_provider_budgets",
        "ck_season_provider_budgets_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', "
        "'qwencloud_direct', 'mock')",
    ),
    (
        "provider_account_budgets",
        "ck_provider_account_budgets_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', "
        "'qwencloud_direct')",
    ),
    (
        "provider_account_authorizations",
        "ck_provider_account_authorizations_backend",
        "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', "
        "'qwencloud_direct')",
    ),
)


def _restore_sqlite_budget_guards() -> None:
    """Restore guards dropped by SQLite table recreation."""

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


def _rewrite_postgresql_reservation_function(definition: str) -> str:
    """Add direct backends to exactly two backend predicates.

    PostgreSQL may deparse the same ``IN`` expression in several equivalent
    forms.  Match alternatives against the original definition in one regex
    pass so a broad fallback cannot match text introduced by an earlier
    replacement.
    """

    replacements = (
        (
            "a.execution_backend IN ('openrouter', 'bedrock')",
            "a.execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', "
            "'qwencloud_direct')",
        ),
        (
            "ARRAY['openrouter'::text, 'bedrock'::text]",
            "ARRAY['openrouter'::text, 'bedrock'::text, 'kimi_direct'::text, "
            "'qwencloud_direct'::text]",
        ),
        (
            "'openrouter'::character varying, 'bedrock'::character varying",
            "'openrouter'::character varying, 'bedrock'::character varying, "
            "'kimi_direct'::character varying, "
            "'qwencloud_direct'::character varying",
        ),
        (
            "'openrouter'::text, 'bedrock'::text",
            "'openrouter'::text, 'bedrock'::text, 'kimi_direct'::text, "
            "'qwencloud_direct'::text",
        ),
        (
            "'openrouter', 'bedrock'",
            "'openrouter', 'bedrock', 'kimi_direct', 'qwencloud_direct'",
        ),
    )
    if "qwencloud_direct" in definition:
        raise RuntimeError("0029 reservation function already names qwencloud_direct")
    replacement_by_source = dict(replacements)
    matcher = re.compile(
        "|".join(
            re.escape(source)
            for source in sorted(replacement_by_source, key=len, reverse=True)
        )
    )
    matches = list(matcher.finditer(definition))
    if len(matches) != 2:
        raise RuntimeError(
            "0029 could not verify the two reservation backend predicates"
        )
    updated = matcher.sub(
        lambda match: replacement_by_source[match.group(0)],
        definition,
    )
    if updated.count("qwencloud_direct") != 2:
        raise RuntimeError("0029 reservation backend rewrite was not exact")
    return updated


def _extend_postgresql_reservation_function() -> None:
    """Extend the owner-authority reservation function without changing its body."""

    bind = op.get_bind()
    definition = bind.execute(
        sa.text(
            "SELECT pg_catalog.pg_get_functiondef("
            "'public.flavourbench_reserve_battle_budget(text)'::pg_catalog.regprocedure)"
        )
    ).scalar_one()
    updated = _rewrite_postgresql_reservation_function(definition)
    op.execute(updated)


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"unsupported database dialect for 0029: {dialect}")
    for table, name, expression in CONSTRAINTS:
        if dialect == "postgresql":
            op.drop_constraint(name, table, type_="check")
            op.create_check_constraint(name, table, expression)
        else:
            with op.batch_alter_table(table, recreate="always") as batch:
                batch.drop_constraint(name, type_="check")
                batch.create_check_constraint(name, expression)
    if dialect == "postgresql":
        _extend_postgresql_reservation_function()
    else:
        _restore_sqlite_budget_guards()


def downgrade() -> None:
    raise RuntimeError(
        "downgrade across research-release archive policy is prohibited; "
        "downgrade across direct-QwenCloud budget authority is prohibited"
    )
