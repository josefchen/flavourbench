"""Use PostgreSQL's SQL COALESCE expression in finish guards.

Revision ID: 0022_postgresql_finish_guard_coalesce
Revises: 0021_postgresql_finish_guard_text_casts

``COALESCE`` is SQL syntax rather than a schema-qualified PostgreSQL function.
The prior guard was created successfully but failed when the trigger first ran.
This forward migration replaces both functions without changing their policy.
"""

from __future__ import annotations

from alembic import op

revision = "0022_postgresql_finish_guard_coalesce"
down_revision = "0021_postgresql_finish_guard_text_casts"
branch_labels = None
depends_on = None


ARM_FUNCTION = "flavourbench_response_arm_normal_finish_guard"
VOTE_FUNCTION = "flavourbench_vote_normal_finish_guard"
NORMAL_FINISH_REASONS_SQL = "'completed', 'end_turn', 'stop', 'stop_sequence'"


def _replace_postgresql_functions() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{ARM_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            IF NEW.status = 'complete'
               AND pg_catalog.lower(pg_catalog.btrim(
                    COALESCE(NEW.finish_reason::text, ''::text)
               )) NOT IN ({NORMAL_FINISH_REASONS_SQL}) THEN
                RAISE EXCEPTION
                    'complete response arm requires a normal provider finish reason';
            END IF;
            RETURN NEW;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.{VOTE_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM public.battles AS battle
                JOIN public.response_arms AS left_arm
                  ON left_arm.id = battle.left_arm_id
                 AND left_arm.battle_id = battle.id
                 AND left_arm.side = 'left'
                 AND left_arm.status = 'complete'
                 AND pg_catalog.lower(pg_catalog.btrim(
                        COALESCE(left_arm.finish_reason::text, ''::text)
                     )) IN ({NORMAL_FINISH_REASONS_SQL})
                JOIN public.response_arms AS right_arm
                  ON right_arm.id = battle.right_arm_id
                 AND right_arm.battle_id = battle.id
                 AND right_arm.side = 'right'
                 AND right_arm.status = 'complete'
                 AND pg_catalog.lower(pg_catalog.btrim(
                        COALESCE(right_arm.finish_reason::text, ''::text)
                     )) IN ({NORMAL_FINISH_REASONS_SQL})
                WHERE battle.id = NEW.battle_id
                  AND battle.status = 'complete'
            ) THEN
                RAISE EXCEPTION
                    'vote requires two normally finished response arms';
            END IF;
            RETURN NEW;
        END;
        $$;
        """
    )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        _replace_postgresql_functions()
    elif dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0022: {dialect}")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        # Keep the executable guard when crossing the corrective boundary.
        _replace_postgresql_functions()
    elif dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0022: {dialect}")
