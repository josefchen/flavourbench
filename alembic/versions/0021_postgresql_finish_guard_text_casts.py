"""Make finish-guard text coercion explicit on PostgreSQL.

Revision ID: 0021_postgresql_finish_guard_text_casts
Revises: 0020_normal_finish_evidence_guards

PostgreSQL could not resolve pg_catalog.coalesce(varchar, unknown) inside the
restricted-search-path trigger function. Explicit text casts preserve the
guard while making INSERT and vote checks executable.
"""

from __future__ import annotations

from alembic import op

revision = "0021_postgresql_finish_guard_text_casts"
down_revision = "0020_normal_finish_evidence_guards"
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
                    pg_catalog.coalesce(NEW.finish_reason::text, ''::text)
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
                        pg_catalog.coalesce(left_arm.finish_reason::text, ''::text)
                     )) IN ({NORMAL_FINISH_REASONS_SQL})
                JOIN public.response_arms AS right_arm
                  ON right_arm.id = battle.right_arm_id
                 AND right_arm.battle_id = battle.id
                 AND right_arm.side = 'right'
                 AND right_arm.status = 'complete'
                 AND pg_catalog.lower(pg_catalog.btrim(
                        pg_catalog.coalesce(right_arm.finish_reason::text, ''::text)
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
        raise RuntimeError(f"unsupported database dialect for 0021: {dialect}")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        _replace_postgresql_functions()
    elif dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0021: {dialect}")
