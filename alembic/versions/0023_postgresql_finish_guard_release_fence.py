"""Fence the executable PostgreSQL normal-finish guards from unsafe rollback.

Revision ID: 0023_postgresql_finish_guard_release_fence
Revises: 0022_postgresql_finish_guard_coalesce

Revision 0022 corrected the PostgreSQL expression and is already deployed.
This append-only release fence reapplies the canonical functions and trigger
bindings, verifies their live metadata, and prevents PostgreSQL downgrade into
the older chain that can reinstall revision 0021's non-executable functions.
"""

from __future__ import annotations

from alembic import op

revision = "0023_postgresql_finish_guard_release_fence"
down_revision = "0022_postgresql_finish_guard_coalesce"
branch_labels = None
depends_on = None


ARM_TRIGGER = "trg_response_arm_normal_finish_guard"
ARM_FUNCTION = "flavourbench_response_arm_normal_finish_guard"
VOTE_TRIGGER = "trg_vote_normal_finish_guard"
VOTE_FUNCTION = "flavourbench_vote_normal_finish_guard"
NORMAL_FINISH_REASONS_SQL = "'completed', 'end_turn', 'stop', 'stop_sequence'"


def _replace_and_verify_postgresql_guards() -> None:
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

        DROP TRIGGER IF EXISTS {ARM_TRIGGER} ON public.response_arms;
        CREATE TRIGGER {ARM_TRIGGER}
        BEFORE INSERT OR UPDATE OF status, finish_reason ON public.response_arms
        FOR EACH ROW EXECUTE FUNCTION public.{ARM_FUNCTION}();

        DROP TRIGGER IF EXISTS {VOTE_TRIGGER} ON public.votes;
        CREATE TRIGGER {VOTE_TRIGGER}
        BEFORE INSERT ON public.votes
        FOR EACH ROW EXECUTE FUNCTION public.{VOTE_FUNCTION}();

        DO $$
        DECLARE
            function_count integer;
            trigger_count integer;
        BEGIN
            SELECT pg_catalog.count(*) INTO function_count
            FROM pg_catalog.pg_proc AS p
            JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
            JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
            WHERE n.nspname = 'public'
              AND p.proname IN ('{ARM_FUNCTION}', '{VOTE_FUNCTION}')
              AND pg_catalog.pg_get_userbyid(p.proowner) = 'flavourbench_owner'
              AND p.prosecdef IS FALSE
              AND l.lanname = 'plpgsql'
              AND pg_catalog.pg_get_function_identity_arguments(p.oid) = ''
              AND pg_catalog.pg_get_function_result(p.oid) = 'trigger'
              AND p.proconfig = ARRAY['search_path=pg_catalog, public']::text[]
              AND pg_catalog.strpos(
                    pg_catalog.lower(p.prosrc),
                    'pg_catalog.coalesce'
                  ) = 0;
            IF function_count <> 2 THEN
                RAISE EXCEPTION
                    '0023 normal-finish guard function verification failed';
            END IF;

            SELECT pg_catalog.count(*) INTO trigger_count
            FROM pg_catalog.pg_trigger AS t
            JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            JOIN pg_catalog.pg_proc AS p ON p.oid = t.tgfoid
            JOIN pg_catalog.pg_namespace AS pn ON pn.oid = p.pronamespace
            WHERE n.nspname = 'public'
              AND pn.nspname = 'public'
              AND NOT t.tgisinternal
              AND t.tgenabled = 'O'
              AND (
                    (
                        t.tgname = '{ARM_TRIGGER}'
                        AND c.relname = 'response_arms'
                        AND p.proname = '{ARM_FUNCTION}'
                        AND t.tgtype = 23
                    ) OR (
                        t.tgname = '{VOTE_TRIGGER}'
                        AND c.relname = 'votes'
                        AND p.proname = '{VOTE_FUNCTION}'
                        AND t.tgtype = 7
                    )
              );
            IF trigger_count <> 2 THEN
                RAISE EXCEPTION
                    '0023 normal-finish guard trigger verification failed';
            END IF;
        END;
        $$;
        """
    )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        _replace_and_verify_postgresql_guards()
    elif dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0023: {dialect}")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        raise RuntimeError(
            "PostgreSQL downgrade across 0023 is prohibited because the older "
            "migration chain can reinstall non-executable normal-finish guards"
        )
    elif dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0023: {dialect}")
