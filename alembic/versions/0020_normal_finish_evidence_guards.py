"""Reject incomplete provider finishes at terminal and voting boundaries.

Revision ID: 0020_normal_finish_evidence_guards
Revises: 0019_commercial_authority_binding

The guards are prospective. Existing historical response arms remain immutable
and available for reliability analyses, but no newly completed arm or vote may
cross the evidence boundary with a truncation or length-limit finish reason.
"""

from __future__ import annotations

from alembic import op

revision = "0020_normal_finish_evidence_guards"
down_revision = "0019_commercial_authority_binding"
branch_labels = None
depends_on = None


ARM_TRIGGER = "trg_response_arm_normal_finish_guard"
ARM_FUNCTION = "flavourbench_response_arm_normal_finish_guard"
VOTE_TRIGGER = "trg_vote_normal_finish_guard"
VOTE_FUNCTION = "flavourbench_vote_normal_finish_guard"
NORMAL_FINISH_REASONS_SQL = "'completed', 'end_turn', 'stop', 'stop_sequence'"


def _create_postgresql_guards() -> None:
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

        DROP TRIGGER IF EXISTS {ARM_TRIGGER} ON public.response_arms;
        CREATE TRIGGER {ARM_TRIGGER}
        BEFORE INSERT OR UPDATE OF status, finish_reason ON public.response_arms
        FOR EACH ROW EXECUTE FUNCTION public.{ARM_FUNCTION}();

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

        DROP TRIGGER IF EXISTS {VOTE_TRIGGER} ON public.votes;
        CREATE TRIGGER {VOTE_TRIGGER}
        BEFORE INSERT ON public.votes
        FOR EACH ROW EXECUTE FUNCTION public.{VOTE_FUNCTION}();
        """
    )


def _create_sqlite_guards() -> None:
    op.execute(
        f"""
        CREATE TRIGGER {ARM_TRIGGER}_insert
        BEFORE INSERT ON response_arms FOR EACH ROW
        WHEN NEW.status = 'complete'
         AND lower(trim(COALESCE(NEW.finish_reason, '')))
             NOT IN ({NORMAL_FINISH_REASONS_SQL})
        BEGIN
            SELECT RAISE(
                ABORT,
                'complete response arm requires a normal provider finish reason'
            );
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_TRIGGER}_update
        BEFORE UPDATE OF status, finish_reason ON response_arms FOR EACH ROW
        WHEN NEW.status = 'complete'
         AND lower(trim(COALESCE(NEW.finish_reason, '')))
             NOT IN ({NORMAL_FINISH_REASONS_SQL})
        BEGIN
            SELECT RAISE(
                ABORT,
                'complete response arm requires a normal provider finish reason'
            );
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {VOTE_TRIGGER}_insert
        BEFORE INSERT ON votes FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1
            FROM battles AS battle
            JOIN response_arms AS left_arm
              ON left_arm.id = battle.left_arm_id
             AND left_arm.battle_id = battle.id
             AND left_arm.side = 'left'
             AND left_arm.status = 'complete'
             AND lower(trim(COALESCE(left_arm.finish_reason, '')))
                 IN ({NORMAL_FINISH_REASONS_SQL})
            JOIN response_arms AS right_arm
              ON right_arm.id = battle.right_arm_id
             AND right_arm.battle_id = battle.id
             AND right_arm.side = 'right'
             AND right_arm.status = 'complete'
             AND lower(trim(COALESCE(right_arm.finish_reason, '')))
                 IN ({NORMAL_FINISH_REASONS_SQL})
            WHERE battle.id = NEW.battle_id
              AND battle.status = 'complete'
        )
        BEGIN
            SELECT RAISE(
                ABORT,
                'vote requires two normally finished response arms'
            );
        END;
        """
    )


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        _create_postgresql_guards()
    elif dialect == "sqlite":
        _create_sqlite_guards()
    else:
        raise RuntimeError(f"unsupported database dialect for 0020: {dialect}")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {VOTE_TRIGGER} ON public.votes")
        op.execute(f"DROP FUNCTION IF EXISTS public.{VOTE_FUNCTION}()")
        op.execute(f"DROP TRIGGER IF EXISTS {ARM_TRIGGER} ON public.response_arms")
        op.execute(f"DROP FUNCTION IF EXISTS public.{ARM_FUNCTION}()")
    elif dialect == "sqlite":
        op.execute(f"DROP TRIGGER IF EXISTS {VOTE_TRIGGER}_insert")
        op.execute(f"DROP TRIGGER IF EXISTS {ARM_TRIGGER}_update")
        op.execute(f"DROP TRIGGER IF EXISTS {ARM_TRIGGER}_insert")
    else:
        raise RuntimeError(f"unsupported database dialect for 0020: {dialect}")
