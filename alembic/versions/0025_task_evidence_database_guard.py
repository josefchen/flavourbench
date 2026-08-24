"""Enforce task-evidence append-only semantics in the database.

Revision ID: 0025_task_evidence_database_guard
Revises: 0024_task_evidence_registry
"""

from __future__ import annotations

from alembic import op

revision = "0025_task_evidence_database_guard"
down_revision = "0024_task_evidence_registry"
branch_labels = None
depends_on = None

FUNCTION_NAME = "flavourbench_task_evidence_append_only"
TRIGGER_NAME = "trg_task_evidence_artifacts_append_only"
SQLITE_UPDATE_TRIGGER = "trg_task_evidence_artifacts_no_update"
SQLITE_DELETE_TRIGGER = "trg_task_evidence_artifacts_no_delete"


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute(
            f"""
            CREATE OR REPLACE FUNCTION public.{FUNCTION_NAME}()
            RETURNS trigger
            LANGUAGE plpgsql
            SECURITY INVOKER
            SET search_path = pg_catalog, public
            AS $$
            BEGIN
                RAISE EXCEPTION 'task-evidence artifacts are append-only';
            END;
            $$;

            DROP TRIGGER IF EXISTS {TRIGGER_NAME}
                ON public.task_evidence_artifacts;
            CREATE TRIGGER {TRIGGER_NAME}
            BEFORE UPDATE OR DELETE ON public.task_evidence_artifacts
            FOR EACH ROW EXECUTE FUNCTION public.{FUNCTION_NAME}();

            DO $$
            DECLARE trigger_count integer;
            BEGIN
                SELECT pg_catalog.count(*) INTO trigger_count
                FROM pg_catalog.pg_trigger AS trigger
                JOIN pg_catalog.pg_class AS relation
                  ON relation.oid = trigger.tgrelid
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                JOIN pg_catalog.pg_proc AS function
                  ON function.oid = trigger.tgfoid
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'task_evidence_artifacts'
                  AND trigger.tgname = '{TRIGGER_NAME}'
                  AND function.proname = '{FUNCTION_NAME}'
                  AND trigger.tgenabled = 'O'
                  AND NOT trigger.tgisinternal;
                IF trigger_count <> 1 THEN
                    RAISE EXCEPTION 'task-evidence append-only trigger verification failed';
                END IF;
            END;
            $$;
            """
        )
    elif dialect == "sqlite":
        op.execute(
            f"""
            CREATE TRIGGER {SQLITE_UPDATE_TRIGGER}
            BEFORE UPDATE ON task_evidence_artifacts
            BEGIN
                SELECT RAISE(ABORT, 'task-evidence artifacts are append-only');
            END
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {SQLITE_DELETE_TRIGGER}
            BEFORE DELETE ON task_evidence_artifacts
            BEGIN
                SELECT RAISE(ABORT, 'task-evidence artifacts are append-only');
            END
            """
        )
    else:
        raise RuntimeError(f"unsupported database dialect for 0025: {dialect}")


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        raise RuntimeError(
            "PostgreSQL downgrade across the task-evidence append-only guard is prohibited"
        )
    if dialect != "sqlite":
        raise RuntimeError(f"unsupported database dialect for 0025: {dialect}")
    op.execute(f"DROP TRIGGER IF EXISTS {SQLITE_UPDATE_TRIGGER}")
    op.execute(f"DROP TRIGGER IF EXISTS {SQLITE_DELETE_TRIGGER}")
