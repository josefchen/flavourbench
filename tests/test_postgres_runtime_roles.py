from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql

ROLE_URLS = {
    "flavourbench_api": os.environ.get("FLAVOURBENCH_TEST_API_DATABASE_URL", ""),
    "flavourbench_worker": os.environ.get("FLAVOURBENCH_TEST_WORKER_DATABASE_URL", ""),
}


pytestmark = pytest.mark.skipif(
    not all(ROLE_URLS.values()),
    reason="dedicated PostgreSQL runtime-role URLs were not provided",
)


def _psycopg_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


@pytest.mark.parametrize(("expected_role", "database_url"), ROLE_URLS.items())
def test_runtime_role_cannot_bypass_evidence_seals(
    expected_role: str,
    database_url: str,
) -> None:
    url = _psycopg_url(database_url)
    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_user, rolsuper, rolcreatedb, rolcreaterole, rolreplication
                FROM pg_roles WHERE rolname = current_user
                """
            )
            identity = cursor.fetchone()
            assert identity == (expected_role, False, False, False, False)
            cursor.execute("SELECT has_schema_privilege(current_user, 'public', 'CREATE')")
            assert cursor.fetchone() == (False,)
            cursor.execute(
                "SELECT has_table_privilege(current_user, 'cost_events', 'UPDATE,DELETE,TRUNCATE')"
            )
            assert cursor.fetchone() == (False,)
            cursor.execute(
                """
                SELECT relname
                FROM pg_class
                WHERE relnamespace = 'public'::regnamespace
                  AND relkind IN ('r', 'p')
                  AND (
                    has_table_privilege(current_user, oid, 'DELETE')
                    OR has_table_privilege(current_user, oid, 'TRUNCATE')
                  )
                ORDER BY relname
                """
            )
            assert cursor.fetchall() == []
            append_only = (
                "epicure_releases",
                "generation_attempts",
                "votes",
                "admission_events",
                "cost_events",
                "bedrock_billing_crosschecks",
                "bedrock_billing_crosscheck_arms",
            )
            for table in append_only:
                cursor.execute(
                    "SELECT has_table_privilege(current_user, %s, 'UPDATE')",
                    (table,),
                )
                assert cursor.fetchone() == (False,)

            if expected_role == "flavourbench_api":
                expected = {
                    ("votes", "INSERT"): True,
                    ("epicure_releases", "INSERT"): True,
                    ("bedrock_billing_crosschecks", "INSERT"): False,
                    ("bedrock_billing_crosscheck_arms", "INSERT"): False,
                    ("generation_attempts", "INSERT"): False,
                    ("run_events", "UPDATE"): False,
                    ("incidents", "UPDATE"): False,
                }
            else:
                expected = {
                    ("votes", "INSERT"): False,
                    ("epicure_releases", "INSERT"): False,
                    ("bedrock_billing_crosschecks", "INSERT"): False,
                    ("generation_attempts", "INSERT"): True,
                    ("tasks", "UPDATE"): False,
                    ("season_models", "UPDATE"): False,
                    ("run_events", "UPDATE"): False,
                    ("incidents", "UPDATE"): False,
                }
            for (table, privilege), permitted in expected.items():
                cursor.execute(
                    "SELECT has_table_privilege(current_user, %s, %s)",
                    (table, privilege),
                )
                assert cursor.fetchone() == (permitted,)
            protected_columns = (
                ("seasons", "budget_used_micros"),
                ("seasons", "budget_reserved_micros"),
                ("season_provider_budgets", "budget_used_micros"),
                ("season_provider_budgets", "budget_reserved_micros"),
                ("provider_account_budgets", "budget_used_micros"),
                ("provider_account_budgets", "budget_reserved_micros"),
                ("controlled_runs", "budget_used_micros"),
                ("controlled_runs", "budget_reserved_micros"),
                ("battles", "reserved_cost_micros"),
                ("battles", "provider_reservations_json"),
            )
            for table, column in protected_columns:
                cursor.execute(
                    "SELECT has_column_privilege(current_user, %s, %s, 'UPDATE')",
                    (table, column),
                )
                assert cursor.fetchone() == (False,)
            if expected_role == "flavourbench_worker":
                cursor.execute(
                    "SELECT has_column_privilege(current_user, 'seasons', 'status', 'UPDATE')"
                )
                assert cursor.fetchone() == (False,)

            expected_functions = {
                "flavourbench_reserve_battle_budget(text)": (
                    expected_role == "flavourbench_api"
                ),
                "flavourbench_settle_battle_budget(text)": True,
                "flavourbench_apply_bedrock_billing_adjustment(text)": (
                    False
                ),
                "flavourbench_register_bedrock_billing_adjustment(text,jsonb)": (
                    expected_role == "flavourbench_api"
                ),
            }
            for signature, permitted in expected_functions.items():
                cursor.execute(
                    "SELECT has_function_privilege(current_user, %s, 'EXECUTE')",
                    (signature,),
                )
                assert cursor.fetchone() == (permitted,)
        connection.rollback()

    forbidden = (
        "ALTER TABLE battles DISABLE TRIGGER ALL",
        "DROP TRIGGER trg_battle_arm_link_guard ON battles",
        "DROP FUNCTION flavourbench_battle_arm_link_guard()",
        "TRUNCATE TABLE battles",
        "DROP TABLE battles",
        "CREATE TABLE public.flavourbench_role_escape_probe (id integer)",
        "SET session_replication_role = replica",
    )
    for statement in forbidden:
        with psycopg.connect(url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    cursor.execute(statement)
            connection.rollback()

    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            event_id = str(uuid.uuid4())
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO run_events (
                        id, entity_type, entity_id, event_type, payload_json, created_at
                    ) VALUES (%s, %s, %s, %s, '{}'::jsonb, CURRENT_TIMESTAMP)
                    """
                ),
                (event_id, "role_probe", expected_role, "runtime_dml_probe"),
            )
            with pytest.raises(psycopg.errors.RaiseException):
                cursor.execute(
                    "UPDATE run_events SET payload_json = '{\"redacted\": true}'::jsonb "
                    "WHERE id = %s",
                    (event_id,),
                )
        connection.rollback()
