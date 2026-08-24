#!/usr/bin/env python3
"""Upgrade the one frozen PostgreSQL 0001 database through a sealed bridge.

Migration 0010 installed one trigger function on three tables.  Its original
PL/pgSQL body refers directly to ``OLD.manifest_sha256`` before first checking
which table fired the trigger.  PostgreSQL therefore raises while migration
0013 backfills ``response_arms``.  Migration 0016 already contains the correct
JSON-safe replacement, but the database cannot reach it without a one-time
bridge.  The same staged rehearsal identified a superseded 0009 arm trigger
that 0016 removes and four quarantined zero-cost mock arms whose new accounting
fields require the canonical mock values before the 0014 integrity preflight.

This script is deliberately specific to the content-addressed frozen backup.
It refuses any other starting revision, schema shape, row inventory, SQL body,
or Alembic head.  Historical migrations remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import sql
from sqlalchemy.engine import make_url

SERVICE_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SQL_PATH = SERVICE_ROOT / "scripts" / "sql" / "legacy_0001_execution_contract_bridge.sql"
SUPERSEDED_TRIGGER_SQL_PATH = (
    SERVICE_ROOT / "scripts" / "sql" / "legacy_0001_superseded_arm_trigger_bridge.sql"
)
MOCK_ACCOUNTING_SQL_PATH = (
    SERVICE_ROOT / "scripts" / "sql" / "legacy_0001_mock_accounting_bridge.sql"
)

SOURCE_BACKUP_SHA256 = "f81b637903a25514a14cefb7363dd9e32b2b4659e65e10b43e25ca9ec8a5bd08"
BRIDGE_SQL_SHA256 = "427f89db5dc8bd779e1d39e2176e4b945d28d214880eb4f10cb1f1f66f3b8b03"
SUPERSEDED_TRIGGER_SQL_SHA256 = "a930d775fd2577fedae774db00652af0085facb8e3ba32183719155391108e4c"
MOCK_ACCOUNTING_SQL_SHA256 = "4ac1f2b483f08e7ce07396da4c86fb64598cbf0306b0c89050210e2985422251"
START_SCHEMA_SHA256 = "4972baa44abad6c2f4d1454291daa6383f23c2efc3fbbd0f3050e3581238031c"
START_CONTENT_ROOT_SHA256 = "6d2ef9476aac0dff33c1d06ce982bc3c6653b47f89fa7b9b291e31a1aeb8f1aa"
UNSAFE_STAGE_BODY_SHA256 = "9f542f9314986b963754b6b65555bd7684a63a2a92bc52ca3fae3c5962df8353"
SAFE_BODY_SHA256 = "f6f2aa7d2b6e6c90a63f0f8a513119cf35717945d3d0e5d7c96fdeb1f23d8123"
SUPERSEDED_ARM_FUNCTION_BODY_SHA256 = (
    "e3aa2f44bc777141462a284674a86a01ec5cdfea914428394f9afdbf31ea3829"
)
MOCK_ACCOUNTING_BEFORE_SHA256 = "a5ceb4113de598a9f441bbb94c3c6cf12c93e1b8ccc8042b3e59350f2b1a2575"
MOCK_ACCOUNTING_AFTER_SHA256 = "ccd088c10f373cff8a83d335c32f7766e5c043a6a9064fb4e23dda2ca8b6b976"
MOCK_FULL_ROWS_BEFORE_SHA256 = "27351d55326e4578b702c03a22e0a98f9b5ff95b6bb831fe910d276567e1aebf"
MOCK_FULL_ROWS_AFTER_SHA256 = "94fcf230d8e72214c99faec0d9bda592591ea28cd66cccb86a5e82cb878bc6e8"

START_REVISION = "0001_initial"
STAGE_REVISION = "0012_snapshot_integrity"
LOCK_NAME = "flavourbench:legacy-0001-upgrade-bridge:v2"
FUNCTION_NAME = "flavourbench_0010_execution_contract_immutable"
SUPERSEDED_ARM_FUNCTION_NAME = "flavourbench_prevent_response_arm_contract_update"

MOCK_ARM_IDS = (
    "18818502-ed30-4a3a-970e-18813c8d8b0d",
    "65f4ebad-591c-4268-adbc-e4f6dffb0bb0",
    "8829771a-9268-4865-a9a6-889a502a2842",
    "afa542d5-5278-417c-ba1a-cd5d23646047",
)

BASELINE_ROW_COUNTS = {
    "admission_events": 2,
    "alembic_version": 1,
    "battles": 2,
    "catalog_models": 12,
    "cost_events": 6,
    "expert_reviewers": 0,
    "incidents": 0,
    "jobs": 2,
    "leaderboard_snapshots": 0,
    "response_arms": 4,
    "run_events": 10,
    "season_models": 12,
    "seasons": 1,
    "tasks": 120,
    "tool_calls": 4,
    "validator_results": 16,
    "votes": 2,
}

STAGE_TRIGGER_BINDINGS = {
    "trg_battle_provider_reservation_immutable": ("battles", FUNCTION_NAME, 19),
    "trg_response_arm_backend_immutable": ("response_arms", FUNCTION_NAME, 19),
    "trg_season_model_backend_immutable": ("season_models", FUNCTION_NAME, 19),
}

SUPERSEDED_TRIGGER_BINDING = {
    "trg_response_arm_contract_immutable": (
        "response_arms",
        SUPERSEDED_ARM_FUNCTION_NAME,
        19,
    )
}

# These are the application-readiness integrity triggers most relevant to this
# upgrade.  The final check also rejects every disabled public user trigger.
FINAL_REQUIRED_TRIGGER_BINDINGS = {
    **STAGE_TRIGGER_BINDINGS,
    "trg_response_arm_normal_finish_guard": (
        "response_arms",
        "flavourbench_response_arm_normal_finish_guard",
        23,
    ),
    "trg_vote_normal_finish_guard": (
        "votes",
        "flavourbench_vote_normal_finish_guard",
        7,
    ),
    "trg_budget_counter_authority_seasons": (
        "seasons",
        "flavourbench_budget_counter_write_guard",
        23,
    ),
    "trg_budget_counter_authority_season_provider_budgets": (
        "season_provider_budgets",
        "flavourbench_budget_counter_write_guard",
        23,
    ),
    "trg_budget_counter_authority_provider_account_budgets": (
        "provider_account_budgets",
        "flavourbench_budget_counter_write_guard",
        23,
    ),
    "trg_budget_counter_authority_controlled_runs": (
        "controlled_runs",
        "flavourbench_budget_counter_write_guard",
        23,
    ),
    "trg_battle_reservation_authority": (
        "battles",
        "flavourbench_battle_reservation_write_guard",
        23,
    ),
    "trg_cost_event_authority": (
        "cost_events",
        "flavourbench_cost_event_authority_guard",
        7,
    ),
    "trg_bedrock_membership_seal": (
        "bedrock_billing_crosscheck_arms",
        "flavourbench_bedrock_membership_seal_guard",
        7,
    ),
    "trg_generation_attempt_arm_authority": (
        "generation_attempts",
        "flavourbench_generation_attempt_arm_guard",
        7,
    ),
    "trg_reviewer_identity_bindings_append_only_v1": (
        "reviewer_identity_bindings",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_qualification_evidence_append_only_v1": (
        "reviewer_qualification_evidence",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_calibration_sets_append_only_v1": (
        "reviewer_calibration_sets",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_calibration_ballots_append_only_v1": (
        "reviewer_calibration_ballots",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_family_admissions_append_only_v1": (
        "reviewer_family_admissions",
        "flavourbench_reviewer_evidence_append_only_v1",
        27,
    ),
    "trg_reviewer_access_credentials_lifecycle_v1": (
        "reviewer_access_credentials",
        "flavourbench_reviewer_credential_lifecycle_v1",
        27,
    ),
    "trg_reviewer_family_admissions_guard_v1": (
        "reviewer_family_admissions",
        "flavourbench_reviewer_family_admission_guard_v1",
        7,
    ),
    "trg_votes_verified_expert_provenance_v1": (
        "votes",
        "flavourbench_verified_expert_vote_guard_v1",
        23,
    ),
    "trg_task_validation_audit_authorizations_append_only_v1": (
        "task_validation_audit_authorizations",
        "flavourbench_task_validation_append_only_v1",
        27,
    ),
    "trg_task_validation_campaign_events_append_only_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_append_only_v1",
        27,
    ),
    "trg_task_validation_campaign_events_authority_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_event_guard_v1",
        7,
    ),
    "trg_task_validation_audit_authorizations_replay_v1": (
        "task_validation_audit_authorizations",
        "flavourbench_task_validation_audit_replay_guard_v1",
        7,
    ),
    "trg_task_validation_campaign_events_replay_v1": (
        "task_validation_campaign_events",
        "flavourbench_task_validation_event_replay_guard_v1",
        7,
    ),
}


class BridgeError(RuntimeError):
    """A fail-closed bridge contract check failed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _normalize_sql(value: object) -> str:
    return " ".join(str(value).split())


def _database_url() -> tuple[str, str]:
    raw = os.environ.get("FLAVOURBENCH_DATABASE_URL", "")
    if not raw:
        raise BridgeError("FLAVOURBENCH_DATABASE_URL is required")
    parsed = make_url(raw)
    if parsed.get_backend_name() != "postgresql" or parsed.database is None:
        raise BridgeError("the legacy bridge only supports a named PostgreSQL database")
    psycopg_url = parsed.set(drivername="postgresql").render_as_string(hide_password=False)
    return psycopg_url, parsed.database


def _current_revision(connection: psycopg.Connection[Any]) -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT version_num FROM public.alembic_version")
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise BridgeError("alembic_version must contain exactly one row")
    return str(rows[0][0])


def _row_counts(connection: psycopg.Connection[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with connection.cursor() as cursor:
        for table_name in BASELINE_ROW_COUNTS:
            cursor.execute(
                sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table_name))
            )
            counts[table_name] = int(cursor.fetchone()[0])
    return counts


def _table_columns(connection: psycopg.Connection[Any], table_name: str) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT attribute.attname
              FROM pg_catalog.pg_attribute AS attribute
              JOIN pg_catalog.pg_class AS table_record
                ON table_record.oid = attribute.attrelid
              JOIN pg_catalog.pg_namespace AS namespace_record
                ON namespace_record.oid = table_record.relnamespace
             WHERE namespace_record.nspname = 'public'
               AND table_record.relname = %s
               AND table_record.relkind IN ('r', 'p')
               AND attribute.attnum > 0
               AND NOT attribute.attisdropped
             ORDER BY attribute.attnum
            """,
            (table_name,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _primary_key_columns(connection: psycopg.Connection[Any], table_name: str) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT attribute.attname
              FROM pg_catalog.pg_constraint AS constraint_record
              CROSS JOIN LATERAL
                   unnest(constraint_record.conkey)
                   WITH ORDINALITY AS key_column(attnum, ordinal)
              JOIN pg_catalog.pg_class AS table_record
                ON table_record.oid = constraint_record.conrelid
              JOIN pg_catalog.pg_namespace AS namespace_record
                ON namespace_record.oid = table_record.relnamespace
              JOIN pg_catalog.pg_attribute AS attribute
                ON attribute.attrelid = table_record.oid
               AND attribute.attnum = key_column.attnum
             WHERE namespace_record.nspname = 'public'
               AND table_record.relname = %s
               AND constraint_record.contype = 'p'
             ORDER BY key_column.ordinal
            """,
            (table_name,),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _baseline_identity_commitments(
    connection: psycopg.Connection[Any],
) -> dict[str, dict[str, object]]:
    """Commit every frozen primary-key set so equal counts cannot hide replacement."""

    commitments: dict[str, dict[str, object]] = {}
    with connection.cursor() as cursor:
        for table_name in BASELINE_ROW_COUNTS:
            if table_name == "alembic_version":
                continue
            primary_key_columns = _primary_key_columns(connection, table_name)
            if not primary_key_columns:
                raise BridgeError(f"baseline table has no primary key: {table_name}")
            selected = sql.SQL(", ").join(
                sql.Identifier(column_name) for column_name in primary_key_columns
            )
            ordering = sql.SQL(", ").join(
                sql.Identifier(column_name) for column_name in primary_key_columns
            )
            cursor.execute(
                sql.SQL("SELECT {} FROM public.{} ORDER BY {}").format(
                    selected,
                    sql.Identifier(table_name),
                    ordering,
                )
            )
            identities = [list(row) for row in cursor.fetchall()]
            commitments[table_name] = {
                "primary_key_columns": primary_key_columns,
                "count": len(identities),
                "identity_sha256": _sha256_bytes(_canonical_bytes(identities)),
            }
    return commitments


def _baseline_content_commitments(
    connection: psycopg.Connection[Any],
    *,
    baseline_columns: dict[str, list[str]] | None = None,
) -> tuple[str, dict[str, dict[str, object]], dict[str, list[str]]]:
    """Hash every original column of every original data row in PK order."""

    columns_by_table = (
        {name: list(columns) for name, columns in baseline_columns.items()}
        if baseline_columns is not None
        else {
            table_name: _table_columns(connection, table_name)
            for table_name in BASELINE_ROW_COUNTS
            if table_name != "alembic_version"
        }
    )
    if set(columns_by_table) != set(BASELINE_ROW_COUNTS) - {"alembic_version"}:
        raise BridgeError("the baseline content projection does not cover all data tables")

    commitments: dict[str, dict[str, object]] = {}
    with connection.cursor() as cursor:
        for table_name in sorted(columns_by_table):
            columns = columns_by_table[table_name]
            if not columns or not set(columns).issubset(
                set(_table_columns(connection, table_name))
            ):
                raise BridgeError(f"baseline columns disappeared from {table_name}")
            primary_key_columns = _primary_key_columns(connection, table_name)
            if not primary_key_columns or not set(primary_key_columns).issubset(columns):
                raise BridgeError(f"baseline primary key is not projectable: {table_name}")
            selected = sql.SQL(", ").join(sql.Identifier(column_name) for column_name in columns)
            ordering = sql.SQL(", ").join(
                sql.Identifier(column_name) for column_name in primary_key_columns
            )
            cursor.execute(
                sql.SQL(
                    "SELECT pg_catalog.to_jsonb(projected) FROM "
                    "(SELECT {} FROM public.{} ORDER BY {}) AS projected"
                ).format(selected, sql.Identifier(table_name), ordering)
            )
            rows = [dict(row[0]) for row in cursor.fetchall()]
            row_commitments = [
                {
                    "primary_key": {
                        column_name: row[column_name] for column_name in primary_key_columns
                    },
                    "row_sha256": _sha256_bytes(_canonical_bytes(row)),
                }
                for row in rows
            ]
            commitments[table_name] = {
                "columns": columns,
                "primary_key_columns": primary_key_columns,
                "row_count": len(rows),
                "row_commitments": row_commitments,
                "table_sha256": _sha256_bytes(_canonical_bytes(rows)),
            }
    return (
        _sha256_bytes(_canonical_bytes(commitments)),
        commitments,
        columns_by_table,
    )


def _mock_accounting_commitment(
    connection: psycopg.Connection[Any],
) -> tuple[str, dict[str, object]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT arm.id, arm.battle_id, arm.status, arm.provider_slug,
                   arm.actual_provider_slug, arm.actual_model_id,
                   arm.generation_id, arm.cost_micros, arm.cost_reconciled,
                   arm.cost_accounting_basis,
                   arm.billing_reconciliation_status,
                   battle.run_class, battle.rank_eligible, battle.data_stratum,
                   season.official, season.status, model.model_id,
                   receipt.id, receipt.kind, receipt.amount_micros,
                   receipt.provider, receipt.generation_id,
                   receipt.accounting_json
              FROM public.response_arms AS arm
              JOIN public.battles AS battle ON battle.id = arm.battle_id
              JOIN public.seasons AS season ON season.id = battle.season_id
              JOIN public.catalog_models AS model ON model.model_id = arm.model_id
              JOIN public.cost_events AS receipt
                ON receipt.arm_id = arm.id AND receipt.kind = 'actual'
             WHERE arm.id = ANY(%s)
             ORDER BY arm.id, receipt.id
            """,
            (list(MOCK_ARM_IDS),),
        )
        rows = [list(row) for row in cursor.fetchall()]
    digest = _sha256_bytes(_canonical_bytes(rows))
    if len(rows) != 4 or {str(row[0]) for row in rows} != set(MOCK_ARM_IDS):
        raise BridgeError("the frozen mock accounting projection is not cardinality four")
    for row in rows:
        if (
            row[2] != "complete"
            or row[3] != "mock"
            or row[4] != "mock"
            or not str(row[5]).startswith("flavourbench/mock-")
            or row[6] != f"mock-{row[0]}"
            or row[7] != 0
            or row[8] is not True
            or row[11] != "exploratory"
            or row[12] is not False
            or row[13] != "development"
            or row[14] is not False
            or row[15] != "draft"
            or row[16] != row[5]
            or row[18] != "actual"
            or row[19] != 0
            or row[20] != "mock"
            or row[21] != row[6]
            or not isinstance(row[22], dict)
            or row[22].get("reconciled") is not True
        ):
            raise BridgeError("a frozen mock row lost its quarantined zero-cost identity")
    summary = {
        "cardinality": len(rows),
        "arm_ids": sorted(str(row[0]) for row in rows),
        "arm_cost_micros_sum": sum(int(row[7]) for row in rows),
        "receipt_amount_micros_sum": sum(int(row[19]) for row in rows),
        "provider": "mock",
        "run_class": "exploratory",
        "data_stratum": "development",
        "rank_eligible": False,
        "season_official": False,
    }
    return digest, summary


def _mock_full_row_commitment(
    connection: psycopg.Connection[Any],
) -> tuple[str, dict[str, dict[str, object]]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT arm.id, pg_catalog.to_jsonb(arm)
              FROM public.response_arms AS arm
             WHERE arm.id = ANY(%s)
             ORDER BY arm.id
            """,
            (list(MOCK_ARM_IDS),),
        )
        rows = [(str(row[0]), dict(row[1])) for row in cursor.fetchall()]
    if len(rows) != 4 or {arm_id for arm_id, _ in rows} != set(MOCK_ARM_IDS):
        raise BridgeError("the full mock-row projection is not cardinality four")
    full_rows = {arm_id: row for arm_id, row in rows}
    return _sha256_bytes(_canonical_bytes(rows)), full_rows


def _mock_field_diff(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if set(before) != set(MOCK_ARM_IDS) or set(after) != set(MOCK_ARM_IDS):
        raise BridgeError("mock full-row diff does not cover the exact frozen arm set")
    differences: list[dict[str, object]] = []
    for arm_id in sorted(MOCK_ARM_IDS):
        before_row = before[arm_id]
        after_row = after[arm_id]
        if set(before_row) != set(after_row):
            raise BridgeError("mock normalization changed the staged response-arm schema")
        changes = [
            {
                "field": field_name,
                "before": before_row[field_name],
                "after": after_row[field_name],
            }
            for field_name in sorted(before_row)
            if before_row[field_name] != after_row[field_name]
        ]
        differences.append({"arm_id": arm_id, "changes": changes})

    expected_changes = [
        {
            "field": "billing_reconciliation_status",
            "before": "unrecorded",
            "after": "not_applicable",
        },
        {
            "field": "cost_accounting_basis",
            "before": "unrecorded",
            "after": "mock_fixture",
        },
    ]
    if any(item["changes"] != expected_changes for item in differences):
        raise BridgeError("mock normalization changed a field outside the exact allowance")
    return differences


def _start_schema_sha256(connection: psycopg.Connection[Any]) -> tuple[str, dict[str, int]]:
    queries = {
        "tables": """
            SELECT c.relname, c.relkind
              FROM pg_catalog.pg_class AS c
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
             ORDER BY c.relname
        """,
        "columns": """
            SELECT c.relname, a.attnum, a.attname,
                   pg_catalog.format_type(a.atttypid, a.atttypmod),
                   a.attnotnull, a.attidentity, a.attgenerated,
                   COALESCE(pg_catalog.pg_get_expr(ad.adbin, ad.adrelid), '')
              FROM pg_catalog.pg_class AS c
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
              JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid
              LEFT JOIN pg_catalog.pg_attrdef AS ad
                ON ad.adrelid = c.oid AND ad.adnum = a.attnum
             WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
               AND a.attnum > 0 AND NOT a.attisdropped
             ORDER BY c.relname, a.attnum
        """,
        "constraints": """
            SELECT c.relname, con.conname, con.contype,
                   pg_catalog.pg_get_constraintdef(con.oid, true)
              FROM pg_catalog.pg_constraint AS con
              JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public'
             ORDER BY c.relname, con.conname
        """,
        "indexes": """
            SELECT tablename, indexname, indexdef
              FROM pg_catalog.pg_indexes
             WHERE schemaname = 'public'
             ORDER BY tablename, indexname
        """,
    }
    inventory: dict[str, list[list[object]]] = {}
    sizes: dict[str, int] = {}
    with connection.cursor() as cursor:
        for label, query in queries.items():
            cursor.execute(query)
            inventory[label] = [list(row) for row in cursor.fetchall()]
            sizes[label] = len(inventory[label])
    return _sha256_bytes(_canonical_bytes(inventory)), sizes


def _initial_object_preflight(connection: psycopg.Connection[Any]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*)
              FROM pg_catalog.pg_trigger AS t
              JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND NOT t.tgisinternal
            """
        )
        trigger_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            SELECT count(*)
              FROM pg_catalog.pg_proc AS p
              JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
             WHERE n.nspname = 'public' AND p.proname LIKE 'flavourbench%'
            """
        )
        function_count = int(cursor.fetchone()[0])
        cursor.execute(
            """
            SELECT c.relname
              FROM pg_catalog.pg_class AS c
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
               AND pg_catalog.pg_get_userbyid(c.relowner) <> current_user
             ORDER BY c.relname
            """
        )
        foreign_owned_tables = [str(row[0]) for row in cursor.fetchall()]
    if trigger_count != 0 or function_count != 0:
        raise BridgeError(
            "the 0001 source must have zero public user triggers and FlavourBench functions"
        )
    if foreign_owned_tables:
        raise BridgeError(
            "the migration role does not own every starting table: "
            + ", ".join(foreign_owned_tables)
        )


def _function_identity(
    connection: psycopg.Connection[Any],
    function_name: str = FUNCTION_NAME,
) -> dict[str, object]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.oid, p.proname,
                   pg_catalog.pg_get_function_identity_arguments(p.oid),
                   pg_catalog.pg_get_function_result(p.oid),
                   l.lanname, p.prosecdef,
                   COALESCE(pg_catalog.array_to_string(p.proconfig, ','), ''),
                   pg_catalog.pg_get_userbyid(p.proowner), p.prosrc
              FROM pg_catalog.pg_proc AS p
              JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
              JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
             WHERE n.nspname = 'public' AND p.proname = %s
            """,
            (function_name,),
        )
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise BridgeError(f"expected exactly one public {function_name}() function")
    row = rows[0]
    identity = {
        "oid": int(row[0]),
        "name": str(row[1]),
        "arguments": str(row[2]),
        "result": str(row[3]),
        "language": str(row[4]),
        "security_definer": bool(row[5]),
        "config": str(row[6]).replace('"', ""),
        "owner": str(row[7]),
        "body_sha256": _sha256_bytes(_normalize_sql(row[8]).encode()),
    }
    if (
        identity["arguments"] != ""
        or identity["result"] != "trigger"
        or identity["language"] != "plpgsql"
        or identity["security_definer"] is not False
    ):
        raise BridgeError("the execution-contract function identity is not the sealed trigger")
    return identity


def _public_trigger_inventory(
    connection: psycopg.Connection[Any],
) -> dict[str, dict[str, object]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT t.oid, t.tgname, c.relname, p.oid, p.proname,
                   t.tgtype, t.tgenabled,
                   pg_catalog.pg_get_triggerdef(t.oid, false)
              FROM pg_catalog.pg_trigger AS t
              JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
              JOIN pg_catalog.pg_proc AS p ON p.oid = t.tgfoid
              JOIN pg_catalog.pg_namespace AS pn ON pn.oid = p.pronamespace
             WHERE n.nspname = 'public' AND pn.nspname = 'public'
               AND NOT t.tgisinternal
             ORDER BY t.tgname
            """
        )
        rows = cursor.fetchall()
    return {
        str(row[1]): {
            "oid": int(row[0]),
            "table": str(row[2]),
            "function_oid": int(row[3]),
            "function": str(row[4]),
            "type": int(row[5]),
            "enabled": str(row[6]),
            "definition_sha256": _sha256_bytes(_normalize_sql(row[7]).encode()),
        }
        for row in rows
    }


def _trigger_bindings(
    connection: psycopg.Connection[Any],
    expected: dict[str, tuple[str, str, int]],
) -> dict[str, dict[str, object]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT t.oid, t.tgname, c.relname, p.oid, p.proname,
                   t.tgtype, t.tgenabled,
                   pg_catalog.pg_get_triggerdef(t.oid, false)
              FROM pg_catalog.pg_trigger AS t
              JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
              JOIN pg_catalog.pg_proc AS p ON p.oid = t.tgfoid
              JOIN pg_catalog.pg_namespace AS pn ON pn.oid = p.pronamespace
             WHERE n.nspname = 'public' AND pn.nspname = 'public'
               AND NOT t.tgisinternal AND t.tgname = ANY(%s)
             ORDER BY t.tgname
            """,
            (list(expected),),
        )
        rows = cursor.fetchall()
    observed = {
        str(row[1]): {
            "oid": int(row[0]),
            "table": str(row[2]),
            "function_oid": int(row[3]),
            "function": str(row[4]),
            "type": int(row[5]),
            "enabled": str(row[6]),
            "definition_sha256": _sha256_bytes(_normalize_sql(row[7]).encode()),
        }
        for row in rows
    }
    if set(observed) != set(expected):
        raise BridgeError("required trigger identities are incomplete or ambiguous")
    for name, (table_name, function_name, trigger_type) in expected.items():
        row = observed[name]
        if (
            row["table"] != table_name
            or row["function"] != function_name
            or row["type"] != trigger_type
            or row["enabled"] != "O"
        ):
            raise BridgeError(f"unsafe trigger binding: {name}")
    return observed


def _assert_no_disabled_public_triggers(connection: psycopg.Connection[Any]) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT t.tgname
              FROM pg_catalog.pg_trigger AS t
              JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND NOT t.tgisinternal
               AND t.tgenabled <> 'O'
             ORDER BY t.tgname
            """
        )
        disabled = [str(row[0]) for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT count(*)
              FROM pg_catalog.pg_trigger AS t
              JOIN pg_catalog.pg_class AS c ON c.oid = t.tgrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND NOT t.tgisinternal
            """
        )
        total = int(cursor.fetchone()[0])
    if disabled:
        raise BridgeError("disabled final integrity triggers: " + ", ".join(disabled))
    return total


def _assert_audit_singleton_index(connection: psycopg.Connection[Any]) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT indexdef
              FROM pg_catalog.pg_indexes
             WHERE schemaname = 'public'
               AND indexname =
                   'uq_task_validation_campaign_events_audit_authorization_type'
            """
        )
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise BridgeError("the final task-validation audit singleton index is missing")
    definition = _normalize_sql(rows[0][0]).lower()
    if "unique index" not in definition or "audit_authorization_id is not null" not in definition:
        raise BridgeError("the final task-validation audit singleton index is unsafe")
    return _sha256_bytes(definition.encode())


def _assert_guarded_update_rolls_back(
    connection: psycopg.Connection[Any],
) -> dict[str, object]:
    """Exercise one immutable-field guard inside a savepoint and prove rollback."""

    arm_id = MOCK_ARM_IDS[0]
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT execution_backend FROM public.response_arms WHERE id = %s",
            (arm_id,),
        )
        original_backend = cursor.fetchone()[0]
    if original_backend != "openrouter":
        raise BridgeError("rollback probe requires the sealed legacy execution backend")

    sqlstate = ""
    message = ""
    with connection.transaction():
        try:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE public.response_arms "
                        "SET execution_backend = 'bedrock' WHERE id = %s",
                        (arm_id,),
                    )
            raise BridgeError("the final immutable-field guard accepted a forbidden update")
        except psycopg.Error as exc:
            sqlstate = str(exc.sqlstate or "")
            message = str(exc).splitlines()[0]
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT execution_backend FROM public.response_arms WHERE id = %s",
                (arm_id,),
            )
            after_savepoint = cursor.fetchone()[0]
        if after_savepoint != original_backend:
            raise BridgeError("the rejected immutable-field update escaped its savepoint")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT execution_backend FROM public.response_arms WHERE id = %s",
            (arm_id,),
        )
        after_transaction = cursor.fetchone()[0]
    if (
        sqlstate != "P0001"
        or "immutable" not in message.lower()
        or after_transaction != original_backend
    ):
        raise BridgeError("the final immutable-field rollback probe was inconclusive")
    return {
        "arm_id": arm_id,
        "attempted_column": "execution_backend",
        "attempted_value": "bedrock",
        "original_value": original_backend,
        "rejected": True,
        "sqlstate": sqlstate,
        "value_unchanged_after_savepoint": True,
        "value_unchanged_after_transaction": True,
    }


def _run_alembic(target: str) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FLAVOURBENCH_SERVICE_ROLE": "migration",
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", target],
        cwd=SERVICE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise BridgeError(f"Alembic upgrade to {target} failed:\n{result.stdout}\n{result.stderr}")


def _write_report(path_template: Path, report: dict[str, object]) -> tuple[Path, str, str]:
    semantic_sha256 = _sha256_bytes(_canonical_bytes(report))
    wrapper = {**report, "semantic_sha256": semantic_sha256}
    encoded = json.dumps(wrapper, indent=2, sort_keys=True).encode() + b"\n"
    path = Path(str(path_template).replace("{semantic_sha256}", semantic_sha256))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BridgeError("refusing to overwrite an existing proof report") from exc
    return path, semantic_sha256, _sha256_bytes(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-backup",
        type=Path,
        required=True,
        help="exact frozen pg_dump archive used for the pre-upgrade recovery point",
    )
    parser.add_argument(
        "--acknowledge-database",
        required=True,
        help="must exactly match the target database name parsed from the URL",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the staged upgrade; without this flag only preflight is run",
    )
    parser.add_argument(
        "--expected-final-revision",
        required=True,
        help="reviewed sole Alembic head; execution refuses if the source tree differs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "write a sanitized JSON proof report once; {semantic_sha256} in the "
            "path is replaced with the report content address"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bridge_script_sha256 = _sha256_bytes(Path(__file__).resolve().read_bytes())
    bridge_sql = BRIDGE_SQL_PATH.read_bytes()
    if _sha256_bytes(bridge_sql) != BRIDGE_SQL_SHA256:
        raise BridgeError("the bridge SQL does not match its reviewed content address")
    superseded_trigger_sql = SUPERSEDED_TRIGGER_SQL_PATH.read_bytes()
    if _sha256_bytes(superseded_trigger_sql) != SUPERSEDED_TRIGGER_SQL_SHA256:
        raise BridgeError("the superseded-trigger SQL does not match its reviewed content address")
    mock_accounting_sql = MOCK_ACCOUNTING_SQL_PATH.read_bytes()
    if _sha256_bytes(mock_accounting_sql) != MOCK_ACCOUNTING_SQL_SHA256:
        raise BridgeError("the mock-accounting SQL does not match its reviewed content address")
    if not args.source_backup.is_file():
        raise BridgeError("the frozen source backup does not exist")
    backup_sha256 = _sha256_bytes(args.source_backup.read_bytes())
    if backup_sha256 != SOURCE_BACKUP_SHA256:
        raise BridgeError("the source backup does not match the frozen content address")

    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("path_separator", "os")
    alembic_head = ScriptDirectory.from_config(config).get_current_head()
    if alembic_head != args.expected_final_revision:
        raise BridgeError(
            f"reviewed final revision {args.expected_final_revision} does not match "
            "the sole Alembic head"
        )

    database_url, database_name = _database_url()
    if args.acknowledge_database != database_name:
        raise BridgeError("--acknowledge-database does not match the target URL")

    report: dict[str, object] = {
        "schema_version": "flavourbench-legacy-0001-upgrade-proof-v3",
        "source_backup_sha256": backup_sha256,
        "bridge_script_sha256": bridge_script_sha256,
        "bridge_sql_sha256s": {
            "execution_contract": BRIDGE_SQL_SHA256,
            "superseded_arm_trigger": SUPERSEDED_TRIGGER_SQL_SHA256,
            "mock_accounting": MOCK_ACCOUNTING_SQL_SHA256,
        },
        "start_revision": START_REVISION,
        "stage_revision": STAGE_REVISION,
        "final_revision": args.expected_final_revision,
        "target_database": database_name,
        "applied": bool(args.apply),
        "started_at": datetime.now(UTC).isoformat(),
    }

    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TIME ZONE 'UTC'")
            cursor.execute(
                "SELECT pg_catalog.pg_advisory_lock(hashtextextended(%s, 0))",
                (LOCK_NAME,),
            )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT current_database(), current_user, "
                    "current_setting('server_version_num')::integer"
                )
                observed_database, migration_role, server_version_num = cursor.fetchone()
            if observed_database != database_name:
                raise BridgeError("connected database identity changed during preflight")
            if not 160000 <= int(server_version_num) < 170000:
                raise BridgeError("the frozen upgrade proof requires PostgreSQL 16")
            if str(migration_role) != "flavourbench_owner":
                raise BridgeError(
                    "the legacy migration must run as the bootstrapped flavourbench_owner role"
                )

            revision = _current_revision(connection)
            if revision != START_REVISION:
                raise BridgeError(
                    f"expected starting revision {START_REVISION}, observed {revision}"
                )
            schema_sha256, schema_inventory_counts = _start_schema_sha256(connection)
            if schema_sha256 != START_SCHEMA_SHA256:
                raise BridgeError(
                    "starting table/schema identity does not match the frozen 0001 backup"
                )
            _initial_object_preflight(connection)
            initial_counts = _row_counts(connection)
            if initial_counts != BASELINE_ROW_COUNTS:
                raise BridgeError("starting row inventory does not match the frozen 0001 backup")
            initial_identities = _baseline_identity_commitments(connection)
            (
                initial_content_root_sha256,
                initial_content_commitments,
                baseline_columns,
            ) = _baseline_content_commitments(connection)
            if initial_content_root_sha256 != START_CONTENT_ROOT_SHA256:
                raise BridgeError("starting row content does not match the frozen 0001 backup")
            report["preflight"] = {
                "schema_sha256": schema_sha256,
                "schema_inventory_counts": schema_inventory_counts,
                "row_counts": initial_counts,
                "identity_commitments": initial_identities,
                "content_root_sha256": initial_content_root_sha256,
                "content_commitments": initial_content_commitments,
                "alembic_version": START_REVISION,
                "original_table_count": 17,
                "content_committed_data_table_count": 16,
                "migration_role": str(migration_role),
                "postgresql_major": 16,
                "public_user_trigger_count": 0,
                "public_flavourbench_function_count": 0,
            }

            if not args.apply:
                report["completed_at"] = datetime.now(UTC).isoformat()
            else:
                _run_alembic(STAGE_REVISION)
                if _current_revision(connection) != STAGE_REVISION:
                    raise BridgeError("database did not stop at the reviewed bridge stage")
                if _row_counts(connection) != initial_counts:
                    raise BridgeError("row counts changed before the bridge was installed")
                stage_content_root_sha256, stage_content_commitments, _ = (
                    _baseline_content_commitments(connection, baseline_columns=baseline_columns)
                )
                if (
                    stage_content_root_sha256 != initial_content_root_sha256
                    or stage_content_commitments != initial_content_commitments
                ):
                    raise BridgeError("an original 0001 field changed before the bridge stage")

                stage_function = _function_identity(connection)
                if stage_function["body_sha256"] != UNSAFE_STAGE_BODY_SHA256:
                    raise BridgeError("0012 does not contain the exact known legacy function body")
                if stage_function["config"] != "":
                    raise BridgeError("the legacy function metadata is not the expected 0010 form")
                stage_triggers = _trigger_bindings(connection, STAGE_TRIGGER_BINDINGS)
                if {row["function_oid"] for row in stage_triggers.values()} != {
                    stage_function["oid"]
                }:
                    raise BridgeError("the three stage triggers do not share the sealed function")

                superseded_function = _function_identity(connection, SUPERSEDED_ARM_FUNCTION_NAME)
                if (
                    superseded_function["body_sha256"] != SUPERSEDED_ARM_FUNCTION_BODY_SHA256
                    or superseded_function["config"] != ""
                ):
                    raise BridgeError(
                        "the 0009 response-arm function is not the exact superseded body"
                    )
                superseded_trigger = _trigger_bindings(connection, SUPERSEDED_TRIGGER_BINDING)
                if {row["function_oid"] for row in superseded_trigger.values()} != {
                    superseded_function["oid"]
                }:
                    raise BridgeError(
                        "the superseded response-arm trigger has an unexpected function"
                    )
                stage_trigger_inventory = _public_trigger_inventory(connection)
                mock_before_sha256, mock_before_summary = _mock_accounting_commitment(connection)
                if mock_before_sha256 != MOCK_ACCOUNTING_BEFORE_SHA256:
                    raise BridgeError(
                        "legacy mock accounting does not match the reviewed pre-state"
                    )
                mock_full_before_sha256, mock_full_before = _mock_full_row_commitment(connection)
                if mock_full_before_sha256 != MOCK_FULL_ROWS_BEFORE_SHA256:
                    raise BridgeError(
                        "legacy full mock rows do not match the reviewed staged pre-state"
                    )

                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.execute(bridge_sql.decode())

                    bridged_function = _function_identity(connection)
                    bridged_triggers = _trigger_bindings(connection, STAGE_TRIGGER_BINDINGS)
                    if bridged_function["oid"] != stage_function["oid"]:
                        raise BridgeError(
                            "CREATE OR REPLACE unexpectedly changed the function identity"
                        )
                    if bridged_function["body_sha256"] != SAFE_BODY_SHA256:
                        raise BridgeError(
                            "the installed bridge function body is not the reviewed body"
                        )
                    if bridged_function["config"] != "search_path=pg_catalog, public":
                        raise BridgeError("the installed bridge function search_path is unsafe")
                    if {
                        name: (row["oid"], row["function_oid"], row["definition_sha256"])
                        for name, row in bridged_triggers.items()
                    } != {
                        name: (row["oid"], row["function_oid"], row["definition_sha256"])
                        for name, row in stage_triggers.items()
                    }:
                        raise BridgeError(
                            "the 0010 trigger identities changed during function replacement"
                        )

                    with connection.cursor() as cursor:
                        cursor.execute(superseded_trigger_sql.decode())
                    after_drop_inventory = _public_trigger_inventory(connection)
                    expected_after_drop = dict(stage_trigger_inventory)
                    removed_trigger = expected_after_drop.pop(
                        "trg_response_arm_contract_immutable", None
                    )
                    if removed_trigger is None or after_drop_inventory != expected_after_drop:
                        raise BridgeError(
                            "the bridge did not remove exactly one superseded trigger"
                        )

                    with connection.cursor() as cursor:
                        cursor.execute(mock_accounting_sql.decode())
                        normalized_rows = cursor.rowcount
                    if normalized_rows != 4:
                        raise BridgeError(
                            "mock-accounting normalization did not update exactly four rows"
                        )
                    mock_after_sha256, mock_after_summary = _mock_accounting_commitment(connection)
                    if mock_after_sha256 != MOCK_ACCOUNTING_AFTER_SHA256:
                        raise BridgeError(
                            "legacy mock accounting does not match the reviewed post-state"
                        )
                    if mock_before_summary != mock_after_summary:
                        raise BridgeError("mock cost, provider, or quarantine semantics changed")
                    mock_full_after_sha256, mock_full_after = _mock_full_row_commitment(connection)
                    if mock_full_after_sha256 != MOCK_FULL_ROWS_AFTER_SHA256:
                        raise BridgeError(
                            "legacy full mock rows do not match the reviewed staged post-state"
                        )
                    mock_exact_field_diff = _mock_field_diff(mock_full_before, mock_full_after)
                    if _row_counts(connection) != initial_counts:
                        raise BridgeError("row counts changed while installing the bridge")
                    if _baseline_identity_commitments(connection) != initial_identities:
                        raise BridgeError(
                            "a frozen baseline row identity changed during the bridge"
                        )
                    bridge_content_root_sha256, bridge_content_commitments, _ = (
                        _baseline_content_commitments(connection, baseline_columns=baseline_columns)
                    )
                    if (
                        bridge_content_root_sha256 != initial_content_root_sha256
                        or bridge_content_commitments != initial_content_commitments
                    ):
                        raise BridgeError("the bridge changed an original 0001 field")

                # Re-read every bridge assertion after commit, not merely within
                # the transaction that could still have been rolled back.
                bridged_function = _function_identity(connection)
                bridged_triggers = _trigger_bindings(connection, STAGE_TRIGGER_BINDINGS)
                if bridged_function["body_sha256"] != SAFE_BODY_SHA256:
                    raise BridgeError("the committed bridge function is not JSON-safe")
                if "trg_response_arm_contract_immutable" in _public_trigger_inventory(connection):
                    raise BridgeError("the superseded arm trigger survived the bridge")
                committed_mock_sha256, _ = _mock_accounting_commitment(connection)
                if committed_mock_sha256 != MOCK_ACCOUNTING_AFTER_SHA256:
                    raise BridgeError("the committed mock normalization is not exact")
                committed_full_mock_sha256, _ = _mock_full_row_commitment(connection)
                if committed_full_mock_sha256 != MOCK_FULL_ROWS_AFTER_SHA256:
                    raise BridgeError("the committed full mock-row post-state is not exact")

                _run_alembic("head")
                final_observed_revision = _current_revision(connection)
                if final_observed_revision != args.expected_final_revision:
                    raise BridgeError("database did not reach the reviewed final Alembic revision")
                final_counts = _row_counts(connection)
                if final_counts != initial_counts:
                    raise BridgeError("baseline table row counts changed during the full upgrade")
                final_identities = _baseline_identity_commitments(connection)
                if final_identities != initial_identities:
                    raise BridgeError(
                        "baseline primary-key identities changed during the full upgrade"
                    )
                final_content_root_sha256, final_content_commitments, _ = (
                    _baseline_content_commitments(connection, baseline_columns=baseline_columns)
                )
                if (
                    final_content_root_sha256 != initial_content_root_sha256
                    or final_content_commitments != initial_content_commitments
                ):
                    raise BridgeError("an original 0001 row value changed during the full upgrade")
                final_function = _function_identity(connection)
                if final_function["body_sha256"] != SAFE_BODY_SHA256:
                    raise BridgeError("the final execution-contract function is not JSON-safe")
                if final_function["config"] != "search_path=pg_catalog, public":
                    raise BridgeError("the final execution-contract function metadata is unsafe")
                final_required_triggers = _trigger_bindings(
                    connection, FINAL_REQUIRED_TRIGGER_BINDINGS
                )
                final_trigger_count = _assert_no_disabled_public_triggers(connection)
                audit_index_sha256 = _assert_audit_singleton_index(connection)
                final_mock_sha256, final_mock_summary = _mock_accounting_commitment(connection)
                if (
                    final_mock_sha256 != MOCK_ACCOUNTING_AFTER_SHA256
                    or final_mock_summary != mock_after_summary
                ):
                    raise BridgeError(
                        "final head changed the preserved mock quarantine or cost evidence"
                    )
                rollback_probe = _assert_guarded_update_rolls_back(connection)
                if (
                    _row_counts(connection) != final_counts
                    or _baseline_identity_commitments(connection) != final_identities
                    or _baseline_content_commitments(connection, baseline_columns=baseline_columns)[
                        0
                    ]
                    != final_content_root_sha256
                    or _mock_accounting_commitment(connection)[0] != final_mock_sha256
                ):
                    raise BridgeError("the rollback probe changed frozen database evidence")

                report["stage"] = {
                    "revision": STAGE_REVISION,
                    "legacy_function": stage_function,
                    "trigger_bindings": stage_triggers,
                    "superseded_function": superseded_function,
                    "superseded_trigger": superseded_trigger,
                    "mock_accounting_before_sha256": mock_before_sha256,
                    "mock_full_rows_before_sha256": mock_full_before_sha256,
                    "baseline_content_root_sha256": stage_content_root_sha256,
                }
                report["bridge"] = {
                    "function": bridged_function,
                    "trigger_bindings_unchanged": True,
                    "superseded_trigger_removed": True,
                    "mock_accounting_rows_normalized": 4,
                    "mock_accounting_after_sha256": mock_after_sha256,
                    "mock_full_rows_after_sha256": mock_full_after_sha256,
                    "mock_exact_field_diff": mock_exact_field_diff,
                    "mock_rows_changed": 4,
                    "mock_field_values_changed": 8,
                    "mock_allowed_fields": [
                        "billing_reconciliation_status",
                        "cost_accounting_basis",
                    ],
                    "mock_accounting_summary": mock_after_summary,
                    "baseline_identity_commitments_unchanged": True,
                    "baseline_content_commitments_unchanged": True,
                    "row_counts_unchanged": True,
                }
                report["final"] = {
                    "revision": final_observed_revision,
                    "function": final_function,
                    "required_integrity_trigger_count": len(final_required_triggers),
                    "all_public_user_trigger_count": final_trigger_count,
                    "all_public_user_triggers_enabled": True,
                    "audit_singleton_index_sha256": audit_index_sha256,
                    "row_counts": final_counts,
                    "identity_commitments": final_identities,
                    "content_root_sha256": final_content_root_sha256,
                    "content_commitments": final_content_commitments,
                    "alembic_revision_transition": {
                        "row_count_before": 1,
                        "revision_before": START_REVISION,
                        "row_count_after": 1,
                        "revision_after": final_observed_revision,
                    },
                    "mock_accounting_sha256": final_mock_sha256,
                    "mock_accounting_summary": final_mock_summary,
                    "quarantined_legacy_fixture_data": True,
                    "rollback_safety": {
                        "source_backup_verified": True,
                        "bridge_operations_single_transaction": True,
                        "alembic_postgresql_transactional_ddl": True,
                        "guarded_update_probe": rollback_probe,
                        "post_probe_counts_identities_content_and_mock_evidence_unchanged": True,
                    },
                    "row_counts_unchanged": True,
                }
                report["completed_at"] = datetime.now(UTC).isoformat()
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_catalog.pg_advisory_unlock(hashtextextended(%s, 0))",
                    (LOCK_NAME,),
                )

    if args.report:
        report_path, semantic_sha256, physical_sha256 = _write_report(args.report, report)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "report": str(report_path),
                    "semantic_sha256": semantic_sha256,
                    "physical_sha256": physical_sha256,
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as exc:
        print(f"legacy upgrade bridge refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
