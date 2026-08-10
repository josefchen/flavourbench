#!/usr/bin/env python3
"""Rehearse the frozen 0001 bridge through the additive 0035 privacy head.

This successor never changes the reviewed v3 bridge or its immutable 0034
proof.  It content-addresses both, runs that exact bridge against a disposable
restore of the same frozen backup, and then independently verifies the 0035
participant-consent, withdrawal, retention, and deletion guard surface before
writing a separate append-only v4 proof.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from flavourbench.database import (
    _PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256,
    _PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS,
    _REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256,
    _REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS,
    _assert_postgresql_participant_lifecycle_guards,
    _assert_postgresql_reviewer_task_validation_guards,
)

SERVICE_ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR_SCRIPT = SERVICE_ROOT / "scripts" / "upgrade_legacy_0001_postgresql.py"
FROZEN_V3_PROOF = (
    SERVICE_ROOT / "artifacts" / "migration-proofs" / "legacy-0001-upgrade-bridge-proof-"
    "81dde02834fd6d5096245fdb9e23b74fd3b7d07a56977596d26e63fe5641fb74.json"
)
SUCCESSOR_BRIDGE_TEST = SERVICE_ROOT / "tests" / "legacy_0001_upgrade_bridge_test.py"

SOURCE_BACKUP_SHA256 = "f81b637903a25514a14cefb7363dd9e32b2b4659e65e10b43e25ca9ec8a5bd08"
PREDECESSOR_SCRIPT_SHA256 = "5c94535e064171a2714e16e49940628f030cf3c541239b6bd3b9bdafb747fb78"
FROZEN_V3_PROOF_SEMANTIC_SHA256 = "81dde02834fd6d5096245fdb9e23b74fd3b7d07a56977596d26e63fe5641fb74"
FROZEN_V3_PROOF_PHYSICAL_SHA256 = "527487c8fdbc664a4b94006a51afe77b81f5292cf07746e347817cfbf6a1e151"
SUCCESSOR_BRIDGE_TEST_SHA256 = "c8457b1950b509df5a3919ed7d7fa8a139e2adf5efbc40a983fda3ec010238d1"
REPAIRED_0035_SOURCE_RELATIVE_PATHS = {
    "migration_0035": "alembic/versions/0035_participant_lifecycle_privacy.py",
    "database_readiness": "src/flavourbench/database.py",
    "model_guards": "src/flavourbench/models.py",
    "participant_lifecycle": "src/flavourbench/participant_lifecycle.py",
    "reviewer_identity": "src/flavourbench/reviewer_identity.py",
    "task_validation_runtime": "src/flavourbench/task_validation_runtime.py",
    "participant_lifecycle_test": "tests/participant_lifecycle_test.py",
    "participant_lifecycle_api_test": "tests/participant_lifecycle_api_test.py",
    "participant_lifecycle_migration_test": "tests/participant_lifecycle_migration_test.py",
    "participant_lifecycle_postgresql_test": "tests/participant_lifecycle_postgresql_test.py",
    "postgresql_candidate_capacity_helper": "tests/postgresql_candidate_capacity_helper.py",
    "reviewer_identity_admission_test": "tests/reviewer_identity_admission_test.py",
    "task_validation_concurrent_submit_helper": (
        "tests/task_validation_concurrent_submit_helper.py"
    ),
    "task_validation_runtime_test": "tests/task_validation_runtime_test.py",
}
REPAIRED_0035_SOURCE_SHA256S = {
    "migration_0035": "9afd6e996b0c5c8a91204e18d64c7365cd0d478827132693be61ef57cf5f1c10",
    "database_readiness": "32bdbdf23613ddf1f56f937fa2910010717636cff1e1348dd27cc089ceea160b",
    "model_guards": "d34180a30774fd17a99a1791b74756112169292413c95fe39755601f83ccaaa8",
    "participant_lifecycle": "a20e5aecb0ef8a7922fce5001e3f87282ad504d55c88f0544bdf53cc3984214d",
    "reviewer_identity": "5f9e0112ea3df17956dc4be2bfb80bace8954305e5eedc769c7c68ff2b9c1b57",
    "task_validation_runtime": "b71af1471be4c8dbca8bf64cc229b351437220240359f265a9077809f5cc83a1",
    "participant_lifecycle_test": (
        "3ca35e2e2502077301170cc7b1a2a1ec2b6c4d9cf50b7e62fe7fa118f0e1e88c"
    ),
    "participant_lifecycle_api_test": (
        "60acf2f7da1d660a635a4ea51d6e4990685b062a41a604b37f83752f93a9b424"
    ),
    "participant_lifecycle_migration_test": (
        "b7439a062a77b59b266c21ffc0f65a5e8c292daa54615dbd873995dd3ec7e72a"
    ),
    "participant_lifecycle_postgresql_test": (
        "b1fef15bfe154b49ae104ebff217e3cc57d6ed79806a19ca79548ee6aa440bc1"
    ),
    "postgresql_candidate_capacity_helper": (
        "629222cc4a6d3297abdb05cacc5fa9e427cae36923425cb4d4027c76214f0e37"
    ),
    "reviewer_identity_admission_test": (
        "ca73954ad379a1a0819bc8898a1c9009b517c8d4856e5f9c1fdfe3f124097a45"
    ),
    "task_validation_concurrent_submit_helper": (
        "cbb9118d07fd28ee68156bca181e14de317e8b2f52470884d0b078a40785bc10"
    ),
    "task_validation_runtime_test": (
        "a71d37560f5e1c4c1ee7d0a881335619077221d8fc0501c6c0aaab0449b4daf8"
    ),
}
START_REVISION = "0001_initial"
FROZEN_V3_FINAL_REVISION = "0034_task_validation_replay_binding"
EXPECTED_FINAL_REVISION = "0035_participant_lifecycle_privacy"
LOCK_NAME = "flavourbench:legacy-0001-upgrade-bridge:v4"
PARTICIPANT_TABLES = (
    "reviewer_enrollment_offers",
    "reviewer_consent_acceptances",
    "reviewer_participation_lifecycles",
    "reviewer_withdrawal_receipts",
    "reviewer_retention_schedules",
    "reviewer_deletion_receipts",
)


class SuccessorBridgeError(RuntimeError):
    """A frozen lineage or 0035 successor assertion failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _verified_json_proof(
    path: Path,
    *,
    expected_semantic_sha256: str | None = None,
    expected_physical_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    encoded = path.read_bytes()
    physical_sha256 = _sha256_bytes(encoded)
    if expected_physical_sha256 is not None and not hmac.compare_digest(
        physical_sha256, expected_physical_sha256
    ):
        raise SuccessorBridgeError("a predecessor proof physical hash changed")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise SuccessorBridgeError("a predecessor proof is not a JSON object")
    semantic_sha256 = value.get("semantic_sha256")
    if not isinstance(semantic_sha256, str):
        raise SuccessorBridgeError("a predecessor proof lacks its semantic hash")
    content = {key: item for key, item in value.items() if key != "semantic_sha256"}
    observed_semantic_sha256 = _sha256_bytes(_canonical_bytes(content))
    if not hmac.compare_digest(semantic_sha256, observed_semantic_sha256):
        raise SuccessorBridgeError("a predecessor proof semantic hash is invalid")
    if expected_semantic_sha256 is not None and not hmac.compare_digest(
        semantic_sha256, expected_semantic_sha256
    ):
        raise SuccessorBridgeError("a predecessor proof semantic hash changed")
    return value, semantic_sha256, physical_sha256


def _verify_repaired_0035_source_pins() -> dict[str, str]:
    observed = {
        label: _sha256_bytes((SERVICE_ROOT / relative_path).read_bytes())
        for label, relative_path in REPAIRED_0035_SOURCE_RELATIVE_PATHS.items()
    }
    if observed != REPAIRED_0035_SOURCE_SHA256S:
        raise SuccessorBridgeError("the repaired 0035 source or regression bundle changed")
    if _sha256_bytes(SUCCESSOR_BRIDGE_TEST.read_bytes()) != SUCCESSOR_BRIDGE_TEST_SHA256:
        raise SuccessorBridgeError("the repaired 0035 successor bridge test changed")
    return dict(sorted(observed.items()))


def _write_report(path_template: Path, report: dict[str, Any]) -> tuple[Path, str, str]:
    semantic_sha256 = _sha256_bytes(_canonical_bytes(report))
    wrapper = {**report, "semantic_sha256": semantic_sha256}
    encoded = json.dumps(wrapper, indent=2, sort_keys=True).encode() + b"\n"
    path = Path(str(path_template).replace("{semantic_sha256}", semantic_sha256))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise SuccessorBridgeError("refusing to overwrite an existing v4 proof") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return path, semantic_sha256, _sha256_bytes(encoded)


def _database_url() -> tuple[str, str]:
    raw = os.environ.get("FLAVOURBENCH_DATABASE_URL", "")
    if not raw:
        raise SuccessorBridgeError("FLAVOURBENCH_DATABASE_URL is required")
    parsed = make_url(raw)
    if parsed.get_backend_name() != "postgresql" or parsed.database is None:
        raise SuccessorBridgeError("the v4 bridge requires a named PostgreSQL database")
    if not any(marker in parsed.database.lower() for marker in ("test", "bridge", "rehearsal")):
        raise SuccessorBridgeError("the v4 bridge only permits a named disposable database")
    return (
        parsed.set(drivername="postgresql").render_as_string(hide_password=False),
        parsed.database,
    )


def _current_revision(connection: psycopg.Connection[Any]) -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT version_num FROM public.alembic_version")
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise SuccessorBridgeError("alembic_version must contain exactly one row")
    return str(rows[0][0])


def _verify_final_participant_surface(database_url: str) -> dict[str, Any]:
    sqlalchemy_url = make_url(database_url).set(drivername="postgresql+psycopg")
    engine = create_engine(sqlalchemy_url)
    try:
        with engine.connect() as connection:
            _assert_postgresql_participant_lifecycle_guards(connection)
            _assert_postgresql_reviewer_task_validation_guards(connection)
            revision = str(connection.scalar(text("SELECT version_num FROM alembic_version")))
            participant_row_counts = {
                table_name: int(connection.scalar(text(f"SELECT count(*) FROM {table_name}")) or 0)
                for table_name in PARTICIPANT_TABLES
            }
            reviewer_count = int(
                connection.scalar(text("SELECT count(*) FROM expert_reviewers")) or 0
            )
            trigger_count = int(
                connection.scalar(
                    text(
                        "SELECT count(*) FROM pg_catalog.pg_trigger AS trigger "
                        "JOIN pg_catalog.pg_class AS relation "
                        "ON relation.oid = trigger.tgrelid "
                        "JOIN pg_catalog.pg_namespace AS namespace "
                        "ON namespace.oid = relation.relnamespace "
                        "WHERE namespace.nspname = 'public' "
                        "AND NOT trigger.tgisinternal"
                    )
                )
                or 0
            )
        reviewer_columns = {
            column["name"] for column in inspect(engine).get_columns("expert_reviewers")
        }
    finally:
        engine.dispose()
    required_reviewer_columns = {
        "privacy_status",
        "privacy_redacted_at",
        "privacy_redaction_receipt_sha256",
    }
    if revision != EXPECTED_FINAL_REVISION:
        raise SuccessorBridgeError("the bridge did not reach the exact 0035 head")
    if any(participant_row_counts.values()) or reviewer_count != 0:
        raise SuccessorBridgeError("0035 created participant identity or lifecycle data")
    if not required_reviewer_columns <= reviewer_columns:
        raise SuccessorBridgeError("0035 reviewer privacy columns are incomplete")
    return {
        "revision": revision,
        "participant_table_row_counts": participant_row_counts,
        "expert_reviewer_row_count": reviewer_count,
        "expert_reviewer_privacy_columns": sorted(required_reviewer_columns),
        "participant_guard_function_body_sha256s": dict(
            sorted(_PARTICIPANT_LIFECYCLE_GUARD_BODY_SHA256.items())
        ),
        "participant_guard_trigger_count": len(_PARTICIPANT_LIFECYCLE_GUARD_TRIGGERS),
        "reviewer_task_guard_function_body_sha256s": dict(
            sorted(_REVIEWER_TASK_VALIDATION_GUARD_BODY_SHA256.items())
        ),
        "reviewer_task_guard_trigger_count": len(_REVIEWER_TASK_VALIDATION_GUARD_TRIGGERS),
        "all_public_user_trigger_count": trigger_count,
        "participant_guard_readiness": True,
        "identities_created": 0,
        "enrollments_created": 0,
        "consent_activated": False,
        "human_contact_performed": False,
        "provider_or_model_calls_performed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-backup", type=Path, required=True)
    parser.add_argument("--acknowledge-database", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-final-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.apply:
        raise SuccessorBridgeError("the v4 artifact is an applied rehearsal proof")
    if args.expected_final_revision != EXPECTED_FINAL_REVISION:
        raise SuccessorBridgeError("the v4 successor is pinned to the 0035 head")
    if not args.source_backup.is_file() or _sha256_bytes(args.source_backup.read_bytes()) != (
        SOURCE_BACKUP_SHA256
    ):
        raise SuccessorBridgeError("the source backup does not match its frozen hash")
    if _sha256_bytes(PREDECESSOR_SCRIPT.read_bytes()) != PREDECESSOR_SCRIPT_SHA256:
        raise SuccessorBridgeError("the reviewed predecessor bridge script changed")
    repaired_0035_source_sha256s = _verify_repaired_0035_source_pins()
    frozen_v3, frozen_v3_semantic, frozen_v3_physical = _verified_json_proof(
        FROZEN_V3_PROOF,
        expected_semantic_sha256=FROZEN_V3_PROOF_SEMANTIC_SHA256,
        expected_physical_sha256=FROZEN_V3_PROOF_PHYSICAL_SHA256,
    )
    if (
        frozen_v3.get("schema_version") != "flavourbench-legacy-0001-upgrade-proof-v3"
        or frozen_v3.get("final_revision") != FROZEN_V3_FINAL_REVISION
    ):
        raise SuccessorBridgeError("the immutable v3 proof lineage is not the frozen 0034 proof")

    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("path_separator", "os")
    if ScriptDirectory.from_config(config).get_current_head() != EXPECTED_FINAL_REVISION:
        raise SuccessorBridgeError("the source tree does not have the reviewed sole 0035 head")
    database_url, database_name = _database_url()
    if args.acknowledge_database != database_name:
        raise SuccessorBridgeError("--acknowledge-database does not match the target URL")

    started_at = datetime.now(UTC).isoformat()
    successor_script_sha256 = _sha256_bytes(Path(__file__).resolve().read_bytes())
    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_user, current_setting('server_version_num')::integer")
            migration_role, server_version_num = cursor.fetchone()
            if migration_role != "flavourbench_owner":
                raise SuccessorBridgeError("the bridge must run as flavourbench_owner")
            if not 160000 <= int(server_version_num) < 170000:
                raise SuccessorBridgeError("the v4 proof requires PostgreSQL 16")
            if _current_revision(connection) != START_REVISION:
                raise SuccessorBridgeError("the rehearsal did not start at frozen 0001")
            cursor.execute(
                "SELECT pg_catalog.pg_advisory_lock(hashtextextended(%s, 0))",
                (LOCK_NAME,),
            )
        try:
            with tempfile.TemporaryDirectory(prefix="flavourbench-bridge-v4-") as temporary:
                temporary_path = Path(temporary)
                base_template = temporary_path / "base-proof-{semantic_sha256}.json"
                environment = os.environ.copy()
                environment["FLAVOURBENCH_DATABASE_URL"] = (
                    make_url(database_url)
                    .set(drivername="postgresql+psycopg")
                    .render_as_string(hide_password=False)
                )
                execution = subprocess.run(
                    [
                        sys.executable,
                        str(PREDECESSOR_SCRIPT),
                        "--source-backup",
                        str(args.source_backup),
                        "--acknowledge-database",
                        database_name,
                        "--apply",
                        "--expected-final-revision",
                        EXPECTED_FINAL_REVISION,
                        "--report",
                        str(base_template),
                    ],
                    cwd=SERVICE_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if execution.returncode:
                    raise SuccessorBridgeError(
                        "the exact predecessor bridge failed during the 0035 rehearsal:\n"
                        f"{execution.stdout}\n{execution.stderr}"
                    )
                base_paths = list(temporary_path.glob("base-proof-*.json"))
                if len(base_paths) != 1:
                    raise SuccessorBridgeError("the predecessor emitted an ambiguous proof set")
                base_report, base_semantic, base_physical = _verified_json_proof(base_paths[0])
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_catalog.pg_advisory_unlock(hashtextextended(%s, 0))",
                    (LOCK_NAME,),
                )

    if (
        base_report.get("schema_version") != "flavourbench-legacy-0001-upgrade-proof-v3"
        or base_report.get("source_backup_sha256") != SOURCE_BACKUP_SHA256
        or base_report.get("bridge_script_sha256") != PREDECESSOR_SCRIPT_SHA256
        or base_report.get("start_revision") != START_REVISION
        or base_report.get("final_revision") != EXPECTED_FINAL_REVISION
        or base_report.get("applied") is not True
    ):
        raise SuccessorBridgeError("the predecessor 0035 execution proof is misbound")
    preflight = base_report.get("preflight")
    final = base_report.get("final")
    if not isinstance(preflight, dict) or not isinstance(final, dict):
        raise SuccessorBridgeError("the predecessor execution proof is incomplete")
    if (
        preflight.get("row_counts") != final.get("row_counts")
        or preflight.get("identity_commitments") != final.get("identity_commitments")
        or preflight.get("content_root_sha256") != final.get("content_root_sha256")
        or preflight.get("content_commitments") != final.get("content_commitments")
    ):
        raise SuccessorBridgeError("the 0035 upgrade changed frozen 0001 evidence")

    successor_checks = _verify_final_participant_surface(database_url)
    report: dict[str, Any] = {
        "schema_version": "flavourbench-legacy-0001-upgrade-proof-v4",
        "source_backup_sha256": SOURCE_BACKUP_SHA256,
        "successor_bridge_script_sha256": successor_script_sha256,
        "successor_bridge_test_sha256": SUCCESSOR_BRIDGE_TEST_SHA256,
        "repaired_0035_source_sha256s": repaired_0035_source_sha256s,
        "predecessor_bridge_script_sha256": PREDECESSOR_SCRIPT_SHA256,
        "start_revision": START_REVISION,
        "final_revision": EXPECTED_FINAL_REVISION,
        "target_database": database_name,
        "applied": True,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "frozen_0034_proof": {
            "schema_version": frozen_v3["schema_version"],
            "semantic_sha256": frozen_v3_semantic,
            "physical_sha256": frozen_v3_physical,
            "final_revision": frozen_v3["final_revision"],
            "bytes_preserved": True,
        },
        "base_bridge_execution_semantic_sha256": base_semantic,
        "base_bridge_execution_physical_sha256": base_physical,
        "base_bridge_execution": base_report,
        "successor_0035_checks": successor_checks,
        "frozen_baseline_evidence_unchanged": True,
    }
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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuccessorBridgeError as exc:
        print(f"legacy 0035 successor bridge refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
