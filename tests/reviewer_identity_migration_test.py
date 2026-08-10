from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect

from flavourbench.database import EXPECTED_SCHEMA_REVISION


def test_reviewer_identity_migration_is_the_linear_head_and_builds_constraints(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'reviewer-identity-migration.sqlite3'}"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("alembic")), "upgrade", "head"],
        cwd=project_root,
        env={
            **os.environ,
            "FLAVOURBENCH_DATABASE_URL": database_url,
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert EXPECTED_SCHEMA_REVISION == "0035_participant_lifecycle_privacy"
    head_migration = (
        project_root / "alembic" / "versions" / "0035_participant_lifecycle_privacy.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0035_participant_lifecycle_privacy"' in head_migration
    assert 'down_revision = "0034_task_validation_replay_binding"' in head_migration

    inspector = inspect(create_engine(database_url))
    expected_tables = {
        "reviewer_identity_bindings",
        "reviewer_access_credentials",
        "reviewer_qualification_evidence",
        "reviewer_calibration_sets",
        "reviewer_calibration_ballots",
        "reviewer_family_admissions",
        "reviewer_enrollment_offers",
        "reviewer_consent_acceptances",
        "reviewer_participation_lifecycles",
        "reviewer_withdrawal_receipts",
        "reviewer_retention_schedules",
        "reviewer_deletion_receipts",
    }
    assert expected_tables <= set(inspector.get_table_names())
    binding_uniques = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("reviewer_identity_bindings")
    }
    assert "uq_reviewer_identity_bindings_season_person" in binding_uniques

    vote_columns = {column["name"]: column for column in inspector.get_columns("votes")}
    assert vote_columns["provenance_status"]["nullable"] is False
    vote_foreign_keys = {
        column
        for constraint in inspector.get_foreign_keys("votes")
        for column in constraint["constrained_columns"]
    }
    assert {
        "reviewer_id",
        "reviewer_identity_binding_id",
        "reviewer_family_admission_id",
    } <= vote_foreign_keys
    vote_indexes = {index["name"] for index in inspector.get_indexes("votes")}
    assert "uq_votes_verified_person_battle" in vote_indexes


def test_postgresql_migration_contains_admission_vote_and_credential_guards() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0030_reviewer_identity_admission.py"
    ).read_text(encoding="utf-8")
    assert 'revision = "0030_reviewer_identity_admission"' in migration
    assert 'down_revision = "0029_qwencloud_direct_backend"' in migration
    assert "uq_reviewer_identity_bindings_season_person" in migration
    assert "flavourbench_reviewer_family_admission_guard_v1" in migration
    assert "flavourbench_reviewer_credential_lifecycle_v1" in migration
    assert "flavourbench_verified_expert_vote_guard_v1" in migration
