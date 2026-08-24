from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text


def test_qwencloud_backend_is_present_in_all_budget_constraints(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'qwencloud-migration.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    result = subprocess.run(
        [str(Path(sys.executable).with_name("alembic")), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    inspector = inspect(engine)
    for table, constraint_name in (
        ("season_provider_budgets", "ck_season_provider_budgets_backend"),
        ("provider_account_budgets", "ck_provider_account_budgets_backend"),
        (
            "provider_account_authorizations",
            "ck_provider_account_authorizations_backend",
        ),
    ):
        constraints = {
            row["name"]: str(row["sqltext"])
            for row in inspector.get_check_constraints(table)
        }
        assert "qwencloud_direct" in constraints[constraint_name]

    with engine.connect() as connection:
        trigger_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).scalars()
        )
    assert {
        "trg_governed_spend_monotonic_season_provider_budgets",
        "trg_governed_spend_monotonic_season_provider_budgets_insert",
        "trg_governed_spend_monotonic_provider_account_budgets",
        "trg_governed_spend_monotonic_provider_account_budgets_insert",
    } <= trigger_names


def test_postgresql_reservation_function_migration_is_fail_closed() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0029_qwencloud_direct_backend.py"
    )
    migration = migration_path.read_text(encoding="utf-8")
    assert 'revision = "0029_qwencloud_direct_backend"' in migration
    assert 'down_revision = "0028_kimi_direct_backend"' in migration
    assert "pg_get_functiondef" in migration
    assert "len(matches) != 2" in migration
    assert migration.count("qwencloud_direct") >= 7
    assert "downgrade across direct-QwenCloud budget authority is prohibited" in migration

    rewrite = runpy.run_path(str(migration_path))[
        "_rewrite_postgresql_reservation_function"
    ]
    bare_postgresql_definition = """
    CREATE FUNCTION public.flavourbench_reserve_battle_budget(p_battle_id text)
    RETURNS void LANGUAGE plpgsql AS $function$
    BEGIN
      PERFORM 1 WHERE a.execution_backend IN ('openrouter', 'bedrock');
      PERFORM 1 WHERE a.execution_backend IN ('openrouter', 'bedrock');
    END;
    $function$;
    """
    updated = rewrite(bare_postgresql_definition)
    assert updated.count("qwencloud_direct") == 2
    assert updated.count("kimi_direct") == 2
    assert "'qwencloud_direct', 'kimi_direct'" not in updated
