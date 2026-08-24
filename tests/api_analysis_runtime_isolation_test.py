from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _production_api_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "FLAVOURBENCH_ENVIRONMENT": "production",
            "FLAVOURBENCH_SERVICE_ROLE": "api",
            "FLAVOURBENCH_DATABASE_URL": (
                "postgresql+psycopg://api:unused@127.0.0.1:1/flavourbench"
            ),
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
            "FLAVOURBENCH_EXECUTION_MODE": "live",
            "FLAVOURBENCH_LIVE_AUTHORIZED": "true",
            "FLAVOURBENCH_SERVICE_TOKEN": "service-" + "1" * 40,
            "FLAVOURBENCH_ADMIN_TOKEN": "admin-" + "2" * 40,
            "FLAVOURBENCH_EXPERT_TOKEN": "expert-" + "3" * 40,
            "FLAVOURBENCH_PSEUDONYM_SECRET": "pseudonym-" + "4" * 40,
            "FLAVOURBENCH_TASK_VALIDATOR_IDENTITY_HMAC_SECRET": ("task-validator-" + "5" * 40),
            "FLAVOURBENCH_REVIEWER_IDENTITY_HMAC_SECRET": "reviewer-" + "6" * 40,
            "FLAVOURBENCH_REVIEWER_CREDENTIAL_HMAC_SECRET": ("reviewer-credential-" + "7" * 40),
            "FLAVOURBENCH_ORGANIZATION_API_KEY_HMAC_SECRET": ("organization-" + "8" * 40),
            "FLAVOURBENCH_RUN_CARD_SIGNING_SECRET": "run-card-" + "9" * 40,
            "FLAVOURBENCH_BUDGET_AUTHORIZATION_SIGNING_SECRET": ("budget-" + "a" * 40),
            "FLAVOURBENCH_TASK_VALIDATION_CAMPAIGN_ENABLED": "false",
            "FLAVOURBENCH_OPENROUTER_API_KEY": "",
            "FLAVOURBENCH_KIMI_API_KEY": "",
            "FLAVOURBENCH_COHERE_API_KEY": "",
            "FLAVOURBENCH_QWENCLOUD_API_KEY": "",
            "DASHSCOPE_API_KEY": "",
            "FLAVOURBENCH_CLOUDFLARE_AI_GATEWAY_TOKEN": "",
            "FLAVOURBENCH_MCP_TOKEN": "",
            "MCP_API_TOKEN": "",
        }
    )
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH", "")) if item
    )
    return environment


def test_production_api_never_loads_snapshot_analysis_runtime() -> None:
    script = r"""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import flavourbench.main as main
from flavourbench.service_ranking import (
    InProcessSnapshotAnalysisForbidden,
    _fit_bradley_terry,
    assert_api_analysis_runtime_clean,
    loaded_analysis_runtime_roots,
)


def assert_clean(stage):
    assert_api_analysis_runtime_clean()
    loaded = loaded_analysis_runtime_roots()
    if loaded:
        raise AssertionError(f"{stage} loaded analysis modules: {loaded}")


class Result:
    def all(self):
        return []


class Session:
    def __init__(self, season):
        self.season = season

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

    def scalar(self, _statement):
        return self.season

    def scalars(self, _statement):
        return Result()


season = SimpleNamespace(
    id="season-isolation-test",
    slug="season-isolation-test",
    status="draft",
    official=False,
    manifest_sha256="f" * 64,
    budget_cap_micros=0,
    budget_used_micros=0,
    budget_reserved_micros=0,
)
session = Session(season)

assert_clean("import")
main.database_readiness = lambda _session, expected_role: {
    "database": "ready",
    "databaseDialect": "sqlite",
    "databaseRole": expected_role,
    "schemaRevision": "test",
}


def session_override():
    yield session


main.app.dependency_overrides[main.get_db] = session_override
main.app.dependency_overrides[main.require_service_token] = lambda: None
main.app.dependency_overrides[main.require_admin_token] = lambda: None
client = TestClient(main.app)

health_response = client.get("/health")
assert health_response.status_code == 200
assert health_response.json()["status"] == "ready"
assert_clean("health")

models_response = client.get("/v1/models", params={"season": season.slug})
assert models_response.status_code == 200
assert models_response.json()["models"] == []
assert_clean("models")

leaderboard_response = client.get(
    "/v1/leaderboards",
    params={
        "season": season.slug,
        "track": "model_arena",
        "rater_cohort": "public",
        "task_family": "all",
    },
)
assert leaderboard_response.status_code == 200
assert leaderboard_response.json()["rows"] == []
assert leaderboard_response.json()["official"] is False
assert_clean("leaderboard")

admin_response = client.post(
    "/v1/admin/leaderboards/snapshot",
    params={"season": season.slug},
)
assert admin_response.status_code == 409
assert "/v1/admin/leaderboards/snapshot-jobs" in admin_response.json()["detail"]
assert_clean("admin refusal")

try:
    main._create_leaderboard_snapshot(
        session,
        season.slug,
        "model_arena",
        "public",
        "all",
        "public_freeform",
        None,
    )
except InProcessSnapshotAnalysisForbidden as exc:
    assert "/v1/admin/leaderboards/snapshot-jobs" in str(exc)
else:
    raise AssertionError("private snapshot helper entered in-process analysis")
assert_clean("snapshot helper refusal")

try:
    _fit_bradley_terry([("a", "b", 1.0), ("a", "b", 0.0)])
except InProcessSnapshotAnalysisForbidden as exc:
    assert "/v1/admin/leaderboards/snapshot-jobs" in str(exc)
else:
    raise AssertionError("production API entered Bradley-Terry fitting")
assert_clean("fit refusal")

print(json.dumps({"status": "pass", "loaded": loaded_analysis_runtime_roots()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_production_api_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"status": "pass", "loaded": []}


def test_production_worker_retains_snapshot_analysis_path(monkeypatch) -> None:
    from flavourbench import service_ranking

    monkeypatch.setattr(
        service_ranking,
        "get_settings",
        lambda: SimpleNamespace(environment="production", service_role="worker"),
    )

    service_ranking.require_snapshot_analysis_process()
