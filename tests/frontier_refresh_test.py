from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.frontier_refresh import (
    FrontierRefreshError,
    _atomic_write,
    _redacted_error,
    _validate_roster,
    run_smokes,
)


def _roster() -> dict[str, object]:
    return {
        "schema_version": "flavourbench-frontier-refresh-roster-v1",
        "slots": [
            {
                "snapshot_model_id": "model-1",
                "provider": "bedrock",
                "endpoint_id": "global.example.model-1",
            },
            {
                "snapshot_model_id": "model-2",
                "provider": "openrouter",
                "endpoint_id": "example/model-2",
            },
        ],
    }


def test_frontier_refresh_roster_requires_unique_model_and_endpoint_ids() -> None:
    roster = _roster()
    assert len(_validate_roster(roster)) == 2
    duplicate = _roster()
    slots = duplicate["slots"]
    assert isinstance(slots, list)
    assert isinstance(slots[1], dict)
    slots[1]["snapshot_model_id"] = "model-1"
    with pytest.raises(FrontierRefreshError, match="model IDs"):
        _validate_roster(duplicate)


def test_frontier_refresh_redacts_credentials_and_account_ids() -> None:
    error = RuntimeError("account 123456789012 rejected sk-secret-value-123456789")
    redacted = _redacted_error(error)
    assert "123456789012" not in redacted
    assert "sk-secret" not in redacted
    assert "<account-redacted>" in redacted
    assert "<credential-redacted>" in redacted


def test_frontier_refresh_artifacts_are_content_addressed(tmp_path: Path) -> None:
    payload = {"schema_version": "test-v1", "value": 4}
    first = _atomic_write(tmp_path, "artifact", payload)
    second = _atomic_write(tmp_path, "artifact", payload)
    assert first == second
    document = json.loads(first.read_text())
    assert document["value"] == 4
    assert document["artifact_sha256"] == first.stem.rsplit("-", 1)[-1]


@pytest.mark.asyncio
async def test_frontier_refresh_rejects_unknown_or_empty_providers(tmp_path: Path) -> None:
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps(_roster()))
    common = {
        "roster_path": roster,
        "bedrock_catalog_path": tmp_path / "bedrock.json",
        "route_catalog_path": tmp_path / "routes.json",
        "tool_catalog_path": tmp_path / "tools.json",
        "output_dir": tmp_path / "output",
        "base_url": "https://example.invalid/api/v1",
        "cap_usd": Decimal("4"),
        "concurrency": 1,
    }
    with pytest.raises(FrontierRefreshError, match="providers"):
        await run_smokes(**common, providers=frozenset())
    with pytest.raises(FrontierRefreshError, match="providers"):
        await run_smokes(**common, providers=frozenset({"unknown"}))
