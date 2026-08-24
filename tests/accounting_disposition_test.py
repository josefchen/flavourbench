from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from flavourbench.accounting_disposition import (
    AccountingDispositionError,
    build_disposition,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "artifacts/season1/current-quality-run/preflight-v9-final-receipts/"
    "20260801T170427Z-b646fc304e20.json"
)
UNRESOLVED = "gen-1785603890-0WsdlhEiviS3F6OYBs81"


def _transport(*, explain_delta: bool) -> httpx.MockTransport:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    before = Decimal(str(source["budget"]["openrouter_key_before"]["usage_daily_usd"]))
    after = Decimal(str(source["budget"]["openrouter_key_after"]["usage_daily_usd"]))
    delta = after - before
    known_ids = [
        str(item["generation_id"])
        for result in source["results"].values()
        if isinstance(result, dict)
        for item in result["generation_metadata"]
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        generation_id = request.url.params["id"]
        if generation_id == UNRESOLVED:
            return httpx.Response(404, json={"error": "not found"})
        assert generation_id in known_ids
        cost = delta if explain_delta and generation_id == known_ids[0] else Decimal(0)
        return httpx.Response(
            200,
            json={
                "data": {
                    "total_cost": format(cost, "f"),
                    "model": "fixture/model",
                    "provider_name": "Fixture",
                    "finish_reason": "stop",
                }
            },
        )

    return httpx.MockTransport(handler)


def test_account_scope_disposition_closes_zero_cost_absent_generation() -> None:
    with httpx.Client(
        transport=_transport(explain_delta=True),
        base_url="https://openrouter.test/api/v1/",
    ) as client:
        disposition = build_disposition(
            source_artifact_path=SOURCE,
            unresolved_generation_id=UNRESOLVED,
            client=client,
        )

    assert disposition["unresolved_generation_incremental_cost_usd"] == "0"
    assert disposition["disposition"]["budget_accounting"] == (
        "closed_at_provider_account_scope"
    )
    assert disposition["disposition"]["source_preflight_eligible"] is False


def test_account_scope_disposition_rejects_unexplained_usage() -> None:
    with httpx.Client(
        transport=_transport(explain_delta=False),
        base_url="https://openrouter.test/api/v1/",
    ) as client:
        with pytest.raises(AccountingDispositionError, match="do not exactly explain"):
            build_disposition(
                source_artifact_path=SOURCE,
                unresolved_generation_id=UNRESOLVED,
                client=client,
            )
