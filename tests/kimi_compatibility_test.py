from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from flavourbench.kimi_compatibility import (
    KimiCompatibilityError,
    KimiTarget,
    _headers,
    _redacted_error,
    freeze_catalog,
    select_targets,
    smoke_target,
)
from flavourbench.mcp_client import McpToolResult
from flavourbench.real_task_bank import sha256_json


def model_entry(model_id: str) -> dict[str, object]:
    return {"id": model_id, "object": "model", "owned_by": "moonshotai"}


@pytest.mark.asyncio
async def test_catalog_is_sorted_and_content_addressed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        return httpx.Response(
            200,
            json={"data": [model_entry("k3-256k"), model_entry("k3")]},
        )

    async with httpx.AsyncClient(
        base_url="https://api.kimi.test/",
        transport=httpx.MockTransport(handler),
    ) as client:
        path, payload = await freeze_catalog(client, tmp_path)

    assert [item["id"] for item in payload["models"]] == ["k3", "k3-256k"]
    assert path.name == f"kimi-catalog-{sha256_json(payload)}.json"
    assert json.loads(path.read_text())["artifact_sha256"] == sha256_json(payload)


def test_target_selection_requires_exact_catalog_id() -> None:
    catalog = {"models": [model_entry("k3")]}
    targets = select_targets(catalog, ["k3"])
    assert targets[0].model_id == "k3"
    with pytest.raises(KimiCompatibilityError, match="absent"):
        select_targets(catalog, ["kimi-k3"])


def test_credentials_and_errors_are_redacted() -> None:
    with pytest.raises(KimiCompatibilityError, match="KIMI_API_KEY"):
        _headers("")
    rendered = _redacted_error(
        RuntimeError("Authorization: Bearer hidden-token api_key=another sk-secretsecret")
    )
    assert "hidden-token" not in rendered
    assert "another" not in rendered
    assert "secretsecret" not in rendered


@pytest.mark.asyncio
async def test_real_contract_shape_uses_tool_then_strict_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "kimi-tool-generation",
                    "model": "k3",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "find_pairings",
                                            "arguments": json.dumps(
                                                {"ingredients": ["pear"]}
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "kimi-final-generation",
                "model": "k3",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "answer_markdown": "Pear, blue cheese, and walnut.",
                                    "ingredient_mentions": ["pear", "blue cheese", "walnut"],
                                    "constraints_addressed": [],
                                    "uncertainties": ["Taste remains subjective."],
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 30,
                    "total_tokens": 110,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        )

    class FakeMcpSession:
        async def __aenter__(self) -> FakeMcpSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def call_tool(self, name: str, arguments: dict[str, object]) -> McpToolResult:
            assert name == "find_pairings"
            assert arguments == {"ingredients": ["pear"]}
            return McpToolResult(
                text="Pear graph",
                structured={"pairings": [{"ingredient": "blue cheese", "score": 0.81}]},
                latency_ms=12,
                is_error=False,
            )

    monkeypatch.setattr("flavourbench.kimi_compatibility.McpSession", FakeMcpSession)
    entry = model_entry("k3")
    target = KimiTarget(
        model_id="k3",
        catalog_entry=entry,
        catalog_entry_sha256=sha256_json(entry),
        catalog_sha256="a" * 64,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "find_pairings",
            "description": "Pair ingredients",
            "parameters": {
                "type": "object",
                "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
                "required": ["ingredients"],
            },
        },
    }
    async with httpx.AsyncClient(
        base_url="https://api.kimi.test/",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await smoke_target(target, tool, "b" * 64, client)

    assert result["status"] == "smoke_passed"
    assert result["real_provider_calls"] == 2
    assert result["real_epicure_calls"] == 1
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["temperature"] == 1
    assert requests[1]["temperature"] == 1
    assert requests[1]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["json_schema"]["strict"] is True
    assert result["returned_model_ids"] == ["k3", "k3"]
    assert result["usage"] == {
        "prompt_tokens": 130,
        "completion_tokens": 50,
        "total_tokens": 180,
        "cached_tokens": 0,
        "reasoning_tokens": 10,
    }


@pytest.mark.asyncio
async def test_returned_model_mismatch_fails_before_epicure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "generation",
                "model": "another-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "no"}}],
            },
        )

    entry = model_entry("k3")
    target = KimiTarget(
        model_id="k3",
        catalog_entry=entry,
        catalog_entry_sha256=sha256_json(entry),
        catalog_sha256="a" * 64,
    )
    async with httpx.AsyncClient(
        base_url="https://api.kimi.test/",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(KimiCompatibilityError, match="expected exact catalog ID"):
            await smoke_target(target, {}, "b" * 64, client)
