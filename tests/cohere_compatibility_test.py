from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from flavourbench.cohere_compatibility import (
    CohereCompatibilityError,
    CohereTarget,
    _headers,
    _raise_for_status,
    _redacted_error,
    freeze_catalog,
    project_cohere_strict_schema,
    project_find_pairings_evidence,
    select_targets,
    smoke_target,
)
from flavourbench.mcp_client import McpToolResult
from flavourbench.real_task_bank import sha256_json


def model_entry(name: str) -> dict[str, object]:
    return {
        "name": name,
        "is_deprecated": False,
        "endpoints": ["chat"],
        "context_length": 128000,
        "features": ["tool-use"],
    }


@pytest.mark.asyncio
async def test_catalog_is_sorted_and_content_addressed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    model_entry("command-a-reasoning-08-2025"),
                    model_entry("command-a-plus-05-2026"),
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.cohere.test/",
        transport=httpx.MockTransport(handler),
    ) as client:
        path, payload = await freeze_catalog(client, tmp_path)

    assert [item["name"] for item in payload["models"]] == [
        "command-a-plus-05-2026",
        "command-a-reasoning-08-2025",
    ]
    assert path.name == f"cohere-catalog-{sha256_json(payload)}.json"
    document = json.loads(path.read_text())
    assert document["artifact_sha256"] == sha256_json(payload)


def test_target_selection_requires_live_chat_model() -> None:
    catalog = {
        "models": [
            model_entry("command-a-plus-05-2026"),
            {
                **model_entry("retired"),
                "is_deprecated": True,
            },
        ]
    }
    targets = select_targets(catalog, ["command-a-plus-05-2026"])
    assert targets[0].model_id == "command-a-plus-05-2026"
    with pytest.raises(CohereCompatibilityError, match="deprecated"):
        select_targets(catalog, ["retired"])
    with pytest.raises(CohereCompatibilityError, match="absent"):
        select_targets(catalog, ["missing"])


def test_credentials_are_required_and_redacted() -> None:
    with pytest.raises(CohereCompatibilityError, match="COHERE_API_KEY"):
        _headers("")
    error = RuntimeError("Authorization: Bearer secret-value api_key=another-secret")
    rendered = _redacted_error(error)
    assert "secret-value" not in rendered
    assert "another-secret" not in rendered


def test_provider_validation_detail_is_preserved_without_raw_headers() -> None:
    response = httpx.Response(
        400,
        json={"message": "invalid request: unsupported parameter"},
        request=httpx.Request(
            "POST",
            "https://api.cohere.test/v2/chat",
            headers={"Authorization": "Bearer secret-value"},
        ),
    )
    with pytest.raises(
        CohereCompatibilityError,
        match="Cohere HTTP 400: invalid request: unsupported parameter",
    ):
        _raise_for_status(response)


def test_strict_schema_projection_removes_only_documented_unsupported_constraints() -> None:
    source = {
        "type": "object",
        "title": "Pairing input",
        "properties": {
            "ingredients": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 120,
                        },
                        "minItems": 1,
                        "maxItems": 12,
                    },
                    {"type": "string", "minLength": 1, "maxLength": 120},
                ]
            },
            "mode": {"type": "string", "enum": ["graph", "direct"]},
        },
        "required": ["ingredients"],
    }

    projected = project_cohere_strict_schema(source)

    assert projected["type"] == "object"
    assert projected["title"] == "Pairing input"
    assert projected["required"] == ["ingredients"]
    assert projected["properties"]["mode"]["enum"] == ["graph", "direct"]
    assert projected["properties"]["ingredients"]["anyOf"][0] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert projected["properties"]["ingredients"]["anyOf"][1] == {"type": "string"}
    assert source["properties"]["ingredients"]["anyOf"][0]["minItems"] == 1


def test_find_pairings_projection_is_bounded_and_deterministic() -> None:
    text = """Pairing graph for: pear
(reference scores: p10=0.02, median=0.09, p90=0.19)

CLUSTERS (primaries grouped by shared secondary connections):
  A [Fruit - 3 nodes, dense]: apple (0.416), apricot (0.351), almond (0.316)
  B [Fruit - 1 node, isolated]: fig (0.319)

BRIDGES (secondaries connecting multiple primaries):
  honey -> apple, fig, apricot (3 of 4 primaries)
  walnut -> apricot, almond (2 of 4 primaries)
"""

    projection, rendered = project_find_pairings_evidence(text)

    assert projection == {
        "projection_version": "find-pairings-synthesis-v2",
        "query": "pear",
        "top_associations": [
            {"ingredient": "apple", "score": 0.416},
            {"ingredient": "apricot", "score": 0.351},
            {"ingredient": "almond", "score": 0.316},
            {"ingredient": "fig", "score": 0.319},
        ],
        "bridge_ingredients": ["honey", "walnut"],
    }
    assert rendered.startswith("Statistical associations for pear:")
    assert "Multi-primary bridge ingredients include honey, walnut" in rendered


@pytest.mark.asyncio
async def test_real_contract_shape_uses_strict_tool_then_json_schema(
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
                    "id": "cohere-tool-generation",
                    "finish_reason": "TOOL_CALL",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "find_pairings",
                                    "arguments": {"ingredients": ["pear"]},
                                },
                            }
                        ],
                    },
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "cohere-final-generation",
                "finish_reason": "COMPLETE",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "answer_markdown": "Pear, blue cheese, and walnut.",
                                    "ingredient_mentions": ["pear", "blue cheese", "walnut"],
                                    "constraints_addressed": [],
                                    "uncertainties": ["Taste remains subjective."],
                                }
                            ),
                        }
                    ],
                },
                "usage": {"input_tokens": 80, "output_tokens": 30},
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
                text="""Pairing graph for: pear
CLUSTERS (primaries grouped by shared secondary connections):
  A [Fruit - 1 node, dense]: blue cheese (0.810)
BRIDGES (secondaries connecting multiple primaries):
  walnut -> pear, blue cheese (2 of 2 primaries)
""",
                structured={"pairings": [{"ingredient": "blue cheese", "score": 0.81}]},
                latency_ms=12,
                is_error=False,
            )

    monkeypatch.setattr(
        "flavourbench.cohere_compatibility.McpSession",
        FakeMcpSession,
    )
    entry = model_entry("command-a-plus-05-2026")
    target = CohereTarget(
        model_id="command-a-plus-05-2026",
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
        base_url="https://api.cohere.test/",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await smoke_target(target, tool, "b" * 64, client)

    assert result["status"] == "smoke_passed"
    assert result["real_provider_calls"] == 2
    assert result["real_epicure_calls"] == 1
    assert requests[0]["strict_tools"] is True
    assert "tool_choice" not in requests[0]
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "response_format" not in requests[0]
    assert "tools" not in requests[1]
    assert requests[1]["thinking"] == {"type": "disabled"}
    assert "statistical" in requests[1]["messages"][0]["content"]
    assert requests[1]["response_format"]["schema"]["additionalProperties"] is False
    assert (
        "minLength"
        not in requests[1]["response_format"]["schema"]["properties"]["answer_markdown"]
    )
    assert result["usage"] == {
        "input_tokens": 130,
        "output_tokens": 50,
        "reasoning_tokens": 0,
        "billed_input_tokens": 0,
        "billed_output_tokens": 0,
    }
