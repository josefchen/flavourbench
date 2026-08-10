from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from flavourbench.frontier_manifest import (
    BudgetExceeded,
    ForecastPolicy,
    PanelResolutionError,
    PanelSlot,
    build_candidate_manifest,
    discover_candidate_manifest,
    resolve_requested_names,
    verify_manifest_content_address,
    write_content_addressed_manifest,
)

TEST_SLOT = PanelSlot(
    slot_id="test-frontier",
    cohort="closed_frontier",
    model_id="vendor/model-v1",
    rationale="Deterministic test coverage.",
)


def _model(model_id: str = "vendor/model-v1") -> dict:
    return {
        "id": model_id,
        "canonical_slug": f"{model_id}-20260715",
        "name": model_id,
        "created": 1_784_000_000,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "test",
        },
        "context_length": 10_000,
        "pricing": {"prompt": "0.002", "completion": "0.003"},
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tool_choice",
            "tools",
            "temperature",
        ],
        "top_provider": {"context_length": 10_000},
    }


def _endpoint(
    tag: str = "provider/flex",
    *,
    prompt: str = "0.001",
    completion: str = "0.002",
    supported: list[str] | None = None,
) -> dict:
    return {
        "name": f"Provider | {tag}",
        "provider_name": "Provider",
        "tag": tag,
        "model_id": "vendor/model-v1",
        "quantization": "unknown",
        "context_length": 10_000,
        "max_completion_tokens": 1_000,
        "status": 0,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": supported
        if supported is not None
        else [
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tool_choice",
            "tools",
            "temperature",
        ],
    }


def _documents(endpoints: list[dict] | None = None) -> tuple[dict, dict]:
    catalog = {"data": [_model()]}
    endpoint_document = {
        "data": {
            "id": "vendor/model-v1",
            "endpoints": endpoints
            if endpoints is not None
            else [
                _endpoint("provider/priority", prompt="0.002", completion="0.003"),
                _endpoint(),
            ],
        }
    }
    return catalog, endpoint_document


def _policy() -> ForecastPolicy:
    return ForecastPolicy(
        arms_per_model=1,
        max_generations_per_arm=2,
        max_prompt_tokens_per_generation=100,
        max_completion_tokens_per_generation=50,
        max_reasoning_tokens_per_generation=20,
    )


def test_manifest_freezes_exact_endpoint_and_conservative_forecast() -> None:
    catalog, endpoints = _documents()
    manifest = build_candidate_manifest(
        catalog,
        {"vendor/model-v1": endpoints},
        cap_usd="1",
        forecast_policy=_policy(),
        panel=[TEST_SLOT],
        requested_names=[],
        observed_at="2026-07-15T00:00:00Z",
    )

    entry = manifest["models"][0]
    assert entry["endpoint"]["tag"] == "provider/flex"
    assert entry["request_policy"]["provider"] == {
        "only": ["provider/flex"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert entry["request_policy"]["endpoint_retention_attestation"] == (
        "not_present_in_endpoint_metadata"
    )
    assert Decimal(entry["forecast"]["per_generation_usd"]) == Decimal("0.2")
    assert Decimal(manifest["budget"]["bounded_forecast_usd"]) == Decimal("0.4")
    assert manifest["status"] == "unranked_candidate"
    assert manifest["generation_calls_made"] == 0
    assert verify_manifest_content_address(manifest)


def test_search_resolution_is_explicit_and_mutable_aliases_are_excluded() -> None:
    catalog = [
        {"id": "openai/gpt-5.6-sol", "name": "GPT 5.6 Sol", "created": 2},
        {"id": "openai/gpt-5.6-sol-pro", "name": "GPT 5.6 Sol Pro", "created": 3},
        {"id": "~anthropic/claude-fable-latest", "name": "Fable Latest", "created": 4},
        {"id": "anthropic/claude-fable-5", "name": "Claude Fable 5", "created": 3},
        {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8", "created": 3},
    ]

    resolutions = resolve_requested_names(catalog, ["5.6", "fable", "opus", "absent"])
    by_query = {item["query"]: item for item in resolutions}
    assert by_query["5.6"]["resolved_model_id"] == "openai/gpt-5.6-sol-pro"
    assert by_query["fable"]["resolved_model_id"] == "anthropic/claude-fable-5"
    assert by_query["fable"]["excluded_mutable_aliases"] == [
        "~anthropic/claude-fable-latest"
    ]
    assert by_query["opus"]["resolved_model_id"] == "anthropic/claude-opus-4.8"
    assert by_query["absent"]["status"] == "unresolved"


def test_endpoint_without_tools_and_structured_output_is_rejected() -> None:
    catalog, endpoints = _documents(
        [_endpoint(supported=["tools"]), _endpoint("provider/no-tools", supported=[])]
    )
    with pytest.raises(PanelResolutionError, match="no eligible endpoint"):
        build_candidate_manifest(
            catalog,
            {"vendor/model-v1": endpoints},
            cap_usd="1",
            forecast_policy=_policy(),
            panel=[TEST_SLOT],
            requested_names=[],
        )


def test_budget_cap_blocks_manifest_before_write() -> None:
    catalog, endpoints = _documents()
    with pytest.raises(BudgetExceeded, match=r"forecast \$0.4 exceeds cap \$0.1"):
        build_candidate_manifest(
            catalog,
            {"vendor/model-v1": endpoints},
            cap_usd="0.1",
            forecast_policy=_policy(),
            panel=[TEST_SLOT],
            requested_names=[],
        )


def test_content_addressed_write_detects_tampering(tmp_path) -> None:
    catalog, endpoints = _documents()
    manifest = build_candidate_manifest(
        catalog,
        {"vendor/model-v1": endpoints},
        cap_usd="1",
        forecast_policy=_policy(),
        panel=[TEST_SLOT],
        requested_names=[],
        observed_at="2026-07-15T00:00:00Z",
    )
    destination = write_content_addressed_manifest(manifest, tmp_path)
    assert destination.name.endswith(f"{manifest['content_address']['digest']}.json")
    assert json.loads(destination.read_text()) == manifest

    manifest["budget"]["cap_usd"] = "2"
    assert not verify_manifest_content_address(manifest)
    with pytest.raises(Exception, match="invalid content address"):
        write_content_addressed_manifest(manifest, tmp_path)


@pytest.mark.asyncio
async def test_live_discovery_surface_is_read_only() -> None:
    catalog, endpoints = _documents()
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json=catalog)
        if request.url.path == "/api/v1/models/vendor/model-v1/endpoints":
            return httpx.Response(200, json=endpoints)
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="https://openrouter.test/api/v1/",
        transport=httpx.MockTransport(handler),
    ) as client:
        manifest = await discover_candidate_manifest(
            api_key="test-only",
            cap_usd="1",
            forecast_policy=_policy(),
            panel=[TEST_SLOT],
            requested_names=[],
            observed_at="2026-07-15T00:00:00Z",
            client=client,
        )

    assert requested_paths == [
        "/api/v1/models",
        "/api/v1/models/vendor/model-v1/endpoints",
    ]
    assert all(
        "chat/completions" not in path and "generations" not in path
        for path in requested_paths
    )
    assert manifest["generation_spend_usd"] == "0"
