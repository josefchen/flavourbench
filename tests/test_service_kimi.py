from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from flavourbench.provider import GenerationSpec, ProviderError
from flavourbench.service_kimi import KimiDirectProvider


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode="mock",
        kimi_api_key="test-key",
        kimi_base_url="https://api.kimi.test/coding/v1",
        kimi_timeout_seconds=30,
        max_provider_attempts=1,
        max_output_tokens=1024,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=7,
        max_tool_rounds=8,
        max_tool_calls_per_round=4,
        max_tool_calls_total=16,
        max_tool_result_bytes=32_768,
        max_cumulative_tool_result_bytes=98_304,
    )


def _spec() -> GenerationSpec:
    return GenerationSpec(
        arm_id="arm-kimi",
        battle_id="battle-kimi",
        prompt="Give one practical pear and miso pairing.",
        category="composition",
        model_id="moonshotai/kimi-k3",
        model_name="Kimi K3",
        provider_slug="kimi-code-direct",
        condition="epicure_off",
        idempotency_key="kimi-direct-test",
        execution_backend="kimi_direct",
        rate_card_json={
            "prompt_price_per_token": "0.000001",
            "completion_price_per_token": "0.000002",
            "request_price": "0",
            "internal_reasoning_price_per_token": "0",
        },
        backend_contract_json={
            "schema_version": "flavourbench-kimi-direct-endpoint-contract-v1",
            "base_url": "https://api.kimi.test/coding/v1",
            "requested_model_id": "k3",
            "expected_actual_provider_slug": "kimi-code-direct",
            "catalog_sha256": "a" * 64,
            "catalog_entry_sha256": "b" * 64,
            "allow_fallbacks": False,
            "season_eligible": False,
        },
        supported_parameters=frozenset(
            {
                "max_tokens",
                "temperature",
                "response_format",
                "structured_outputs",
                "tool_choice",
                "tools",
            }
        ),
        decoding_parameters={"max_tokens": 1024, "temperature": 0.2},
        expected_actual_model_id="k3",
        expected_actual_provider_slug="kimi-code-direct",
        endpoint_contract_sha256="c" * 64,
        protocol_bundle_sha256="d" * 64,
    )


@pytest.mark.asyncio
async def test_direct_kimi_uses_exact_model_without_openrouter_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_kimi.get_settings", _settings)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-kimi-direct",
                "model": "k3",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer_markdown": "Pair pear with white miso and walnut.",
                                    "ingredient_mentions": ["pear", "white miso", "walnut"],
                                    "constraints_addressed": [],
                                    "uncertainties": ["Balance depends on miso salinity."],
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    provider = KimiDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.kimi.test/coding/v1/",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate(_spec())
    await provider.aclose()

    assert len(requests) == 1
    assert requests[0]["model"] == "k3"
    assert "provider" not in requests[0]
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert result.actual_model_id == "k3"
    assert result.provider_slug == "kimi-code-direct"
    assert result.generation_ids == ["chatcmpl-kimi-direct"]
    assert result.cost_micros == 20
    assert result.cost_reconciled is False
    assert result.cost_accounting_basis == "frozen_rate_card_times_kimi_returned_usage"
    assert result.billing_reconciliation_status == "provider_charge_unavailable"


@pytest.mark.asyncio
async def test_direct_kimi_rejects_model_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_kimi.get_settings", _settings)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-substitution",
                "model": "kimi-for-coding",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            },
        )

    provider = KimiDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.kimi.test/coding/v1/",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError, match="Kimi request failed"):
        await provider.generate(_spec())
    await provider.aclose()
