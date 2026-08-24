from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from flavourbench.config import Settings
from flavourbench.epicure_native_powered_runner import _read_secret_file
from flavourbench.provider import GenerationSpec, ProviderError
from flavourbench.service_zai import (
    ZAI_CODING_ACCOUNTING_BASIS,
    ZAI_CODING_BILLING_STATUS,
    ZaiCodingDirectProvider,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode="mock",
        zai_coding_api_key="fake-zai-key",
        zai_coding_base_url="https://api.z.ai/api/anthropic",
        zai_coding_timeout_seconds=30,
        max_provider_attempts=1,
        max_output_tokens=1024,
        decoding_temperature=0.0,
        decoding_top_p=1.0,
        decoding_seed=7,
        max_tool_rounds=1,
        max_tool_calls_per_round=1,
        max_tool_calls_total=1,
        max_tool_result_bytes=4096,
        max_cumulative_tool_result_bytes=4096,
    )


def test_zai_continuation_key_takes_precedence_without_exposing_either_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_CODING_API_KEY", "fake-primary-key")
    monkeypatch.setenv("ZAI_CODING_API_KEY2", "fake-continuation-key")
    settings = Settings(_env_file=None)
    assert settings.zai_coding_api_key == "fake-continuation-key"


def test_powered_runner_maps_optional_zai_continuation_key(tmp_path: Path) -> None:
    source = tmp_path / "secrets.env"
    source.write_text(
        "OPENROUTER_API_KEY=fake-openrouter\n"
        "KIMI_API_KEY=fake-kimi\n"
        "COHERE_API_KEY=fake-cohere\n"
        "ZAI_CODING_API_KEY=fake-primary\n"
        "ZAI_CODING_API_KEY2=fake-continuation\n",
        encoding="utf-8",
    )
    loaded = _read_secret_file(source)
    assert loaded["FLAVOURBENCH_ZAI_CODING_API_KEY"] == "fake-primary"
    assert loaded["FLAVOURBENCH_ZAI_CODING_API_KEY2"] == "fake-continuation"


def _spec(*, authorized: bool = True) -> GenerationSpec:
    permission = {
        "provider": "Z.ai",
        "grantee": "Josef Chen",
        "scope": "one_finite_flavourbench_benchmark_run",
        "permanent_running_function_authorized": False,
        "user_attested_written_permission": authorized,
        "permission_quote_sha256": "a" * 64,
    }
    return GenerationSpec(
        arm_id="arm-zai",
        battle_id="battle-zai",
        prompt="Select three ingredients from the supplied candidates.",
        category="composition",
        model_id="z-ai/glm-5.3",
        model_name="GLM-5.3",
        provider_slug="zai-coding-plan-direct",
        condition="epicure_off",
        idempotency_key="zai-limited-run-test",
        execution_backend="zai_coding_direct",
        rate_card_json={
            "prompt_price_per_token": "0",
            "completion_price_per_token": "0",
            "request_price": "0",
            "internal_reasoning_price_per_token": "0",
        },
        backend_contract_json={
            "schema_version": "flavourbench-zai-coding-anthropic-contract-v1",
            "base_url": "https://api.z.ai/api/anthropic",
            "requested_model_id": "glm-5.3",
            "expected_actual_provider_slug": "zai-coding-plan-direct",
            "catalog_sha256": "b" * 64,
            "catalog_entry_sha256": "c" * 64,
            "identity_kind": "official_named_release",
            "rank_eligible_after_complete_block": True,
            "limited_run_permission": permission,
            "allow_fallbacks": False,
        },
        supported_parameters=frozenset({"max_tokens", "temperature"}),
        decoding_parameters={"max_tokens": 1024, "temperature": 0.0},
        expected_actual_model_id="glm-5.3",
        expected_actual_provider_slug="zai-coding-plan-direct",
        endpoint_contract_sha256="d" * 64,
        protocol_bundle_sha256="e" * 64,
        final_response_mode="plain_text",
    )


@pytest.mark.asyncio
async def test_zai_uses_anthropic_bearer_route_and_exact_glm_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_kimi.get_settings", _settings)
    requests: list[tuple[httpx.Request, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request, payload))
        return httpx.Response(
            200,
            json={
                "type": "message",
                "id": "glm-limited-run-generation-1",
                "model": "glm-5.3",
                "content": [{"type": "text", "text": "FINAL_SELECTION: pear, miso, walnut"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 8},
            },
        )

    provider = ZaiCodingDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.z.ai/api/anthropic/",
        headers={"Authorization": "Bearer fake-zai-key"},
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate(_spec())
    await provider.aclose()

    assert len(requests) == 1
    request, payload = requests[0]
    assert request.url.path == "/api/anthropic/v1/messages"
    assert request.headers["authorization"] == "Bearer fake-zai-key"
    assert payload["model"] == "glm-5.3"
    assert "provider" not in payload
    assert result.actual_model_id == "glm-5.3"
    assert result.provider_slug == "zai-coding-plan-direct"
    assert result.cost_micros == 0
    assert result.cost_reconciled is False
    assert result.cost_accounting_basis == ZAI_CODING_ACCOUNTING_BASIS
    assert result.billing_reconciliation_status == ZAI_CODING_BILLING_STATUS


@pytest.mark.asyncio
async def test_zai_rejects_contract_without_the_written_limited_run_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_kimi.get_settings", _settings)
    provider = ZaiCodingDirectProvider()
    with pytest.raises(ProviderError, match="limited-run permission"):
        await provider.generate(_spec(authorized=False))
    await provider.aclose()
