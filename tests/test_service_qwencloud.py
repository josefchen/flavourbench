from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from flavourbench.budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from flavourbench.provider import GenerationSpec, ProviderError
from flavourbench.qwencloud_catalog import QWEN38_TOOL_AUTO_INSTRUCTION
from flavourbench.real_task_bank import sha256_json
from flavourbench.service_qwencloud import (
    QWENCLOUD_ACCOUNTING_BASIS,
    QWENCLOUD_MESSAGE_CANONICALIZATION,
    QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS,
    QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS,
    QWENCLOUD_TOOL_CHOICE_TRANSPORT,
    QwenCloudDirectProvider,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode="mock",
        qwencloud_api_key="sk-ws-test-not-real",
        qwencloud_base_url=(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ),
        qwencloud_timeout_seconds=30,
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
    model_id = "qwen3.7-max-2026-06-08"
    return GenerationSpec(
        arm_id="arm-qwencloud",
        battle_id="battle-qwencloud",
        prompt="Give one practical pear and miso pairing.",
        category="composition",
        model_id=model_id,
        model_name="Qwen 3.7 Max 2026-06-08",
        provider_slug="qwencloud-direct",
        condition="epicure_off",
        idempotency_key="qwencloud-direct-test",
        final_response_mode="plain_text",
        execution_backend="qwencloud_direct",
        rate_card_json={
            "prompt_price_per_token": "0.0000025",
            "completion_price_per_token": "0.0000075",
            "request_price": "0",
            "internal_reasoning_price_per_token": "0",
        },
        backend_contract_json={
            "schema_version": "flavourbench-qwencloud-direct-endpoint-contract-v1",
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "requested_model_id": model_id,
            "expected_actual_provider_slug": "qwencloud-direct",
            "catalog_sha256": "a" * 64,
            "catalog_entry_sha256": "b" * 64,
            "identity_kind": "immutable_dated_release",
            "structured_outputs_supported": False,
            "allow_fallbacks": False,
            "cost_reconciliation": "provider_charge_unavailable",
            "season_eligible": False,
            "rank_eligible": False,
        },
        supported_parameters=frozenset(
            {"max_tokens", "temperature", "tool_choice", "tools", "top_p"}
        ),
        decoding_parameters={"max_tokens": 1024, "temperature": 0.2, "top_p": 0.95},
        expected_actual_model_id=model_id,
        expected_actual_provider_slug="qwencloud-direct",
        endpoint_contract_sha256="c" * 64,
        protocol_bundle_sha256="d" * 64,
    )


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flavourbench.service_qwencloud.get_settings", _settings)
    monkeypatch.setattr("flavourbench.service_kimi.get_settings", _settings)


def _mutable_alias_spec(*, opted_in: bool) -> GenerationSpec:
    base = _spec()
    contract = {
        **base.backend_contract_json,
        "requested_model_id": "qwen3.8-max",
        "identity_kind": "mutable_alias",
        "catalog_observed_at": "2026-08-08T16:29:04Z",
        "catalog_pinned_at_observation": True,
        "model_identity_label": "catalog_pinned_at_observation_not_a_frozen_model",
        "mutable_alias_execution_requires_explicit_opt_in": True,
        "provider_rate_status": "unpublished_at_catalog_observation",
        "cost_reconciliation": "provider_rate_and_charge_unavailable",
        "official": False,
        "tool_choice_transport_mode": QWENCLOUD_TOOL_CHOICE_TRANSPORT,
        "tool_choice_required_supported": False,
        "required_success_postcondition": (
            "at_least_one_successful_real_epicure_tool_trace"
        ),
        "tool_selection_system_instruction": QWEN38_TOOL_AUTO_INSTRUCTION,
        "tool_selection_system_instruction_sha256": sha256_json(
            QWEN38_TOOL_AUTO_INSTRUCTION
        ),
        "message_canonicalization": QWENCLOUD_MESSAGE_CANONICALIZATION,
        "predecessor_failure_artifact_sha256": "e" * 64,
    }
    return replace(
        base,
        model_id="qwen3.8-max",
        model_name="Qwen 3.8 Max (catalog-pinned mutable alias)",
        expected_actual_model_id="qwen3.8-max",
        backend_contract_json=contract,
        rate_card_json={
            "prompt_price_per_token": "0",
            "completion_price_per_token": "0",
            "request_price": "0",
            "internal_reasoning_price_per_token": "0",
        },
        allow_mutable_alias_exploratory=opted_in,
    )


@pytest.mark.asyncio
async def test_qwencloud_plain_text_uses_exact_model_and_keeps_cost_unreconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-qwencloud-direct",
                "model": "qwen3.7-max-2026-06-08",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Pair pear with white miso and walnut."},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = QwenCloudDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate(_spec())
    await provider.aclose()

    assert len(requests) == 1
    assert requests[0]["model"] == "qwen3.7-max-2026-06-08"
    assert "provider" not in requests[0]
    assert "response_format" not in requests[0]
    assert "reasoning" not in requests[0]
    assert result.actual_model_id == "qwen3.7-max-2026-06-08"
    assert result.provider_slug == "qwencloud-direct"
    assert result.generation_ids == ["chatcmpl-qwencloud-direct"]
    assert result.cost_micros == 63
    assert result.cost_reconciled is False
    assert result.cost_accounting_basis == QWENCLOUD_ACCOUNTING_BASIS
    assert result.billing_reconciliation_status == "provider_charge_unavailable"


@pytest.mark.asyncio
async def test_qwencloud_rejects_unsupported_structured_final_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    provider = QwenCloudDirectProvider()
    with pytest.raises(ProviderError, match="lacks frozen structured-output support"):
        await provider.generate(replace(_spec(), final_response_mode="structured_json"))
    await provider.aclose()


@pytest.mark.asyncio
async def test_qwencloud_rejects_unfrozen_reasoning_translation_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    provider = QwenCloudDirectProvider()
    spec = replace(
        _spec(),
        supported_parameters=_spec().supported_parameters | {"reasoning"},
        intermediate_reasoning_effort="low",
        final_reasoning_effort="low",
    )
    with pytest.raises(ProviderError, match="no frozen low/high/max"):
        await provider.generate(spec)
    await provider.aclose()


@pytest.mark.asyncio
async def test_mutable_qwen38_alias_requires_explicit_opt_in_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    provider = QwenCloudDirectProvider()
    with pytest.raises(ProviderError, match="explicit catalog-pinned exploratory"):
        await provider.generate(_mutable_alias_spec(opted_in=False))
    await provider.aclose()


@pytest.mark.asyncio
async def test_mutable_qwen38_alias_records_usage_but_never_invents_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "qwen3.8-max"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-qwen38-alias-observation",
                "model": "qwen3.8-max",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Use pear, white miso, and toasted walnut."},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    provider = QwenCloudDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate(_mutable_alias_spec(opted_in=True))
    await provider.aclose()

    assert result.actual_model_id == "qwen3.8-max"
    assert result.cost_micros == 0
    assert result.cost_reconciled is False
    assert result.cost_accounting_basis == QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
    assert result.billing_reconciliation_status == (
        QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
    )
    assert result.generation_metadata[0]["provider_cost_known"] is False
    assert result.generation_metadata[0]["cost_micros"] == 0


@pytest.mark.asyncio
async def test_qwen38_successor_uses_auto_and_documented_tool_continuation_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-successor-{len(requests)}",
                "model": "qwen3.8-max",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "accepted"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    provider = QwenCloudDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
        transport=httpx.MockTransport(handler),
    )
    spec = _mutable_alias_spec(opted_in=True)
    provider._spec_by_arm[spec.arm_id] = spec  # noqa: SLF001
    await provider._post(  # noqa: SLF001
        {
            "model": "qwen3.8-max",
            "messages": [
                {"role": "system", "content": "System contract."},
                {"role": "user", "content": "Use evidence."},
                {
                    "role": "assistant",
                    "content": "Planning note.",
                    "reasoning_details": [{"unsupported": True}],
                    "refusal": None,
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "find_pairings",
                        "description": "Find pairings.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 16,
        },
        "successor-initial",
        arm_id=spec.arm_id,
        phase="tool_round_0",
    )
    await provider._post(  # noqa: SLF001
        {
            "model": "qwen3.8-max",
            "messages": [
                {"role": "system", "content": "System contract."},
                {"role": "user", "content": "Use evidence."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "find_pairings",
                                "arguments": "{}",
                            },
                        }
                    ],
                    "audio": None,
                    "function_call": None,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "find_pairings",
                    "content": "bounded result",
                },
            ],
            "tools": [],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
        "successor-continuation",
        arm_id=spec.arm_id,
        phase="tool_round_1",
    )
    await provider.aclose()

    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["messages"][0]["content"].endswith(
        QWEN38_TOOL_AUTO_INSTRUCTION
    )
    assert requests[0]["messages"][2] == {
        "role": "assistant",
        "content": "Planning note.",
    }
    assert requests[1]["messages"][2].keys() == {
        "role",
        "content",
        "tool_calls",
    }
    assert requests[1]["messages"][3] == {
        "role": "tool",
        "content": "bounded result",
        "tool_call_id": "call-1",
    }


def test_qwencloud_has_installation_wide_governed_cap() -> None:
    assert provider_account_hard_cap_micros("qwencloud_direct") == 100_000_000
    assert len(provider_account_scope_sha256("qwencloud_direct")) == 64
