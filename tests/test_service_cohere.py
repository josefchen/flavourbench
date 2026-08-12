from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from flavourbench.provider import GenerationSpec, _assistant_continuation_message
from flavourbench.service_cohere import (
    COHERE_ACCOUNTING_BASIS,
    COHERE_REASONING_PORTABLE_FINAL_MODE,
    COHERE_REASONING_PORTABLE_SELECTION_MODE,
    CohereDirectProvider,
    _openai_response,
    _request_payload,
    _safe_http_error_detail,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        execution_mode="mock",
        cohere_api_key="test-key",
        cohere_base_url="https://api.cohere.test",
        cohere_timeout_seconds=30,
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
        arm_id="arm-cohere",
        battle_id="battle-cohere",
        prompt="Give one practical pear and miso pairing.",
        category="composition",
        model_id="cohere/command-a-plus-05-2026",
        model_name="Command A Plus",
        provider_slug="cohere-direct",
        condition="epicure_off",
        idempotency_key="cohere-direct-test",
        execution_backend="cohere_direct",
        rate_card_json={
            "prompt_price_per_token": "0.000001",
            "completion_price_per_token": "0.000002",
            "request_price": "0",
            "internal_reasoning_price_per_token": "0",
        },
        backend_contract_json={
            "schema_version": "flavourbench-cohere-direct-endpoint-contract-v1",
            "base_url": "https://api.cohere.test",
            "requested_model_id": "command-a-plus-05-2026",
            "expected_actual_provider_slug": "cohere-direct",
            "catalog_sha256": "a" * 64,
            "catalog_entry_sha256": "b" * 64,
            "allow_fallbacks": False,
            "season_eligible": False,
        },
        supported_parameters=frozenset(
            {
                "max_tokens",
                "temperature",
                "top_p",
                "seed",
                "response_format",
                "structured_outputs",
                "tool_choice",
                "tools",
            }
        ),
        decoding_parameters={
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
            "seed": 7,
        },
        expected_actual_model_id="command-a-plus-05-2026",
        expected_actual_provider_slug="cohere-direct",
        endpoint_contract_sha256="c" * 64,
        protocol_bundle_sha256="d" * 64,
    )


@pytest.mark.asyncio
async def test_direct_cohere_normalizes_v2_response_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_cohere.get_settings", _settings)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "cohere-generation-1",
                "finish_reason": "COMPLETE",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "not retained"},
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "answer_markdown": "Pair pear with white miso.",
                                    "ingredient_mentions": ["pear", "white miso"],
                                    "constraints_addressed": [],
                                    "uncertainties": ["Adjust for salinity."],
                                }
                            ),
                        },
                    ],
                },
                "usage": {
                    "tokens": {"input_tokens": 10, "output_tokens": 5},
                    "billed_units": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        )

    provider = CohereDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.cohere.test/",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.generate(_spec())
    await provider.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "command-a-plus-05-2026"
    assert request["p"] == 0.95
    assert "top_p" not in request and "provider" not in request
    assert request["response_format"]["type"] == "json_object"
    assert result.actual_model_id == "command-a-plus-05-2026"
    assert result.provider_slug == "cohere-direct"
    assert result.generation_ids == ["cohere-generation-1"]
    assert result.cost_micros == 20
    assert result.cost_reconciled is False
    assert result.cost_accounting_basis == COHERE_ACCOUNTING_BASIS
    assert result.billing_reconciliation_status == "provider_charge_unavailable"


def test_cohere_tool_round_trip_translation_is_lossless() -> None:
    request = _request_payload(
        {
            "model": "command-a-plus-05-2026",
            "max_tokens": 2048,
            "reasoning": {"effort": "low", "exclude": True},
            "messages": [
                {"role": "user", "content": "Find a pairing."},
                {
                    "role": "assistant",
                    "content": "I will query Epicure.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "find_pairings",
                                "arguments": '{"ingredients":["pear","miso"]}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "name": "find_pairings",
                    "content": "Pairing graph for pear and miso.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "find_pairings",
                        "parameters": {
                            "type": "object",
                            "properties": {"ingredients": {"type": "array"}},
                            "required": ["ingredients"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        }
    )
    assert "tool_choice" not in request
    assert request["strict_tools"] is True
    assert request["messages"][-1]["content"].endswith(
        "Do not return a text-only response in this turn."
    )
    assert request["thinking"] == {"type": "enabled", "token_budget": 512}
    assert request["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert request["messages"][1]["content"] == "I will query Epicure."
    assert "tool_plan" not in request["messages"][1]
    assert request["messages"][2]["tool_call_id"] == "call-1"

    normalized = _openai_response(
        {
            "id": "generation-tool",
            "finish_reason": "TOOL_CALL",
            "message": {
                "tool_plan": "I will query Epicure.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "find_pairings",
                            "arguments": '{"ingredients":["pear","miso"]}',
                        },
                    }
                ],
            },
            "usage": {"tokens": {"input_tokens": 8, "output_tokens": 4}},
        },
        response_model="command-a-plus-05-2026",
    )
    assert normalized["choices"][0]["finish_reason"] == "tool_calls"
    assert normalized["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
    assert normalized["choices"][0]["message"]["_cohere_tool_plan"] == ("I will query Epicure.")


def test_cohere_reasoning_continuation_preserves_opaque_blocks_but_hides_thinking() -> None:
    raw_content = [
        {"type": "thinking", "thinking": "private chain", "signature": "signed-1"},
        {"type": "text", "text": "Visible planning note."},
    ]
    normalized = _openai_response(
        {
            "id": "generation-reasoning",
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": raw_content},
            "usage": {"tokens": {"input_tokens": 8, "output_tokens": 4}},
        },
        response_model="command-a-reasoning-08-2025",
    )
    message = normalized["choices"][0]["message"]
    assert message["content"] == "Visible planning note."
    assert "private chain" not in message["content"]
    assert message["_cohere_content"] == raw_content

    continuation = _assistant_continuation_message(
        message,
        empty_content_fallback="Planning completed without a visible note.",
    )
    projected = _request_payload(
        {
            "model": "command-a-reasoning-08-2025",
            "messages": [
                {"role": "user", "content": "Plan a pairing."},
                continuation,
                {"role": "user", "content": "Continue."},
            ],
        }
    )
    assert projected["messages"][1]["content"] == raw_content
    assert projected["messages"][1]["content"][0]["thinking"] == "private chain"


def test_cohere_tool_turn_replays_tool_plan_content_and_exact_result_id() -> None:
    raw_content = [{"type": "thinking", "thinking": "choose one tool", "signature": "signed-2"}]
    normalized = _openai_response(
        {
            "id": "generation-tool-plan",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "content": raw_content,
                "tool_plan": "Query the pairing graph.",
                "tool_calls": [
                    {
                        "id": "exact-call-id",
                        "type": "function",
                        "function": {
                            "name": "find_pairings",
                            "arguments": '{"ingredients":["pear","miso"]}',
                        },
                    }
                ],
            },
            "usage": {"tokens": {"input_tokens": 8, "output_tokens": 4}},
        },
        response_model="command-a-reasoning-08-2025",
    )
    assistant = normalized["choices"][0]["message"]
    projected = _request_payload(
        {
            "model": "command-a-reasoning-08-2025",
            "messages": [
                {"role": "user", "content": "Find a pairing."},
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": "exact-call-id",
                    "name": "find_pairings",
                    "content": "Pairing graph for pear and miso.",
                },
            ],
        }
    )
    replayed = projected["messages"][1]
    assert replayed["content"] == raw_content
    assert replayed["tool_plan"] == "Query the pairing graph."
    assert replayed["tool_calls"][0]["id"] == "exact-call-id"
    assert projected["messages"][2]["tool_call_id"] == "exact-call-id"


def test_cohere_http_error_detail_is_bounded_and_header_free() -> None:
    request = httpx.Request(
        "POST",
        "https://api.cohere.test/v2/chat",
        headers={"Authorization": "Bearer must-not-appear"},
    )
    response = httpx.Response(
        400,
        request=request,
        json={"message": "invalid request: " + "x" * 800},
    )
    detail = _safe_http_error_detail(response)
    assert len(detail) == 500
    assert detail.startswith("invalid request:")
    assert "must-not-appear" not in detail


@pytest.mark.asyncio
async def test_cohere_reasoning_disables_thinking_only_for_portable_json_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_cohere.get_settings", _settings)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cohere-selection-1",
                "finish_reason": "COMPLETE",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": '{"name":"neighbors","arguments":{"ingredient":"pear"}}',
                        }
                    ],
                },
                "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
            },
        )

    spec = _spec()
    spec = GenerationSpec(
        **{
            **spec.__dict__,
            "model_id": "cohere/command-a-reasoning-08-2025",
            "expected_actual_model_id": "command-a-reasoning-08-2025",
            "evidence_protocol": "portable_text_tool_v1",
            "backend_contract_json": {
                **spec.backend_contract_json,
                "requested_model_id": "command-a-reasoning-08-2025",
                "portable_tool_selection_reasoning": (COHERE_REASONING_PORTABLE_SELECTION_MODE),
                "portable_final_reasoning": COHERE_REASONING_PORTABLE_FINAL_MODE,
            },
        }
    )
    provider = CohereDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.cohere.test/",
        transport=httpx.MockTransport(handler),
    )
    provider._spec_by_arm[spec.arm_id] = spec
    try:
        await provider._post(
            {
                "model": "command-a-reasoning-08-2025",
                "messages": [{"role": "user", "content": "Return exact JSON."}],
                "max_tokens": 1024,
            },
            "selection-key",
            arm_id=spec.arm_id,
            phase="portable_tool_selection",
        )
        await provider._post(
            {
                "model": "command-a-reasoning-08-2025",
                "messages": [{"role": "user", "content": "Return exact choice."}],
                "max_tokens": 128,
            },
            "final-key",
            arm_id=spec.arm_id,
            phase="final",
        )
    finally:
        provider._spec_by_arm.pop(spec.arm_id, None)
        await provider.aclose()

    assert [request["thinking"] for request in requests] == [
        {"type": "disabled"},
        {"type": "disabled"},
    ]


@pytest.mark.asyncio
async def test_cohere_plus_selection_protocol_preserves_three_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flavourbench.service_cohere.get_settings", _settings)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cohere-selection-legacy-1",
                "finish_reason": "COMPLETE",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"selection":"A,C,F"}'}],
                },
                "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
            },
        )

    spec = GenerationSpec(
        **{
            **_spec().__dict__,
            "final_response_mode": "plain_text",
            "evidence_protocol": "selection_text_v1",
            "backend_contract_json": {
                **_spec().backend_contract_json,
                "portable_phase_reasoning": "thinking_enabled_512_for_schema",
            },
        }
    )
    provider = CohereDirectProvider()
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://api.cohere.test/",
        transport=httpx.MockTransport(handler),
    )
    provider._spec_by_arm[spec.arm_id] = spec
    try:
        value = await provider._post(
            {
                "model": "command-a-plus-05-2026",
                "messages": [{"role": "user", "content": "Select three labels."}],
                "max_tokens": 1024,
            },
            "selection-legacy-key",
            arm_id=spec.arm_id,
            phase="final",
        )
    finally:
        provider._spec_by_arm.pop(spec.arm_id, None)
        await provider.aclose()

    assert requests[0]["response_format"]["type"] == "json_object"
    selection_schema = requests[0]["response_format"]["schema"]["properties"]["selection"]
    assert len(selection_schema["enum"]) == 56
    assert "pattern" not in selection_schema
    assert requests[0]["thinking"] == {"type": "enabled", "token_budget": 512}
    assert value["choices"][0]["message"]["content"] == "FINAL_SELECTION: A,C,F"
