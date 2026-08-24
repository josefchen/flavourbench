from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import flavourbench.service_bedrock as service_bedrock
from flavourbench.bedrock_auth import BedrockLaneSettings
from flavourbench.bedrock_manifest import (
    BedrockEndpointContract,
    BedrockManifestError,
    BedrockPriceContract,
    sanitized_arn,
)
from flavourbench.budget_policy import provider_account_scope_sha256
from flavourbench.provider import (
    GenerationSpec,
    ProviderAttemptEvent,
    ProviderError,
    ToolTrace,
    UncertainDeliveryError,
)
from flavourbench.service_bedrock import BedrockServiceProvider


def _lane_settings() -> BedrockLaneSettings:
    return BedrockLaneSettings.from_environ(
        {
            "FLAVOURBENCH_BEDROCK_ENABLED": "true",
            "FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED": "true",
            "FLAVOURBENCH_BEDROCK_HARD_CAP_USD": "5000",
            "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD": "5",
            "FLAVOURBENCH_BEDROCK_STAGE": "season",
            "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_EVIDENCE_SHA256": "9" * 64,
            "FLAVOURBENCH_BEDROCK_ACCOUNT_SCOPE_SHA256": (
                provider_account_scope_sha256("bedrock")
            ),
            "FLAVOURBENCH_BEDROCK_PROFILE_SCOPE": "global",
            "AWS_REGION": "eu-west-1",
        }
    )


def _contract() -> BedrockEndpointContract:
    return BedrockEndpointContract(
        canonical_model_id="anthropic/claude-test",
        bedrock_target_id="global.anthropic.claude-test-v1:0",
        bedrock_target_arn=sanitized_arn(
            "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/"
            "global.anthropic.claude-test-v1:0"
        ),
        endpoint_kind="inference_profile",
        expected_foundation_model_ids=("anthropic.claude-test-v1:0",),
        destination_model_arns=(
            sanitized_arn(
                "arn:aws:bedrock:eu-west-1::foundation-model/"
                "anthropic.claude-test-v1:0"
            ),
        ),
        region="eu-west-1",
        profile_scope="global",
        supports_converse=True,
        supports_tool_use=True,
        supports_structured_output=True,
        capability_evidence_uri="artifact://bedrock-contract/test",
        capability_evidence_sha256="a" * 64,
        price=BedrockPriceContract(
            input_per_million_usd="2",
            output_per_million_usd="10",
            cache_read_per_million_usd=None,
            cache_write_per_million_usd=None,
            source_uri="https://aws.amazon.com/bedrock/pricing/",
            observed_at="2026-07-15T00:00:00Z",
        ),
        openrouter_fallback_model_id="anthropic/claude-test",
        season_eligible=True,
    )


def test_provisioned_throughput_is_not_officially_eligible_without_commitment_ledger() -> None:
    contract = _contract()
    with pytest.raises(BedrockManifestError, match="hourly commitment accounting"):
        replace(
            contract,
            bedrock_target_id="provisioned-model/test-throughput",
            bedrock_target_arn=sanitized_arn(
                "arn:aws:bedrock:eu-west-1:123456789012:provisioned-model/test-throughput"
            ),
            endpoint_kind="provisioned_throughput",
            profile_scope="in_region",
        )


def _spec(*, condition: str = "epicure_off", arm_id: str = "arm-bedrock") -> GenerationSpec:
    contract = _contract()
    return GenerationSpec(
        arm_id=arm_id,
        battle_id="battle-bedrock",
        prompt="Design a practical tomato dish.",
        category="cookability",
        model_id=contract.canonical_model_id,
        model_name="Claude Test",
        provider_slug="amazon-bedrock",
        condition=condition,
        idempotency_key=f"flavourbench:{arm_id}",
        execution_backend="bedrock",
        backend_contract_json=contract.payload(),
        decoding_parameters={
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.95,
        },
        expected_actual_model_id=contract.canonical_model_id,
        expected_actual_provider_slug="amazon-bedrock",
        endpoint_contract_sha256="b" * 64,
        protocol_bundle_sha256="c" * 64,
        expected_epicure_release_id="epicure-test-v1",
        expected_epicure_bundle_sha256="d" * 64,
        expected_epicure_application_sha256="e" * 64,
        expected_epicure_tool_schema_sha256="f" * 64,
        provider_budget_cap_micros=5_000_000_000,
        provider_account_budget_cap_micros=5_000_000_000,
        provider_account_scope_sha256=provider_account_scope_sha256("bedrock"),
        provider_authorization_envelope_sha256="7" * 64,
        provider_account_authorization_envelope_sha256="6" * 64,
        provider_credential_binding_sha256="5" * 64,
        provider_credential_scope_sha256=provider_account_scope_sha256("bedrock"),
        contract_smoke_registry_sha256="9" * 64,
    )


def _response(
    *,
    request_id: str,
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict[str, Any]:
    return {
        "ResponseMetadata": {
            "RequestId": request_id,
            "HTTPStatusCode": 200,
        },
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "metrics": {"latencyMs": 23},
    }


def _final_answer() -> dict[str, Any]:
    return {
        "answer_markdown": "Roast the tomatoes, then finish with acid and herbs.",
        "ingredient_mentions": ["tomato"],
        "constraints_addressed": ["practical cookability"],
        "uncertainties": ["Seasoning depends on tomato acidity."],
    }


class _FakeRuntime:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class _FakeControl:
    def __init__(self, *, target_arn: str | None = None) -> None:
        self.target_arn = target_arn or (
            "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/"
            "global.anthropic.claude-test-v1:0"
        )

    def get_inference_profile(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "inferenceProfileIdentifier": "global.anthropic.claude-test-v1:0"
        }
        return {
            "ResponseMetadata": {"RequestId": "control-plane-request"},
            "inferenceProfileArn": self.target_arn,
            "models": [
                {
                    "modelArn": (
                        "arn:aws:bedrock:eu-west-1::foundation-model/"
                        "anthropic.claude-test-v1:0"
                    )
                }
            ],
            "status": "ACTIVE",
        }


def _provider(*, runtime: Any, **kwargs: Any) -> BedrockServiceProvider:
    return BedrockServiceProvider(
        runtime=runtime,
        control=_FakeControl(),
        lane_settings=_lane_settings(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_bedrock_service_generates_and_journals_hashed_identity() -> None:
    raw_request_id = "aws-request-id-that-must-not-be-retained"
    runtime = _FakeRuntime(
        [
            _response(
                request_id=raw_request_id,
                content=[{"text": json.dumps(_final_answer())}],
            )
        ]
    )
    events: list[ProviderAttemptEvent] = []
    provider = _provider(
        attempt_sink=events.append,
        runtime=runtime,
    )

    result = await provider.generate(_spec())

    request_hash = hashlib.sha256(raw_request_id.encode()).hexdigest()
    assert result.answer_markdown == _final_answer()["answer_markdown"]
    assert result.actual_model_id == "anthropic/claude-test"
    assert result.provider_slug == "amazon-bedrock"
    assert result.generation_id == f"bedrock:{request_hash}"
    assert result.generation_ids == [f"bedrock:{request_hash}"]
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.cost_micros == 700
    assert result.cost_reconciled is True
    assert len(result.generation_metadata) == len(result.generation_ids) == 1
    assert result.generation_metadata[0]["generation_id"] == result.generation_id
    assert result.generation_metadata[0]["model"] == result.actual_model_id
    assert result.generation_metadata[0]["provider"] == result.provider_slug
    assert [event.event_type for event in events] == [
        "bedrock_control_plane_attested",
        "request_started",
        "response_received",
        "accounting_reconciled",
    ]
    serialized_events = json.dumps([event.__dict__ for event in events], sort_keys=True)
    assert raw_request_id not in serialized_events
    assert request_hash in serialized_events
    assert runtime.requests[0]["modelId"] == "global.anthropic.claude-test-v1:0"
    assert "outputConfig" in runtime.requests[0]


@pytest.mark.asyncio
async def test_bedrock_service_reconciles_accepted_invalid_response() -> None:
    runtime = _FakeRuntime(
        [
            _response(
                request_id="accepted-but-invalid",
                content=[{"text": "not-json"}],
                input_tokens=11,
                output_tokens=7,
            )
        ]
    )
    provider = _provider(
        runtime=runtime,
    )
    spec = _spec(arm_id="arm-invalid")

    with pytest.raises(ProviderError) as caught:
        await provider.generate(spec)
    failure = await provider.reconcile_failure(spec, caught.value)

    assert failure is not None
    assert failure.provider_slug == "amazon-bedrock"
    assert failure.prompt_tokens == 11
    assert failure.completion_tokens == 7
    assert failure.cost_micros == 92
    assert failure.cost_reconciled is True
    assert len(failure.generation_ids) == 1


@pytest.mark.asyncio
async def test_bedrock_service_retains_reservation_when_usage_is_malformed() -> None:
    response = _response(
        request_id="accepted-without-usable-accounting",
        content=[{"text": json.dumps(_final_answer())}],
    )
    response.pop("usage")
    provider = _provider(
        runtime=_FakeRuntime([response]),
    )
    spec = _spec(arm_id="arm-missing-usage")

    with pytest.raises(ProviderError) as caught:
        await provider.generate(spec)
    failure = await provider.reconcile_failure(spec, caught.value)

    assert failure is not None
    assert failure.generation_ids
    assert failure.cost_micros == 0
    assert failure.cost_reconciled is False
    assert failure.generation_metadata[0]["accounting_basis"] == (
        "accepted_response_usage_unparsed"
    )


@pytest.mark.asyncio
async def test_bedrock_service_never_fabricates_a_missing_aws_request_id() -> None:
    runtime = _FakeRuntime(
        [
            _response(
                request_id="",
                content=[{"text": json.dumps(_final_answer())}],
            )
        ]
    )
    events: list[ProviderAttemptEvent] = []
    provider = _provider(
        attempt_sink=events.append,
        runtime=runtime,
    )
    spec = _spec(arm_id="arm-missing-request-id")

    with pytest.raises(ProviderError):
        await provider.generate(spec)
    failure = await provider.reconcile_failure(spec, ProviderError("identity missing"))

    assert failure is not None
    assert failure.generation_id.startswith("bedrock-unverifiable:")
    assert failure.generation_metadata[0]["provider_request_id_present"] is False
    assert "aws_request_id_sha256" not in failure.generation_metadata[0]
    assert any(
        event.metadata.get("identity_evidence")
        == "accepted_response_without_provider_request_id"
        for event in events
    )


@pytest.mark.asyncio
async def test_bedrock_service_rejects_contradictory_returned_model_identity() -> None:
    response = _response(
        request_id="returned-model-mismatch",
        content=[{"text": json.dumps(_final_answer())}],
    )
    response["modelId"] = "global.anthropic.some-other-model-v1:0"
    provider = _provider(
        runtime=_FakeRuntime([response]),
    )
    spec = _spec(arm_id="arm-model-mismatch")

    with pytest.raises(ProviderError, match="Bedrock generation failed") as caught:
        await provider.generate(spec)
    failure = await provider.reconcile_failure(spec, caught.value)

    assert failure is not None
    assert failure.cost_reconciled is True
    assert failure.generation_ids


@pytest.mark.asyncio
async def test_bedrock_service_redacts_returned_account_arn() -> None:
    raw_arn = (
        "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/"
        "global.anthropic.claude-test-v1:0"
    )
    response = _response(
        request_id="returned-account-arn",
        content=[{"text": json.dumps(_final_answer())}],
    )
    response["modelId"] = raw_arn
    provider = _provider(
        runtime=_FakeRuntime([response]),
    )

    result = await provider.generate(_spec(arm_id="arm-account-arn"))

    serialized = json.dumps(result.generation_metadata, sort_keys=True)
    assert raw_arn not in serialized
    assert "<account-redacted>" in serialized
    assert hashlib.sha256(raw_arn.encode()).hexdigest() in serialized


@pytest.mark.asyncio
async def test_bedrock_service_never_falls_back_after_uncertain_delivery() -> None:
    events: list[ProviderAttemptEvent] = []

    class BrokenRuntime:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            raise RuntimeError("connection dropped after dispatch")

    provider = _provider(
        attempt_sink=events.append,
        runtime=BrokenRuntime(),
    )

    with pytest.raises(UncertainDeliveryError, match="automatic fallback is forbidden"):
        await provider.generate(_spec(arm_id="arm-uncertain"))
    assert [event.event_type for event in events] == [
        "bedrock_control_plane_attested",
        "request_started",
        "uncertain_delivery",
    ]


@pytest.mark.parametrize(
    "code",
    [
        "ServiceUnavailableException",
        "ThrottlingException",
        "ModelNotReadyException",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_ambiguous_service_errors_hold_the_reservation(
    code: str,
) -> None:
    events: list[ProviderAttemptEvent] = []

    class AwsServiceError(RuntimeError):
        def __init__(self) -> None:
            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    class BrokenRuntime:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            raise AwsServiceError()

    provider = _provider(
        attempt_sink=events.append,
        runtime=BrokenRuntime(),
    )

    with pytest.raises(UncertainDeliveryError, match="automatic fallback is forbidden"):
        await provider.generate(_spec(arm_id=f"arm-ambiguous-{code}"))
    assert [event.event_type for event in events] == [
        "bedrock_control_plane_attested",
        "request_started",
        "uncertain_delivery",
    ]


@pytest.mark.asyncio
async def test_bedrock_service_rejects_a_seed_it_cannot_send() -> None:
    provider = _provider(
        runtime=_FakeRuntime([]),
    )
    spec = _spec(arm_id="arm-seed")
    spec = replace(
        spec,
        decoding_parameters={**(spec.decoding_parameters or {}), "seed": 20260715},
    )

    with pytest.raises(ProviderError, match="does not accept the frozen seed"):
        await provider.generate(spec)


@pytest.mark.asyncio
async def test_bedrock_service_executes_epicure_tool_with_complete_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(
        [
            _response(
                request_id="tool-round",
                content=[
                    {
                        "toolUse": {
                            "toolUseId": "tool-use-1",
                            "name": "find_pairings",
                            "input": {"ingredients": ["tomato"]},
                        }
                    }
                ],
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=4,
            ),
            _response(
                request_id="final-round",
                content=[{"text": json.dumps(_final_answer())}],
                input_tokens=20,
                output_tokens=10,
            ),
        ]
    )
    mcp_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeMcpSession:
        async def __aenter__(self) -> FakeMcpSession:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        async def list_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "find_pairings",
                    "description": "Return pairing evidence.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ingredients": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["ingredients"],
                        "additionalProperties": False,
                    },
                }
            ]

        async def attest_runtime(
            self,
            *,
            expected: dict[str, str],
            tools: list[dict[str, Any]],
        ) -> dict[str, Any]:
            assert tools[0]["name"] == "find_pairings"
            return {**expected, "attested": True}

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
        ) -> SimpleNamespace:
            mcp_calls.append((name, arguments))
            return SimpleNamespace(
                text='{"pairings":["basil"]}',
                structured={"pairings": ["basil"]},
                is_error=False,
                latency_ms=5,
            )

    monkeypatch.setattr(service_bedrock, "McpSession", FakeMcpSession)
    events: list[ProviderAttemptEvent] = []
    traces: list[ToolTrace] = []
    provider = _provider(
        attempt_sink=events.append,
        tool_sink=lambda _arm_id, trace: traces.append(trace),
        runtime=runtime,
    )

    result = await provider.generate(_spec(condition="epicure_on", arm_id="arm-tool"))

    assert mcp_calls == [("find_pairings", {"ingredients": ["tomato"]})]
    assert len(result.tool_traces) == 1
    assert result.tool_traces == traces
    assert result.tool_traces[0].tool_call_id == "tool-use-1"
    assert result.cost_micros == 200
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 14
    event_types = [event.event_type for event in events]
    assert event_types.count("request_started") == 2
    assert "mcp_session_attested" in event_types
    assert "mcp_call_started" in event_types
    assert "mcp_call_completed" in event_types
    second_messages = runtime.requests[1]["messages"]
    tool_results = [
        block["toolResult"]
        for message in second_messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert tool_results[0]["toolUseId"] == "tool-use-1"
