from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import pytest

from flavourbench.bedrock_auth import (
    BedrockConfigurationError,
    BedrockLaneSettings,
    create_boto3_clients,
)
from flavourbench.bedrock_budget import BedrockBudgetSnapshot, BedrockCostGovernor
from flavourbench.bedrock_manifest import (
    BedrockCatalogDiscoverer,
    BedrockEndpointContract,
    BedrockManifestError,
    BedrockPriceContract,
    assert_public_catalog_safe,
    endpoint_manifest_payload,
    sanitized_arn,
    validate_contract_against_catalog,
)
from flavourbench.bedrock_provider import (
    BEDROCK_FINAL_SCHEMA,
    BedrockConverseProvider,
    BedrockGenerationSpec,
    BedrockInferenceConfig,
    BedrockProviderError,
    BedrockRouteUnavailable,
    BedrockToolDefinition,
    BedrockToolExecution,
    _usage,
    structured_output_config,
)
from flavourbench.bedrock_routing import (
    BedrockPrimaryRouter,
    OpenRouterFallbackResult,
)

FAKE_ACCOUNT_ID = "123456" + "789012"


def bedrock_settings(**overrides: str | None) -> BedrockLaneSettings:
    values: dict[str, str] = {
        "FLAVOURBENCH_BEDROCK_ENABLED": "true",
        "FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED": "true",
        "FLAVOURBENCH_BEDROCK_HARD_CAP_USD": "5000",
        "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_CAP_USD": "5",
        "FLAVOURBENCH_BEDROCK_STAGE": "contract_smoke",
        "FLAVOURBENCH_BEDROCK_PROFILE_SCOPE": "global",
        "AWS_REGION": "eu-west-1",
    }
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value
    return BedrockLaneSettings.from_environ(values)


def price_contract() -> BedrockPriceContract:
    return BedrockPriceContract(
        input_per_million_usd="2",
        output_per_million_usd="10",
        cache_read_per_million_usd="0.2",
        cache_write_per_million_usd="2.5",
        source_uri="https://aws.amazon.com/bedrock/pricing/",
        observed_at="2026-07-15T00:00:00Z",
    )


def endpoint_contract(
    *,
    season_eligible: bool = True,
    temperature_top_p_mutually_exclusive: bool = False,
) -> BedrockEndpointContract:
    return BedrockEndpointContract(
        canonical_model_id="anthropic/claude-test",
        bedrock_target_id="global.anthropic.claude-test-v1:0",
        bedrock_target_arn=sanitized_arn(
            f"arn:aws:bedrock:eu-west-1:{FAKE_ACCOUNT_ID}:inference-profile/"
            "global.anthropic.claude-test-v1:0"
        ),
        endpoint_kind="inference_profile",
        expected_foundation_model_ids=("anthropic.claude-test-v1:0",),
        destination_model_arns=(
            sanitized_arn(
                "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-test-v1:0"
            ),
            sanitized_arn(
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-test-v1:0"
            ),
        ),
        region="eu-west-1",
        profile_scope="global",
        supports_converse=True,
        supports_tool_use=True,
        supports_structured_output=True,
        capability_evidence_uri="artifact://bedrock-contract/test",
        capability_evidence_sha256="a" * 64,
        price=price_contract(),
        openrouter_fallback_model_id="anthropic/claude-test",
        season_eligible=season_eligible,
        temperature_top_p_mutually_exclusive=(
            temperature_top_p_mutually_exclusive
        ),
    )


def test_settings_require_explicit_region_and_never_retain_bearer_value() -> None:
    secret = "bedrock-secret-that-must-not-appear"
    settings = bedrock_settings(AWS_BEARER_TOKEN_BEDROCK=secret)

    assert settings.region == "eu-west-1"
    assert settings.profile_scope == "global"
    assert settings.auth_mode_hint == "bedrock_bearer_token_env"
    assert secret not in repr(settings)
    assert not hasattr(settings, "bearer_token")

    with pytest.raises(BedrockConfigurationError, match="explicit AWS_REGION"):
        bedrock_settings(AWS_REGION="")
    with pytest.raises(BedrockConfigurationError, match="contract-smoke evidence"):
        bedrock_settings(FLAVOURBENCH_BEDROCK_STAGE="exploratory")


@pytest.mark.parametrize("bearer", [None, "", "   "])
def test_settings_record_default_chain_when_bearer_is_absent_or_blank(
    bearer: str | None,
) -> None:
    settings = bedrock_settings(AWS_BEARER_TOKEN_BEDROCK=bearer)
    assert settings.auth_mode_hint == "boto3_default_chain"


@pytest.mark.parametrize("value", ["global", "in_region", "us", "eu", "apac"])
def test_settings_reject_profile_scope_as_sdk_region(value: str) -> None:
    with pytest.raises(BedrockConfigurationError, match="physical AWS region"):
        bedrock_settings(AWS_REGION=value)


def test_cap_alias_is_supported_and_conflicts_fail_closed() -> None:
    alias_only = bedrock_settings(
        FLAVOURBENCH_BEDROCK_HARD_CAP_USD=None,
        FLAVOURBENCH_BEDROCK_CAP_USD="5000",
    )
    assert alias_only.hard_cap_usd == Decimal("5000")

    with pytest.raises(BedrockConfigurationError, match="conflict"):
        bedrock_settings(FLAVOURBENCH_BEDROCK_CAP_USD="4999")


def test_boto3_client_factory_passes_only_region_and_optional_endpoints() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("session", kwargs))

        def client(self, service_name: str, **kwargs: Any) -> object:
            calls.append((service_name, kwargs))
            return {"service": service_name}

    settings = bedrock_settings(AWS_BEARER_TOKEN_BEDROCK="never-forward-this")
    clients = create_boto3_clients(settings, session_factory=FakeSession)

    assert clients.region == "eu-west-1"
    assert calls == [
        ("session", {"region_name": "eu-west-1"}),
        ("bedrock", {}),
        ("bedrock-runtime", {}),
    ]
    assert "never-forward-this" not in repr(calls)


def test_boto3_client_factory_forwards_non_secret_transport_config() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    transport_config = object()

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("session", kwargs))

        def client(self, service_name: str, **kwargs: Any) -> object:
            calls.append((service_name, kwargs))
            return {"service": service_name}

    create_boto3_clients(
        bedrock_settings(),
        session_factory=FakeSession,
        client_config=transport_config,
    )
    assert calls == [
        ("session", {"region_name": "eu-west-1"}),
        ("bedrock", {"config": transport_config}),
        ("bedrock-runtime", {"config": transport_config}),
    ]


def test_contract_smoke_stage_is_bounded_inside_the_5000_usd_hard_cap() -> None:
    governor = BedrockCostGovernor(bedrock_settings())

    admitted = governor.decide(
        BedrockBudgetSnapshot(Decimal("4.20"), Decimal("0")),
        worst_case_reservation_usd=Decimal("0.04"),
    )
    stopped = governor.decide(
        BedrockBudgetSnapshot(Decimal("4.20"), Decimal("0")),
        worst_case_reservation_usd=Decimal("0.10"),
    )
    hard_stopped = governor.decide(
        BedrockBudgetSnapshot(Decimal("4.99"), Decimal("0")),
        worst_case_reservation_usd=Decimal("0.02"),
    )

    assert admitted.status == "admit"
    assert admitted.effective_stage_cap_usd == Decimal("5")
    assert admitted.hard_cap_usd == Decimal("5000")
    assert stopped.status == "stop_admission"
    assert hard_stopped.status == "hard_stop"


class FakeControlClient:
    def list_foundation_models(self, **kwargs: Any) -> dict[str, Any]:
        assert not kwargs
        return {
            "ResponseMetadata": {
                "RequestId": "discovery-request-id",
                "HTTPHeaders": {
                    "x-test-account": FAKE_ACCOUNT_ID,
                    "authorization": "Bearer fake-test-only-credential",
                },
            },
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-test-v1:0",
                    "modelArn": (
                        "arn:aws:bedrock:eu-west-1::foundation-model/"
                        "anthropic.claude-test-v1:0"
                    ),
                    "modelName": "Claude Test",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                }
            ]
        }

    def list_inference_profiles(self, **kwargs: Any) -> dict[str, Any]:
        if not kwargs:
            return {
                "nextToken": "page-2",
                "inferenceProfileSummaries": [],
            }
        assert kwargs == {"nextToken": "page-2"}
        return {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "global.anthropic.claude-test-v1:0",
                    "inferenceProfileArn": (
                        f"arn:aws:bedrock:eu-west-1:{FAKE_ACCOUNT_ID}:inference-profile/"
                        "global.anthropic.claude-test-v1:0"
                    ),
                    "inferenceProfileName": "Global Claude Test",
                    "status": "ACTIVE",
                    "type": "SYSTEM_DEFINED",
                    "models": [
                        {
                            "modelArn": (
                                "arn:aws:bedrock:eu-west-1::foundation-model/"
                                "anthropic.claude-test-v1:0"
                            )
                        },
                        {
                            "modelArn": (
                                "arn:aws:bedrock:us-east-1::foundation-model/"
                                "anthropic.claude-test-v1:0"
                            )
                        },
                    ],
                }
            ]
        }

    def list_provisioned_model_throughputs(self, **kwargs: Any) -> dict[str, Any]:
        assert not kwargs
        return {"provisionedModelSummaries": []}


def test_catalog_discovery_and_manifest_freeze_are_deterministic_and_no_inference() -> None:
    snapshot = BedrockCatalogDiscoverer(
        FakeControlClient(), region="eu-west-1", profile_scope="global"
    ).discover(discovered_at="2026-07-15T00:00:00Z")
    contract = endpoint_contract(season_eligible=False)

    validate_contract_against_catalog(contract, snapshot)
    manifest = endpoint_manifest_payload(snapshot, [contract])

    assert len(snapshot.targets) == 2
    assert snapshot.catalog_sha256 == BedrockCatalogDiscoverer(
        FakeControlClient(), region="eu-west-1", profile_scope="global"
    ).discover(discovered_at="2026-07-15T00:00:00Z").catalog_sha256
    assert manifest["rank_eligible"] is False
    frozen = manifest["contracts"][0]
    assert frozen["profile_scope"] == "global"
    assert [item["redacted"] for item in frozen["destination_model_arns"]] == sorted(
        item.redacted for item in contract.destination_model_arns
    )
    assert frozen["bedrock_target_arn"]["redacted"].endswith(
        "<account-redacted>:inference-profile/global.anthropic.claude-test-v1:0"
    )
    assert len(frozen["bedrock_target_arn"]["original_sha256"]) == 64
    assert FAKE_ACCOUNT_ID not in str(manifest)
    assert "fake-test-only-credential" not in str(manifest)
    assert "authorization" not in str(manifest).lower()
    assert len(frozen["profile_scope_sha256"]) == 64


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: str, arguments: dict[str, Any]) -> BedrockToolExecution:
        self.calls.append((name, dict(arguments)))
        return BedrockToolExecution({"score": 0.83, "source": "epicure-test"})


class FakeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = [
            {
                "ResponseMetadata": {"RequestId": "request-tool", "HTTPStatusCode": 200},
                "modelId": "global.anthropic.claude-test-v1:0",
                "stopReason": "tool_use",
                "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                "metrics": {"latencyMs": 13},
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool-1",
                                    "name": "pairing_score",
                                    "input": {"ingredient_a": "tomato", "ingredient_b": "miso"},
                                }
                            }
                        ],
                    }
                },
            },
            {
                "ResponseMetadata": {"RequestId": "request-final", "HTTPStatusCode": 200},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 300, "outputTokens": 50, "totalTokens": 350},
                "metrics": {"latencyMs": 21},
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "text": (
                                    '{"answer_markdown":"Use miso sparingly.",'
                                    '"ingredient_mentions":["tomato","miso"],'
                                    '"constraints_addressed":[],"uncertainties":[]}'
                                )
                            }
                        ],
                    }
                },
            },
        ]

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_converse_tool_loop_structured_output_and_provenance() -> None:
    runtime = FakeRuntimeClient()
    executor = FakeToolExecutor()
    provider = BedrockConverseProvider(
        runtime,
        endpoint_contract(),
        tool_executor=executor,
    )
    tool = BedrockToolDefinition(
        name="pairing_score",
        description="Score a culinary pairing",
        input_schema={
            "type": "object",
            "properties": {
                "ingredient_a": {"type": "string"},
                "ingredient_b": {"type": "string"},
            },
            "required": ["ingredient_a", "ingredient_b"],
            "additionalProperties": False,
        },
    )
    result = await provider.generate(
        BedrockGenerationSpec(
            arm_id="arm-test",
            canonical_model_id="anthropic/claude-test",
            prompt="How should I combine tomato and miso?",
            system_prompt="Return only the frozen structured answer.",
            inference=BedrockInferenceConfig(max_tokens=3000, temperature=0.2, top_p=0.95),
            tools=(tool,),
            request_metadata={"benchmark": "flavourbench"},
        )
    )

    assert len(runtime.calls) == 2
    first = runtime.calls[0]
    assert first["modelId"] == "global.anthropic.claude-test-v1:0"
    assert first["outputConfig"]["textFormat"]["type"] == "json_schema"
    assert first["toolConfig"]["tools"][0]["toolSpec"]["strict"] is True
    assert "minLength" not in first["outputConfig"]["textFormat"]["structure"][
        "jsonSchema"
    ]["schema"]
    assert runtime.calls[1]["messages"][-1]["content"][0]["toolResult"][
        "toolUseId"
    ] == "tool-1"
    assert executor.calls == [
        ("pairing_score", {"ingredient_a": "tomato", "ingredient_b": "miso"})
    ]
    assert result.request_ids == ("request-tool", "request-final")
    assert result.identity.ingress_region == "eu-west-1"
    assert result.identity.profile_scope == "global"
    assert FAKE_ACCOUNT_ID not in result.identity.requested_model_or_profile_arn_redacted
    assert len(result.identity.requested_model_or_profile_arn_sha256) == 64
    assert result.identity.returned_model_ids == (
        "global.anthropic.claude-test-v1:0",
    )
    assert result.usage.input_tokens == 400
    assert result.usage.output_tokens == 70
    assert result.service_latency_ms == 34
    assert result.response_latencies_ms == (13, 21)
    assert result.retries == 0
    assert len(result.response_schema_sha256) == 64
    assert len(result.tool_schema_sha256) == 64
    assert result.cost.estimated_cost_micros == 1500
    assert result.cost.cost_source == "frozen_rate_card_estimate"
    assert result.cost.independent_billing_reconciliation_status == "not_reconciled"
    assert result.provider_substitution is False


@pytest.mark.asyncio
async def test_model_specific_sampling_conflict_fails_before_runtime() -> None:
    runtime = FakeRuntimeClient()
    provider = BedrockConverseProvider(
        runtime,
        endpoint_contract(temperature_top_p_mutually_exclusive=True),
    )

    with pytest.raises(BedrockProviderError, match="temperature or top_p"):
        await provider.generate(
            BedrockGenerationSpec(
                arm_id="sampling-conflict",
                canonical_model_id="anthropic/claude-test",
                prompt="test",
                system_prompt="test",
                inference=BedrockInferenceConfig(
                    max_tokens=64,
                    temperature=0.2,
                    top_p=0.95,
                ),
            )
        )

    assert runtime.calls == []


def test_bedrock_schema_rejects_unsupported_string_constraints() -> None:
    schema = {**BEDROCK_FINAL_SCHEMA, "properties": {"value": {"type": "string", "minLength": 1}}}
    with pytest.raises(BedrockProviderError, match="minLength"):
        structured_output_config(schema)


@pytest.mark.parametrize("missing", ["usage", "inputTokens", "outputTokens", "totalTokens"])
def test_bedrock_usage_requires_every_billable_response_field(missing: str) -> None:
    response: dict[str, Any] = {
        "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}
    }
    if missing == "usage":
        response.pop("usage")
    else:
        response["usage"].pop(missing)

    with pytest.raises(BedrockProviderError, match="usage"):
        _usage(response)


def test_optional_cache_usage_fields_default_to_zero() -> None:
    usage = _usage(
        {"usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}}
    )

    assert usage.cache_read_input_tokens == 0
    assert usage.cache_write_input_tokens == 0


def test_public_catalog_secret_scan_rejects_identity_and_credentials() -> None:
    assert_public_catalog_safe({"artifact_sha256": "1" * 64})
    with pytest.raises(BedrockManifestError, match="AWS account ID"):
        assert_public_catalog_safe({"metadata": {"owner": FAKE_ACCOUNT_ID}})
    with pytest.raises(BedrockManifestError, match="forbidden credential"):
        assert_public_catalog_safe({"authorization": "not-printed"})
    with pytest.raises(BedrockManifestError, match="credential-bearing URI"):
        assert_public_catalog_safe({"source": "https://user:password@example.invalid/catalog"})


class UnavailablePrimary:
    async def generate(self, spec: BedrockGenerationSpec) -> Any:
        raise BedrockRouteUnavailable("Bedrock rejected the route before generation: Throttling")


class FakeFallback:
    canonical_model_id = "anthropic/claude-test"

    async def generate(self, spec: BedrockGenerationSpec) -> OpenRouterFallbackResult:
        return OpenRouterFallbackResult(
            canonical_model_id=spec.canonical_model_id,
            actual_model_id="anthropic/claude-test",
            provider_slug="fixed-openrouter-endpoint",
            payload={"answer": "fallback"},
        )


class InvalidOutputPrimary:
    async def generate(self, spec: BedrockGenerationSpec) -> Any:
        raise BedrockProviderError("Bedrock structured output was not valid JSON")


@pytest.mark.asyncio
async def test_openrouter_fallback_is_same_model_substitution_unranked_and_unpooled() -> None:
    router = BedrockPrimaryRouter(
        UnavailablePrimary(),
        canonical_model_id="anthropic/claude-test",
        fallback=FakeFallback(),
        allow_openrouter_fallback=True,
    )
    result = await router.generate(
        BedrockGenerationSpec(
            arm_id="arm-fallback",
            canonical_model_id="anthropic/claude-test",
            prompt="test",
            system_prompt="test",
            inference=BedrockInferenceConfig(max_tokens=1000),
        )
    )

    assert result.route == "openrouter_fallback"
    assert result.provider_substitution is True
    assert result.rank_eligible is False
    assert result.unpooled is True
    assert result.pooling_group == "unranked_provider_substitution"


def test_router_rejects_a_different_canonical_fallback() -> None:
    fallback = FakeFallback()
    fallback.canonical_model_id = "anthropic/different-model"
    with pytest.raises(ValueError, match="same canonical model"):
        BedrockPrimaryRouter(
            UnavailablePrimary(),
            canonical_model_id="anthropic/claude-test",
            fallback=fallback,
            allow_openrouter_fallback=True,
        )


@pytest.mark.asyncio
async def test_invalid_bedrock_output_never_triggers_fallback() -> None:
    router = BedrockPrimaryRouter(
        InvalidOutputPrimary(),
        canonical_model_id="anthropic/claude-test",
        fallback=FakeFallback(),
        allow_openrouter_fallback=True,
    )
    with pytest.raises(BedrockProviderError, match="not valid JSON"):
        await router.generate(
            BedrockGenerationSpec(
                arm_id="arm-invalid",
                canonical_model_id="anthropic/claude-test",
                prompt="test",
                system_prompt="test",
                inference=BedrockInferenceConfig(max_tokens=1000),
            )
        )
