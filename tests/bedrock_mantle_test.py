from __future__ import annotations

import copy
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from flavourbench.bedrock_budget import BedrockAdmissionDecision
from flavourbench.bedrock_manifest import BedrockPriceContract
from flavourbench.bedrock_mantle import (
    GPT56_TIER_POLICIES,
    MantleAdmissionTicket,
    MantleBudgetFinalization,
    MantleContractError,
    MantleEndpointContract,
    MantleGenerationSpec,
    MantleInferenceConfig,
    MantleProviderError,
    MantleReservationRequest,
    MantleResponsesProvider,
    MantleRouteUnavailable,
    MantleToolDefinition,
    MantleToolExecution,
    MantleTransportError,
    MantleTransportResponse,
    admission_ticket_from_governor_decision,
    frontier_contract_plan,
    run,
    worst_case_cost_micros,
)
from flavourbench.bedrock_mantle_routing import (
    MantleOpenRouterFallbackResult,
    MantlePrimaryRouter,
)


def price_contract() -> BedrockPriceContract:
    return BedrockPriceContract(
        input_per_million_usd="2",
        output_per_million_usd="10",
        cache_read_per_million_usd="0.2",
        cache_write_per_million_usd=None,
        source_uri="https://aws.amazon.com/bedrock/pricing/",
        observed_at="2026-07-15T00:00:00Z",
    )


def endpoint_contract(
    *,
    tier: str = "sol",
    region: str = "us-east-1",
    structured_output_mode: str = "responses_json_schema",
    season_eligible: bool = True,
) -> MantleEndpointContract:
    return MantleEndpointContract(
        tier=tier,  # type: ignore[arg-type]
        canonical_model_id=f"openai/gpt-5.6-{tier}",
        mantle_model_id=f"openai.gpt-5.6-{tier}",
        ingress_region=region,
        model_catalog_uri="artifact://bedrock-mantle/models/test",
        model_catalog_sha256="a" * 64,
        model_entry_sha256="d" * 64,
        model_catalog_observed_at="2026-07-15T00:00:00Z",
        supports_responses=True,
        supports_client_side_tools=True,
        supports_strict_tools=True,
        structured_output_mode=structured_output_mode,  # type: ignore[arg-type]
        capability_evidence_uri="artifact://bedrock-mantle/contract-smoke",
        capability_evidence_sha256="b" * 64,
        price=price_contract(),
        openrouter_fallback_model_id=f"openai/gpt-5.6-{tier}",
        season_eligible=season_eligible,
    )


def generation_spec(*, tools: tuple[MantleToolDefinition, ...] = ()) -> MantleGenerationSpec:
    return MantleGenerationSpec(
        arm_id="arm-mantle-test",
        canonical_model_id="openai/gpt-5.6-sol",
        prompt="How should I combine tomato and miso?",
        system_prompt="Use Epicure evidence and return the frozen JSON object.",
        inference=MantleInferenceConfig(
            max_output_tokens=100,
            max_input_tokens_per_response=1_000,
            reasoning_effort="medium",
        ),
        tools=tools,
        request_metadata={"benchmark": "flavourbench"},
    )


class FakeBudget:
    def __init__(self, *, admitted: bool = True, reserved_delta: int = 0) -> None:
        self.admitted = admitted
        self.reserved_delta = reserved_delta
        self.requests: list[MantleReservationRequest] = []
        self.finalizations: list[MantleBudgetFinalization] = []

    async def reserve(self, request: MantleReservationRequest) -> MantleAdmissionTicket:
        self.requests.append(request)
        return MantleAdmissionTicket(
            reservation_id="reservation-test",
            arm_id=request.arm_id,
            canonical_model_id=request.canonical_model_id,
            contract_sha256=request.contract_sha256,
            reserved_cost_micros=(
                request.worst_case_estimated_cost_micros + self.reserved_delta
            ),
            hard_cap_micros=5_000_000,
            admitted=self.admitted,
            admission_status="admit" if self.admitted else "hard_stop",
            admission_evidence_sha256="c" * 64,
        )

    async def finalize(self, finalization: MantleBudgetFinalization) -> None:
        self.finalizations.append(finalization)


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> MantleToolExecution:
        self.calls.append((name, dict(arguments)))
        return MantleToolExecution(
            {"score": 0.83, "source": "epicure-private-mcp"}
        )


def tool_definition() -> MantleToolDefinition:
    return MantleToolDefinition(
        name="pairing_score",
        description="Query the private Epicure MCP pairing model",
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


def tool_response() -> dict[str, Any]:
    return {
        "id": "resp-tool",
        "model": "openai.gpt-5.6-sol",
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
        "output": [
            {
                "type": "function_call",
                "id": "item-tool",
                "call_id": "call-tool",
                "name": "pairing_score",
                "arguments": '{"ingredient_a":"tomato","ingredient_b":"miso"}',
            }
        ],
    }


def final_response() -> dict[str, Any]:
    return {
        "id": "resp-final",
        "model": "openai.gpt-5.6-sol",
        "status": "completed",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            '{"answer_markdown":"Use miso sparingly.",'
                            '"ingredient_mentions":["tomato","miso"],'
                            '"constraints_addressed":[],"uncertainties":[]}'
                        ),
                    }
                ],
            }
        ],
    }


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[dict[str, Any], str]] = []

    async def create_response(
        self, payload: dict[str, Any], *, idempotency_key: str
    ) -> MantleTransportResponse:
        self.calls.append((copy.deepcopy(payload), idempotency_key))
        index = len(self.calls)
        return MantleTransportResponse(
            body=self.responses.pop(0),
            status_code=200,
            aws_request_id=f"aws-request-{index}",
            elapsed_ms=10 + index,
        )


def test_model_specific_regions_and_no_global_profile_are_hard_contracts() -> None:
    assert GPT56_TIER_POLICIES["sol"].documented_ingress_regions == (
        "us-east-1",
        "us-east-2",
    )
    assert GPT56_TIER_POLICIES["terra"].documented_ingress_regions == (
        "us-east-1",
        "us-east-2",
        "us-west-2",
    )
    assert endpoint_contract().profile_scope == "in_region"
    assert endpoint_contract().endpoint_base_url == (
        "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    )
    with pytest.raises(MantleContractError, match="not documented"):
        endpoint_contract(tier="sol", region="us-west-2")
    with pytest.raises(MantleContractError, match="cross-region profile"):
        replace(
            endpoint_contract(),
            mantle_model_id="global.openai.gpt-5.6-sol",
        )


def test_contract_canonical_binding_fallback_and_capability_gates() -> None:
    with pytest.raises(MantleContractError, match="canonical"):
        MantleEndpointContract(
            tier="sol",
            canonical_model_id="openai/gpt-5.6-luna",
            mantle_model_id="openai.gpt-5.6-sol",
            ingress_region="us-east-1",
            model_catalog_uri="artifact://bedrock-mantle/models/test",
            model_catalog_sha256="a" * 64,
            model_entry_sha256="d" * 64,
            model_catalog_observed_at="2026-07-15T00:00:00Z",
            supports_responses=True,
            supports_client_side_tools=True,
            supports_strict_tools=True,
            structured_output_mode="responses_json_schema",
            capability_evidence_uri="artifact://test",
            capability_evidence_sha256="b" * 64,
            price=price_contract(),
            openrouter_fallback_model_id="openai/gpt-5.6-luna",
        )
    with pytest.raises(MantleContractError, match="season eligibility"):
        endpoint_contract(
            structured_output_mode="client_validation_only",
            season_eligible=True,
        )
    with pytest.raises(MantleContractError, match="unsafe public provenance"):
        replace(
            endpoint_contract(),
            model_catalog_uri="https://user:password@example.invalid/models",
        )


@pytest.mark.asyncio
async def test_client_side_epicure_loop_json_schema_provenance_and_estimated_cost() -> None:
    transport = FakeTransport([tool_response(), final_response()])
    budget = FakeBudget()
    executor = FakeToolExecutor()
    provider = MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
        tool_executor=executor,
    )
    result = await provider.generate(generation_spec(tools=(tool_definition(),)))

    assert len(transport.calls) == 2
    first, first_idempotency_key = transport.calls[0]
    assert first["model"] == "openai.gpt-5.6-sol"
    assert first["store"] is False
    assert first["text"]["format"]["type"] == "json_schema"
    assert first["text"]["format"]["strict"] is True
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][0]["strict"] is True
    assert "mcp" not in first
    assert "previous_response_id" not in first
    assert len(first_idempotency_key) == 64
    second = transport.calls[1][0]
    assert any(item.get("type") == "function_call" for item in second["input"])
    tool_output = next(
        item for item in second["input"] if item.get("type") == "function_call_output"
    )
    assert tool_output["call_id"] == "call-tool"
    assert executor.calls == [
        ("pairing_score", {"ingredient_a": "tomato", "ingredient_b": "miso"})
    ]
    assert result.answer_markdown == "Use miso sparingly."
    assert result.request_ids == ("aws-request-1", "aws-request-2")
    assert result.response_ids == ("resp-tool", "resp-final")
    assert result.identity.returned_model_ids == (
        "openai.gpt-5.6-sol",
    )
    assert result.identity.ingress_region == "us-east-1"
    assert result.identity.profile_scope == "in_region"
    assert result.store is False
    assert result.usage.input_tokens == 300
    assert result.usage.cached_input_tokens == 10
    assert result.usage.reasoning_output_tokens == 10
    assert result.cost.estimated_cost_micros == 1_282
    assert result.cost.cost_source == "frozen_rate_card_estimate"
    assert result.cost.charged_cost_status == "not_available_in_responses_api"
    assert result.rounds[0].request_sha256 == (
        __import__("hashlib").sha256(
            result.rounds[0].request_payload_json.encode()
        ).hexdigest()
    )
    assert json.loads(result.tool_traces[0].result_json)["source"] == (
        "epicure-private-mcp"
    )
    assert budget.requests[0].worst_case_estimated_cost_micros == 4_903_200
    assert budget.finalizations == [
        MantleBudgetFinalization(
            reservation_id="reservation-test",
            outcome="success_estimate_only",
            estimated_cost_micros=1_282,
            estimate_complete=True,
            note=(
                "Responses usage priced from frozen rate card; AWS charged cost is "
                "not claimed."
            ),
        )
    ]


@pytest.mark.asyncio
async def test_tool_only_exploratory_mode_uses_local_final_validation() -> None:
    transport = FakeTransport([tool_response(), final_response()])
    budget = FakeBudget()
    executor = FakeToolExecutor()
    contract = endpoint_contract(
        structured_output_mode="client_validation_only",
        season_eligible=False,
    )
    result = await MantleResponsesProvider(
        transport,
        contract,
        budget=budget,
        tool_executor=executor,
    ).generate(generation_spec(tools=(tool_definition(),)))

    first = transport.calls[0][0]
    assert "text" not in first
    assert "Return the final answer only as JSON matching this schema" in first[
        "instructions"
    ]
    assert first["tools"][0]["type"] == "function"
    assert "mcp" not in first
    assert result.structured_output_enforcement == "client_validation_only"
    assert result.rank_eligible is False
    assert result.answer_markdown == "Use miso sparingly."


@pytest.mark.asyncio
async def test_budget_denial_happens_before_any_mantle_call() -> None:
    transport = FakeTransport([final_response()])
    budget = FakeBudget(admitted=False)
    provider = MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
    )
    with pytest.raises(MantleProviderError, match="admission"):
        await provider.generate(generation_spec())
    assert transport.calls == []
    assert budget.finalizations == []


@pytest.mark.asyncio
async def test_non_strict_tools_and_impossible_context_bounds_fail_before_admission() -> None:
    transport = FakeTransport([final_response()])
    budget = FakeBudget()
    provider = MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
        tool_executor=FakeToolExecutor(),
    )
    with pytest.raises(MantleProviderError, match="strict Mantle function tools"):
        await provider.generate(
            generation_spec(tools=(replace(tool_definition(), strict=False),))
        )
    impossible = replace(
        generation_spec(),
        inference=MantleInferenceConfig(
            max_output_tokens=100,
            max_input_tokens_per_response=272_001,
        ),
    )
    with pytest.raises(MantleProviderError, match="input bound exceeds"):
        await provider.generate(impossible)
    assert budget.requests == []
    assert transport.calls == []


def test_worst_case_reservation_never_assumes_cache_discount() -> None:
    inference = MantleInferenceConfig(
        max_output_tokens=100,
        max_input_tokens_per_response=1_000,
    )
    assert worst_case_cost_micros(
        endpoint_contract(), inference, maximum_responses=9
    ) == 4_903_200
    assert Decimal(4_903_200) / Decimal(1_000_000) == Decimal("4.9032")


def test_shared_bedrock_governor_decision_binds_the_mantle_ticket() -> None:
    request = MantleReservationRequest(
        arm_id="arm-governed",
        canonical_model_id="openai/gpt-5.6-sol",
        contract_sha256="e" * 64,
        worst_case_estimated_cost_micros=4_903_200,
    )
    decision = BedrockAdmissionDecision(
        status="admit",
        admitted=True,
        stage="contract_smoke",
        requested_reservation_usd=Decimal("4.9032"),
        exposure_before_usd=Decimal("0"),
        exposure_after_usd=Decimal("4.9032"),
        effective_stage_cap_usd=Decimal("5000"),
        hard_cap_usd=Decimal("5000"),
        reason="within every boundary",
    )
    ticket = admission_ticket_from_governor_decision(
        request,
        decision,
        reservation_id="reservation-governed",
        admission_evidence_sha256="f" * 64,
    )
    assert ticket.admitted is True
    assert ticket.reserved_cost_micros == 4_903_200
    assert ticket.hard_cap_micros == 5_000_000_000
    with pytest.raises(MantleProviderError, match="does not match"):
        admission_ticket_from_governor_decision(
            request,
            replace(
                decision,
                requested_reservation_usd=Decimal("4.9031"),
            ),
            reservation_id="reservation-governed",
            admission_evidence_sha256="f" * 64,
        )


class RetryThenSuccessTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__([final_response()])
        self.attempts = 0

    async def create_response(
        self, payload: dict[str, Any], *, idempotency_key: str
    ) -> MantleTransportResponse:
        self.attempts += 1
        if self.attempts <= 2:
            raise MantleTransportError(
                "socket was never opened",
                dispatch_state="not_sent",
            )
        return await super().create_response(
            payload,
            idempotency_key=idempotency_key,
        )


@pytest.mark.asyncio
async def test_only_proven_not_sent_failures_retry_at_most_twice() -> None:
    transport = RetryThenSuccessTransport()
    budget = FakeBudget()
    result = await MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
    ).generate(generation_spec())
    assert transport.attempts == 3
    assert result.retries == 2


class AlwaysFailTransport:
    def __init__(self, error: MantleTransportError) -> None:
        self.error = error
        self.calls = 0

    async def create_response(
        self, payload: dict[str, Any], *, idempotency_key: str
    ) -> MantleTransportResponse:
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_ambiguous_failure_is_not_retried_and_holds_full_reservation() -> None:
    transport = AlwaysFailTransport(
        MantleTransportError("timeout", dispatch_state="ambiguous")
    )
    budget = FakeBudget()
    provider = MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
    )
    with pytest.raises(MantleProviderError, match="without retry-safe evidence"):
        await provider.generate(generation_spec())
    assert transport.calls == 1
    assert budget.finalizations[0].outcome == "failed_hold_full_reservation"


@pytest.mark.asyncio
async def test_explicit_pre_inference_route_rejection_allows_fallback_and_release() -> None:
    transport = AlwaysFailTransport(
        MantleTransportError(
            "model absent",
            status_code=404,
            error_code="model_not_found",
            dispatch_state="rejected_before_inference",
            route_unavailable=True,
        )
    )
    budget = FakeBudget()
    provider = MantleResponsesProvider(
        transport,
        endpoint_contract(),
        budget=budget,
    )
    with pytest.raises(MantleRouteUnavailable, match="model_not_found"):
        await provider.generate(generation_spec())
    assert transport.calls == 1
    assert budget.finalizations[0].outcome == "not_sent_release_allowed"


class UnavailablePrimary:
    async def generate(self, spec: MantleGenerationSpec) -> Any:
        raise MantleRouteUnavailable("model_not_found")


class ExactFallback:
    canonical_model_id = "openai/gpt-5.6-sol"

    async def generate(self, spec: MantleGenerationSpec) -> MantleOpenRouterFallbackResult:
        return MantleOpenRouterFallbackResult(
            canonical_model_id=spec.canonical_model_id,
            actual_model_id="openai/gpt-5.6-sol",
            provider_slug="fixed-openrouter-endpoint",
            payload={"answer": "fallback"},
        )


@pytest.mark.asyncio
async def test_openrouter_fallback_is_exact_model_unranked_and_unpooled() -> None:
    result = await MantlePrimaryRouter(
        UnavailablePrimary(),
        canonical_model_id="openai/gpt-5.6-sol",
        fallback=ExactFallback(),
        allow_openrouter_fallback=True,
    ).generate(generation_spec())
    assert result.route == "openrouter_fallback"
    assert result.provider_substitution is True
    assert result.rank_eligible is False
    assert result.unpooled is True
    assert result.pooling_group == "unranked_provider_substitution"


def test_dry_run_plan_never_reads_credentials_or_calls_a_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "bedrock-secret-never-copy-or-print"
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", secret)
    plan = frontier_contract_plan()
    assert plan["provider_calls_made"] is False
    assert plan["inference_calls"] == 0
    assert plan["credential_environment_read"] is False
    assert plan["server_side_tools_enabled"] is False
    assert plan["epicure_tool_ownership"] == "flavourbench_client_side_only"
    assert secret not in json.dumps(plan)
    assert run([]) == 0
    assert secret not in capsys.readouterr().out


def test_dry_run_validates_a_canonical_contract_and_forecasts_reservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = endpoint_contract()
    path = tmp_path / "mantle-contract.json"
    path.write_text(json.dumps(contract.payload(), sort_keys=True), encoding="utf-8")
    assert run(["--contract", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["provider_calls_made"] is False
    assert output["contract_sha256"] == contract.sha256
    assert output["worst_case_reservation_micros"] == 5_112_000
    assert output["validated_contract"]["profile_scope"] == "in_region"
