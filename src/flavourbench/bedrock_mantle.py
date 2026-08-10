"""Isolated Amazon Bedrock Mantle Responses lane for OpenAI GPT-5.6.

The Mantle endpoint is not a ``bedrock-runtime`` Converse endpoint.  It uses
an OpenAI-compatible Responses surface and a model catalog that is independent
from ``ListFoundationModels``.  This module therefore keeps Mantle contracts,
transport, provenance, and client-side Epicure orchestration separate from the
existing Converse adapter.

Nothing in this module reads ``AWS_BEARER_TOKEN_BEDROCK``.  A caller must
inject an already authenticated transport.  The provided HTTP transport also
receives an already configured ``httpx.AsyncClient`` and never inspects or
persists its authentication headers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from .bedrock_budget import BedrockAdmissionDecision
from .bedrock_manifest import (
    BedrockManifestError,
    BedrockPriceContract,
    assert_public_catalog_safe,
)
from .bedrock_provider import BEDROCK_FINAL_SCHEMA, validate_bedrock_json_schema

MANTLE_CONTRACT_SCHEMA_VERSION = "flavourbench-bedrock-mantle-contract-v1"
MANTLE_PLAN_SCHEMA_VERSION = "flavourbench-bedrock-mantle-plan-v1"
MANTLE_API_REFERENCE = "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html"
GPT56_LAUNCH_REFERENCE = "https://aws.amazon.com/about-aws/whats-new/2026/07/openai-gpt-sol-terra/"

MantleTier = Literal["sol", "terra", "luna"]
StructuredOutputMode = Literal["responses_json_schema", "client_validation_only", "unverified"]
DispatchState = Literal["not_sent", "rejected_before_inference", "ambiguous", "accepted"]
BudgetOutcome = Literal[
    "success_estimate_only",
    "not_sent_release_allowed",
    "failed_hold_full_reservation",
]


@dataclass(frozen=True)
class GPT56TierPolicy:
    tier: MantleTier
    canonical_model_id: str
    mantle_model_id: str
    documented_ingress_regions: tuple[str, ...]
    model_card_uri: str
    context_window_tokens: int = 272_000


GPT56_TIER_POLICIES: Mapping[MantleTier, GPT56TierPolicy] = {
    "sol": GPT56TierPolicy(
        tier="sol",
        canonical_model_id="openai/gpt-5.6-sol",
        mantle_model_id="openai.gpt-5.6-sol",
        documented_ingress_regions=("us-east-1", "us-east-2"),
        model_card_uri=(
            "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html"
        ),
    ),
    "terra": GPT56TierPolicy(
        tier="terra",
        canonical_model_id="openai/gpt-5.6-terra",
        mantle_model_id="openai.gpt-5.6-terra",
        documented_ingress_regions=("us-east-1", "us-east-2", "us-west-2"),
        model_card_uri=(
            "https://docs.aws.amazon.com/bedrock/latest/userguide/"
            "model-card-openai-gpt-56-terra.html"
        ),
    ),
    "luna": GPT56TierPolicy(
        tier="luna",
        canonical_model_id="openai/gpt-5.6-luna",
        mantle_model_id="openai.gpt-5.6-luna",
        documented_ingress_regions=("us-east-1", "us-east-2", "us-west-2"),
        model_card_uri=(
            "https://docs.aws.amazon.com/bedrock/latest/userguide/"
            "model-card-openai-gpt-56-luna.html"
        ),
    ),
}


class MantleContractError(ValueError):
    """A Mantle endpoint contract is incomplete or unsafe."""


class MantleProviderError(RuntimeError):
    """A Mantle request or response violated the frozen benchmark contract."""


class MantleRouteUnavailable(MantleProviderError):
    """Mantle explicitly rejected a route before starting inference."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MantleContractError(f"{field_name} must be a lowercase SHA-256")


def _money_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).quantize(Decimal("1"), rounding=ROUND_CEILING))


@dataclass(frozen=True)
class MantleEndpointContract:
    """A model-specific, in-region Mantle Responses binding.

    ``mantle_model_id`` must come from an immutable snapshot of the Mantle
    ``/models`` response.  It is deliberately not guessed from the display
    name or the separate Bedrock Runtime catalog.
    """

    tier: MantleTier
    canonical_model_id: str
    mantle_model_id: str
    ingress_region: str
    model_catalog_uri: str
    model_catalog_sha256: str
    model_entry_sha256: str
    model_catalog_observed_at: str
    supports_responses: bool
    supports_client_side_tools: bool
    supports_strict_tools: bool
    structured_output_mode: StructuredOutputMode
    capability_evidence_uri: str
    capability_evidence_sha256: str
    price: BedrockPriceContract
    openrouter_fallback_model_id: str | None = None
    season_eligible: bool = False

    def __post_init__(self) -> None:
        policy = GPT56_TIER_POLICIES.get(self.tier)
        if policy is None:  # pragma: no cover - Literal protects typed callers
            raise MantleContractError("unsupported GPT-5.6 Mantle tier")
        if self.canonical_model_id != policy.canonical_model_id:
            raise MantleContractError(
                "Mantle tier and canonical FlavourBench model ID do not match"
            )
        if not self.mantle_model_id or self.mantle_model_id.strip() != self.mantle_model_id:
            raise MantleContractError("Mantle model ID must be non-empty and normalized")
        lowered_model_id = self.mantle_model_id.lower()
        if lowered_model_id.startswith(("global.", "us.", "eu.", "apac.")):
            raise MantleContractError(
                "GPT-5.6 Mantle must bind an in-region model, not a cross-region profile"
            )
        if self.mantle_model_id != policy.mantle_model_id:
            raise MantleContractError(
                "Mantle model ID must equal the model-specific ID documented by AWS"
            )
        if self.ingress_region not in policy.documented_ingress_regions:
            raise MantleContractError(
                f"GPT-5.6 {self.tier} is not documented in {self.ingress_region}"
            )
        _require_sha256(self.model_catalog_sha256, field_name="model catalog digest")
        _require_sha256(self.model_entry_sha256, field_name="model entry digest")
        _require_sha256(
            self.capability_evidence_sha256,
            field_name="capability evidence digest",
        )
        if (
            not self.model_catalog_uri
            or not self.model_catalog_observed_at
            or not self.capability_evidence_uri
        ):
            raise MantleContractError("Mantle catalog and capability evidence must be attributable")
        if self.openrouter_fallback_model_id not in {None, self.canonical_model_id}:
            raise MantleContractError(
                "OpenRouter fallback must bind exactly the same canonical model"
            )
        if self.structured_output_mode not in {
            "responses_json_schema",
            "client_validation_only",
            "unverified",
        }:
            raise MantleContractError("unsupported Mantle structured-output mode")
        if self.season_eligible and not all(
            (
                self.supports_responses,
                self.supports_client_side_tools,
                self.supports_strict_tools,
                self.structured_output_mode == "responses_json_schema",
            )
        ):
            raise MantleContractError(
                "season eligibility requires proven Responses, client tools, strict tools, "
                "and Responses json_schema support"
            )
        # Normalize/validate the frozen rate card now, not after admission.
        self.price.normalized()
        try:
            assert_public_catalog_safe(self.payload(), path="$mantle_endpoint_contract")
        except BedrockManifestError as error:
            raise MantleContractError(
                "Mantle endpoint contract contains unsafe public provenance"
            ) from error

    @property
    def endpoint_base_url(self) -> str:
        return f"https://bedrock-mantle.{self.ingress_region}.api.aws/openai/v1"

    @property
    def profile_scope(self) -> Literal["in_region"]:
        return "in_region"

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANTLE_CONTRACT_SCHEMA_VERSION,
            "tier": self.tier,
            "canonical_model_id": self.canonical_model_id,
            "mantle_model_id": self.mantle_model_id,
            "endpoint_kind": "bedrock_mantle_responses",
            "endpoint_base_url": self.endpoint_base_url,
            "ingress_region": self.ingress_region,
            "profile_scope": self.profile_scope,
            "documented_ingress_regions": list(
                GPT56_TIER_POLICIES[self.tier].documented_ingress_regions
            ),
            "context_window_tokens": GPT56_TIER_POLICIES[self.tier].context_window_tokens,
            "model_card_uri": GPT56_TIER_POLICIES[self.tier].model_card_uri,
            "model_catalog_uri": self.model_catalog_uri,
            "model_catalog_sha256": self.model_catalog_sha256,
            "model_entry_sha256": self.model_entry_sha256,
            "model_catalog_observed_at": self.model_catalog_observed_at,
            "supports_responses": self.supports_responses,
            "supports_client_side_tools": self.supports_client_side_tools,
            "supports_strict_tools": self.supports_strict_tools,
            "structured_output_mode": self.structured_output_mode,
            "capability_evidence_uri": self.capability_evidence_uri,
            "capability_evidence_sha256": self.capability_evidence_sha256,
            "price": {**self.price.normalized(), "price_sha256": self.price.sha256},
            "openrouter_fallback_model_id": self.openrouter_fallback_model_id,
            "season_eligible": self.season_eligible,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.payload())

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> MantleEndpointContract:
        if value.get("schema_version") != MANTLE_CONTRACT_SCHEMA_VERSION:
            raise MantleContractError("unsupported Mantle endpoint-contract schema")
        raw_price = value.get("price")
        if not isinstance(raw_price, Mapping):
            raise MantleContractError("Mantle endpoint contract has no frozen price")
        return cls(
            tier=str(value.get("tier") or ""),  # type: ignore[arg-type]
            canonical_model_id=str(value.get("canonical_model_id") or ""),
            mantle_model_id=str(value.get("mantle_model_id") or ""),
            ingress_region=str(value.get("ingress_region") or ""),
            model_catalog_uri=str(value.get("model_catalog_uri") or ""),
            model_catalog_sha256=str(value.get("model_catalog_sha256") or ""),
            model_entry_sha256=str(value.get("model_entry_sha256") or ""),
            model_catalog_observed_at=str(value.get("model_catalog_observed_at") or ""),
            supports_responses=value.get("supports_responses") is True,
            supports_client_side_tools=value.get("supports_client_side_tools") is True,
            supports_strict_tools=value.get("supports_strict_tools") is True,
            structured_output_mode=str(value.get("structured_output_mode") or "unverified"),  # type: ignore[arg-type]
            capability_evidence_uri=str(value.get("capability_evidence_uri") or ""),
            capability_evidence_sha256=str(value.get("capability_evidence_sha256") or ""),
            price=BedrockPriceContract(
                input_per_million_usd=str(raw_price.get("input_per_million_usd") or ""),
                output_per_million_usd=str(raw_price.get("output_per_million_usd") or ""),
                cache_read_per_million_usd=(
                    str(raw_price["cache_read_per_million_usd"])
                    if raw_price.get("cache_read_per_million_usd") is not None
                    else None
                ),
                cache_write_per_million_usd=(
                    str(raw_price["cache_write_per_million_usd"])
                    if raw_price.get("cache_write_per_million_usd") is not None
                    else None
                ),
                source_uri=str(raw_price.get("source_uri") or ""),
                observed_at=str(raw_price.get("observed_at") or ""),
            ),
            openrouter_fallback_model_id=(
                str(value["openrouter_fallback_model_id"])
                if value.get("openrouter_fallback_model_id") is not None
                else None
            ),
            season_eligible=value.get("season_eligible") is True,
        )


@dataclass(frozen=True)
class MantleToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    strict: bool = True

    def as_responses_tool(self) -> dict[str, Any]:
        if not self.name or len(self.name) > 64:
            raise MantleProviderError("Mantle tool names must contain 1-64 characters")
        validate_bedrock_json_schema(self.input_schema)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.input_schema),
            "strict": self.strict,
        }


@dataclass(frozen=True)
class MantleToolExecution:
    content: object
    is_error: bool = False


class MantleToolExecutor(Protocol):
    async def execute(self, name: str, arguments: Mapping[str, Any]) -> MantleToolExecution: ...


@dataclass(frozen=True)
class MantleInferenceConfig:
    max_output_tokens: int
    max_input_tokens_per_response: int
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None
    verbosity: Literal["low", "medium", "high"] | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0 or self.max_input_tokens_per_response <= 0:
            raise MantleProviderError("Mantle token bounds must be positive")

    def request_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"max_output_tokens": self.max_output_tokens}
        if self.reasoning_effort is not None:
            fields["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity is not None:
            fields["text"] = {"verbosity": self.verbosity}
        return fields


@dataclass(frozen=True)
class MantleGenerationSpec:
    arm_id: str
    canonical_model_id: str
    prompt: str
    system_prompt: str
    inference: MantleInferenceConfig
    tools: tuple[MantleToolDefinition, ...] = ()
    request_metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MantleUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    def plus(self, other: MantleUsage) -> MantleUsage:
        return MantleUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_output_tokens=(self.reasoning_output_tokens + other.reasoning_output_tokens),
        )


@dataclass(frozen=True)
class MantleCostProvenance:
    estimated_cost_micros: int | None
    estimate_complete: bool
    pricing_sha256: str
    pricing_source_uri: str
    pricing_observed_at: str
    cost_source: str = "frozen_rate_card_estimate"
    charged_cost_status: str = "not_available_in_responses_api"
    independently_reconciled_cost_micros: int | None = None
    billing_reconciliation_sha256: str | None = None


@dataclass(frozen=True)
class MantleToolTrace:
    round_index: int
    call_id: str
    name: str
    arguments: Mapping[str, Any] | None
    arguments_json: str
    is_error: bool
    error_kind: str | None
    result_json: str
    result_sha256: str


@dataclass(frozen=True)
class MantleRoundProvenance:
    round_index: int
    request_payload_json: str
    request_sha256: str
    response_payload_json: str
    response_sha256: str
    response_id: str
    aws_request_id: str
    returned_model_id: str
    status: str
    wall_clock_latency_ms: int
    usage: MantleUsage


@dataclass(frozen=True)
class MantleIdentityProvenance:
    canonical_model_id: str
    requested_mantle_model_id: str
    returned_model_ids: tuple[str, ...]
    provider: str
    endpoint_kind: str
    ingress_region: str
    profile_scope: Literal["in_region"]
    endpoint_base_url: str
    model_catalog_sha256: str
    contract_sha256: str
    provider_substitution: bool = False


@dataclass(frozen=True)
class MantleGenerationResult:
    answer_markdown: str
    output_json: Mapping[str, Any]
    finish_reason: str
    usage: MantleUsage
    wall_clock_latency_ms: int
    request_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    retries: int
    response_schema_sha256: str
    tool_schema_sha256: str
    request_metadata_sha256: str
    tool_traces: tuple[MantleToolTrace, ...]
    rounds: tuple[MantleRoundProvenance, ...]
    identity: MantleIdentityProvenance
    cost: MantleCostProvenance
    structured_output_enforcement: StructuredOutputMode
    store: Literal[False] = False
    rank_eligible: bool = False
    provider_substitution: bool = False
    unpooled: bool = False


@dataclass(frozen=True)
class MantleTransportResponse:
    body: Mapping[str, Any]
    status_code: int
    aws_request_id: str
    elapsed_ms: int


class MantleTransportError(RuntimeError):
    """Sanitized transport failure with explicit dispatch certainty."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str = "",
        aws_request_id: str = "",
        dispatch_state: DispatchState = "ambiguous",
        route_unavailable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.aws_request_id = aws_request_id
        self.dispatch_state = dispatch_state
        self.route_unavailable = route_unavailable


class MantleResponsesTransport(Protocol):
    async def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> MantleTransportResponse: ...


class HttpxMantleResponsesTransport:
    """Thin transport around an already authenticated HTTP client.

    Authentication construction is intentionally out of scope: this object
    never reads environment credentials and provenance never records headers.
    Network ambiguity is reported as ambiguous and is never automatically
    retried or used to trigger provider substitution.
    """

    _route_rejection_statuses = frozenset({400, 401, 403, 404, 422, 429})
    _route_error_codes = frozenset(
        {
            "access_denied",
            "model_not_found",
            "model_not_available",
            "unsupported_region",
            "rate_limit_exceeded",
        }
    )

    def __init__(self, client: httpx.AsyncClient, *, endpoint_base_url: str) -> None:
        expected_prefix = "https://bedrock-mantle."
        if not endpoint_base_url.startswith(expected_prefix) or not endpoint_base_url.endswith(
            ".api.aws/openai/v1"
        ):
            raise MantleContractError("Mantle transport requires an AWS Mantle v1 endpoint")
        self.client = client
        self.endpoint_base_url = endpoint_base_url.rstrip("/")

    async def create_response(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> MantleTransportResponse:
        started = time.monotonic()
        try:
            response = await self.client.post(
                f"{self.endpoint_base_url}/responses",
                json=dict(payload),
                headers={"Idempotency-Key": idempotency_key},
            )
        except httpx.RequestError as error:
            raise MantleTransportError(
                f"Mantle transport failed ambiguously: {type(error).__name__}",
                dispatch_state="ambiguous",
            ) from error
        elapsed_ms = round((time.monotonic() - started) * 1000)
        request_id = response.headers.get("x-amzn-requestid") or response.headers.get(
            "x-amzn-request-id", ""
        )
        try:
            body = response.json()
        except ValueError as error:
            raise MantleTransportError(
                "Mantle returned a non-JSON response",
                status_code=response.status_code,
                aws_request_id=request_id,
                dispatch_state=(
                    "rejected_before_inference"
                    if response.status_code in self._route_rejection_statuses
                    else "ambiguous"
                ),
            ) from error
        if not isinstance(body, Mapping):
            raise MantleTransportError(
                "Mantle returned a non-object response",
                status_code=response.status_code,
                aws_request_id=request_id,
                dispatch_state="ambiguous",
            )
        if response.status_code >= 400:
            raw_error = body.get("error")
            error_map = raw_error if isinstance(raw_error, Mapping) else {}
            error_code = str(error_map.get("code") or error_map.get("type") or "")
            rejected = response.status_code in self._route_rejection_statuses
            raise MantleTransportError(
                f"Mantle rejected the request with HTTP {response.status_code}",
                status_code=response.status_code,
                error_code=error_code,
                aws_request_id=request_id,
                dispatch_state="rejected_before_inference" if rejected else "ambiguous",
                route_unavailable=(
                    rejected
                    and (
                        error_code in self._route_error_codes
                        or response.status_code in {401, 403, 404, 429}
                    )
                ),
            )
        return MantleTransportResponse(
            body=dict(body),
            status_code=response.status_code,
            aws_request_id=request_id,
            elapsed_ms=elapsed_ms,
        )


@dataclass(frozen=True)
class MantleReservationRequest:
    arm_id: str
    canonical_model_id: str
    contract_sha256: str
    worst_case_estimated_cost_micros: int


@dataclass(frozen=True)
class MantleAdmissionTicket:
    reservation_id: str
    arm_id: str
    canonical_model_id: str
    contract_sha256: str
    reserved_cost_micros: int
    hard_cap_micros: int
    admitted: bool
    admission_status: str
    admission_evidence_sha256: str


def admission_ticket_from_governor_decision(
    request: MantleReservationRequest,
    decision: BedrockAdmissionDecision,
    *,
    reservation_id: str,
    admission_evidence_sha256: str,
) -> MantleAdmissionTicket:
    """Bind the shared 85/95/100 Bedrock governor to a Mantle reservation.

    A PostgreSQL transaction should persist the governor decision and reserve
    the exposure before returning this ticket.  This helper only validates and
    normalizes that already-transactional decision; it performs no I/O.
    """

    if not reservation_id:
        raise MantleProviderError("Mantle reservation ID must be non-empty")
    try:
        _require_sha256(
            admission_evidence_sha256,
            field_name="Mantle admission evidence digest",
        )
    except MantleContractError as error:
        raise MantleProviderError(str(error)) from error
    requested_micros = _money_micros(decision.requested_reservation_usd)
    if requested_micros != request.worst_case_estimated_cost_micros:
        raise MantleProviderError(
            "Mantle governor decision does not match the requested worst-case reserve"
        )
    return MantleAdmissionTicket(
        reservation_id=reservation_id,
        arm_id=request.arm_id,
        canonical_model_id=request.canonical_model_id,
        contract_sha256=request.contract_sha256,
        reserved_cost_micros=requested_micros,
        hard_cap_micros=_money_micros(decision.hard_cap_usd),
        admitted=decision.admitted,
        admission_status=decision.status,
        admission_evidence_sha256=admission_evidence_sha256,
    )


@dataclass(frozen=True)
class MantleBudgetFinalization:
    reservation_id: str
    outcome: BudgetOutcome
    estimated_cost_micros: int | None
    estimate_complete: bool
    note: str


class MantleBudgetController(Protocol):
    """Transactional budget boundary implemented by the worker/storage lane."""

    async def reserve(self, request: MantleReservationRequest) -> MantleAdmissionTicket: ...

    async def finalize(self, finalization: MantleBudgetFinalization) -> None: ...


def worst_case_cost_micros(
    contract: MantleEndpointContract,
    inference: MantleInferenceConfig,
    *,
    maximum_responses: int,
) -> int:
    """Reserve all tokens at non-cache rates; never assume a cache discount."""

    if maximum_responses <= 0:
        raise MantleProviderError("maximum Mantle response count must be positive")
    price = contract.price.normalized()
    context_window = GPT56_TIER_POLICIES[contract.tier].context_window_tokens
    if inference.max_output_tokens >= context_window:
        raise MantleProviderError("Mantle output bound must be below the context window")
    # Responses has no request parameter that enforces a caller-declared input
    # token ceiling. Reserve the entire remaining documented context window so
    # a tokenizer-estimation error cannot under-reserve a paid call.
    reserved_input_tokens = max(
        inference.max_input_tokens_per_response,
        context_window - inference.max_output_tokens,
    )
    input_cost = Decimal(reserved_input_tokens) * Decimal(price["input_per_million_usd"])
    output_cost = Decimal(inference.max_output_tokens) * Decimal(price["output_per_million_usd"])
    # A USD-per-million-token number is numerically micro-USD per token.
    return int(
        ((input_cost + output_cost) * maximum_responses).quantize(
            Decimal("1"), rounding=ROUND_CEILING
        )
    )


def _usage(value: Mapping[str, Any]) -> MantleUsage:
    raw = value.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}

    def count(mapping: Mapping[str, Any], name: str) -> int:
        amount = mapping.get(name, 0)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise MantleProviderError(f"Mantle returned invalid usage.{name}")
        return amount

    input_details_raw = usage.get("input_tokens_details")
    input_details = input_details_raw if isinstance(input_details_raw, Mapping) else {}
    output_details_raw = usage.get("output_tokens_details")
    output_details = output_details_raw if isinstance(output_details_raw, Mapping) else {}
    return MantleUsage(
        input_tokens=count(usage, "input_tokens"),
        output_tokens=count(usage, "output_tokens"),
        total_tokens=count(usage, "total_tokens"),
        cached_input_tokens=count(input_details, "cached_tokens"),
        reasoning_output_tokens=count(output_details, "reasoning_tokens"),
    )


def _cost(usage: MantleUsage, contract: MantleEndpointContract) -> MantleCostProvenance:
    price = contract.price
    complete = not (usage.cached_input_tokens and price.cache_read_per_million_usd is None)
    estimated: int | None = None
    if complete:
        uncached = usage.input_tokens - usage.cached_input_tokens
        if uncached < 0:
            raise MantleProviderError("Mantle cached input exceeds total input usage")
        micros = Decimal(uncached) * Decimal(price.input_per_million_usd)
        micros += Decimal(usage.output_tokens) * Decimal(price.output_per_million_usd)
        if usage.cached_input_tokens:
            micros += Decimal(usage.cached_input_tokens) * Decimal(
                price.cache_read_per_million_usd or "0"
            )
        estimated = int(micros.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return MantleCostProvenance(
        estimated_cost_micros=estimated,
        estimate_complete=complete,
        pricing_sha256=price.sha256,
        pricing_source_uri=price.source_uri,
        pricing_observed_at=price.observed_at,
    )


def _validated_answer(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise MantleProviderError("Mantle final output was not valid JSON") from error
    expected = {
        "answer_markdown",
        "ingredient_mentions",
        "constraints_addressed",
        "uncertainties",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise MantleProviderError("Mantle final output did not match the frozen object schema")
    if not isinstance(value["answer_markdown"], str) or not value["answer_markdown"]:
        raise MantleProviderError("Mantle returned an empty culinary answer")
    for field_name in expected - {"answer_markdown"}:
        items = value[field_name]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise MantleProviderError(f"Mantle final output has invalid {field_name}")
    return value


def _output_text(body: Mapping[str, Any]) -> str:
    output = body.get("output")
    if not isinstance(output, Sequence) or isinstance(output, str | bytes):
        raise MantleProviderError("Mantle response has invalid output items")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, Sequence) or isinstance(content, str | bytes):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _function_calls(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    output = body.get("output")
    if not isinstance(output, Sequence) or isinstance(output, str | bytes):
        raise MantleProviderError("Mantle response has invalid output items")
    return [
        item for item in output if isinstance(item, Mapping) and item.get("type") == "function_call"
    ]


def _safe_request_metadata(values: Mapping[str, str]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise MantleProviderError("Mantle metadata must contain only strings")
        lowered = key.lower()
        if any(marker in lowered for marker in ("token", "secret", "password", "credential")):
            raise MantleProviderError("credential-like Mantle metadata is forbidden")
        safe[key] = value
    return safe


def _response_format() -> dict[str, Any]:
    validate_bedrock_json_schema(BEDROCK_FINAL_SCHEMA)
    return {
        "type": "json_schema",
        "name": "flavourbench_answer",
        "description": "A blinded culinary benchmark answer",
        "schema": BEDROCK_FINAL_SCHEMA,
        "strict": True,
    }


class MantleResponsesProvider:
    """Run one fixed Mantle model through a bounded client-side Epicure loop."""

    def __init__(
        self,
        transport: MantleResponsesTransport,
        contract: MantleEndpointContract,
        *,
        budget: MantleBudgetController,
        tool_executor: MantleToolExecutor | None = None,
        max_tool_rounds: int = 8,
        max_tool_calls_per_round: int = 4,
        max_tool_calls_total: int = 16,
        max_pre_dispatch_retries: int = 2,
    ) -> None:
        if not contract.supports_responses:
            raise MantleProviderError("Mantle endpoint lacks frozen Responses evidence")
        if contract.structured_output_mode == "unverified":
            raise MantleProviderError("Mantle structured-output capability is not frozen")
        if min(max_tool_rounds, max_tool_calls_per_round, max_tool_calls_total) < 1:
            raise MantleProviderError("Mantle tool bounds must be positive")
        if not 0 <= max_pre_dispatch_retries <= 2:
            raise MantleProviderError("Mantle permits at most two pre-dispatch retries")
        self.transport = transport
        self.contract = contract
        self.budget = budget
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls_per_round = max_tool_calls_per_round
        self.max_tool_calls_total = max_tool_calls_total
        self.max_pre_dispatch_retries = max_pre_dispatch_retries

    async def _create(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
        completed_responses: int,
    ) -> tuple[MantleTransportResponse, int]:
        retries = 0
        while True:
            try:
                return (
                    await self.transport.create_response(
                        payload,
                        idempotency_key=idempotency_key,
                    ),
                    retries,
                )
            except MantleTransportError as error:
                # Only an injected transport that can prove it never dispatched
                # may allow a retry. HTTP timeouts and all 5xx responses remain
                # ambiguous and hold the full reservation.
                if error.dispatch_state == "not_sent" and retries < self.max_pre_dispatch_retries:
                    retries += 1
                    await asyncio.sleep(0)
                    continue
                if (
                    completed_responses == 0
                    and error.dispatch_state == "rejected_before_inference"
                    and error.route_unavailable
                ):
                    raise MantleRouteUnavailable(
                        "Mantle explicitly rejected the model route before inference: "
                        f"{error.error_code or error.status_code or 'unknown'}"
                    ) from error
                raise MantleProviderError(
                    "Mantle request failed without retry-safe evidence: "
                    f"{error.error_code or error.status_code or error.dispatch_state}"
                ) from error

    async def generate(self, spec: MantleGenerationSpec) -> MantleGenerationResult:
        if spec.canonical_model_id != self.contract.canonical_model_id:
            raise MantleProviderError("generation and Mantle canonical model IDs differ")
        context_window = GPT56_TIER_POLICIES[self.contract.tier].context_window_tokens
        if spec.inference.max_input_tokens_per_response > context_window:
            raise MantleProviderError("Mantle input bound exceeds the context window")
        if spec.tools and not all(
            (
                self.contract.supports_client_side_tools,
                self.contract.supports_strict_tools,
            )
        ):
            raise MantleProviderError("Mantle tool capability is not frozen")
        if spec.tools and self.tool_executor is None:
            raise MantleProviderError("Mantle tools require a client-side Epicure executor")
        if any(not tool.strict for tool in spec.tools):
            raise MantleProviderError("FlavourBench requires strict Mantle function tools")

        tool_payload = [tool.as_responses_tool() for tool in spec.tools]
        response_schema_sha256 = _sha256(BEDROCK_FINAL_SCHEMA)
        tool_schema_sha256 = _sha256(tool_payload)
        metadata = {
            **_safe_request_metadata(spec.request_metadata),
            "flavourbench_arm_sha256": hashlib.sha256(spec.arm_id.encode()).hexdigest(),
            "flavourbench_contract_sha256": self.contract.sha256,
            "flavourbench_schema_sha256": response_schema_sha256,
            "flavourbench_tools_sha256": tool_schema_sha256,
        }
        metadata_sha256 = _sha256(metadata)
        maximum_responses = self.max_tool_rounds + 1
        reserve_micros = worst_case_cost_micros(
            self.contract,
            spec.inference,
            maximum_responses=maximum_responses,
        )
        reservation_request = MantleReservationRequest(
            arm_id=spec.arm_id,
            canonical_model_id=spec.canonical_model_id,
            contract_sha256=self.contract.sha256,
            worst_case_estimated_cost_micros=reserve_micros,
        )
        ticket = await self.budget.reserve(reservation_request)
        if (
            not ticket.admitted
            or not ticket.reservation_id
            or ticket.arm_id != spec.arm_id
            or ticket.canonical_model_id != spec.canonical_model_id
            or ticket.contract_sha256 != self.contract.sha256
            or ticket.reserved_cost_micros < reserve_micros
            or ticket.hard_cap_micros < ticket.reserved_cost_micros
            or len(ticket.admission_evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in ticket.admission_evidence_sha256
            )
        ):
            raise MantleProviderError("Mantle hard-budget admission was denied or malformed")

        history: list[dict[str, Any]] = [
            {"role": "user", "content": spec.prompt},
        ]
        common: dict[str, Any] = {
            "model": self.contract.mantle_model_id,
            "instructions": (
                spec.system_prompt
                if self.contract.structured_output_mode == "responses_json_schema"
                else (
                    spec.system_prompt
                    + "\nReturn the final answer only as JSON matching this schema: "
                    + _canonical_json(BEDROCK_FINAL_SCHEMA).decode("utf-8")
                )
            ),
            "store": False,
            "metadata": metadata,
            **spec.inference.request_fields(),
        }
        if self.contract.structured_output_mode == "responses_json_schema":
            text_config = dict(common.get("text") or {})
            text_config["format"] = _response_format()
            common["text"] = text_config
        if tool_payload:
            common.update(
                {
                    "tools": tool_payload,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                }
            )

        started = time.monotonic()
        total_usage = MantleUsage()
        traces: list[MantleToolTrace] = []
        rounds: list[MantleRoundProvenance] = []
        response_ids: list[str] = []
        request_ids: list[str] = []
        returned_models: list[str] = []
        total_tool_calls = 0
        total_retries = 0
        invalid_argument_repairs = 0
        final_answer: dict[str, Any] | None = None
        finish_reason = "unknown"
        terminal_outcome: BudgetOutcome = "failed_hold_full_reservation"
        final_cost: MantleCostProvenance | None = None

        try:
            for round_index in range(maximum_responses):
                payload = {**common, "input": history}
                payload_json = _canonical_json(payload).decode("utf-8")
                idempotency_key = _sha256(
                    {
                        "arm_id": spec.arm_id,
                        "contract_sha256": self.contract.sha256,
                        "request_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
                        "round_index": round_index,
                    }
                )
                response, retry_count = await self._create(
                    payload,
                    idempotency_key=idempotency_key,
                    completed_responses=len(rounds),
                )
                total_retries += retry_count
                if response.status_code != 200:
                    raise MantleProviderError(
                        f"Mantle successful transport returned HTTP {response.status_code}"
                    )
                body = dict(response.body)
                response_id = str(body.get("id") or "")
                request_id = response.aws_request_id
                returned_model = str(body.get("model") or "")
                status = str(body.get("status") or "")
                if not response_id or not request_id:
                    raise MantleProviderError("Mantle omitted a response or AWS request ID")
                if returned_model != self.contract.mantle_model_id:
                    raise MantleProviderError("Mantle returned a different model identity")
                if status not in {"completed", "incomplete"}:
                    raise MantleProviderError(f"Mantle returned unsupported status {status!r}")
                round_usage = _usage(body)
                total_usage = total_usage.plus(round_usage)
                response_json = _canonical_json(body).decode("utf-8")
                rounds.append(
                    MantleRoundProvenance(
                        round_index=round_index,
                        request_payload_json=payload_json,
                        request_sha256=hashlib.sha256(payload_json.encode()).hexdigest(),
                        response_payload_json=response_json,
                        response_sha256=hashlib.sha256(response_json.encode()).hexdigest(),
                        response_id=response_id,
                        aws_request_id=request_id,
                        returned_model_id=returned_model,
                        status=status,
                        wall_clock_latency_ms=response.elapsed_ms,
                        usage=round_usage,
                    )
                )
                response_ids.append(response_id)
                request_ids.append(request_id)
                if returned_model not in returned_models:
                    returned_models.append(returned_model)
                if status == "incomplete":
                    details = body.get("incomplete_details")
                    reason = (
                        str(details.get("reason") or "unknown")
                        if isinstance(details, Mapping)
                        else "unknown"
                    )
                    raise MantleProviderError(f"Mantle response was incomplete: {reason}")

                calls = _function_calls(body)
                if not calls:
                    final_answer = _validated_answer(_output_text(body))
                    finish_reason = "completed"
                    break
                if round_index >= self.max_tool_rounds:
                    raise MantleProviderError("Mantle exhausted the frozen tool-round cap")
                if len(calls) > self.max_tool_calls_per_round:
                    raise MantleProviderError("Mantle tool fan-out exceeded the per-round cap")
                total_tool_calls += len(calls)
                if total_tool_calls > self.max_tool_calls_total:
                    raise MantleProviderError("Mantle tool calls exceeded the generation cap")

                output_items = body.get("output")
                assert isinstance(output_items, Sequence)
                history.extend(dict(item) for item in output_items if isinstance(item, Mapping))
                for call in calls:
                    call_id = str(call.get("call_id") or call.get("id") or "")
                    name = str(call.get("name") or "")
                    arguments_json = str(call.get("arguments") or "")
                    if not call_id or not name:
                        raise MantleProviderError("Mantle returned an invalid function call")
                    arguments: Mapping[str, Any] | None = None
                    execution: MantleToolExecution
                    error_kind: str | None = None
                    try:
                        parsed = json.loads(arguments_json)
                        if not isinstance(parsed, Mapping):
                            raise ValueError("function arguments are not an object")
                        arguments = dict(parsed)
                    except (json.JSONDecodeError, ValueError):
                        invalid_argument_repairs += 1
                        if invalid_argument_repairs > 1:
                            raise MantleProviderError(
                                "Mantle exceeded the single invalid-argument repair turn"
                            ) from None
                        error_kind = "invalid_tool_arguments"
                        execution = MantleToolExecution(
                            {
                                "error": "invalid_tool_arguments",
                                "instruction": "Return one valid JSON object for this tool.",
                            },
                            is_error=True,
                        )
                    else:
                        if name not in {tool.name for tool in spec.tools}:
                            invalid_argument_repairs += 1
                            if invalid_argument_repairs > 1:
                                raise MantleProviderError(
                                    "Mantle exceeded the single unknown-tool repair turn"
                                )
                            error_kind = "unknown_tool"
                            execution = MantleToolExecution(
                                {
                                    "error": "unknown_tool",
                                    "available_tools": sorted(tool.name for tool in spec.tools),
                                },
                                is_error=True,
                            )
                        else:
                            assert self.tool_executor is not None
                            execution = await self.tool_executor.execute(name, arguments)
                            if execution.is_error:
                                error_kind = "tool_execution_error"
                    result_json = _canonical_json(execution.content).decode("utf-8")
                    traces.append(
                        MantleToolTrace(
                            round_index=round_index,
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                            arguments_json=arguments_json,
                            is_error=execution.is_error,
                            error_kind=error_kind,
                            result_json=result_json,
                            result_sha256=hashlib.sha256(result_json.encode()).hexdigest(),
                        )
                    )
                    history.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result_json,
                        }
                    )

            if final_answer is None:
                raise MantleProviderError("Mantle returned no final structured response")
            final_cost = _cost(total_usage, self.contract)
            terminal_outcome = "success_estimate_only"
            identity = MantleIdentityProvenance(
                canonical_model_id=self.contract.canonical_model_id,
                requested_mantle_model_id=self.contract.mantle_model_id,
                returned_model_ids=tuple(returned_models),
                provider="amazon-bedrock-mantle",
                endpoint_kind="bedrock_mantle_responses",
                ingress_region=self.contract.ingress_region,
                profile_scope="in_region",
                endpoint_base_url=self.contract.endpoint_base_url,
                model_catalog_sha256=self.contract.model_catalog_sha256,
                contract_sha256=self.contract.sha256,
            )
            return MantleGenerationResult(
                answer_markdown=final_answer["answer_markdown"],
                output_json=final_answer,
                finish_reason=finish_reason,
                usage=total_usage,
                wall_clock_latency_ms=round((time.monotonic() - started) * 1000),
                request_ids=tuple(request_ids),
                response_ids=tuple(response_ids),
                retries=total_retries,
                response_schema_sha256=response_schema_sha256,
                tool_schema_sha256=tool_schema_sha256,
                request_metadata_sha256=metadata_sha256,
                tool_traces=tuple(traces),
                rounds=tuple(rounds),
                identity=identity,
                cost=final_cost,
                structured_output_enforcement=self.contract.structured_output_mode,
                rank_eligible=self.contract.season_eligible,
            )
        except MantleRouteUnavailable:
            # An explicit, pre-inference rejection may release the reservation.
            terminal_outcome = "not_sent_release_allowed"
            raise
        finally:
            await self.budget.finalize(
                MantleBudgetFinalization(
                    reservation_id=ticket.reservation_id,
                    outcome=terminal_outcome,
                    estimated_cost_micros=(
                        final_cost.estimated_cost_micros if final_cost is not None else None
                    ),
                    estimate_complete=(
                        final_cost.estimate_complete if final_cost is not None else False
                    ),
                    note=(
                        "Responses usage priced from frozen rate card; AWS charged cost is "
                        "not claimed."
                        if final_cost is not None
                        else "Hold the full reservation unless dispatch was explicitly rejected "
                        "before inference."
                    ),
                )
            )


def frontier_contract_plan() -> dict[str, Any]:
    """Return a credential-free, zero-call plan for the three frontier tiers."""

    models = []
    for tier in ("sol", "terra", "luna"):
        policy = GPT56_TIER_POLICIES[tier]
        models.append(
            {
                "tier": tier,
                "canonical_model_id": policy.canonical_model_id,
                "documented_ingress_regions": list(policy.documented_ingress_regions),
                "context_window_tokens": policy.context_window_tokens,
                "profile_scope": "in_region",
                "mantle_model_id": policy.mantle_model_id,
                "model_card_uri": policy.model_card_uri,
                "openrouter_fallback_model_id": policy.canonical_model_id,
                "rank_eligible": False,
                "blockers": [
                    "freeze a content-addressed Mantle /models snapshot confirming the "
                    "AWS-documented model ID",
                    "freeze Responses function-tool conformance evidence",
                    "freeze Responses text.format json_schema conformance evidence",
                    "freeze a model-specific AWS rate card",
                    "complete a paid contract smoke under a separately admitted reservation",
                ],
            }
        )
    return {
        "schema_version": MANTLE_PLAN_SCHEMA_VERSION,
        "operation": "plan_bedrock_mantle_gpt56_contracts",
        "provider_calls_made": False,
        "inference_calls": 0,
        "credential_environment_read": False,
        "api": "OpenAI-compatible Responses API",
        "api_reference": MANTLE_API_REFERENCE,
        "gpt56_endpoint_base_url_pattern": ("https://bedrock-mantle.{region}.api.aws/openai/v1"),
        "region_evidence": GPT56_LAUNCH_REFERENCE,
        "server_side_tools_enabled": False,
        "epicure_tool_ownership": "flavourbench_client_side_only",
        "store": False,
        "tool_only_exploratory_mode": {
            "structured_output_mode": "client_validation_only",
            "native_json_schema_claimed": False,
            "rank_eligible": False,
        },
        "models": models,
        "api_uncertainties": [
            "The generic Responses surface supports text.format json_schema, but GPT-5.6 "
            "model-specific conformance must be contract-smoked before ranking.",
            "The AWS request-ID response header must be confirmed by contract smoke.",
            "Responses usage is not an AWS invoice; costs remain rate-card estimates until "
            "independent billing reconciliation.",
        ],
    }


def _read_contract(path: Path) -> MantleEndpointContract:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MantleContractError(f"could not read Mantle contract: {path}") from error
    if not isinstance(value, Mapping):
        raise MantleContractError("Mantle contract document must be an object")
    contract = MantleEndpointContract.from_payload(value)
    if value != contract.payload():
        raise MantleContractError("Mantle contract is not canonical")
    return contract


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a zero-call Bedrock Mantle GPT-5.6 contract plan"
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=8_000)
    parser.add_argument("--max-output-tokens", type=int, default=3_000)
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    arguments = parser.parse_args(argv)
    plan = frontier_contract_plan()
    if arguments.contract is not None:
        contract = _read_contract(arguments.contract)
        inference = MantleInferenceConfig(
            max_output_tokens=arguments.max_output_tokens,
            max_input_tokens_per_response=arguments.max_input_tokens,
        )
        plan = {
            **plan,
            "validated_contract": contract.payload(),
            "contract_sha256": contract.sha256,
            "maximum_responses_per_arm": arguments.max_tool_rounds + 1,
            "worst_case_reservation_micros": worst_case_cost_micros(
                contract,
                inference,
                maximum_responses=arguments.max_tool_rounds + 1,
            ),
            "ready_for_inference": contract.season_eligible,
        }
    print(json.dumps(plan, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
