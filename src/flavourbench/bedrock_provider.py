"""Isolated Amazon Bedrock Converse adapter for FlavourBench.

It implements the Bedrock Runtime Converse shapes used by the worker for
client-side tool use and structured JSON output. The transport remains isolated
from OpenRouter so a frozen arm cannot change providers silently.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol

from .bedrock_manifest import BedrockEndpointContract

BEDROCK_FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_markdown": {"type": "string"},
        "ingredient_mentions": {"type": "array", "items": {"type": "string"}},
        "constraints_addressed": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_markdown",
        "ingredient_mentions",
        "constraints_addressed",
        "uncertainties",
    ],
    "additionalProperties": False,
}

_UNSUPPORTED_BEDROCK_SCHEMA_KEYS = frozenset(
    {
        "$id",
        "$schema",
        "default",
        "discriminator",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "maxItems",
        "multipleOf",
        "oneOf",
        "pattern",
        "patternProperties",
        "title",
        "uniqueItems",
    }
)

_BEDROCK_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "definitions",
        "description",
        "enum",
        "format",
        "items",
        "minItems",
        "properties",
        "required",
        "type",
    }
)
_BEDROCK_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)


class BedrockProviderError(RuntimeError):
    """A Bedrock request or response violated the frozen contract."""


class BedrockRouteUnavailable(BedrockProviderError):
    """Bedrock rejected a request before returning a generation."""


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def count_tokens(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BedrockToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    strict: bool = True

    def as_converse_tool(self) -> dict[str, Any]:
        if not self.name or len(self.name) > 64:
            raise BedrockProviderError("Bedrock tool names must contain 1-64 characters")
        validate_bedrock_json_schema(self.input_schema)
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": dict(self.input_schema)},
                "strict": self.strict,
            }
        }


@dataclass(frozen=True)
class BedrockToolExecution:
    content: object
    is_error: bool = False


class BedrockToolExecutor(Protocol):
    async def execute(self, name: str, arguments: Mapping[str, Any]) -> BedrockToolExecution: ...


@dataclass(frozen=True)
class BedrockInferenceConfig:
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()

    def as_converse_config(self) -> dict[str, Any]:
        if self.max_tokens <= 0:
            raise BedrockProviderError("max_tokens must be positive")
        value: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            value["temperature"] = self.temperature
        if self.top_p is not None:
            value["topP"] = self.top_p
        if self.stop_sequences:
            value["stopSequences"] = list(self.stop_sequences)
        return value


@dataclass(frozen=True)
class BedrockGenerationSpec:
    arm_id: str
    canonical_model_id: str
    prompt: str
    system_prompt: str
    inference: BedrockInferenceConfig
    tools: tuple[BedrockToolDefinition, ...] = ()
    request_metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BedrockUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    def plus(self, other: BedrockUsage) -> BedrockUsage:
        return BedrockUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
        )


@dataclass(frozen=True)
class BedrockToolTrace:
    round_index: int
    tool_use_id: str
    name: str
    arguments: Mapping[str, Any]
    is_error: bool
    result_sha256: str


@dataclass(frozen=True)
class BedrockCostProvenance:
    estimated_cost_micros: int | None
    estimate_complete: bool
    pricing_sha256: str
    pricing_source_uri: str
    pricing_observed_at: str
    cost_source: str = "frozen_rate_card_estimate"
    independent_billing_reconciliation_status: str = "not_reconciled"
    independently_reconciled_cost_micros: int | None = None
    billing_reconciliation_sha256: str | None = None


@dataclass(frozen=True)
class BedrockIdentityProvenance:
    canonical_model_id: str
    requested_model_or_profile_id: str
    requested_model_or_profile_arn_redacted: str
    requested_model_or_profile_arn_sha256: str
    expected_foundation_model_ids: tuple[str, ...]
    frozen_destination_model_arns: tuple[str, ...]
    frozen_destination_model_arn_sha256s: tuple[str, ...]
    returned_model_ids: tuple[str, ...]
    returned_model_id_sha256s: tuple[str, ...]
    provider: str
    ingress_region: str
    profile_scope: str
    profile_scope_sha256: str
    endpoint_kind: str
    actual_execution_region: str | None
    actual_foundation_model_id: str | None
    identity_evidence: str
    provider_substitution: bool = False


@dataclass(frozen=True)
class BedrockGenerationResult:
    answer_markdown: str
    output_json: Mapping[str, Any]
    finish_reason: str
    usage: BedrockUsage
    wall_clock_latency_ms: int
    service_latency_ms: int
    response_latencies_ms: tuple[int, ...]
    request_ids: tuple[str, ...]
    retries: int
    response_schema_sha256: str
    tool_schema_sha256: str
    tool_traces: tuple[BedrockToolTrace, ...]
    identity: BedrockIdentityProvenance
    cost: BedrockCostProvenance
    rank_eligible: bool
    provider_substitution: bool = False
    unpooled: bool = False


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def validate_bedrock_json_schema(schema: Mapping[str, Any]) -> None:
    """Reject schema features Bedrock's documented structured subset omits."""

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            unsupported = _UNSUPPORTED_BEDROCK_SCHEMA_KEYS.intersection(value)
            if unsupported:
                raise BedrockProviderError(
                    "Bedrock structured output does not support: " + ", ".join(sorted(unsupported))
                )
            if "additionalProperties" in value and value["additionalProperties"] is not False:
                raise BedrockProviderError(
                    "Bedrock requires additionalProperties=false when the keyword is present"
                )
            if (value.get("type") == "object" or "properties" in value) and value.get(
                "additionalProperties"
            ) is not False:
                raise BedrockProviderError(
                    "Bedrock strict object schemas require additionalProperties=false"
                )
            if "minItems" in value and value["minItems"] not in {0, 1}:
                raise BedrockProviderError("Bedrock supports minItems only with values 0 or 1")
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for child in value:
                visit(child)

    visit(schema)


def project_bedrock_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically widen a JSON schema to Bedrock's documented subset.

    Epicure continues to validate the original MCP arguments at execution.
    This projection is used only for Bedrock strict tool guidance. Unsupported
    bounds are removed, and Pydantic discriminated ``oneOf`` unions become the
    supported ``anyOf`` form. Unknown keywords fail closed so a future schema
    change cannot be silently weakened.
    """

    def project(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "oneOf":
                if "anyOf" in value:
                    raise BedrockProviderError(
                        f"Bedrock schema projection found oneOf and anyOf at {path}"
                    )
                if not isinstance(child, Sequence) or isinstance(child, str | bytes):
                    raise BedrockProviderError(
                        f"Bedrock schema projection found invalid oneOf at {path}"
                    )
                result["anyOf"] = [
                    project(item, path=f"{path}.oneOf[{index}]")
                    if isinstance(item, Mapping)
                    else _raise_projection(path, "oneOf member")
                    for index, item in enumerate(child)
                ]
                continue
            if key in _UNSUPPORTED_BEDROCK_SCHEMA_KEYS:
                continue
            if key not in _BEDROCK_SCHEMA_KEYS:
                raise BedrockProviderError(
                    f"Bedrock schema projection found unsupported keyword {key!r} at {path}"
                )
            if key in {"properties", "$defs", "definitions"}:
                if not isinstance(child, Mapping):
                    raise BedrockProviderError(
                        f"Bedrock schema projection found invalid {key} at {path}"
                    )
                result[key] = {
                    str(name): project(item, path=f"{path}.{key}.{name}")
                    if isinstance(item, Mapping)
                    else _raise_projection(path, f"{key} member")
                    for name, item in child.items()
                }
                continue
            if key in {"anyOf", "allOf"}:
                if not isinstance(child, Sequence) or isinstance(child, str | bytes):
                    raise BedrockProviderError(
                        f"Bedrock schema projection found invalid {key} at {path}"
                    )
                result[key] = [
                    project(item, path=f"{path}.{key}[{index}]")
                    if isinstance(item, Mapping)
                    else _raise_projection(path, f"{key} member")
                    for index, item in enumerate(child)
                ]
                continue
            if key == "items":
                if not isinstance(child, Mapping):
                    raise BedrockProviderError(
                        f"Bedrock schema projection found invalid items at {path}"
                    )
                result[key] = project(child, path=f"{path}.items")
                continue
            if key == "additionalProperties":
                if child is not False:
                    raise BedrockProviderError(
                        "Bedrock schema projection refuses non-false additionalProperties"
                    )
                result[key] = False
                continue
            if key == "minItems":
                if child in {0, 1} and not isinstance(child, bool):
                    result[key] = child
                continue
            if key == "format":
                if child not in _BEDROCK_STRING_FORMATS:
                    raise BedrockProviderError(
                        f"Bedrock schema projection found unsupported format at {path}"
                    )
                result[key] = child
                continue
            if key == "$ref":
                if not isinstance(child, str) or not child.startswith(
                    ("#/$defs/", "#/definitions/")
                ):
                    raise BedrockProviderError(
                        f"Bedrock schema projection refuses an external reference at {path}"
                    )
                result[key] = child
                continue
            result[key] = child
        if result.get("type") == "object" or "properties" in result:
            result["additionalProperties"] = False
        return result

    def _raise_projection(path: str, field: str) -> Any:
        raise BedrockProviderError(f"Bedrock schema projection found invalid {field} at {path}")

    projected = project(schema, path="$")
    validate_bedrock_json_schema(projected)
    return projected


def structured_output_config(
    schema: Mapping[str, Any] = BEDROCK_FINAL_SCHEMA,
) -> dict[str, Any]:
    validate_bedrock_json_schema(schema)
    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": _canonical_json(schema).decode("utf-8"),
                    "name": "flavourbench_answer",
                    "description": "A blinded culinary benchmark answer",
                }
            },
        }
    }


def _usage(response: Mapping[str, Any]) -> BedrockUsage:
    value = response.get("usage")
    if not isinstance(value, Mapping):
        raise BedrockProviderError("Bedrock returned no usage object")
    usage = value

    def count(name: str, *, required: bool) -> int:
        if required and name not in usage:
            raise BedrockProviderError(f"Bedrock returned no usage.{name}")
        raw = usage.get(name, 0)
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise BedrockProviderError(f"Bedrock returned invalid usage.{name}")
        return raw

    return BedrockUsage(
        input_tokens=count("inputTokens", required=True),
        output_tokens=count("outputTokens", required=True),
        total_tokens=count("totalTokens", required=True),
        cache_read_input_tokens=count("cacheReadInputTokens", required=False),
        cache_write_input_tokens=count("cacheWriteInputTokens", required=False),
    )


def _cost(usage: BedrockUsage, contract: BedrockEndpointContract) -> BedrockCostProvenance:
    price = contract.price
    complete = not (
        usage.cache_read_input_tokens and price.cache_read_per_million_usd is None
    ) and not (usage.cache_write_input_tokens and price.cache_write_per_million_usd is None)
    estimated: int | None = None
    if complete:
        # A price quoted per million tokens numerically equals micro-USD per token.
        micros = Decimal(usage.input_tokens) * Decimal(price.input_per_million_usd)
        micros += Decimal(usage.output_tokens) * Decimal(price.output_per_million_usd)
        if usage.cache_read_input_tokens:
            micros += Decimal(usage.cache_read_input_tokens) * Decimal(
                price.cache_read_per_million_usd or "0"
            )
        if usage.cache_write_input_tokens:
            micros += Decimal(usage.cache_write_input_tokens) * Decimal(
                price.cache_write_per_million_usd or "0"
            )
        estimated = int(micros.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return BedrockCostProvenance(
        estimated_cost_micros=estimated,
        estimate_complete=complete,
        pricing_sha256=price.sha256,
        pricing_source_uri=price.source_uri,
        pricing_observed_at=price.observed_at,
    )


def _safe_request_metadata(values: Mapping[str, str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise BedrockProviderError("Bedrock request metadata must contain only strings")
        lowered = key.lower()
        if any(marker in lowered for marker in ("token", "secret", "password", "credential")):
            raise BedrockProviderError("credential-like Bedrock request metadata is forbidden")
        metadata[key] = value
    return metadata


def _response_metadata(response: Mapping[str, Any]) -> tuple[str, int]:
    raw = response.get("ResponseMetadata")
    metadata = raw if isinstance(raw, Mapping) else {}
    status = metadata.get("HTTPStatusCode", 200)
    if status != 200:
        raise BedrockProviderError(f"Bedrock returned HTTP status {status}")
    request_id = str(metadata.get("RequestId") or "")
    metrics_raw = response.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, Mapping) else {}
    latency = metrics.get("latencyMs", 0)
    if not isinstance(latency, int) or isinstance(latency, bool) or latency < 0:
        raise BedrockProviderError("Bedrock returned an invalid metrics.latencyMs")
    return request_id, latency


def _output_message(response: Mapping[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("message"), Mapping):
        raise BedrockProviderError("Bedrock returned no output message")
    message = dict(output["message"])
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        raise BedrockProviderError("Bedrock returned invalid message content")
    return message


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping) and "text" in block
    )


def _validated_answer(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise BedrockProviderError("Bedrock structured output was not valid JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("answer_markdown"), str):
        raise BedrockProviderError("Bedrock answer did not match the FlavourBench object schema")
    if not value["answer_markdown"]:
        raise BedrockProviderError("Bedrock returned an empty culinary answer")
    for key in ("ingredient_mentions", "constraints_addressed", "uncertainties"):
        if not isinstance(value.get(key), list) or not all(
            isinstance(item, str) for item in value[key]
        ):
            raise BedrockProviderError(f"Bedrock answer has an invalid {key}")
    if set(value) != {
        "answer_markdown",
        "ingredient_mentions",
        "constraints_addressed",
        "uncertainties",
    }:
        raise BedrockProviderError("Bedrock answer contains unexpected fields")
    return value


def _aws_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return ""
    details = response.get("Error")
    return str(details.get("Code") or "") if isinstance(details, Mapping) else ""


def _validated_returned_model_identity(
    value: str,
    contract: BedrockEndpointContract,
) -> tuple[str, str]:
    """Validate a provider-returned identity and redact account-scoped ARNs."""

    digest = hashlib.sha256(value.encode()).hexdigest()
    if value in {
        contract.bedrock_target_id,
        *contract.expected_foundation_model_ids,
    }:
        return value, digest
    arn_by_digest = {
        contract.bedrock_target_arn.original_sha256: contract.bedrock_target_arn.redacted,
        **{item.original_sha256: item.redacted for item in contract.destination_model_arns},
    }
    if digest in arn_by_digest:
        return arn_by_digest[digest], digest
    raise BedrockProviderError(
        "Bedrock returned a model identity outside the frozen endpoint contract"
    )


class BedrockConverseProvider:
    """Run one fixed Bedrock endpoint through a bounded Converse tool loop."""

    _fallback_safe_rejections = frozenset(
        {
            "AccessDeniedException",
            "ResourceNotFoundException",
            "ValidationException",
        }
    )

    def __init__(
        self,
        runtime: BedrockRuntimeClient,
        contract: BedrockEndpointContract,
        *,
        tool_executor: BedrockToolExecutor | None = None,
        max_tool_rounds: int = 8,
        max_tool_calls_per_round: int = 4,
        max_tool_calls_total: int = 16,
    ) -> None:
        if not contract.supports_converse or not contract.supports_structured_output:
            raise BedrockProviderError("endpoint lacks frozen Converse/structured-output evidence")
        if max_tool_rounds < 1 or max_tool_calls_per_round < 1 or max_tool_calls_total < 1:
            raise BedrockProviderError("Bedrock tool bounds must be positive")
        self.runtime = runtime
        self.contract = contract
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls_per_round = max_tool_calls_per_round
        self.max_tool_calls_total = max_tool_calls_total

    async def _converse(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = await asyncio.to_thread(self.runtime.converse, **dict(payload))
        except Exception as error:
            code = _aws_error_code(error)
            if code in self._fallback_safe_rejections:
                raise BedrockRouteUnavailable(
                    f"Bedrock rejected the route before generation: {code}"
                ) from error
            raise BedrockProviderError(
                "Bedrock Converse failed without safe fallback evidence: "
                f"{code or type(error).__name__}"
            ) from error
        if not isinstance(response, Mapping):
            raise BedrockProviderError("Bedrock returned a non-object response")
        return response

    async def generate(self, spec: BedrockGenerationSpec) -> BedrockGenerationResult:
        if spec.canonical_model_id != self.contract.canonical_model_id:
            raise BedrockProviderError("generation and Bedrock contract canonical IDs differ")
        if spec.tools and not self.contract.supports_tool_use:
            raise BedrockProviderError("endpoint lacks frozen Bedrock tool-use evidence")
        if spec.tools and self.tool_executor is None:
            raise BedrockProviderError("Bedrock tool definitions require a client-side executor")
        if (
            self.contract.temperature_top_p_mutually_exclusive
            and spec.inference.temperature is not None
            and spec.inference.top_p is not None
        ):
            raise BedrockProviderError(
                "frozen Bedrock endpoint permits temperature or top_p, not both"
            )

        start = time.monotonic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": spec.prompt}]}]
        tool_payload = [tool.as_converse_tool() for tool in spec.tools]
        response_schema_sha256 = _sha256(BEDROCK_FINAL_SCHEMA)
        tool_schema_sha256 = _sha256(tool_payload)
        request_metadata = {
            **_safe_request_metadata(spec.request_metadata),
            "flavourbench_arm_sha256": hashlib.sha256(spec.arm_id.encode()).hexdigest(),
            "flavourbench_schema_sha256": response_schema_sha256,
            "flavourbench_tools_sha256": tool_schema_sha256,
        }
        common: dict[str, Any] = {
            "modelId": self.contract.bedrock_target_id,
            "system": [{"text": spec.system_prompt}],
            "inferenceConfig": spec.inference.as_converse_config(),
            "outputConfig": structured_output_config(),
            "requestMetadata": request_metadata,
        }
        if spec.tools:
            common["toolConfig"] = {
                "tools": tool_payload,
                "toolChoice": {"auto": {}},
            }

        usage = BedrockUsage()
        request_ids: list[str] = []
        returned_model_ids: list[str] = []
        returned_model_id_sha256s: list[str] = []
        service_latency = 0
        response_latencies: list[int] = []
        traces: list[BedrockToolTrace] = []
        total_tool_calls = 0
        final_message: Mapping[str, Any] | None = None
        finish_reason = "unknown"

        for round_index in range(self.max_tool_rounds + 1):
            response = await self._converse({**common, "messages": messages})
            returned = response.get("modelId") or response.get("model")
            if isinstance(returned, str) and returned:
                safe_returned, returned_sha256 = _validated_returned_model_identity(
                    returned,
                    self.contract,
                )
                if safe_returned not in returned_model_ids:
                    returned_model_ids.append(safe_returned)
                    returned_model_id_sha256s.append(returned_sha256)
            request_id, latency = _response_metadata(response)
            if request_id:
                request_ids.append(request_id)
            service_latency += latency
            response_latencies.append(latency)
            usage = usage.plus(_usage(response))
            message = _output_message(response)
            messages.append(message)
            finish_reason = str(response.get("stopReason") or "unknown")
            if finish_reason != "tool_use":
                final_message = message
                break
            if round_index >= self.max_tool_rounds:
                raise BedrockProviderError("Bedrock exhausted the frozen tool-round cap")

            content = message.get("content")
            tool_uses = [
                block["toolUse"]
                for block in content
                if isinstance(block, Mapping) and isinstance(block.get("toolUse"), Mapping)
            ]
            if not tool_uses:
                raise BedrockProviderError("Bedrock reported tool_use without a tool request")
            if len(tool_uses) > self.max_tool_calls_per_round:
                raise BedrockProviderError("Bedrock tool fan-out exceeded the per-round cap")
            total_tool_calls += len(tool_uses)
            if total_tool_calls > self.max_tool_calls_total:
                raise BedrockProviderError("Bedrock tool calls exceeded the generation cap")

            result_blocks: list[dict[str, Any]] = []
            for call_index, tool_use in enumerate(tool_uses):
                tool_use_id = str(tool_use.get("toolUseId") or "")
                name = str(tool_use.get("name") or "")
                arguments = tool_use.get("input")
                if not tool_use_id or not name or not isinstance(arguments, Mapping):
                    raise BedrockProviderError("Bedrock returned an invalid tool-use block")
                assert self.tool_executor is not None
                set_context = getattr(self.tool_executor, "set_context", None)
                if callable(set_context):
                    set_context(round_index, call_index, tool_use_id)
                execution = await self.tool_executor.execute(name, arguments)
                if isinstance(execution.content, str):
                    result_content = [{"text": execution.content}]
                else:
                    result_content = [{"json": execution.content}]
                result = {
                    "toolUseId": tool_use_id,
                    "content": result_content,
                    **({"status": "error"} if execution.is_error else {}),
                }
                result_blocks.append({"toolResult": result})
                traces.append(
                    BedrockToolTrace(
                        round_index=round_index,
                        tool_use_id=tool_use_id,
                        name=name,
                        arguments=dict(arguments),
                        is_error=execution.is_error,
                        result_sha256=_sha256(execution.content),
                    )
                )
            messages.append({"role": "user", "content": result_blocks})

        if final_message is None:
            raise BedrockProviderError("Bedrock returned no final response")
        if finish_reason not in {"end_turn", "stop_sequence"}:
            raise BedrockProviderError(f"Bedrock final stop reason was {finish_reason}")
        answer = _validated_answer(_text(final_message))
        identity = BedrockIdentityProvenance(
            canonical_model_id=self.contract.canonical_model_id,
            requested_model_or_profile_id=self.contract.bedrock_target_id,
            requested_model_or_profile_arn_redacted=(self.contract.bedrock_target_arn.redacted),
            requested_model_or_profile_arn_sha256=(
                self.contract.bedrock_target_arn.original_sha256
            ),
            expected_foundation_model_ids=self.contract.expected_foundation_model_ids,
            frozen_destination_model_arns=tuple(
                item.redacted for item in self.contract.destination_model_arns
            ),
            frozen_destination_model_arn_sha256s=tuple(
                item.original_sha256 for item in self.contract.destination_model_arns
            ),
            returned_model_ids=tuple(returned_model_ids),
            returned_model_id_sha256s=tuple(returned_model_id_sha256s),
            provider="amazon-bedrock",
            ingress_region=self.contract.region,
            profile_scope=self.contract.profile_scope,
            profile_scope_sha256=self.contract.profile_scope_sha256,
            endpoint_kind=self.contract.endpoint_kind,
            # Converse identifies the requested target but does not attest which
            # destination Region/profile member served a cross-Region request.
            actual_execution_region=None,
            actual_foundation_model_id=None,
            identity_evidence="frozen_catalog_target_plus_aws_request_ids",
        )
        return BedrockGenerationResult(
            answer_markdown=answer["answer_markdown"],
            output_json=answer,
            finish_reason=finish_reason,
            usage=usage,
            wall_clock_latency_ms=round((time.monotonic() - start) * 1000),
            service_latency_ms=service_latency,
            response_latencies_ms=tuple(response_latencies),
            request_ids=tuple(request_ids),
            retries=0,
            response_schema_sha256=response_schema_sha256,
            tool_schema_sha256=tool_schema_sha256,
            tool_traces=tuple(traces),
            identity=identity,
            cost=_cost(usage, self.contract),
            rank_eligible=self.contract.season_eligible,
        )
