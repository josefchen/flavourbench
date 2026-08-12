"""Governed Cohere Chat V2 adapter for FlavourBench workers.

The benchmark's execution loop uses an OpenAI-shaped internal protocol. This
module translates that protocol to Cohere Chat V2 while preserving the exact
model request, bounded tool loop, response IDs, returned usage, and complete
Epicure traces. No provider fallback is available on this route.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import re
import time
from collections.abc import Mapping
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import httpx

from .budget_policy import provider_account_scope_sha256
from .cohere_compatibility import project_cohere_strict_schema
from .config import get_settings
from .execution_policy import PORTABLE_TEXT_TOOL_PROTOCOL_V1, SELECTION_TEXT_PROTOCOL_V1
from .provider import (
    PORTABLE_EPICURE_TOOL_NAMES,
    AttemptIdFactory,
    AttemptSink,
    GenerationFailureResult,
    GenerationResult,
    GenerationSpec,
    OpenRouterProvider,
    ProviderAttemptEvent,
    ProviderError,
    ToolSink,
    UncertainDeliveryError,
    _canonical_sha256,
    _safe_retry_delay_seconds,
)

COHERE_DIRECT_CONTRACT_SCHEMA = "flavourbench-cohere-direct-endpoint-contract-v1"
COHERE_DIRECT_PROVIDER_SLUG = "cohere-direct"
COHERE_ACCOUNTING_BASIS = "frozen_rate_card_times_cohere_returned_usage"
COHERE_REQUIRED_TOOL_INSTRUCTION = (
    "Call at least one of the supplied tools now. Do not return a text-only response in this turn."
)
COHERE_REASONING_PORTABLE_SELECTION_MODE = "thinking_disabled_for_exact_json_selection"
COHERE_REASONING_PORTABLE_FINAL_MODE = "thinking_disabled_for_exact_choice"
COHERE_REASONING_MODEL = "command-a-reasoning-08-2025"
COHERE_PLUS_MODEL = "command-a-plus-05-2026"
COHERE_PLUS_SELECTION_FORMAT = "cohere_json_schema_tool_selection_v1"
COHERE_PLUS_FINAL_FORMAT = "cohere_json_schema_choice_v1"
COHERE_PLUS_PHASE_REASONING = "thinking_enabled_512_for_schema"
COHERE_SELECTION_VALUES = tuple(
    ",".join(labels) for labels in itertools.combinations("ABCDEFGH", 3)
)


def _safe_http_error_detail(response: httpx.Response) -> str:
    """Retain a bounded provider validation message without request headers or bodies."""

    detail = ""
    try:
        value = response.json()
    except (json.JSONDecodeError, ValueError):
        value = None
    if isinstance(value, Mapping):
        candidate = value.get("message")
        error = value.get("error")
        if not candidate and isinstance(error, Mapping):
            candidate = error.get("message") or error.get("type")
        if candidate:
            detail = str(candidate)
    detail = re.sub(r"\s+", " ", detail).strip()
    return detail[:500] or "provider supplied no validation detail"


def _micros(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise ProviderError("Cohere rate-card cost must be finite and non-negative")
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ProviderError(f"Cohere rate-card field is invalid: {field}") from error
    if not parsed.is_finite() or parsed < 0:
        raise ProviderError(f"Cohere rate-card field is invalid: {field}")
    return parsed


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}
    )


def _cohere_messages(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ProviderError("Cohere messages must be a list")
    projected: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            raise ProviderError("Cohere message entry must be an object")
        role = str(raw.get("role") or "")
        if role in {"system", "user"}:
            projected.append({"role": role, "content": _text_content(raw.get("content"))})
            continue
        if role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant"}
            # Cohere's V2 contract requires callers to append the provider's
            # assistant message to the conversation before returning tool
            # results.  Keep provider-native content blocks opaque here: the
            # benchmark-visible projection remains `_text_content`, while the
            # transport replay retains thinking blocks and any future content
            # block metadata verbatim.
            reasoning_details = raw.get("reasoning_details")
            cohere_transport = (
                reasoning_details
                if isinstance(reasoning_details, Mapping)
                and reasoning_details.get("type") == "cohere_v2_opaque_continuation"
                else {}
            )
            opaque_content = raw.get("_cohere_content", cohere_transport.get("content"))
            if opaque_content is not None:
                if not isinstance(opaque_content, (str, list)):
                    raise ProviderError("Cohere opaque assistant content is malformed")
                assistant["content"] = opaque_content
            else:
                visible_content = _text_content(raw.get("content"))
                if visible_content:
                    assistant["content"] = visible_content
            opaque_tool_plan = raw.get("_cohere_tool_plan", cohere_transport.get("tool_plan"))
            if opaque_tool_plan is not None:
                if not isinstance(opaque_tool_plan, str):
                    raise ProviderError("Cohere opaque assistant tool plan is malformed")
                assistant["tool_plan"] = opaque_tool_plan
            raw_calls = raw.get("tool_calls")
            if isinstance(raw_calls, list) and raw_calls:
                calls: list[dict[str, Any]] = []
                for raw_call in raw_calls:
                    if not isinstance(raw_call, Mapping):
                        raise ProviderError("Cohere assistant tool call is malformed")
                    function = raw_call.get("function")
                    if not isinstance(function, Mapping):
                        raise ProviderError("Cohere assistant tool function is malformed")
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments or {}, sort_keys=True)
                    calls.append(
                        {
                            "id": str(raw_call.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(function.get("name") or ""),
                                "arguments": arguments,
                            },
                        }
                    )
                assistant["tool_calls"] = calls
            if len(assistant) == 1:
                assistant["content"] = ""
            projected.append(assistant)
            continue
        if role == "tool":
            result = _text_content(raw.get("content"))
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": str(raw.get("tool_call_id") or ""),
                    "content": [
                        {
                            "type": "document",
                            "document": {
                                "data": json.dumps(
                                    {
                                        "tool": str(raw.get("name") or "epicure"),
                                        "result": result,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                            },
                        }
                    ],
                }
            )
            continue
        raise ProviderError(f"Cohere message role is unsupported: {role}")
    return projected


def _cohere_usage(value: Mapping[str, Any]) -> dict[str, int]:
    usage = value.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    tokens = usage.get("tokens")
    tokens = tokens if isinstance(tokens, Mapping) else usage
    billed = usage.get("billed_units")
    billed = billed if isinstance(billed, Mapping) else {}
    return {
        "prompt_tokens": int(tokens.get("input_tokens") or 0),
        "completion_tokens": int(tokens.get("output_tokens") or 0),
        "reasoning_tokens": int(tokens.get("reasoning_tokens") or 0),
        "billed_input_tokens": int(billed.get("input_tokens") or 0),
        "billed_output_tokens": int(billed.get("output_tokens") or 0),
    }


def _finish_reason(value: object) -> str:
    return {
        "COMPLETE": "stop",
        "STOP_SEQUENCE": "stop",
        "TOOL_CALL": "tool_calls",
        "MAX_TOKENS": "length",
    }.get(str(value or "").upper(), str(value or "unknown").lower())


def _thinking(effort: str, max_tokens: int) -> dict[str, Any]:
    if effort in {"none", "minimal"}:
        return {"type": "disabled"}
    budgets = {"low": 512, "medium": 1024, "high": 2048}
    if effort == "max":
        budget = max(1, max_tokens - 1024)
    else:
        budget = budgets.get(effort, 512)
    return {"type": "enabled", "token_budget": min(budget, max(1, max_tokens - 1))}


def _request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "stream": False,
        "messages": _cohere_messages(payload.get("messages")),
    }
    for name in ("max_tokens", "temperature", "seed"):
        if name in payload:
            projected[name] = payload[name]
    if "top_p" in payload:
        projected["p"] = payload["top_p"]
    reasoning = payload.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, Mapping):
            raise ProviderError("Cohere reasoning control must be an object")
        projected["thinking"] = _thinking(
            str(reasoning.get("effort") or "low"),
            int(payload.get("max_tokens") or 4096),
        )
    raw_tools = payload.get("tools")
    if raw_tools is not None:
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ProviderError("Cohere tools must be a non-empty list")
        projected["tools"] = project_cohere_strict_schema(raw_tools)
        # The full Epicure catalog includes parameter-free discovery tools,
        # which Cohere excludes from strict-tools mode. The schemas remain
        # bounded and the benchmark validates every returned call client-side.
        all_have_required = all(
            bool(((item.get("function") or {}).get("parameters") or {}).get("required"))
            for item in projected["tools"]
            if isinstance(item, Mapping)
        )
        projected["strict_tools"] = all_have_required
        # Command A+ currently rejects the native tool_choice field. The
        # matched protocol supplies an explicit evidence-decision instruction,
        # and the shared execution loop rejects the arm unless it records at
        # least one successful real Epicure trace. This emulates the benchmark's
        # required-tool invariant without claiming native provider enforcement.
        if payload.get("tool_choice") == "required":
            messages = projected["messages"]
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {
                    **messages[-1],
                    "content": (
                        str(messages[-1].get("content") or "").rstrip()
                        + "\n\n"
                        + COHERE_REQUIRED_TOOL_INSTRUCTION
                    ),
                }
            else:
                messages.append({"role": "user", "content": COHERE_REQUIRED_TOOL_INSTRUCTION})
    response_format = payload.get("response_format")
    if response_format is not None:
        if raw_tools is not None or not isinstance(response_format, Mapping):
            raise ProviderError("Cohere response format cannot be combined with tools")
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, Mapping) else None
        projected["response_format"] = {
            "type": "json_object",
            "schema": project_cohere_strict_schema(schema or {}),
        }
    return projected


def _json_schema_payload(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": dict(schema),
        },
    }


def _cohere_plus_phase_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        phase in {"portable_tool_selection", "final"}
        and contract.get("portable_phase_reasoning") != COHERE_PLUS_PHASE_REASONING
    ):
        raise ProviderError("Cohere A Plus phase reasoning is absent from its contract")
    if phase == "portable_tool_selection":
        if contract.get("portable_tool_selection_format") != COHERE_PLUS_SELECTION_FORMAT:
            raise ProviderError("Cohere A Plus selection format is absent from its contract")
        return {
            **payload,
            "response_format": _json_schema_payload(
                "epicure_tool_selection",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": sorted(PORTABLE_EPICURE_TOOL_NAMES),
                        },
                        "arguments_json": {"type": "string"},
                    },
                    "required": ["name", "arguments_json"],
                    "additionalProperties": False,
                },
            ),
            "reasoning": {"effort": "low", "exclude": True},
        }
    if phase == "final":
        if contract.get("portable_final_format") != COHERE_PLUS_FINAL_FORMAT:
            raise ProviderError("Cohere A Plus final format is absent from its contract")
        return {
            **payload,
            "response_format": _json_schema_payload(
                "flavourbench_choice",
                {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": ["A", "B", "C", "D"]}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            ),
            "reasoning": {"effort": "low", "exclude": True},
        }
    return dict(payload)


def _normalize_cohere_plus_phase(value: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if phase not in {"portable_tool_selection", "final"}:
        return value
    choices = value.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    if not isinstance(message, dict):
        raise ProviderError("Cohere A Plus response omitted its assistant message")
    try:
        parsed = json.loads(str(message.get("content") or ""))
    except json.JSONDecodeError as error:
        raise ProviderError("Cohere A Plus structured response is not JSON") from error
    if not isinstance(parsed, dict):
        raise ProviderError("Cohere A Plus structured response is not an object")
    if phase == "portable_tool_selection":
        if set(parsed) != {"name", "arguments_json"} or not isinstance(
            parsed["arguments_json"], str
        ):
            raise ProviderError("Cohere A Plus tool selection has an invalid shape")
        try:
            arguments = json.loads(parsed["arguments_json"])
        except json.JSONDecodeError as error:
            raise ProviderError("Cohere A Plus tool arguments are not JSON") from error
        if not isinstance(arguments, dict):
            raise ProviderError("Cohere A Plus tool arguments are not an object")
        message["content"] = json.dumps(
            {"name": parsed["name"], "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        if set(parsed) != {"choice"} or parsed["choice"] not in {"A", "B", "C", "D"}:
            raise ProviderError("Cohere A Plus final choice has an invalid shape")
        message["content"] = f"FINAL_CHOICE: {parsed['choice']}"
    return value


def _cohere_plus_selection_payload(
    payload: dict[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if contract.get("portable_phase_reasoning") != COHERE_PLUS_PHASE_REASONING:
        raise ProviderError("Cohere A Plus bounded reasoning is absent from its contract")
    return {
        **payload,
        "response_format": _json_schema_payload(
            "flavourbench_selection",
            {
                "type": "object",
                "properties": {
                    "selection": {
                        "type": "string",
                        "enum": list(COHERE_SELECTION_VALUES),
                    }
                },
                "required": ["selection"],
                "additionalProperties": False,
            },
        ),
        "reasoning": {"effort": "low", "exclude": True},
    }


def _normalize_cohere_plus_selection(value: dict[str, Any]) -> dict[str, Any]:
    choices = value.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    if not isinstance(message, dict):
        raise ProviderError("Cohere A Plus selection response omitted its assistant message")
    try:
        parsed = json.loads(str(message.get("content") or ""))
    except json.JSONDecodeError as error:
        raise ProviderError("Cohere A Plus selection response is not JSON") from error
    selection = parsed.get("selection") if isinstance(parsed, dict) else None
    labels = str(selection or "").split(",")
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"selection"}
        or len(labels) != 3
        or len(set(labels)) != 3
        or labels != sorted(labels)
        or any(label not in set("ABCDEFGH") for label in labels)
    ):
        raise ProviderError("Cohere A Plus selection response has an invalid shape")
    message["content"] = f"FINAL_SELECTION: {selection}"
    return value


def _openai_response(value: Mapping[str, Any], *, response_model: str) -> dict[str, Any]:
    message = value.get("message")
    if not isinstance(message, Mapping):
        raise ProviderError("Cohere response has no assistant message")
    content = message.get("content")
    text = _text_content(content)
    raw_calls = message.get("tool_calls")
    tool_calls: list[dict[str, Any]] = []
    if isinstance(raw_calls, list):
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise ProviderError("Cohere returned a malformed tool call")
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                raise ProviderError("Cohere returned a malformed tool function")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, sort_keys=True)
            tool_calls.append(
                {
                    "id": str(raw_call.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": arguments,
                    },
                }
            )
    normalized_message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
        # Internal transport-only fields.  They are deliberately separate
        # from `content`, which is the publishable text projection.
        "_cohere_content": content,
        # The shared provider loop already preserves `reasoning_details` across
        # assistant continuations.  Use that established opaque transport so
        # Cohere continuity does not require changing the historically frozen
        # provider source used to reconstruct earlier runs.
        "reasoning_details": {
            "type": "cohere_v2_opaque_continuation",
            "content": content,
        },
    }
    tool_plan = message.get("tool_plan")
    if tool_plan is not None:
        if not isinstance(tool_plan, str):
            raise ProviderError("Cohere returned a malformed tool plan")
        normalized_message["_cohere_tool_plan"] = tool_plan
        normalized_message["reasoning_details"]["tool_plan"] = tool_plan
    if tool_calls:
        normalized_message["tool_calls"] = tool_calls
    usage = _cohere_usage(value)
    return {
        "id": str(value.get("id") or ""),
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": _finish_reason(value.get("finish_reason")),
                "message": normalized_message,
            }
        ],
        "usage": usage,
        "_cohere_usage": usage,
    }


class CohereDirectProvider(OpenRouterProvider):
    """Execute one frozen Cohere model through Chat V2."""

    def __init__(
        self,
        attempt_sink: AttemptSink | None = None,
        tool_sink: ToolSink | None = None,
        attempt_id_factory: AttemptIdFactory | None = None,
    ) -> None:
        self.settings = get_settings()
        self.attempt_sink = attempt_sink
        self.tool_sink = tool_sink
        self.attempt_id_factory = attempt_id_factory
        self._issued_attempt_ids: set[str] = set()
        self._attempt_by_generation: dict[str, ProviderAttemptEvent] = {}
        self._backend_tool_schema_by_arm: dict[str, str] = {}
        self._spec_by_arm: dict[str, GenerationSpec] = {}
        self._accounting_by_generation: dict[str, dict[str, Any]] = {}
        self.client = httpx.AsyncClient(
            base_url=self.settings.cohere_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {self.settings.cohere_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Client-Name": "Epicure-FlavourBench",
            },
            timeout=self.settings.cohere_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    def _provider_preferences(
        self,
        provider_slug: str,
        rate_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {}

    def _validate_contract(self, spec: GenerationSpec) -> str:
        contract = spec.backend_contract_json
        requested_model = str(contract.get("requested_model_id") or "")
        if (
            contract.get("schema_version") != COHERE_DIRECT_CONTRACT_SCHEMA
            or not requested_model
            or spec.expected_actual_provider_slug != COHERE_DIRECT_PROVIDER_SLUG
            or str(contract.get("base_url") or "").rstrip("/")
            != self.settings.cohere_base_url.rstrip("/")
            or str(contract.get("catalog_sha256") or "") in {"", "unresolved", "unfrozen"}
            or str(contract.get("catalog_entry_sha256") or "") in {"", "unresolved", "unfrozen"}
        ):
            raise ProviderError("direct Cohere endpoint differs from its frozen contract")
        if self.settings.execution_mode == "live":
            if not self.settings.cohere_api_key:
                raise ProviderError("direct Cohere execution requires a configured API key")
            if (
                spec.execution_backend != "cohere_direct"
                or spec.provider_credential_scope_sha256
                != provider_account_scope_sha256("cohere_direct")
            ):
                raise ProviderError("direct Cohere generation lacks its governed account binding")
        return requested_model

    def _rate_card_accounting(
        self,
        *,
        generation_id: str,
        arm_id: str,
        response_model: str,
        usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        spec = self._spec_by_arm.get(arm_id)
        if spec is None:
            raise ProviderError("direct Cohere response has no active generation contract")
        prompt_tokens = int(usage.get("billed_input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage.get("billed_output_tokens") or usage.get("completion_tokens") or 0
        )
        reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
        prices = spec.rate_card_json
        amount = (
            _decimal(prices.get("request_price", "0"), field="request_price")
            + Decimal(prompt_tokens)
            * _decimal(
                prices.get("prompt_price_per_token", "0"),
                field="prompt_price_per_token",
            )
            + Decimal(completion_tokens)
            * _decimal(
                prices.get("completion_price_per_token", "0"),
                field="completion_price_per_token",
            )
            + Decimal(reasoning_tokens)
            * _decimal(
                prices.get("internal_reasoning_price_per_token", "0"),
                field="internal_reasoning_price_per_token",
            )
        )
        return {
            "generation_id": generation_id,
            "cost_micros": _micros(amount),
            "provider": COHERE_DIRECT_PROVIDER_SLUG,
            "model": response_model,
            "reconciled": False,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "accounting_basis": COHERE_ACCOUNTING_BASIS,
            "billing_reconciliation_status": "provider_charge_unavailable",
        }

    async def _post(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
        *,
        arm_id: str = "",
        phase: str = "unknown",
        governance_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = self._spec_by_arm.get(arm_id)
        selection_text = bool(
            spec is not None and spec.evidence_protocol == SELECTION_TEXT_PROTOCOL_V1
        )
        reasoning_override = {
            "portable_tool_selection": (
                "portable_tool_selection_reasoning",
                COHERE_REASONING_PORTABLE_SELECTION_MODE,
            ),
            "final": (
                "portable_final_reasoning",
                COHERE_REASONING_PORTABLE_FINAL_MODE,
            ),
        }.get(phase)
        if (
            reasoning_override is not None
            and str(payload.get("model") or "") == COHERE_REASONING_MODEL
            and spec is not None
            and (
                spec.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
                or (selection_text and phase == "final")
            )
        ):
            contract_field, expected_mode = reasoning_override
            if spec is None or spec.backend_contract_json.get(contract_field) != expected_mode:
                raise ProviderError(
                    "Cohere portable phase override is absent from its frozen contract"
                )
            payload = {
                **payload,
                "reasoning": {"effort": "none", "exclude": True},
            }
        if (
            str(payload.get("model") or "") == COHERE_PLUS_MODEL
            and spec is not None
            and spec.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
        ):
            contract = spec.backend_contract_json if spec is not None else {}
            payload = _cohere_plus_phase_payload(payload, phase=phase, contract=contract)
        elif (
            str(payload.get("model") or "") == COHERE_PLUS_MODEL
            and selection_text
            and phase == "final"
        ):
            payload = _cohere_plus_selection_payload(
                payload,
                contract=spec.backend_contract_json,
            )
        request_payload = _request_payload(payload)
        last_error: Exception | None = None
        for attempt in range(self.settings.max_provider_attempts):
            attempt_event = ProviderAttemptEvent(
                attempt_id=self._new_attempt_id(arm_id, phase, attempt),
                arm_id=arm_id,
                request_key_sha256=hashlib.sha256(idempotency_key.encode()).hexdigest(),
                phase=f"cohere_direct_{phase}",
                attempt_index=attempt,
                event_type="request_started",
                payload_sha256=_canonical_sha256(request_payload),
                metadata={"execution_backend": "cohere_direct", **dict(governance_metadata or {})},
            )
            self._emit_attempt(attempt_event)
            try:
                response = await self.client.post("v2/chat", json=request_payload)
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw, Mapping):
                    raise ProviderError("Cohere returned a non-object response")
                generation_id = str(raw.get("id") or "")
                spec = self._spec_by_arm.get(arm_id)
                if not generation_id or spec is None:
                    raise ProviderError("Cohere response omitted its generation identity")
                value = _openai_response(raw, response_model=spec.expected_actual_model_id)
                if (
                    spec.expected_actual_model_id == COHERE_PLUS_MODEL
                    and spec.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
                ):
                    value = _normalize_cohere_plus_phase(value, phase=phase)
                elif (
                    spec.expected_actual_model_id == COHERE_PLUS_MODEL
                    and spec.evidence_protocol == SELECTION_TEXT_PROTOCOL_V1
                    and phase == "final"
                ):
                    value = _normalize_cohere_plus_selection(value)
                usage = value.get("usage")
                usage = usage if isinstance(usage, Mapping) else {}
                accounting = self._rate_card_accounting(
                    generation_id=generation_id,
                    arm_id=arm_id,
                    response_model=spec.expected_actual_model_id,
                    usage=usage,
                )
                value["_flavourbench_retries"] = attempt
                completed = ProviderAttemptEvent(
                    **{
                        **attempt_event.__dict__,
                        "event_type": "response_received",
                        "generation_id": generation_id,
                        "http_status": response.status_code,
                        "payload_sha256": _canonical_sha256(value),
                        "metadata": {
                            **attempt_event.metadata,
                            "response_model": spec.expected_actual_model_id,
                            "finish_reason": str(
                                ((value.get("choices") or [{}])[0] or {}).get("finish_reason")
                                or "unknown"
                            ),
                            "identity_evidence": (
                                "authenticated_catalog_exact_request_and_generation_id"
                            ),
                        },
                    }
                )
                self._attempt_by_generation[generation_id] = completed
                self._accounting_by_generation[generation_id] = accounting
                self._emit_attempt(completed)
                return value
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                last_error = error
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "pre_send_failure",
                            "error_type": type(error).__name__,
                        }
                    )
                )
                if attempt + 1 < self.settings.max_provider_attempts:
                    delay = _safe_retry_delay_seconds(
                        idempotency_key=idempotency_key,
                        attempt=attempt,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            except httpx.ReadTimeout as error:
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "uncertain_delivery",
                            "error_type": type(error).__name__,
                        }
                    )
                )
                raise UncertainDeliveryError(
                    "Cohere response timed out after possible acceptance; reconcile before retry"
                ) from error
            except httpx.HTTPStatusError as error:
                last_error = error
                status = error.response.status_code
                provider_error_detail = _safe_http_error_detail(error.response)
                event_type = (
                    "uncertain_delivery" if status == 408 or status >= 500 else "request_rejected"
                )
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": event_type,
                            "http_status": status,
                            "error_type": type(error).__name__,
                            "metadata": {
                                **attempt_event.metadata,
                                "provider_error_detail": provider_error_detail,
                            },
                        }
                    )
                )
                if event_type == "uncertain_delivery":
                    raise UncertainDeliveryError(
                        "Cohere returned an ambiguous failure after possible dispatch"
                    ) from error
                if status == 429 and attempt + 1 < self.settings.max_provider_attempts:
                    delay = _safe_retry_delay_seconds(
                        idempotency_key=idempotency_key,
                        attempt=attempt,
                        retry_after=error.response.headers.get("Retry-After", ""),
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            except (TypeError, ValueError, ProviderError) as error:
                last_error = error
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "invalid_response",
                            "error_type": type(error).__name__,
                        }
                    )
                )
                break
        if isinstance(last_error, httpx.HTTPStatusError):
            raise ProviderError(
                "Cohere request rejected with HTTP "
                f"{last_error.response.status_code}: "
                f"{_safe_http_error_detail(last_error.response)}"
            ) from last_error
        raise ProviderError(f"Cohere request failed: {type(last_error).__name__}") from last_error

    async def _generation_cost(self, generation_id: str) -> dict[str, Any]:
        accounting = self._accounting_by_generation.get(generation_id)
        if accounting is None:
            return {
                "generation_id": generation_id,
                "cost_micros": 0,
                "provider": "unknown",
                "model": "unknown",
                "reconciled": False,
                "billing_reconciliation_status": "missing_direct_response_usage",
            }
        prior = self._attempt_by_generation.get(generation_id)
        if prior is not None:
            self._emit_attempt(
                ProviderAttemptEvent(
                    **{
                        **prior.__dict__,
                        "event_type": "accounting_reconciled",
                        "metadata": dict(accounting),
                    }
                )
            )
        return dict(accounting)

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        requested_model = self._validate_contract(spec)
        request_spec = replace(spec, model_id=requested_model)
        self._spec_by_arm[spec.arm_id] = request_spec
        started = time.monotonic()
        try:
            result = await super().generate(request_spec)
        finally:
            self._spec_by_arm.pop(spec.arm_id, None)
        result.latency_ms = round((time.monotonic() - started) * 1000)
        result.cost_accounting_basis = COHERE_ACCOUNTING_BASIS
        result.billing_reconciliation_status = "provider_charge_unavailable"
        return result

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        requested_model = self._validate_contract(spec)
        request_spec = replace(spec, model_id=requested_model)
        self._spec_by_arm[spec.arm_id] = request_spec
        try:
            result = await super().reconcile_failure(request_spec, error)
        finally:
            self._spec_by_arm.pop(spec.arm_id, None)
        if result is not None:
            result.cost_accounting_basis = COHERE_ACCOUNTING_BASIS
            result.billing_reconciliation_status = "provider_charge_unavailable"
        return result
