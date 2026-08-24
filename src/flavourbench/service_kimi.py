"""Governed direct-Kimi adapter for production FlavourBench workers.

The Kimi Code endpoint is OpenAI Chat Completions compatible, but it is not an
OpenRouter route. This adapter reuses the benchmark's bounded tool loop while
keeping credentials, identity evidence, request journaling, and cost
provenance in a separate backend. Provider fallback is intentionally absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import httpx

from .budget_policy import provider_account_scope_sha256
from .config import get_settings
from .provider import (
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

KIMI_DIRECT_CONTRACT_SCHEMA = "flavourbench-kimi-direct-anthropic-contract-v2"
KIMI_DIRECT_PROVIDER_SLUG = "kimi-code-direct"
KIMI_ANTHROPIC_VERSION = "2023-06-01"


def _safe_http_error_detail(response: httpx.Response) -> str:
    """Return a bounded provider validation message without credentials."""

    detail = ""
    try:
        value = response.json()
    except (json.JSONDecodeError, ValueError):
        value = None
    if isinstance(value, Mapping):
        error = value.get("error")
        candidate = value.get("message")
        if not candidate and isinstance(error, Mapping):
            candidate = error.get("message") or error.get("type")
        if candidate:
            detail = str(candidate)
    detail = re.sub(r"\s+", " ", detail).strip()
    return detail[:500] or "provider supplied no validation detail"


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


def _anthropic_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the benchmark's text-only OpenAI shape to Anthropic Messages."""

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ProviderError("Kimi messages must be a non-empty list")
    system_parts: list[str] = []
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise ProviderError("Kimi message entry must be an object")
        role = str(raw.get("role") or "")
        content = _text_content(raw.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            raise ProviderError(f"Kimi message role is unsupported: {role}")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ProviderError("Kimi request has no user or assistant messages")
    if payload.get("tools") is not None or payload.get("response_format") is not None:
        raise ProviderError("Kimi direct benchmark route requires the portable text protocol")
    if payload.get("reasoning") is not None:
        raise ProviderError("Kimi Anthropic reasoning overrides are not enabled for this run")
    projected: dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "max_tokens": int(payload.get("max_tokens") or 0),
        "messages": messages,
    }
    if system_parts:
        projected["system"] = "\n\n".join(system_parts)
    # Kimi's Anthropic-compatible endpoint does not expose deterministic seed.
    # top_p=1 is the provider default, so the matched run sends temperature=0
    # and otherwise leaves sampling at that default.
    if "temperature" in payload:
        projected["temperature"] = payload["temperature"]
    return projected


def _openai_response(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("type") != "message":
        raise ProviderError("Kimi returned a non-message response")
    generation_id = str(value.get("id") or "")
    response_model = str(value.get("model") or "")
    if not generation_id or not response_model:
        raise ProviderError("Kimi response omitted its generation identity")
    content = value.get("content")
    text = _text_content(content)
    usage = value.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    stop_reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(str(value.get("stop_reason") or ""), str(value.get("stop_reason") or "unknown"))
    return {
        "id": generation_id,
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": stop_reason,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_details": {
                        "type": "kimi_anthropic_opaque_continuation",
                        "content": content,
                    },
                },
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "reasoning_tokens": 0,
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        },
    }


def _micros(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise ProviderError("Kimi rate-card cost must be finite and non-negative")
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ProviderError(f"Kimi rate-card field is invalid: {field}") from error
    if not parsed.is_finite() or parsed < 0:
        raise ProviderError(f"Kimi rate-card field is invalid: {field}")
    return parsed


class KimiDirectProvider(OpenRouterProvider):
    """Execute one frozen Kimi model without passing through OpenRouter."""

    direct_provider_name = "Kimi"
    execution_backend = "kimi_direct"
    provider_slug = KIMI_DIRECT_PROVIDER_SLUG
    contract_schema = KIMI_DIRECT_CONTRACT_SCHEMA
    credential_setting = "kimi_api_key"
    credential_environment_name = "FLAVOURBENCH_KIMI_API_KEY"
    base_url_setting = "kimi_base_url"
    timeout_setting = "kimi_timeout_seconds"
    accounting_basis = "frozen_rate_card_times_kimi_returned_usage"

    def _authorization_headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key}

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
        api_key = str(getattr(self.settings, self.credential_setting, ""))
        base_url = str(getattr(self.settings, self.base_url_setting))
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                **self._authorization_headers(api_key),
                "anthropic-version": KIMI_ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Epicure-FlavourBench/0.1",
            },
            timeout=getattr(self.settings, self.timeout_setting),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    def _provider_preferences(
        self,
        provider_slug: str,
        rate_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # OpenRouter's provider-routing object is not part of Kimi's API.
        return {}

    def _validate_contract(self, spec: GenerationSpec) -> str:
        contract = spec.backend_contract_json
        requested_model = str(contract.get("requested_model_id") or "")
        base_url = str(contract.get("base_url") or "").rstrip("/")
        configured_base_url = str(getattr(self.settings, self.base_url_setting)).rstrip("/")
        if (
            contract.get("schema_version") != self.contract_schema
            or not requested_model
            or requested_model != spec.expected_actual_model_id
            or spec.expected_actual_provider_slug != self.provider_slug
            or base_url != configured_base_url
            or str(contract.get("catalog_sha256") or "") in {"", "unresolved", "unfrozen"}
            or str(contract.get("catalog_entry_sha256") or "") in {"", "unresolved", "unfrozen"}
        ):
            raise ProviderError(
                f"direct {self.direct_provider_name} endpoint differs from its frozen "
                "backend contract"
            )
        if self.settings.execution_mode == "live":
            if not getattr(self.settings, self.credential_setting, ""):
                raise ProviderError(
                    f"direct {self.direct_provider_name} execution requires "
                    f"{self.credential_environment_name}"
                )
            if (
                spec.execution_backend != self.execution_backend
                or spec.provider_credential_scope_sha256
                != provider_account_scope_sha256(self.execution_backend)
            ):
                raise ProviderError(
                    f"direct {self.direct_provider_name} generation lacks its governed "
                    "account binding"
                )
        return requested_model

    def _rate_card_accounting(
        self,
        *,
        generation_id: str,
        arm_id: str,
        response_model: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        spec = self._spec_by_arm.get(arm_id)
        if spec is None:
            raise ProviderError(
                f"direct {self.direct_provider_name} response has no active frozen "
                "generation contract"
            )
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        completion_details = usage.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, dict) else {}
        reasoning_tokens = int(
            usage.get("reasoning_tokens") or completion_details.get("reasoning_tokens") or 0
        )
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
            "provider": self.provider_slug,
            "model": response_model,
            # The managed endpoint returns token usage but no charged amount.
            # This is a rate-card estimate, not provider billing reconciliation.
            "reconciled": False,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "accounting_basis": self.accounting_basis,
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
        request_payload = _anthropic_request_payload(payload)
        last_error: Exception | None = None
        for attempt in range(self.settings.max_provider_attempts):
            attempt_event = ProviderAttemptEvent(
                attempt_id=self._new_attempt_id(arm_id, phase, attempt),
                arm_id=arm_id,
                request_key_sha256=hashlib.sha256(idempotency_key.encode()).hexdigest(),
                phase=f"{self.execution_backend}_{phase}",
                attempt_index=attempt,
                event_type="request_started",
                payload_sha256=_canonical_sha256(request_payload),
                metadata={
                    "execution_backend": self.execution_backend,
                    **dict(governance_metadata or {}),
                },
            )
            self._emit_attempt(attempt_event)
            try:
                response = await self.client.post("v1/messages", json=request_payload)
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw, Mapping):
                    raise ProviderError(
                        f"{self.direct_provider_name} returned a non-object response"
                    )
                value = _openai_response(raw)
                generation_id = str(value.get("id") or "")
                response_model = str(value.get("model") or "")
                spec = self._spec_by_arm.get(arm_id)
                if (
                    not generation_id
                    or spec is None
                    or response_model != spec.expected_actual_model_id
                ):
                    raise ProviderError(
                        f"{self.direct_provider_name} omitted or substituted the frozen "
                        "model identity"
                    )
                usage = value.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                accounting = self._rate_card_accounting(
                    generation_id=generation_id,
                    arm_id=arm_id,
                    response_model=response_model,
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
                            "response_model": response_model,
                            "finish_reason": str(
                                ((value.get("choices") or [{}])[0] or {}).get("finish_reason")
                                or "unknown"
                            ),
                            "identity_evidence": "direct_response_model_and_generation_id",
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
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **attempt_event.__dict__,
                                "event_type": "retry_scheduled",
                                "error_type": type(error).__name__,
                                "metadata": {
                                    **attempt_event.metadata,
                                    "retry_reason": "connect_error_before_send",
                                    "backoff_seconds": delay,
                                },
                            }
                        )
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
                    f"{self.direct_provider_name} response timed out after possible "
                    "acceptance; reconcile before retry"
                ) from error
            except httpx.HTTPStatusError as error:
                last_error = error
                status = error.response.status_code
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
                                "provider_error_detail": _safe_http_error_detail(error.response),
                            },
                        }
                    )
                )
                if event_type == "uncertain_delivery":
                    raise UncertainDeliveryError(
                        f"{self.direct_provider_name} returned an ambiguous failure after "
                        "possible dispatch"
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
                f"{self.direct_provider_name} request rejected with HTTP "
                f"{last_error.response.status_code}: "
                f"{_safe_http_error_detail(last_error.response)}"
            ) from last_error
        raise ProviderError(
            f"{self.direct_provider_name} request failed: {type(last_error).__name__}"
        ) from last_error

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
        result.cost_accounting_basis = self.accounting_basis
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
            result.cost_accounting_basis = self.accounting_basis
            result.billing_reconciliation_status = "provider_charge_unavailable"
        return result
