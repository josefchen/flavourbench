"""Governed QwenCloud Chat Completions adapter for FlavourBench workers.

QwenCloud's international endpoint is OpenAI Chat Completions compatible.  The
adapter nevertheless keeps it as a separate execution backend so an
authenticated catalog snapshot, exact returned model identity, provider usage,
and rate-card estimates cannot be confused with OpenRouter evidence.  QwenCloud
does not expose a per-generation charged-amount lookup through this interface;
all completed calls therefore remain cost-unreconciled and unranked.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import get_settings
from .provider import (
    GenerationFailureResult,
    GenerationResult,
    GenerationSpec,
    ProviderError,
)
from .qwencloud_catalog import (
    QWEN38_TOOL_AUTO_INSTRUCTION,
    _is_pay_as_you_go_api_key,
    _safe_base_url,
)
from .real_task_bank import sha256_json
from .service_kimi import KimiDirectProvider

QWENCLOUD_DIRECT_CONTRACT_SCHEMA = "flavourbench-qwencloud-direct-endpoint-contract-v1"
QWENCLOUD_DIRECT_PROVIDER_SLUG = "qwencloud-direct"
QWENCLOUD_ACCOUNTING_BASIS = "frozen_rate_card_times_qwencloud_returned_usage"
QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS = (
    "qwencloud_returned_usage_with_full_unpriced_budget_ceiling"
)
QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS = "provider_rate_and_charge_unavailable"
QWENCLOUD_MUTABLE_ALIAS_IDENTITY_LABEL = "catalog_pinned_at_observation_not_a_frozen_model"
QWENCLOUD_TOOL_CHOICE_TRANSPORT = "auto_with_required_success_postcondition"
QWENCLOUD_MESSAGE_CANONICALIZATION = "official_qwen_chat_tool_continuation_shape_v1"


def _canonicalize_qwen_messages(
    messages: object,
    *,
    add_tool_instruction: bool,
) -> list[dict[str, Any]]:
    """Project shared-loop messages onto Alibaba's documented Chat tool shape."""

    if not isinstance(messages, list) or not messages:
        raise ProviderError("QwenCloud request has no message list")
    projected: list[dict[str, Any]] = []
    system_instruction_added = False
    for raw in messages:
        if not isinstance(raw, dict):
            raise ProviderError("QwenCloud request contains a non-object message")
        role = str(raw.get("role") or "")
        if role in {"system", "user"}:
            content = raw.get("content")
            if not isinstance(content, str):
                raise ProviderError("QwenCloud text message content must be a string")
            if role == "system" and add_tool_instruction and not system_instruction_added:
                content = f"{content}\n\n{QWEN38_TOOL_AUTO_INSTRUCTION}"
                system_instruction_added = True
            projected.append({"role": role, "content": content})
        elif role == "assistant":
            assistant = {"role": "assistant", "content": raw.get("content")}
            tool_calls = raw.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    raise ProviderError("QwenCloud assistant tool_calls must be an array")
                assistant["tool_calls"] = tool_calls
            projected.append(assistant)
        elif role == "tool":
            tool_call_id = raw.get("tool_call_id")
            content = raw.get("content")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ProviderError("QwenCloud tool result lacks a tool_call_id")
            if not isinstance(content, str):
                raise ProviderError("QwenCloud tool result content must be a string")
            projected.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id,
                }
            )
        else:
            raise ProviderError(f"QwenCloud request contains unsupported role: {role}")
    if add_tool_instruction and not system_instruction_added:
        raise ProviderError("QwenCloud tool request lacks its frozen system instruction")
    return projected


class QwenCloudDirectProvider(KimiDirectProvider):
    """Execute one catalog-pinned QwenCloud route without provider fallback."""

    direct_provider_name = "QwenCloud"
    execution_backend = "qwencloud_direct"
    provider_slug = QWENCLOUD_DIRECT_PROVIDER_SLUG
    contract_schema = QWENCLOUD_DIRECT_CONTRACT_SCHEMA
    credential_setting = "qwencloud_api_key"
    credential_environment_name = "FLAVOURBENCH_QWENCLOUD_API_KEY or DASHSCOPE_API_KEY"
    base_url_setting = "qwencloud_base_url"
    timeout_setting = "qwencloud_timeout_seconds"
    accounting_basis = QWENCLOUD_ACCOUNTING_BASIS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Validate the credential destination before constructing a client with
        # an Authorization header.  Tests use the real approved host with a
        # MockTransport; production may additionally use workspace MaaS hosts.
        settings = get_settings()
        api_key = str(settings.qwencloud_api_key)
        if api_key and not _is_pay_as_you_go_api_key(api_key):
            raise ProviderError(
                "automated QwenCloud execution requires a pay-as-you-go Model Studio "
                "credential; "
                "Token Plan and Coding Plan credentials are prohibited"
            )
        _safe_base_url(str(settings.qwencloud_base_url))
        super().__init__(*args, **kwargs)

    def _validate_contract(self, spec: GenerationSpec) -> str:
        requested_model = super()._validate_contract(spec)
        contract = spec.backend_contract_json
        identity_kind = str(contract.get("identity_kind") or "")
        structured_supported = contract.get("structured_outputs_supported")
        expected_cost_reconciliation = (
            "provider_rate_and_charge_unavailable"
            if identity_kind == "mutable_alias"
            else "provider_charge_unavailable"
        )
        if (
            identity_kind not in {"immutable_dated_release", "mutable_alias"}
            or not isinstance(structured_supported, bool)
            or contract.get("allow_fallbacks") is not False
            or contract.get("season_eligible") is not False
            or contract.get("rank_eligible") is not False
            or contract.get("cost_reconciliation") != expected_cost_reconciliation
            or str(contract.get("catalog_sha256") or "") in {"", "unresolved", "unfrozen"}
            or str(contract.get("catalog_entry_sha256") or "") in {"", "unresolved", "unfrozen"}
        ):
            raise ProviderError("QwenCloud route is not a fail-closed exploratory contract")
        if identity_kind == "mutable_alias":
            observed_at = str(contract.get("catalog_observed_at") or "")
            try:
                parsed_observation = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ProviderError(
                    "QwenCloud mutable-alias contract lacks a valid catalog observation"
                ) from error
            if (
                requested_model != "qwen3.8-max"
                or parsed_observation.tzinfo is None
                or contract.get("official") is not False
                or contract.get("model_identity_label") != QWENCLOUD_MUTABLE_ALIAS_IDENTITY_LABEL
                or contract.get("catalog_pinned_at_observation") is not True
                or contract.get("mutable_alias_execution_requires_explicit_opt_in") is not True
                or contract.get("provider_rate_status") != "unpublished_at_catalog_observation"
                or not spec.allow_mutable_alias_exploratory
                or contract.get("tool_choice_transport_mode") != QWENCLOUD_TOOL_CHOICE_TRANSPORT
                or contract.get("tool_choice_required_supported") is not False
                or contract.get("required_success_postcondition")
                != "at_least_one_successful_real_epicure_tool_trace"
                or contract.get("tool_selection_system_instruction") != QWEN38_TOOL_AUTO_INSTRUCTION
                or contract.get("tool_selection_system_instruction_sha256")
                != sha256_json(QWEN38_TOOL_AUTO_INSTRUCTION)
                or contract.get("message_canonicalization") != QWENCLOUD_MESSAGE_CANONICALIZATION
                or len(str(contract.get("predecessor_failure_artifact_sha256") or "")) != 64
            ):
                raise ProviderError(
                    "QwenCloud mutable alias requires the explicit catalog-pinned "
                    "exploratory contract"
                )
        elif spec.allow_mutable_alias_exploratory:
            raise ProviderError(
                "mutable-alias exploratory opt-in cannot be applied to a dated release"
            )
        if spec.final_response_mode == "structured_json" and not structured_supported:
            raise ProviderError(
                "QwenCloud model lacks frozen structured-output support; use an unranked "
                "plain-text contract smoke"
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
        if spec is not None and spec.backend_contract_json.get("identity_kind") == "mutable_alias":
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            completion_details = usage.get("completion_tokens_details")
            completion_details = completion_details if isinstance(completion_details, dict) else {}
            reasoning_tokens = int(
                usage.get("reasoning_tokens") or completion_details.get("reasoning_tokens") or 0
            )
            return {
                "generation_id": generation_id,
                # The direct catalog and public documentation expose no price
                # for this alias.  Zero means cost unknown, never free; the
                # complete pre-admitted ceiling remains reserved separately.
                "cost_micros": 0,
                "provider": self.provider_slug,
                "model": response_model,
                "reconciled": False,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "accounting_basis": QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS,
                "billing_reconciliation_status": (QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS),
                "provider_cost_known": False,
            }
        return super()._rate_card_accounting(
            generation_id=generation_id,
            arm_id=arm_id,
            response_model=response_model,
            usage=usage,
        )

    async def _post(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
        *,
        arm_id: str = "",
        phase: str = "unknown",
        governance_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_payload = dict(payload)
        spec = self._spec_by_arm.get(arm_id)
        contract = spec.backend_contract_json if spec is not None else {}
        successor_contract = (
            contract.get("tool_choice_transport_mode") == QWENCLOUD_TOOL_CHOICE_TRANSPORT
            and contract.get("message_canonicalization") == QWENCLOUD_MESSAGE_CANONICALIZATION
        )
        if successor_contract:
            initial_tool_selection = phase == "tool_round_0"
            request_payload["messages"] = _canonicalize_qwen_messages(
                request_payload.get("messages"),
                add_tool_instruction=initial_tool_selection,
            )
            if request_payload.get("tool_choice") == "required":
                if not initial_tool_selection:
                    raise ProviderError(
                        "QwenCloud required tool choice appeared outside initial selection"
                    )
                request_payload["tool_choice"] = "auto"

        # Model Studio documents thinking controls, but not a portable mapping
        # from FlavourBench's low/high/max effort scale for this frozen Chat
        # Completions route.  Reject the control instead of inventing one.
        if request_payload.get("reasoning") is not None:
            raise ProviderError(
                "QwenCloud route has no frozen low/high/max reasoning-effort translation"
            )
        return await super()._post(
            request_payload,
            idempotency_key,
            arm_id=arm_id,
            phase=phase,
            governance_metadata=governance_metadata,
        )

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        result = await super().generate(spec)
        # Repeat the boundary at the adapter edge so a future base-class change
        # cannot silently promote provider-usage estimates to reconciled cost.
        result.cost_reconciled = False
        if spec.backend_contract_json.get("identity_kind") == "mutable_alias":
            result.cost_accounting_basis = QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
            result.billing_reconciliation_status = QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
        else:
            result.cost_accounting_basis = self.accounting_basis
            result.billing_reconciliation_status = "provider_charge_unavailable"
        return result

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        result = await super().reconcile_failure(spec, error)
        if (
            result is not None
            and spec.backend_contract_json.get("identity_kind") == "mutable_alias"
        ):
            result.cost_accounting_basis = QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
            result.billing_reconciliation_status = QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
        return result
