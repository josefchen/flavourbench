from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import get_settings
from .endpoint_contract import endpoint_contract_sha256
from .execution_policy import (
    DIRECT_TOOL_CONTRACT_PROTOCOL,
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOLS,
    PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    ExecutionPolicy,
    assert_legacy_paid_cli_allowed,
)
from .mcp_client import McpSession
from .protocol_contract import build_protocol_bundle
from .provider import (
    GenerationResult,
    GenerationSpec,
    OpenRouterProvider,
    ProviderAttemptEvent,
    ToolTrace,
    response_schema_sha256,
    system_prompt_sha256,
)
from .run_journal import RunJournal
from .tool_contract import TOOL_CONTRACT_PROMPT, required_tool_contract

CONFIRMATION = "UNRANKED_REAL_SMOKE"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_PROVIDER = "darkbloom"
DEFAULT_PROMPT = (
    "Design a make-ahead vegetarian starter for six using watermelon, green olive, and mint, "
    "with no added sugar. Explain salt, acid, texture, aromatic intensity, and tasting "
    "uncertainty."
)
REQUIRED_ENDPOINT_PARAMETERS = {
    "max_tokens",
    "response_format",
    "structured_outputs",
    "tool_choice",
    "tools",
}
ENDPOINT_EXECUTION_CONTRACT_FIELDS = (
    "model_id",
    "provider_name",
    "tag",
    "quantization",
    "context_length",
    "max_completion_tokens",
    "supported_parameters",
)
LIVE_PROTOCOL_SCHEMA_VERSION = "flavourbench-live-development-protocol-v10"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def endpoint_execution_contract(endpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the non-observational endpoint fields that govern execution."""

    contract = {field: endpoint.get(field) for field in ENDPOINT_EXECUTION_CONTRACT_FIELDS}
    contract["supported_parameters"] = sorted(contract.get("supported_parameters") or [])
    return contract


def endpoint_execution_contract_sha256(endpoint: dict[str, Any]) -> str:
    return _sha256(endpoint_execution_contract(endpoint))


def frozen_generation_contract(model: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact GenerationSpec contract before any paid request."""

    settings = get_settings()
    supported = frozenset(str(item) for item in endpoint.get("supported_parameters") or [])
    decoding_candidates: dict[str, int | float] = {
        "max_tokens": settings.max_output_tokens,
        "temperature": settings.decoding_temperature,
        "top_p": settings.decoding_top_p,
        "seed": settings.decoding_seed,
    }
    decoding = {name: value for name, value in decoding_candidates.items() if name in supported}
    expected_model = str(model.get("canonical_slug") or "")
    expected_provider = str(endpoint.get("provider_name") or "")
    endpoint_document_sha256 = endpoint_execution_contract_sha256(endpoint)
    if not expected_model or not expected_provider:
        raise RuntimeError("catalog endpoint lacks an exact actual model/provider identity")
    max_completion_tokens = endpoint.get("max_completion_tokens")
    # OpenRouter uses null when the upstream endpoint does not publish a
    # separate completion ceiling. It is metadata-unknown, not an assertion
    # that max_tokens is unsupported. The required max_tokens capability and
    # our client-side bound remain frozen in the request contract.
    if max_completion_tokens is not None and (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens <= 0
        or settings.max_output_tokens > max_completion_tokens
    ):
        raise RuntimeError("configured max output exceeds the endpoint completion limit")
    return {
        "supported_parameters": supported,
        "decoding_parameters": decoding,
        "expected_actual_model_id": expected_model,
        "expected_actual_provider_slug": expected_provider,
        "endpoint_contract_sha256": endpoint_contract_sha256(
            model_id=str(model.get("id") or ""),
            provider_slug=str(endpoint.get("tag") or ""),
            expected_actual_model_id=expected_model,
            expected_actual_provider_slug=expected_provider,
            supported_parameters=sorted(supported),
            decoding=decoding,
            endpoint_max_completion_tokens=max_completion_tokens,
            endpoint_document_sha256=endpoint_document_sha256,
        ),
    }


def build_live_protocol_bundle(
    *,
    candidate_manifest_sha256: str,
    dataset_work_item_id: str,
    dataset_task_id: str,
    prompt: str,
    category: str,
    model: dict[str, Any],
    endpoint: dict[str, Any],
    generation_contract: dict[str, Any],
    execution_policy: ExecutionPolicy,
    provenance: dict[str, Any],
    tool_schema_sha256: str,
    run_purpose: str,
    final_response_mode: str,
    selected_conditions: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Bind a paid live request to its exact model, task, policy, and Epicure runtime."""

    epicure_release_id = str(provenance.get("release_id") or "")
    epicure_bundle_sha256 = str(provenance.get("bundle_sha256") or "")
    epicure_application_sha256 = str(provenance.get("application_sha256") or "")
    sha_fields = {
        "epicure_bundle_sha256": epicure_bundle_sha256,
        "epicure_application_sha256": epicure_application_sha256,
        "tool_schema_sha256": tool_schema_sha256,
        "execution_policy_sha256": execution_policy.sha256,
        "endpoint_contract_sha256": str(generation_contract.get("endpoint_contract_sha256") or ""),
    }
    for field, value in sha_fields.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RuntimeError(f"live protocol has an invalid {field}")
    if not epicure_release_id:
        raise RuntimeError("live protocol has no Epicure release identity")

    route_registry_sha256 = candidate_manifest_sha256 or _sha256(
        {
            "schema_version": "flavourbench-live-route-binding-v1",
            "model": model,
            "endpoint": endpoint_execution_contract(endpoint),
        }
    )
    run_binding = {
        "schema_version": "flavourbench-live-run-binding-v2",
        "candidate_manifest_sha256": candidate_manifest_sha256 or None,
        "route_registry_sha256": route_registry_sha256,
        "dataset_work_item_id": dataset_work_item_id or None,
        "dataset_task_id": dataset_task_id or None,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "category": category,
        "run_purpose": run_purpose,
        "requested_model_id": str(model.get("id") or ""),
        "canonical_model_slug": str(model.get("canonical_slug") or ""),
        "provider_tag": str(endpoint.get("tag") or ""),
        "endpoint_contract_sha256": generation_contract["endpoint_contract_sha256"],
        "execution_policy_sha256": execution_policy.sha256,
        "final_response_mode": final_response_mode,
        "matched_planning": execution_policy.matched_planning,
        "max_intermediate_tokens": execution_policy.max_intermediate_tokens,
        "required_tool_contract_max_intermediate_tokens": (
            execution_policy.required_tool_contract_max_intermediate_tokens
        ),
        "evidence_protocol": execution_policy.evidence_protocol,
        "required_tool_contract_protocol": (execution_policy.required_tool_contract_protocol),
        "required_tool_contract_sha256": required_tool_contract(execution_policy)[
            "content_address"
        ]["digest"],
        "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
        "intermediate_reasoning_effort": execution_policy.intermediate_reasoning_effort,
        "final_reasoning_effort": execution_policy.final_reasoning_effort,
        "response_schema_sha256": response_schema_sha256(final_response_mode),
    }
    if generation_contract.get("allow_mutable_alias_exploratory") is True:
        run_binding["allow_mutable_alias_exploratory"] = True
    if selected_conditions is not None:
        conditions = tuple(selected_conditions)
        if (
            not conditions
            or len(set(conditions)) != len(conditions)
            or not set(conditions) <= {"epicure_off", "epicure_on"}
        ):
            raise RuntimeError("live protocol has invalid selected conditions")
        run_binding["selected_conditions"] = list(conditions)
    run_binding_sha256 = _sha256(run_binding)
    core_bundle, core_bundle_sha256 = build_protocol_bundle(
        tool_registry_sha256=tool_schema_sha256,
        epicure_release_id=epicure_release_id,
        epicure_bundle_sha256=epicure_bundle_sha256,
        epicure_application_sha256=epicure_application_sha256,
        analysis_plan_sha256=run_binding_sha256,
        model_smoke_registry_sha256=route_registry_sha256,
        final_response_mode=final_response_mode,
        evidence_protocol=execution_policy.evidence_protocol,
        required_tool_contract_sha256=required_tool_contract(execution_policy)["content_address"][
            "digest"
        ],
        settings=get_settings(),
    )
    bundle = {
        "schema_version": LIVE_PROTOCOL_SCHEMA_VERSION,
        "core_protocol_bundle": core_bundle,
        "core_protocol_bundle_sha256": core_bundle_sha256,
        "run_binding": run_binding,
        "run_binding_sha256": run_binding_sha256,
    }
    return bundle, _sha256(bundle)


def _normalise_provider(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("Infinity")


async def _openrouter_get(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"OpenRouter returned an invalid document for {path}")
    return payload


async def _key_status(client: httpx.AsyncClient) -> dict[str, Any]:
    data = (await _openrouter_get(client, "key")).get("data") or {}
    return {
        "limit_usd": data.get("limit"),
        "limit_remaining_usd": data.get("limit_remaining"),
        "usage_daily_usd": data.get("usage_daily"),
        "usage_monthly_usd": data.get("usage_monthly"),
        "is_free_tier": data.get("is_free_tier"),
    }


async def _endpoint_contract(
    client: httpx.AsyncClient, model_id: str, provider_slug: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    author, slug = model_id.split("/", 1)
    model_data = (await _openrouter_get(client, f"model/{quote(author)}/{quote(slug, safe=':')}"))[
        "data"
    ]
    endpoint_data = (
        await _openrouter_get(client, f"models/{quote(author)}/{quote(slug, safe=':')}/endpoints")
    )["data"]
    endpoints = endpoint_data.get("endpoints") or []
    expected = _normalise_provider(provider_slug)
    exact_matches = [
        endpoint
        for endpoint in endpoints
        if _normalise_provider(str(endpoint.get("tag") or "")) == expected
    ]
    matches = exact_matches or [
        endpoint
        for endpoint in endpoints
        if expected
        in {
            _normalise_provider(str(endpoint.get("tag") or "").split("/")[0]),
            _normalise_provider(str(endpoint.get("provider_name") or "")),
        }
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one endpoint for {model_id}@{provider_slug}; found {len(matches)}"
        )
    endpoint = matches[0]
    supported = set(endpoint.get("supported_parameters") or [])
    missing = REQUIRED_ENDPOINT_PARAMETERS - supported
    if missing:
        raise RuntimeError(
            f"{model_id}@{provider_slug} is missing contract parameters: {sorted(missing)}"
        )
    safe_model = {
        "id": model_data.get("id"),
        "canonical_slug": model_data.get("canonical_slug"),
        "name": model_data.get("name"),
        "context_length": model_data.get("context_length"),
        "pricing": model_data.get("pricing") or {},
        "supported_parameters": model_data.get("supported_parameters") or [],
    }
    safe_endpoint = {
        "name": endpoint.get("name"),
        "provider_name": endpoint.get("provider_name"),
        "tag": endpoint.get("tag"),
        "model_id": endpoint.get("model_id"),
        "quantization": endpoint.get("quantization"),
        "context_length": endpoint.get("context_length"),
        "max_completion_tokens": endpoint.get("max_completion_tokens"),
        "pricing": endpoint.get("pricing") or {},
        "supported_parameters": sorted(supported),
        "uptime_last_1d": endpoint.get("uptime_last_1d"),
    }
    return safe_model, safe_endpoint


def _worst_case_cost_usd(
    endpoint: dict[str, Any],
    *,
    prompt: str,
    include_tool_contract: bool,
    execution_policy: ExecutionPolicy | None = None,
    conditions: Sequence[str] = ("epicure_off", "epicure_on"),
) -> Decimal:
    settings = get_settings()
    policy = execution_policy or ExecutionPolicy.from_settings(
        settings,
        pair_arm_scheduling="concurrent",
    )
    pricing = endpoint.get("pricing") or {}
    prompt_rate = _decimal(pricing.get("prompt"))
    completion_rate = _decimal(pricing.get("completion"))
    reasoning_rate = _decimal(pricing.get("internal_reasoning") or 0)
    request_rate = _decimal(pricing.get("request") or 0)
    if not all(
        rate.is_finite() for rate in (prompt_rate, completion_rate, reasoning_rate, request_rate)
    ):
        return Decimal("Infinity")
    # Deliberately conservative: every Epicure round may add the full bounded
    # tool payload and a full completion before the final response.
    possible_tool_calls = min(
        policy.max_tool_calls_total,
        policy.max_tool_rounds * policy.max_tool_calls_per_round,
    )
    tool_context_bytes = min(
        policy.max_cumulative_tool_result_bytes,
        policy.max_tool_result_bytes * possible_tool_calls,
    )
    approximate_request_tokens = Decimal(
        len(prompt.encode("utf-8"))
        + policy.approximate_non_user_prompt_bytes
        + policy.tool_catalog_bytes_bound
        + tool_context_bytes
    ) / Decimal(3)
    selected = tuple(conditions)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= {"epicure_off", "epicure_on"}
    ):
        raise RuntimeError("forecast conditions must be a unique non-empty Epicure subset")
    total_requests = 0
    total_output_token_count = 0
    if "epicure_off" in selected:
        off_intermediate_requests = 0
        if policy.matched_planning:
            off_intermediate_requests += 1
            if policy.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS:
                off_intermediate_requests += 1
        off_requests = 1 + off_intermediate_requests
        total_requests += off_requests
        total_output_token_count += (
            policy.max_output_tokens + off_intermediate_requests * policy.max_intermediate_tokens
            if policy.matched_planning or policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else off_requests * policy.max_output_tokens
        )
    if "epicure_on" in selected:
        on_intermediate_requests = (
            1
            if policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else policy.max_tool_rounds
        )
        if policy.matched_planning and policy.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS:
            on_intermediate_requests += 1
        on_requests = 1 + on_intermediate_requests
        total_requests += on_requests
        total_output_token_count += (
            policy.max_output_tokens + on_intermediate_requests * policy.max_intermediate_tokens
            if policy.matched_planning or policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else on_requests * policy.max_output_tokens
        )
    if include_tool_contract:
        direct_tool_contract = (
            policy.evidence_protocol in GOVERNED_EPICURE_PROTOCOLS
            and policy.required_tool_contract_protocol == DIRECT_TOOL_CONTRACT_PROTOCOL
        )
        contract_requests = policy.max_tool_rounds + 1 if direct_tool_contract else on_requests
        total_requests += contract_requests
        total_output_token_count += (
            (
                policy.max_tool_rounds
                + (
                    0
                    if direct_tool_contract
                    else 1
                    if policy.evidence_protocol in GOVERNED_EPICURE_PROTOCOLS
                    else 0
                )
            )
            * (
                policy.required_tool_contract_max_intermediate_tokens
                if direct_tool_contract
                else policy.max_intermediate_tokens
            )
            + policy.max_output_tokens
            if policy.matched_planning
            else on_requests * policy.max_output_tokens
        )
    input_tokens = Decimal(total_requests) * approximate_request_tokens
    output_tokens = Decimal(total_output_token_count)
    return (
        prompt_rate * input_tokens
        + completion_rate * output_tokens
        + reasoning_rate * output_tokens
        + request_rate * total_requests
    )


async def _epicure_attestation() -> tuple[dict[str, Any], str]:
    settings = get_settings()
    provenance_url = os.environ.get("FLAVOURBENCH_EPICURE_PROVENANCE_URL")
    if not provenance_url:
        provenance_url = settings.mcp_url.removesuffix("/mcp").rstrip("/") + "/provenance"
    headers = {"Authorization": f"Bearer {settings.mcp_token}"} if settings.mcp_token else {}
    async with httpx.AsyncClient(timeout=settings.mcp_timeout_seconds, headers=headers) as client:
        response = await client.get(provenance_url)
        response.raise_for_status()
        provenance = response.json()
    async with McpSession() as mcp:
        tools = await mcp.list_tools()
    tool_sha = _sha256(tools)
    expected_tool_sha = settings.epicure_tool_schema_sha256
    if expected_tool_sha not in {"", "unresolved"} and expected_tool_sha != tool_sha:
        raise RuntimeError("runtime Epicure tool catalog differs from the configured hash")
    expected_release = settings.epicure_release_id
    if (
        expected_release not in {"", "unresolved", "unresolved-1790-development-only"}
        and provenance.get("release_id") != expected_release
    ):
        raise RuntimeError("runtime Epicure release differs from the configured identity")
    expected_bundle = settings.epicure_bundle_sha256
    if (
        expected_bundle not in {"", "unresolved"}
        and provenance.get("bundle_sha256") != expected_bundle
    ):
        raise RuntimeError("runtime Epicure bundle differs from the configured hash")
    expected_application = settings.epicure_application_sha256
    if (
        expected_application not in {"", "unresolved"}
        and provenance.get("application_sha256") != expected_application
    ):
        raise RuntimeError("runtime Epicure application differs from the configured hash")
    return provenance, tool_sha


def verify_expected_epicure_attestation(
    provenance: dict[str, Any],
    tool_schema_sha256: str,
    *,
    expected_release_id: str = "",
    expected_bundle_sha256: str = "",
    expected_application_sha256: str = "",
    expected_tool_schema_sha256: str = "",
) -> None:
    """Fail before generation when the live local Epicure runtime drifts."""

    expected = {
        "release_id": expected_release_id,
        "bundle_sha256": expected_bundle_sha256,
        "application_sha256": expected_application_sha256,
        "tool_schema_sha256": expected_tool_schema_sha256,
    }
    observed = {
        "release_id": str(provenance.get("release_id") or ""),
        "bundle_sha256": str(provenance.get("bundle_sha256") or ""),
        "application_sha256": str(provenance.get("application_sha256") or ""),
        "tool_schema_sha256": tool_schema_sha256,
    }
    for field, expected_value in expected.items():
        if expected_value and observed[field] != expected_value:
            raise RuntimeError(f"live Epicure {field} differs from the frozen execution contract")


def _result_payload(result: GenerationResult) -> dict[str, Any]:
    return {
        "answer_markdown": result.answer_markdown,
        "output_json": result.output_json,
        "actual_model_id": result.actual_model_id,
        "actual_provider": result.provider_slug,
        "generation_id": result.generation_id,
        "generation_ids": result.generation_ids,
        "generation_metadata": result.generation_metadata,
        "decoding": result.decoding_json,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cost_micros": result.cost_micros,
        "cost_reconciled": result.cost_reconciled,
        "latency_ms": result.latency_ms,
        "retries": result.retries,
        "finish_reason": result.finish_reason,
        "final_response_mode": result.final_response_mode,
        "structured_output_requested": result.structured_output_requested,
        "structured_output_valid": result.structured_output_valid,
        "backend_response_schema_sha256": result.backend_response_schema_sha256,
        "backend_tool_schema_sha256": result.backend_tool_schema_sha256,
        "cost_accounting_basis": result.cost_accounting_basis,
        "billing_reconciliation_status": result.billing_reconciliation_status,
        "intermediate_outputs": result.intermediate_outputs,
        "tool_trace": [
            {
                "round_index": trace.round_index,
                "name": trace.name,
                "arguments": trace.arguments,
                "result": trace.result,
                "result_sha256": hashlib.sha256(trace.result.encode()).hexdigest(),
                "latency_ms": trace.latency_ms,
                "is_error": trace.is_error,
            }
            for trace in result.tool_traces
        ],
    }


def _model_identity_matches(actual: str, model: dict[str, Any]) -> bool:
    expected = str(model.get("canonical_slug") or "")
    return bool(expected) and actual == expected


def _provider_identity_matches(actual: str, endpoint: dict[str, Any]) -> bool:
    expected = str(endpoint.get("provider_name") or "")
    return bool(expected) and actual == expected


async def live_smoke(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    if settings.execution_mode != "live" or not settings.live_authorized:
        raise RuntimeError("live smoke requires live execution and explicit authorization flags")
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"pass --confirm {CONFIRMATION} to acknowledge real external calls")
    if args.contract_only and args.skip_tool_contract:
        raise RuntimeError("--contract-only and --skip-tool-contract are mutually exclusive")
    requested_conditions = tuple(getattr(args, "condition", None) or ())
    if args.contract_only and requested_conditions:
        raise RuntimeError("--condition cannot be combined with --contract-only")
    selected_conditions = (
        ("epicure_on",)
        if args.contract_only
        else requested_conditions or ("epicure_off", "epicure_on")
    )
    if len(set(selected_conditions)) != len(selected_conditions):
        raise RuntimeError("each selected Epicure condition may appear only once")
    if args.cap_usd < 0:
        raise RuntimeError("the smoke cap cannot be negative")
    candidate_manifest_sha256 = getattr(args, "candidate_manifest_sha256", "")
    if candidate_manifest_sha256 and (
        len(candidate_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in candidate_manifest_sha256)
    ):
        raise RuntimeError("candidate manifest SHA-256 must be 64 lowercase hex characters")
    dataset_work_item_id = getattr(args, "dataset_work_item_id", "")
    if dataset_work_item_id and (
        len(dataset_work_item_id) != 64
        or any(character not in "0123456789abcdef" for character in dataset_work_item_id)
    ):
        raise RuntimeError("dataset work-item ID must be 64 lowercase hex characters")
    dataset_task_id = getattr(args, "dataset_task_id", "")
    if dataset_task_id and not dataset_work_item_id:
        raise RuntimeError("dataset task ID requires a dataset work-item ID")
    expected_endpoint_sha256 = getattr(args, "expected_endpoint_execution_sha256", "")
    if expected_endpoint_sha256 and (
        len(expected_endpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_endpoint_sha256)
    ):
        raise RuntimeError("expected endpoint execution SHA-256 is malformed")
    execution_policy = ExecutionPolicy.from_settings(
        settings,
        pair_arm_scheduling=(
            "sequential" if getattr(args, "sequential_arms", False) else "concurrent"
        ),
        final_response_mode=(
            "plain_text" if getattr(args, "plain_text_final", False) else "structured_json"
        ),
        matched_planning=(
            getattr(args, "evidence_protocol", "legacy_v6") in MATCHED_EVIDENCE_PROTOCOLS
        ),
        evidence_protocol=getattr(args, "evidence_protocol", "legacy_v6"),
        intermediate_reasoning_effort=getattr(args, "intermediate_reasoning_effort", None),
        final_reasoning_effort=getattr(args, "final_reasoning_effort", None),
        tool_catalog_bytes_bound=int(getattr(args, "tool_catalog_bytes_bound", 0)),
        epicure_on_tool_required=bool(getattr(args, "require_epicure_call", False)),
    )
    expected_execution_policy_sha256 = getattr(args, "expected_execution_policy_sha256", "")
    if expected_execution_policy_sha256 and (
        len(expected_execution_policy_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_execution_policy_sha256
        )
    ):
        raise RuntimeError("expected execution-policy SHA-256 is malformed")
    if (
        expected_execution_policy_sha256
        and execution_policy.sha256 != expected_execution_policy_sha256
    ):
        raise RuntimeError("runtime settings differ from the frozen execution policy")
    if args.cap_usd == 0 and (
        settings.openrouter_max_prompt_price_per_mtok != 0
        or settings.openrouter_max_completion_price_per_mtok != 0
    ):
        raise RuntimeError("zero-cost smoke requires both OpenRouter max-price limits to be zero")

    run_id = str(getattr(args, "frozen_run_id", "") or uuid.uuid4())
    frozen_attempt_slots = getattr(args, "frozen_attempt_slots", None)
    attempt_id_factory = None
    if frozen_attempt_slots is not None:
        if not isinstance(frozen_attempt_slots, list) or not frozen_attempt_slots:
            raise RuntimeError("frozen attempt slots must be a non-empty list")
        slot_map: dict[tuple[str, str, int], str] = {}
        frozen_ids: set[str] = set()
        for slot in frozen_attempt_slots:
            if not isinstance(slot, dict):
                raise RuntimeError("frozen attempt slot must be an object")
            key = (
                str(slot.get("arm_id") or ""),
                str(slot.get("phase") or ""),
                int(slot.get("attempt_index", -1)),
            )
            attempt_id = str(slot.get("attempt_id") or "")
            if (
                not all(key[:2])
                or key[2] < 0
                or not attempt_id
                or key in slot_map
                or attempt_id in frozen_ids
            ):
                raise RuntimeError("frozen attempt slots are malformed or duplicated")
            slot_map[key] = attempt_id
            frozen_ids.add(attempt_id)

        def resolve_attempt_id(arm_id: str, phase: str, attempt_index: int) -> str:
            try:
                return slot_map[(arm_id, phase, attempt_index)]
            except KeyError as error:
                raise RuntimeError(
                    "external request has no pre-frozen attempt-ID slot: "
                    f"{arm_id}/{phase}/{attempt_index}"
                ) from error

        attempt_id_factory = resolve_attempt_id
    started_at = datetime.now(UTC)
    output_dir = Path(args.output_dir)
    journal = RunJournal.create(
        output_dir,
        run_id=run_id,
        metadata={
            "run_class": "engineering_live_smoke",
            "candidate_manifest_sha256": candidate_manifest_sha256 or None,
            "dataset_work_item_id": dataset_work_item_id or None,
            "dataset_task_id": dataset_task_id or None,
            "requested_model_id": args.model_id,
            "requested_provider": args.provider_slug,
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "category": args.category,
            "contract_only": bool(args.contract_only),
            "epicure_conditions": list(selected_conditions),
        },
    )
    attempt_events: list[dict[str, Any]] = []
    mcp_trace_events: list[dict[str, Any]] = []

    def sink(event: ProviderAttemptEvent) -> None:
        payload = asdict(event)
        # The append+fsync must complete before OpenRouterProvider may send
        # request bytes. A journal failure propagates and aborts provider I/O.
        journal.append("provider_attempt", payload)
        attempt_events.append(payload)

    def tool_sink(arm_id: str, trace: ToolTrace) -> None:
        payload = {
            "arm_id": arm_id,
            "round_index": trace.round_index,
            "name": trace.name,
            "arguments": trace.arguments,
            "result": trace.result,
            "result_sha256": hashlib.sha256(trace.result.encode()).hexdigest(),
            "latency_ms": trace.latency_ms,
            "is_error": trace.is_error,
        }
        # The complete MCP result is durable before another paid tool round.
        journal.append("mcp_trace", payload)
        mcp_trace_events.append(payload)

    account_headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    async with httpx.AsyncClient(
        base_url=settings.openrouter_accounting_base_url.rstrip("/") + "/",
        headers=account_headers,
        timeout=settings.openrouter_timeout_seconds,
    ) as catalog_client:
        before = await _key_status(catalog_client)
        journal.append("openrouter_key_status", {"position": "before", **before})
        model, endpoint = await _endpoint_contract(
            catalog_client, args.model_id, args.provider_slug
        )
        if endpoint.get("tag") != args.provider_slug:
            raise RuntimeError("requested provider must equal the exact current endpoint tag")
        expected_canonical_slug = getattr(args, "expected_canonical_model_slug", "")
        if expected_canonical_slug and model.get("canonical_slug") != expected_canonical_slug:
            raise RuntimeError(
                "current canonical model slug differs from the frozen execution contract"
            )
        if (
            expected_endpoint_sha256
            and endpoint_execution_contract_sha256(endpoint) != expected_endpoint_sha256
        ):
            raise RuntimeError(
                "current endpoint execution contract differs from the frozen manifest"
            )
        forecast = _worst_case_cost_usd(
            endpoint,
            prompt=args.prompt,
            include_tool_contract=not args.skip_tool_contract,
            execution_policy=execution_policy,
            conditions=selected_conditions,
        )
        if forecast > args.cap_usd:
            raise RuntimeError(
                f"forecast cost ${forecast:.6f} exceeds smoke cap ${args.cap_usd:.6f}"
            )
        provenance, tool_schema_sha256 = await _epicure_attestation()
        verify_expected_epicure_attestation(
            provenance,
            tool_schema_sha256,
            expected_release_id=getattr(args, "expected_epicure_release_id", ""),
            expected_bundle_sha256=getattr(args, "expected_epicure_bundle_sha256", ""),
            expected_application_sha256=getattr(args, "expected_epicure_application_sha256", ""),
            expected_tool_schema_sha256=getattr(args, "expected_epicure_tool_schema_sha256", ""),
        )
        generation_contract = frozen_generation_contract(model, endpoint)
        run_purpose = (
            "tool_contract"
            if args.contract_only
            else "epicure_on_off_pair"
            if selected_conditions == ("epicure_off", "epicure_on")
            else "epicure_condition_subset"
        )
        protocol_bundle, protocol_bundle_sha256 = build_live_protocol_bundle(
            candidate_manifest_sha256=candidate_manifest_sha256,
            dataset_work_item_id=dataset_work_item_id,
            dataset_task_id=dataset_task_id,
            prompt=args.prompt,
            category=args.category,
            model=model,
            endpoint=endpoint,
            generation_contract=generation_contract,
            execution_policy=execution_policy,
            provenance=provenance,
            tool_schema_sha256=tool_schema_sha256,
            run_purpose=run_purpose,
            final_response_mode=execution_policy.final_response_mode,
            selected_conditions=(
                selected_conditions if run_purpose == "epicure_condition_subset" else None
            ),
        )
        generation_contract.update(
            {
                "final_response_mode": execution_policy.final_response_mode,
                "matched_planning": execution_policy.matched_planning,
                "intermediate_max_tokens": execution_policy.max_intermediate_tokens,
                "required_tool_contract_max_intermediate_tokens": (
                    execution_policy.required_tool_contract_max_intermediate_tokens
                ),
                "evidence_protocol": execution_policy.evidence_protocol,
                "required_tool_contract_protocol": (
                    execution_policy.required_tool_contract_protocol
                ),
                "required_tool_contract_sha256": required_tool_contract(execution_policy)[
                    "content_address"
                ]["digest"],
                "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
                "intermediate_reasoning_effort": execution_policy.intermediate_reasoning_effort,
                "final_reasoning_effort": execution_policy.final_reasoning_effort,
                "protocol_bundle_sha256": protocol_bundle_sha256,
                "expected_epicure_release_id": str(provenance["release_id"]),
                "expected_epicure_bundle_sha256": str(provenance["bundle_sha256"]),
                "expected_epicure_application_sha256": str(provenance["application_sha256"]),
                "expected_epicure_tool_schema_sha256": tool_schema_sha256,
            }
        )

        provider = OpenRouterProvider(
            attempt_sink=sink,
            tool_sink=tool_sink,
            attempt_id_factory=attempt_id_factory,
        )
        provider_routing_controls = provider._provider_preferences(  # noqa: SLF001
            args.provider_slug
        )
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        incomplete_generation_metadata: list[dict[str, Any]] = []
        try:
            if not args.contract_only:
                specs = [
                    GenerationSpec(
                        arm_id=f"{run_id}:{condition}",
                        battle_id=run_id,
                        prompt=args.prompt,
                        category=args.category,
                        model_id=args.model_id,
                        model_name=str(model.get("name") or args.model_id),
                        provider_slug=args.provider_slug,
                        condition=condition,
                        idempotency_key=f"flavourbench-live-smoke:{run_id}:{condition}",
                        **{
                            **generation_contract,
                            "epicure_on_tool_required": (
                                condition == "epicure_on"
                                and execution_policy.epicure_on_tool_required
                            ),
                        },
                    )
                    for condition in selected_conditions
                ]
                if getattr(args, "sequential_arms", False):
                    outcomes: list[GenerationResult | Exception] = []
                    for spec in specs:
                        try:
                            outcomes.append(await provider.generate(spec))
                        except Exception as error:  # preserve a failed paid arm in artifact
                            outcomes.append(error)
                else:
                    outcomes = await asyncio.gather(
                        *(provider.generate(spec) for spec in specs), return_exceptions=True
                    )
                for spec, outcome in zip(specs, outcomes, strict=True):
                    if isinstance(outcome, GenerationResult):
                        results[spec.condition] = _result_payload(outcome)
                        if not _model_identity_matches(outcome.actual_model_id, model):
                            errors[f"{spec.condition}_model_identity"] = (
                                f"returned model {outcome.actual_model_id!r} differs from the "
                                "catalog identity"
                            )
                        if not _provider_identity_matches(outcome.provider_slug, endpoint):
                            errors[f"{spec.condition}_provider_identity"] = (
                                f"returned provider {outcome.provider_slug!r} differs from the "
                                "fixed endpoint provider"
                            )
                    else:
                        errors[spec.condition] = f"{type(outcome).__name__}: {outcome}"

            if not args.skip_tool_contract:
                try:
                    contract = await provider.generate(
                        GenerationSpec(
                            arm_id=f"{run_id}:tool_contract",
                            battle_id=run_id,
                            prompt=TOOL_CONTRACT_PROMPT,
                            category="evidence",
                            model_id=args.model_id,
                            model_name=str(model.get("name") or args.model_id),
                            provider_slug=args.provider_slug,
                            condition="epicure_on",
                            idempotency_key=f"flavourbench-live-smoke:{run_id}:tool-contract",
                            tool_choice="required",
                            tool_contract_diagnostic=True,
                            **generation_contract,
                        )
                    )
                    results["tool_contract"] = _result_payload(contract)
                    if not _model_identity_matches(contract.actual_model_id, model):
                        errors["tool_contract_model_identity"] = (
                            f"returned model {contract.actual_model_id!r} differs from the "
                            "catalog identity"
                        )
                    if not _provider_identity_matches(contract.provider_slug, endpoint):
                        errors["tool_contract_provider_identity"] = (
                            f"returned provider {contract.provider_slug!r} differs from the "
                            "fixed endpoint provider"
                        )
                    if not contract.tool_traces:
                        errors["tool_contract"] = "required tool call produced no Epicure trace"
                    elif not any(
                        trace.name == "find_pairings" and not trace.is_error
                        for trace in contract.tool_traces
                    ):
                        errors["tool_contract"] = (
                            "contract did not complete a successful find_pairings call"
                        )
                except Exception as exc:  # artifact must preserve failed paid/free attempts
                    errors["tool_contract"] = f"{type(exc).__name__}: {exc}"
        finally:
            accounted_ids = {
                str(metadata.get("generation_id") or "")
                for result in results.values()
                for metadata in result.get("generation_metadata") or []
            }
            received_ids = {
                str(event.get("generation_id") or "")
                for event in attempt_events
                if event.get("event_type") == "response_received"
            }
            for generation_id in sorted(received_ids - accounted_ids - {""}):
                incomplete_generation_metadata.append(
                    await provider._generation_cost(generation_id)  # noqa: SLF001
                )
            await provider.aclose()
        after = await _key_status(catalog_client)
        journal.append("openrouter_key_status", {"position": "after", **after})

    total_cost_micros = sum(
        int(result.get("cost_micros") or 0) for result in results.values()
    ) + sum(int(item.get("cost_micros") or 0) for item in incomplete_generation_metadata)
    reconciliation_states = [
        *(bool(result.get("cost_reconciled")) for result in results.values()),
        *(bool(item.get("reconciled")) for item in incomplete_generation_metadata),
    ]
    all_reconciled = bool(reconciliation_states) and all(reconciliation_states)
    journal_descriptor = journal.finalize(
        {
            "status": "generation_complete" if not errors and all_reconciled else "failed",
            "condition_names": sorted(results),
            "error_keys": sorted(errors),
            "generation_ids": sorted(
                {
                    str(generation_id)
                    for result in results.values()
                    for generation_id in result.get("generation_ids") or []
                    if generation_id
                }
            ),
            "actual_cost_micros": total_cost_micros,
            "all_generation_costs_reconciled": all_reconciled,
        }
    )
    artifact: dict[str, Any] = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": run_id,
        "status": "complete" if not errors and all_reconciled else "failed_or_unreconciled",
        "run_class": "engineering_live_smoke",
        "run_purpose": run_purpose,
        "requested_conditions": list(selected_conditions),
        "candidate_manifest_sha256": candidate_manifest_sha256 or None,
        "dataset_work_item_id": dataset_work_item_id or None,
        "dataset_task_id": dataset_task_id or None,
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "category": args.category,
        "requested_model_id": args.model_id,
        "requested_provider": args.provider_slug,
        "model_contract": model,
        "endpoint_contract": endpoint,
        "endpoint_contract_sha256": _sha256(endpoint),
        "endpoint_execution_contract_sha256": endpoint_execution_contract_sha256(endpoint),
        "provider_routing_controls": provider_routing_controls,
        "provider_routing_controls_sha256": _sha256(provider_routing_controls),
        "frozen_generation_contract": {
            "supported_parameters": sorted(generation_contract["supported_parameters"]),
            "decoding_parameters": generation_contract["decoding_parameters"],
            "expected_actual_model_id": generation_contract["expected_actual_model_id"],
            "expected_actual_provider_slug": generation_contract["expected_actual_provider_slug"],
            "endpoint_contract_sha256": generation_contract["endpoint_contract_sha256"],
            "final_response_mode": generation_contract["final_response_mode"],
            "matched_planning": generation_contract["matched_planning"],
            "intermediate_max_tokens": generation_contract["intermediate_max_tokens"],
            "required_tool_contract_max_intermediate_tokens": generation_contract[
                "required_tool_contract_max_intermediate_tokens"
            ],
            "evidence_protocol": generation_contract["evidence_protocol"],
            "required_tool_contract_protocol": generation_contract[
                "required_tool_contract_protocol"
            ],
            "required_tool_contract_sha256": generation_contract["required_tool_contract_sha256"],
            "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
            "intermediate_reasoning_effort": generation_contract["intermediate_reasoning_effort"],
            "final_reasoning_effort": generation_contract["final_reasoning_effort"],
            "protocol_bundle_sha256": generation_contract["protocol_bundle_sha256"],
            "expected_epicure_release_id": generation_contract["expected_epicure_release_id"],
            "expected_epicure_bundle_sha256": generation_contract["expected_epicure_bundle_sha256"],
            "expected_epicure_application_sha256": generation_contract[
                "expected_epicure_application_sha256"
            ],
            "expected_epicure_tool_schema_sha256": generation_contract[
                "expected_epicure_tool_schema_sha256"
            ],
        },
        "protocol_bundle": protocol_bundle,
        "protocol_bundle_sha256": protocol_bundle_sha256,
        "required_tool_contract": required_tool_contract(execution_policy),
        "execution_policy": execution_policy.document(),
        "execution_policy_sha256": execution_policy.sha256,
        "decoding": {
            "temperature": settings.decoding_temperature,
            "top_p": settings.decoding_top_p,
            "seed": settings.decoding_seed,
            "max_output_tokens": settings.max_output_tokens,
            "max_tool_rounds": settings.max_tool_rounds,
            "max_tool_calls_per_round": settings.max_tool_calls_per_round,
            "max_tool_calls_total": settings.max_tool_calls_total,
            "max_tool_result_bytes": settings.max_tool_result_bytes,
            "max_cumulative_tool_result_bytes": (settings.max_cumulative_tool_result_bytes),
            "max_provider_attempts": settings.max_provider_attempts,
            "parallel_tool_calls_enforcement": "bounded_sequential_execution",
        },
        "system_prompt_sha256": {
            condition: system_prompt_sha256(
                condition,
                execution_policy.final_response_mode,
                execution_policy.evidence_protocol,
            )
            for condition in selected_conditions
        },
        "response_schema_sha256": response_schema_sha256(execution_policy.final_response_mode),
        "epicure": provenance,
        "epicure_tool_schema_sha256": tool_schema_sha256,
        "budget": {
            "cap_usd": str(args.cap_usd),
            "forecast_worst_case_usd": str(forecast),
            "actual_cost_micros": total_cost_micros,
            "all_generation_costs_reconciled": all_reconciled,
            "openrouter_key_before": before,
            "openrouter_key_after": after,
        },
        "results": results,
        "errors": errors,
        "provider_attempt_events": attempt_events,
        "mcp_trace_events": mcp_trace_events,
        "incomplete_generation_metadata": incomplete_generation_metadata,
        "run_journal": journal_descriptor.payload(),
        "limitations": [
            "This is an unranked engineering run and cannot enter any leaderboard.",
            "No human preference or expert judgment was collected.",
            "The Epicure bundle is explicitly an unmatched exploratory runtime release.",
            "A contract smoke is not a substitute for blinded, voted benchmark collection.",
            "The endpoint tag is request-enforced; returned accounting identifies the provider "
            "but may not independently attest a provider sub-region or serving tier.",
        ],
    }
    artifact["artifact_sha256"] = _sha256(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{started_at:%Y%m%dT%H%M%SZ}-{artifact['artifact_sha256'][:12]}.json"
    output_path = output_dir / filename
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {output_path}")
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    output_path.chmod(0o644)
    return {
        "status": artifact["status"],
        "rank_eligible": False,
        "artifact": str(output_path.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "actual_cost_micros": total_cost_micros,
        "conditions": sorted(results),
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real, permanently unranked OpenRouter + Epicure engineering calls."
    )
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("0"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--provider-slug", default=DEFAULT_PROVIDER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--category",
        choices=["substitution", "composition", "cookability", "evidence"],
        default="cookability",
    )
    parser.add_argument("--skip-tool-contract", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument(
        "--condition",
        action="append",
        choices=["epicure_off", "epicure_on"],
        help="Run only the selected condition; repeat for an explicit subset.",
    )
    parser.add_argument(
        "--plain-text-final",
        action="store_true",
        help="Collect a lossless natural-language answer instead of a JSON wrapper.",
    )
    parser.add_argument("--tool-catalog-bytes-bound", type=int, default=0)
    parser.add_argument(
        "--require-epicure-call",
        action="store_true",
        help="Require at least one successful real Epicure call in the Epicure-on arm.",
    )
    parser.add_argument(
        "--evidence-protocol",
        choices=["legacy_v6", *sorted(GOVERNED_EPICURE_PROTOCOLS)],
        default="legacy_v6",
    )
    parser.add_argument(
        "--intermediate-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument(
        "--final-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument("--output-dir", default="artifacts/live-smoke")
    parser.add_argument("--candidate-manifest-sha256", default="")
    parser.add_argument(
        "--sequential-arms",
        action="store_true",
        help="Run the Epicure-off arm to completion before starting the Epicure-on arm.",
    )
    parser.add_argument(
        "--dataset-work-item-id",
        default="",
        help="Optional immutable exploratory-dataset work-item SHA-256.",
    )
    parser.add_argument(
        "--dataset-task-id",
        default="",
        help="Optional candidate task ID associated with --dataset-work-item-id.",
    )
    parser.add_argument("--expected-canonical-model-slug", default="")
    parser.add_argument("--expected-endpoint-execution-sha256", default="")
    parser.add_argument("--expected-execution-policy-sha256", default="")
    parser.add_argument("--expected-epicure-release-id", default="")
    parser.add_argument("--expected-epicure-bundle-sha256", default="")
    parser.add_argument("--expected-epicure-application-sha256", default="")
    parser.add_argument("--expected-epicure-tool-schema-sha256", default="")
    return parser


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-live-smoke")
    try:
        summary = asyncio.run(live_smoke(_parser().parse_args()))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1) from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    run()
