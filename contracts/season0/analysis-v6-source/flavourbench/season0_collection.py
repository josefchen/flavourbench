"""Collect real FlavourBench Season 0 Epicure-off/on response arms."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Never

import httpx

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import project_bedrock_json_schema
from .mcp_client import McpSession, tool_catalog_sha256
from .real_task_bank import sha256_json, sha256_text

ARM_SCHEMA = "flavourbench-season0-real-response-arm-v1"
SUMMARY_SCHEMA = "flavourbench-season0-real-collection-summary-v1"
COLLECTOR_VERSION = "season0-real-collector-v2"
COLLECTOR_SOURCE_SHA256 = sha256_text(Path(__file__).read_text(encoding="utf-8"))
CALIBRATION_CONFIRMATION = "RUN_REAL_SEASON0_CALIBRATION_V1"
SCORED_CONFIRMATION = "RUN_REAL_SEASON0_SCORED_V1"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONDITIONS = ("epicure_off", "epicure_on")

BASE_SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary reasoning
benchmark. Answer the user's exact question. Prioritize explicit constraints, coherent flavour
logic, practical cookability, and calibrated claims. Give a complete answer in plain Markdown.
Be concise and finish cleanly; normally stay below 700 words.
Never identify your model, developer, or provider. Do not provide formal medical, allergen,
food-safety, or cultural-authenticity certification."""

EPICURE_SYSTEM_PROMPT = """You have read-only access to Epicure's culinary evidence tools. Use
them when they can materially improve the answer, and integrate results critically. Tool outputs
encode learned statistical relationships, not proof of chemistry, safety, authenticity, human
preference, or universal quality. Do not mention the benchmark condition or tool implementation."""


class CollectionError(RuntimeError):
    """The real collection contract or one of its immutable inputs was invalid."""


class SafePreInferenceError(CollectionError):
    """A provider explicitly rejected a request before a usable generation."""


class UncertainDeliveryError(CollectionError):
    """A request may have been accepted but no reconcilable response was received."""


class ReconciledArmFailure(CollectionError):
    """A paid response violated the contract but retains its complete partial trace."""

    def __init__(self, message: str, partial_result: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.partial_result = dict(partial_result)


def _unhandled_delivery_state(error: Exception) -> str:
    """Conservatively classify transport errors not wrapped by a provider adapter."""
    if type(error).__name__ in {
        "ConnectionClosedError",
        "ReadTimeoutError",
        "ResponseStreamingError",
    }:
        return "uncertain"
    return "safe_pre_inference"


@dataclass(frozen=True)
class WorkItem:
    arm_id: str
    phase: str
    task: Mapping[str, Any]
    model: Mapping[str, Any]
    condition: str


@dataclass
class BudgetGovernor:
    hard_caps: dict[str, Decimal]
    spent: dict[str, Decimal]
    active: dict[str, dict[str, Decimal]]
    lock: asyncio.Lock

    @classmethod
    def create(
        cls,
        *,
        bedrock_cap: Decimal,
        openrouter_cap: Decimal,
        initial_spent: Mapping[str, Decimal] | None = None,
    ) -> BudgetGovernor:
        initial = initial_spent or {}
        return cls(
            hard_caps={"bedrock": bedrock_cap, "openrouter": openrouter_cap},
            spent={
                "bedrock": Decimal(initial.get("bedrock", 0)),
                "openrouter": Decimal(initial.get("openrouter", 0)),
            },
            active={"bedrock": {}, "openrouter": {}},
            lock=asyncio.Lock(),
        )

    async def reserve(self, provider: str, arm_id: str, amount: Decimal) -> None:
        if amount <= 0:
            raise CollectionError("budget reservation must be positive")
        async with self.lock:
            projected = self.spent[provider] + sum(self.active[provider].values()) + amount
            if projected > self.hard_caps[provider] * Decimal("0.85"):
                raise CollectionError(f"{provider} admission reached 85% of its hard cap")
            self.active[provider][arm_id] = amount

    async def finalize(self, provider: str, arm_id: str, actual: Decimal | None) -> None:
        async with self.lock:
            reservation = self.active[provider].pop(arm_id, Decimal(0))
            charged = reservation if actual is None else actual
            self.spent[provider] += charged
            if self.spent[provider] > self.hard_caps[provider]:
                raise CollectionError(f"{provider} hard cap was exceeded")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CollectionError(f"expected a JSON object: {path}")
    return value


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise CollectionError(f"content-address conflict at {destination}")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _artifact_sha(document: Mapping[str, Any], *, label: str) -> str:
    claimed = document.get("artifact_sha256")
    if not isinstance(claimed, str):
        raise CollectionError(f"{label} has no artifact hash")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if sha256_json(body) != claimed:
        raise CollectionError(f"{label} artifact hash is invalid")
    return claimed


def _selected_tasks(task_bank: Mapping[str, Any], per_family: int) -> list[Mapping[str, Any]]:
    tasks = task_bank.get("tasks")
    if not isinstance(tasks, list) or task_bank.get("synthetic_tasks") != 0:
        raise CollectionError("real Season 0 task bank is invalid")
    selected: list[Mapping[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [
            task for task in tasks if isinstance(task, Mapping) and task.get("family") == family
        ]
        if len(family_tasks) < per_family:
            raise CollectionError(f"task bank has fewer than {per_family} tasks for {family}")
        if per_family == 1:
            selected.append(family_tasks[len(family_tasks) // 2])
        else:
            selected.extend(family_tasks[:per_family])
    return selected


def build_work_items(
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    *,
    phase: str,
    per_family: int,
) -> list[WorkItem]:
    tasks = _selected_tasks(task_bank, per_family)
    models = model_manifest.get("models")
    if not isinstance(models, list) or len(models) != 12:
        raise CollectionError("Season 0 requires an exact 12-model manifest")
    if task_bank.get("task_set_sha256") != model_manifest.get("task_set_sha256"):
        raise CollectionError("task and model manifests bind different task sets")
    model_set_sha = str(model_manifest.get("model_set_sha256") or "")
    task_set_sha = str(task_bank.get("task_set_sha256") or "")
    execution_contract_sha = sha256_json(model_manifest.get("execution_contract") or {})
    output: list[WorkItem] = []
    for task in tasks:
        for model in models:
            if not isinstance(model, Mapping):
                raise CollectionError("model manifest contains an invalid model")
            for condition in CONDITIONS:
                identity = {
                    "schema_version": ARM_SCHEMA,
                    "phase": phase,
                    "task_set_sha256": task_set_sha,
                    "model_set_sha256": model_set_sha,
                    "execution_contract_sha256": execution_contract_sha,
                    "epicure_intervention_artifact_sha256": model_manifest.get(
                        "epicure_intervention_artifact_sha256"
                    ),
                    "task_id": task["task_id"],
                    "task_sha256": task["task_sha256"],
                    "season_model_id": model["season_model_id"],
                    "canonical_model_id": model["canonical_model_id"],
                    "condition": condition,
                    "system_prompt_sha256": sha256_text(
                        BASE_SYSTEM_PROMPT
                        + (
                            "\n\n" + EPICURE_SYSTEM_PROMPT
                            if condition == "epicure_on"
                            else ""
                        )
                    ),
                    "seed": 20260716,
                }
                output.append(
                    WorkItem(
                        arm_id=sha256_json(identity),
                        phase=phase,
                        task=task,
                        model=model,
                        condition=condition,
                    )
                )
    output.sort(key=lambda item: sha256_text(f"season0-order:{item.arm_id}"))
    if len({item.arm_id for item in output}) != len(output):
        raise CollectionError("work-item arm IDs are not unique")
    return output


def _answer_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, str | bytes):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping)
        ).strip()
    return ""


def _bedrock_message(response: Mapping[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
        raise CollectionError("Bedrock returned no assistant message")
    return {"role": str(message.get("role") or "assistant"), "content": message["content"]}


def _usage_bedrock(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise CollectionError("Bedrock returned no usage")
    return {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
        "total_tokens": int(usage.get("totalTokens") or 0),
        "cache_read_input_tokens": int(usage.get("cacheReadInputTokens") or 0),
        "cache_write_input_tokens": int(usage.get("cacheWriteInputTokens") or 0),
    }


def _bedrock_result(
    *,
    started: float,
    final_answer: str,
    finish_reason: str,
    usages: Sequence[Mapping[str, int]],
    traces: Sequence[Mapping[str, Any]],
    response_latencies: Sequence[int],
    request_id_sha256s: Sequence[str],
    request_payload_sha256s: Sequence[str],
    provider_models: Sequence[str],
) -> dict[str, Any]:
    usage = {
        key: sum(int(part.get(key) or 0) for part in usages)
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        )
    }
    return {
        "answer_markdown": final_answer,
        "finish_reason": finish_reason,
        "usage": usage,
        "provider_calls": len(response_latencies),
        "tool_trace": list(traces),
        "real_epicure_calls": len(traces),
        "request_id_sha256s": list(request_id_sha256s),
        "request_payload_sha256s": list(request_payload_sha256s),
        "returned_model_ids": list(provider_models),
        "actual_provider_name": "Amazon Bedrock",
        "actual_cost_usd": None,
        "cost_status": "aws_usage_captured_rate_card_reconciliation_pending",
        "response_latencies_ms": list(response_latencies),
        "wall_clock_latency_ms": round((time.monotonic() - started) * 1000),
    }


def _bounded_text(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    bounded = encoded[:limit]
    while bounded:
        try:
            value = bounded.decode("utf-8")
            return value + "\n[Tool result truncated by the frozen FlavourBench contract.]"
        except UnicodeDecodeError:
            bounded = bounded[:-1]
    return "[Tool result omitted after UTF-8 boundary failure.]"


async def _execute_tool(
    mcp: McpSession,
    *,
    round_index: int,
    name: str,
    arguments: Mapping[str, Any],
    result_byte_limit: int,
) -> tuple[dict[str, Any], object, bool]:
    started = time.monotonic()
    try:
        result = await mcp.call_tool(name, dict(arguments))
        full_value: object = result.structured or result.text
        full_text = result.text or json.dumps(result.structured, ensure_ascii=False, sort_keys=True)
        is_error = result.is_error
        latency_ms = result.latency_ms
    except Exception as error:  # noqa: BLE001 - return one repairable tool error to the model
        full_value = {
            "error": type(error).__name__,
            "message": str(error)[:400],
        }
        full_text = json.dumps(full_value, ensure_ascii=False)
        is_error = True
        latency_ms = round((time.monotonic() - started) * 1000)
    trace = {
        "round_index": round_index,
        "name": name,
        "arguments": dict(arguments),
        "arguments_sha256": sha256_json(arguments),
        "result": full_value,
        "result_sha256": sha256_json(full_value),
        "latency_ms": latency_ms,
        "is_error": is_error,
        "model_visible_result": _bounded_text(full_text, result_byte_limit),
        "model_visible_result_sha256": sha256_text(_bounded_text(full_text, result_byte_limit)),
    }
    return trace, full_value, is_error


def _bedrock_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for tool in tools:
        schema = tool.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise CollectionError("Epicure tool has no input schema")
        output.append(
            {
                "toolSpec": {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description") or "Epicure evidence tool"),
                    "inputSchema": {"json": project_bedrock_json_schema(schema)},
                }
            }
        )
    return output


async def _run_bedrock_arm(
    runtime: Any,
    item: WorkItem,
    tools: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    system = BASE_SYSTEM_PROMPT + (
        "\n\n" + EPICURE_SYSTEM_PROMPT if item.condition == "epicure_on" else ""
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": str(item.task["prompt"])}]}
    ]
    bedrock_tools = _bedrock_tools(tools) if item.condition == "epicure_on" else []
    usages: list[dict[str, int]] = []
    traces: list[dict[str, Any]] = []
    response_latencies: list[int] = []
    request_id_sha256s: list[str] = []
    request_payload_sha256s: list[str] = []
    provider_models: list[str] = []
    finish_reason = "unknown"
    final_answer = ""
    tool_error_repairs = 0
    total_tool_calls = 0
    cumulative_tool_bytes = 0
    max_rounds = int(contract["max_tool_rounds"])

    try:
        async with McpSession() if item.condition == "epicure_on" else _NullMcp() as mcp:
            for round_index in range(max_rounds + 1):
                request: dict[str, Any] = {
                "modelId": item.model["requested_endpoint_id"],
                "system": [{"text": system}],
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": int(contract["final_answer_max_tokens"]),
                    "temperature": 0.2,
                },
                "requestMetadata": {
                    "flavourbench_phase": item.phase,
                    "flavourbench_arm": item.arm_id[:32],
                    "flavourbench_condition": item.condition,
                },
            }
                if bedrock_tools:
                    request["toolConfig"] = {
                        "tools": bedrock_tools,
                        "toolChoice": {"auto": {}},
                    }
                request_payload_sha256s.append(sha256_json(request))
                call_started = time.monotonic()
                response = await asyncio.to_thread(runtime.converse, **request)
                response_latencies.append(
                    round((time.monotonic() - call_started) * 1000)
                )
                usages.append(_usage_bedrock(response))
                metadata = response.get("ResponseMetadata")
                request_id = (
                    metadata.get("RequestId") if isinstance(metadata, Mapping) else None
                )
                if request_id:
                    request_id_sha256s.append(sha256_text(str(request_id)))
                returned_model = response.get("modelId") or response.get("model")
                if isinstance(returned_model, str) and returned_model not in provider_models:
                    provider_models.append(returned_model)
                assistant = _bedrock_message(response)
                messages.append(assistant)
                finish_reason = str(response.get("stopReason") or "unknown")
                if finish_reason != "tool_use":
                    final_answer = _answer_text(assistant)
                    break
                if item.condition != "epicure_on" or round_index >= max_rounds:
                    raise CollectionError("Bedrock exhausted the Epicure tool-round cap")
                content = assistant["content"]
                calls = [
                    block["toolUse"]
                    for block in content
                    if isinstance(block, Mapping)
                    and isinstance(block.get("toolUse"), Mapping)
                ]
                if not calls or len(calls) > int(contract["max_tool_calls_per_round"]):
                    raise CollectionError(
                        "Bedrock tool fan-out violated the frozen contract"
                    )
                total_tool_calls += len(calls)
                if total_tool_calls > int(contract["max_tool_calls_total"]):
                    raise CollectionError("Bedrock exceeded the total tool-call cap")
                result_blocks: list[dict[str, Any]] = []
                assert isinstance(mcp, McpSession)
                for call in calls:
                    name = str(call.get("name") or "")
                    arguments = call.get("input")
                    if not name or not isinstance(arguments, Mapping):
                        arguments = {}
                    trace, _, is_error = await _execute_tool(
                        mcp,
                        round_index=round_index,
                        name=name,
                        arguments=arguments,
                        result_byte_limit=int(contract["max_tool_result_bytes"]),
                    )
                    traces.append(trace)
                    cumulative_tool_bytes += len(
                        trace["model_visible_result"].encode("utf-8")
                    )
                    if cumulative_tool_bytes > int(
                        contract["max_cumulative_tool_result_bytes"]
                    ):
                        raise CollectionError(
                            "Epicure evidence exceeded the cumulative byte cap"
                        )
                    if is_error:
                        tool_error_repairs += 1
                        if tool_error_repairs > 1:
                            raise CollectionError(
                                "tool arguments remained invalid after one repair"
                            )
                    result_blocks.append(
                        {
                            "toolResult": {
                                "toolUseId": str(call.get("toolUseId") or ""),
                                "content": [{"text": trace["model_visible_result"]}],
                                **({"status": "error"} if is_error else {}),
                            }
                        }
                    )
                messages.append({"role": "user", "content": result_blocks})
        if not final_answer:
            raise CollectionError("Bedrock returned an empty final answer")
    except CollectionError as error:
        partial = _bedrock_result(
            started=started,
            final_answer=final_answer,
            finish_reason=finish_reason,
            usages=usages,
            traces=traces,
            response_latencies=response_latencies,
            request_id_sha256s=request_id_sha256s,
            request_payload_sha256s=request_payload_sha256s,
            provider_models=provider_models,
        )
        if response_latencies:
            raise ReconciledArmFailure(str(error), partial) from error
        raise
    return _bedrock_result(
        started=started,
        final_answer=final_answer,
        finish_reason=finish_reason,
        usages=usages,
        traces=traces,
        response_latencies=response_latencies,
        request_id_sha256s=request_id_sha256s,
        request_payload_sha256s=request_payload_sha256s,
        provider_models=provider_models,
    )


class _NullMcp:
    async def __aenter__(self) -> _NullMcp:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


def _openrouter_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description") or "Epicure evidence tool"),
                "parameters": tool["inputSchema"],
            },
        }
        for tool in tools
    ]


def _or_token_limit(model: Mapping[str, Any], value: int) -> dict[str, int]:
    supported = set(model["endpoint"].get("supported_parameters") or [])
    return {
        "max_tokens" if "max_tokens" in supported else "max_completion_tokens": value
    }


def _or_message(response: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        if response.get("id"):
            raise UncertainDeliveryError("OpenRouter returned a generation without a choice")
        error = response.get("error")
        message = str(error.get("message") or "no choice") if isinstance(error, Mapping) else ""
        raise SafePreInferenceError(f"OpenRouter rejected the request: {message[:300]}")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise UncertainDeliveryError("OpenRouter choice contains no assistant message")
    normalized = {
        "role": "assistant",
        "content": message.get("content"),
        **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {}),
    }
    return choices[0], normalized


async def _post_openrouter(
    client: httpx.AsyncClient,
    payload: Mapping[str, Any],
    arm_id: str,
    call_index: int,
) -> Mapping[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await client.post(
                "chat/completions",
                json=dict(payload),
                headers={
                    "Idempotency-Key": sha256_text(f"{arm_id}:{call_index}"),
                    "cf-aig-metadata": json.dumps(
                        {
                            "benchmark": "flavourbench-season0",
                            "arm_sha256": arm_id[:32],
                            "call_index": call_index,
                        },
                        separators=(",", ":"),
                    ),
                },
            )
            if response.status_code in {429, 503}:
                last_error = SafePreInferenceError(f"OpenRouter HTTP {response.status_code}")
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, Mapping):
                raise UncertainDeliveryError("OpenRouter returned a non-object response")
            return value
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            last_error = error
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
        except httpx.ReadTimeout as error:
            raise UncertainDeliveryError(
                "OpenRouter read timeout after possible acceptance"
            ) from error
        except httpx.HTTPStatusError as error:
            raise SafePreInferenceError(f"OpenRouter HTTP {error.response.status_code}") from error
    raise SafePreInferenceError(
        f"OpenRouter pre-inference request failure: {type(last_error).__name__}"
    )


async def _or_accounting(
    client: httpx.AsyncClient, generation_id: str, attempts: int = 7
) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            response = await client.get("generation", params={"id": generation_id})
            response.raise_for_status()
            data = response.json().get("data")
            if isinstance(data, Mapping) and "total_cost" in data:
                return {
                    "generation_id": generation_id,
                    "total_cost_usd": str(data.get("total_cost") or "0"),
                    "provider_name": str(data.get("provider_name") or ""),
                    "model": str(data.get("model") or ""),
                    "tokens_prompt": int(data.get("tokens_prompt") or 0),
                    "tokens_completion": int(data.get("tokens_completion") or 0),
                    "native_tokens_prompt": int(data.get("native_tokens_prompt") or 0),
                    "native_tokens_completion": int(data.get("native_tokens_completion") or 0),
                    "generation_time_ms": int(data.get("generation_time") or 0),
                    "upstream_latency_ms": int(data.get("latency") or 0),
                    "reconciled": True,
                }
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            pass
        if attempt + 1 < attempts:
            await asyncio.sleep(0.5 * (2**attempt))
    raise UncertainDeliveryError(f"OpenRouter cost did not reconcile: {generation_id}")


async def _finalize_openrouter_result(
    *,
    accounting_client: httpx.AsyncClient,
    item: WorkItem,
    responses: Sequence[Mapping[str, Any]],
    request_payload_sha256s: Sequence[str],
    traces: Sequence[Mapping[str, Any]],
    final_answer: str,
    finish_reason: str,
    started: float,
) -> dict[str, Any]:
    generation_ids = [str(response.get("id") or "") for response in responses]
    if not generation_ids or any(not value for value in generation_ids):
        raise UncertainDeliveryError("OpenRouter response omitted a generation ID")
    accounting = await asyncio.gather(
        *(
            _or_accounting(accounting_client, generation_id)
            for generation_id in generation_ids
        )
    )
    model_identity_verified = all(
        entry["model"] == item.model["canonical_model_id"] for entry in accounting
    )
    provider_identity_verified = all(
        entry["provider_name"] == item.model["provider_name"] for entry in accounting
    )
    actual_cost = sum(
        (Decimal(str(entry["total_cost_usd"])) for entry in accounting), Decimal(0)
    )
    usage = {
        "input_tokens": sum(
            int((response.get("usage") or {}).get("prompt_tokens") or 0)
            for response in responses
        ),
        "output_tokens": sum(
            int((response.get("usage") or {}).get("completion_tokens") or 0)
            for response in responses
        ),
        "reasoning_tokens": sum(
            int((response.get("usage") or {}).get("reasoning_tokens") or 0)
            for response in responses
        ),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return {
        "answer_markdown": final_answer,
        "finish_reason": finish_reason,
        "usage": usage,
        "provider_calls": len(responses),
        "tool_trace": list(traces),
        "real_epicure_calls": len(traces),
        "generation_ids": generation_ids,
        "generation_accounting": accounting,
        "request_payload_sha256s": list(request_payload_sha256s),
        "returned_model_ids": sorted({entry["model"] for entry in accounting}),
        "actual_provider_names": sorted(
            {entry["provider_name"] for entry in accounting}
        ),
        "actual_provider_name": (
            accounting[0]["provider_name"] if accounting else ""
        ),
        "model_identity_verified": model_identity_verified,
        "provider_identity_verified": provider_identity_verified,
        "actual_cost_usd": format(actual_cost, "f"),
        "cost_status": "openrouter_generation_metadata_reconciled",
        "wall_clock_latency_ms": round((time.monotonic() - started) * 1000),
    }


async def _run_openrouter_arm(
    generation_client: httpx.AsyncClient,
    accounting_client: httpx.AsyncClient,
    item: WorkItem,
    tools: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    system = BASE_SYSTEM_PROMPT + (
        "\n\n" + EPICURE_SYSTEM_PROMPT if item.condition == "epicure_on" else ""
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": str(item.task["prompt"])},
    ]
    provider = {
        "only": [item.model["provider_slug"]],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    available_tools = _openrouter_tools(tools) if item.condition == "epicure_on" else []
    responses: list[Mapping[str, Any]] = []
    request_payload_sha256s: list[str] = []
    traces: list[dict[str, Any]] = []
    final_answer = ""
    finish_reason = "unknown"
    total_tool_calls = 0
    tool_error_repairs = 0
    cumulative_tool_bytes = 0
    max_rounds = int(contract["max_tool_rounds"])

    async def reconciled_failure(message: str) -> Never:
        partial = await _finalize_openrouter_result(
            accounting_client=accounting_client,
            item=item,
            responses=responses,
            request_payload_sha256s=request_payload_sha256s,
            traces=traces,
            final_answer=final_answer,
            finish_reason=finish_reason,
            started=started,
        )
        raise ReconciledArmFailure(message, partial)

    async with McpSession() if item.condition == "epicure_on" else _NullMcp() as mcp:
        for round_index in range(max_rounds + 1):
            payload: dict[str, Any] = {
                "model": item.model["requested_endpoint_id"],
                "messages": messages,
                **_or_token_limit(item.model, int(contract["final_answer_max_tokens"])),
                "provider": provider,
                "usage": {"include": True},
            }
            supported = set(item.model["endpoint"].get("supported_parameters") or [])
            if "temperature" in supported:
                payload["temperature"] = 0.2
            if available_tools:
                payload["tools"] = available_tools
                payload["tool_choice"] = "auto"
            request_payload_sha256s.append(sha256_json(payload))
            response = await _post_openrouter(
                generation_client, payload, item.arm_id, len(responses)
            )
            responses.append(response)
            choice, assistant = _or_message(response)
            finish_reason = str(choice.get("finish_reason") or "unknown")
            calls = assistant.get("tool_calls") or []
            if not calls:
                final_answer = _answer_text(assistant)
                break
            if item.condition != "epicure_on" or round_index >= max_rounds:
                await reconciled_failure(
                    "OpenRouter exhausted the Epicure tool-round cap"
                )
            if not isinstance(calls, list) or len(calls) > int(
                contract["max_tool_calls_per_round"]
            ):
                await reconciled_failure(
                    "OpenRouter tool fan-out violated the frozen contract"
                )
            total_tool_calls += len(calls)
            if total_tool_calls > int(contract["max_tool_calls_total"]):
                await reconciled_failure("OpenRouter exceeded the total tool-call cap")
            messages.append(assistant)
            assert isinstance(mcp, McpSession)
            for call in calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
                raw_arguments = function.get("arguments") if isinstance(function, Mapping) else "{}"
                try:
                    arguments = json.loads(str(raw_arguments or "{}"))
                    if not isinstance(arguments, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
                trace, _, is_error = await _execute_tool(
                    mcp,
                    round_index=round_index,
                    name=name,
                    arguments=arguments,
                    result_byte_limit=int(contract["max_tool_result_bytes"]),
                )
                traces.append(trace)
                cumulative_tool_bytes += len(
                    trace["model_visible_result"].encode("utf-8")
                )
                if cumulative_tool_bytes > int(contract["max_cumulative_tool_result_bytes"]):
                    await reconciled_failure(
                        "Epicure evidence exceeded the cumulative byte cap"
                    )
                if is_error:
                    tool_error_repairs += 1
                    if tool_error_repairs > 1:
                        await reconciled_failure(
                            "tool arguments remained invalid after one repair"
                        )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "name": name,
                        "content": trace["model_visible_result"],
                    }
                )
    if not final_answer:
        await reconciled_failure("OpenRouter returned an empty final answer")
    result = await _finalize_openrouter_result(
        accounting_client=accounting_client,
        item=item,
        responses=responses,
        request_payload_sha256s=request_payload_sha256s,
        traces=traces,
        final_answer=final_answer,
        finish_reason=finish_reason,
        started=started,
    )
    if not result["model_identity_verified"]:
        raise ReconciledArmFailure(
            "OpenRouter returned a different canonical model", result
        )
    if not result["provider_identity_verified"]:
        raise ReconciledArmFailure("OpenRouter returned a different provider", result)
    return result


def _existing_collection_state(
    arm_directory: Path,
    event_directory: Path,
) -> tuple[set[str], set[str], dict[str, Decimal]]:
    terminal: set[str] = set()
    latest_by_arm: dict[str, dict[str, Any]] = {}
    initial_spent = {"bedrock": Decimal(0), "openrouter": Decimal(0)}
    if not arm_directory.exists() and not event_directory.exists():
        return terminal, set(), initial_spent
    for path in arm_directory.glob("arm-*.json"):
        try:
            document = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        arm_id = document.get("arm_id")
        if not isinstance(arm_id, str):
            continue
        prior = latest_by_arm.get(arm_id)
        if prior is None or str(document.get("completed_at") or "") > str(
            prior.get("completed_at") or ""
        ):
            latest_by_arm[arm_id] = document
    for arm_id, document in latest_by_arm.items():
        terminal.add(arm_id)
        model = document.get("model")
        provider = str(model.get("provider") or "") if isinstance(model, Mapping) else ""
        if provider not in initial_spent:
            continue
        result = document.get("result")
        actual = result.get("actual_cost_usd") if isinstance(result, Mapping) else None
        amount = (
            Decimal(str(actual))
            if actual is not None
            else Decimal(str(document.get("reservation_usd") or "0"))
        )
        initial_spent[provider] += amount

    started_events: dict[str, list[dict[str, Any]]] = {}
    for path in event_directory.glob("event-*-request-started-*.json"):
        try:
            event = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        arm_id = event.get("arm_id")
        if isinstance(arm_id, str):
            started_events.setdefault(arm_id, []).append(event)
    orphaned = set(started_events) - terminal
    for arm_id in orphaned:
        for event in started_events[arm_id]:
            provider = str(event.get("provider") or "")
            if provider in initial_spent:
                initial_spent[provider] += Decimal(
                    str(event.get("reservation_usd") or "0")
                )
    return terminal, orphaned, initial_spent


def _reservation_for(
    item: WorkItem,
    cost_envelope: Mapping[str, Any] | None = None,
) -> Decimal:
    if cost_envelope is not None:
        models = cost_envelope.get("models")
        model = (
            models.get(item.model["season_model_id"])
            if isinstance(models, Mapping)
            else None
        )
        if not isinstance(model, Mapping):
            raise CollectionError("cost envelope has no reservation for a season model")
        return Decimal(str(model["scored_arm_reservation_usd"]))
    if item.model["provider"] == "bedrock":
        return Decimal("1")
    smoke_cost = Decimal(str(item.model.get("compatibility_cost_usd") or "0"))
    multiplier = Decimal("3") if item.condition == "epicure_on" else Decimal("1.5")
    return max(Decimal("0.10"), smoke_cost * multiplier)


async def collect(
    *,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    epicure_intervention: Mapping[str, Any],
    output_dir: Path,
    phase: str,
    per_family: int,
    runtime: Any,
    openrouter_base_url: str,
    openrouter_api_key: str,
    cloudflare_gateway_token: str,
    bedrock_cap: Decimal,
    openrouter_cap: Decimal,
    bedrock_concurrency: int,
    openrouter_concurrency: int,
    cost_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_bank_sha = _artifact_sha(task_bank, label="task bank")
    model_manifest_sha = _artifact_sha(model_manifest, label="model manifest")
    epicure_sha = _artifact_sha(epicure_intervention, label="Epicure intervention")
    cost_envelope_sha: str | None = None
    if phase == "scored":
        if cost_envelope is None:
            raise CollectionError("scored collection requires a frozen cost envelope")
        cost_envelope_sha = _artifact_sha(cost_envelope, label="cost envelope")
        if (
            cost_envelope.get("status") != "frozen_for_scored_admission"
            or cost_envelope.get("model_manifest_artifact_sha256")
            != model_manifest_sha
            or cost_envelope.get("execution_contract_sha256")
            != sha256_json(model_manifest["execution_contract"])
            or cost_envelope.get("forecast_within_admission_caps") is not True
        ):
            raise CollectionError("cost envelope is not bound and safe for this scored run")
    elif cost_envelope is not None:
        raise CollectionError("calibration cannot consume a scored cost envelope")
    if model_manifest.get("epicure_intervention_artifact_sha256") != epicure_sha:
        raise CollectionError("model manifest binds a different Epicure intervention")
    tools = epicure_intervention.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, Mapping) for tool in tools):
        raise CollectionError("Epicure intervention contains no frozen tool catalog")
    async with McpSession() as mcp:
        live_tools = await mcp.list_tools()
    if tool_catalog_sha256(live_tools) != epicure_intervention["runtime"]["tool_catalog_sha256"]:
        raise CollectionError("live Epicure tool catalog drifted before collection")

    contract = model_manifest["execution_contract"]
    work_items = build_work_items(
        task_bank, model_manifest, phase=phase, per_family=per_family
    )
    arm_dir = output_dir / "arms"
    event_dir = output_dir / "events"
    terminal_existing, orphaned_request_events, initial_spent = (
        _existing_collection_state(arm_dir, event_dir)
    )
    pending = [
        item
        for item in work_items
        if item.arm_id not in terminal_existing
        and item.arm_id not in orphaned_request_events
    ]
    governor = BudgetGovernor.create(
        bedrock_cap=bedrock_cap,
        openrouter_cap=openrouter_cap,
        initial_spent=initial_spent,
    )
    bedrock_semaphore = asyncio.Semaphore(bedrock_concurrency)
    openrouter_semaphore = asyncio.Semaphore(openrouter_concurrency)
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": f"Epicure FlavourBench Season 0 {phase}",
    }
    if "gateway.ai.cloudflare.com" in openrouter_base_url:
        if not cloudflare_gateway_token:
            raise CollectionError("Cloudflare AI Gateway token is required")
        headers.update(
            {
                "cf-aig-authorization": f"Bearer {cloudflare_gateway_token}",
                "cf-aig-skip-cache": "true",
                "cf-aig-collect-log-payload": "false",
            }
        )
    accounting_headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Accept": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": f"Epicure FlavourBench Season 0 {phase}",
    }
    records: list[dict[str, Any]] = []
    records_lock = asyncio.Lock()

    async with (
        httpx.AsyncClient(
            base_url=openrouter_base_url.rstrip("/") + "/",
            headers=headers,
            timeout=240,
        ) as generation_client,
        httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1/",
            headers=accounting_headers,
            timeout=120,
        ) as accounting_client,
    ):

        async def one(item: WorkItem) -> None:
            provider = str(item.model["provider"])
            reservation = _reservation_for(item, cost_envelope)
            semaphore = (
                bedrock_semaphore if provider == "bedrock" else openrouter_semaphore
            )
            started = _utc_now()
            status = "not_admitted"
            delivery_state = "not_sent"
            result: dict[str, Any] | None = None
            error_type: str | None = None
            error_message: str | None = None
            actual_cost: Decimal | None = None
            admitted = False
            async with semaphore:
                try:
                    await governor.reserve(provider, item.arm_id, reservation)
                    admitted = True
                except CollectionError as error:
                    error_type = type(error).__name__
                    error_message = str(error)[:800]
                if admitted:
                    event = {
                        "schema_version": "flavourbench-season0-request-event-v1",
                        "event": "request_started",
                        "arm_id": item.arm_id,
                        "phase": phase,
                        "provider": provider,
                        "reservation_usd": format(reservation, "f"),
                        "recorded_at": _utc_now(),
                    }
                    _atomic_write(
                        event_dir, f"event-{item.arm_id}-request-started", event
                    )
                    started = _utc_now()
                    status = "failed"
                    delivery_state = "safe_pre_inference"
                    try:
                        if provider == "bedrock":
                            result = await _run_bedrock_arm(
                                runtime, item, tools, contract
                            )
                        else:
                            result = await _run_openrouter_arm(
                                generation_client,
                                accounting_client,
                                item,
                                tools,
                                contract,
                            )
                        status = "success"
                        delivery_state = "reconciled"
                        if result.get("actual_cost_usd") is not None:
                            actual_cost = Decimal(str(result["actual_cost_usd"]))
                    except UncertainDeliveryError as error:
                        delivery_state = "uncertain"
                        error_type = type(error).__name__
                        error_message = str(error)[:800]
                    except SafePreInferenceError as error:
                        error_type = type(error).__name__
                        error_message = str(error)[:800]
                    except ReconciledArmFailure as error:
                        delivery_state = "reconciled"
                        result = error.partial_result
                        if result.get("actual_cost_usd") is not None:
                            actual_cost = Decimal(str(result["actual_cost_usd"]))
                        error_type = type(error).__name__
                        error_message = str(error)[:800]
                    except CollectionError as error:
                        # The provider returned a usable, billable response, but the arm
                        # violated the frozen execution contract (for example the tool
                        # round cap). This is a reconciled invalid response, not a
                        # pre-inference transport failure.
                        delivery_state = "reconciled"
                        error_type = type(error).__name__
                        error_message = str(error)[:800]
                    except Exception as error:  # noqa: BLE001 - persist redacted arm failure
                        delivery_state = _unhandled_delivery_state(error)
                        error_type = type(error).__name__
                        error_message = re.sub(
                            r"(?<!\d)\d{12}(?!\d)", "<account-redacted>", str(error)
                        )[:800]
                    await governor.finalize(provider, item.arm_id, actual_cost)
            record: dict[str, Any] = {
                "schema_version": ARM_SCHEMA,
                "status": status,
                "delivery_state": delivery_state,
                "arm_id": item.arm_id,
                "phase": phase,
                "condition": item.condition,
                "synthetic": False,
                "task": {
                    "task_id": item.task["task_id"],
                    "family": item.task["family"],
                    "task_sha256": item.task["task_sha256"],
                    "prompt": item.task["prompt"],
                    "prompt_sha256": item.task["prompt_sha256"],
                    "source_question_id": item.task["source_question_id"],
                },
                "model": {
                    "season_model_id": item.model["season_model_id"],
                    "display_name": item.model["display_name"],
                    "canonical_model_id": item.model["canonical_model_id"],
                    "provider": provider,
                    "requested_endpoint_id": item.model["requested_endpoint_id"],
                    "provider_slug": item.model.get("provider_slug"),
                    "compatibility_artifact_sha256": item.model[
                        "compatibility_artifact_sha256"
                    ],
                },
                "contracts": {
                    "task_bank_artifact_sha256": task_bank_sha,
                    "model_manifest_artifact_sha256": model_manifest_sha,
                    "task_set_sha256": task_bank["task_set_sha256"],
                    "model_set_sha256": model_manifest["model_set_sha256"],
                    "epicure_intervention_artifact_sha256": epicure_sha,
                    "epicure_release_id": epicure_intervention["runtime"]["release_id"],
                    "epicure_bundle_sha256": epicure_intervention["runtime"]["bundle_sha256"],
                    "epicure_application_sha256": epicure_intervention["runtime"][
                        "application_sha256"
                    ],
                    "epicure_tool_catalog_sha256": epicure_intervention["runtime"][
                        "tool_catalog_sha256"
                    ],
                    "system_prompt_sha256": sha256_text(
                        BASE_SYSTEM_PROMPT
                        + ("\n\n" + EPICURE_SYSTEM_PROMPT if item.condition == "epicure_on" else "")
                    ),
                    "execution_contract_sha256": sha256_json(contract),
                    "normalization_mode": "lossless_client_text_wrapper_v1",
                    "collector_version": COLLECTOR_VERSION,
                    "collector_source_sha256": COLLECTOR_SOURCE_SHA256,
                    "cost_envelope_artifact_sha256": cost_envelope_sha,
                },
                "reservation_usd": format(reservation, "f"),
                "started_at": started,
                "completed_at": _utc_now(),
                "result": result,
                "error_type": error_type,
                "error": error_message,
                "rank_eligible": status == "success" and delivery_state == "reconciled",
            }
            path = _atomic_write(arm_dir, f"arm-{item.arm_id}", record)
            record["path"] = str(path)
            async with records_lock:
                records.append(record)

        await asyncio.gather(*(one(item) for item in pending))

    all_records: list[dict[str, Any]] = []
    for path in arm_dir.glob("arm-*.json"):
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("arm_id") in {item.arm_id for item in work_items}:
            all_records.append(value)
    latest_by_arm: dict[str, dict[str, Any]] = {}
    for record in sorted(all_records, key=lambda value: str(value.get("completed_at") or "")):
        latest_by_arm[str(record["arm_id"])] = record
    final_records = list(latest_by_arm.values())
    successes = [record for record in final_records if record.get("status") == "success"]
    not_admitted_count = sum(
        record.get("status") == "not_admitted" for record in final_records
    )
    if orphaned_request_events or len(final_records) != len(work_items):
        collection_status = "incomplete_uncertain_delivery"
    elif not_admitted_count:
        collection_status = "budget_admission_closed"
    else:
        collection_status = "collection_complete"
    actual_or_cost = sum(
        (
            Decimal(str(record["result"]["actual_cost_usd"]))
            for record in successes
            if record["model"]["provider"] == "openrouter"
            and record.get("result", {}).get("actual_cost_usd") is not None
        ),
        Decimal(0),
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "phase": phase,
        "status": collection_status,
        "synthetic_arms": 0,
        "task_bank_artifact_sha256": task_bank_sha,
        "model_manifest_artifact_sha256": model_manifest_sha,
        "epicure_intervention_artifact_sha256": epicure_sha,
        "cost_envelope_artifact_sha256": cost_envelope_sha,
        "counts": {
            "planned_arms": len(work_items),
            "terminal_arms": len(final_records),
            "success": len(successes),
            "failed": sum(record.get("status") == "failed" for record in final_records),
            "not_admitted": not_admitted_count,
            "orphaned_request_events": len(orphaned_request_events),
            "uncertain": sum(
                record.get("delivery_state") == "uncertain" for record in final_records
            ),
            "by_provider": dict(Counter(record["model"]["provider"] for record in successes)),
            "by_condition": dict(Counter(record["condition"] for record in successes)),
            "real_epicure_calls": sum(
                int(record["result"].get("real_epicure_calls") or 0) for record in successes
            ),
            "real_provider_calls": sum(
                int(record["result"].get("provider_calls") or 0) for record in successes
            ),
        },
        "cost": {
            "openrouter_actual_reconciled_usd": format(actual_or_cost, "f"),
            "bedrock_budget_exposure_usd": format(governor.spent["bedrock"], "f"),
            "openrouter_budget_exposure_usd": format(governor.spent["openrouter"], "f"),
            "bedrock_hard_cap_usd": format(bedrock_cap, "f"),
            "openrouter_hard_cap_usd": format(openrouter_cap, "f"),
        },
        "models": {
            model_id: {
                "success": sum(
                    record.get("status") == "success"
                    for record in final_records
                    if record["model"]["season_model_id"] == model_id
                ),
                "failed": sum(
                    record.get("status") == "failed"
                    for record in final_records
                    if record["model"]["season_model_id"] == model_id
                ),
            }
            for model_id in sorted(
                {record["model"]["season_model_id"] for record in final_records}
            )
        },
        "arm_artifact_sha256s": sorted(
            str(record["artifact_sha256"]) for record in final_records
        ),
    }
    path = _atomic_write(output_dir, "collection-summary", summary)
    return {**summary, "summary_path": str(path)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--epicure-intervention", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "scored"), required=True)
    parser.add_argument("--tasks-per-family", type=int, required=True)
    parser.add_argument("--bedrock-cap-usd", type=Decimal, required=True)
    parser.add_argument("--openrouter-cap-usd", type=Decimal, required=True)
    parser.add_argument("--bedrock-concurrency", type=int, default=4)
    parser.add_argument("--openrouter-concurrency", type=int, default=2)
    parser.add_argument("--openrouter-base-url", required=True)
    parser.add_argument("--cost-envelope", type=Path)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    expected_confirmation = (
        CALIBRATION_CONFIRMATION if args.phase == "calibration" else SCORED_CONFIRMATION
    )
    if args.confirmation != expected_confirmation:
        raise CollectionError(f"real collection requires --confirmation {expected_confirmation}")
    if args.phase == "calibration" and args.tasks_per_family != 1:
        raise CollectionError("calibration requires exactly one real task per family")
    if args.phase == "scored" and args.tasks_per_family != 30:
        raise CollectionError("scored collection requires all 30 tasks per family")
    if not 1 <= args.bedrock_concurrency <= 12 or not 1 <= args.openrouter_concurrency <= 5:
        raise CollectionError("collection concurrency is outside the frozen safety bounds")
    if args.bedrock_cap_usd <= 0 or args.bedrock_cap_usd > Decimal("5000"):
        raise CollectionError("Bedrock sub-cap is outside the authorized $5,000 cap")
    if args.openrouter_cap_usd <= 0 or args.openrouter_cap_usd > Decimal("100"):
        raise CollectionError("OpenRouter sub-cap is outside the authorized $100 cap")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    if not api_key:
        raise CollectionError("OpenRouter API key is required")
    bedrock_settings = BedrockLaneSettings.from_environ()
    if not bedrock_settings.enabled or not bedrock_settings.live_authorized:
        raise CollectionError("Bedrock live collection is not authorized")
    if bedrock_settings.stage != "season":
        raise CollectionError("real Season 0 collection requires the Bedrock season stage")
    if args.bedrock_cap_usd > bedrock_settings.hard_cap_usd:
        raise CollectionError("the requested Bedrock sub-cap exceeds the authorized hard cap")
    clients = create_boto3_clients(bedrock_settings)
    task_bank = _load(args.task_bank)
    model_manifest = _load(args.model_manifest)
    epicure_intervention = _load(args.epicure_intervention)
    cost_envelope = _load(args.cost_envelope) if args.cost_envelope else None
    summary = asyncio.run(
        collect(
            task_bank=task_bank,
            model_manifest=model_manifest,
            epicure_intervention=epicure_intervention,
            output_dir=args.output_dir,
            phase=args.phase,
            per_family=args.tasks_per_family,
            runtime=clients.runtime,
            openrouter_base_url=args.openrouter_base_url,
            openrouter_api_key=api_key,
            cloudflare_gateway_token=os.environ.get("CLOUDFLARE_AI_GATEWAY_TOKEN") or "",
            bedrock_cap=args.bedrock_cap_usd,
            openrouter_cap=args.openrouter_cap_usd,
            bedrock_concurrency=args.bedrock_concurrency,
            openrouter_concurrency=args.openrouter_concurrency,
            cost_envelope=cost_envelope,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
