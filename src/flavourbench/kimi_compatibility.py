"""Direct Kimi Code K3 + real Epicure compatibility checks for FlavourBench.

This lane targets the authenticated Kimi Code managed API, not Moonshot Open
Platform and not OpenRouter. It freezes the exact model catalog, requires one
``find_pairings`` call, executes that call through the private Epicure MCP
service, and requests a JSON-Schema-constrained final answer. Results remain
unranked compatibility evidence until a scored-season manifest admits them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .mcp_client import McpSession, tool_catalog_sha256
from .provider import FINAL_SCHEMA
from .real_task_bank import sha256_json, sha256_text

CATALOG_SCHEMA_VERSION = "flavourbench-kimi-catalog-v1"
ARM_SCHEMA_VERSION = "flavourbench-kimi-epicure-contract-smoke-v1"
SUMMARY_SCHEMA_VERSION = "flavourbench-kimi-contract-summary-v1"
CONFIRMATION = "RUN_REAL_KIMI_EPICURE_CONTRACT_SMOKES_V1"
DEFAULT_MODELS = ("k3",)
DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1/"
SYSTEM_PROMPT = """You are completing an unranked FlavourBench compatibility task.
Call the supplied Epicure function exactly once before answering. Its output describes learned
statistical relationships, not chemical, safety, nutrition, cultural, or preference ground truth.
Do not identify your model or provider."""
FINAL_SYSTEM_PROMPT = """Write a concise culinary answer using the supplied statistical evidence.
Treat the evidence as explanatory rather than ground truth. Return only valid JSON matching the
supplied schema. Do not identify your model or provider."""
PROMPT = (
    "Design two plausible savoury pairing directions for pear and explain one bridge ingredient."
)
TEMPERATURE = 1


class KimiCompatibilityError(RuntimeError):
    """A Kimi catalog or compatibility contract was incomplete."""


class KimiAttemptError(KimiCompatibilityError):
    """A Kimi smoke failed after zero or more external requests were attempted."""

    def __init__(
        self,
        cause: Exception,
        *,
        provider_request_attempts: int,
        provider_calls: int,
        epicure_calls: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.provider_request_attempts = provider_request_attempts
        self.provider_calls = provider_calls
        self.epicure_calls = epicure_calls


@dataclass(frozen=True)
class KimiTarget:
    model_id: str
    catalog_entry: Mapping[str, Any]
    catalog_entry_sha256: str
    catalog_sha256: str


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise KimiCompatibilityError("content-addressed Kimi artifact conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _redacted_error(error: Exception) -> str:
    value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", str(error))
    value = re.sub(r"(?i)(api[_ -]?key)[=: ]+[A-Za-z0-9._-]+", r"\1=<redacted>", value)
    value = re.sub(r"\bsk-[A-Za-z0-9._-]{12,}\b", "sk-<redacted>", value)
    return value[:600]


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise KimiCompatibilityError("KIMI_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Epicure-FlavourBench/0.1",
    }


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, Mapping):
                raw = body.get("error") or body.get("message") or body.get("detail")
                if isinstance(raw, Mapping):
                    raw = raw.get("message") or raw.get("type") or raw
                detail = str(raw or "")
        except (ValueError, TypeError):
            detail = ""
        suffix = f": {detail[:400]}" if detail else ""
        raise KimiCompatibilityError(f"Kimi HTTP {response.status_code}{suffix}") from error


async def freeze_catalog(
    client: httpx.AsyncClient,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    response = await client.get("models")
    _raise_for_status(response)
    body = response.json()
    models = body.get("data") if isinstance(body, Mapping) else None
    if not isinstance(models, list):
        raise KimiCompatibilityError("Kimi model catalog returned no data list")
    entries = sorted(
        (dict(item) for item in models if isinstance(item, Mapping)),
        key=lambda item: str(item.get("id") or ""),
    )
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "provider": "kimi_code_direct",
        "catalog_endpoint": "https://api.kimi.com/coding/v1/models",
        "models": entries,
        "model_count": len(entries),
        "official": False,
        "rank_eligible": False,
    }
    return _atomic_write(output_dir, "kimi-catalog", payload), payload


def select_targets(
    catalog: Mapping[str, Any],
    model_ids: Sequence[str],
) -> list[KimiTarget]:
    models = catalog.get("models")
    if not isinstance(models, list):
        raise KimiCompatibilityError("Kimi catalog contains no models")
    by_id = {
        str(item.get("id")): dict(item)
        for item in models
        if isinstance(item, Mapping) and item.get("id")
    }
    catalog_sha = sha256_json(catalog)
    targets: list[KimiTarget] = []
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if entry is None:
            raise KimiCompatibilityError(
                f"requested Kimi model is absent from the authenticated catalog: {model_id}"
            )
        targets.append(
            KimiTarget(
                model_id=model_id,
                catalog_entry=entry,
                catalog_entry_sha256=sha256_json(entry),
                catalog_sha256=catalog_sha,
            )
        )
    return targets


def load_find_pairings(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise KimiCompatibilityError("invalid Epicure tool catalog") from error
    if not isinstance(value, list):
        raise KimiCompatibilityError("Epicure tool catalog must be an array")
    tools = [dict(item) for item in value if isinstance(item, Mapping)]
    match = next((item for item in tools if item.get("name") == "find_pairings"), None)
    if match is None or not isinstance(match.get("inputSchema"), Mapping):
        raise KimiCompatibilityError("Epicure catalog has no find_pairings schema")
    projected = {
        "type": "function",
        "function": {
            "name": "find_pairings",
            "description": str(match.get("description") or "Explore ingredient pairings"),
            "parameters": dict(match["inputSchema"]),
        },
    }
    return projected, tool_catalog_sha256(tools)


def _choice(response: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise KimiCompatibilityError("Kimi returned no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise KimiCompatibilityError("Kimi completion choice has no message")
    return choices[0], message


def _tool_call(message: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        raise KimiCompatibilityError("Kimi did not return exactly one Epicure tool call")
    call = calls[0]
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "find_pairings":
        raise KimiCompatibilityError("Kimi called an unexpected tool")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise KimiCompatibilityError("Kimi returned invalid tool arguments") from error
    if not isinstance(arguments, Mapping):
        raise KimiCompatibilityError("Kimi returned non-object tool arguments")
    return str(call.get("id") or ""), dict(arguments)


def _content_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") in {"text", "output_text"}
        ).strip()
    return ""


def _final(message: Mapping[str, Any]) -> dict[str, Any]:
    text = _content_text(message)
    if not text:
        raise KimiCompatibilityError("Kimi final response is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise KimiCompatibilityError("Kimi final response is not JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(FINAL_SCHEMA["properties"]):
        raise KimiCompatibilityError("Kimi final response does not match the frozen schema")
    if not isinstance(parsed.get("answer_markdown"), str) or not parsed["answer_markdown"].strip():
        raise KimiCompatibilityError("Kimi final answer is empty")
    for field in ("ingredient_mentions", "constraints_addressed", "uncertainties"):
        if not isinstance(parsed.get(field), list) or not all(
            isinstance(item, str) for item in parsed[field]
        ):
            raise KimiCompatibilityError(f"Kimi final field is invalid: {field}")
    return parsed


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(prompt_details, Mapping):
        prompt_details = {}
    if not isinstance(completion_details, Mapping):
        completion_details = {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }


def _assert_returned_model(response: Mapping[str, Any], target: KimiTarget) -> str:
    returned = str(response.get("model") or "")
    if not returned:
        raise KimiCompatibilityError("Kimi response omitted the actual model")
    if returned != target.model_id:
        raise KimiCompatibilityError(
            f"Kimi returned model {returned}, expected exact catalog ID {target.model_id}"
        )
    return returned


async def smoke_target(
    target: KimiTarget,
    tool: Mapping[str, Any],
    tool_catalog_sha: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    started = time.monotonic()
    provider_request_attempts = 0
    provider_calls = 0
    epicure_calls = 0
    try:
        tool_request = {
            "model": target.model_id,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            "tools": [tool],
            "tool_choice": "required",
            "temperature": TEMPERATURE,
            "max_tokens": 4096,
        }
        provider_request_attempts += 1
        tool_http = await client.post("chat/completions", json=tool_request)
        _raise_for_status(tool_http)
        provider_calls += 1
        tool_response = tool_http.json()
        if not isinstance(tool_response, Mapping):
            raise KimiCompatibilityError("Kimi tool response is not an object")
        tool_returned_model = _assert_returned_model(tool_response, target)
        tool_choice, assistant = _choice(tool_response)
        tool_call_id, arguments = _tool_call(assistant)

        async with McpSession() as mcp:
            epicure_calls += 1
            result = await mcp.call_tool("find_pairings", arguments)
        if result.is_error:
            raise KimiCompatibilityError("Epicure returned a tool error")
        evidence = result.structured or {"text": result.text}

        final_request = {
            "model": target.model_id,
            "stream": False,
            "messages": [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{PROMPT}\n\nStatistical evidence (JSON):\n"
                        f"{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n\n"
                        "Return the JSON object now."
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "flavourbench_answer",
                    "strict": True,
                    "schema": FINAL_SCHEMA,
                },
            },
            "temperature": TEMPERATURE,
            "max_tokens": 4096,
        }
        provider_request_attempts += 1
        final_http = await client.post("chat/completions", json=final_request)
        _raise_for_status(final_http)
        provider_calls += 1
        final_response = final_http.json()
        if not isinstance(final_response, Mapping):
            raise KimiCompatibilityError("Kimi final response is not an object")
        final_returned_model = _assert_returned_model(final_response, target)
        final_choice, final_message = _choice(final_response)
        output = _final(final_message)
        generation_ids = [
            str(tool_response.get("id") or ""),
            str(final_response.get("id") or ""),
        ]
        if any(not generation_id for generation_id in generation_ids):
            raise KimiCompatibilityError("Kimi response omitted a generation ID")
        usages = [_usage(tool_response), _usage(final_response)]
    except KimiAttemptError:
        raise
    except Exception as error:
        raise KimiAttemptError(
            error,
            provider_request_attempts=provider_request_attempts,
            provider_calls=provider_calls,
            epicure_calls=epicure_calls,
        ) from error

    return {
        "schema_version": ARM_SCHEMA_VERSION,
        "status": "smoke_passed",
        "provider": "kimi_code_direct",
        "api_contract": "OpenAI-compatible Chat Completions",
        "requested_model_id": target.model_id,
        "returned_model_ids": [tool_returned_model, final_returned_model],
        "catalog_sha256": target.catalog_sha256,
        "catalog_entry_sha256": target.catalog_entry_sha256,
        "catalog_entry": dict(target.catalog_entry),
        "prompt": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "final_system_prompt_sha256": sha256_text(FINAL_SYSTEM_PROMPT),
        "response_schema_sha256": sha256_json(FINAL_SCHEMA),
        "epicure_tool_catalog_sha256": tool_catalog_sha,
        "projected_tool_sha256": sha256_json(tool),
        "provider_contract": {
            "base_url": DEFAULT_BASE_URL,
            "tool_choice": "required",
            "structured_output": "openai_json_schema_strict",
            "max_tokens": 4096,
            "temperature": TEMPERATURE,
        },
        "output_json": output,
        "complete_epicure_trace": [
            {
                "round_index": 0,
                "tool_call_id": tool_call_id,
                "name": "find_pairings",
                "arguments": arguments,
                "result": evidence,
                "result_sha256": sha256_json(evidence),
                "latency_ms": result.latency_ms,
                "is_error": result.is_error,
            }
        ],
        "generation_ids": generation_ids,
        "usage": {
            field: sum(item[field] for item in usages)
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
            )
        },
        "finish_reasons": [
            str(tool_choice.get("finish_reason") or "unknown"),
            str(final_choice.get("finish_reason") or "unknown"),
        ],
        "request_payload_sha256s": [
            sha256_json(tool_request),
            sha256_json(final_request),
        ],
        "wall_clock_latency_ms": round((time.monotonic() - started) * 1000),
        "provider_request_attempts": provider_request_attempts,
        "real_provider_calls": provider_calls,
        "real_epicure_calls": epicure_calls,
        "cost_status": "managed_service_returned_no_per_generation_cost",
        "official": False,
        "rank_eligible": False,
    }


async def execute(
    *,
    output_dir: Path,
    tool_catalog_path: Path,
    model_ids: Sequence[str],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[Path, dict[str, Any]]:
    tool, tool_sha = load_find_pairings(tool_catalog_path)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers=_headers(api_key),
        timeout=300,
    ) as client:
        catalog_path, catalog = await freeze_catalog(client, output_dir / "catalog")
        targets = select_targets(catalog, model_ids)
        artifacts: list[dict[str, Any]] = []
        for target in targets:
            try:
                payload = await smoke_target(target, tool, tool_sha, client)
            except Exception as error:  # noqa: BLE001 - immutable redacted failure evidence
                attempt = error if isinstance(error, KimiAttemptError) else None
                payload = {
                    "schema_version": ARM_SCHEMA_VERSION,
                    "status": "failed",
                    "provider": "kimi_code_direct",
                    "requested_model_id": target.model_id,
                    "catalog_sha256": target.catalog_sha256,
                    "catalog_entry_sha256": target.catalog_entry_sha256,
                    "error_type": attempt.cause_type if attempt else type(error).__name__,
                    "error": _redacted_error(error),
                    "provider_request_attempts": (
                        attempt.provider_request_attempts if attempt else 0
                    ),
                    "real_provider_calls": attempt.provider_calls if attempt else 0,
                    "real_epicure_calls": attempt.epicure_calls if attempt else 0,
                    "official": False,
                    "rank_eligible": False,
                }
            path = _atomic_write(
                output_dir / "compatibility",
                f"kimi-{sha256_text(target.model_id)[:12]}",
                payload,
            )
            artifacts.append(
                {
                    "model_id": target.model_id,
                    "status": payload["status"],
                    "artifact_path": str(path),
                    "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                    "provider_calls": int(payload.get("real_provider_calls") or 0),
                    "provider_request_attempts": int(payload.get("provider_request_attempts") or 0),
                    "epicure_calls": int(payload.get("real_epicure_calls") or 0),
                    "error_type": payload.get("error_type"),
                }
            )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "provider": "kimi_code_direct",
        "catalog_artifact_path": str(catalog_path),
        "catalog_artifact_sha256": catalog_path.stem.rsplit("-", 1)[-1],
        "counts": {
            "targets": len(artifacts),
            "smoke_passed": sum(item["status"] == "smoke_passed" for item in artifacts),
            "failed": sum(item["status"] == "failed" for item in artifacts),
            "provider_calls": sum(item["provider_calls"] for item in artifacts),
            "provider_request_attempts": sum(
                item["provider_request_attempts"] for item in artifacts
            ),
            "epicure_calls": sum(item["epicure_calls"] for item in artifacts),
        },
        "artifacts": artifacts,
        "official": False,
        "rank_eligible": False,
    }
    return _atomic_write(output_dir, "kimi-contract-summary", summary), summary


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--confirm", required=True)
    arguments = parser.parse_args(argv)
    if arguments.confirm != CONFIRMATION:
        raise KimiCompatibilityError(f"exact confirmation required: {CONFIRMATION}")
    api_key = os.environ.get("KIMI_API_KEY") or ""
    path, summary = asyncio.run(
        execute(
            output_dir=arguments.output_dir,
            tool_catalog_path=arguments.tool_catalog,
            model_ids=tuple(arguments.models or DEFAULT_MODELS),
            api_key=api_key,
            base_url=arguments.base_url,
        )
    )
    print(json.dumps({"summary": str(path), "counts": summary["counts"]}, sort_keys=True))


if __name__ == "__main__":
    run()
