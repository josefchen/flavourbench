"""Real OpenRouter + Epicure compatibility smokes for Season 0 fallback slots."""

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
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from .execution_policy import assert_legacy_paid_cli_allowed
from .mcp_client import McpSession, tool_catalog_sha256
from .provider import FINAL_SCHEMA
from .real_task_bank import sha256_json, sha256_text
from .season0_compatibility import PROMPT, SYSTEM_PROMPT

SCHEMA_VERSION = "flavourbench-season0-openrouter-compatibility-v1"
CONFIRMATION = "RUN_REAL_SEASON0_OPENROUTER_EPICURE_COMPATIBILITY_V1"


class OpenRouterCompatibilityError(RuntimeError):
    """An exact OpenRouter route failed the compatibility contract."""


@dataclass(frozen=True)
class OpenRouterTarget:
    display_name: str
    model_id: str
    canonical_slug: str
    provider_slug: str
    provider_name: str
    supported_parameters: tuple[str, ...]
    endpoint_document_sha256: str
    pricing: Mapping[str, Any]
    source_manifest_sha256: str


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
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_bytes() != data:
            raise OpenRouterCompatibilityError("content-addressed compatibility conflict")
        return path
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o600)
    return path


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise OpenRouterCompatibilityError(f"invalid JSON input: {path}") from error


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    direct = manifest.get("artifact_sha256") or manifest.get("content_sha256")
    if isinstance(direct, str) and len(direct) == 64:
        return direct
    address = manifest.get("content_address")
    if not isinstance(address, Mapping):
        raise OpenRouterCompatibilityError("OpenRouter manifest has no content address")
    value = address.get("sha256") or address.get("digest")
    if not isinstance(value, str) or len(value) != 64:
        raise OpenRouterCompatibilityError("OpenRouter manifest has an invalid content hash")
    return value


def load_targets(roster_path: Path, manifest_path: Path) -> list[OpenRouterTarget]:
    roster = _load(roster_path)
    manifest = _load(manifest_path)
    if not isinstance(roster, Mapping) or not isinstance(manifest, Mapping):
        raise OpenRouterCompatibilityError("roster and OpenRouter manifest must be objects")
    slots = roster.get("slots")
    routes = manifest.get("routes")
    models = routes if isinstance(routes, list) else manifest.get("models")
    if not isinstance(slots, list) or not isinstance(models, list):
        raise OpenRouterCompatibilityError("roster or manifest collections are invalid")
    by_model_id: dict[str, Mapping[str, Any]] = {}
    for item in models:
        if not isinstance(item, Mapping):
            continue
        model = item.get("model")
        model_id = item.get("model_id")
        if not isinstance(model_id, str) and isinstance(model, Mapping):
            model_id = model.get("id")
        if isinstance(model_id, str):
            by_model_id[model_id] = item
    source_sha = _manifest_sha256(manifest)
    output: list[OpenRouterTarget] = []
    required = {"tools", "tool_choice"}
    for slot in slots:
        if not isinstance(slot, Mapping) or slot.get("provider") != "openrouter":
            continue
        model_id = str(slot.get("endpoint_id") or "")
        item = by_model_id.get(model_id)
        if item is None:
            raise OpenRouterCompatibilityError(
                f"OpenRouter roster model is absent from its frozen catalog: {model_id}"
            )
        model = item["model"]
        endpoint = item.get("endpoint")
        if not isinstance(endpoint, Mapping):
            raise OpenRouterCompatibilityError(f"OpenRouter model has no endpoint: {model_id}")
        provider_slug = str(slot.get("provider_slug") or "")
        if endpoint.get("tag") != provider_slug:
            raise OpenRouterCompatibilityError(f"provider tag drift for {model_id}")
        canonical_slug = str(slot.get("canonical_slug") or "")
        item_canonical = item.get("canonical_slug") or model.get("canonical_slug")
        if item_canonical != canonical_slug:
            raise OpenRouterCompatibilityError(f"canonical model drift for {model_id}")
        supported = tuple(sorted(str(value) for value in endpoint.get("supported_parameters", [])))
        if not required.issubset(supported) or not {
            "max_tokens",
            "max_completion_tokens",
        }.intersection(supported):
            raise OpenRouterCompatibilityError(
                f"OpenRouter endpoint lacks required tools: {model_id}"
            )
        output.append(
            OpenRouterTarget(
                display_name=str(slot.get("canonical_name") or model.get("name") or model_id),
                model_id=model_id,
                canonical_slug=canonical_slug,
                provider_slug=provider_slug,
                provider_name=str(endpoint.get("provider_name") or ""),
                supported_parameters=supported,
                endpoint_document_sha256=str(item.get("endpoint_document_sha256") or ""),
                pricing=dict(endpoint.get("pricing") or {}),
                source_manifest_sha256=source_sha,
            )
        )
    return output


def load_find_pairings(path: Path) -> tuple[dict[str, Any], str]:
    value = _load(path)
    if not isinstance(value, list):
        raise OpenRouterCompatibilityError("Epicure tool catalog must be an array")
    tools = [dict(item) for item in value if isinstance(item, Mapping)]
    match = next((item for item in tools if item.get("name") == "find_pairings"), None)
    if match is None or not isinstance(match.get("inputSchema"), Mapping):
        raise OpenRouterCompatibilityError("Epicure catalog has no find_pairings schema")
    return match, tool_catalog_sha256(tools)


def _headers(
    api_key: str,
    gateway_token: str,
    base_url: str,
    *,
    request_title: str = "Epicure FlavourBench Season 0 compatibility",
) -> dict[str, str]:
    values = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": request_title,
    }
    if "gateway.ai.cloudflare.com" in base_url:
        if not gateway_token:
            raise OpenRouterCompatibilityError("Cloudflare gateway token is required")
        values.update(
            {
                "cf-aig-authorization": f"Bearer {gateway_token}",
                "cf-aig-skip-cache": "true",
                "cf-aig-collect-log-payload": "false",
            }
        )
    return values


def _choice(response: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        error = response.get("error")
        error_code = ""
        error_message = ""
        if isinstance(error, Mapping):
            error_code = str(error.get("code") or "")[:80]
            error_message = re.sub(
                r"sk-[A-Za-z0-9_-]{12,}",
                "<credential-redacted>",
                str(error.get("message") or ""),
            )[:240]
        choice_count = len(choices) if isinstance(choices, list) else None
        raise OpenRouterCompatibilityError(
            "OpenRouter returned no choice "
            f"(choice_count={choice_count}, error_code={error_code!r}, "
            f"error_message={error_message!r})"
        )
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise OpenRouterCompatibilityError("OpenRouter choice has no message")
    return choices[0], message


def _final(message: Mapping[str, Any]) -> dict[str, Any]:
    parsed = message.get("parsed")
    if isinstance(parsed, Mapping):
        value: Any = dict(parsed)
    else:
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenRouterCompatibilityError("OpenRouter final response contains no text")
        cleaned = content.strip()
        if not cleaned:
            raise OpenRouterCompatibilityError("OpenRouter final response is empty")
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            value = {
                "answer_markdown": cleaned,
                "ingredient_mentions": [],
                "constraints_addressed": [],
                "uncertainties": [],
            }
    if (
        isinstance(value, dict)
        and set(value) == set(FINAL_SCHEMA["properties"])
        and isinstance(value.get("answer_markdown"), str)
        and value["answer_markdown"].strip()
    ):
        return value
    if isinstance(message.get("content"), str) and str(message["content"]).strip():
        return {
            "answer_markdown": str(message["content"]).strip(),
            "ingredient_mentions": [],
            "constraints_addressed": [],
            "uncertainties": [],
        }
    raise OpenRouterCompatibilityError("OpenRouter final answer is empty")


def _token_limit(target: OpenRouterTarget, value: int) -> dict[str, int]:
    field = "max_tokens" if "max_tokens" in target.supported_parameters else "max_completion_tokens"
    return {field: value}


def _structured_response_format(target: OpenRouterTarget) -> dict[str, Any]:
    required = {"response_format", "structured_outputs"}
    if not required.issubset(target.supported_parameters):
        raise OpenRouterCompatibilityError(
            f"OpenRouter endpoint lacks strict structured output: {target.model_id}"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "flavourbench_answer",
            "strict": True,
            "schema": FINAL_SCHEMA,
        },
    }


def _finish_contract(finish_reason: str) -> tuple[str, str | None, str | None]:
    if finish_reason == "stop":
        return "smoke_passed", None, None
    return (
        "failed_incomplete_finish",
        "IncompleteFinish",
        f"final generation ended with {finish_reason}",
    )


async def _accounting(
    client: httpx.AsyncClient, generation_id: str, attempts: int = 6
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
                    "reconciled": True,
                }
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            pass
        if attempt + 1 < attempts:
            await asyncio.sleep(0.5 * (2**attempt))
    raise OpenRouterCompatibilityError(f"generation accounting did not reconcile: {generation_id}")


async def _smoke(
    target: OpenRouterTarget,
    tool: Mapping[str, Any],
    tool_catalog_sha: str,
    generation_client: httpx.AsyncClient,
    accounting_client: httpx.AsyncClient,
    *,
    artifact_schema_version: str = SCHEMA_VERSION,
    first_max_tokens: int = 700,
    final_max_tokens: int = 4_096,
    require_structured_output: bool = False,
) -> dict[str, Any]:
    if not 128 <= first_max_tokens <= 8_192:
        raise OpenRouterCompatibilityError("tool-turn token limit is outside service bounds")
    if not 128 <= final_max_tokens <= 8_192:
        raise OpenRouterCompatibilityError("final-turn token limit is outside service bounds")
    start = time.monotonic()
    tool_payload = {
        "type": "function",
        "function": {
            "name": "find_pairings",
            "description": str(tool.get("description") or "Explore ingredient pairings"),
            "parameters": tool["inputSchema"],
        },
    }
    provider = {
        "only": [target.provider_slug],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    request_hashes: list[str] = []
    responses: list[Mapping[str, Any]] = []

    first_payload = {
        "model": target.model_id,
        "messages": messages,
        "tools": [tool_payload],
        "tool_choice": "required",
        **_token_limit(target, first_max_tokens),
        "provider": provider,
        "usage": {"include": True},
    }
    request_hashes.append(sha256_json(first_payload))
    first_http = await generation_client.post("chat/completions", json=first_payload)
    first_http.raise_for_status()
    first = first_http.json()
    if not isinstance(first, Mapping):
        raise OpenRouterCompatibilityError("OpenRouter tool response is not an object")
    responses.append(first)
    _first_choice, assistant = _choice(first)
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        raise OpenRouterCompatibilityError("model did not make exactly one Epicure call")
    function = calls[0].get("function")
    if not isinstance(function, Mapping) or function.get("name") != "find_pairings":
        raise OpenRouterCompatibilityError("model called an unexpected tool")
    try:
        arguments = json.loads(str(function.get("arguments") or "{}"))
    except json.JSONDecodeError as error:
        raise OpenRouterCompatibilityError("model returned invalid tool arguments") from error
    if not isinstance(arguments, dict):
        raise OpenRouterCompatibilityError("model returned non-object tool arguments")
    async with McpSession() as mcp:
        result = await mcp.call_tool("find_pairings", arguments)
    if result.is_error:
        raise OpenRouterCompatibilityError("Epicure returned a tool error")
    messages.extend(
        [
            dict(assistant),
            {
                "role": "tool",
                "tool_call_id": str(calls[0].get("id") or ""),
                "name": "find_pairings",
                "content": result.text,
            },
            {"role": "user", "content": "Return the final structured answer now."},
        ]
    )
    final_payload = {
        "model": target.model_id,
        "messages": messages,
        **_token_limit(target, final_max_tokens),
        "provider": provider,
        "usage": {"include": True},
    }
    if require_structured_output:
        final_payload["response_format"] = _structured_response_format(target)
    request_hashes.append(sha256_json(final_payload))
    final_http = await generation_client.post("chat/completions", json=final_payload)
    final_http.raise_for_status()
    final_response = final_http.json()
    if not isinstance(final_response, Mapping):
        raise OpenRouterCompatibilityError("OpenRouter final response is not an object")
    responses.append(final_response)
    final_choice, final_message = _choice(final_response)
    output = _final(final_message)
    generation_ids = [str(response.get("id") or "") for response in responses]
    if any(not generation_id for generation_id in generation_ids):
        raise OpenRouterCompatibilityError("OpenRouter response omitted a generation ID")
    accounting = await asyncio.gather(
        *(_accounting(accounting_client, generation_id) for generation_id in generation_ids)
    )
    if accounting[-1]["model"] != target.canonical_slug:
        raise OpenRouterCompatibilityError(
            f"returned model {accounting[-1]['model']} differs from {target.canonical_slug}"
        )
    if any(item["provider_name"] != target.provider_name for item in accounting):
        raise OpenRouterCompatibilityError("returned provider differs from the frozen endpoint")
    usage = {
        "prompt_tokens": sum(
            int((response.get("usage") or {}).get("prompt_tokens") or 0) for response in responses
        ),
        "completion_tokens": sum(
            int((response.get("usage") or {}).get("completion_tokens") or 0)
            for response in responses
        ),
    }
    finish_reason = str(final_choice.get("finish_reason") or "unknown")
    status, error_type, error = _finish_contract(finish_reason)
    return {
        "schema_version": artifact_schema_version,
        "status": status,
        "error_type": error_type,
        "error": error,
        "display_name": target.display_name,
        "requested_model_id": target.model_id,
        "canonical_slug": target.canonical_slug,
        "requested_provider_slug": target.provider_slug,
        "returned_provider_name": target.provider_name,
        "source_manifest_sha256": target.source_manifest_sha256,
        "endpoint_document_sha256": target.endpoint_document_sha256,
        "supported_parameters": list(target.supported_parameters),
        "pricing": target.pricing,
        "prompt": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "response_schema_sha256": sha256_json(FINAL_SCHEMA),
        "normalization_mode": "lossless_client_text_wrapper_v1",
        "provider_structured_output_required": require_structured_output,
        "epicure_tool_catalog_sha256": tool_catalog_sha,
        "projected_tool_sha256": sha256_json(tool_payload),
        "output_json": output,
        "complete_epicure_trace": [
            {
                "round_index": 0,
                "name": "find_pairings",
                "arguments": arguments,
                "result": result.structured or result.text,
                "result_sha256": sha256_json(result.structured or result.text),
                "latency_ms": result.latency_ms,
                "is_error": result.is_error,
            }
        ],
        "generation_ids": generation_ids,
        "generation_accounting": accounting,
        "cost_usd": format(
            sum((Decimal(item["total_cost_usd"]) for item in accounting), Decimal(0)),
            "f",
        ),
        "usage": usage,
        "finish_reason": finish_reason,
        "generation_costs_reconciled": True,
        "request_payload_sha256s": request_hashes,
        "wall_clock_latency_ms": round((time.monotonic() - start) * 1000),
        "real_provider_calls": 2,
        "real_epicure_calls": 1,
        "official": False,
        "rank_eligible": False,
    }


async def execute(
    *,
    targets: Sequence[OpenRouterTarget],
    tool: Mapping[str, Any],
    tool_catalog_sha: str,
    output_dir: Path,
    base_url: str,
    api_key: str,
    gateway_token: str,
) -> dict[str, Any]:
    headers = _headers(api_key, gateway_token, base_url)
    accounting_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "HTTP-Referer": "https://epicure.kaikaku.ai/flavourbench",
        "X-Title": "Epicure FlavourBench Season 0 compatibility",
    }
    artifacts: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=240) as gen,
        httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1/",
            headers=accounting_headers,
            timeout=120,
        ) as accounting,
    ):
        for target in targets:
            try:
                payload = await _smoke(target, tool, tool_catalog_sha, gen, accounting)
            except Exception as error:  # noqa: BLE001 - persist a redacted compatibility failure
                message = re.sub(r"(?<!\d)\d{12}(?!\d)", "<account-redacted>", str(error))
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "display_name": target.display_name,
                    "requested_model_id": target.model_id,
                    "canonical_slug": target.canonical_slug,
                    "requested_provider_slug": target.provider_slug,
                    "source_manifest_sha256": target.source_manifest_sha256,
                    "endpoint_document_sha256": target.endpoint_document_sha256,
                    "error_type": type(error).__name__,
                    "error": message[:600],
                    "official": False,
                    "rank_eligible": False,
                }
            path = _atomic_write(
                output_dir,
                f"compatibility-{sha256_text(target.model_id)[:12]}",
                payload,
            )
            artifacts.append(
                {
                    "model_id": target.model_id,
                    "display_name": target.display_name,
                    "status": payload["status"],
                    "path": str(path),
                    "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                    "cost_usd": str(payload.get("cost_usd") or "0"),
                    "error_type": payload.get("error_type"),
                }
            )
    return {
        "schema_version": "flavourbench-season0-openrouter-compatibility-summary-v1",
        "counts": {
            "targets": len(artifacts),
            "smoke_passed": sum(item["status"] == "smoke_passed" for item in artifacts),
            "failed": sum(item["status"] == "failed" for item in artifacts),
        },
        "cost_usd": format(sum((Decimal(item["cost_usd"]) for item in artifacts), Decimal(0)), "f"),
        "artifacts": artifacts,
        "official": False,
        "rank_eligible": False,
    }


def run(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-season0-openrouter-compatibility")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season0/compatibility"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("10"))
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    if args.confirmation != CONFIRMATION:
        raise OpenRouterCompatibilityError(f"live smoke requires --confirmation {CONFIRMATION}")
    if args.cap_usd <= 0 or args.cap_usd > Decimal("15"):
        raise OpenRouterCompatibilityError("OpenRouter compatibility sub-cap must be in (0, 15]")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    if not api_key:
        raise OpenRouterCompatibilityError("OpenRouter API key is required")
    targets = load_targets(args.roster, args.manifest)
    if Decimal(len(targets)) * Decimal("2") > args.cap_usd:
        raise OpenRouterCompatibilityError("$2-per-target reservation exceeds the sub-cap")
    tool, tool_catalog_sha = load_find_pairings(args.tool_catalog)
    summary = asyncio.run(
        execute(
            targets=targets,
            tool=tool,
            tool_catalog_sha=tool_catalog_sha,
            output_dir=args.output_dir,
            base_url=args.base_url,
            api_key=api_key,
            gateway_token=os.environ.get("CLOUDFLARE_AI_GATEWAY_TOKEN") or "",
        )
    )
    path = _atomic_write(args.output_dir, "openrouter-compatibility-summary", summary)
    print(json.dumps({**summary, "summary_path": str(path)}, indent=2))


if __name__ == "__main__":
    run()
