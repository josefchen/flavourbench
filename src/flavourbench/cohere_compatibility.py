"""Direct Cohere V2 + real Epicure compatibility checks for FlavourBench.

This lane is intentionally separate from OpenRouter and Amazon Bedrock. It first
freezes the authenticated Cohere model catalog, then requires a strict
``find_pairings`` call, executes that call against the private Epicure MCP
service, and finally asks the same model for a JSON-Schema-constrained answer.

Cohere V2 does not allow ``response_format`` and ``tools`` in the same request,
so the final normalization request receives the original prompt and the exact
Epicure evidence in a fresh, explicitly hashed message. This limitation is
recorded in every artifact and keeps these checks unranked until a season
protocol explicitly approves the two-phase contract.
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

CATALOG_SCHEMA_VERSION = "flavourbench-cohere-catalog-v1"
ARM_SCHEMA_VERSION = "flavourbench-cohere-epicure-contract-smoke-v1"
SUMMARY_SCHEMA_VERSION = "flavourbench-cohere-contract-summary-v1"
CONFIRMATION = "RUN_REAL_COHERE_EPICURE_CONTRACT_SMOKES_V1"
DEFAULT_MODELS = (
    "command-a-plus-05-2026",
    "command-a-reasoning-08-2025",
)
COHERE_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "allOf",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "uniqueItems",
    }
)
SYSTEM_PROMPT = """You are completing an unranked FlavourBench compatibility task.
Use the Epicure tool exactly as instructed. Epicure outputs describe learned statistical
relationships, not chemical, safety, nutrition, cultural, or preference ground truth.
Do not identify your model or provider."""
FINAL_SYSTEM_PROMPT = """Write a concise culinary answer. Treat supplied associations as statistical
evidence, not ground truth. Return only valid JSON matching the supplied schema."""
PROMPT = (
    "Design two plausible savoury pairing directions for pear and explain one bridge ingredient."
)


class CohereCompatibilityError(RuntimeError):
    """A Cohere catalog or compatibility contract was incomplete."""


class CohereAttemptError(CohereCompatibilityError):
    """A smoke failed after zero or more external requests were attempted."""

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
class CohereTarget:
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
            raise CohereCompatibilityError("content-addressed Cohere artifact conflict")
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
    return value[:600]


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise CohereCompatibilityError("COHERE_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client-Name": "Epicure-FlavourBench",
    }


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, Mapping):
                detail = str(body.get("message") or body.get("detail") or body.get("error") or "")
        except (ValueError, TypeError):
            detail = ""
        suffix = f": {detail[:400]}" if detail else ""
        raise CohereCompatibilityError(f"Cohere HTTP {response.status_code}{suffix}") from error


async def freeze_catalog(
    client: httpx.AsyncClient,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    models: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, str | int] = {"page_size": 1000}
        if page_token:
            params["page_token"] = page_token
        response = await client.get("v1/models", params=params)
        _raise_for_status(response)
        body = response.json()
        page = body.get("models") if isinstance(body, Mapping) else None
        if not isinstance(page, list):
            raise CohereCompatibilityError("Cohere model catalog returned no model list")
        models.extend(dict(item) for item in page if isinstance(item, Mapping))
        raw_next = body.get("next_page_token")
        page_token = str(raw_next) if raw_next else None
        if not page_token:
            break
    models.sort(key=lambda item: str(item.get("name") or ""))
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "provider": "cohere_direct",
        "catalog_endpoint": "https://api.cohere.com/v1/models",
        "models": models,
        "model_count": len(models),
        "official": False,
        "rank_eligible": False,
    }
    return _atomic_write(output_dir, "cohere-catalog", payload), payload


def select_targets(
    catalog: Mapping[str, Any],
    model_ids: Sequence[str],
) -> list[CohereTarget]:
    models = catalog.get("models")
    if not isinstance(models, list):
        raise CohereCompatibilityError("Cohere catalog contains no models")
    catalog_sha = sha256_json(catalog)
    by_name = {
        str(item.get("name")): dict(item)
        for item in models
        if isinstance(item, Mapping) and item.get("name")
    }
    targets: list[CohereTarget] = []
    for model_id in model_ids:
        entry = by_name.get(model_id)
        if entry is None:
            raise CohereCompatibilityError(
                f"requested Cohere model is absent from the authenticated catalog: {model_id}"
            )
        endpoints = {str(item).lower() for item in entry.get("endpoints") or []}
        if bool(entry.get("is_deprecated")) or "chat" not in endpoints:
            raise CohereCompatibilityError(
                f"requested Cohere model is deprecated or lacks Chat: {model_id}"
            )
        targets.append(
            CohereTarget(
                model_id=model_id,
                catalog_entry=entry,
                catalog_entry_sha256=sha256_json(entry),
                catalog_sha256=catalog_sha,
            )
        )
    return targets


def project_cohere_strict_schema(value: Any) -> Any:
    """Project JSON Schema into Cohere's documented strict-tools subset.

    The full, immutable Epicure tool catalog remains the source contract and is
    hashed separately. This projection removes only keywords Cohere documents
    as unsupported; it does not rewrite types, fields, required properties, or
    descriptions.
    """

    if isinstance(value, Mapping):
        return {
            str(key): project_cohere_strict_schema(item)
            for key, item in value.items()
            if key not in COHERE_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [project_cohere_strict_schema(item) for item in value]
    return value


def load_find_pairings(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CohereCompatibilityError("invalid Epicure tool catalog") from error
    if not isinstance(value, list):
        raise CohereCompatibilityError("Epicure tool catalog must be an array")
    tools = [dict(item) for item in value if isinstance(item, Mapping)]
    match = next((item for item in tools if item.get("name") == "find_pairings"), None)
    if match is None or not isinstance(match.get("inputSchema"), Mapping):
        raise CohereCompatibilityError("Epicure catalog has no find_pairings schema")
    projected = {
        "type": "function",
        "function": {
            "name": "find_pairings",
            "description": str(match.get("description") or "Explore ingredient pairings"),
            "parameters": project_cohere_strict_schema(match["inputSchema"]),
        },
    }
    return projected, tool_catalog_sha256(tools)


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        usage = response.get("meta")
        usage = usage.get("tokens") if isinstance(usage, Mapping) else {}
    if not isinstance(usage, Mapping):
        usage = {}
    tokens = usage.get("tokens")
    if not isinstance(tokens, Mapping):
        tokens = usage
    billed = usage.get("billed_units")
    if not isinstance(billed, Mapping):
        billed = {}
    return {
        "input_tokens": int(tokens.get("input_tokens") or tokens.get("inputTokens") or 0),
        "output_tokens": int(tokens.get("output_tokens") or tokens.get("outputTokens") or 0),
        "reasoning_tokens": int(
            tokens.get("reasoning_tokens") or tokens.get("reasoningTokens") or 0
        ),
        "billed_input_tokens": int(billed.get("input_tokens") or 0),
        "billed_output_tokens": int(billed.get("output_tokens") or 0),
    }


def _tool_call(response: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    message = response.get("message")
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        raise CohereCompatibilityError("Cohere did not return exactly one Epicure tool call")
    call = calls[0]
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != "find_pairings":
        raise CohereCompatibilityError("Cohere called an unexpected tool")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise CohereCompatibilityError("Cohere returned invalid tool arguments") from error
    if not isinstance(arguments, Mapping):
        raise CohereCompatibilityError("Cohere returned non-object tool arguments")
    return str(call.get("id") or ""), dict(arguments)


def project_find_pairings_evidence(text: str) -> tuple[dict[str, Any], str]:
    """Create a deterministic bounded synthesis view of a find_pairings graph."""

    query_match = re.search(r"^Pairing graph for:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    query = query_match.group(1).strip() if query_match else ""
    associations: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\s+[A-Z]\s+\[", line):
            continue
        _, _, nodes = line.partition(":")
        for ingredient, score in re.findall(r"([^,()]+?)\s+\((-?\d+(?:\.\d+)?)\)", nodes):
            associations.append(
                {
                    "ingredient": ingredient.strip(),
                    "score": float(score),
                }
            )
    bridges: list[dict[str, Any]] = []
    in_bridges = False
    for line in text.splitlines():
        if line.startswith("BRIDGES "):
            in_bridges = True
            continue
        if not in_bridges or not line.strip():
            continue
        match = re.match(r"^\s+(.+?)\s+->\s+(.+?)\s+\(\d+\s+of\s+\d+\s+primaries\)\s*$", line)
        if not match:
            continue
        bridges.append(
            {
                "ingredient": match.group(1).strip(),
                "connects": [
                    ingredient.strip()
                    for ingredient in match.group(2).split(",")
                    if ingredient.strip()
                ],
            }
        )
    if not query or not associations or not bridges:
        raise CohereCompatibilityError(
            "Epicure find_pairings result could not be projected deterministically"
        )
    projection = {
        "projection_version": "find-pairings-synthesis-v2",
        "query": query,
        "top_associations": associations[:6],
        "bridge_ingredients": [item["ingredient"] for item in bridges[:5]],
    }
    association_text = ", ".join(
        f"{item['ingredient']} ({item['score']:.3f})" for item in projection["top_associations"]
    )
    bridge_text = ", ".join(projection["bridge_ingredients"])
    rendered = (
        f"Statistical associations for {query}: {association_text}. "
        f"Multi-primary bridge ingredients include {bridge_text}."
    )
    return projection, rendered


def _final(response: Mapping[str, Any]) -> dict[str, Any]:
    message = response.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list):
        raise CohereCompatibilityError("Cohere final response has no content blocks")
    text = "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text"
    ).strip()
    if not text:
        raise CohereCompatibilityError("Cohere final response is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise CohereCompatibilityError("Cohere final response is not JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(FINAL_SCHEMA["properties"]):
        raise CohereCompatibilityError("Cohere final response does not match the frozen schema")
    if not isinstance(parsed.get("answer_markdown"), str) or not parsed["answer_markdown"].strip():
        raise CohereCompatibilityError("Cohere final answer is empty")
    for field in ("ingredient_mentions", "constraints_addressed", "uncertainties"):
        if not isinstance(parsed.get(field), list) or not all(
            isinstance(item, str) for item in parsed[field]
        ):
            raise CohereCompatibilityError(f"Cohere final field is invalid: {field}")
    return parsed


async def smoke_target(
    target: CohereTarget,
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
            "strict_tools": True,
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "max_tokens": 700,
        }
        provider_request_attempts += 1
        tool_http = await client.post("v2/chat", json=tool_request)
        _raise_for_status(tool_http)
        provider_calls += 1
        tool_response = tool_http.json()
        if not isinstance(tool_response, Mapping):
            raise CohereCompatibilityError("Cohere tool response is not an object")
        tool_call_id, arguments = _tool_call(tool_response)
        async with McpSession() as mcp:
            epicure_calls += 1
            result = await mcp.call_tool("find_pairings", arguments)
        if result.is_error:
            raise CohereCompatibilityError("Epicure returned a tool error")

        evidence = result.structured or {"text": result.text}
        evidence_projection, rendered_evidence = project_find_pairings_evidence(result.text)
        projected_response_schema = project_cohere_strict_schema(FINAL_SCHEMA)
        final_request = {
            "model": target.model_id,
            "stream": False,
            "messages": [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{PROMPT} {rendered_evidence} Return the JSON object now.",
                },
            ],
            "response_format": {
                "type": "json_object",
                "schema": projected_response_schema,
            },
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        provider_request_attempts += 1
        final_http = await client.post("v2/chat", json=final_request)
        _raise_for_status(final_http)
        provider_calls += 1
        final_response = final_http.json()
        if not isinstance(final_response, Mapping):
            raise CohereCompatibilityError("Cohere final response is not an object")
        output = _final(final_response)
        generation_ids = [
            str(tool_response.get("id") or ""),
            str(final_response.get("id") or ""),
        ]
        if any(not item for item in generation_ids):
            raise CohereCompatibilityError("Cohere response omitted a generation ID")
        usages = [_usage(tool_response), _usage(final_response)]
    except CohereAttemptError:
        raise
    except Exception as error:
        raise CohereAttemptError(
            error,
            provider_request_attempts=provider_request_attempts,
            provider_calls=provider_calls,
            epicure_calls=epicure_calls,
        ) from error
    return {
        "schema_version": ARM_SCHEMA_VERSION,
        "status": "smoke_passed",
        "provider": "cohere_direct",
        "requested_model_id": target.model_id,
        "catalog_sha256": target.catalog_sha256,
        "catalog_entry_sha256": target.catalog_entry_sha256,
        "catalog_entry": dict(target.catalog_entry),
        "prompt": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "final_system_prompt_sha256": sha256_text(FINAL_SYSTEM_PROMPT),
        "response_schema_sha256": sha256_json(FINAL_SCHEMA),
        "projected_response_schema_sha256": sha256_json(projected_response_schema),
        "epicure_tool_catalog_sha256": tool_catalog_sha,
        "projected_tool_sha256": sha256_json(tool),
        "synthesis_evidence_projection": evidence_projection,
        "synthesis_evidence_projection_sha256": sha256_json(evidence_projection),
        "synthesis_evidence_mode": "deterministic_find_pairings_graph_v2",
        "provider_contract": {
            "api": "Cohere Chat V2",
            "strict_tools": True,
            "tool_choice": "omitted_command_a_plus_rejects_tool_choice",
            "tool_invocation_required_by_prompt_and_validated_client_side": True,
            "thinking": "disabled_for_contract_smoke",
            "normalization": "separate_json_schema_request_after_tool_evidence",
            "response_format_with_tools_supported": False,
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
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "billed_input_tokens",
                "billed_output_tokens",
            )
        },
        "finish_reasons": [
            str(tool_response.get("finish_reason") or "unknown"),
            str(final_response.get("finish_reason") or "unknown"),
        ],
        "request_payload_sha256s": [
            sha256_json(tool_request),
            sha256_json(final_request),
        ],
        "wall_clock_latency_ms": round((time.monotonic() - started) * 1000),
        "provider_request_attempts": provider_request_attempts,
        "real_provider_calls": provider_calls,
        "real_epicure_calls": epicure_calls,
        "cost_status": "no_per_generation_cost_returned_by_provider",
        "official": False,
        "rank_eligible": False,
    }


async def execute(
    *,
    output_dir: Path,
    tool_catalog_path: Path,
    model_ids: Sequence[str],
    api_key: str,
    base_url: str = "https://api.cohere.com/",
) -> tuple[Path, dict[str, Any]]:
    tool, tool_sha = load_find_pairings(tool_catalog_path)
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        headers=_headers(api_key),
        timeout=240,
    ) as client:
        catalog_path, catalog = await freeze_catalog(client, output_dir / "catalog")
        targets = select_targets(catalog, model_ids)
        artifacts: list[dict[str, Any]] = []
        for target in targets:
            try:
                payload = await smoke_target(target, tool, tool_sha, client)
            except Exception as error:  # noqa: BLE001 - immutable redacted failure evidence
                attempt = error if isinstance(error, CohereAttemptError) else None
                payload = {
                    "schema_version": ARM_SCHEMA_VERSION,
                    "status": "failed",
                    "provider": "cohere_direct",
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
                f"cohere-{sha256_text(target.model_id)[:12]}",
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
        "provider": "cohere_direct",
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
    return _atomic_write(output_dir, "cohere-contract-summary", summary), summary


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--base-url", default="https://api.cohere.com/")
    parser.add_argument("--confirm", required=True)
    arguments = parser.parse_args(argv)
    if arguments.confirm != CONFIRMATION:
        raise CohereCompatibilityError(f"exact confirmation required: {CONFIRMATION}")
    api_key = os.environ.get("COHERE_API_KEY") or ""
    path, summary = asyncio.run(
        execute(
            output_dir=arguments.output_dir,
            tool_catalog_path=arguments.tool_catalog,
            model_ids=tuple(arguments.models or DEFAULT_MODELS),
            api_key=api_key,
            base_url=arguments.base_url,
        )
    )
    print(
        json.dumps(
            {
                "summary": str(path),
                "counts": summary["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
