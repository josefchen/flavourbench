"""Real Bedrock + Epicure compatibility smokes for the Season 0 roster."""

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

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import (
    BEDROCK_FINAL_SCHEMA,
    project_bedrock_json_schema,
)
from .execution_policy import assert_legacy_paid_cli_allowed
from .mcp_client import McpSession, tool_catalog_sha256
from .real_task_bank import sha256_json, sha256_text

SCHEMA_VERSION = "flavourbench-season0-bedrock-compatibility-v1"
CONFIRMATION = "RUN_REAL_SEASON0_BEDROCK_EPICURE_COMPATIBILITY_V1"
SYSTEM_PROMPT = """You are completing a FlavourBench compatibility task. You must call the
available Epicure find_pairings tool exactly once before answering. Use the result critically.
Then return a complete culinary answer in plain Markdown and do not identify your model or
provider."""
PROMPT = (
    "Design two plausible savoury pairing directions for pear and explain one bridge ingredient."
)


class CompatibilityError(RuntimeError):
    """A compatibility run or its immutable inputs were invalid."""


@dataclass(frozen=True)
class CompatibilityTarget:
    display_name: str
    target_id: str
    target_arn: Mapping[str, Any]
    foundation_model_ids: tuple[str, ...]
    foundation_model_arns: tuple[Mapping[str, Any], ...]
    endpoint_kind: str
    catalog_sha256: str


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    data = rendered.encode("utf-8") + b"\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_bytes() != data:
            raise CompatibilityError("compatibility artifact content-address conflict")
        return path
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o600)
    return path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CompatibilityError(f"invalid JSON input: {path}") from error


def load_targets(roster_path: Path, catalog_path: Path) -> list[CompatibilityTarget]:
    roster = _load_json(roster_path)
    catalog = _load_json(catalog_path)
    if not isinstance(roster, Mapping) or not isinstance(catalog, Mapping):
        raise CompatibilityError("roster and catalog must be objects")
    targets = catalog.get("targets")
    slots = roster.get("slots")
    if not isinstance(targets, list) or not isinstance(slots, list):
        raise CompatibilityError("roster or catalog collections are invalid")
    by_id = {
        str(target["target_id"]): target
        for target in targets
        if isinstance(target, Mapping) and isinstance(target.get("target_id"), str)
    }
    catalog_sha = str(catalog.get("catalog_sha256") or "")
    selected: list[CompatibilityTarget] = []
    for slot in slots:
        if not isinstance(slot, Mapping) or slot.get("provider") != "bedrock":
            continue
        target_id = str(slot.get("endpoint_id") or "")
        target = by_id.get(target_id)
        if target is None:
            raise CompatibilityError(f"roster target is absent from catalog: {target_id}")
        selected.append(
            CompatibilityTarget(
                display_name=str(slot.get("canonical_name") or target.get("display_name") or ""),
                target_id=target_id,
                target_arn=dict(target["target_arn"]),
                foundation_model_ids=tuple(str(value) for value in target["foundation_model_ids"]),
                foundation_model_arns=tuple(
                    dict(value) for value in target["foundation_model_arns"]
                ),
                endpoint_kind=str(target.get("endpoint_kind") or ""),
                catalog_sha256=catalog_sha,
            )
        )
    return selected


def load_find_pairings(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise CompatibilityError("Epicure tool catalog must be an array")
    tools = [item for item in value if isinstance(item, Mapping)]
    match = next((item for item in tools if item.get("name") == "find_pairings"), None)
    if match is None or not isinstance(match.get("inputSchema"), Mapping):
        raise CompatibilityError("Epicure catalog has no valid find_pairings tool")
    projected = project_bedrock_json_schema(dict(match["inputSchema"]))
    return (
        {
            "name": "find_pairings",
            "description": str(match.get("description") or "Explore ingredient pairings"),
            "inputSchema": projected,
        },
        tool_catalog_sha256([dict(item) for item in tools]),
    )


def _message(response: Mapping[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    if not isinstance(message, Mapping):
        raise CompatibilityError("Bedrock response has no message")
    content = message.get("content")
    if not isinstance(content, list):
        raise CompatibilityError("Bedrock response message has invalid content")
    return {"role": str(message.get("role") or "assistant"), "content": content}


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, Mapping) and "text" in block
    )


def _normalize_final(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise CompatibilityError("Bedrock final answer is empty")
    return {
        "answer_markdown": cleaned,
        "ingredient_mentions": [],
        "constraints_addressed": [],
        "uncertainties": [],
    }


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(raw.get("inputTokens") or 0),
        "output_tokens": int(raw.get("outputTokens") or 0),
        "total_tokens": int(raw.get("totalTokens") or 0),
    }


async def _smoke_target(
    runtime: Any,
    target: CompatibilityTarget,
    tool: Mapping[str, Any],
    raw_tool_schema_sha256: str,
    *,
    artifact_schema_version: str = SCHEMA_VERSION,
    request_phase: str = "season0_compatibility",
) -> dict[str, Any]:
    started = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": PROMPT}]}]
    bedrock_tool = {
        "toolSpec": {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {"json": tool["inputSchema"]},
        }
    }
    common: dict[str, Any] = {
        "modelId": target.target_id,
        "system": [{"text": SYSTEM_PROMPT}],
        "inferenceConfig": {"maxTokens": 700, "temperature": 0.2},
        "requestMetadata": {
            "flavourbench_phase": request_phase,
            "flavourbench_target": sha256_text(target.target_id),
        },
    }
    usages: list[dict[str, int]] = []
    response_latencies_ms: list[int] = []
    tool_trace: list[dict[str, Any]] = []
    request_ids: list[str] = []
    output_json: dict[str, Any] | None = None
    async with McpSession() as mcp:
        for round_index in range(3):
            if round_index == 0:
                round_contract = {
                    "toolConfig": {
                        "tools": [bedrock_tool],
                        "toolChoice": {"tool": {"name": "find_pairings"}},
                    }
                }
            else:
                round_contract = {
                    "toolConfig": {
                        "tools": [bedrock_tool],
                        "toolChoice": {"auto": {}},
                    }
                }
            round_started = time.monotonic()
            response = await asyncio.to_thread(
                runtime.converse,
                **{**common, **round_contract, "messages": messages},
            )
            response_latencies_ms.append(round((time.monotonic() - round_started) * 1000))
            usages.append(_usage(response))
            metadata = response.get("ResponseMetadata")
            request_id = metadata.get("RequestId") if isinstance(metadata, Mapping) else None
            if request_id:
                request_ids.append(sha256_text(str(request_id)))
            assistant = _message(response)
            messages.append(assistant)
            stop_reason = str(response.get("stopReason") or "unknown")
            if stop_reason != "tool_use":
                if round_index == 0:
                    raise CompatibilityError("model ignored the required Epicure tool call")
                output_json = _normalize_final(_text(assistant))
                break
            content = assistant["content"]
            uses = [
                block["toolUse"]
                for block in content
                if isinstance(block, Mapping) and isinstance(block.get("toolUse"), Mapping)
            ]
            if len(uses) != 1 or uses[0].get("name") != "find_pairings":
                raise CompatibilityError("model did not make exactly one allowed Epicure call")
            use = uses[0]
            arguments = use.get("input")
            if not isinstance(arguments, Mapping):
                raise CompatibilityError("model returned invalid Epicure arguments")
            result = await mcp.call_tool("find_pairings", dict(arguments))
            content_value: object = result.structured or result.text
            tool_trace.append(
                {
                    "round_index": round_index,
                    "name": "find_pairings",
                    "arguments": dict(arguments),
                    "result": content_value,
                    "result_sha256": sha256_json(content_value),
                    "latency_ms": result.latency_ms,
                    "is_error": result.is_error,
                }
            )
            result_content = (
                [{"json": content_value}]
                if isinstance(content_value, Mapping)
                else [{"text": str(content_value)}]
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": str(use.get("toolUseId") or ""),
                                "content": result_content,
                                **({"status": "error"} if result.is_error else {}),
                            }
                        }
                    ],
                }
            )
    if output_json is None or not tool_trace or any(trace["is_error"] for trace in tool_trace):
        raise CompatibilityError("compatibility task lacked a valid final answer or Epicure trace")
    usage = {
        key: sum(round_usage[key] for round_usage in usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    return {
        "schema_version": artifact_schema_version,
        "status": "smoke_passed",
        "display_name": target.display_name,
        "requested_target_id": target.target_id,
        "returned_model_id": None,
        "expected_foundation_model_ids": list(target.foundation_model_ids),
        "target_arn": target.target_arn,
        "foundation_model_arns": list(target.foundation_model_arns),
        "endpoint_kind": target.endpoint_kind,
        "catalog_sha256": target.catalog_sha256,
        "prompt": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "response_schema_sha256": sha256_json(BEDROCK_FINAL_SCHEMA),
        "normalization_mode": "lossless_client_text_wrapper_v1",
        "provider_structured_output_required": False,
        "raw_epicure_tool_schema_sha256": raw_tool_schema_sha256,
        "projected_tool_schema_sha256": sha256_json(bedrock_tool),
        "output_json": output_json,
        "complete_epicure_trace": tool_trace,
        "usage": usage,
        "response_latencies_ms": response_latencies_ms,
        "wall_clock_latency_ms": round((time.monotonic() - started) * 1000),
        "request_id_sha256s": request_ids,
        "provider_calls": len(usages),
        "real_epicure_calls": len(tool_trace),
        "official": False,
        "rank_eligible": False,
    }


async def execute(
    *,
    runtime: Any,
    targets: Sequence[CompatibilityTarget],
    tool: Mapping[str, Any],
    raw_tool_schema_sha256: str,
    output_dir: Path,
    concurrency: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(target: CompatibilityTarget) -> tuple[CompatibilityTarget, dict[str, Any]]:
        async with semaphore:
            try:
                return target, await _smoke_target(runtime, target, tool, raw_tool_schema_sha256)
            except Exception as error:  # noqa: BLE001 - persist a redacted compatibility failure
                message = re.sub(r"(?<!\d)\d{12}(?!\d)", "<account-redacted>", str(error))
                return target, {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "display_name": target.display_name,
                    "requested_target_id": target.target_id,
                    "expected_foundation_model_ids": list(target.foundation_model_ids),
                    "catalog_sha256": target.catalog_sha256,
                    "error_type": type(error).__name__,
                    "error": message[:600],
                    "official": False,
                    "rank_eligible": False,
                }

    outcomes = await asyncio.gather(*(one(target) for target in targets))
    artifacts: list[dict[str, Any]] = []
    for target, payload in outcomes:
        path = _atomic_write(
            output_dir,
            f"compatibility-{sha256_text(target.target_id)[:12]}",
            payload,
        )
        artifacts.append(
            {
                "target_id": target.target_id,
                "display_name": target.display_name,
                "status": payload["status"],
                "path": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "error_type": payload.get("error_type"),
            }
        )
    return {
        "schema_version": "flavourbench-season0-bedrock-compatibility-summary-v1",
        "counts": {
            "targets": len(artifacts),
            "smoke_passed": sum(item["status"] == "smoke_passed" for item in artifacts),
            "failed": sum(item["status"] == "failed" for item in artifacts),
        },
        "artifacts": artifacts,
        "real_provider_calls": sum(
            int(payload.get("provider_calls") or 0) for _target, payload in outcomes
        ),
        "real_epicure_calls": sum(
            int(payload.get("real_epicure_calls") or 0) for _target, payload in outcomes
        ),
        "official": False,
        "rank_eligible": False,
    }


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-season0-compatibility")
    parser = argparse.ArgumentParser(description="Run real Bedrock + Epicure roster smokes")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--tool-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season0/compatibility"))
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--cap-usd", type=float, default=20.0)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.confirmation != CONFIRMATION:
        raise CompatibilityError(f"live compatibility requires --confirmation {CONFIRMATION}")
    if not 0 < args.cap_usd <= 20:
        raise CompatibilityError("compatibility smoke sub-cap must be in (0, 20]")
    if not 1 <= args.concurrency <= 8:
        raise CompatibilityError("compatibility concurrency must be between one and eight")
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized:
        raise CompatibilityError("Bedrock compatibility requires live authorization")
    if settings.hard_cap_usd < args.cap_usd:
        raise CompatibilityError("compatibility sub-cap exceeds the authorized Bedrock cap")
    targets = load_targets(args.roster, args.catalog)
    if len(targets) * 2 > args.cap_usd:
        raise CompatibilityError("conservative $2-per-target reservation exceeds sub-cap")
    tool, raw_tool_sha = load_find_pairings(args.tool_catalog)
    clients = create_boto3_clients(settings)
    summary = asyncio.run(
        execute(
            runtime=clients.runtime,
            targets=targets,
            tool=tool,
            raw_tool_schema_sha256=raw_tool_sha,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
        )
    )
    path = _atomic_write(args.output_dir, "compatibility-summary", summary)
    print(json.dumps({**summary, "summary_path": str(path)}, indent=2))


if __name__ == "__main__":
    run()
