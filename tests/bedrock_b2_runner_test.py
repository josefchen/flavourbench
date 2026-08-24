from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from bedrock_b2_manifest_test import _endpoint, _task_ids, _write

from flavourbench.bedrock_auth import BedrockLaneSettings
from flavourbench.bedrock_b2_manifest import build_b2_manifest, select_tasks, write_b2_manifest
from flavourbench.bedrock_b2_runner import (
    BedrockB2RunnerError,
    _execute_with_clients,
    counterbalanced_arms,
    execute_b2,
)
from flavourbench.mcp_client import tool_catalog_sha256

ROOT = Path(__file__).parents[1]
REAL_ENDPOINT = ROOT / (
    "artifacts/bedrock/contracts/"
    "bedrock-smoke-manifest-13e55aa50acea7ac5ba06ccf055e4d19eadb01e7a92007b996bce41d5a8293f3.json"
)
REAL_EPICURE = ROOT / "contracts/epicure/exploratory-unmatched-1790-runtime.json"
REAL_TOOLS = ROOT / (
    "contracts/epicure/"
    "tool-catalog-666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd.json"
)


def _manifest(tmp_path: Path) -> tuple[Path, str]:
    endpoint = _endpoint(tmp_path / "endpoint.json")
    epicure = _write(tmp_path / "epicure.json", {"release": "real"})
    tools = _write(tmp_path / "tools.json", [{"name": "find_pairings"}])
    document = build_b2_manifest(
        endpoint_contract_paths=[endpoint],
        epicure_contract_path=epicure,
        tool_contract_path=tools,
        tasks=select_tasks(_task_ids()),
        frozen_at="2026-07-16T00:00:00Z",
    )
    return write_b2_manifest(document, tmp_path / "manifests"), document["content_address"][
        "digest"
    ]


def test_counterbalance_keeps_complete_pairs(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    manifest = json.loads(path.read_bytes())
    arms = counterbalanced_arms(manifest)
    assert len(arms) == 16
    assert all(
        {arms[index]["condition"], arms[index + 1]["condition"]} == {"epicure_off", "epicure_on"}
        for index in range(0, len(arms), 2)
    )
    assert {arms[index]["condition"] for index in range(0, len(arms), 2)} == {
        "epicure_off",
        "epicure_on",
    }


def test_public_execution_boundary_forbids_adapter_injection() -> None:
    parameters = inspect.signature(execute_b2).parameters
    assert "arm_runner" not in parameters
    assert "runtime" not in parameters
    assert "mcp_factory" not in parameters
    assert "attestor" not in parameters


@pytest.mark.asyncio
async def test_requires_distinct_b2_confirmation_before_any_call(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    with pytest.raises(BedrockB2RunnerError, match="exact execution confirmation"):
        await execute_b2(
            manifest_path=path,
            expected_manifest_sha256=digest,
            output_directory=tmp_path / "run",
            ledger_path=tmp_path / "ledger.jsonl",
            endpoint_directory=tmp_path,
            catalog_directory=tmp_path,
            evidence_directory=tmp_path,
            epicure_contract_directory=tmp_path,
            tool_contract_directory=tmp_path,
            confirmation="RUN_B1",
        )


class _B2Runtime:
    def __init__(self) -> None:
        self.converse_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        self.count_calls.append(kwargs)
        return {
            "inputTokens": 900,
            "ResponseMetadata": {
                "RequestId": f"count-{len(self.count_calls)}",
                "HTTPStatusCode": 200,
            },
        }

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        last_content = kwargs["messages"][-1]["content"]
        after_tool = any("toolResult" in block for block in last_content)
        if "toolConfig" in kwargs and not after_tool:
            return {
                "ResponseMetadata": {
                    "RequestId": f"tool-{len(self.converse_calls)}",
                    "HTTPStatusCode": 200,
                },
                "modelId": kwargs["modelId"],
                "stopReason": "tool_use",
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                "metrics": {"latencyMs": 1},
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": f"use-{len(self.converse_calls)}",
                                    "name": "find_pairings",
                                    "input": {"ingredients": ["tomato", "basil"]},
                                }
                            }
                        ],
                    }
                },
            }
        answer = {
            "answer_markdown": "A bounded culinary benchmark answer.",
            "ingredient_mentions": ["tomato", "basil"],
            "constraints_addressed": ["prompt constraints"],
            "uncertainties": ["final seasoning requires tasting"],
        }
        return {
            "ResponseMetadata": {
                "RequestId": f"final-{len(self.converse_calls)}",
                "HTTPStatusCode": 200,
            },
            "modelId": kwargs["modelId"],
            "stopReason": "end_turn",
            "usage": {"inputTokens": 200, "outputTokens": 80, "totalTokens": 280},
            "metrics": {"latencyMs": 1},
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": json.dumps(answer)}],
                }
            },
        }


class _ToolResult:
    structured = {"pairings": [{"ingredient": "basil", "score": 0.8}]}
    text = ""
    latency_ms = 1
    is_error = False


class _B2Mcp:
    def __init__(self) -> None:
        self.tools = json.loads(REAL_TOOLS.read_bytes())
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _B2Mcp:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _ToolResult:
        self.calls.append((name, arguments))
        return _ToolResult()


@pytest.mark.asyncio
async def test_private_test_harness_exercises_real_adapter_contract(tmp_path: Path) -> None:
    document = build_b2_manifest(
        endpoint_contract_paths=[REAL_ENDPOINT],
        epicure_contract_path=REAL_EPICURE,
        tool_contract_path=REAL_TOOLS,
        tasks=select_tasks(_task_ids()),
        frozen_at="2026-07-16T00:00:00Z",
    )
    digest = document["content_address"]["digest"]
    runtime = _B2Runtime()
    mcp = _B2Mcp()
    epicure = json.loads(REAL_EPICURE.read_bytes())

    async def attestor() -> dict[str, Any]:
        return {
            "release_id": epicure["release_id"],
            "bundle_sha256": epicure["bundle_sha256"],
            "application_sha256": epicure["application_sha256"],
            "ingredient_count": epicure["ingredient_count"],
            "embedding_dimensions": epicure["embedding_dimensions"],
        }

    settings = BedrockLaneSettings.from_environ(
        {
            "FLAVOURBENCH_BEDROCK_ENABLED": "true",
            "FLAVOURBENCH_BEDROCK_LIVE_AUTHORIZED": "true",
            "FLAVOURBENCH_BEDROCK_HARD_CAP_USD": "5000",
            "FLAVOURBENCH_BEDROCK_STAGE": "exploratory",
            "FLAVOURBENCH_BEDROCK_CONTRACT_SMOKE_EVIDENCE_SHA256": "a" * 64,
            "FLAVOURBENCH_BEDROCK_PROFILE_SCOPE": "global",
            "AWS_REGION": "eu-west-1",
        }
    )
    summary_path = await _execute_with_clients(
        manifest=document,
        expected_manifest_sha256=digest,
        output_directory=tmp_path / "run",
        ledger_path=tmp_path / "ledger.jsonl",
        settings=settings,
        runtime=runtime,
        mcp_factory=lambda: mcp,
        attestor=attestor,
        endpoint_directory=REAL_ENDPOINT.parent,
        catalog_directory=ROOT / "artifacts/bedrock/catalog",
        evidence_directory=ROOT / "contracts/evidence",
        epicure_contract_directory=REAL_EPICURE.parent,
        tool_contract_directory=REAL_TOOLS.parent,
    )

    summary = json.loads(summary_path.read_bytes())
    assert summary["counts"] == {"complete": 16, "failed": 0, "total": 16}
    assert len(runtime.converse_calls) == 24
    assert len(runtime.count_calls) == 24
    assert len(mcp.calls) == 8
    assert tool_catalog_sha256(mcp.tools) == epicure["tool_schema_sha256"]
    assert summary["official"] is False and summary["rank_eligible"] is False
