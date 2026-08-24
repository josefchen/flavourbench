from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .config import get_settings

MCP_PROTOCOL_VERSION = "2025-06-18"


def _decode_response(body: str) -> dict[str, Any]:
    stripped = body.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for line in stripped.splitlines():
        if line.startswith("data:"):
            value = line[5:].strip()
            if value and value != "[DONE]":
                return json.loads(value)
    raise RuntimeError("MCP returned an unsupported response")


def _read_result(envelope: dict[str, Any]) -> Any:
    if envelope.get("error"):
        error = envelope["error"]
        message = (
            error.get("message", "MCP protocol error") if isinstance(error, dict) else str(error)
        )
        raise RuntimeError(message)
    if "result" not in envelope:
        raise RuntimeError("MCP response was missing a result")
    return envelope["result"]


def tool_catalog_sha256(tools: list[dict[str, Any]]) -> str:
    canonical = json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class McpToolResult:
    text: str
    structured: dict[str, Any]
    latency_ms: int
    is_error: bool


class McpSession:
    def __init__(self) -> None:
        settings = get_settings()
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if settings.mcp_token:
            headers["Authorization"] = f"Bearer {settings.mcp_token}"
        self.url = settings.mcp_url
        self.client = httpx.AsyncClient(headers=headers, timeout=settings.mcp_timeout_seconds)
        self.session_id: str | None = None
        self.request_id = 0

    async def __aenter__(self) -> McpSession:
        envelope, headers = await self._post(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "epicure-flavourbench", "version": "0.1.0"},
            },
            include_session=False,
        )
        _read_result(envelope)
        self.session_id = headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError("MCP did not establish a session")
        await self.client.post(
            self.url,
            headers=self._session_headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.session_id:
            try:
                await self.client.delete(self.url, headers=self._session_headers())
            except httpx.HTTPError:
                pass
        await self.client.aclose()

    def _session_headers(self) -> dict[str, str]:
        if not self.session_id:
            return {}
        return {
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "MCP-Session-Id": self.session_id,
        }

    async def _post(
        self,
        method: str,
        params: dict[str, Any],
        *,
        include_session: bool = True,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        self.request_id += 1
        response = await self.client.post(
            self.url,
            headers=self._session_headers() if include_session else {},
            json={"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params},
        )
        response.raise_for_status()
        return _decode_response(response.text), response.headers

    async def list_tools(self) -> list[dict[str, Any]]:
        envelope, _ = await self._post("tools/list", {})
        result = _read_result(envelope)
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise RuntimeError("MCP tool catalog was invalid")
        return [tool for tool in tools if isinstance(tool, dict)]

    async def attest_runtime(
        self,
        *,
        expected: Mapping[str, str],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Attest the Epicure identity used by this exact MCP session."""

        provenance_url = self.url.removesuffix("/mcp").rstrip("/") + "/provenance"
        response = await self.client.get(provenance_url)
        response.raise_for_status()
        provenance = response.json()
        if not isinstance(provenance, dict):
            raise RuntimeError("Epicure provenance endpoint returned an invalid document")
        session_tools = tools if tools is not None else await self.list_tools()
        actual = {
            "release_id": provenance.get("release_id"),
            "bundle_sha256": provenance.get("bundle_sha256"),
            "application_sha256": provenance.get("application_sha256"),
            "tool_schema_sha256": tool_catalog_sha256(session_tools),
        }
        for field, expected_value in expected.items():
            if expected_value in {
                "",
                "unfrozen",
                "unresolved",
                "unresolved-1790-development-only",
            }:
                raise RuntimeError(f"frozen Epicure {field} is unresolved")
            if actual.get(field) != expected_value:
                raise RuntimeError(
                    f"runtime Epicure {field} differs from the frozen season contract"
                )
        if int(provenance.get("ingredient_count") or 0) <= 0:
            raise RuntimeError("Epicure provenance did not attest a non-empty ingredient bundle")
        return {
            **actual,
            "ingredient_count": int(provenance["ingredient_count"]),
            "embedding_dimensions": int(provenance.get("embedding_dimensions") or 0),
            "tool_count": len(session_tools),
            "mcp_protocol_version": MCP_PROTOCOL_VERSION,
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        start = time.monotonic()
        envelope, _ = await self._post("tools/call", {"name": name, "arguments": arguments})
        result = _read_result(envelope)
        if not isinstance(result, dict):
            raise RuntimeError("MCP tool result was invalid")
        structured = result.get("structuredContent")
        structured = structured if isinstance(structured, dict) else {}
        text = ""
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text") or "")
        if not text and structured:
            text = json.dumps(structured, sort_keys=True)
        return McpToolResult(
            text=text,
            structured=structured,
            latency_ms=round((time.monotonic() - start) * 1000),
            is_error=bool(result.get("isError")),
        )


async def attest_epicure_runtime(
    expected: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify the live MCP application, data bundle, release, and tool contract."""

    settings = get_settings()
    async with McpSession() as mcp:
        tools = await mcp.list_tools()
        frozen = expected or {
            "release_id": settings.epicure_release_id,
            "bundle_sha256": settings.epicure_bundle_sha256,
            "application_sha256": settings.epicure_application_sha256,
            "tool_schema_sha256": settings.epicure_tool_schema_sha256,
        }
        return await mcp.attest_runtime(expected=frozen, tools=tools)
