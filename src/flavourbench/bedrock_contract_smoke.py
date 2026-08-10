"""Two-arm, rank-ineligible Amazon Bedrock + Epicure contract smoke.

Dry-run is the default and creates no SDK or MCP client.  Live execution is a
separate, exactly confirmed path with a locked local cost ledger.  This module
is intentionally not imported by the normal FlavourBench worker.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Protocol

import httpx

from .bedrock_auth import (
    BedrockClients,
    BedrockConfigurationError,
    BedrockLaneSettings,
    create_boto3_clients,
)
from .bedrock_contract import (
    LoadedSmokeContract,
    freeze_smoke_manifest,
    load_smoke_contract,
)
from .bedrock_manifest import BedrockManifestError, assert_public_catalog_safe
from .bedrock_provider import (
    BedrockConverseProvider,
    BedrockGenerationResult,
    BedrockGenerationSpec,
    BedrockInferenceConfig,
    BedrockRouteUnavailable,
    BedrockRuntimeClient,
    BedrockToolDefinition,
    BedrockToolExecution,
    project_bedrock_json_schema,
)
from .bedrock_smoke_ledger import (
    BedrockSmokeLedger,
    BedrockSmokeLedgerError,
    canonical_json,
    sha256_json,
)
from .config import get_settings
from .execution_policy import assert_legacy_paid_cli_allowed
from .mcp_client import McpSession, tool_catalog_sha256

EXECUTION_CONFIRMATION = "RUN_BEDROCK_EPICURE_CONTRACT_SMOKE_V8_SAFE_RESPONSE_HASH"
PROTOCOL_ID = "bedrock_epicure_contract_smoke_v8"
SMOKE_TEMPERATURE = 0.2
SMOKE_TOP_P: float | None = None
SMOKE_MAX_OUTPUT_TOKENS = 2048
SMOKE_TOOL_NAMES = ("find_pairings",)
DEFAULT_PROMPT = (
    "Design a make-ahead vegetarian starter for six using watermelon, green olive, and mint, "
    "with no added sugar. Explain salt, acid, texture, aromatic intensity, and tasting "
    "uncertainty."
)
OFF_SYSTEM_PROMPT = (
    "You are completing an authored, blinded culinary benchmark contract test without Epicure. "
    "Use your own reasoning. Return only the required structured answer object. Do not claim "
    "that an external culinary tool or database was consulted."
)
ON_SYSTEM_PROMPT = (
    "You are completing an authored, blinded culinary benchmark contract test with Epicure. "
    "Before answering, call at least one relevant Epicure tool using the provided schemas. "
    "Use the returned evidence, distinguish it from tasting uncertainty, and return only the "
    "required structured answer object."
)
ARTIFACT_SCHEMA_VERSION = "flavourbench-bedrock-epicure-smoke-arm-v8"
SUMMARY_SCHEMA_VERSION = "flavourbench-bedrock-epicure-smoke-summary-v8"
EPICURE_CONTRACT_SCHEMA_VERSION = "flavourbench-epicure-runtime-contract-v1"


class BedrockSmokeError(RuntimeError):
    """A smoke contract cannot be executed or safely resumed."""


@dataclass(frozen=True)
class FrozenEpicureIdentity:
    path: Path
    file_sha256: str
    release_id: str
    bundle_sha256: str
    application_sha256: str
    tool_schema_sha256: str
    ingredient_count: int
    embedding_dimensions: int

    def payload(self) -> dict[str, Any]:
        return {
            "contract_filename": self.path.name,
            "contract_file_sha256": self.file_sha256,
            "release_id": self.release_id,
            "bundle_sha256": self.bundle_sha256,
            "application_sha256": self.application_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "ingredient_count": self.ingredient_count,
            "embedding_dimensions": self.embedding_dimensions,
        }


@dataclass(frozen=True)
class FrozenEpicureToolCatalog:
    path: Path
    file_sha256: str
    raw_tool_schema_sha256: str
    bedrock_tool_schema_sha256: str
    raw_tools: tuple[dict[str, Any], ...]
    bedrock_tools: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "fixture_filename": self.path.name,
            "fixture_file_sha256": self.file_sha256,
            "raw_tool_schema_sha256": self.raw_tool_schema_sha256,
            "bedrock_tool_schema_sha256": self.bedrock_tool_schema_sha256,
            "raw_tool_count": len(self.raw_tools),
            "bedrock_tool_count": len(self.bedrock_tools),
            "bedrock_tool_names": [tool["name"] for tool in self.bedrock_tools],
            "projection": "aws-draft-2020-12-supported-subset-v1",
        }


def _load_epicure_contract(path: str | Path) -> FrozenEpicureIdentity:
    contract_path = Path(path)
    if contract_path.is_symlink() or not contract_path.is_file():
        raise BedrockSmokeError("Epicure runtime contract must be a regular file")
    raw = contract_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BedrockSmokeError("Epicure runtime contract is invalid JSON") from error
    if not isinstance(value, dict):
        raise BedrockSmokeError("Epicure runtime contract must be an object")
    assert_public_catalog_safe(value, path="$epicure_runtime_contract")
    if value.get("schema_version") != EPICURE_CONTRACT_SCHEMA_VERSION:
        raise BedrockSmokeError("unsupported Epicure runtime contract schema")
    digests = {
        key: str(value.get(key) or "")
        for key in ("bundle_sha256", "application_sha256", "tool_schema_sha256")
    }
    if any(
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests.values()
    ):
        raise BedrockSmokeError("Epicure runtime contract contains an invalid digest")
    ingredient_count = value.get("ingredient_count")
    dimensions = value.get("embedding_dimensions")
    if (
        not isinstance(ingredient_count, int)
        or isinstance(ingredient_count, bool)
        or ingredient_count <= 0
        or not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions <= 0
        or value.get("rank_eligible") is not False
    ):
        raise BedrockSmokeError("Epicure runtime contract lineage fields are invalid")
    return FrozenEpicureIdentity(
        path=contract_path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        release_id=str(value.get("release_id") or ""),
        bundle_sha256=digests["bundle_sha256"],
        application_sha256=digests["application_sha256"],
        tool_schema_sha256=digests["tool_schema_sha256"],
        ingredient_count=ingredient_count,
        embedding_dimensions=dimensions,
    )


def project_epicure_tool_catalog(
    catalog: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the prompt-relevant MCP subset while retaining raw-catalog lineage."""

    projected: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(catalog):
        if raw_tool.get("name") not in SMOKE_TOOL_NAMES:
            continue
        tool = copy.deepcopy(dict(raw_tool))
        schema = tool.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise BedrockSmokeError(f"Epicure tool catalog entry {index} has no input schema")
        tool["inputSchema"] = project_bedrock_json_schema(schema)
        projected.append(tool)
    if [tool.get("name") for tool in projected] != list(SMOKE_TOOL_NAMES):
        raise BedrockSmokeError("Epicure smoke tool subset is missing or out of order")
    return projected


def _load_epicure_tool_catalog(
    path: str | Path,
    *,
    expected_raw_sha256: str,
) -> FrozenEpicureToolCatalog:
    fixture_path = Path(path)
    if fixture_path.is_symlink() or not fixture_path.is_file():
        raise BedrockSmokeError("Epicure tool catalog fixture must be a regular file")
    raw = fixture_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BedrockSmokeError("Epicure tool catalog fixture is invalid JSON") from error
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(tool, dict) for tool in value)
    ):
        raise BedrockSmokeError("Epicure tool catalog fixture must be a non-empty list")
    assert_public_catalog_safe(value, path="$frozen_epicure_tool_catalog")
    raw_tools = [dict(tool) for tool in value]
    raw_digest = tool_catalog_sha256(raw_tools)
    if raw_digest != expected_raw_sha256:
        raise BedrockSmokeError("Epicure tool catalog fixture differs from the runtime contract")
    if fixture_path.name != f"tool-catalog-{raw_digest}.json":
        raise BedrockSmokeError("Epicure tool catalog fixture is not content-addressed")
    projected = project_epicure_tool_catalog(raw_tools)
    for definition in _tools_from_catalog(projected):
        definition.as_converse_tool()
    return FrozenEpicureToolCatalog(
        path=fixture_path,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_tool_schema_sha256=raw_digest,
        bedrock_tool_schema_sha256=tool_catalog_sha256(projected),
        raw_tools=tuple(copy.deepcopy(raw_tools)),
        bedrock_tools=tuple(copy.deepcopy(projected)),
    )


class McpLike(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


McpFactory = Callable[[], Any]
Attestor = Callable[[], Awaitable[dict[str, Any]]]


async def attest_epicure_provenance_document() -> dict[str, Any]:
    """Read only public lineage fields from the private Epicure HTTP endpoint."""

    settings = get_settings()
    provenance_url = os.environ.get("FLAVOURBENCH_EPICURE_PROVENANCE_URL")
    if not provenance_url:
        provenance_url = settings.mcp_url.removesuffix("/mcp").rstrip("/") + "/provenance"
    headers = {"Authorization": f"Bearer {settings.mcp_token}"} if settings.mcp_token else {}
    async with httpx.AsyncClient(headers=headers, timeout=settings.mcp_timeout_seconds) as client:
        response = await client.get(provenance_url)
        response.raise_for_status()
        raw = response.json()
    if not isinstance(raw, Mapping):
        raise BedrockSmokeError("Epicure provenance endpoint returned an invalid document")
    result = {
        "release_id": raw.get("release_id"),
        "bundle_sha256": raw.get("bundle_sha256"),
        "application_sha256": raw.get("application_sha256"),
        "ingredient_count": raw.get("ingredient_count"),
        "embedding_dimensions": raw.get("embedding_dimensions"),
    }
    assert_public_catalog_safe(result, path="$epicure_provenance")
    return result


@dataclass(frozen=True)
class SmokeBounds:
    max_output_tokens: int
    max_input_tokens_per_call: int
    max_tool_rounds: int
    max_tool_calls_per_round: int
    max_tool_calls_total: int
    max_tool_result_bytes: int
    max_cumulative_tool_result_bytes: int

    def __post_init__(self) -> None:
        if not 64 <= self.max_output_tokens <= 4096:
            raise BedrockSmokeError("max output tokens must be between 64 and 4096")
        if not 512 <= self.max_input_tokens_per_call <= 100_000:
            raise BedrockSmokeError("max input tokens per call must be between 512 and 100000")
        if not 1 <= self.max_tool_rounds <= 4:
            raise BedrockSmokeError("contract smoke permits one to four tool rounds")
        if not 1 <= self.max_tool_calls_per_round <= 4:
            raise BedrockSmokeError("contract smoke permits one to four calls per tool round")
        if not 1 <= self.max_tool_calls_total <= 8:
            raise BedrockSmokeError("contract smoke permits one to eight total tool calls")
        if self.max_tool_calls_total < self.max_tool_calls_per_round:
            raise BedrockSmokeError("total tool-call cap cannot be below the per-round cap")
        if not 1024 <= self.max_tool_result_bytes <= 65_536:
            raise BedrockSmokeError("per-tool result bytes must be between 1024 and 65536")
        if not self.max_tool_result_bytes <= self.max_cumulative_tool_result_bytes <= 131_072:
            raise BedrockSmokeError(
                "cumulative tool bytes must contain one result and stay bounded"
            )

    def payload(self) -> dict[str, int]:
        return asdict(self)


def _reservation_micros(
    loaded: LoadedSmokeContract,
    bounds: SmokeBounds,
    *,
    epicure_applied: bool,
) -> int:
    calls = bounds.max_tool_rounds + 1 if epicure_applied else 1
    price = loaded.contract.price
    micros = Decimal(calls * bounds.max_input_tokens_per_call) * Decimal(
        price.input_per_million_usd
    )
    micros += Decimal(calls * bounds.max_output_tokens) * Decimal(price.output_per_million_usd)
    return max(1, int(micros.quantize(Decimal("1"), rounding=ROUND_CEILING)))


def _delivered_rate_card_cost_micros(
    loaded: LoadedSmokeContract,
    responses: Sequence[Mapping[str, Any]],
) -> int | None:
    """Reconcile delivered responses even when normalized output validation fails."""

    if not responses or any(item.get("usage_complete") is not True for item in responses):
        return None
    price = loaded.contract.price
    total = Decimal("0")
    for response in responses:
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return None
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
        cache_read = usage.get("cacheReadInputTokens", 0)
        cache_write = usage.get("cacheWriteInputTokens", 0)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (input_tokens, output_tokens, cache_read, cache_write)
        ):
            return None
        if cache_read and price.cache_read_per_million_usd is None:
            return None
        if cache_write and price.cache_write_per_million_usd is None:
            return None
        total += Decimal(input_tokens) * Decimal(price.input_per_million_usd)
        total += Decimal(output_tokens) * Decimal(price.output_per_million_usd)
        total += Decimal(cache_read) * Decimal(price.cache_read_per_million_usd or "0")
        total += Decimal(cache_write) * Decimal(price.cache_write_per_million_usd or "0")
    return int(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _count_tokens_model_id(loaded: LoadedSmokeContract) -> str:
    model_ids = loaded.contract.expected_foundation_model_ids
    if len(model_ids) != 1 or not model_ids[0]:
        raise BedrockSmokeError("B1 CountTokens requires one frozen in-region foundation model ID")
    return model_ids[0]


def _artifact_payload_digest(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    return sha256_json(payload)


def _write_content_addressed(
    directory: Path,
    *,
    prefix: str,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    assert_public_catalog_safe(payload, path="$bedrock_smoke_artifact")
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    assert_public_catalog_safe(document, path="$bedrock_smoke_artifact")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    rendered = canonical_json(document) + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    if destination.exists():
        if destination.read_bytes() != rendered:
            temporary.unlink()
            raise BedrockSmokeError("content-addressed Bedrock artifact conflicts")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination, digest


def _load_artifact(directory: Path, filename: str, digest: str) -> dict[str, Any]:
    path = directory / filename
    if (
        not digest
        or len(digest) != 64
        or path.is_symlink()
        or not path.is_file()
        or not filename.endswith(f"-{digest}.json")
    ):
        raise BedrockSmokeError("linked Bedrock smoke artifact is missing or malformed")
    try:
        document = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise BedrockSmokeError("linked Bedrock smoke artifact is invalid JSON") from error
    if not isinstance(document, dict) or document.get("artifact_sha256") != digest:
        raise BedrockSmokeError("linked Bedrock smoke artifact identity is invalid")
    if _artifact_payload_digest(document) != digest:
        raise BedrockSmokeError("linked Bedrock smoke artifact digest does not verify")
    assert_public_catalog_safe(document, path="$linked_bedrock_smoke_artifact")
    return document


_AWS_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
_WHITESPACE = re.compile(r"\s+")
_ERROR_SANITIZER_VERSION = "aws-error-redaction-v1"


def _aws_error_response(error: BaseException) -> Mapping[str, Any] | None:
    """Find the first botocore-style response without persisting exception text."""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        if isinstance(response, Mapping):
            return response
        current = current.__cause__ or current.__context__
    return None


def _safe_aws_error_details(error: BaseException) -> dict[str, Any]:
    """Return bounded, public-safe AWS diagnostics and a raw-message digest."""

    response = _aws_error_response(error)
    if response is None:
        return {
            "aws_error_code": None,
            "aws_http_status": None,
            "aws_request_id": None,
            "aws_error_message_sha256": None,
            "aws_error_message_sanitized": None,
            "aws_error_sanitizer_version": _ERROR_SANITIZER_VERSION,
        }
    error_block = response.get("Error")
    error_block = error_block if isinstance(error_block, Mapping) else {}
    metadata = response.get("ResponseMetadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_message = str(error_block.get("Message") or "")
    sanitized = _WHITESPACE.sub(" ", _AWS_ACCOUNT_ID.sub("<account-redacted>", raw_message)).strip()
    if len(sanitized) > 512:
        sanitized = sanitized[:509] + "..."
    result = {
        "aws_error_code": str(error_block.get("Code") or "") or None,
        "aws_http_status": (
            metadata.get("HTTPStatusCode")
            if isinstance(metadata.get("HTTPStatusCode"), int)
            else None
        ),
        "aws_request_id": str(metadata.get("RequestId") or "") or None,
        "aws_error_message_sha256": (
            hashlib.sha256(raw_message.encode("utf-8")).hexdigest() if raw_message else None
        ),
        "aws_error_message_sanitized": sanitized or None,
        "aws_error_sanitizer_version": _ERROR_SANITIZER_VERSION,
    }
    try:
        assert_public_catalog_safe(result, path="$sanitized_aws_error")
    except BedrockManifestError:
        result["aws_error_message_sanitized"] = "<diagnostic-redacted>"
        assert_public_catalog_safe(result, path="$sanitized_aws_error")
    return result


def _safe_usage(value: object) -> dict[str, int]:
    usage = value if isinstance(value, Mapping) else {}
    result: dict[str, int] = {}
    for key in (
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
    ):
        raw = usage.get(key, 0)
        result[key] = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else 0
    return result


def _safe_response_sha256(value: object) -> tuple[str, str]:
    """Hash SDK responses without retaining non-JSON runtime objects."""

    try:
        return sha256_json(value), "canonical_json"
    except (TypeError, ValueError):
        pass

    def project(item: object) -> object:
        if item is None or isinstance(item, str | int | float | bool):
            return item
        if isinstance(item, Decimal):
            return {"decimal": str(item)}
        if isinstance(item, bytes | bytearray):
            raw = bytes(item)
            return {
                "binary_byte_count": len(raw),
                "binary_sha256": hashlib.sha256(raw).hexdigest(),
            }
        if isinstance(item, Mapping):
            return {str(key): project(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            return [project(child) for child in item]
        rendered = repr(item).encode("utf-8", errors="replace")
        return {
            "python_type": f"{type(item).__module__}.{type(item).__qualname__}",
            "repr_sha256": hashlib.sha256(rendered).hexdigest(),
        }

    return sha256_json(project(value)), "sanitized_sdk_projection"


def _exception_type_chain(error: BaseException) -> list[str]:
    chain: list[str] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited and len(chain) < 8:
        visited.add(id(current))
        chain.append(f"{type(current).__module__}.{type(current).__qualname__}")
        current = current.__cause__ or current.__context__
    return chain


def _assert_anthropic_use_case_ready(control_client: Any) -> None:
    """Check the free control-plane prerequisite without retaining form data."""

    try:
        response = control_client.get_use_case_for_model_access()
    except Exception as error:
        diagnostic = _safe_aws_error_details(error)
        code = diagnostic.get("aws_error_code")
        message = str(diagnostic.get("aws_error_message_sanitized") or "").lower()
        if code == "ResourceNotFoundException" or "use case details" in message:
            raise BedrockSmokeError(
                "Anthropic Bedrock use-case details are not submitted; no inference "
                "reservation was created"
            ) from error
        raise BedrockSmokeError(
            "Anthropic Bedrock use-case preflight failed before inference: "
            f"{code or type(error).__name__}"
        ) from error
    if not isinstance(response, Mapping):
        raise BedrockSmokeError(
            "Anthropic Bedrock use-case preflight returned a non-object response"
        )
    form_data = response.get("formData")
    if not isinstance(form_data, str | bytes | bytearray) or len(form_data) < 10:
        raise BedrockSmokeError(
            "Anthropic Bedrock use-case preflight did not confirm submitted details"
        )
    # Deliberately do not hash, serialize, log, or return the account form.


class LedgerRuntimeClient:
    """Prove each input bound, then fsync a paid delivery boundary."""

    def __init__(
        self,
        runtime: BedrockRuntimeClient,
        ledger: BedrockSmokeLedger,
        *,
        run_key: str,
        arm_id: str,
        reservation_id: str,
        reservation_micros: int,
        expected_target_id: str,
        count_tokens_model_id: str,
        max_input_tokens_per_call: int,
        max_converse_calls: int,
    ) -> None:
        self.runtime = runtime
        self.ledger = ledger
        self.run_key = run_key
        self.arm_id = arm_id
        self.reservation_id = reservation_id
        self.reservation_micros = reservation_micros
        self.expected_target_id = expected_target_id
        self.count_tokens_model_id = count_tokens_model_id
        self.max_input_tokens_per_call = max_input_tokens_per_call
        self.max_converse_calls = max_converse_calls
        self.call_count = 0
        self.count_tokens_attempt_count = 0
        self.counted_input_tokens: list[int] = []
        self.response_evidence: list[dict[str, Any]] = []

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.ledger.append(
            event_type,
            run_key=self.run_key,
            arm_id=self.arm_id,
            reservation_id=self.reservation_id,
            reservation_micros=self.reservation_micros,
            payload=payload,
        )

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        if kwargs.get("modelId") != self.expected_target_id:
            raise BedrockSmokeError("runtime payload differs from the frozen Bedrock target")
        call_index = self.call_count + 1
        if call_index > self.max_converse_calls:
            raise BedrockSmokeError("Converse call count exceeds the frozen reservation")
        count_input = {
            key: copy.deepcopy(kwargs[key])
            for key in (
                "messages",
                "system",
                "toolConfig",
                "additionalModelRequestFields",
            )
            if key in kwargs
        }
        if "messages" not in count_input:
            raise BedrockSmokeError("Converse payload has no messages to count")
        self.count_tokens_attempt_count += 1
        count_attempt = self.count_tokens_attempt_count
        count_payload = {
            "modelId": self.count_tokens_model_id,
            "input": {"converse": count_input},
        }
        self._append(
            "count_tokens_request_started",
            {
                "call_index": call_index,
                "count_tokens_attempt": count_attempt,
                "count_tokens_model_id": self.count_tokens_model_id,
                "count_tokens_payload_sha256": sha256_json(count_payload),
            },
        )
        try:
            count_response = self.runtime.count_tokens(**count_payload)
        except Exception as error:
            diagnostic = _safe_aws_error_details(error)
            self._append(
                "count_tokens_failed_pre_send",
                {
                    "call_index": call_index,
                    "count_tokens_attempt": count_attempt,
                    "failure_class": type(error).__name__,
                    **diagnostic,
                },
            )
            raise BedrockSmokeError("CountTokens preflight failed before paid Converse") from error
        if not isinstance(count_response, Mapping):
            self._append(
                "count_tokens_failed_pre_send",
                {
                    "call_index": call_index,
                    "count_tokens_attempt": count_attempt,
                    "failure_class": "NonMappingCountTokensResponse",
                },
            )
            raise BedrockSmokeError("CountTokens returned a non-object response")
        count = count_response.get("inputTokens")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            self._append(
                "count_tokens_failed_pre_send",
                {
                    "call_index": call_index,
                    "count_tokens_attempt": count_attempt,
                    "failure_class": "InvalidInputTokenCount",
                    "count_tokens_response_sha256": sha256_json(count_response),
                },
            )
            raise BedrockSmokeError("CountTokens returned an invalid inputTokens value")
        metadata = count_response.get("ResponseMetadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        status = metadata.get("HTTPStatusCode", 200)
        if status != 200:
            self._append(
                "count_tokens_failed_pre_send",
                {
                    "call_index": call_index,
                    "count_tokens_attempt": count_attempt,
                    "failure_class": "NonSuccessCountTokensStatus",
                    "http_status": status if isinstance(status, int) else None,
                },
            )
            raise BedrockSmokeError("CountTokens returned a non-success status")
        admitted = count <= self.max_input_tokens_per_call
        self._append(
            "count_tokens_response_received",
            {
                "call_index": call_index,
                "count_tokens_attempt": count_attempt,
                "aws_request_id": str(metadata.get("RequestId") or "") or None,
                "input_tokens": count,
                "max_input_tokens_per_call": self.max_input_tokens_per_call,
                "admitted_for_paid_converse": admitted,
                "count_tokens_response_sha256": sha256_json(count_response),
            },
        )
        if not admitted:
            raise BedrockSmokeError("CountTokens input exceeds the frozen per-call reservation")
        self.counted_input_tokens.append(count)
        self._append(
            "converse_request_started",
            {
                "call_index": call_index,
                "request_payload_sha256": sha256_json(kwargs),
                "requested_target_id": self.expected_target_id,
            },
        )
        # This counter means a durable paid-call boundary exists. It is set
        # only after CountTokens admission and the fsynced ledger append.
        self.call_count = call_index
        try:
            response = self.runtime.converse(**kwargs)
        except Exception as error:
            diagnostic = _safe_aws_error_details(error)
            self._append(
                "converse_delivery_uncertain",
                {
                    "call_index": call_index,
                    "failure_class": type(error).__name__,
                    **diagnostic,
                },
            )
            raise
        if not isinstance(response, Mapping):
            self._append(
                "converse_delivery_uncertain",
                {"call_index": call_index, "failure_class": "NonMappingResponse"},
            )
            return response
        metadata = response.get("ResponseMetadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        metrics = response.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        raw_usage = response.get("usage")
        usage_complete = isinstance(raw_usage, Mapping) and all(
            isinstance(raw_usage.get(key), int)
            and not isinstance(raw_usage.get(key), bool)
            and raw_usage.get(key) >= 0
            for key in ("inputTokens", "outputTokens", "totalTokens")
        )
        response_sha256, response_hash_mode = _safe_response_sha256(response)
        evidence = {
            "call_index": call_index,
            "aws_request_id": str(metadata.get("RequestId") or "") or None,
            "http_status": (
                metadata.get("HTTPStatusCode")
                if isinstance(metadata.get("HTTPStatusCode"), int)
                else None
            ),
            "returned_model_id": str(response.get("modelId") or response.get("model") or "")
            or None,
            "stop_reason": str(response.get("stopReason") or "") or None,
            "usage": _safe_usage(response.get("usage")),
            "usage_complete": usage_complete,
            "service_latency_ms": (
                metrics.get("latencyMs") if isinstance(metrics.get("latencyMs"), int) else None
            ),
            "response_sha256": response_sha256,
            "response_hash_mode": response_hash_mode,
        }
        self._append("converse_response_received", evidence)
        self.response_evidence.append(evidence)
        return response


@dataclass
class EpicureExecutor:
    mcp: McpLike
    allowed_tools: frozenset[str]
    max_result_bytes: int
    max_cumulative_bytes: int
    cumulative_bytes: int = 0
    traces: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.traces = []

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> BedrockToolExecution:
        if name not in self.allowed_tools:
            raise BedrockSmokeError("model requested a tool outside the frozen Epicure catalog")
        result = await self.mcp.call_tool(name, dict(arguments))
        structured = getattr(result, "structured", {})
        text = str(getattr(result, "text", "") or "")
        content: object = structured if isinstance(structured, dict) and structured else text
        rendered = canonical_json(content)
        byte_count = len(rendered)
        result_digest = hashlib.sha256(rendered).hexdigest()
        is_error = bool(getattr(result, "is_error", False))
        accepted = (
            byte_count <= self.max_result_bytes
            and self.cumulative_bytes + byte_count <= self.max_cumulative_bytes
        )
        if accepted:
            self.cumulative_bytes += byte_count
            returned_content = content
        else:
            is_error = True
            returned_content = {
                "error": "Epicure tool result exceeded the frozen smoke byte budget",
                "original_result_sha256": result_digest,
                "original_result_bytes": byte_count,
            }
        assert self.traces is not None
        self.traces.append(
            {
                "name": name,
                "arguments": dict(arguments),
                "result": content if accepted else None,
                "result_sha256": result_digest,
                "result_bytes": byte_count,
                "latency_ms": int(getattr(result, "latency_ms", 0) or 0),
                "is_error": is_error,
                "accepted_within_byte_budget": accepted,
            }
        )
        return BedrockToolExecution(returned_content, is_error=is_error)


def _tools_from_catalog(catalog: list[dict[str, Any]]) -> tuple[BedrockToolDefinition, ...]:
    definitions: list[BedrockToolDefinition] = []
    names: set[str] = set()
    for tool in catalog:
        name = str(tool.get("name") or "")
        schema = tool.get("inputSchema")
        if not name or name in names or not isinstance(schema, Mapping):
            raise BedrockSmokeError("Epicure MCP returned an invalid or duplicate tool schema")
        names.add(name)
        definitions.append(
            BedrockToolDefinition(
                name=name,
                description=str(tool.get("description") or f"Epicure tool {name}"),
                input_schema=dict(schema),
                strict=True,
            )
        )
    if not definitions:
        raise BedrockSmokeError("Epicure MCP returned an empty tool catalog")
    return tuple(definitions)


def _source_payload(
    loaded: LoadedSmokeContract,
    *,
    prompt: str,
    bounds: SmokeBounds,
    epicure_identity: FrozenEpicureIdentity,
    frozen_tool_catalog: FrozenEpicureToolCatalog,
    attested_epicure_identity: Mapping[str, Any],
    raw_tool_catalog: list[dict[str, Any]],
    bedrock_tool_catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_ID,
        "decoding": {
            "temperature": SMOKE_TEMPERATURE,
            "top_p": SMOKE_TOP_P,
        },
        "manifest_filename": loaded.manifest_path.name,
        "manifest_sha256": loaded.manifest_sha256,
        "catalog_filename": str(loaded.document.get("catalog_filename") or ""),
        "catalog_sha256": loaded.catalog.catalog_sha256,
        "capability_price_evidence_filename": loaded.evidence_path.name,
        "capability_price_evidence_sha256": loaded.evidence_sha256,
        "canonical_model_id": loaded.contract.canonical_model_id,
        "bedrock_target_id": loaded.contract.bedrock_target_id,
        "endpoint_contract_sha256": loaded.contract.sha256,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_class": "newly_authored_non_public_non_pii_engineering_task",
        "bounds": bounds.payload(),
        "epicure_contract": epicure_identity.payload(),
        "epicure_tool_catalog_fixture": frozen_tool_catalog.payload(),
        "attested_epicure_identity": dict(attested_epicure_identity),
        "epicure_raw_tool_catalog": copy.deepcopy(raw_tool_catalog),
        "epicure_raw_tool_schema_sha256": tool_catalog_sha256(raw_tool_catalog),
        "epicure_bedrock_tool_catalog": copy.deepcopy(bedrock_tool_catalog),
        "epicure_bedrock_tool_schema_sha256": tool_catalog_sha256(bedrock_tool_catalog),
    }


def _result_payload(result: BedrockGenerationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["rank_eligible"] = False
    payload["billing_actual_cost_micros"] = None
    payload["billing_actual_reconciliation_status"] = "not_reconciled"
    return payload


def _validate_result_bounds(
    result: BedrockGenerationResult,
    *,
    condition: str,
    bounds: SmokeBounds,
    reservation_micros: int,
) -> None:
    call_limit = bounds.max_tool_rounds + 1 if condition == "epicure_on" else 1
    if len(result.response_latencies_ms) > call_limit:
        raise BedrockSmokeError("Bedrock response count exceeded its reservation model")
    if result.usage.input_tokens > call_limit * bounds.max_input_tokens_per_call:
        raise BedrockSmokeError("Bedrock input usage exceeded its reserved token bound")
    if result.usage.output_tokens > call_limit * bounds.max_output_tokens:
        raise BedrockSmokeError("Bedrock output usage exceeded its reserved token bound")
    estimated = result.cost.estimated_cost_micros
    if not result.cost.estimate_complete or estimated is None:
        raise BedrockSmokeError("Bedrock rate-card estimate is incomplete")
    if estimated > reservation_micros:
        raise BedrockSmokeError("Bedrock rate-card estimate exceeded its reservation")
    if result.rank_eligible:
        raise BedrockSmokeError("contract smoke unexpectedly became rank-eligible")
    if condition == "epicure_off" and result.tool_traces:
        raise BedrockSmokeError("Epicure-off arm returned a tool trace")
    if condition == "epicure_on" and not result.tool_traces:
        raise BedrockSmokeError("Epicure-on contract arm did not call Epicure")


def _terminal_artifact(
    ledger: BedrockSmokeLedger,
    *,
    run_key: str,
    arm_id: str,
    artifact_directory: Path,
) -> dict[str, Any] | None:
    entries = ledger.arm_entries(run_key=run_key, arm_id=arm_id)
    reservations = [entry for entry in entries if entry["event_type"] == "reservation_created"]
    for reservation in reversed(reservations):
        reservation_id = reservation["reservation_id"]
        related = [entry for entry in entries if entry["reservation_id"] == reservation_id]
        terminal = next(
            (
                entry
                for entry in related
                if entry["event_type"]
                in {
                    "reservation_settled_rate_card_estimate",
                    "reservation_held_uncertain",
                    "reservation_released_pre_send",
                    "reservation_released_service_rejection",
                }
            ),
            None,
        )
        recorded = next(
            (
                entry
                for entry in reversed(related)
                if entry["event_type"] == "arm_artifact_recorded"
            ),
            None,
        )
        if terminal and terminal["event_type"] == "reservation_settled_rate_card_estimate":
            artifact = _load_artifact(
                artifact_directory,
                str(terminal.get("artifact_filename") or ""),
                str(terminal.get("artifact_sha256") or ""),
            )
            if artifact.get("status") != "complete":
                raise BedrockSmokeError(
                    f"arm {arm_id} has a terminal delivered invalid response; "
                    "paid replay is blocked"
                )
            return artifact
        if terminal and terminal["event_type"] == "reservation_held_uncertain":
            raise BedrockSmokeError(
                f"arm {arm_id} has an unresolved uncertain reservation; paid replay is blocked"
            )
        if terminal and terminal["event_type"] == "reservation_released_service_rejection":
            raise BedrockSmokeError(
                f"arm {arm_id} has a terminal provider route rejection; paid replay is blocked"
            )
        if terminal:
            continue
        reservation_micros = int(reservation["reservation_micros"])
        if recorded and recorded.get("status") == "complete":
            artifact = _load_artifact(
                artifact_directory,
                str(recorded.get("artifact_filename") or ""),
                str(recorded.get("artifact_sha256") or ""),
            )
            estimated = recorded.get("rate_card_estimated_cost_micros")
            if not isinstance(estimated, int) or estimated < 0 or estimated > reservation_micros:
                raise BedrockSmokeError("recoverable artifact has invalid cost evidence")
            ledger.append(
                "reservation_settled_rate_card_estimate",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={
                    "rate_card_estimated_cost_micros": estimated,
                    "billing_actual_reconciliation_status": "not_reconciled",
                    "artifact_filename": recorded["artifact_filename"],
                    "artifact_sha256": recorded["artifact_sha256"],
                    "recovered_after_artifact_fsync": True,
                },
            )
            return artifact
        requests = [entry for entry in related if entry["event_type"] == "converse_request_started"]
        if not requests:
            ledger.append(
                "reservation_released_pre_send",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={"reason": "restart_found_no_converse_request_boundary"},
            )
            continue
        ledger.append(
            "reservation_held_uncertain",
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
            payload={
                "reason": "restart_found_provider_request_without_settled_complete_artifact",
                "artifact_filename": recorded.get("artifact_filename") if recorded else None,
                "artifact_sha256": recorded.get("artifact_sha256") if recorded else None,
                "billing_actual_reconciliation_status": "uncertain_not_reconciled",
            },
        )
        raise BedrockSmokeError(
            f"arm {arm_id} has ambiguous provider delivery; full reservation is held"
        )
    return None


async def _execute_arm(
    *,
    loaded: LoadedSmokeContract,
    settings: BedrockLaneSettings,
    runtime: BedrockRuntimeClient,
    ledger: BedrockSmokeLedger,
    artifact_directory: Path,
    run_key: str,
    condition: str,
    prompt: str,
    bounds: SmokeBounds,
    frozen_epicure_identity: FrozenEpicureIdentity,
    frozen_tool_catalog: FrozenEpicureToolCatalog,
    attested_epicure_identity: Mapping[str, Any],
    raw_tool_catalog: list[dict[str, Any]],
    bedrock_tool_catalog: list[dict[str, Any]],
    mcp: McpLike,
) -> dict[str, Any]:
    arm_id = sha256_json({"run_key": run_key, "condition": condition})
    recovered = _terminal_artifact(
        ledger,
        run_key=run_key,
        arm_id=arm_id,
        artifact_directory=artifact_directory,
    )
    if recovered is not None:
        return recovered
    epicure_applied = condition == "epicure_on"
    reservation_micros = _reservation_micros(loaded, bounds, epicure_applied=epicure_applied)
    attempt_index = ledger.next_attempt_index(run_key=run_key, arm_id=arm_id)
    reservation_id = sha256_json(
        {"run_key": run_key, "arm_id": arm_id, "attempt_index": attempt_index}
    )
    ledger.reserve(
        settings=settings,
        run_key=run_key,
        arm_id=arm_id,
        reservation_id=reservation_id,
        reservation_micros=reservation_micros,
        payload={
            "condition": condition,
            "attempt_index": attempt_index,
            "reservation_basis": "frozen_rate_card_worst_case_token_bound",
            "billing_actual_reconciliation_status": "not_reconciled",
            "manifest_sha256": loaded.manifest_sha256,
            "endpoint_contract_sha256": loaded.contract.sha256,
        },
    )
    journaled_runtime = LedgerRuntimeClient(
        runtime,
        ledger,
        run_key=run_key,
        arm_id=arm_id,
        reservation_id=reservation_id,
        reservation_micros=reservation_micros,
        expected_target_id=loaded.contract.bedrock_target_id,
        count_tokens_model_id=_count_tokens_model_id(loaded),
        max_input_tokens_per_call=bounds.max_input_tokens_per_call,
        max_converse_calls=(bounds.max_tool_rounds + 1 if epicure_applied else 1),
    )
    tools = _tools_from_catalog(bedrock_tool_catalog) if epicure_applied else ()
    executor = EpicureExecutor(
        mcp=mcp,
        allowed_tools=frozenset(tool.name for tool in tools),
        max_result_bytes=bounds.max_tool_result_bytes,
        max_cumulative_bytes=bounds.max_cumulative_tool_result_bytes,
    )
    provider = BedrockConverseProvider(
        journaled_runtime,
        loaded.contract,
        tool_executor=executor if epicure_applied else None,
        max_tool_rounds=bounds.max_tool_rounds,
        max_tool_calls_per_round=bounds.max_tool_calls_per_round,
        max_tool_calls_total=bounds.max_tool_calls_total,
    )
    source = _source_payload(
        loaded,
        prompt=prompt,
        bounds=bounds,
        epicure_identity=frozen_epicure_identity,
        frozen_tool_catalog=frozen_tool_catalog,
        attested_epicure_identity=attested_epicure_identity,
        raw_tool_catalog=raw_tool_catalog,
        bedrock_tool_catalog=bedrock_tool_catalog,
    )
    try:
        result = await provider.generate(
            BedrockGenerationSpec(
                arm_id=arm_id,
                canonical_model_id=loaded.contract.canonical_model_id,
                prompt=prompt,
                system_prompt=ON_SYSTEM_PROMPT if epicure_applied else OFF_SYSTEM_PROMPT,
                inference=BedrockInferenceConfig(
                    max_tokens=bounds.max_output_tokens,
                    temperature=SMOKE_TEMPERATURE,
                    top_p=SMOKE_TOP_P,
                ),
                tools=tools,
                request_metadata={
                    "flavourbench_protocol": PROTOCOL_ID,
                    "flavourbench_condition": condition,
                },
            )
        )
        _validate_result_bounds(
            result,
            condition=condition,
            bounds=bounds,
            reservation_micros=reservation_micros,
        )
        artifact_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "complete",
            "run_key": run_key,
            "arm_id": arm_id,
            "condition": condition,
            "epicure_applied": epicure_applied,
            "rank_eligible": False,
            "official": False,
            "source": source,
            "generation": _result_payload(result),
            "complete_epicure_mcp_trace": executor.traces if epicure_applied else [],
            "count_tokens_preflight": {
                "model_id": journaled_runtime.count_tokens_model_id,
                "max_input_tokens_per_call": bounds.max_input_tokens_per_call,
                "input_tokens": list(journaled_runtime.counted_input_tokens),
                "free_api_attempts": journaled_runtime.count_tokens_attempt_count,
            },
            "reservation_micros": reservation_micros,
            "rate_card_estimated_cost_micros": result.cost.estimated_cost_micros,
            "cost_classification": "frozen_rate_card_estimate_not_billing_actual",
            "billing_actual_cost_micros": None,
            "billing_actual_reconciliation_status": "not_reconciled",
        }
        path, digest = _write_content_addressed(
            artifact_directory,
            prefix=f"bedrock-smoke-{condition}",
            payload=artifact_payload,
        )
        ledger.append(
            "arm_artifact_recorded",
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
            payload={
                "status": "complete",
                "artifact_filename": path.name,
                "artifact_sha256": digest,
                "rate_card_estimated_cost_micros": result.cost.estimated_cost_micros,
            },
        )
        ledger.append(
            "reservation_settled_rate_card_estimate",
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
            payload={
                "rate_card_estimated_cost_micros": result.cost.estimated_cost_micros,
                "billing_actual_reconciliation_status": "not_reconciled",
                "artifact_filename": path.name,
                "artifact_sha256": digest,
                "recovered_after_artifact_fsync": False,
            },
        )
        return {**artifact_payload, "artifact_sha256": digest}
    except Exception as error:
        requests_started = journaled_runtime.call_count
        service_rejected = isinstance(error, BedrockRouteUnavailable)
        delivered_estimate = _delivered_rate_card_cost_micros(
            loaded, journaled_runtime.response_evidence
        )
        delivered_invalid = delivered_estimate is not None
        pre_send = not requests_started and not service_rejected
        failure_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": (
                "failed_pre_generation_rejection"
                if service_rejected
                else "failed_delivered_invalid_response"
                if delivered_invalid
                else "failed_uncertain"
                if requests_started
                else "failed_pre_send"
            ),
            "run_key": run_key,
            "arm_id": arm_id,
            "condition": condition,
            "epicure_applied": epicure_applied,
            "rank_eligible": False,
            "official": False,
            "source": source,
            "failure_class": type(error).__name__,
            "failure_type_chain": _exception_type_chain(error),
            "failure_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "provider_error": _safe_aws_error_details(error),
            "converse_calls_started": requests_started,
            "delivered_response_evidence": copy.deepcopy(journaled_runtime.response_evidence),
            "complete_epicure_mcp_trace": executor.traces if epicure_applied else [],
            "count_tokens_preflight": {
                "model_id": journaled_runtime.count_tokens_model_id,
                "max_input_tokens_per_call": bounds.max_input_tokens_per_call,
                "input_tokens": list(journaled_runtime.counted_input_tokens),
                "free_api_attempts": journaled_runtime.count_tokens_attempt_count,
            },
            "reservation_micros": reservation_micros,
            "rate_card_estimated_cost_micros": (
                0 if service_rejected or pre_send else delivered_estimate
            ),
            "cost_classification": (
                "service_rejected_pre_generation_rate_card_zero"
                if service_rejected
                else "delivered_invalid_response_rate_card_estimate_not_billing_actual"
                if delivered_invalid
                else "failure_pre_send_rate_card_zero"
                if pre_send
                else "uncertain_full_reservation_held"
            ),
            "billing_actual_cost_micros": None,
            "billing_actual_reconciliation_status": (
                "not_reconciled"
                if service_rejected or delivered_invalid or pre_send
                else "uncertain_not_reconciled"
            ),
        }
        path, digest = _write_content_addressed(
            artifact_directory,
            prefix=f"bedrock-smoke-{condition}-failure",
            payload=failure_payload,
        )
        ledger.append(
            "arm_artifact_recorded",
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
            payload={
                "status": failure_payload["status"],
                "artifact_filename": path.name,
                "artifact_sha256": digest,
                "rate_card_estimated_cost_micros": (
                    0 if service_rejected or pre_send else delivered_estimate
                ),
            },
        )
        if service_rejected:
            ledger.append(
                "reservation_released_service_rejection",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={
                    "reason": "provider_returned_explicit_pre_generation_route_rejection",
                    "artifact_filename": path.name,
                    "artifact_sha256": digest,
                    "rate_card_estimated_cost_micros": 0,
                    "billing_actual_reconciliation_status": "not_reconciled",
                },
            )
        elif delivered_invalid:
            ledger.append(
                "reservation_settled_rate_card_estimate",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={
                    "reason": "provider_response_delivered_but_normalized_output_invalid",
                    "rate_card_estimated_cost_micros": delivered_estimate,
                    "billing_actual_reconciliation_status": "not_reconciled",
                    "artifact_filename": path.name,
                    "artifact_sha256": digest,
                },
            )
        elif requests_started:
            ledger.append(
                "reservation_held_uncertain",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={
                    "reason": "provider_call_started_without_complete_cost-reconciled_arm",
                    "artifact_filename": path.name,
                    "artifact_sha256": digest,
                    "billing_actual_reconciliation_status": "uncertain_not_reconciled",
                },
            )
        else:
            ledger.append(
                "reservation_released_pre_send",
                run_key=run_key,
                arm_id=arm_id,
                reservation_id=reservation_id,
                reservation_micros=reservation_micros,
                payload={
                    "reason": "failure_before_converse_request_boundary",
                    "artifact_filename": path.name,
                    "artifact_sha256": digest,
                },
            )
        raise BedrockSmokeError(
            f"{condition} failed; immutable failure evidence was recorded at {path.name}"
        ) from error


def _run_key(
    loaded: LoadedSmokeContract,
    *,
    prompt: str,
    bounds: SmokeBounds,
    epicure_identity: FrozenEpicureIdentity,
    tool_catalog: FrozenEpicureToolCatalog,
) -> str:
    return sha256_json(
        {
            "protocol": PROTOCOL_ID,
            "manifest_sha256": loaded.manifest_sha256,
            "endpoint_contract_sha256": loaded.contract.sha256,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "off_system_prompt_sha256": hashlib.sha256(
                OFF_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "on_system_prompt_sha256": hashlib.sha256(ON_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "bounds": bounds.payload(),
            "decoding": {
                "temperature": SMOKE_TEMPERATURE,
                "top_p": SMOKE_TOP_P,
            },
            "epicure_identity": epicure_identity.payload(),
            "epicure_tool_catalog_fixture": tool_catalog.payload(),
            "count_tokens_model_id": _count_tokens_model_id(loaded),
        }
    )


async def execute_contract_smoke(
    arguments: argparse.Namespace,
    *,
    runtime_client: BedrockRuntimeClient | None = None,
    client_factory: Callable[[BedrockLaneSettings], BedrockClients] = create_boto3_clients,
    mcp_factory: McpFactory = McpSession,
    attestor: Attestor = attest_epicure_provenance_document,
    lane_settings: BedrockLaneSettings | None = None,
) -> dict[str, Any]:
    if arguments.confirm != EXECUTION_CONFIRMATION:
        raise BedrockSmokeError(f"--execute requires --confirm {EXECUTION_CONFIRMATION}")
    if arguments.prompt != DEFAULT_PROMPT:
        raise BedrockSmokeError(
            "B1 smoke is frozen to the authored non-PII prompt; custom prompts are prohibited"
        )
    assert_public_catalog_safe({"prompt": arguments.prompt}, path="$smoke_prompt")
    settings = lane_settings or BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized:
        raise BedrockSmokeError("Bedrock live execution is not explicitly enabled and authorized")
    if settings.stage != "contract_smoke":
        raise BedrockSmokeError("this runner is restricted to the contract_smoke stage")
    loaded = _loaded_from_arguments(arguments)
    if settings.region != loaded.contract.region:
        raise BedrockSmokeError("AWS ingress region differs from the frozen endpoint contract")
    if settings.profile_scope != loaded.contract.profile_scope:
        raise BedrockSmokeError("profile scope differs from the frozen endpoint contract")
    if runtime_client is None:
        clients = client_factory(settings)
        _assert_anthropic_use_case_ready(clients.control)
        runtime_client = clients.runtime
    bounds = _bounds_from_arguments(arguments)
    configured_epicure = _load_epicure_contract(arguments.epicure_contract)
    frozen_tool_catalog = _load_epicure_tool_catalog(
        arguments.epicure_tool_catalog,
        expected_raw_sha256=configured_epicure.tool_schema_sha256,
    )
    run_key = _run_key(
        loaded,
        prompt=arguments.prompt,
        bounds=bounds,
        epicure_identity=configured_epicure,
        tool_catalog=frozen_tool_catalog,
    )
    ledger = BedrockSmokeLedger(arguments.ledger)
    artifact_directory = Path(arguments.output_dir) / "arms"
    try:
        attested = await attestor()
    except Exception as error:
        raise BedrockSmokeError(
            "Epicure provenance preflight failed before any Bedrock inference call: "
            f"{type(error).__name__}"
        ) from error
    for key, configured_key in (
        ("release_id", "release_id"),
        ("bundle_sha256", "bundle_sha256"),
        ("application_sha256", "application_sha256"),
        ("ingredient_count", "ingredient_count"),
        ("embedding_dimensions", "embedding_dimensions"),
    ):
        if attested.get(key) != getattr(configured_epicure, configured_key):
            raise BedrockSmokeError("live Epicure identity differs from the frozen configuration")
    assert_public_catalog_safe(attested, path="$attested_epicure_identity")

    try:
        mcp_context = mcp_factory()
        async with mcp_context as mcp:
            try:
                tool_catalog = await mcp.list_tools()
            except Exception as error:
                raise BedrockSmokeError(
                    "Epicure tool-catalog preflight failed before any Bedrock inference call: "
                    f"{type(error).__name__}"
                ) from error
            actual_tool_sha256 = tool_catalog_sha256(tool_catalog)
            if actual_tool_sha256 != configured_epicure.tool_schema_sha256:
                raise BedrockSmokeError("live Epicure tool catalog differs from its frozen digest")
            if tool_catalog != list(frozen_tool_catalog.raw_tools):
                raise BedrockSmokeError("live Epicure tool catalog differs from the frozen fixture")
            bedrock_tool_catalog = project_epicure_tool_catalog(tool_catalog)
            if tool_catalog_sha256(
                bedrock_tool_catalog
            ) != frozen_tool_catalog.bedrock_tool_schema_sha256 or bedrock_tool_catalog != list(
                frozen_tool_catalog.bedrock_tools
            ):
                raise BedrockSmokeError(
                    "Bedrock Epicure schema projection differs from the frozen fixture"
                )
            definitions = _tools_from_catalog(bedrock_tool_catalog)
            # Validate every projected Bedrock schema before the Epicure-off
            # arm can spend money. MCP retains the full raw constraints.
            for definition in definitions:
                definition.as_converse_tool()
            attested = {
                **attested,
                "tool_schema_sha256": actual_tool_sha256,
                "tool_count": len(tool_catalog),
            }
            assert_public_catalog_safe(attested, path="$complete_attested_epicure_identity")
            assert runtime_client is not None
            off = await _execute_arm(
                loaded=loaded,
                settings=settings,
                runtime=runtime_client,
                ledger=ledger,
                artifact_directory=artifact_directory,
                run_key=run_key,
                condition="epicure_off",
                prompt=arguments.prompt,
                bounds=bounds,
                frozen_epicure_identity=configured_epicure,
                frozen_tool_catalog=frozen_tool_catalog,
                attested_epicure_identity=attested,
                raw_tool_catalog=tool_catalog,
                bedrock_tool_catalog=bedrock_tool_catalog,
                mcp=mcp,
            )
            on = await _execute_arm(
                loaded=loaded,
                settings=settings,
                runtime=runtime_client,
                ledger=ledger,
                artifact_directory=artifact_directory,
                run_key=run_key,
                condition="epicure_on",
                prompt=arguments.prompt,
                bounds=bounds,
                frozen_epicure_identity=configured_epicure,
                frozen_tool_catalog=frozen_tool_catalog,
                attested_epicure_identity=attested,
                raw_tool_catalog=tool_catalog,
                bedrock_tool_catalog=bedrock_tool_catalog,
                mcp=mcp,
            )
    except (BedrockSmokeError, BedrockSmokeLedgerError):
        raise
    except Exception as error:
        raise BedrockSmokeError(
            "Epicure MCP session failed; completed Bedrock arm artifacts, if any, "
            "remain resumable: "
            f"{type(error).__name__}"
        ) from error

    summary_payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "decoding": {
            "temperature": SMOKE_TEMPERATURE,
            "top_p": SMOKE_TOP_P,
        },
        "status": "complete",
        "run_key": run_key,
        "rank_eligible": False,
        "official": False,
        "manifest_filename": loaded.manifest_path.name,
        "manifest_sha256": loaded.manifest_sha256,
        "catalog_sha256": loaded.catalog.catalog_sha256,
        "endpoint_contract_sha256": loaded.contract.sha256,
        "epicure_tool_catalog_fixture": frozen_tool_catalog.payload(),
        "canonical_model_id": loaded.contract.canonical_model_id,
        "bedrock_target_id": loaded.contract.bedrock_target_id,
        "count_tokens_model_id": _count_tokens_model_id(loaded),
        "arms": [
            {
                "condition": "epicure_off",
                "artifact_sha256": off["artifact_sha256"],
                "rate_card_estimated_cost_micros": off["rate_card_estimated_cost_micros"],
                "count_tokens_input_counts": off["count_tokens_preflight"]["input_tokens"],
            },
            {
                "condition": "epicure_on",
                "artifact_sha256": on["artifact_sha256"],
                "rate_card_estimated_cost_micros": on["rate_card_estimated_cost_micros"],
                "count_tokens_input_counts": on["count_tokens_preflight"]["input_tokens"],
            },
        ],
        "billing_actual_cost_micros": None,
        "billing_actual_reconciliation_status": "not_reconciled",
        "ledger": ledger.descriptor(),
    }
    summary_path, summary_digest = _write_content_addressed(
        Path(arguments.output_dir) / "summaries",
        prefix="bedrock-epicure-contract-smoke-summary",
        payload=summary_payload,
    )
    return {
        "operation": "bedrock_epicure_contract_smoke",
        "status": "complete",
        "run_key": run_key,
        "summary": str(summary_path),
        "summary_sha256": summary_digest,
        "bedrock_converse_calls_made": sum(
            len(arm["generation"]["response_latencies_ms"]) for arm in (off, on)
        ),
        "bedrock_count_tokens_calls_made": sum(
            int(arm["count_tokens_preflight"]["free_api_attempts"]) for arm in (off, on)
        ),
        "mcp_calls_made": len(on["complete_epicure_mcp_trace"]),
        "rank_eligible": False,
        "billing_actual_reconciliation_status": "not_reconciled",
        "governed_exposure_usd": str(ledger.exposure().governed_exposure_usd),
    }


def _loaded_from_arguments(arguments: argparse.Namespace) -> LoadedSmokeContract:
    if not all(
        (
            arguments.manifest,
            arguments.catalog,
            arguments.evidence,
            arguments.epicure_contract,
            arguments.epicure_tool_catalog,
            arguments.expected_manifest_sha256,
        )
    ):
        raise BedrockSmokeError(
            "--manifest, --catalog, --evidence, --epicure-contract, "
            "--epicure-tool-catalog, and --expected-manifest-sha256 are required"
        )
    loaded = load_smoke_contract(
        manifest_path=arguments.manifest,
        catalog_path=arguments.catalog,
        evidence_path=arguments.evidence,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
    )
    if not loaded.contract.temperature_top_p_mutually_exclusive:
        raise BedrockSmokeError(
            "v8 smoke requires frozen evidence that temperature and top_p are mutually exclusive"
        )
    return loaded


def _bounds_from_arguments(arguments: argparse.Namespace) -> SmokeBounds:
    return SmokeBounds(
        max_output_tokens=arguments.max_output_tokens,
        max_input_tokens_per_call=arguments.max_input_tokens_per_call,
        max_tool_rounds=arguments.max_tool_rounds,
        max_tool_calls_per_round=arguments.max_tool_calls_per_round,
        max_tool_calls_total=arguments.max_tool_calls_total,
        max_tool_result_bytes=arguments.max_tool_result_bytes,
        max_cumulative_tool_result_bytes=arguments.max_cumulative_tool_result_bytes,
    )


def dry_run_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.prompt != DEFAULT_PROMPT:
        raise BedrockSmokeError(
            "B1 smoke is frozen to the authored non-PII prompt; custom prompts are prohibited"
        )
    assert_public_catalog_safe({"prompt": arguments.prompt}, path="$smoke_prompt")
    loaded = _loaded_from_arguments(arguments)
    bounds = _bounds_from_arguments(arguments)
    epicure_identity = _load_epicure_contract(arguments.epicure_contract)
    frozen_tool_catalog = _load_epicure_tool_catalog(
        arguments.epicure_tool_catalog,
        expected_raw_sha256=epicure_identity.tool_schema_sha256,
    )
    run_key = _run_key(
        loaded,
        prompt=arguments.prompt,
        bounds=bounds,
        epicure_identity=epicure_identity,
        tool_catalog=frozen_tool_catalog,
    )
    settings = BedrockLaneSettings.from_environ()
    if settings.region and settings.region != loaded.contract.region:
        raise BedrockSmokeError("configured ingress region differs from the frozen contract")
    if settings.profile_scope != loaded.contract.profile_scope:
        raise BedrockSmokeError("configured profile scope differs from the frozen contract")
    ledger_path = Path(arguments.ledger)
    exposure = None
    if ledger_path.exists():
        exposure = str(BedrockSmokeLedger(ledger_path).exposure().governed_exposure_usd)
    plan = {
        "operation": "bedrock_epicure_contract_smoke_dry_run",
        "protocol": PROTOCOL_ID,
        "decoding": {
            "temperature": SMOKE_TEMPERATURE,
            "top_p": SMOKE_TOP_P,
        },
        "dry_run": True,
        "external_calls_made": 0,
        "bedrock_inference_calls_made": 0,
        "epicure_mcp_calls_made": 0,
        "run_key": run_key,
        "manifest_filename": loaded.manifest_path.name,
        "manifest_sha256": loaded.manifest_sha256,
        "catalog_sha256": loaded.catalog.catalog_sha256,
        "capability_price_evidence_sha256": loaded.evidence_sha256,
        "endpoint_contract_sha256": loaded.contract.sha256,
        "epicure_tool_catalog_fixture": frozen_tool_catalog.payload(),
        "canonical_model_id": loaded.contract.canonical_model_id,
        "bedrock_target_id": loaded.contract.bedrock_target_id,
        "count_tokens_model_id": _count_tokens_model_id(loaded),
        "region": loaded.contract.region,
        "profile_scope": loaded.contract.profile_scope,
        "bounds": bounds.payload(),
        "arm_reservations_micros": {
            "epicure_off": _reservation_micros(loaded, bounds, epicure_applied=False),
            "epicure_on": _reservation_micros(loaded, bounds, epicure_applied=True),
        },
        "hard_cap_usd": str(settings.hard_cap_usd),
        "effective_stage_cap_usd": str(settings.effective_stage_cap_usd),
        "existing_governed_exposure_usd": exposure,
        "rank_eligible": False,
        "billing_actual_reconciliation_status": "not_reconciled",
        "execute_confirmation": EXECUTION_CONFIRMATION,
    }
    assert_public_catalog_safe(plan, path="$bedrock_smoke_dry_run")
    return plan


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Freeze, plan, or exactly confirm one Bedrock + Epicure contract smoke"
    )
    value.add_argument("--freeze-contract", action="store_true")
    value.add_argument("--catalog", type=Path)
    value.add_argument("--evidence", type=Path)
    value.add_argument("--epicure-contract", type=Path)
    value.add_argument("--epicure-tool-catalog", type=Path)
    value.add_argument("--target-id", default="")
    value.add_argument("--canonical-model-id", default="")
    value.add_argument("--manifest", type=Path)
    value.add_argument("--expected-manifest-sha256", default="")
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/bedrock/smoke"))
    value.add_argument("--ledger", type=Path, default=Path("artifacts/bedrock/smoke/ledger.jsonl"))
    value.add_argument("--prompt", default=DEFAULT_PROMPT)
    value.add_argument("--max-output-tokens", type=int, default=SMOKE_MAX_OUTPUT_TOKENS)
    value.add_argument("--max-input-tokens-per-call", type=int, default=12_000)
    value.add_argument("--max-tool-rounds", type=int, default=2)
    value.add_argument("--max-tool-calls-per-round", type=int, default=2)
    value.add_argument("--max-tool-calls-total", type=int, default=4)
    value.add_argument("--max-tool-result-bytes", type=int, default=16_384)
    value.add_argument("--max-cumulative-tool-result-bytes", type=int, default=32_768)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--confirm", default="")
    return value


def run(argv: Sequence[str] | None = None) -> int:
    assert_legacy_paid_cli_allowed("flavourbench-bedrock-contract-smoke")
    arguments = parser().parse_args(argv)
    try:
        if arguments.freeze_contract:
            if arguments.execute:
                raise BedrockSmokeError("--freeze-contract and --execute are mutually exclusive")
            if not all(
                (
                    arguments.catalog,
                    arguments.evidence,
                    arguments.target_id,
                    arguments.canonical_model_id,
                )
            ):
                raise BedrockSmokeError(
                    "contract freeze requires catalog, evidence, target ID, and canonical model ID"
                )
            path = freeze_smoke_manifest(
                catalog_path=arguments.catalog,
                evidence_path=arguments.evidence,
                target_id=arguments.target_id,
                canonical_model_id=arguments.canonical_model_id,
                output_directory=arguments.output_dir,
            )
            print(
                json.dumps(
                    {
                        "operation": "freeze_bedrock_smoke_contract",
                        "external_calls_made": 0,
                        "output": str(path),
                        "manifest_sha256": path.stem.rsplit("-", 1)[-1],
                        "rank_eligible": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.execute:
            result = asyncio.run(execute_contract_smoke(arguments))
        else:
            result = dry_run_plan(arguments)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        BedrockConfigurationError,
        BedrockManifestError,
        BedrockSmokeError,
        BedrockSmokeLedgerError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
