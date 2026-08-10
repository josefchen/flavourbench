from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import httpx

from .config import get_settings
from .execution_policy import (
    DIRECT_TOOL_CONTRACT_PROTOCOL,
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOL_V1,
    MATCHED_EVIDENCE_PROTOCOL_V2,
    MATCHED_EVIDENCE_PROTOCOLS,
    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
    PORTABLE_TEXT_TOOL_PROTOCOL_V1,
)
from .mcp_client import McpSession
from .tool_contract import (
    TOOL_CONTRACT_NAME,
    TOOL_CONTRACT_SYSTEM_PROMPT,
)

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_markdown": {"type": "string", "minLength": 1},
        "ingredient_mentions": {"type": "array", "items": {"type": "string"}},
        "constraints_addressed": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_markdown",
        "ingredient_mentions",
        "constraints_addressed",
        "uncertainties",
    ],
    "additionalProperties": False,
}
FINAL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(FINAL_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
PLAIN_TEXT_RESPONSE_SCHEMA = {"type": "string", "minLength": 1}
PLAIN_TEXT_RESPONSE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(PLAIN_TEXT_RESPONSE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary reasoning benchmark.
Answer the user's exact culinary question. Prioritize practical cookability, explicit constraints,
coherent flavour logic, and calibrated claims. Never identify yourself, your developer, provider,
or model family. Never state or imply whether you used an external tool, retrieval system, MCP
server, or condition-specific resource. Do not claim that embedding similarity proves chemical
causation, enjoyment,
authenticity, safety, or universal culinary quality. Return only the requested structured object."""

PLAIN_TEXT_SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary reasoning
benchmark. Answer the user's exact culinary question. Prioritize practical cookability, explicit
constraints, coherent flavour logic, and calibrated claims. Never identify yourself, your
developer, provider, or model family. Never state or imply whether you used an external tool,
retrieval system, MCP server, or condition-specific resource. Do not claim that embedding
similarity proves chemical causation, enjoyment, authenticity, safety, or universal culinary
quality. Return only the final culinary answer in clear natural-language Markdown."""

EPICURE_PROMPT = """You have access to Epicure's read-only culinary evidence tools. Use them only
when relevant and integrate results critically. Tool outputs are learned statistical relationships,
not ground truth for safety, nutrition, culture, chemistry, or human preference. For open-ended
pairing or recipe design, prefer find_pairings before composing the final answer."""

MATCHED_EVIDENCE_PROTOCOL = MATCHED_EVIDENCE_PROTOCOL_V1
MATCHED_EVIDENCE_SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary
reasoning benchmark. Follow the staged instructions exactly. Answer the user's culinary question
only when asked for the final answer. Prioritize practical cookability, explicit constraints,
coherent flavour logic, and calibrated claims. If a culinary evidence tool catalog is exposed, use
it only when relevant and treat its learned statistical relationships as explanatory evidence, not
ground truth for safety, nutrition, culture, chemistry, authenticity, enjoyment, or human
preference. Never identify yourself, your developer, provider, model family, tool availability, or
evaluation condition."""
MATCHED_EVIDENCE_V2_SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary
reasoning benchmark. Follow the staged instructions exactly. Answer the user's culinary question
only when asked for the final answer. Prioritize practical cookability, explicit constraints,
coherent flavour logic, and calibrated claims. Statistical pairings, embeddings, networks, and
similarity scores may suggest candidates or corroborate a hypothesis; they do not establish causal
mechanisms, ingredient function, sensory intensity, safety, authenticity, enjoyment, or universal
quality. Resolve conflicts using culinary technique and physical principles, and state material
uncertainty. Never identify yourself, your developer, provider, model family, tool availability,
or evaluation condition."""
MATCHED_TOOL_ACCESS_SYSTEM_PROMPT = """You are participating in FlavourBench, a blinded culinary
reasoning benchmark. Answer the user's exact culinary question. Prioritize practical cookability,
explicit constraints, coherent flavour logic, and calibrated claims. If a culinary evidence tool
catalog is available and its evidence would materially improve the answer, use it critically. If
no such catalog is available or it is not useful, answer from culinary knowledge. Statistical
pairings, embeddings, networks, and similarity scores may suggest candidates or corroborate a
bounded hypothesis; they do not establish causal mechanisms, ingredient function, sensory
intensity, safety, authenticity, enjoyment, or universal quality. Never identify yourself, your
developer, provider, model family, tool availability, or evaluation condition."""
PORTABLE_TEXT_TOOL_SYSTEM_PROMPT = """You are participating in FlavourBench, an executable
culinary benchmark. Follow each staged instruction exactly. The benchmark may ask for one
provider-neutral Epicure tool request as text; emit only the requested JSON object in that turn.
Treat Epicure output as the benchmark's answer source for the exact query, while avoiding broader
claims about safety, nutrition, authenticity, enjoyment, or causal chemistry. In the final turn,
emit exactly one line in the form `FINAL_CHOICE: X`. Never identify yourself, your developer,
provider, model family, tool availability, or evaluation condition."""
MATCHED_PLANNING_INSTRUCTION = (
    "Draft a compact checklist of the constraints and culinary decisions needed for the final "
    "answer. Do not write the final answer yet and do not mention model identity or evaluation."
)
MATCHED_EVIDENCE_DECISION_INSTRUCTION = (
    "Decide whether external culinary evidence would materially improve the answer. If relevant "
    "evidence tools are exposed, call the most useful tool or tools now. Otherwise return one "
    "short evidence-needs note. Do not draft the final answer."
)
PORTABLE_NO_TOOL_INSTRUCTION = (
    "No external evidence is available in this condition. Return one short note stating what "
    "evidence would resolve the exact question. Do not answer the question yet."
)
PORTABLE_FINAL_CHOICE_INSTRUCTION = (
    "Answer now with no analysis or explanation. Return exactly one line in this form: "
    "FINAL_CHOICE: X, replacing X with A, B, C, or D."
)
PORTABLE_EPICURE_TOOL_NAMES = frozenset(
    {"neighbors", "pairing_score", "compare_on_axis", "cultural_profile"}
)
MATCHED_EVIDENCE_V2_FINAL_INSTRUCTION = (
    "Integrate the available information critically. Begin with culinary technique and physical "
    "principles. Use statistical pairing, embedding, network, or similarity evidence only to "
    "suggest candidates or corroborate a bounded hypothesis. Never infer binding, thickening, "
    "sweetness, acidity, safety, or a causal mechanism from similarity. Discard evidence that "
    "conflicts with the recipe constraints or established technique. Use calibrated language; "
    "do not say that the data proves or confirms a mechanistic or functional claim. "
)


def response_schema_sha256(final_response_mode: str = "structured_json") -> str:
    if final_response_mode == "structured_json":
        return FINAL_SCHEMA_SHA256
    if final_response_mode == "plain_text":
        return PLAIN_TEXT_RESPONSE_SCHEMA_SHA256
    raise ValueError(f"unsupported final response mode: {final_response_mode}")


def system_prompt_text(
    condition: str,
    final_response_mode: str = "structured_json",
    evidence_protocol: str = "legacy_v6",
) -> str:
    if condition not in {"epicure_off", "epicure_on"}:
        raise ValueError(f"unsupported benchmark condition: {condition}")
    if final_response_mode not in {"structured_json", "plain_text"}:
        raise ValueError(f"unsupported final response mode: {final_response_mode}")
    if evidence_protocol == MATCHED_EVIDENCE_PROTOCOL_V1:
        prompt = MATCHED_EVIDENCE_SYSTEM_PROMPT
    elif evidence_protocol == MATCHED_EVIDENCE_PROTOCOL_V2:
        prompt = MATCHED_EVIDENCE_V2_SYSTEM_PROMPT
    elif evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1:
        prompt = MATCHED_TOOL_ACCESS_SYSTEM_PROMPT
    elif evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1:
        prompt = PORTABLE_TEXT_TOOL_SYSTEM_PROMPT
    elif evidence_protocol == "legacy_v6":
        base = (
            SYSTEM_PROMPT if final_response_mode == "structured_json" else PLAIN_TEXT_SYSTEM_PROMPT
        )
        prompt = base + ("\n\n" + EPICURE_PROMPT if condition == "epicure_on" else "")
    else:
        raise ValueError(f"unsupported evidence protocol: {evidence_protocol}")
    return prompt


def system_prompt_sha256(
    condition: str,
    final_response_mode: str = "structured_json",
    evidence_protocol: str = "legacy_v6",
) -> str:
    prompt = system_prompt_text(condition, final_response_mode, evidence_protocol)
    return hashlib.sha256(prompt.encode()).hexdigest()


def _portable_tool_instruction(tools: list[dict[str, Any]]) -> str:
    catalog = [
        {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "arguments_schema": tool.get("inputSchema") or {"type": "object"},
        }
        for tool in tools
        if tool.get("name") in PORTABLE_EPICURE_TOOL_NAMES
    ]
    if {item["name"] for item in catalog} != PORTABLE_EPICURE_TOOL_NAMES:
        raise ProviderError("portable Epicure catalog is incomplete")
    return (
        "Select exactly one Epicure operation that resolves the user's exact question. Return "
        "only one JSON object with exactly two keys: name and arguments. The name must be from "
        "the catalog and arguments must satisfy its schema. Do not answer the question yet.\n\n"
        "EPICURE_CATALOG="
        + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _parse_portable_tool_request(text: str) -> tuple[str, dict[str, Any]]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ProviderError("portable Epicure selection was not one JSON object") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"name", "arguments"}
        or payload.get("name") not in PORTABLE_EPICURE_TOOL_NAMES
        or not isinstance(payload.get("arguments"), dict)
    ):
        raise ProviderError("portable Epicure selection has an invalid shape")
    return str(payload["name"]), dict(payload["arguments"])


@dataclass(frozen=True)
class ToolTrace:
    round_index: int
    name: str
    arguments: dict[str, Any]
    result: str
    latency_ms: int
    is_error: bool
    call_index: int = 0
    tool_call_id: str = ""
    structured_content: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationSpec:
    arm_id: str
    battle_id: str
    prompt: str
    category: str
    model_id: str
    model_name: str
    provider_slug: str
    condition: str
    idempotency_key: str
    final_response_mode: str = "structured_json"
    matched_planning: bool = False
    intermediate_max_tokens: int = 700
    required_tool_contract_max_intermediate_tokens: int = 2_048
    evidence_protocol: str = "legacy_v6"
    intermediate_reasoning_effort: str | None = None
    final_reasoning_effort: str | None = None
    required_tool_contract_protocol: str = DIRECT_TOOL_CONTRACT_PROTOCOL
    required_tool_contract_sha256: str = "unfrozen"
    execution_backend: str = "openrouter"
    rate_card_json: dict[str, Any] = field(default_factory=dict)
    backend_contract_json: dict[str, Any] = field(default_factory=dict)
    tool_choice: str | dict[str, Any] = "auto"
    epicure_on_tool_required: bool = False
    tool_contract_diagnostic: bool = False
    supported_parameters: frozenset[str] | None = None
    decoding_parameters: dict[str, int | float] | None = None
    expected_actual_model_id: str | None = None
    expected_actual_provider_slug: str | None = None
    endpoint_contract_sha256: str = "unfrozen"
    protocol_bundle_sha256: str = "unfrozen"
    expected_epicure_release_id: str = "unresolved"
    expected_epicure_bundle_sha256: str = "unresolved"
    expected_epicure_application_sha256: str = "unresolved"
    expected_epicure_tool_schema_sha256: str = "unresolved"
    provider_budget_cap_micros: int = 0
    provider_account_budget_cap_micros: int = 0
    provider_account_scope_sha256: str = "unresolved"
    provider_authorization_envelope_sha256: str = "unresolved"
    provider_account_authorization_envelope_sha256: str = "unresolved"
    provider_credential_binding_sha256: str = "unresolved"
    provider_credential_scope_sha256: str = "unresolved"
    contract_smoke_registry_sha256: str = "unresolved"
    # Mutable provider aliases are never normal benchmark identities.  The one
    # governed exception is an explicitly requested, permanently exploratory
    # contract run whose catalog observation is content-addressed.
    allow_mutable_alias_exploratory: bool = False


@dataclass
class GenerationResult:
    answer_markdown: str
    output_json: dict[str, Any]
    actual_model_id: str
    provider_slug: str
    generation_id: str
    generation_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_micros: int = 0
    cost_reconciled: bool = True
    latency_ms: int = 0
    retries: int = 0
    finish_reason: str = "stop"
    tool_traces: list[ToolTrace] = field(default_factory=list)
    generation_metadata: list[dict[str, Any]] = field(default_factory=list)
    decoding_json: dict[str, Any] = field(default_factory=dict)
    epicure_attestation: dict[str, Any] = field(default_factory=dict)
    backend_response_schema_sha256: str = "unresolved"
    backend_tool_schema_sha256: str = "unresolved"
    cost_accounting_basis: str = "unrecorded"
    billing_reconciliation_status: str = "unrecorded"
    final_response_mode: str = "structured_json"
    structured_output_requested: bool = True
    structured_output_valid: bool | None = True
    intermediate_outputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationFailureResult:
    """Accounting evidence for accepted requests whose arm cannot be scored."""

    error: Exception
    actual_model_id: str
    provider_slug: str
    generation_id: str
    generation_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    cost_micros: int
    cost_reconciled: bool
    retries: int
    generation_metadata: list[dict[str, Any]]
    decoding_json: dict[str, Any]
    latency_ms: int = 0
    tool_traces: list[ToolTrace] = field(default_factory=list)
    backend_response_schema_sha256: str = "unresolved"
    backend_tool_schema_sha256: str = "unresolved"
    cost_accounting_basis: str = "unrecorded"
    billing_reconciliation_status: str = "unrecorded"


@dataclass(frozen=True)
class ProviderAttemptEvent:
    """Append-only lifecycle event for one external generation attempt."""

    attempt_id: str
    arm_id: str
    request_key_sha256: str
    phase: str
    attempt_index: int
    event_type: str
    generation_id: str = ""
    http_status: int | None = None
    error_type: str = ""
    payload_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


AttemptSink = Callable[[ProviderAttemptEvent], None]
ToolSink = Callable[[str, ToolTrace], None]
AttemptIdFactory = Callable[[str, str, int], str]


class ProviderError(RuntimeError):
    pass


class UncertainDeliveryError(ProviderError):
    """The provider may have accepted a request whose response was not received."""


class ResponseEnvelopeError(ProviderError):
    """A safely classified HTTP-200 response was not a chat-completions result."""


# OpenRouter documents these codes as request timeout, rate limiting, upstream
# provider failure, and no available provider.  When they arrive in an explicit
# JSON error envelope there is no chat-completion generation to reconcile.  Keep
# this allow-list deliberately narrow: unknown and non-transient codes remain
# terminal, even when a caller has retry capacity left.
RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES = frozenset({408, 429, 502, 503})


def _verified_openrouter_generation_identity(
    spec: GenerationSpec,
    generation_ids: list[str],
    generation_metadata: list[dict[str, Any]],
) -> tuple[str, str]:
    """Require exact identity evidence for every request in a multi-round arm."""

    expected_model = spec.expected_actual_model_id
    expected_provider = spec.expected_actual_provider_slug
    if not expected_model or not expected_provider:
        raise ProviderError("OpenRouter generation lacks a frozen identity contract")
    if (
        not generation_ids
        or any(not generation_id for generation_id in generation_ids)
        or len(generation_ids) != len(set(generation_ids))
        or len(generation_metadata) != len(generation_ids)
    ):
        raise ProviderError("OpenRouter generation identity coverage is incomplete")
    observed_ids = [str(item.get("generation_id") or "") for item in generation_metadata]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(generation_ids):
        raise ProviderError("OpenRouter generation metadata does not cover the exact request set")
    for item in generation_metadata:
        if (
            str(item.get("model") or "unknown") != expected_model
            or str(item.get("provider") or "unknown") != expected_provider
        ):
            raise ProviderError("OpenRouter substituted or omitted identity in a multi-round arm")
    return expected_model, expected_provider


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _safe_retry_delay_seconds(
    *, idempotency_key: str, attempt: int, retry_after: str = ""
) -> float:
    """Stagger only failures known to precede provider acceptance."""

    try:
        base = max(0.0, float(retry_after))
    except ValueError:
        base = 0.4 * (2**attempt)
    jitter_milliseconds = int(hashlib.sha256(idempotency_key.encode()).hexdigest()[:8], 16) % 1000
    return min(30.0, base + jitter_milliseconds / 1000)


def _money_to_micros(value: object) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("provider cost is not a decimal amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("provider cost must be finite and non-negative")
    return int((amount * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _extract_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
    return ""


def _assistant_continuation_message(
    message: dict[str, Any], *, empty_content_fallback: str
) -> dict[str, Any]:
    """Preserve provider reasoning continuity without publishing hidden reasoning."""

    continuation: dict[str, Any] = {"role": "assistant"}
    reasoning_details = message.get("reasoning_details")
    if reasoning_details is not None:
        continuation["reasoning_details"] = reasoning_details
        continuation["content"] = message.get("content")
    else:
        content = message.get("content")
        continuation["content"] = (
            content if _extract_content(message).strip() else empty_content_fallback
        )
    return continuation


def _parse_final(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProviderError("provider returned invalid final JSON") from exc
    return _validate_final(value)


def _validate_final(value: object) -> dict[str, Any]:
    required = set(FINAL_SCHEMA["required"])
    if not isinstance(value, dict) or set(value) != required:
        raise ProviderError("provider final response did not match the FlavourBench schema")
    answer = value.get("answer_markdown")
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderError("provider final response contained an empty answer")
    for key in ("ingredient_mentions", "constraints_addressed", "uncertainties"):
        items = value.get(key)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ProviderError("provider final response contained a malformed string array")
    return value


class MockProvider:
    async def aclose(self) -> None:
        return None

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        start = time.monotonic()
        await asyncio.sleep(0.08)
        evidence_line = (
            "I would use the structured pairing evidence as a shortlist, then test seasoning, "
            "texture, and intensity in the actual preparation."
            if spec.condition == "epicure_on"
            else "I would validate the combination with a small tasting batch before scaling it."
        )
        category_guidance = {
            "substitution": (
                "Match the missing ingredient's function first: structure, fat, salt, acidity, "
                "aroma, or browning."
            ),
            "composition": (
                "Choose one bridge ingredient, one source of acidity, and a deliberate texture "
                "contrast."
            ),
            "cookability": (
                "Sequence the preparation around the slowest component and include a visible "
                "doneness cue."
            ),
            "evidence": (
                "Treat model geometry as decision support, not proof of preference or causation."
            ),
        }
        guidance = category_guidance.get(spec.category, category_guidance["composition"])
        answer = (
            f"**Recommendation**\n\n{guidance} {evidence_line}\n\n"
            "**Practical check**\n\nStart with a small batch, record the adjustment, "
            "and keep the user's stated constraint as the hard boundary."
        )
        output = {
            "answer_markdown": answer,
            "ingredient_mentions": [],
            "constraints_addressed": ["user-stated constraint"],
            "uncertainties": ["Sensory quality requires tasting in the final preparation."],
        }
        traces = []
        if spec.condition == "epicure_on":
            traces.append(
                ToolTrace(
                    round_index=0,
                    name="find_pairings",
                    arguments={"ingredients": ["development fixture"]},
                    result="Mock Epicure evidence for an unranked engineering battle.",
                    latency_ms=4,
                    is_error=False,
                )
            )
        return GenerationResult(
            answer_markdown=answer,
            output_json=output,
            actual_model_id=spec.model_id,
            provider_slug="mock",
            generation_id=f"mock-{spec.arm_id}",
            generation_ids=[f"mock-{spec.arm_id}"],
            prompt_tokens=len(spec.prompt.split()),
            completion_tokens=len(answer.split()),
            latency_ms=round((time.monotonic() - start) * 1000),
            tool_traces=traces,
            decoding_json={
                "max_tokens": get_settings().max_output_tokens,
                "temperature": get_settings().decoding_temperature,
                "top_p": get_settings().decoding_top_p,
                "seed": get_settings().decoding_seed,
            },
            backend_response_schema_sha256=response_schema_sha256(spec.final_response_mode),
            backend_tool_schema_sha256=_canonical_sha256([]),
            cost_accounting_basis="mock_fixture",
            billing_reconciliation_status="not_applicable",
            final_response_mode=spec.final_response_mode,
            structured_output_requested=spec.final_response_mode == "structured_json",
            structured_output_valid=(
                True if spec.final_response_mode == "structured_json" else None
            ),
        )


class OpenRouterProvider:
    def __init__(
        self,
        attempt_sink: AttemptSink | None = None,
        tool_sink: ToolSink | None = None,
        attempt_id_factory: AttemptIdFactory | None = None,
    ) -> None:
        self.settings = get_settings()
        self.attempt_sink = attempt_sink
        self.tool_sink = tool_sink
        self.attempt_id_factory = attempt_id_factory
        self._issued_attempt_ids: set[str] = set()
        self._attempt_by_generation: dict[str, ProviderAttemptEvent] = {}
        self._backend_tool_schema_by_arm: dict[str, str] = {}
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_http_referer,
            "X-Title": self.settings.openrouter_title,
            "X-OpenRouter-Cache": "false",
        }
        if "gateway.ai.cloudflare.com" in self.settings.openrouter_base_url:
            headers["cf-aig-authorization"] = f"Bearer {self.settings.cloudflare_ai_gateway_token}"
            headers["cf-aig-skip-cache"] = "true"
            headers["cf-aig-collect-log-payload"] = "false"
        self.client = httpx.AsyncClient(
            base_url=self.settings.openrouter_base_url.rstrip("/"),
            headers=headers,
            timeout=self.settings.openrouter_timeout_seconds,
        )
        self.accounting_client = httpx.AsyncClient(
            base_url=getattr(
                self.settings,
                "openrouter_accounting_base_url",
                "https://openrouter.ai/api/v1",
            ).rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Accept": "application/json",
                "HTTP-Referer": self.settings.openrouter_http_referer,
                "X-Title": self.settings.openrouter_title,
            },
            timeout=self.settings.openrouter_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.accounting_client.aclose()

    def _emit_attempt(self, event: ProviderAttemptEvent) -> None:
        if self.attempt_sink is not None:
            self.attempt_sink(event)

    def _emit_tool(self, arm_id: str, trace: ToolTrace) -> None:
        if self.tool_sink is not None:
            self.tool_sink(arm_id, trace)

    def _new_attempt_id(self, arm_id: str, phase: str, attempt_index: int) -> str:
        """Issue one unique attempt ID, optionally from a pre-frozen plan.

        Production callers normally use random UUIDs. Narrow qualification runs can
        inject an exact slot resolver so every possible external request is frozen
        before any request bytes leave the process. A missing or duplicate slot is a
        hard pre-I/O failure.
        """

        attempt_id = (
            self.attempt_id_factory(arm_id, phase, attempt_index)
            if self.attempt_id_factory is not None
            else str(uuid.uuid4())
        )
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ProviderError("attempt-ID factory returned an empty identifier")
        if attempt_id in self._issued_attempt_ids:
            raise ProviderError("attempt-ID factory reused an identifier")
        self._issued_attempt_ids.add(attempt_id)
        return attempt_id

    @staticmethod
    def _request_contract(payload: dict[str, Any]) -> dict[str, Any]:
        """Project request semantics without retaining prompts or tool payloads."""

        provider = payload.get("provider")
        provider_contract = dict(provider) if isinstance(provider, dict) else {}
        tools = payload.get("tools")
        tool_contracts: list[dict[str, str]] = []
        if isinstance(tools, list):
            for tool in tools:
                function = tool.get("function") if isinstance(tool, dict) else None
                if not isinstance(function, dict):
                    continue
                parameters = function.get("parameters")
                tool_contracts.append(
                    {
                        "name": str(function.get("name") or ""),
                        "parameters_sha256": _canonical_sha256(parameters),
                    }
                )
        response_format = payload.get("response_format")
        return {
            "model": str(payload.get("model") or ""),
            "provider": provider_contract,
            "reasoning": payload.get("reasoning"),
            "reasoning_field_present": "reasoning" in payload,
            "response_format_sha256": (
                _canonical_sha256(response_format) if response_format is not None else None
            ),
            "response_format_present": "response_format" in payload,
            "tool_choice": payload.get("tool_choice"),
            "tools": tool_contracts,
            "tools_present": "tools" in payload,
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "seed": payload.get("seed"),
            "message_count": (
                len(payload.get("messages") or [])
                if isinstance(payload.get("messages"), list)
                else 0
            ),
            "messages_sha256": _canonical_sha256(payload.get("messages") or []),
        }

    @staticmethod
    def _safe_envelope_scalar(value: object) -> str | int | float | bool | None:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if (
            isinstance(value, str)
            and len(value) <= 64
            and all(character.isalnum() or character in "._:/-" for character in value)
        ):
            return value
        return "present_redacted"

    @classmethod
    def classify_response_envelope(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Classify response shape without retaining provider messages or response bodies."""

        error = value.get("error")
        if error is not None:
            error_mapping = error if isinstance(error, dict) else {}
            metadata = error_mapping.get("metadata")
            metadata_mapping = metadata if isinstance(metadata, dict) else {}
            provider = (
                error_mapping.get("provider")
                or metadata_mapping.get("provider_name")
                or metadata_mapping.get("provider")
                or value.get("provider")
            )
            error_code = cls._safe_envelope_scalar(error_mapping.get("code"))
            return {
                "classification": "openrouter_error_envelope",
                "accepted_chat_completion": False,
                "error_code": error_code,
                "error_type": cls._safe_envelope_scalar(error_mapping.get("type")),
                "provider": cls._safe_envelope_scalar(provider),
                "retryable": error_code in RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES,
            }
        if "success" in value and ("result" in value or "errors" in value):
            return {
                "classification": "gateway_api_envelope",
                "accepted_chat_completion": False,
                "error_code": None,
                "error_type": None,
                "provider": cls._safe_envelope_scalar(value.get("provider")),
                "retryable": False,
            }
        if "output" in value or value.get("object") in {"response", "response.completed"}:
            return {
                "classification": "responses_api_schema_mismatch",
                "accepted_chat_completion": False,
                "error_code": None,
                "error_type": None,
                "provider": cls._safe_envelope_scalar(value.get("provider")),
                "retryable": False,
            }
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            return {
                "classification": "chat_completions",
                "accepted_chat_completion": True,
                "error_code": None,
                "error_type": None,
                "provider": cls._safe_envelope_scalar(value.get("provider")),
                "retryable": False,
            }
        return {
            "classification": "unknown_non_chat_completion_envelope",
            "accepted_chat_completion": False,
            "error_code": None,
            "error_type": None,
            "provider": cls._safe_envelope_scalar(value.get("provider")),
            "retryable": False,
        }

    def _provider_preferences(
        self,
        provider_slug: str,
        rate_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preferences: dict[str, Any] = {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }
        if provider_slug and provider_slug not in {"openrouter", "mock"}:
            preferences["only"] = [provider_slug]
        if self.settings.openrouter_zdr:
            preferences["zdr"] = True
        rate_card = rate_card or {}
        prompt_per_token = rate_card.get("prompt_price_per_token")
        completion_per_token = rate_card.get("completion_price_per_token")
        prompt_max = (
            float(Decimal(str(prompt_per_token)) * Decimal(1_000_000))
            if prompt_per_token is not None
            else getattr(self.settings, "openrouter_max_prompt_price_per_mtok", None)
        )
        completion_max = (
            float(Decimal(str(completion_per_token)) * Decimal(1_000_000))
            if completion_per_token is not None
            else getattr(self.settings, "openrouter_max_completion_price_per_mtok", None)
        )
        if prompt_max is not None or completion_max is not None:
            preferences["max_price"] = {
                **({"prompt": prompt_max} if prompt_max is not None else {}),
                **({"completion": completion_max} if completion_max is not None else {}),
            }
        return preferences

    async def _post(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
        *,
        arm_id: str = "",
        phase: str = "unknown",
        governance_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_provider_attempts):
            request_contract = self._request_contract(payload)
            attempt_event = ProviderAttemptEvent(
                attempt_id=self._new_attempt_id(arm_id, phase, attempt),
                arm_id=arm_id,
                request_key_sha256=hashlib.sha256(idempotency_key.encode()).hexdigest(),
                phase=phase,
                attempt_index=attempt,
                event_type="request_started",
                payload_sha256=_canonical_sha256(payload),
                metadata={
                    **dict(governance_metadata or {}),
                    "request_contract": request_contract,
                    "request_contract_sha256": _canonical_sha256(request_contract),
                },
            )
            # The start event must be durable before bytes are sent. A failing
            # sink aborts the request instead of creating unaccounted spend.
            self._emit_attempt(attempt_event)
            try:
                response = await self.client.post(
                    "chat/completions",
                    json=payload,
                    headers={
                        "cf-aig-metadata": json.dumps(
                            {
                                "benchmark": "flavourbench",
                                "request_key_sha256": attempt_event.request_key_sha256[:32],
                                "phase": phase,
                            },
                            separators=(",", ":"),
                        )
                    },
                )
                response.raise_for_status()
                value = response.json()
                if not isinstance(value, dict):
                    raise ProviderError("OpenRouter returned an invalid response")
                response_envelope = self.classify_response_envelope(value)
                openrouter_cache = response.headers.get("X-OpenRouter-Cache-Status", "").upper()
                cloudflare_cache = response.headers.get("cf-aig-cache-status", "").upper()
                envelope_is_fresh = (
                    openrouter_cache != "HIT"
                    and cloudflare_cache != "HIT"
                    and (
                        "gateway.ai.cloudflare.com" not in self.settings.openrouter_base_url
                        or cloudflare_cache in {"MISS", "BYPASS"}
                    )
                )
                if response_envelope["accepted_chat_completion"] is not True:
                    classification = str(response_envelope["classification"])
                    error_code = response_envelope.get("error_code")
                    suffix = f" code={error_code}" if error_code is not None else ""
                    last_error = ResponseEnvelopeError(
                        f"OpenRouter returned {classification}{suffix}"
                    )
                    if classification == "openrouter_error_envelope":
                        # A structured error is a provider rejection, not a
                        # generation.  Never emit response_received or attach an
                        # ID from a malformed envelope, because both would make
                        # downstream accounting attempt to reconcile a cost.
                        self._emit_attempt(
                            ProviderAttemptEvent(
                                **{
                                    **attempt_event.__dict__,
                                    "event_type": "request_rejected",
                                    "http_status": response.status_code,
                                    "error_type": type(last_error).__name__,
                                    "metadata": {
                                        **attempt_event.metadata,
                                        "openrouter_cache_status": openrouter_cache,
                                        "cloudflare_cache_status": cloudflare_cache,
                                        "response_envelope": response_envelope,
                                    },
                                }
                            )
                        )
                        if (
                            response_envelope.get("retryable") is True
                            and envelope_is_fresh
                            and attempt + 1 < self.settings.max_provider_attempts
                        ):
                            delay = _safe_retry_delay_seconds(
                                idempotency_key=idempotency_key,
                                attempt=attempt,
                                retry_after=response.headers.get("Retry-After", ""),
                            )
                            self._emit_attempt(
                                ProviderAttemptEvent(
                                    **{
                                        **attempt_event.__dict__,
                                        "event_type": "retry_scheduled",
                                        "http_status": response.status_code,
                                        "error_type": type(last_error).__name__,
                                        "metadata": {
                                            **attempt_event.metadata,
                                            "retry_reason": (
                                                f"retryable_openrouter_error_envelope_{error_code}"
                                            ),
                                            "backoff_seconds": delay,
                                        },
                                    }
                                )
                            )
                            await asyncio.sleep(delay)
                            continue
                        break
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **attempt_event.__dict__,
                                "event_type": "invalid_response",
                                "http_status": response.status_code,
                                "error_type": type(last_error).__name__,
                                "metadata": {
                                    **attempt_event.metadata,
                                    "openrouter_cache_status": openrouter_cache,
                                    "cloudflare_cache_status": cloudflare_cache,
                                    "response_envelope": response_envelope,
                                },
                            }
                        )
                    )
                    break

                value["_flavourbench_retries"] = attempt
                generation_id = str(value.get("id") or "")
                completed = ProviderAttemptEvent(
                    **{
                        **attempt_event.__dict__,
                        "event_type": "response_received",
                        "generation_id": generation_id,
                        "http_status": response.status_code,
                        "payload_sha256": _canonical_sha256(value),
                        "metadata": {
                            "response_model": str(value.get("model") or ""),
                            "finish_reason": str(
                                ((value.get("choices") or [{}])[0] or {}).get("finish_reason")
                                or "unknown"
                            ),
                            "native_finish_reason": str(
                                ((value.get("choices") or [{}])[0] or {}).get(
                                    "native_finish_reason"
                                )
                                or ""
                            ),
                            "openrouter_cache_status": openrouter_cache,
                            "cloudflare_cache_status": cloudflare_cache,
                            "response_envelope": response_envelope,
                        },
                    }
                )
                if generation_id:
                    self._attempt_by_generation[generation_id] = completed
                self._emit_attempt(completed)
                if openrouter_cache == "HIT" or cloudflare_cache == "HIT":
                    raise ProviderError("cached provider responses are not rank eligible")
                if (
                    "gateway.ai.cloudflare.com" in self.settings.openrouter_base_url
                    and cloudflare_cache not in {"MISS", "BYPASS"}
                ):
                    raise ProviderError(
                        "Cloudflare gateway did not attest a fresh provider response"
                    )
                return value
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "pre_send_failure",
                            "error_type": type(exc).__name__,
                        }
                    )
                )
                if attempt + 1 < self.settings.max_provider_attempts:
                    delay = _safe_retry_delay_seconds(
                        idempotency_key=idempotency_key,
                        attempt=attempt,
                    )
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **attempt_event.__dict__,
                                "event_type": "retry_scheduled",
                                "error_type": type(exc).__name__,
                                "metadata": {
                                    **attempt_event.metadata,
                                    "retry_reason": "connect_error_before_send",
                                    "backoff_seconds": delay,
                                },
                            }
                        )
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            except httpx.ReadTimeout as exc:
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "uncertain_delivery",
                            "error_type": type(exc).__name__,
                        }
                    )
                )
                raise UncertainDeliveryError(
                    "OpenRouter response timed out after possible acceptance; "
                    "reconcile before retry"
                ) from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status == 408 or status >= 500:
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **attempt_event.__dict__,
                                "event_type": "uncertain_delivery",
                                "http_status": status,
                                "error_type": type(exc).__name__,
                            }
                        )
                    )
                    raise UncertainDeliveryError(
                        "OpenRouter gateway returned an ambiguous failure after possible "
                        "upstream dispatch; reconcile before retry"
                    ) from exc
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "request_rejected",
                            "http_status": status,
                            "error_type": type(exc).__name__,
                        }
                    )
                )
                if status == 429 and attempt + 1 < self.settings.max_provider_attempts:
                    delay = _safe_retry_delay_seconds(
                        idempotency_key=idempotency_key,
                        attempt=attempt,
                        retry_after=exc.response.headers.get("Retry-After", ""),
                    )
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **attempt_event.__dict__,
                                "event_type": "retry_scheduled",
                                "http_status": status,
                                "error_type": type(exc).__name__,
                                "metadata": {
                                    **attempt_event.metadata,
                                    "retry_reason": "http_429_request_rejected",
                                    "backoff_seconds": delay,
                                },
                            }
                        )
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            except (ValueError, ProviderError) as exc:
                last_error = exc
                self._emit_attempt(
                    ProviderAttemptEvent(
                        **{
                            **attempt_event.__dict__,
                            "event_type": "invalid_response",
                            "error_type": type(exc).__name__,
                        }
                    )
                )
                break
        if isinstance(last_error, ResponseEnvelopeError):
            raise ProviderError(str(last_error)) from last_error
        raise ProviderError(
            f"OpenRouter request failed: {type(last_error).__name__}"
        ) from last_error

    async def _generation_cost(self, generation_id: str) -> dict[str, Any]:
        if not generation_id:
            return {
                "generation_id": "",
                "cost_micros": 0,
                "provider": "unknown",
                "model": "unknown",
                "reconciled": False,
            }
        attempts = getattr(self.settings, "openrouter_accounting_attempts", 6)
        initial_delay = getattr(self.settings, "openrouter_accounting_initial_delay_seconds", 0.5)
        for attempt in range(attempts):
            try:
                response = await self.accounting_client.get(
                    "generation", params={"id": generation_id}
                )
                response.raise_for_status()
                data = response.json().get("data") or {}
                reconciled = "total_cost" in data
                result = {
                    "generation_id": generation_id,
                    "cost_micros": _money_to_micros(data.get("total_cost", 0)),
                    "provider": str(data.get("provider_name") or "unknown"),
                    "model": str(data.get("model") or "unknown"),
                    "reconciled": reconciled,
                    "tokens_prompt": int(data.get("tokens_prompt") or 0),
                    "tokens_completion": int(data.get("tokens_completion") or 0),
                    "native_tokens_prompt": int(data.get("native_tokens_prompt") or 0),
                    "native_tokens_completion": int(data.get("native_tokens_completion") or 0),
                    "generation_time_ms": int(data.get("generation_time") or 0),
                    "upstream_latency_ms": int(data.get("latency") or 0),
                }
                prior = self._attempt_by_generation.get(generation_id)
                if prior is not None:
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            **{
                                **prior.__dict__,
                                "event_type": "accounting_reconciled",
                                "metadata": result,
                            }
                        )
                    )
                return result
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                if attempt + 1 < attempts:
                    await asyncio.sleep(initial_delay * (2**attempt))
        return {
            "generation_id": generation_id,
            "cost_micros": 0,
            "provider": "unknown",
            "model": "unknown",
            "reconciled": False,
        }

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        """Reconcile every accepted request even when answer processing fails."""

        accepted = [
            (generation_id, event)
            for generation_id, event in self._attempt_by_generation.items()
            if event.arm_id == spec.arm_id
        ]
        if not accepted:
            return None
        generation_ids = [generation_id for generation_id, _ in accepted]
        generation_metadata: list[dict[str, Any]] = []
        cost_micros = 0
        cost_reconciled = True
        actual_model = spec.expected_actual_model_id or spec.model_id
        actual_provider = spec.expected_actual_provider_slug or spec.provider_slug
        prompt_tokens = 0
        completion_tokens = 0
        retries = 0
        for generation_id, event in accepted:
            accounting = await self._generation_cost(generation_id)
            generation_metadata.append(accounting)
            cost_micros += int(accounting["cost_micros"])
            cost_reconciled = cost_reconciled and bool(accounting["reconciled"])
            prompt_tokens += int(accounting.get("tokens_prompt") or 0)
            completion_tokens += int(accounting.get("tokens_completion") or 0)
            retries += event.attempt_index
            if accounting["provider"] != "unknown":
                actual_provider = str(accounting["provider"])
            if accounting["model"] != "unknown":
                actual_model = str(accounting["model"])
            elif event.metadata.get("response_model"):
                actual_model = str(event.metadata["response_model"])
        frozen_decoding = spec.decoding_parameters or {
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.decoding_temperature,
            "top_p": self.settings.decoding_top_p,
            "seed": self.settings.decoding_seed,
        }
        decoding_json = {
            name: frozen_decoding.get(name, "provider_fixed_unsupported")
            for name in ("max_tokens", "temperature", "top_p", "seed")
        }
        return GenerationFailureResult(
            error=error,
            actual_model_id=actual_model,
            provider_slug=actual_provider,
            generation_id=generation_ids[-1],
            generation_ids=generation_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_micros=cost_micros,
            cost_reconciled=cost_reconciled,
            retries=retries,
            generation_metadata=generation_metadata,
            decoding_json=decoding_json,
            backend_response_schema_sha256=response_schema_sha256(spec.final_response_mode),
            backend_tool_schema_sha256=self._backend_tool_schema_by_arm.get(
                spec.arm_id, _canonical_sha256([])
            ),
            cost_accounting_basis="openrouter_generation_metadata",
            billing_reconciliation_status="provider_generation_metadata",
        )

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        start = time.monotonic()
        if spec.final_response_mode not in {"structured_json", "plain_text"}:
            raise ProviderError("unsupported final response mode")
        matched_evidence = spec.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS
        matched_tool_access = spec.evidence_protocol == MATCHED_TOOL_ACCESS_PROTOCOL_V1
        portable_text_tool = spec.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
        if spec.evidence_protocol not in {"legacy_v6", *GOVERNED_EPICURE_PROTOCOLS}:
            raise ProviderError("unsupported frozen evidence protocol")
        if matched_evidence and not spec.matched_planning:
            raise ProviderError("matched-evidence protocol requires matched planning in both arms")
        if matched_tool_access and spec.matched_planning:
            raise ProviderError("matched-tool-access protocol prohibits staged planning")
        if portable_text_tool and spec.matched_planning:
            raise ProviderError("portable text-tool protocol prohibits staged planning")
        if spec.required_tool_contract_protocol != DIRECT_TOOL_CONTRACT_PROTOCOL:
            raise ProviderError("unsupported frozen required-tool contract protocol")
        if (
            self.settings.execution_mode == "live"
            and (matched_evidence or portable_text_tool)
            and (
                len(spec.required_tool_contract_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in spec.required_tool_contract_sha256
                )
            )
        ):
            raise ProviderError("live generation lacks a frozen required-tool contract")
        if spec.epicure_on_tool_required and (
            spec.condition != "epicure_on" or not (matched_evidence or portable_text_tool)
        ):
            raise ProviderError(
                "required Epicure treatment must be an Epicure-on matched-evidence arm"
            )
        direct_tool_contract = bool(spec.tool_contract_diagnostic)
        if direct_tool_contract and (
            spec.condition != "epicure_on"
            or spec.tool_choice != "required"
            or spec.required_tool_contract_protocol != DIRECT_TOOL_CONTRACT_PROTOCOL
        ):
            raise ProviderError("invalid required-tool diagnostic contract")
        system_prompt = (
            TOOL_CONTRACT_SYSTEM_PROMPT
            if direct_tool_contract
            else system_prompt_text(
                spec.condition,
                spec.final_response_mode,
                spec.evidence_protocol,
            )
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": spec.prompt},
        ]
        traces: list[ToolTrace] = []
        generation_ids: list[str] = []
        retries = 0
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        total_tool_calls = 0
        cumulative_tool_result_bytes = 0
        intermediate_outputs: list[dict[str, Any]] = []
        epicure_attestation: dict[str, Any] = {}
        backend_tool_schema_sha256 = _canonical_sha256([])
        self._backend_tool_schema_by_arm[spec.arm_id] = backend_tool_schema_sha256

        supported = spec.supported_parameters
        frozen_decoding = spec.decoding_parameters
        if self.settings.execution_mode == "live" and (
            supported is None
            or frozen_decoding is None
            or spec.endpoint_contract_sha256 in {"", "unfrozen", "unresolved"}
            or not spec.expected_actual_model_id
            or not spec.expected_actual_provider_slug
            or spec.protocol_bundle_sha256 in {"", "unfrozen", "unresolved"}
        ):
            raise ProviderError("live generation requires a complete frozen endpoint contract")
        supported = supported or frozenset({"max_tokens", "temperature", "top_p", "seed"})
        candidates = frozen_decoding or {
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.decoding_temperature,
            "top_p": self.settings.decoding_top_p,
            "seed": self.settings.decoding_seed,
        }
        unknown_decoding = set(candidates) - {
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        }
        if unknown_decoding:
            raise ProviderError(
                "frozen contract contains unsupported decoding fields: "
                + ", ".join(sorted(unknown_decoding))
            )
        unsupported_decoding = set(candidates) - supported
        if unsupported_decoding:
            raise ProviderError(
                "frozen decoding is not supported by the fixed endpoint: "
                + ", ".join(sorted(unsupported_decoding))
            )
        decoding = dict(candidates)
        if "max_tokens" not in decoding:
            raise ProviderError("fixed endpoint does not support a bounded output-token limit")
        if (
            not isinstance(spec.intermediate_max_tokens, int)
            or isinstance(spec.intermediate_max_tokens, bool)
            or spec.intermediate_max_tokens <= 0
            or spec.intermediate_max_tokens > int(decoding["max_tokens"])
        ):
            raise ProviderError("invalid frozen intermediate-token limit")
        if direct_tool_contract and (
            not isinstance(spec.required_tool_contract_max_intermediate_tokens, int)
            or isinstance(spec.required_tool_contract_max_intermediate_tokens, bool)
            or spec.required_tool_contract_max_intermediate_tokens <= 0
            or spec.required_tool_contract_max_intermediate_tokens > int(decoding["max_tokens"])
        ):
            raise ProviderError("invalid frozen required-tool intermediate-token limit")
        required = {"max_tokens"}
        if spec.final_response_mode == "structured_json":
            required.update({"response_format", "structured_outputs"})
        if spec.condition == "epicure_on" and not portable_text_tool:
            required.update({"tool_choice", "tools"})
        missing = required - supported
        if missing:
            raise ProviderError(
                "fixed endpoint is missing required parameters: " + ", ".join(sorted(missing))
            )
        effective_decoding = {
            name: decoding.get(name, "provider_fixed_unsupported")
            for name in ("max_tokens", "temperature", "top_p", "seed")
        }
        intermediate_decoding = {
            **decoding,
            "max_tokens": spec.intermediate_max_tokens,
        }
        reasoning_efforts = {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}
        if (
            spec.intermediate_reasoning_effort not in reasoning_efforts
            or spec.final_reasoning_effort not in reasoning_efforts
        ):
            raise ProviderError("frozen reasoning effort is unsupported")
        if (
            spec.intermediate_reasoning_effort is not None
            or spec.final_reasoning_effort is not None
        ) and "reasoning" not in supported:
            raise ProviderError("fixed endpoint does not support the frozen reasoning controls")
        intermediate_reasoning = (
            {
                "reasoning": {
                    "effort": spec.intermediate_reasoning_effort,
                    "exclude": True,
                }
            }
            if spec.intermediate_reasoning_effort is not None
            else {}
        )
        final_reasoning = (
            {
                "reasoning": {
                    "effort": spec.final_reasoning_effort,
                    "exclude": True,
                }
            }
            if spec.final_reasoning_effort is not None
            else {}
        )

        if (
            spec.matched_planning
            and (matched_evidence or spec.condition == "epicure_off")
            and not direct_tool_contract
        ):
            planning_response = await self._post(
                {
                    "model": spec.model_id,
                    "messages": [
                        *messages,
                        {
                            "role": "user",
                            "content": MATCHED_PLANNING_INSTRUCTION,
                        },
                    ],
                    **intermediate_decoding,
                    **intermediate_reasoning,
                    "provider": self._provider_preferences(
                        spec.provider_slug,
                        spec.rate_card_json,
                    ),
                },
                f"{spec.idempotency_key}:planning",
                arm_id=spec.arm_id,
                phase="planning",
                governance_metadata={
                    "provider_account_authorization_envelope_sha256": (
                        spec.provider_account_authorization_envelope_sha256
                    ),
                    "provider_credential_binding_sha256": (spec.provider_credential_binding_sha256),
                },
            )
            retries += int(planning_response.get("_flavourbench_retries", 0))
            planning_generation_id = str(planning_response.get("id") or "")
            generation_ids.append(planning_generation_id)
            usage = planning_response.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
            planning_choices = planning_response.get("choices") or []
            if not planning_choices:
                raise ProviderError("OpenRouter returned no planning choice")
            planning_choice = planning_choices[0]
            planning_text = _extract_content(planning_choice.get("message") or {}).strip()
            planning_finish = str(planning_choice.get("finish_reason") or "unknown")
            if planning_finish not in {"stop", "end_turn"}:
                raise ProviderError("matched planning response did not finish normally")
            intermediate_outputs.append(
                {
                    "phase": "planning",
                    "round_index": 0,
                    "generation_id": planning_generation_id,
                    "finish_reason": planning_finish,
                    "truncated": False,
                    "content": planning_text,
                    "visible_content_status": (
                        "present" if planning_text else "reasoning_only_or_suppressed"
                    ),
                    "tool_call_count": 0,
                }
            )
            messages.append(
                _assistant_continuation_message(
                    planning_choice.get("message") or {},
                    empty_content_fallback="Planning completed without a visible note.",
                )
            )

        if matched_evidence and not direct_tool_contract:
            messages.append({"role": "user", "content": MATCHED_EVIDENCE_DECISION_INSTRUCTION})
            if spec.condition == "epicure_off":
                evidence_response = await self._post(
                    {
                        "model": spec.model_id,
                        "messages": messages,
                        **intermediate_decoding,
                        **intermediate_reasoning,
                        "provider": self._provider_preferences(
                            spec.provider_slug,
                            spec.rate_card_json,
                        ),
                    },
                    f"{spec.idempotency_key}:evidence-decision",
                    arm_id=spec.arm_id,
                    phase="evidence_decision",
                    governance_metadata={
                        "provider_account_authorization_envelope_sha256": (
                            spec.provider_account_authorization_envelope_sha256
                        ),
                        "provider_credential_binding_sha256": (
                            spec.provider_credential_binding_sha256
                        ),
                    },
                )
                retries += int(evidence_response.get("_flavourbench_retries", 0))
                evidence_generation_id = str(evidence_response.get("id") or "")
                generation_ids.append(evidence_generation_id)
                usage = evidence_response.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
                evidence_choices = evidence_response.get("choices") or []
                if not evidence_choices:
                    raise ProviderError("OpenRouter returned no evidence-decision choice")
                evidence_choice = evidence_choices[0]
                evidence_finish = str(evidence_choice.get("finish_reason") or "unknown")
                evidence_note = _extract_content(evidence_choice.get("message") or {}).strip()
                if evidence_finish not in {"stop", "end_turn"}:
                    raise ProviderError(
                        "matched evidence-decision response did not finish normally"
                    )
                intermediate_outputs.append(
                    {
                        "phase": "evidence_decision",
                        "round_index": 0,
                        "generation_id": evidence_generation_id,
                        "finish_reason": evidence_finish,
                        "truncated": False,
                        "content": evidence_note,
                        "visible_content_status": (
                            "present" if evidence_note else "reasoning_only_or_suppressed"
                        ),
                        "tool_call_count": 0,
                    }
                )
                messages.append(
                    _assistant_continuation_message(
                        evidence_choice.get("message") or {},
                        empty_content_fallback="No external evidence was selected.",
                    )
                )

        if spec.condition == "epicure_on" and portable_text_tool:
            mcp_session_attempt_id = self._new_attempt_id(spec.arm_id, "mcp_session", 0)
            mcp_session_request_key = hashlib.sha256(
                f"{spec.idempotency_key}:mcp-session".encode()
            ).hexdigest()
            self._emit_attempt(
                ProviderAttemptEvent(
                    attempt_id=mcp_session_attempt_id,
                    arm_id=spec.arm_id,
                    request_key_sha256=mcp_session_request_key,
                    phase="mcp_session",
                    attempt_index=0,
                    event_type="mcp_session_started",
                    payload_sha256=spec.expected_epicure_tool_schema_sha256,
                    metadata={"protocol_bundle_sha256": spec.protocol_bundle_sha256},
                )
            )
            async with McpSession() as mcp:
                mcp_tools = await mcp.list_tools()
                epicure_attestation = await mcp.attest_runtime(
                    expected={
                        "release_id": spec.expected_epicure_release_id,
                        "bundle_sha256": spec.expected_epicure_bundle_sha256,
                        "application_sha256": spec.expected_epicure_application_sha256,
                        "tool_schema_sha256": spec.expected_epicure_tool_schema_sha256,
                    },
                    tools=mcp_tools,
                )
                attestation_sha256 = _canonical_sha256(epicure_attestation)
                self._emit_attempt(
                    ProviderAttemptEvent(
                        attempt_id=mcp_session_attempt_id,
                        arm_id=spec.arm_id,
                        request_key_sha256=mcp_session_request_key,
                        phase="mcp_attestation",
                        attempt_index=0,
                        event_type="mcp_session_attested",
                        payload_sha256=attestation_sha256,
                        metadata={
                            "attestation": epicure_attestation,
                            "attestation_sha256": attestation_sha256,
                            "protocol_bundle_sha256": spec.protocol_bundle_sha256,
                        },
                    )
                )
                exposed_tools = [
                    tool for tool in mcp_tools if tool.get("name") in PORTABLE_EPICURE_TOOL_NAMES
                ]
                portable_catalog = [
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema") or {"type": "object"},
                    }
                    for tool in exposed_tools
                ]
                backend_tool_schema_sha256 = _canonical_sha256(portable_catalog)
                self._backend_tool_schema_by_arm[spec.arm_id] = backend_tool_schema_sha256
                messages.append(
                    {"role": "user", "content": _portable_tool_instruction(exposed_tools)}
                )
                selection_response = await self._post(
                    {
                        "model": spec.model_id,
                        "messages": messages,
                        **intermediate_decoding,
                        **intermediate_reasoning,
                        "provider": self._provider_preferences(
                            spec.provider_slug,
                            spec.rate_card_json,
                        ),
                    },
                    f"{spec.idempotency_key}:portable-tool-selection",
                    arm_id=spec.arm_id,
                    phase="portable_tool_selection",
                    governance_metadata={
                        "provider_account_authorization_envelope_sha256": (
                            spec.provider_account_authorization_envelope_sha256
                        ),
                        "provider_credential_binding_sha256": (
                            spec.provider_credential_binding_sha256
                        ),
                    },
                )
                retries += int(selection_response.get("_flavourbench_retries", 0))
                selection_generation_id = str(selection_response.get("id") or "")
                generation_ids.append(selection_generation_id)
                usage = selection_response.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
                selection_choices = selection_response.get("choices") or []
                if not selection_choices:
                    raise ProviderError("provider returned no portable tool-selection choice")
                selection_choice = selection_choices[0]
                selection_finish = str(selection_choice.get("finish_reason") or "unknown")
                selection_message = selection_choice.get("message") or {}
                selection_text = _extract_content(selection_message).strip()
                if selection_finish not in {"stop", "end_turn"}:
                    raise ProviderError("portable tool-selection turn did not finish normally")
                intermediate_outputs.append(
                    {
                        "phase": "portable_tool_selection",
                        "round_index": 0,
                        "generation_id": selection_generation_id,
                        "finish_reason": selection_finish,
                        "truncated": False,
                        "content": selection_text,
                        "visible_content_status": "present" if selection_text else "empty",
                        "tool_call_count": 1,
                    }
                )
                name, arguments = _parse_portable_tool_request(selection_text)
                total_tool_calls = 1
                mcp_call_attempt_id = self._new_attempt_id(spec.arm_id, "mcp_tool_0_0", 0)
                mcp_call_key = hashlib.sha256(
                    f"{spec.idempotency_key}:mcp:0:0:{name}".encode()
                ).hexdigest()
                self._emit_attempt(
                    ProviderAttemptEvent(
                        attempt_id=mcp_call_attempt_id,
                        arm_id=spec.arm_id,
                        request_key_sha256=mcp_call_key,
                        phase="mcp_tool_0_0",
                        attempt_index=0,
                        event_type="mcp_call_started",
                        payload_sha256=_canonical_sha256({"name": name, "arguments": arguments}),
                        metadata={"tool_name": name, "transport": "portable_text_tool_v1"},
                    )
                )
                try:
                    result = await mcp.call_tool(name, arguments)
                except (RuntimeError, httpx.HTTPError) as error:
                    result_text = f"Epicure MCP service error: {type(error).__name__}"
                    self._emit_attempt(
                        ProviderAttemptEvent(
                            attempt_id=mcp_call_attempt_id,
                            arm_id=spec.arm_id,
                            request_key_sha256=mcp_call_key,
                            phase="mcp_tool_0_0",
                            attempt_index=0,
                            event_type="mcp_call_failed",
                            error_type=type(error).__name__,
                            payload_sha256=hashlib.sha256(result_text.encode()).hexdigest(),
                            metadata={"tool_name": name},
                        )
                    )
                    raise ProviderError("Epicure MCP service call failed") from error
                result_text = result.text
                result_bytes = result_text.encode()
                cumulative_tool_result_bytes = len(result_bytes)
                if cumulative_tool_result_bytes > self.settings.max_cumulative_tool_result_bytes:
                    raise ProviderError("cumulative Epicure tool evidence exceeded its cap")
                if len(result_bytes) > self.settings.max_tool_result_bytes:
                    bounded = result_bytes[: self.settings.max_tool_result_bytes]
                    while bounded:
                        try:
                            model_result_text = bounded.decode()
                            break
                        except UnicodeDecodeError:
                            bounded = bounded[:-1]
                    else:
                        model_result_text = ""
                    model_result_text += "\n[FlavourBench truncated this Epicure result.]"
                else:
                    model_result_text = result_text
                trace = ToolTrace(
                    round_index=0,
                    name=name,
                    arguments=arguments,
                    result=result_text,
                    latency_ms=result.latency_ms,
                    is_error=result.is_error,
                    call_index=0,
                    tool_call_id=f"portable:{selection_generation_id}",
                    structured_content=result.structured,
                )
                traces.append(trace)
                self._emit_tool(spec.arm_id, trace)
                self._emit_attempt(
                    ProviderAttemptEvent(
                        attempt_id=mcp_call_attempt_id,
                        arm_id=spec.arm_id,
                        request_key_sha256=mcp_call_key,
                        phase="mcp_tool_0_0",
                        attempt_index=0,
                        event_type="mcp_call_completed",
                        payload_sha256=hashlib.sha256(result_text.encode()).hexdigest(),
                        metadata={
                            "tool_name": name,
                            "latency_ms": result.latency_ms,
                            "is_error": result.is_error,
                        },
                    )
                )
                if result.is_error:
                    raise ProviderError("portable Epicure tool returned an error")
                messages.append(
                    _assistant_continuation_message(
                        selection_message,
                        empty_content_fallback=selection_text,
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Epicure executed {name} with your arguments and returned:\n"
                            f"{model_result_text}\n\n"
                            "Use this result to answer the original question."
                        ),
                    }
                )

        elif spec.condition == "epicure_on":
            mcp_session_attempt_id = self._new_attempt_id(spec.arm_id, "mcp_session", 0)
            mcp_session_request_key = hashlib.sha256(
                f"{spec.idempotency_key}:mcp-session".encode()
            ).hexdigest()
            self._emit_attempt(
                ProviderAttemptEvent(
                    attempt_id=mcp_session_attempt_id,
                    arm_id=spec.arm_id,
                    request_key_sha256=mcp_session_request_key,
                    phase="mcp_session",
                    attempt_index=0,
                    event_type="mcp_session_started",
                    payload_sha256=spec.expected_epicure_tool_schema_sha256,
                    metadata={"protocol_bundle_sha256": spec.protocol_bundle_sha256},
                )
            )
            async with McpSession() as mcp:
                mcp_tools = await mcp.list_tools()
                epicure_attestation = await mcp.attest_runtime(
                    expected={
                        "release_id": spec.expected_epicure_release_id,
                        "bundle_sha256": spec.expected_epicure_bundle_sha256,
                        "application_sha256": spec.expected_epicure_application_sha256,
                        "tool_schema_sha256": spec.expected_epicure_tool_schema_sha256,
                    },
                    tools=mcp_tools,
                )
                attestation_sha256 = _canonical_sha256(epicure_attestation)
                self._emit_attempt(
                    ProviderAttemptEvent(
                        attempt_id=mcp_session_attempt_id,
                        arm_id=spec.arm_id,
                        request_key_sha256=mcp_session_request_key,
                        phase="mcp_attestation",
                        attempt_index=0,
                        event_type="mcp_session_attested",
                        payload_sha256=attestation_sha256,
                        metadata={
                            "attestation": epicure_attestation,
                            "attestation_sha256": attestation_sha256,
                            "protocol_bundle_sha256": spec.protocol_bundle_sha256,
                        },
                    )
                )
                exposed_mcp_tools = (
                    [tool for tool in mcp_tools if tool.get("name") == TOOL_CONTRACT_NAME]
                    if direct_tool_contract
                    else mcp_tools
                )
                if direct_tool_contract and len(exposed_mcp_tools) != 1:
                    raise ProviderError(
                        "attested Epicure catalog does not expose exactly one find_pairings tool"
                    )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema") or {"type": "object"},
                        },
                    }
                    for tool in exposed_mcp_tools
                    if isinstance(tool.get("name"), str)
                ]
                backend_tool_schema_sha256 = _canonical_sha256(tools)
                self._backend_tool_schema_by_arm[spec.arm_id] = backend_tool_schema_sha256
                tool_round_limit = (
                    min(self.settings.max_tool_rounds, 2)
                    if direct_tool_contract
                    else self.settings.max_tool_rounds
                )
                for round_index in range(tool_round_limit):
                    round_decoding = (
                        {
                            **decoding,
                            "max_tokens": (spec.required_tool_contract_max_intermediate_tokens),
                        }
                        if direct_tool_contract
                        else intermediate_decoding
                    )
                    response = await self._post(
                        {
                            "model": spec.model_id,
                            "messages": messages,
                            "tools": tools,
                            "tool_choice": (
                                "required"
                                if round_index == 0 and spec.epicure_on_tool_required
                                else spec.tool_choice
                                if round_index == 0
                                else "auto"
                            ),
                            **round_decoding,
                            **intermediate_reasoning,
                            "provider": self._provider_preferences(
                                spec.provider_slug,
                                spec.rate_card_json,
                            ),
                        },
                        f"{spec.idempotency_key}:tool:{round_index}",
                        arm_id=spec.arm_id,
                        phase=f"tool_round_{round_index}",
                        governance_metadata={
                            "provider_account_authorization_envelope_sha256": (
                                spec.provider_account_authorization_envelope_sha256
                            ),
                            "provider_credential_binding_sha256": (
                                spec.provider_credential_binding_sha256
                            ),
                        },
                    )
                    retries += int(response.get("_flavourbench_retries", 0))
                    generation_ids.append(str(response.get("id") or ""))
                    usage = response.get("usage") or {}
                    prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    completion_tokens += int(usage.get("completion_tokens") or 0)
                    reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
                    choices = response.get("choices") or []
                    if not choices:
                        raise ProviderError("OpenRouter returned no choices")
                    message = choices[0].get("message") or {}
                    tool_calls = message.get("tool_calls") or []
                    intermediate_outputs.append(
                        {
                            "phase": "tool_selection",
                            "round_index": round_index,
                            "generation_id": str(response.get("id") or ""),
                            "finish_reason": str(choices[0].get("finish_reason") or "unknown"),
                            "truncated": str(choices[0].get("finish_reason") or "unknown")
                            == "length",
                            "content": _extract_content(message),
                            "tool_call_count": len(tool_calls),
                        }
                    )
                    if direct_tool_contract and round_index == 0:
                        if len(tool_calls) != 1:
                            raise ProviderError(
                                "required-tool diagnostic did not emit exactly one initial call"
                            )
                        initial_function = tool_calls[0].get("function") or {}
                        if initial_function.get("name") != TOOL_CONTRACT_NAME:
                            raise ProviderError(
                                "required-tool diagnostic called a non-contract tool"
                            )
                    if not tool_calls:
                        finish_reason = str(choices[0].get("finish_reason") or "unknown")
                        if (matched_evidence or matched_tool_access) and finish_reason not in {
                            "stop",
                            "end_turn",
                        }:
                            raise ProviderError(
                                "provider evidence-decision turn did not finish normally"
                            )
                        messages.append(
                            _assistant_continuation_message(
                                message,
                                empty_content_fallback="No external evidence was selected.",
                            )
                        )
                        break
                    if len(tool_calls) > self.settings.max_tool_calls_per_round:
                        raise ProviderError(
                            "provider tool-call fan-out "
                            f"({len(tool_calls)}) exceeded the per-round cap "
                            f"({self.settings.max_tool_calls_per_round})"
                        )
                    total_tool_calls += len(tool_calls)
                    effective_total_tool_cap = (
                        min(self.settings.max_tool_calls_total, 2)
                        if direct_tool_contract
                        else self.settings.max_tool_calls_total
                    )
                    if total_tool_calls > effective_total_tool_cap:
                        raise ProviderError("provider tool calls exceeded the generation cap")
                    messages.append(message)
                    round_had_error = False
                    for call_index, call in enumerate(tool_calls):
                        function = call.get("function") or {}
                        name = str(function.get("name") or "")
                        try:
                            arguments = json.loads(function.get("arguments") or "{}")
                            if not isinstance(arguments, dict):
                                raise ValueError
                        except (json.JSONDecodeError, ValueError) as exc:
                            arguments = {}
                            error_kind = "invalid_tool_arguments"
                            structured_content = {"flavourbench_error_kind": error_kind}
                            result_text = (
                                f"Tool call error: {type(exc).__name__}. Repair the arguments once "
                                "or continue without this tool."
                            )
                            is_error = True
                            latency_ms = 0
                        else:
                            error_kind = None
                            mcp_call_attempt_id = self._new_attempt_id(
                                spec.arm_id,
                                f"mcp_tool_{round_index}_{call_index}",
                                0,
                            )
                            mcp_call_key = hashlib.sha256(
                                (
                                    f"{spec.idempotency_key}:mcp:{round_index}:{call_index}:{name}"
                                ).encode()
                            ).hexdigest()
                            self._emit_attempt(
                                ProviderAttemptEvent(
                                    attempt_id=mcp_call_attempt_id,
                                    arm_id=spec.arm_id,
                                    request_key_sha256=mcp_call_key,
                                    phase=f"mcp_tool_{round_index}_{call_index}",
                                    attempt_index=0,
                                    event_type="mcp_call_started",
                                    payload_sha256=_canonical_sha256(
                                        {"name": name, "arguments": arguments}
                                    ),
                                    metadata={"tool_name": name},
                                )
                            )
                            mcp_call_started = time.monotonic()
                            try:
                                result = await mcp.call_tool(name, arguments)
                            except (RuntimeError, httpx.HTTPError) as exc:
                                result_text = f"Epicure MCP service error: {type(exc).__name__}"
                                structured_content = {
                                    "flavourbench_error_kind": "mcp_service_error"
                                }
                                is_error = True
                                latency_ms = round((time.monotonic() - mcp_call_started) * 1000)
                                trace = ToolTrace(
                                    round_index=round_index,
                                    name=name,
                                    arguments=arguments,
                                    result=result_text,
                                    latency_ms=latency_ms,
                                    is_error=True,
                                    call_index=call_index,
                                    tool_call_id=str(call.get("id") or ""),
                                    structured_content=structured_content,
                                )
                                traces.append(trace)
                                self._emit_tool(spec.arm_id, trace)
                                self._emit_attempt(
                                    ProviderAttemptEvent(
                                        attempt_id=mcp_call_attempt_id,
                                        arm_id=spec.arm_id,
                                        request_key_sha256=mcp_call_key,
                                        phase=f"mcp_tool_{round_index}_{call_index}",
                                        attempt_index=0,
                                        event_type="mcp_call_failed",
                                        error_type=type(exc).__name__,
                                        payload_sha256=hashlib.sha256(
                                            result_text.encode()
                                        ).hexdigest(),
                                        metadata={"tool_name": name},
                                    )
                                )
                                raise ProviderError("Epicure MCP service call failed") from exc
                            result_text = result.text
                            structured_content = result.structured
                            if result.is_error:
                                error_kind = "tool_returned_error"
                                structured_content = {
                                    **structured_content,
                                    "flavourbench_error_kind": error_kind,
                                }
                            is_error = result.is_error
                            latency_ms = result.latency_ms
                            self._emit_attempt(
                                ProviderAttemptEvent(
                                    attempt_id=mcp_call_attempt_id,
                                    arm_id=spec.arm_id,
                                    request_key_sha256=mcp_call_key,
                                    phase=f"mcp_tool_{round_index}_{call_index}",
                                    attempt_index=0,
                                    event_type="mcp_call_completed",
                                    payload_sha256=hashlib.sha256(result_text.encode()).hexdigest(),
                                    metadata={
                                        "tool_name": name,
                                        "latency_ms": latency_ms,
                                        "is_error": is_error,
                                    },
                                )
                            )
                        result_bytes = result_text.encode()
                        cumulative_tool_result_bytes += len(result_bytes)
                        cumulative_cap_exceeded = (
                            cumulative_tool_result_bytes
                            > self.settings.max_cumulative_tool_result_bytes
                        )
                        if len(result_bytes) > self.settings.max_tool_result_bytes:
                            bounded = result_bytes[: self.settings.max_tool_result_bytes]
                            while bounded:
                                try:
                                    model_result_text = bounded.decode()
                                    break
                                except UnicodeDecodeError:
                                    bounded = bounded[:-1]
                            else:
                                model_result_text = ""
                            model_result_text += (
                                "\n[FlavourBench truncated this tool result before returning it to "
                                "the model; the complete trace remains in the audit record.]"
                            )
                        else:
                            model_result_text = result_text
                        trace = ToolTrace(
                            round_index=round_index,
                            name=name,
                            arguments=arguments,
                            result=result_text,
                            latency_ms=latency_ms,
                            is_error=is_error,
                            call_index=call_index,
                            tool_call_id=str(call.get("id") or ""),
                            structured_content=structured_content,
                        )
                        traces.append(trace)
                        # This sink is a durability boundary. The complete MCP
                        # response is committed before another paid request can
                        # be sent, including calls that exhaust the repair law.
                        self._emit_tool(spec.arm_id, trace)
                        if cumulative_cap_exceeded:
                            raise ProviderError("cumulative Epicure tool evidence exceeded its cap")
                        if is_error:
                            round_had_error = True
                            if error_kind == "invalid_tool_arguments" and round_index >= 1:
                                raise ProviderError("tool call remained invalid after one repair")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id"),
                                "name": name,
                                "content": model_result_text,
                            }
                        )
                    if not round_had_error:
                        break

                if spec.epicure_on_tool_required:
                    if not traces:
                        raise ProviderError(
                            "required Epicure treatment produced no real tool trace"
                        )
                    if not any(not trace.is_error for trace in traces):
                        raise ProviderError(
                            "required Epicure treatment produced no successful tool call"
                        )

        if portable_text_tool:
            final_instruction = PORTABLE_FINAL_CHOICE_INSTRUCTION
        else:
            final_instruction = (
                MATCHED_EVIDENCE_V2_FINAL_INSTRUCTION
                if spec.evidence_protocol
                in {
                    MATCHED_EVIDENCE_PROTOCOL_V2,
                    MATCHED_TOOL_ACCESS_PROTOCOL_V1,
                }
                else ""
            ) + (
                "Return the final answer now. Use the required JSON schema and do not mention "
                "model identity."
                if spec.final_response_mode == "structured_json"
                else "Return only the final culinary answer now in clear Markdown. Do not mention "
                "model identity, tools, retrieval, or the evaluation condition."
            )
        messages.append({"role": "user", "content": final_instruction})
        final_payload: dict[str, Any] = {
            "model": spec.model_id,
            "messages": messages,
            **decoding,
            **final_reasoning,
            "provider": self._provider_preferences(
                spec.provider_slug,
                spec.rate_card_json,
            ),
        }
        if spec.final_response_mode == "structured_json":
            final_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "flavourbench_answer",
                    "strict": True,
                    "schema": FINAL_SCHEMA,
                },
            }
        final_response = await self._post(
            final_payload,
            f"{spec.idempotency_key}:final",
            arm_id=spec.arm_id,
            phase="final",
            governance_metadata={
                "provider_account_authorization_envelope_sha256": (
                    spec.provider_account_authorization_envelope_sha256
                ),
                "provider_credential_binding_sha256": (spec.provider_credential_binding_sha256),
            },
        )
        retries += int(final_response.get("_flavourbench_retries", 0))
        generation_id = str(final_response.get("id") or "")
        generation_ids.append(generation_id)
        usage = final_response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        choices = final_response.get("choices") or []
        if not choices:
            raise ProviderError("OpenRouter returned no final choice")
        choice = choices[0]
        final_finish_reason = str(choice.get("finish_reason") or "unknown")
        if final_finish_reason not in {"stop", "end_turn"}:
            raise ProviderError("provider final response did not finish normally")
        message = choice.get("message") or {}
        parsed = message.get("parsed")
        if spec.final_response_mode == "structured_json":
            output = (
                _validate_final(parsed)
                if isinstance(parsed, dict)
                else _parse_final(_extract_content(message))
            )
        else:
            answer = _extract_content(message).strip()
            if not answer:
                raise ProviderError("provider returned an empty plain-text final answer")
            output = {
                "answer_markdown": answer,
                "ingredient_mentions": [],
                "constraints_addressed": [],
                "uncertainties": [],
            }

        cost_micros = 0
        cost_reconciled = True
        generation_metadata: list[dict[str, Any]] = []
        for item_id in generation_ids:
            accounting = await self._generation_cost(item_id)
            generation_metadata.append(accounting)
            cost_micros += int(accounting["cost_micros"])
            cost_reconciled = cost_reconciled and bool(accounting["reconciled"])
        actual_model, actual_provider = _verified_openrouter_generation_identity(
            spec,
            generation_ids,
            generation_metadata,
        )
        return GenerationResult(
            answer_markdown=output["answer_markdown"],
            output_json=output,
            actual_model_id=actual_model,
            provider_slug=actual_provider,
            generation_id=generation_id,
            generation_ids=[item_id for item_id in generation_ids if item_id],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_micros=cost_micros,
            cost_reconciled=cost_reconciled,
            latency_ms=round((time.monotonic() - start) * 1000),
            retries=retries,
            finish_reason=final_finish_reason,
            tool_traces=traces,
            generation_metadata=generation_metadata,
            decoding_json=effective_decoding,
            epicure_attestation=epicure_attestation,
            backend_response_schema_sha256=response_schema_sha256(spec.final_response_mode),
            backend_tool_schema_sha256=backend_tool_schema_sha256,
            cost_accounting_basis="openrouter_generation_metadata",
            billing_reconciliation_status="provider_generation_metadata",
            final_response_mode=spec.final_response_mode,
            structured_output_requested=spec.final_response_mode == "structured_json",
            structured_output_valid=(
                True if spec.final_response_mode == "structured_json" else None
            ),
            intermediate_outputs=intermediate_outputs,
        )


class MultiBackendProvider:
    """Dispatch each arm only to its frozen execution backend."""

    def __init__(
        self,
        attempt_sink: AttemptSink | None = None,
        tool_sink: ToolSink | None = None,
    ) -> None:
        from .service_bedrock import BedrockServiceProvider
        from .service_cohere import CohereDirectProvider
        from .service_kimi import KimiDirectProvider
        from .service_qwencloud import QwenCloudDirectProvider

        self.openrouter = OpenRouterProvider(
            attempt_sink=attempt_sink,
            tool_sink=tool_sink,
        )
        self.bedrock = BedrockServiceProvider(
            attempt_sink=attempt_sink,
            tool_sink=tool_sink,
        )
        self.kimi_direct = KimiDirectProvider(
            attempt_sink=attempt_sink,
            tool_sink=tool_sink,
        )
        self.cohere_direct = CohereDirectProvider(
            attempt_sink=attempt_sink,
            tool_sink=tool_sink,
        )
        self.qwencloud_direct = QwenCloudDirectProvider(
            attempt_sink=attempt_sink,
            tool_sink=tool_sink,
        )

    async def generate(self, spec: GenerationSpec) -> GenerationResult:
        if spec.execution_backend == "openrouter":
            return await self.openrouter.generate(spec)
        if spec.execution_backend == "bedrock":
            return await self.bedrock.generate(spec)
        if spec.execution_backend == "kimi_direct":
            return await self.kimi_direct.generate(spec)
        if spec.execution_backend == "cohere_direct":
            return await self.cohere_direct.generate(spec)
        if spec.execution_backend == "qwencloud_direct":
            return await self.qwencloud_direct.generate(spec)
        raise ProviderError(f"unsupported live execution backend: {spec.execution_backend}")

    async def reconcile_failure(
        self,
        spec: GenerationSpec,
        error: Exception,
    ) -> GenerationFailureResult | None:
        if spec.execution_backend == "openrouter":
            return await self.openrouter.reconcile_failure(spec, error)
        if spec.execution_backend == "bedrock":
            return await self.bedrock.reconcile_failure(spec, error)
        if spec.execution_backend == "kimi_direct":
            return await self.kimi_direct.reconcile_failure(spec, error)
        if spec.execution_backend == "cohere_direct":
            return await self.cohere_direct.reconcile_failure(spec, error)
        if spec.execution_backend == "qwencloud_direct":
            return await self.qwencloud_direct.reconcile_failure(spec, error)
        return None

    async def aclose(self) -> None:
        await self.openrouter.aclose()
        await self.bedrock.aclose()
        await self.kimi_direct.aclose()
        await self.cohere_direct.aclose()
        await self.qwencloud_direct.aclose()


def get_provider(
    attempt_sink: AttemptSink | None = None,
    tool_sink: ToolSink | None = None,
) -> MockProvider | MultiBackendProvider:
    return (
        MockProvider()
        if get_settings().execution_mode == "mock"
        else MultiBackendProvider(attempt_sink=attempt_sink, tool_sink=tool_sink)
    )
