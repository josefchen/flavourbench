"""Read-only OpenRouter discovery and content-addressed candidate-panel freezing.

This module deliberately does not generate model output.  It snapshots the live
OpenRouter catalog and endpoint contracts needed for a later, explicitly
authorised FlavourBench run.  A produced manifest is always unranked and every
endpoint remains pending a paid contract smoke test and governance approval.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MANIFEST_SCHEMA_VERSION = "flavourbench-openrouter-candidate-manifest-v1"
REQUIRED_PARAMETERS = frozenset(
    {"max_tokens", "response_format", "structured_outputs", "tool_choice", "tools"}
)
DEFAULT_REQUESTED_NAMES = ("5.6", "fable", "opus")
REQUESTED_NAME_PREFERENCES = {
    "5.6": "openai/gpt-5.6-sol-pro",
    "fable": "anthropic/claude-fable-5",
    "opus": "anthropic/claude-opus-4.8",
}


class ManifestError(RuntimeError):
    """Base error for candidate-manifest discovery and validation."""


class PanelResolutionError(ManifestError):
    """Raised when a curated panel slot cannot be resolved or made eligible."""


class BudgetExceeded(ManifestError):
    """Raised before writing a manifest when its bounded forecast exceeds the cap."""


@dataclass(frozen=True)
class PanelSlot:
    """One intentionally curated coverage slot in the candidate panel."""

    slot_id: str
    cohort: str
    model_id: str
    rationale: str
    open_weight_candidate: bool = False


DEFAULT_PANEL: tuple[PanelSlot, ...] = (
    PanelSlot(
        "closed-openai-frontier",
        "closed_frontier",
        "openai/gpt-5.6-sol-pro",
        "OpenAI's highest 5.6 Sol tier; resolves the requested '5.6' family search.",
    ),
    PanelSlot(
        "closed-anthropic-frontier",
        "closed_frontier",
        "anthropic/claude-fable-5",
        "Stable, dated-canonical Anthropic Fable release; excludes the mutable latest alias.",
    ),
    PanelSlot(
        "closed-google-frontier",
        "closed_frontier",
        "google/gemini-3.1-pro-preview",
        "Google frontier-family coverage with the exact canonical release frozen.",
    ),
    PanelSlot(
        "closed-xai-frontier",
        "closed_frontier",
        "x-ai/grok-4.5",
        "Independent xAI frontier-family coverage.",
    ),
    PanelSlot(
        "open-deepseek-frontier",
        "open_weight_frontier",
        "deepseek/deepseek-v4-pro",
        "Large current DeepSeek open-weight candidate with a published weight repository.",
        True,
    ),
    PanelSlot(
        "open-qwen-frontier",
        "open_weight_frontier",
        "qwen/qwen3.5-397b-a17b",
        "Large sparse Qwen open-weight candidate from a distinct model family.",
        True,
    ),
    PanelSlot(
        "open-zai-frontier",
        "open_weight_frontier",
        "z-ai/glm-5.2",
        "Large GLM open-weight candidate for architecture and training-family diversity.",
        True,
    ),
    PanelSlot(
        "open-nvidia-frontier",
        "open_weight_frontier",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "NVIDIA Nemotron open-weight candidate for additional family and serving diversity.",
        True,
    ),
    PanelSlot(
        "efficiency-openai",
        "efficiency",
        "openai/gpt-5.6-luna",
        "Lower-cost OpenAI 5.6 tier for quality-cost-frontier measurement.",
    ),
    PanelSlot(
        "efficiency-google",
        "efficiency",
        "google/gemini-3.1-flash-lite",
        "Low-cost Google tier for quality-cost and latency-frontier measurement.",
    ),
    PanelSlot(
        "reasoning-anthropic",
        "reasoning",
        "anthropic/claude-opus-4.8",
        "Stable Opus release for high-compute reasoning coverage; excludes latest aliases.",
    ),
    PanelSlot(
        "reasoning-deepseek",
        "reasoning",
        "deepseek/deepseek-r1-0528",
        "Open-weight reasoning-specialised reference from a second family.",
        True,
    ),
)


@dataclass(frozen=True)
class ForecastPolicy:
    """Explicit upper bounds used only to reserve a future candidate workload."""

    arms_per_model: int = 20
    max_generations_per_arm: int = 9
    max_prompt_tokens_per_generation: int = 8_000
    max_completion_tokens_per_generation: int = 1_800
    max_reasoning_tokens_per_generation: int = 1_800

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_FORECAST_POLICY = ForecastPolicy()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ManifestError(f"{field} is not a decimal: {value!r}") from error
    if not parsed.is_finite() or parsed < 0:
        raise ManifestError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _stable_model_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not value.startswith("~")


def _normalise_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _catalog_index(catalog: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for model in catalog:
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            index[model_id] = model
    return index


def _search_matches(query: str, catalog: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    needle = _normalise_search(query)
    if not needle:
        return []

    def score(model: Mapping[str, Any]) -> tuple[int, int, str]:
        model_id = str(model.get("id") or "")
        canonical = str(model.get("canonical_slug") or "")
        name = str(model.get("name") or "")
        fields = [
            _normalise_search(model_id),
            _normalise_search(canonical),
            _normalise_search(name),
        ]
        if needle in fields:
            rank = 0
        elif any(field.startswith(needle) for field in fields):
            rank = 1
        else:
            rank = 2
        created = model.get("created")
        created_value = created if isinstance(created, int) else 0
        return (rank, -created_value, model_id)

    matches = []
    for model in catalog:
        haystack = " ".join(
            _normalise_search(str(model.get(field) or ""))
            for field in ("id", "canonical_slug", "name")
        )
        if needle in haystack:
            matches.append(model)
    return sorted(matches, key=score)


def resolve_requested_names(
    catalog: Sequence[Mapping[str, Any]],
    requested_names: Iterable[str],
    *,
    preferences: Mapping[str, str] = REQUESTED_NAME_PREFERENCES,
) -> list[dict[str, Any]]:
    """Resolve explicit searches without silently choosing among ambiguous matches."""

    index = _catalog_index(catalog)
    resolutions: list[dict[str, Any]] = []
    for raw_query in requested_names:
        query = raw_query.strip()
        if not query:
            continue
        matches = _search_matches(query, catalog)
        stable_matches = [model for model in matches if _stable_model_id(model.get("id"))]
        mutable_aliases = [
            str(model.get("id")) for model in matches if not _stable_model_id(model.get("id"))
        ]
        preferred_id = preferences.get(query.lower())
        preferred = index.get(preferred_id) if preferred_id else None

        if preferred is not None and preferred in stable_matches:
            status = "resolved_by_declared_policy"
            resolved_id: str | None = preferred_id
            basis = f"declared stable preference for search '{query}'"
        elif len(stable_matches) == 1:
            status = "resolved_unique_match"
            resolved_id = str(stable_matches[0].get("id"))
            basis = "only stable catalog match"
        elif not stable_matches:
            status = "unresolved"
            resolved_id = None
            basis = "no stable catalog match"
        else:
            status = "ambiguous"
            resolved_id = None
            basis = "multiple stable matches and no declared preference"

        resolutions.append(
            {
                "query": query,
                "status": status,
                "resolved_model_id": resolved_id,
                "resolution_basis": basis,
                "stable_matches": [str(model.get("id")) for model in stable_matches],
                "excluded_mutable_aliases": mutable_aliases,
            }
        )
    return resolutions


def _validate_model(model: Mapping[str, Any], slot: PanelSlot) -> None:
    model_id = model.get("id")
    canonical_slug = model.get("canonical_slug")
    if model_id != slot.model_id:
        raise PanelResolutionError(f"slot {slot.slot_id} resolved to the wrong model: {model_id}")
    if not _stable_model_id(model_id) or not _stable_model_id(canonical_slug):
        raise PanelResolutionError(f"slot {slot.slot_id} uses a mutable or absent model slug")
    architecture = model.get("architecture")
    if not isinstance(architecture, Mapping):
        raise PanelResolutionError(f"{model_id} has no architecture contract")
    input_modalities = set(architecture.get("input_modalities") or [])
    output_modalities = set(architecture.get("output_modalities") or [])
    if "text" not in input_modalities or "text" not in output_modalities:
        raise PanelResolutionError(f"{model_id} does not have text input and text output")
    supported = set(model.get("supported_parameters") or [])
    missing = REQUIRED_PARAMETERS - supported
    if missing:
        raise PanelResolutionError(f"{model_id} is missing model parameters: {sorted(missing)}")
    if slot.open_weight_candidate and not model.get("hugging_face_id"):
        raise PanelResolutionError(
            f"{model_id} lacks a catalog weight-repository reference for its open-weight slot"
        )


def _effective_rate(pricing: Mapping[str, Any], field: str, *, max_prompt_tokens: int) -> Decimal:
    values: list[Decimal] = []
    if pricing.get(field) is not None:
        values.append(_decimal(pricing[field], field=f"pricing.{field}"))
    overrides = pricing.get("overrides") or []
    if isinstance(overrides, list):
        for index, override in enumerate(overrides):
            if not isinstance(override, Mapping) or override.get(field) is None:
                continue
            threshold = override.get("min_prompt_tokens")
            if isinstance(threshold, int) and max_prompt_tokens >= threshold:
                values.append(
                    _decimal(
                        override[field],
                        field=f"pricing.overrides[{index}].{field}",
                    )
                )
    if not values:
        return Decimal(0)
    return max(values)


def _generation_cost(
    endpoint: Mapping[str, Any], policy: ForecastPolicy
) -> tuple[Decimal, dict[str, str]]:
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise ManifestError(f"endpoint {endpoint.get('tag')} has no pricing contract")
    if pricing.get("prompt") is None or pricing.get("completion") is None:
        raise ManifestError(f"endpoint {endpoint.get('tag')} has unpriced text tokens")
    prompt_rate = _effective_rate(
        pricing,
        "prompt",
        max_prompt_tokens=policy.max_prompt_tokens_per_generation,
    )
    completion_rate = _effective_rate(
        pricing,
        "completion",
        max_prompt_tokens=policy.max_prompt_tokens_per_generation,
    )
    reasoning_rate = _effective_rate(
        pricing,
        "internal_reasoning",
        max_prompt_tokens=policy.max_prompt_tokens_per_generation,
    )
    request_rate = _effective_rate(
        pricing,
        "request",
        max_prompt_tokens=policy.max_prompt_tokens_per_generation,
    )
    cost = (
        prompt_rate * policy.max_prompt_tokens_per_generation
        + completion_rate * policy.max_completion_tokens_per_generation
        + reasoning_rate * policy.max_reasoning_tokens_per_generation
        + request_rate
    )
    return cost, {
        "prompt_usd_per_token": _decimal_text(prompt_rate),
        "completion_usd_per_token": _decimal_text(completion_rate),
        "internal_reasoning_usd_per_token": _decimal_text(reasoning_rate),
        "request_usd": _decimal_text(request_rate),
    }


def _eligible_endpoint(
    endpoint: Mapping[str, Any], model_id: str, policy: ForecastPolicy
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    supported = set(endpoint.get("supported_parameters") or [])
    missing = REQUIRED_PARAMETERS - supported
    if missing:
        reasons.append(f"missing parameters {sorted(missing)}")
    tag = endpoint.get("tag")
    if not isinstance(tag, str) or not tag:
        reasons.append("missing fixed endpoint tag")
    if endpoint.get("model_id") != model_id:
        reasons.append("endpoint model_id does not match catalog model")
    context_length = endpoint.get("context_length")
    required_context = (
        policy.max_prompt_tokens_per_generation + policy.max_completion_tokens_per_generation
    )
    if not isinstance(context_length, int) or context_length < required_context:
        reasons.append(f"context length below required {required_context}")
    max_completion = endpoint.get("max_completion_tokens")
    if (
        isinstance(max_completion, int)
        and max_completion < policy.max_completion_tokens_per_generation
    ):
        reasons.append(
            f"completion limit below required {policy.max_completion_tokens_per_generation}"
        )
    try:
        _generation_cost(endpoint, policy)
    except ManifestError as error:
        reasons.append(str(error))
    return not reasons, reasons


def _select_endpoint(
    model_id: str,
    endpoints_document: Mapping[str, Any],
    policy: ForecastPolicy,
    excluded_tags: frozenset[str] = frozenset(),
) -> tuple[Mapping[str, Any], int]:
    data = endpoints_document.get("data")
    endpoints = data.get("endpoints") if isinstance(data, Mapping) else None
    if not isinstance(endpoints, list):
        raise PanelResolutionError(f"{model_id} returned no endpoint list")
    eligible: list[tuple[Decimal, int, Mapping[str, Any]]] = []
    rejection_details: list[str] = []
    for position, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, Mapping):
            continue
        if str(endpoint.get("tag") or "") in excluded_tags:
            rejection_details.append(
                f"{endpoint.get('tag')}: excluded by prior immutable contract evidence"
            )
            continue
        accepted, reasons = _eligible_endpoint(endpoint, model_id, policy)
        if accepted:
            cost, _ = _generation_cost(endpoint, policy)
            eligible.append((cost, position, endpoint))
        else:
            rejection_details.append(f"{endpoint.get('tag')}: {', '.join(reasons)}")
    if not eligible:
        detail = "; ".join(rejection_details[:5])
        raise PanelResolutionError(f"{model_id} has no eligible endpoint ({detail})")
    eligible.sort(key=lambda item: (item[0], item[1]))
    return eligible[0][2], len(eligible)


def _model_snapshot(model: Mapping[str, Any]) -> dict[str, Any]:
    architecture = model.get("architecture")
    architecture = architecture if isinstance(architecture, Mapping) else {}
    pricing = model.get("pricing")
    return {
        "id": model.get("id"),
        "canonical_slug": model.get("canonical_slug"),
        "name": model.get("name"),
        "created": model.get("created"),
        "context_length": model.get("context_length"),
        "architecture": {
            "modality": architecture.get("modality"),
            "input_modalities": sorted(architecture.get("input_modalities") or []),
            "output_modalities": sorted(architecture.get("output_modalities") or []),
            "tokenizer": architecture.get("tokenizer"),
            "instruct_type": architecture.get("instruct_type"),
        },
        "supported_parameters": sorted(model.get("supported_parameters") or []),
        "pricing": pricing if isinstance(pricing, Mapping) else {},
        "hugging_face_id": model.get("hugging_face_id") or None,
        "top_provider": model.get("top_provider")
        if isinstance(model.get("top_provider"), Mapping)
        else {},
    }


def _endpoint_snapshot(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    pricing = endpoint.get("pricing")
    return {
        "name": endpoint.get("name"),
        "provider_name": endpoint.get("provider_name"),
        "tag": endpoint.get("tag"),
        "model_id": endpoint.get("model_id"),
        "quantization": endpoint.get("quantization"),
        "context_length": endpoint.get("context_length"),
        "max_completion_tokens": endpoint.get("max_completion_tokens"),
        "status": endpoint.get("status"),
        "pricing": pricing if isinstance(pricing, Mapping) else {},
        "supported_parameters": sorted(endpoint.get("supported_parameters") or []),
        "uptime_last_1d": endpoint.get("uptime_last_1d"),
        "uptime_last_7d": endpoint.get("uptime_last_7d"),
        "uptime_last_30d": endpoint.get("uptime_last_30d"),
        "latency_last_30m": endpoint.get("latency_last_30m"),
        "throughput_last_30m": endpoint.get("throughput_last_30m"),
    }


def build_candidate_manifest(
    catalog_document: Mapping[str, Any],
    endpoints_by_model: Mapping[str, Mapping[str, Any]],
    *,
    cap_usd: Decimal | str,
    forecast_policy: ForecastPolicy = DEFAULT_FORECAST_POLICY,
    panel: Sequence[PanelSlot] = DEFAULT_PANEL,
    requested_names: Iterable[str] = DEFAULT_REQUESTED_NAMES,
    endpoint_exclusions: Mapping[str, Iterable[str]] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build and hash a validated, unranked candidate manifest without generation."""

    forecast_policy.validate()
    cap = _decimal(cap_usd, field="cap_usd")
    if cap <= 0:
        raise ValueError("cap_usd must be greater than zero")
    catalog = catalog_document.get("data")
    if not isinstance(catalog, list):
        raise ManifestError("OpenRouter catalog returned no model list")
    models = [model for model in catalog if isinstance(model, Mapping)]
    catalog_index = _catalog_index(models)
    if len({slot.slot_id for slot in panel}) != len(panel):
        raise ValueError("panel slot IDs must be unique")
    if len({slot.model_id for slot in panel}) != len(panel):
        raise ValueError("panel model IDs must be unique")

    model_entries: list[dict[str, Any]] = []
    forecast_total = Decimal(0)
    normalized_exclusions = {
        model_id: frozenset(tags) for model_id, tags in (endpoint_exclusions or {}).items()
    }
    for slot in panel:
        model = catalog_index.get(slot.model_id)
        if model is None:
            raise PanelResolutionError(f"panel model is absent from live catalog: {slot.model_id}")
        _validate_model(model, slot)
        endpoints_document = endpoints_by_model.get(slot.model_id)
        if endpoints_document is None:
            raise PanelResolutionError(f"no endpoint document for {slot.model_id}")
        endpoint, eligible_count = _select_endpoint(
            slot.model_id,
            endpoints_document,
            forecast_policy,
            normalized_exclusions.get(slot.model_id, frozenset()),
        )
        per_generation, effective_rates = _generation_cost(endpoint, forecast_policy)
        per_arm = per_generation * forecast_policy.max_generations_per_arm
        per_model = per_arm * forecast_policy.arms_per_model
        forecast_total += per_model
        endpoint_tag = str(endpoint["tag"])
        model_entries.append(
            {
                "slot": asdict(slot),
                "model": _model_snapshot(model),
                "endpoint": _endpoint_snapshot(endpoint),
                "endpoint_document_sha256": _sha256(endpoints_document),
                "endpoint_selection": {
                    "method": (
                        "lowest bounded text-generation cost among strictly eligible endpoints; "
                        "OpenRouter endpoint order breaks exact price ties"
                    ),
                    "eligible_endpoint_count": eligible_count,
                    "selected_exact_tag": endpoint_tag,
                    "pending_checks": [
                        "no-spend tool/structured-output contract smoke",
                        "paid identity and provider-return contract smoke after authorisation",
                        "provider data-policy and jurisdiction review",
                    ],
                },
                "request_policy": {
                    "provider": {
                        "only": [endpoint_tag],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                        "data_collection": "deny",
                    },
                    "policy_scope": "request_enforced",
                    "endpoint_retention_attestation": "not_present_in_endpoint_metadata",
                    "official_eligibility": "pending_contract_smoke_and_governance_approval",
                },
                "open_weight_evidence": {
                    "catalog_hugging_face_id": model.get("hugging_face_id") or None,
                    "license_review_status": "pending",
                }
                if slot.open_weight_candidate
                else None,
                "forecast": {
                    "effective_rates": effective_rates,
                    "per_generation_usd": _decimal_text(per_generation),
                    "per_arm_usd": _decimal_text(per_arm),
                    "per_model_usd": _decimal_text(per_model),
                },
            }
        )

    resolutions = resolve_requested_names(models, requested_names)
    unresolved = [item["query"] for item in resolutions if item["status"] == "unresolved"]
    ambiguous = [item["query"] for item in resolutions if item["status"] == "ambiguous"]
    within_cap = forecast_total <= cap
    if not within_cap:
        raise BudgetExceeded(
            f"bounded forecast ${_decimal_text(forecast_total)} exceeds cap "
            f"${_decimal_text(cap)}; no manifest was written"
        )

    observed = observed_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "unranked_candidate",
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": observed,
        "source": {
            "catalog_url": f"{OPENROUTER_API_BASE}/models",
            "endpoint_url_template": f"{OPENROUTER_API_BASE}/models/:author/:slug/endpoints",
            "catalog_model_count": len(models),
            "catalog_document_sha256": _sha256(catalog_document),
        },
        "selection": {
            "method": (
                "curator-defined 4 closed-frontier, 4 open-weight-frontier, 2 efficiency, "
                "and 2 reasoning coverage slots; live capability checks and deterministic "
                "endpoint cost selection"
            ),
            "performance_claim": "none; inclusion is coverage, not a ranking",
            "model_count": len(model_entries),
            "requested_name_resolutions": resolutions,
            "unresolved_requested_names": unresolved,
            "ambiguous_requested_names": ambiguous,
            "endpoint_exclusions": {
                model_id: sorted(tags)
                for model_id, tags in sorted(normalized_exclusions.items())
                if tags
            },
        },
        "budget": {
            "currency": "USD",
            "cap_usd": _decimal_text(cap),
            "bounded_forecast_usd": _decimal_text(forecast_total),
            "headroom_usd": _decimal_text(cap - forecast_total),
            "within_cap": True,
            "forecast_policy": asdict(forecast_policy),
            "forecast_notes": [
                "Every arm reserves all nine calls: eight Epicure tool rounds plus one final call.",
                (
                    "Prompt, visible completion, separately listed internal-reasoning, and "
                    "request rates are bounded."
                ),
                (
                    "Forecast is a reservation bound for the stated candidate workload, "
                    "not an invoice quote."
                ),
                "Actual future spend must be reconciled from OpenRouter generation metadata.",
            ],
        },
        "models": model_entries,
        "governance": {
            "manifest_class": "discovery_snapshot_only",
            "freeze_status": "candidate_not_season_manifest",
            "data_policy": (
                "data_collection=deny is frozen in each request policy; endpoint metadata does not "
                "independently attest retention"
            ),
            "required_before_scored_collection": [
                "Gate A approvals",
                "provider and endpoint data-policy approval",
                "endpoint-specific tools and structured-output smoke pass",
                "actual model/provider identity verification",
                "final prompt, tool, schema, and Epicure lineage freeze",
            ],
        },
    }
    digest = _sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return payload


def verify_manifest_content_address(manifest: Mapping[str, Any]) -> bool:
    content_address = manifest.get("content_address")
    if not isinstance(content_address, Mapping):
        return False
    digest = content_address.get("digest")
    unhashed = dict(manifest)
    unhashed.pop("content_address", None)
    return (
        content_address.get("algorithm") == "sha256"
        and isinstance(digest, str)
        and digest == _sha256(unhashed)
        and content_address.get("uri") == f"sha256:{digest}"
    )


def write_content_addressed_manifest(
    manifest: Mapping[str, Any], output_directory: str | Path
) -> Path:
    """Atomically write a verified manifest using its digest as the filename."""

    if not verify_manifest_content_address(manifest):
        raise ManifestError("refusing to write a manifest with an invalid content address")
    digest = str(manifest["content_address"]["digest"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"flavourbench-openrouter-unranked-{digest}.json"
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


async def _get_document(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ManifestError(f"OpenRouter returned a non-object document for {path}")
    return payload


async def discover_candidate_manifest(
    *,
    api_key: str,
    cap_usd: Decimal | str,
    forecast_policy: ForecastPolicy = DEFAULT_FORECAST_POLICY,
    panel: Sequence[PanelSlot] = DEFAULT_PANEL,
    requested_names: Iterable[str] = DEFAULT_REQUESTED_NAMES,
    endpoint_exclusions: Mapping[str, Iterable[str]] | None = None,
    observed_at: str | None = None,
    api_base: str = OPENROUTER_API_BASE,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Read catalog/endpoints and freeze a candidate manifest; never call generation APIs."""

    if not api_key:
        raise ValueError("an OpenRouter API key is required for endpoint discovery")
    own_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=api_base.rstrip("/") + "/",
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    try:
        catalog_document = await _get_document(client, "models")
        catalog = catalog_document.get("data")
        if not isinstance(catalog, list):
            raise ManifestError("OpenRouter catalog returned no model list")
        catalog_ids = {
            model.get("id")
            for model in catalog
            if isinstance(model, Mapping) and isinstance(model.get("id"), str)
        }
        absent = [slot.model_id for slot in panel if slot.model_id not in catalog_ids]
        if absent:
            raise PanelResolutionError(f"panel models absent from live catalog: {absent}")

        async def fetch_endpoints(model_id: str) -> tuple[str, dict[str, Any]]:
            author, slug = model_id.split("/", 1)
            path = f"models/{quote(author, safe='')}/{quote(slug, safe=':~._-')}/endpoints"
            return model_id, await _get_document(client, path)

        endpoint_pairs = await asyncio.gather(*(fetch_endpoints(slot.model_id) for slot in panel))
        return build_candidate_manifest(
            catalog_document,
            dict(endpoint_pairs),
            cap_usd=cap_usd,
            forecast_policy=forecast_policy,
            panel=panel,
            requested_names=requested_names,
            endpoint_exclusions=endpoint_exclusions,
            observed_at=observed_at,
        )
    finally:
        if own_client:
            await client.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read live OpenRouter metadata and write a no-spend, unranked, "
            "content-addressed FlavourBench candidate manifest."
        )
    )
    parser.add_argument("--cap-usd", required=True, help="Exact USD reservation cap")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--requested-name",
        action="append",
        default=[],
        help="Additional catalog search to resolve and record",
    )
    parser.add_argument("--arms-per-model", type=int, default=20)
    parser.add_argument("--max-generations-per-arm", type=int, default=9)
    parser.add_argument("--max-prompt-tokens", type=int, default=8_000)
    parser.add_argument("--max-completion-tokens", type=int, default=1_800)
    parser.add_argument("--max-reasoning-tokens", type=int, default=1_800)
    parser.add_argument(
        "--exclude-endpoint",
        action="append",
        default=[],
        metavar="MODEL_ID=ENDPOINT_TAG",
        help="Exclude an endpoint using prior immutable contract evidence.",
    )
    return parser


async def _async_run(arguments: argparse.Namespace) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY or FLAVOURBENCH_OPENROUTER_API_KEY is required; "
            "the key is used only for read-only catalog and endpoint requests"
        )
    requested_names = tuple(dict.fromkeys((*DEFAULT_REQUESTED_NAMES, *arguments.requested_name)))
    policy = ForecastPolicy(
        arms_per_model=arguments.arms_per_model,
        max_generations_per_arm=arguments.max_generations_per_arm,
        max_prompt_tokens_per_generation=arguments.max_prompt_tokens,
        max_completion_tokens_per_generation=arguments.max_completion_tokens,
        max_reasoning_tokens_per_generation=arguments.max_reasoning_tokens,
    )
    exclusions: dict[str, list[str]] = {}
    for value in arguments.exclude_endpoint:
        if "=" not in value:
            raise SystemExit("--exclude-endpoint must be MODEL_ID=ENDPOINT_TAG")
        model_id, endpoint_tag = value.split("=", 1)
        if not model_id or not endpoint_tag:
            raise SystemExit("--exclude-endpoint must include both model and endpoint")
        exclusions.setdefault(model_id, []).append(endpoint_tag)
    manifest = await discover_candidate_manifest(
        api_key=api_key,
        cap_usd=arguments.cap_usd,
        forecast_policy=policy,
        requested_names=requested_names,
        endpoint_exclusions=exclusions,
    )
    return write_content_addressed_manifest(manifest, arguments.output_directory)


def run() -> None:
    destination = asyncio.run(_async_run(_parser().parse_args()))
    print(destination)


if __name__ == "__main__":
    run()
