"""Authenticated, no-generation QwenCloud catalog freezing.

This module deliberately stops at discovery.  A catalog row is not evidence that
the endpoint satisfies FlavourBench's tool loop, structured-final contract,
identity checks, cost accounting, or culinary evaluation protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .real_task_bank import sha256_json

CATALOG_SCHEMA_VERSION = "flavourbench-qwencloud-catalog-v1"
CANDIDATE_SCHEMA_VERSION = "flavourbench-qwencloud-candidate-extension-v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
CATALOG_ENDPOINT = f"{DEFAULT_BASE_URL}/models"
PROVIDER_MODEL_DOCUMENTATION = (
    "https://www.alibabacloud.com/help/en/model-studio/text-generation-model/"
)
PROVIDER_PRICING_DOCUMENTATION = "https://www.alibabacloud.com/help/en/model-studio/model-pricing"
QWEN37_MODEL_DOCUMENTATION = "https://www.alibabacloud.com/help/en/model-studio/qwen3-7-max"
QWENCLOUD_FUNCTION_CALLING_DOCUMENTATION = (
    "https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling"
)
QWENCLOUD_ERROR_DOCUMENTATION = "https://www.alibabacloud.com/help/en/model-studio/error-code"
QWEN38_TOOL_AUTO_INSTRUCTION = (
    "For this exploratory Epicure treatment, call at least one exposed Epicure tool "
    "now before continuing."
)
CONFIRMATION = "FREEZE_AUTHENTICATED_QWENCLOUD_CATALOG_NO_GENERATION_V1"

# Mainline aliases are useful discovery evidence but are never frozen as scored
# endpoints.  Every executable candidate below has an immutable dated release
# or a stable open-weight model identifier.
DISCOVERY_ALIASES = ("qwen3.8-max", "qwen3.7-max")
DEFAULT_CANDIDATES = (
    "qwen3.7-max-2026-06-08",
    "qwen3.7-plus-2026-05-26",
    "qwen3.7-flash-2026-07-15",
    "qwen3.5-397b-a17b",
)
DISPLAY_NAMES = {
    "qwen3.8-max": "Qwen 3.8 Max (mutable alias)",
    "qwen3.7-max": "Qwen 3.7 Max (mutable alias)",
    "qwen3.7-max-2026-06-08": "Qwen 3.7 Max 2026-06-08",
    "qwen3.7-plus-2026-05-26": "Qwen 3.7 Plus 2026-05-26",
    "qwen3.7-flash-2026-07-15": "Qwen 3.7 Flash 2026-07-15",
    "qwen3.5-397b-a17b": "Qwen 3.5 397B A17B",
}


class QwenCloudCatalogError(RuntimeError):
    """The authenticated QwenCloud catalog could not be frozen safely."""


def _nonnegative_decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise QwenCloudCatalogError(f"{field} is not a valid decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise QwenCloudCatalogError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _safe_base_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    subscription_only_host = hostname.startswith(("token-plan.", "coding.")) or hostname in {
        "coding-intl.dashscope.aliyuncs.com",
        "coding.dashscope.aliyuncs.com",
    }
    approved_host = not subscription_only_host and (
        hostname == "dashscope-intl.aliyuncs.com" or hostname.endswith(".maas.aliyuncs.com")
    )
    if parsed.scheme != "https" or not approved_host or parsed.username or parsed.password:
        raise QwenCloudCatalogError(
            "QwenCloud pay-as-you-go credentials may only be sent to an approved HTTPS host"
        )
    return value.rstrip("/")


def _is_pay_as_you_go_api_key(value: str) -> bool:
    """Accept general Model Studio keys while excluding isolated plan keys."""

    return value.startswith("sk-") and not value.startswith("sk-sp-")


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
            raise QwenCloudCatalogError("content-addressed QwenCloud artifact conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def verify_content_address(document: Mapping[str, Any], schema_version: str) -> bool:
    digest = document.get("artifact_sha256")
    payload = dict(document)
    payload.pop("artifact_sha256", None)
    return (
        document.get("schema_version") == schema_version
        and isinstance(digest, str)
        and digest == sha256_json(payload)
    )


async def fetch_authenticated_catalog(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 60,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Fetch the OpenAI-compatible model list; never invoke a generation route."""

    if not _is_pay_as_you_go_api_key(api_key):
        raise QwenCloudCatalogError("DASHSCOPE_API_KEY is missing or is not a pay-as-you-go key")
    approved = _safe_base_url(base_url)
    async with httpx.AsyncClient(
        base_url=f"{approved}/",
        timeout=timeout_seconds,
        transport=transport,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    ) as client:
        response = await client.get("models")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise QwenCloudCatalogError(
            f"QwenCloud catalog request failed with HTTP {response.status_code}"
        ) from error
    body = response.json()
    if not isinstance(body, Mapping) or body.get("object") != "list":
        raise QwenCloudCatalogError("QwenCloud returned an invalid model-list envelope")
    rows = body.get("data")
    if not isinstance(rows, list) or not rows:
        raise QwenCloudCatalogError("QwenCloud returned an empty or invalid model list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise QwenCloudCatalogError("QwenCloud model list contains a non-object row")
        model_id = str(raw.get("id") or "")
        if not model_id or model_id in seen:
            raise QwenCloudCatalogError("QwenCloud model identities are empty or duplicated")
        seen.add(model_id)
        normalized.append(dict(raw))
    normalized.sort(key=lambda row: str(row["id"]))
    return {"object": "list", "data": normalized}, response.headers.get("Date")


def build_catalog_artifact(
    *,
    catalog: Mapping[str, Any],
    observed_at: str,
    response_date: str | None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    rows = catalog.get("data")
    if catalog.get("object") != "list" or not isinstance(rows, list) or not rows:
        raise QwenCloudCatalogError("cannot freeze an invalid QwenCloud model list")
    ids = [str(row.get("id") or "") for row in rows if isinstance(row, Mapping)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids) or any(not model_id for model_id in ids):
        raise QwenCloudCatalogError("cannot freeze incomplete or duplicated QwenCloud identities")
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "provider": "qwencloud",
        "catalog_endpoint": f"{_safe_base_url(base_url)}/models",
        "observed_at": observed_at,
        "provider_response_date": response_date,
        "model_count": len(rows),
        "models": [dict(row) for row in sorted(rows, key=lambda row: str(row["id"]))],
        "request_accounting": {
            "catalog_requests": 1,
            "provider_generation_requests": 0,
            "epicure_calls": 0,
            "spend_usd": "0",
        },
        "official": False,
        "rank_eligible": False,
        "quality_observations": 0,
    }


def build_candidate_extension(
    *,
    catalog_artifact: Mapping[str, Any],
    candidate_ids: Sequence[str] = DEFAULT_CANDIDATES,
    discovery_aliases: Sequence[str] = DISCOVERY_ALIASES,
) -> dict[str, Any]:
    if not verify_content_address(catalog_artifact, CATALOG_SCHEMA_VERSION):
        raise QwenCloudCatalogError("QwenCloud catalog content address does not verify")
    rows = catalog_artifact.get("models")
    if not isinstance(rows, list):
        raise QwenCloudCatalogError("QwenCloud catalog artifact has no model rows")
    by_id = {
        str(row.get("id")): dict(row) for row in rows if isinstance(row, Mapping) and row.get("id")
    }
    requested = [*discovery_aliases, *candidate_ids]
    absent = [model_id for model_id in requested if model_id not in by_id]
    if absent:
        raise QwenCloudCatalogError(
            "requested QwenCloud models are absent from the authenticated catalog: "
            + ", ".join(absent)
        )

    def project(model_id: str, *, alias_only: bool) -> dict[str, Any]:
        row = by_id[model_id]
        return {
            "model_id": model_id,
            "display_name": DISPLAY_NAMES.get(model_id, model_id),
            "provider": "QwenCloud direct API",
            "execution_backend": "qwencloud_direct",
            "catalog_row_sha256": sha256_json(row),
            "identity_kind": "mutable_alias" if alias_only else "frozen_candidate",
            "compatibility_state": "discovered",
            "contract_smoke_passed": False,
            "season_eligible": False,
            "scheduled_arms": 0,
            "observed_arms": 0,
            "provider_generation_calls": 0,
            "epicure_calls": 0,
            "quality_judgments": 0,
            "rank_eligible": False,
            "result_inclusion": "excluded_until_governed_real_run",
        }

    candidates = [project(model_id, alias_only=False) for model_id in candidate_ids]
    aliases = [project(model_id, alias_only=True) for model_id in discovery_aliases]
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "provider": "qwencloud",
        "catalog_artifact_sha256": catalog_artifact["artifact_sha256"],
        "catalog_observed_at": catalog_artifact["observed_at"],
        "catalog_endpoint": catalog_artifact["catalog_endpoint"],
        "provider_model_documentation": PROVIDER_MODEL_DOCUMENTATION,
        "provider_documentation_observed_date": "2026-07-14",
        "catalog_model_count": catalog_artifact["model_count"],
        "discovery_aliases": aliases,
        "candidates": candidates,
        "counts": {
            "discovery_aliases": len(aliases),
            "frozen_candidates": len(candidates),
            "observed_arms": 0,
            "provider_generation_calls": 0,
            "epicure_calls": 0,
            "quality_judgments": 0,
            "rankable_comparisons": 0,
        },
        "requested_user_label": {
            "label": "qwen3.8-max",
            "status": "authenticated_catalog_discovered_mutable_alias",
            "included_as_model": True,
            "frozen_execution_candidate": False,
            "observed_arms": 0,
            "rank_eligible": False,
        },
        "claim_boundary": {
            "catalog_discovery_only": True,
            "tool_compatibility_unverified": True,
            "structured_output_compatibility_unverified": True,
            "actual_returned_model_unverified": True,
            "cost_accounting_unverified": True,
            "openrouter_routes_are_separate_strata": True,
            "automatic_fallback_permitted": False,
            "official": False,
            "rank_eligible": False,
        },
    }


def build_unranked_qwen37_route_manifest(
    *,
    catalog_artifact: Mapping[str, Any],
    cap_usd: Decimal | str = "2",
    model_id: str = "qwen3.7-max-2026-06-08",
    allow_mutable_alias_exploratory: bool = False,
    tool_auto_successor_failure_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze one executable, permanently unranked QwenCloud route.

    The authenticated model catalog supplies identity evidence only.  Capability
    and list-price fields are kept separate and preserve their documented
    limitations: function calling is supported, structured outputs are not, and
    the Chat Completions API exposes usage rather than a charged amount.  The
    mutable Qwen 3.8 alias is admitted only with an explicit opt-in and is
    labelled as catalog-pinned at one observation, never as a frozen model.
    """

    if not verify_content_address(catalog_artifact, CATALOG_SCHEMA_VERSION):
        raise QwenCloudCatalogError("QwenCloud catalog content address does not verify")
    mutable_alias = model_id == "qwen3.8-max"
    tool_auto_successor = tool_auto_successor_failure_sha256 is not None
    if mutable_alias and not allow_mutable_alias_exploratory:
        raise QwenCloudCatalogError(
            "qwen3.8-max requires explicit mutable-alias exploratory opt-in"
        )
    if model_id not in {"qwen3.7-max-2026-06-08", "qwen3.8-max"}:
        raise QwenCloudCatalogError(
            "only the dated Qwen 3.7 release and explicitly exploratory Qwen 3.8 "
            "alias have execution contracts"
        )
    if tool_auto_successor and (
        not mutable_alias
        or len(str(tool_auto_successor_failure_sha256)) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(tool_auto_successor_failure_sha256)
        )
    ):
        raise QwenCloudCatalogError(
            "tool-auto successor requires one exact prior Qwen 3.8 failure artifact"
        )
    rows = catalog_artifact.get("models")
    if not isinstance(rows, list):
        raise QwenCloudCatalogError("QwenCloud catalog artifact has no model rows")
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("id") or "") == model_id
    ]
    if len(matches) != 1:
        raise QwenCloudCatalogError("dated QwenCloud candidate is absent or duplicated")
    cap = _nonnegative_decimal(cap_usd, field="cap_usd")
    if cap <= 0 or cap > Decimal("100"):
        raise QwenCloudCatalogError("QwenCloud route cap must be in (0, 100]")

    catalog_row = matches[0]
    catalog_sha = str(catalog_artifact["artifact_sha256"])
    catalog_entry_sha = sha256_json(catalog_row)
    provider_slug = "qwencloud-direct"
    request_bound = 18
    prompt_token_bound = 16_000
    completion_token_bound = 8_192
    if mutable_alias:
        pricing = {
            "currency": "USD",
            "prompt": "0",
            "completion": "0",
            "internal_reasoning": "0",
            "request": "0",
            "source": PROVIDER_PRICING_DOCUMENTATION,
            "source_observed_date": "2026-08-08",
            "status": "provider_rate_unpublished_at_catalog_observation",
            "provider_rate_known": False,
            "operational_reservation_ceiling_usd": _decimal_text(cap),
            "operational_reservation_request_bound": request_bound,
            "zero_values_mean": "unknown_cost_not_free",
        }
    else:
        pricing = {
            "currency": "USD",
            "prompt": "0.0000025",
            "completion": "0.0000075",
            "internal_reasoning": "0",
            "request": "0",
            "source": PROVIDER_PRICING_DOCUMENTATION,
            "source_observed_date": "2026-08-08",
            "status": "frozen_public_rate_card_provider_charge_unavailable",
            "provider_rate_known": True,
        }
    endpoint = {
        "model_id": model_id,
        "provider_name": provider_slug,
        "tag": provider_slug,
        "quantization": "provider_managed_unpublished",
        "context_length": None if mutable_alias else 1_000_000,
        "max_completion_tokens": None if mutable_alias else 65_536,
        "pricing": pricing,
        "supported_parameters": [
            "max_tokens",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
    }
    backend_contract = {
        "schema_version": "flavourbench-qwencloud-direct-endpoint-contract-v1",
        "base_url": DEFAULT_BASE_URL,
        "requested_model_id": model_id,
        "expected_actual_provider_slug": provider_slug,
        "catalog_sha256": catalog_sha,
        "catalog_entry_sha256": catalog_entry_sha,
        "identity_kind": "mutable_alias" if mutable_alias else "immutable_dated_release",
        "identity_evidence": (
            "authenticated_catalog_row_at_observation_exact_request_and_response_alias"
            if mutable_alias
            else "authenticated_catalog_exact_request_and_response_model"
        ),
        "catalog_observed_at": catalog_artifact["observed_at"],
        "catalog_pinned_at_observation": mutable_alias,
        "model_identity_label": (
            "catalog_pinned_at_observation_not_a_frozen_model"
            if mutable_alias
            else "immutable_dated_release"
        ),
        "mutable_alias_execution_requires_explicit_opt_in": mutable_alias,
        "function_calling_supported": ("unverified_contract_target" if mutable_alias else True),
        "structured_outputs_supported": False,
        "reasoning_effort_translation": "unsupported_omitted",
        "allow_fallbacks": False,
        "cost_accounting": (
            "provider_usage_with_unpriced_budget_ceiling"
            if mutable_alias
            else "provider_usage_times_frozen_rate_card"
        ),
        "cost_reconciliation": (
            "provider_rate_and_charge_unavailable"
            if mutable_alias
            else "provider_charge_unavailable"
        ),
        "provider_rate_status": (
            "unpublished_at_catalog_observation" if mutable_alias else "published_rate_card"
        ),
        "data_policy": "public_nonpersonal_contract_smoke_only",
        "openrouter_alternate_route": "separate_stratum_only_no_identity_pooling",
        "season_eligible": False,
        "rank_eligible": False,
        "official": False,
    }
    if tool_auto_successor:
        backend_contract.update(
            {
                "function_calling_supported": "successor_contract_pending_real_retest",
                "tool_choice_transport_mode": "auto_with_required_success_postcondition",
                "tool_choice_required_supported": False,
                "tool_choice_required_rejection_diagnosis": (
                    "inference_from_initial_tool_selection_http_400_and_provider_contract"
                ),
                "required_success_postcondition": (
                    "at_least_one_successful_real_epicure_tool_trace"
                ),
                "tool_selection_system_instruction": QWEN38_TOOL_AUTO_INSTRUCTION,
                "tool_selection_system_instruction_sha256": sha256_json(
                    QWEN38_TOOL_AUTO_INSTRUCTION
                ),
                "message_canonicalization": ("official_qwen_chat_tool_continuation_shape_v1"),
                "predecessor_failure_artifact_sha256": (tool_auto_successor_failure_sha256),
                "function_calling_documentation": (QWENCLOUD_FUNCTION_CALLING_DOCUMENTATION),
                "error_documentation": QWENCLOUD_ERROR_DOCUMENTATION,
            }
        )
    backend_contract_sha = sha256_json(backend_contract)

    # A pair can issue at most eighteen requests under the eight-round bound.
    # Qwen 3.7 uses its published list price.  Qwen 3.8 has no published direct
    # rate at this catalog observation, so its complete explicit allowance is
    # retained as exposure rather than fabricating a cost estimate.
    forecast = (
        cap
        if mutable_alias
        else Decimal(request_bound)
        * (
            Decimal(prompt_token_bound) * Decimal(pricing["prompt"])
            + Decimal(completion_token_bound) * Decimal(pricing["completion"])
        )
    )
    if forecast > cap:
        raise QwenCloudCatalogError(
            f"bounded QwenCloud forecast ${_decimal_text(forecast)} exceeds cap "
            f"${_decimal_text(cap)}"
        )

    payload: dict[str, Any] = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "status": "unranked_candidate",
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": catalog_artifact["observed_at"],
        "manifest_role": (
            "qwencloud_mutable_alias_tool_auto_successor_contract_smoke_candidate"
            if tool_auto_successor
            else "qwencloud_mutable_alias_public_contract_smoke_candidate"
            if mutable_alias
            else "qwencloud_public_contract_smoke_candidate"
        ),
        "source": {
            "catalog_artifact_sha256": catalog_sha,
            "catalog_entry_sha256": catalog_entry_sha,
            "catalog_endpoint": catalog_artifact["catalog_endpoint"],
            "model_documentation": QWEN37_MODEL_DOCUMENTATION,
            "pricing_documentation": PROVIDER_PRICING_DOCUMENTATION,
            **(
                {
                    "function_calling_documentation": (QWENCLOUD_FUNCTION_CALLING_DOCUMENTATION),
                    "error_documentation": QWENCLOUD_ERROR_DOCUMENTATION,
                    "predecessor_failure_artifact_sha256": (tool_auto_successor_failure_sha256),
                }
                if tool_auto_successor
                else {}
            ),
        },
        "selection": {
            "method": (
                "one authenticated mutable alias pinned only at catalog observation"
                if mutable_alias
                else "one authenticated dated model identity; no alias substitution"
            ),
            "model_count": 1,
            "performance_claim": "none; route inclusion is not a quality result",
        },
        "budget": {
            "currency": "USD",
            "cap_usd": _decimal_text(cap),
            "bounded_forecast_usd": _decimal_text(forecast),
            "within_cap": True,
            "forecast_policy": {
                "request_bound": request_bound,
                "prompt_tokens_per_request_bound": prompt_token_bound,
                "completion_tokens_per_request_bound": completion_token_bound,
                "provider_rate_known": not mutable_alias,
                "full_ceiling_retained_when_cost_unknown": mutable_alias,
            },
        },
        "models": [
            {
                "slot": {
                    "slot_id": (
                        "qwencloud-qwen38-max-alias-observation-exploratory"
                        if mutable_alias
                        else "qwencloud-qwen37-max-dated-exploratory"
                    ),
                    "cohort": "current_frontier_exploratory",
                    "model_id": model_id,
                    "rationale": (
                        "mutable alias for an explicitly labelled real tool contract smoke"
                        if mutable_alias
                        else "dated QwenCloud identity for a real tool contract smoke"
                    ),
                    "open_weight_candidate": False,
                },
                "model": {
                    "id": model_id,
                    "canonical_slug": model_id,
                    "name": (
                        "Qwen 3.8 Max (catalog-pinned mutable alias)"
                        if mutable_alias
                        else "Qwen 3.7 Max 2026-06-08"
                    ),
                    "description": (
                        "Mutable QwenCloud alias pinned only to one authenticated catalog "
                        "observation; not a frozen model."
                        if mutable_alias
                        else "Catalog-pinned dated QwenCloud managed model."
                    ),
                    "catalog_entry": catalog_row,
                    "context_length": endpoint["context_length"],
                    "supported_parameters": endpoint["supported_parameters"],
                },
                "endpoint": endpoint,
                "endpoint_document_sha256": sha256_json(endpoint),
                "backend_contract": backend_contract,
                "backend_contract_sha256": backend_contract_sha,
                "cost_accounting_policy": backend_contract["cost_accounting"],
                "execution_route": {
                    "policy": (
                        "exact_direct_qwencloud_tool_auto_successor_v2"
                        if tool_auto_successor
                        else "exact_direct_qwencloud_v1"
                    ),
                    "preferred_backend": "qwencloud_direct",
                    "selected_backend": "qwencloud_direct",
                    "selection_frozen_before_generation": True,
                    "fallback_used": False,
                    "generation_time_automatic_fallback": False,
                    "evidence": {
                        "catalog_sha256": catalog_sha,
                        "catalog_entry_sha256": catalog_entry_sha,
                    },
                },
                "request_policy": {
                    "provider": {
                        "only": [provider_slug],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                        "data_collection": "public_nonpersonal_contract_smoke_only",
                    },
                    "policy_scope": "fixed_public_prompt_only",
                    "official_eligibility": "development_only",
                    **(
                        {
                            "tool_choice_transport": "auto",
                            "required_tool_success_enforced_after_response": True,
                            "message_canonicalization": (
                                "official_qwen_chat_tool_continuation_shape_v1"
                            ),
                        }
                        if tool_auto_successor
                        else {}
                    ),
                },
                "contract_evidence": {
                    "contract_status": "not_run",
                    "real_provider_calls": 0,
                    "real_epicure_calls": 0,
                    "quality_judgments": 0,
                    "rank_eligible": False,
                    "identity_label": backend_contract["model_identity_label"],
                },
            }
        ],
        "governance": {
            "official": False,
            "rank_eligible": False,
            "season_eligible": False,
            "data_policy": "public non-personal smoke prompt only",
            "required_before_any_generation": [
                "preflight against the frozen Epicure attestation",
                "transactional QwenCloud account and season budget reservation",
                "one exact human-PI execution authorization",
                *(["explicit --allow-mutable-alias-exploratory opt-in"] if mutable_alias else []),
            ],
            "required_before_official_collection": [
                "structured-output contract pass",
                "provider charged-amount reconciliation",
                "immutable model identity",
                "approved provider data policy",
            ],
            "cross_provider_identity_policy": (
                "Direct QwenCloud and OpenRouter Qwen routes are separate strata; "
                "automatic fallback and identity pooling are prohibited."
            ),
            "model_identity_policy": backend_contract["model_identity_label"],
        },
    }
    digest = sha256_json(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return payload


def write_qwencloud_route_manifest(
    manifest: Mapping[str, Any],
    output_directory: Path,
) -> Path:
    """Atomically write a verified route manifest with its complete digest."""

    content_address = manifest.get("content_address")
    digest = (
        str(content_address.get("digest") or "") if isinstance(content_address, Mapping) else ""
    )
    unhashed = dict(manifest)
    unhashed.pop("content_address", None)
    if (
        len(digest) != 64
        or digest != sha256_json(unhashed)
        or content_address.get("algorithm") != "sha256"
        or content_address.get("uri") != f"sha256:{digest}"
    ):
        raise QwenCloudCatalogError("QwenCloud route manifest content address does not verify")
    rendered = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"qwencloud-route-manifest-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise QwenCloudCatalogError("content-addressed QwenCloud route conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def build_unranked_qwen38_alias_route_manifest(
    *,
    catalog_artifact: Mapping[str, Any],
    cap_usd: Decimal | str = "2",
    allow_mutable_alias_exploratory: bool = False,
    tool_auto_successor_failure_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the opt-in Qwen 3.8 alias route without laundering its identity."""

    return build_unranked_qwen37_route_manifest(
        catalog_artifact=catalog_artifact,
        cap_usd=cap_usd,
        model_id="qwen3.8-max",
        allow_mutable_alias_exploratory=allow_mutable_alias_exploratory,
        tool_auto_successor_failure_sha256=tool_auto_successor_failure_sha256,
    )


async def _async_run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"catalog freeze requires --confirm {CONFIRMATION}")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    catalog, response_date = await fetch_authenticated_catalog(
        api_key=api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    catalog_payload = build_catalog_artifact(
        catalog=catalog,
        observed_at=observed_at,
        response_date=response_date,
        base_url=args.base_url,
    )
    catalog_path = _atomic_write(args.output_dir, "qwencloud-model-catalog", catalog_payload)
    catalog_document = json.loads(catalog_path.read_text(encoding="utf-8"))
    extension_payload = build_candidate_extension(catalog_artifact=catalog_document)
    extension_path = _atomic_write(
        args.output_dir,
        "qwencloud-candidate-extension",
        extension_payload,
    )
    return catalog_path, extension_path


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    catalog_path, extension_path = asyncio.run(_async_run(args))
    print(catalog_path)
    print(extension_path)


if __name__ == "__main__":
    run()
