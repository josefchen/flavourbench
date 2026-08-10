from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_ENDPOINT_PARAMETERS = frozenset(
    {
        "max_tokens",
        "response_format",
        "structured_outputs",
        "tool_choice",
        "tools",
    }
)
DECODING_PARAMETERS = frozenset({"max_tokens", "temperature", "top_p", "seed"})
UNFROZEN_VALUES = frozenset({"", "unfrozen", "unresolved"})


def normalized_supported_parameters(values: Sequence[str]) -> list[str]:
    """Return the unique, deterministic endpoint capability snapshot."""

    return sorted(set(values))


def normalized_decoding(values: Mapping[str, Any]) -> dict[str, int | float]:
    """Canonicalize only the decoding fields FlavourBench is allowed to send."""

    return {
        name: values[name]
        for name in sorted(DECODING_PARAMETERS)
        if name in values and values[name] is not None
    }


def endpoint_contract_payload(
    *,
    model_id: str,
    provider_slug: str,
    expected_actual_model_id: str,
    expected_actual_provider_slug: str,
    supported_parameters: Sequence[str],
    decoding: Mapping[str, Any],
    endpoint_max_completion_tokens: int | None,
    endpoint_document_sha256: str,
    routing_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact per-season endpoint contract covered by its digest."""

    return {
        "model_id": model_id,
        "provider_slug": provider_slug,
        "expected_actual_model_id": expected_actual_model_id,
        "expected_actual_provider_slug": expected_actual_provider_slug,
        "supported_parameters": normalized_supported_parameters(supported_parameters),
        "decoding": normalized_decoding(decoding),
        "endpoint_max_completion_tokens": endpoint_max_completion_tokens,
        "endpoint_document_sha256": endpoint_document_sha256,
        "routing_policy": dict(
            routing_policy
            or {
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        ),
    }


def endpoint_contract_sha256(**contract: Any) -> str:
    payload = endpoint_contract_payload(**contract)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
