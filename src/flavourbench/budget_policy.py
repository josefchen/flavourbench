"""Immutable account-wide spend policy for every paid FlavourBench backend.

The scope is deliberately installation-wide rather than credential-derived.  A
rotated API key or AWS credential therefore cannot mint a fresh budget.  Raising
either ceiling requires a reviewed protocol/code change, not an environment
variable or a season-freeze request.
"""

from __future__ import annotations

import hashlib
import json

PROVIDER_ACCOUNT_HARD_CAP_MICROS = {
    "bedrock": 5_000_000_000,
    "cohere_direct": 100_000_000,
    "kimi_direct": 100_000_000,
    "openrouter": 100_000_000,
    "qwencloud_direct": 100_000_000,
    "zai_coding_direct": 100_000_000,
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def provider_account_scope_sha256(execution_backend: str) -> str:
    """Return the non-resettable scope used by the shared account ledger."""

    if execution_backend not in PROVIDER_ACCOUNT_HARD_CAP_MICROS:
        raise ValueError("unsupported paid execution backend")
    return canonical_sha256(
        {
            "schema_version": "flavourbench-provider-account-scope-v1",
            "execution_backend": execution_backend,
            "scope": "flavourbench-installation-wide-all-credentials",
        }
    )


def provider_account_hard_cap_micros(execution_backend: str) -> int:
    try:
        return PROVIDER_ACCOUNT_HARD_CAP_MICROS[execution_backend]
    except KeyError as error:
        raise ValueError("unsupported paid execution backend") from error
