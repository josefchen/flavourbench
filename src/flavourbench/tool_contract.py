"""Frozen, unranked transport diagnostic for Epicure tool integration."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .execution_policy import DIRECT_TOOL_CONTRACT_PROTOCOL, ExecutionPolicy

TOOL_CONTRACT_SCHEMA_VERSION = "flavourbench-required-tool-contract-v2"
TOOL_CONTRACT_NAME = "find_pairings"
TOOL_CONTRACT_SYSTEM_PROMPT = """You are participating in a blinded, unranked
FlavourBench tool-integration diagnostic. On your first response, call the only exposed culinary
evidence tool exactly once with valid arguments. Do not provide an answer before the tool result.
After the result is returned, provide a concise practical culinary answer. Never identify the
model, provider, tool infrastructure, or evaluation condition."""
TOOL_CONTRACT_PROMPT = (
    "Call find_pairings once for watermelon, green olive, and mint. After the evidence returns, "
    "propose one practical starter and distinguish suggestive pairing evidence from claims that "
    "still require tasting."
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def required_tool_contract(policy: ExecutionPolicy) -> dict[str, Any]:
    """Return the complete content-addressed diagnostic contract."""

    policy.validate()
    payload: dict[str, Any] = {
        "schema_version": TOOL_CONTRACT_SCHEMA_VERSION,
        "protocol": DIRECT_TOOL_CONTRACT_PROTOCOL,
        "lane_class": "unranked_endpoint_capability_diagnostic",
        "official": False,
        "rank_eligible": False,
        "system_prompt_sha256": text_sha256(TOOL_CONTRACT_SYSTEM_PROMPT),
        "user_prompt_sha256": text_sha256(TOOL_CONTRACT_PROMPT),
        "first_provider_phase": "tool_round_0",
        "tool_catalog": {
            "mode": "singleton_from_attested_epicure_catalog",
            "required_tool_name": TOOL_CONTRACT_NAME,
            "tool_choice": "required",
            "initial_call_count": {"minimum": 1, "maximum": 1},
        },
        "limits": {
            "max_tool_rounds": policy.max_tool_rounds,
            "max_tool_calls_per_round": 1,
            "max_tool_calls_total": min(policy.max_tool_calls_total, 2),
            "invalid_argument_repair_turns": policy.tool_argument_repair_turns,
            "max_intermediate_tokens": (
                policy.required_tool_contract_max_intermediate_tokens
            ),
            "max_output_tokens": policy.max_output_tokens,
        },
        "reasoning": {
            "intermediate_effort": policy.intermediate_reasoning_effort,
            "final_effort": policy.final_reasoning_effort,
            "exclude_from_provider_response": True,
        },
        "final_response_mode": policy.final_response_mode,
        "excluded_from_metrics": [
            "preference",
            "uplift",
            "answer_quality",
            "benchmark_cost",
            "benchmark_latency",
        ],
    }
    digest = _sha256(payload)
    return {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
