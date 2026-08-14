"""Freeze the dated 26-model FlavourBench frontier successor roster.

The successor retains sixteen complete response blocks from the powered-v31
run and schedules ten complete replacement/addition blocks.  Endpoint
selection is based only on the public route catalogue, before score data from
any successor call exists.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest import _fetch_endpoints, _select_exact
from .frontier_manifest import verify_manifest_content_address

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
MANIFEST_ROLE = "epicure_selection_powered_frontier_successor_20260814"
CAP_USD = Decimal("200")
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
REQUESTS_PER_NEW_BLOCK = PRIMARY_TASKS + REPEAT_TASKS
PROMPT_TOKEN_BOUND = 4_096

RETAINED_MODEL_IDS = (
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-luna-pro",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.6-flash",
    "moonshotai/kimi-k3",
    "qwen/qwen3.8-max",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-flash-0731",
    "minimax/minimax-m3",
    "nvidia/nemotron-3.5-lightning",
    "mistralai/mistral-large-2512",
    "tencent/hy3",
)

OPENROUTER_SUCCESSORS: dict[str, dict[str, Any]] = {
    "x-ai/grok-4.6": {
        "slot_id": "frontier-refresh-09",
        "route": "xai/zdr",
        "cohort": "xai",
        "rationale": "Grok 4.6 on the exact native xAI zero-data-retention route.",
        "max_output_tokens": 2_048,
    },
    "deepseek/deepseek-v4-pro-0813": {
        "slot_id": "frontier-refresh-13",
        "route": "deepseek",
        "cohort": "deepseek",
        "rationale": "Dated DeepSeek V4 Pro 0813 on the exact native DeepSeek route.",
        "max_output_tokens": 2_048,
    },
    "meta/muse-spark-1.2": {
        "slot_id": "frontier-refresh-21",
        "route": "meta",
        "cohort": "meta",
        "rationale": "Muse Spark 1.2 on the exact first-party Meta route.",
        "max_output_tokens": 4_096,
    },
    "meta/muse-glimmer-30b": {
        "slot_id": "frontier-refresh-22",
        "route": "deepinfra/bf16",
        "cohort": "meta",
        "rationale": "Muse Glimmer 30B on the exact DeepInfra BF16 route.",
        "max_output_tokens": 2_048,
    },
    "anthropic/claude-fable-5": {
        "slot_id": "frontier-refresh-23",
        "route": "amazon-bedrock/claude-on-aws",
        "cohort": "anthropic",
        "rationale": "Claude Fable 5 on the exact Amazon Bedrock route.",
        "max_output_tokens": 2_048,
    },
    "qwen/qwen3.8-2.4t-a95b": {
        "slot_id": "frontier-refresh-24",
        "route": "digitalocean",
        "cohort": "qwen",
        "rationale": "Qwen3.8 2.4T A95B on the exact DigitalOcean route.",
        "max_output_tokens": 4_096,
    },
    "bytedance-seed/seed-2-1-turbo": {
        "slot_id": "frontier-refresh-25",
        "route": "seed/fp8",
        "cohort": "bytedance-seed",
        "rationale": "Seed 2.1 Turbo on the exact first-party Seed FP8 route.",
        "max_output_tokens": 2_048,
    },
    "thinkingmachines/inkling": {
        "slot_id": "frontier-refresh-26",
        "route": "baseten/fp8",
        "cohort": "thinking-machines",
        "rationale": "Inkling on the exact BaseTen FP8 route.",
        "max_output_tokens": 2_048,
    },
}

DIRECT_COHERE_MODEL_IDS = (
    "cohere/command-a-plus-05-2026",
    "cohere/command-a-reasoning-08-2025",
)
NEW_MODEL_IDS = (
    "x-ai/grok-4.6",
    "deepseek/deepseek-v4-pro-0813",
    *DIRECT_COHERE_MODEL_IDS,
    "meta/muse-spark-1.2",
    "meta/muse-glimmer-30b",
    "anthropic/claude-fable-5",
    "qwen/qwen3.8-2.4t-a95b",
    "bytedance-seed/seed-2-1-turbo",
    "thinkingmachines/inkling",
)
PANEL_ORDER = (
    "openai/gpt-5.6-sol-pro",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-luna-pro",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.6-flash",
    "x-ai/grok-4.6",
    "moonshotai/kimi-k3",
    "qwen/qwen3.8-max",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-pro-0813",
    "deepseek/deepseek-v4-flash-0731",
    "minimax/minimax-m3",
    "nvidia/nemotron-3.5-lightning",
    "mistralai/mistral-large-2512",
    "tencent/hy3",
    "cohere/command-a-plus-05-2026",
    "cohere/command-a-reasoning-08-2025",
    "meta/muse-spark-1.2",
    "meta/muse-glimmer-30b",
    "anthropic/claude-fable-5",
    "qwen/qwen3.8-2.4t-a95b",
    "bytedance-seed/seed-2-1-turbo",
    "thinkingmachines/inkling",
)


class FrontierRefresh26Error(RuntimeError):
    """The dated 26-model successor could not be frozen safely."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrontierRefresh26Error(f"manifest is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise FrontierRefresh26Error(f"manifest content address failed: {path}")
    return value


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _block_envelope(endpoint: Mapping[str, Any], *, max_output_tokens: int) -> Decimal:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * PROMPT_TOKEN_BOUND
    completion = Decimal(str(pricing.get("completion") or 0)) * max_output_tokens
    reasoning = Decimal(str(pricing.get("internal_reasoning") or 0)) * max_output_tokens
    request = Decimal(str(pricing.get("request") or 0))
    return (prompt + completion + reasoning + request) * REQUESTS_PER_NEW_BLOCK


def _canonical_slug(endpoint: Mapping[str, Any]) -> str:
    name = str(endpoint.get("name") or "")
    if " | " not in name:
        raise FrontierRefresh26Error("OpenRouter endpoint lacks a canonical dated identity")
    return name.split(" | ", 1)[1]


def _openrouter_entry(
    template: Mapping[str, Any],
    *,
    model_id: str,
    spec: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    endpoint = copy.deepcopy(dict(endpoint))
    if endpoint.get("model_id") != model_id or endpoint.get("tag") != spec["route"]:
        raise FrontierRefresh26Error(f"endpoint identity changed for {model_id}")
    supported = sorted(str(value) for value in endpoint.get("supported_parameters") or [])
    endpoint["supported_parameters"] = supported
    value = copy.deepcopy(dict(template))
    value["slot"] = {
        "slot_id": spec["slot_id"],
        "model_id": model_id,
        "cohort": spec["cohort"],
        "open_weight_candidate": model_id.startswith(("meta/", "qwen/", "deepseek/")),
        "rationale": spec["rationale"],
    }
    value["model"] = {
        "id": model_id,
        "canonical_slug": _canonical_slug(endpoint),
        "name": str(endpoint["model_name"]),
        "description": spec["rationale"],
        "context_length": int(endpoint["context_length"]),
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "modality": "text->text",
        },
        "supported_parameters": supported,
        "top_provider": {
            "context_length": int(endpoint["context_length"]),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "is_moderated": False,
        },
    }
    value["endpoint"] = endpoint
    value["endpoint_document_sha256"] = _sha256(endpoint)
    value["endpoint_selection"] = {
        "method": "exact dated endpoint selected before successor score collection",
        "selected_exact_tag": spec["route"],
        "eligible_endpoint_count": 1,
        "quality_observations_used": 0,
        "observed_at": observed_at,
        "automatic_fallback": False,
    }
    value["request_policy"] = {
        "official_eligibility": "successor_collection_only",
        "policy_scope": "request_enforced",
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "only": [spec["route"]],
            "require_parameters": True,
        },
    }
    value["execution_route"] = {
        "preferred_backend": "openrouter",
        "selected_backend": "openrouter",
        "policy": "exact_openrouter_provider_only_v1",
        "fallback_used": False,
        "generation_time_automatic_fallback": False,
        "selection_frozen_before_generation": True,
        "selection_reason": "dated_frontier_successor_route_freeze",
    }
    value["contract_evidence"] = {
        "status": "route_frozen_before_complete_successor_block",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_complete_primary_and_repeat_blocks": True,
    }
    value["backend_contract"] = {}
    value["backend_contract_sha256"] = "unfrozen"
    value["cost_accounting_policy"] = "provider_generation_metadata"
    value["forecast"] = {
        "primary_tasks": PRIMARY_TASKS,
        "repeat_tasks": REPEAT_TASKS,
        "prompt_token_bound": PROMPT_TOKEN_BOUND,
        "route_max_output_tokens": spec["max_output_tokens"],
        "new_provider_calls": REQUESTS_PER_NEW_BLOCK,
        "model_block_worst_case_usd": _decimal_text(
            _block_envelope(endpoint, max_output_tokens=int(spec["max_output_tokens"]))
        ),
    }
    value["open_weight_evidence"] = None
    return value


def _address(document: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(document))
    value.pop("content_address", None)
    digest = _sha256(value)
    value["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return value


async def build(*, current_manifest_path: Path, direct_manifest_path: Path) -> dict[str, Any]:
    current = _load_manifest(current_manifest_path)
    direct = _load_manifest(direct_manifest_path)
    current_by_id = {str(row["model"]["id"]): row for row in current["models"]}
    direct_by_id = {str(row["model"]["id"]): row for row in direct["models"]}
    if not set(RETAINED_MODEL_IDS) <= set(current_by_id):
        raise FrontierRefresh26Error("current manifest lacks a retained response source")
    if not set(DIRECT_COHERE_MODEL_IDS) <= set(direct_by_id):
        raise FrontierRefresh26Error("direct manifest lacks the exact Cohere successors")

    endpoint_sets = await asyncio.gather(
        *(_fetch_endpoints(model_id) for model_id in OPENROUTER_SUCCESSORS)
    )
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    template = current_by_id["x-ai/grok-4.5"]
    by_id: dict[str, dict[str, Any]] = {}
    for model_id in RETAINED_MODEL_IDS:
        entry = copy.deepcopy(current_by_id[model_id])
        entry["forecast"] = {
            "response_source": "complete_powered_v31_block_reused_exactly",
            "new_provider_calls": 0,
            "model_block_worst_case_usd": "0",
        }
        by_id[model_id] = entry
    for (model_id, spec), endpoints in zip(
        OPENROUTER_SUCCESSORS.items(), endpoint_sets, strict=True
    ):
        endpoint = _select_exact(model_id=model_id, tag=str(spec["route"]), endpoints=endpoints)
        by_id[model_id] = _openrouter_entry(
            template,
            model_id=model_id,
            spec=spec,
            endpoint=endpoint,
            observed_at=observed_at,
        )
    for model_id, slot_id in zip(
        DIRECT_COHERE_MODEL_IDS, ("frontier-refresh-19", "frontier-refresh-20"), strict=True
    ):
        entry = copy.deepcopy(direct_by_id[model_id])
        entry["slot"] = {
            **dict(entry["slot"]),
            "slot_id": slot_id,
            "model_id": model_id,
        }
        entry["forecast"] = {
            "primary_tasks": PRIMARY_TASKS,
            "repeat_tasks": REPEAT_TASKS,
            "route_max_output_tokens": 1_800,
            "new_provider_calls": REQUESTS_PER_NEW_BLOCK,
            "model_block_worst_case_usd": "0",
            "billing_note": "frozen public rate card; provider charge remains separately observed",
        }
        entry["contract_evidence"] = {
            **dict(entry["contract_evidence"]),
            "status": "exact_direct_route_requires_complete_successor_block",
            "quality_observations": 0,
        }
        by_id[model_id] = entry
    if set(by_id) != set(PANEL_ORDER):
        raise FrontierRefresh26Error("the 26-model panel is incomplete")
    entries = [by_id[model_id] for model_id in PANEL_ORDER]
    slots = [str(entry["slot"]["slot_id"]) for entry in entries]
    if len(set(slots)) != len(PANEL_ORDER):
        raise FrontierRefresh26Error("successor slots are not unique")

    per_model_forecast = {
        model_id: str(by_id[model_id]["forecast"]["model_block_worst_case_usd"])
        for model_id in NEW_MODEL_IDS
    }
    bounded = sum((Decimal(value) for value in per_model_forecast.values()), Decimal())
    if bounded > CAP_USD:
        raise FrontierRefresh26Error(
            f"successor block envelope ${_decimal_text(bounded)} exceeds ${CAP_USD}"
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "status": "unranked_candidate",
        "manifest_role": MANIFEST_ROLE,
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": observed_at,
        "source": {
            "powered_v35_manifest": {
                "semantic_sha256": current["content_address"]["digest"],
                "physical_sha256": _sha256_file(current_manifest_path),
            },
            "direct_cohere_manifest": {
                "semantic_sha256": direct["content_address"]["digest"],
                "physical_sha256": _sha256_file(direct_manifest_path),
            },
        },
        "selection": {
            "method": "dated_frontier_refresh_before_successor_score_collection",
            "model_count": 26,
            "quality_observations_used": 0,
            "route_counts": {"openrouter": 24, "cohere_direct": 2},
            "panel_order": list(PANEL_ORDER),
            "retained_complete_model_blocks": list(RETAINED_MODEL_IDS),
            "new_complete_model_blocks": list(NEW_MODEL_IDS),
        },
        "run_design": {
            "track": "epicure_scored_combinatorial_culinary_decisions",
            "primary_tasks": PRIMARY_TASKS,
            "repeat_tasks": REPEAT_TASKS,
            "new_provider_calls": REQUESTS_PER_NEW_BLOCK * len(NEW_MODEL_IDS),
            "composite_primary_cells": PRIMARY_TASKS * len(PANEL_ORDER),
            "composite_repeat_cells": REPEAT_TASKS * len(PANEL_ORDER),
            "response_source_pooling_within_model": False,
        },
        "models": entries,
        "routing_policy": {
            "resolved_before_successor_generation": True,
            "generation_time_automatic_fallback": False,
            "provider_substitution": "prohibited",
            "direct_cohere_retained": True,
        },
        "budget": {
            "currency": "USD",
            "cap_usd": _decimal_text(CAP_USD),
            "bounded_forecast_usd": _decimal_text(bounded),
            "headroom_usd": _decimal_text(CAP_USD - bounded),
            "per_model_worst_case_usd": dict(sorted(per_model_forecast.items())),
            "forecast_scope": "ten new or replacement 704-response blocks only",
            "within_cap": True,
        },
        "governance": {
            "scope": "automated Epicure-scored leaderboard; no human ratings",
            "historical_scores_reused_only_for_unchanged_exact_model_route_blocks": True,
        },
    }
    return _address(document)


def write_manifest(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_manifest_content_address(document):
        raise FrontierRefresh26Error("refusing to write an invalid manifest")
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = output_directory / f"flavourbench-frontier-refresh-26-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise FrontierRefresh26Error("content-addressed manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            current_manifest_path=args.current_manifest,
            direct_manifest_path=args.direct_manifest,
        )
    )
    path = write_manifest(document, args.output_directory)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_sha256": document["content_address"]["digest"],
                "model_count": len(document["models"]),
                "new_model_blocks": len(NEW_MODEL_IDS),
                "new_provider_calls": document["run_design"]["new_provider_calls"],
                "bounded_forecast_usd": document["budget"]["bounded_forecast_usd"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
