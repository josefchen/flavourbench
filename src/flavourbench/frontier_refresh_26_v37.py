"""Freeze the transport-repaired 26-model frontier successor."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest import _fetch_endpoints, _select_exact
from .frontier_manifest import verify_manifest_content_address
from .frontier_refresh_26 import (
    CAP_USD,
    PANEL_ORDER,
    PRIMARY_TASKS,
    PROMPT_TOKEN_BOUND,
    REPEAT_TASKS,
    REQUESTS_PER_NEW_BLOCK,
    _address,
    _block_envelope,
    _decimal_text,
    _load_manifest,
    _sha256,
    _sha256_file,
    write_manifest,
)

RETAINED_BASE_MODEL_IDS = (
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
RETAINED_COHERE_MODEL_IDS = (
    "cohere/command-a",
    "cohere/command-r-plus-08-2024",
)
NEW_MODEL_IDS = (
    "x-ai/grok-4.6",
    "deepseek/deepseek-v4-pro-0813",
    "meta/muse-spark-1.2",
    "meta/muse-glimmer-30b",
    "anthropic/claude-fable-5",
    "qwen/qwen3.8-2.4t-a95b",
    "bytedance-seed/seed-2-1-turbo",
    "thinkingmachines/inkling",
)
FINAL_PANEL_ORDER = (
    *PANEL_ORDER[:18],
    *RETAINED_COHERE_MODEL_IDS,
    *PANEL_ORDER[20:],
)
ROUTE_REPAIRS = {
    "deepseek/deepseek-v4-pro-0813": "gmicloud/fp8",
    "anthropic/claude-fable-5": "anthropic",
}
MAX_OUTPUT_TOKENS_BY_MODEL = {
    "x-ai/grok-4.6": 2_048,
    "deepseek/deepseek-v4-pro-0813": 2_048,
    "meta/muse-spark-1.2": 4_096,
    "meta/muse-glimmer-30b": 2_048,
    "anthropic/claude-fable-5": 2_048,
    "qwen/qwen3.8-2.4t-a95b": 4_096,
    "bytedance-seed/seed-2-1-turbo": 4_096,
    "thinkingmachines/inkling": 2_048,
}


class FrontierRefresh26V37Error(RuntimeError):
    """The transport-repaired roster could not be frozen."""


def _pilot_commitment(directory: Path, *, expected_plan_sha256: str) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    physical: list[str] = []
    for path in sorted((directory / "responses/primary").glob("*/response-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(value)
        recorded = str(payload.pop("artifact_sha256", ""))
        if recorded != _sha256(payload) or value.get("plan_sha256") != expected_plan_sha256:
            raise FrontierRefresh26V37Error("v36 transport response failed integrity")
        responses.append(value)
        physical.append(_sha256_file(path))
    if len(responses) != 40:
        raise FrontierRefresh26V37Error("v36 transport pilot is not exact")
    by_model: dict[str, dict[str, int]] = {}
    for row in responses:
        model_id = str(row["model_id"])
        counts = by_model.setdefault(model_id, {"completed": 0, "failed": 0})
        counts[str(row["status"])] += 1
    if len(by_model) != 10 or any(sum(counts.values()) != 4 for counts in by_model.values()):
        raise FrontierRefresh26V37Error("v36 pilot model coverage changed")
    return {
        "plan_sha256": expected_plan_sha256,
        "response_count": len(responses),
        "completed_count": sum(row["status"] == "completed" for row in responses),
        "failed_count": sum(row["status"] == "failed" for row in responses),
        "status_counts_by_model": dict(sorted(by_model.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in responses)
        ),
        "response_physical_set_sha256": _sha256(sorted(physical)),
        "attempt_journal_physical_sha256": _sha256_file(
            directory / "attempts/provider-attempts.jsonl"
        ),
        "scores_or_selections_used": False,
        "responses_used_as_final_score_data": False,
    }


def _replace_endpoint(
    entry: dict[str, Any], *, endpoint: Mapping[str, Any], observed_at: str
) -> None:
    endpoint = copy.deepcopy(dict(endpoint))
    model_id = str(entry["model"]["id"])
    supported = sorted(str(value) for value in endpoint.get("supported_parameters") or [])
    endpoint["supported_parameters"] = supported
    entry["model"].update(
        {
            "canonical_slug": str(endpoint["name"]).split(" | ", 1)[1],
            "name": str(endpoint["model_name"]),
            "context_length": int(endpoint["context_length"]),
            "supported_parameters": supported,
        }
    )
    entry["endpoint"] = endpoint
    entry["endpoint_document_sha256"] = _sha256(endpoint)
    entry["endpoint_selection"] = {
        "method": "exact transport-repair endpoint selected before successor scores",
        "selected_exact_tag": endpoint["tag"],
        "eligible_endpoint_count": 1,
        "quality_observations_used": 0,
        "observed_at": observed_at,
        "automatic_fallback": False,
    }
    entry["request_policy"]["provider"]["only"] = [endpoint["tag"]]
    entry["execution_route"].update(
        {
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
            "selection_frozen_before_generation": True,
            "selection_reason": "v36 transport-only repair",
        }
    )
    max_tokens = MAX_OUTPUT_TOKENS_BY_MODEL[model_id]
    entry["forecast"] = {
        "primary_tasks": PRIMARY_TASKS,
        "repeat_tasks": REPEAT_TASKS,
        "prompt_token_bound": PROMPT_TOKEN_BOUND,
        "route_max_output_tokens": max_tokens,
        "new_provider_calls": REQUESTS_PER_NEW_BLOCK,
        "model_block_worst_case_usd": _decimal_text(
            _block_envelope(endpoint, max_output_tokens=max_tokens)
        ),
    }


async def build(
    *,
    source_manifest_path: Path,
    current_manifest_path: Path,
    v36_plan_sha256: str,
    v36_pilot_directory: Path,
) -> dict[str, Any]:
    source = _load_manifest(source_manifest_path)
    current = _load_manifest(current_manifest_path)
    pilot = _pilot_commitment(v36_pilot_directory, expected_plan_sha256=v36_plan_sha256)
    current_by_id = {str(row["model"]["id"]): row for row in current["models"]}
    endpoint_sets = await asyncio.gather(
        *(_fetch_endpoints(model_id) for model_id in ROUTE_REPAIRS)
    )
    endpoints = {
        model_id: _select_exact(model_id=model_id, tag=tag, endpoints=rows)
        for (model_id, tag), rows in zip(ROUTE_REPAIRS.items(), endpoint_sets, strict=True)
    }
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    observed_at = str(source["observed_at"])
    entries: dict[str, dict[str, Any]] = {
        str(row["model"]["id"]): copy.deepcopy(row) for row in document["models"]
    }
    for model_id in ROUTE_REPAIRS:
        _replace_endpoint(entries[model_id], endpoint=endpoints[model_id], observed_at=observed_at)
    for model_id, slot_id in zip(
        RETAINED_COHERE_MODEL_IDS, ("frontier-refresh-19", "frontier-refresh-20"), strict=True
    ):
        entry = copy.deepcopy(current_by_id[model_id])
        entry["slot"] = {**dict(entry["slot"]), "slot_id": slot_id}
        entry["forecast"] = {
            "response_source": "complete_powered_v35_block_reused_exactly",
            "new_provider_calls": 0,
            "model_block_worst_case_usd": "0",
        }
        entries[model_id] = entry
    for model_id in RETAINED_BASE_MODEL_IDS:
        entries[model_id]["forecast"] = {
            "response_source": "complete_powered_v31_block_reused_exactly",
            "new_provider_calls": 0,
            "model_block_worst_case_usd": "0",
        }
    document["models"] = [entries[model_id] for model_id in FINAL_PANEL_ORDER]
    if len(document["models"]) != 26 or len(entries) != 28:
        raise FrontierRefresh26V37Error("v37 roster replacement is incomplete")

    forecasts = {
        model_id: str(entries[model_id]["forecast"]["model_block_worst_case_usd"])
        for model_id in NEW_MODEL_IDS
    }
    bounded = sum((Decimal(value) for value in forecasts.values()), Decimal())
    if bounded > CAP_USD:
        raise FrontierRefresh26V37Error("v37 exact new-block envelope exceeds cap")
    document.update(
        {
            "manifest_role": "epicure_selection_powered_frontier_successor_v37",
            "status": "unranked_candidate",
            "official_results_authorised": False,
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "selection": {
                "method": "transport_repaired_before_complete_successor_blocks",
                "model_count": 26,
                "quality_observations_used": 0,
                "route_counts": {"openrouter": 26},
                "panel_order": list(FINAL_PANEL_ORDER),
                "retained_base_model_blocks": list(RETAINED_BASE_MODEL_IDS),
                "retained_cohere_model_blocks": list(RETAINED_COHERE_MODEL_IDS),
                "new_complete_model_blocks": list(NEW_MODEL_IDS),
            },
            "run_design": {
                **dict(document["run_design"]),
                "new_provider_calls": len(NEW_MODEL_IDS) * REQUESTS_PER_NEW_BLOCK,
                "composite_primary_cells": 26 * PRIMARY_TASKS,
                "composite_repeat_cells": 26 * REPEAT_TASKS,
                "response_source_pooling_within_model": False,
            },
            "budget": {
                "currency": "USD",
                "cap_usd": _decimal_text(CAP_USD),
                "bounded_forecast_usd": _decimal_text(bounded),
                "headroom_usd": _decimal_text(CAP_USD - bounded),
                "per_model_worst_case_usd": dict(sorted(forecasts.items())),
                "forecast_scope": "eight new or replacement 704-response blocks only",
                "within_cap": True,
            },
            "route_refresh": {
                "schema_version": "flavourbench-selection-route-refresh-v37",
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_manifest_path),
                },
                "current_v35_manifest": {
                    "semantic_sha256": current["content_address"]["digest"],
                    "physical_sha256": _sha256_file(current_manifest_path),
                },
                "v36_transport_pilot": pilot,
                "repairs": {
                    "deepseek/deepseek-v4-pro-0813": "native route 404; exact GMICloud FP8",
                    "anthropic/claude-fable-5": "Bedrock refusals; exact Anthropic route",
                    "qwen/qwen3.8-2.4t-a95b": "reasoning disabled in plan after one length finish",
                    "bytedance-seed/seed-2-1-turbo": (
                        "reasoning disabled in plan after four length finishes"
                    ),
                    "cohere": "direct monthly cap; retain complete exact OpenRouter Cohere blocks",
                },
                "pilot_scores_or_selections_used": False,
                "pilot_responses_used_as_final_score_data": False,
            },
        }
    )
    return _address(document)


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--v36-plan-sha256", required=True)
    parser.add_argument("--v36-pilot-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_manifest_path=args.source_manifest,
            current_manifest_path=args.current_manifest,
            v36_plan_sha256=args.v36_plan_sha256,
            v36_pilot_directory=args.v36_pilot_directory,
        )
    )
    if not verify_manifest_content_address(document):
        raise FrontierRefresh26V37Error("v37 manifest did not self-verify")
    print(write_manifest(document, args.output_directory))


if __name__ == "__main__":
    run()
