"""Freeze the 20-model Epicure-native automated FlavourBench refresh."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .execution_policy import PORTABLE_TEXT_TOOL_PROTOCOL_V1, ExecutionPolicy
from .frontier_contract_runner import select_candidates
from .frontier_manifest import (
    ForecastPolicy,
    PanelSlot,
    discover_candidate_manifest,
    verify_manifest_content_address,
)
from .real_dataset_runner import (
    build_balanced_work_items,
    derive_pair_forecast,
    load_epicure_native_task_inventory,
    select_balanced_tasks,
)

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
MANIFEST_ROLE = "epicure_native_automated_frontier_refresh_v1"
SELECTION_SEED = "flavourbench-epicure-native-frontier-refresh-20-v1"
TASKS_PER_FAMILY = 8
ASSIGNMENTS_PER_MODEL = 32
CAP_USD = Decimal("100")
ADMISSION_FRACTION = Decimal("0.85")

OPENROUTER_PANEL = (
    PanelSlot("frontier-refresh-01", "openai", "openai/gpt-5.6-sol-pro", "Sol quality tier."),
    PanelSlot(
        "frontier-refresh-02", "openai", "openai/gpt-5.6-terra-pro", "Terra quality-cost tier."
    ),
    PanelSlot("frontier-refresh-03", "openai", "openai/gpt-5.6-luna-pro", "Luna efficiency tier."),
    PanelSlot("frontier-refresh-04", "anthropic", "anthropic/claude-fable-5", "Claude Fable 5."),
    PanelSlot("frontier-refresh-05", "anthropic", "anthropic/claude-opus-5", "Claude Opus 5."),
    PanelSlot("frontier-refresh-06", "anthropic", "anthropic/claude-sonnet-5", "Claude Sonnet 5."),
    PanelSlot(
        "frontier-refresh-07",
        "google",
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro.",
    ),
    PanelSlot("frontier-refresh-08", "google", "google/gemini-3.6-flash", "Gemini 3.6 Flash."),
    PanelSlot("frontier-refresh-09", "xai", "x-ai/grok-4.5", "Grok 4.5."),
    PanelSlot(
        "frontier-refresh-10",
        "kimi",
        "moonshotai/kimi-k3",
        "Kimi K3 through an exact preselected OpenRouter route.",
    ),
    PanelSlot(
        "frontier-refresh-11",
        "qwen",
        "qwen/qwen3.8-max",
        "Qwen3.8 Max through the exact OpenRouter pay-as-you-go route.",
    ),
    PanelSlot("frontier-refresh-12", "zai", "z-ai/glm-5.2", "GLM 5.2."),
    PanelSlot(
        "frontier-refresh-13",
        "deepseek",
        "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro.",
    ),
    PanelSlot(
        "frontier-refresh-14",
        "deepseek",
        "deepseek/deepseek-v4-flash-0731",
        "DeepSeek V4 Flash.",
    ),
    PanelSlot("frontier-refresh-15", "minimax", "minimax/minimax-m3", "MiniMax M3."),
    PanelSlot(
        "frontier-refresh-16",
        "nvidia",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "Nemotron 3 Ultra.",
    ),
    PanelSlot(
        "frontier-refresh-17",
        "mistral",
        "mistralai/mistral-large-2512",
        "Mistral Large 3 2512.",
    ),
    PanelSlot("frontier-refresh-18", "tencent", "tencent/hy3", "Tencent Hy3."),
)

EXECUTION_POLICY = ExecutionPolicy(
    max_output_tokens=2_048,
    max_intermediate_tokens=1_024,
    max_tool_rounds=1,
    max_tool_result_bytes=4_096,
    max_cumulative_tool_result_bytes=4_096,
    max_tool_calls_per_round=1,
    max_tool_calls_total=1,
    max_provider_attempts=2,
    decoding_temperature=0.0,
    decoding_top_p=1.0,
    decoding_seed=20_260_810,
    final_response_mode="plain_text",
    matched_planning=False,
    evidence_protocol=PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    intermediate_reasoning_effort=None,
    final_reasoning_effort=None,
    tool_catalog_bytes_bound=16_000,
    epicure_on_tool_required=True,
)

KIMI_OPENROUTER_ROUTE = "wafer"
KIMI_OPENROUTER_ROUTE_EXCLUSIONS = frozenset(
    {
        "baseten/fp8",
        "chutes/mxfp4",
        "deepinfra/bf16",
        "digitalocean",
        "fireworks",
        "fireworks/fast",
        "modal/mxfp4",
        "moonshotai/mxfp4",
        "morph",
        "morph/fast",
        "together",
    }
)
GLM_OPENROUTER_ROUTE = "coreweave/fp4"
GLM_OPENROUTER_ROUTE_EXCLUSIONS = frozenset(
    {
        "baidu/fp8",
        "decart/fp4",
        "deepinfra/fp4",
        "inceptron/fp4",
        "novita/fp8",
    }
)
DEEPSEEK_FLASH_OPENROUTER_ROUTE = "cloudflare/fp8"
DEEPSEEK_FLASH_OPENROUTER_ROUTE_EXCLUSIONS = frozenset(
    {
        "akashml/fp8",
        "baidu/fp8",
        "decart/fp4",
        "deepinfra/fp4",
        "digitalocean",
        "fireworks",
        "morph",
    }
)


class FrontierRefreshError(RuntimeError):
    """The 20-model refresh could not be frozen safely."""


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


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrontierRefreshError(f"manifest must be a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierRefreshError(f"invalid manifest JSON: {path}") from error
    if not isinstance(document, dict) or not verify_manifest_content_address(document):
        raise FrontierRefreshError(f"manifest content address does not verify: {path}")
    return document


def _openrouter_entries(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = copy.deepcopy(discovery.get("models"))
    if not isinstance(entries, list) or len(entries) != len(OPENROUTER_PANEL):
        raise FrontierRefreshError("OpenRouter discovery did not resolve the exact 18-model panel")
    for entry in entries:
        entry["execution_route"] = {
            "policy": "exact_openrouter_endpoint_v1",
            "preferred_backend": "openrouter",
            "selected_backend": "openrouter",
            "selection_frozen_before_generation": True,
            "selection_reason": "live_capability_and_price_contract",
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
        }
        entry["backend_contract"] = {}
        entry["backend_contract_sha256"] = "unfrozen"
        entry["cost_accounting_policy"] = "provider_generation_metadata"
        entry["contract_evidence"] = {
            "status": "live_read_only_endpoint_contract_selected",
            "generation_calls": 0,
            "quality_observations": 0,
        }
    return entries


def _direct_entries(
    routed_manifest: Mapping[str, Any], cohere_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    routed_models = routed_manifest.get("models")
    if not isinstance(routed_models, list):
        raise FrontierRefreshError("the routed source manifest has no model entries")
    kimi_matches = [
        copy.deepcopy(entry)
        for entry in routed_models
        if isinstance(entry, Mapping)
        and str((entry.get("model") or {}).get("id") or "") == "moonshotai/kimi-k3"
    ]
    if len(kimi_matches) != 1:
        raise FrontierRefreshError("expected exactly one direct Kimi entry")
    kimi_entry = kimi_matches[0]
    if (kimi_entry.get("execution_route") or {}).get("selected_backend") != "kimi_direct" or (
        kimi_entry.get("endpoint") or {}
    ).get("tag") != "kimi-code-direct":
        raise FrontierRefreshError("the routed Kimi source is not the direct K3 contract")
    kimi_contract = {
        **dict(kimi_entry["backend_contract"]),
        "schema_version": "flavourbench-kimi-direct-anthropic-contract-v2",
        "base_url": "https://api.kimi.com/coding",
        "transport": "anthropic_messages_v1",
        "reasoning_parameter_translation": "provider_default_for_matched_run",
        "portable_text_protocol_required": True,
    }
    kimi_entry["backend_contract"] = kimi_contract
    kimi_entry["backend_contract_sha256"] = _sha256(kimi_contract)
    kimi_entry["execution_route"] = {
        **dict(kimi_entry["execution_route"]),
        "policy": "direct_kimi_anthropic_messages_v1",
        "selection_reason": "exact_direct_kimi_k3_anthropic_contract_passed",
    }
    kimi_entry["slot"] = {
        **dict(kimi_entry["slot"]),
        "slot_id": "frontier-refresh-10",
        "cohort": "kimi",
    }

    cohere = cohere_manifest.get("models")
    if not isinstance(cohere, list):
        raise FrontierRefreshError("the direct Cohere source manifest has no model entries")
    cohere_entries = [copy.deepcopy(entry) for entry in cohere if isinstance(entry, Mapping)]
    if len(cohere_entries) != 2:
        raise FrontierRefreshError("expected exactly two Cohere direct entries")
    expected_cohere = {
        "cohere/command-a-plus-05-2026": "frontier-refresh-19",
        "cohere/command-a-reasoning-08-2025": "frontier-refresh-20",
    }
    for entry in cohere_entries:
        model_id = str(entry["model"]["id"])
        if model_id not in expected_cohere:
            raise FrontierRefreshError(f"unexpected Cohere model: {model_id}")
        entry["slot"] = {
            **dict(entry["slot"]),
            "slot_id": expected_cohere[model_id],
            "cohort": "cohere",
        }
        contract = dict(entry["backend_contract"])
        if model_id == "cohere/command-a-reasoning-08-2025":
            contract.update(
                {
                    "portable_tool_selection_reasoning": (
                        "thinking_disabled_for_exact_json_selection"
                    ),
                    "portable_final_reasoning": "thinking_disabled_for_exact_choice",
                }
            )
        else:
            contract.update(
                {
                    "portable_tool_selection_format": "cohere_json_schema_tool_selection_v1",
                    "portable_final_format": "cohere_json_schema_choice_v1",
                    "portable_phase_reasoning": "thinking_enabled_512_for_schema",
                }
            )
        entry["backend_contract"] = contract
        entry["backend_contract_sha256"] = _sha256(contract)
    return [kimi_entry, *cohere_entries]


def _base_payload(
    *,
    entries: list[dict[str, Any]],
    discovery: Mapping[str, Any],
    task_source: Mapping[str, Any],
    routed_path: Path,
    routed_manifest: Mapping[str, Any],
    cohere_path: Path,
    cohere_manifest: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unranked_candidate",
        "manifest_role": MANIFEST_ROLE,
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": observed_at,
        "source": {
            "openrouter_discovery_manifest_sha256": discovery["content_address"]["digest"],
            "routed_manifest_sha256": routed_manifest["content_address"]["digest"],
            "routed_manifest_physical_sha256": _sha256_file(routed_path),
            "cohere_manifest_sha256": cohere_manifest["content_address"]["digest"],
            "cohere_manifest_physical_sha256": _sha256_file(cohere_path),
            "epicure_native_task_artifact_sha256": task_source["artifact_sha256"],
            "epicure_native_task_set_sha256": task_source["task_set_sha256"],
            "epicure_provenance": task_source["epicure_provenance"],
        },
        "selection": {
            "method": "fixed_frontier_coverage_panel_before_generation",
            "model_count": 20,
            "quality_observations_used": 0,
            "route_counts": {"openrouter": 17, "kimi_direct": 1, "cohere_direct": 2},
            "panel_order": [entry["model"]["id"] for entry in entries],
        },
        "run_design": {
            "track": "epicure_native_exact",
            "tasks_per_family_in_pool": TASKS_PER_FAMILY,
            "selection_seed": SELECTION_SEED,
            "selected_task_count": 32,
            "assignments_per_model": ASSIGNMENTS_PER_MODEL,
            "conditions": ["epicure_off", "epicure_on"],
            "expected_pairs": 640,
            "expected_arms": 1_280,
            "primary_metric": "macro_exact_choice_accuracy",
            "execution_policy": EXECUTION_POLICY.document(),
            "execution_policy_sha256": EXECUTION_POLICY.sha256,
        },
        "models": entries,
        "routing_policy": {
            "resolved_before_generation": True,
            "generation_time_automatic_fallback": False,
            "provider_substitution": "prohibited",
            "qwen_route": "openrouter_pay_as_you_go_only",
            "kimi_route": "direct_kimi_k3_only",
            "glm_route": "openrouter_exact_route_only",
        },
        "budget": {
            "currency": "USD",
            "cap_usd": _decimal_text(CAP_USD),
            "admission_fraction": _decimal_text(ADMISSION_FRACTION),
            "admission_ceiling_usd": _decimal_text(CAP_USD * ADMISSION_FRACTION),
            "bounded_forecast_usd": "0",
            "within_cap": True,
        },
        "governance": {
            "manifest_class": "automated_epicure_native_refresh_candidate",
            "scope": "automated_exact-answer leaderboard; no human ratings",
        },
    }


def _address(payload: Mapping[str, Any]) -> dict[str, Any]:
    document = copy.deepcopy(dict(payload))
    document.pop("content_address", None)
    digest = _sha256(document)
    document["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return document


def _apply_exact_forecast(
    manifest: dict[str, Any], task_artifact: Path
) -> tuple[dict[str, Any], Decimal]:
    task_inventory, _ = load_epicure_native_task_inventory(task_artifact)
    selected_tasks, registry_sha = select_balanced_tasks(
        tasks_per_family=TASKS_PER_FAMILY,
        seed=SELECTION_SEED,
        tasks=task_inventory,
    )
    candidates = select_candidates(manifest)
    work_items = build_balanced_work_items(
        manifest_sha256=manifest["content_address"]["digest"],
        task_registry_digest=registry_sha,
        selected_tasks=selected_tasks,
        candidates=candidates,
        execution_policy=EXECUTION_POLICY,
        assignments_per_model=ASSIGNMENTS_PER_MODEL,
    )
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for work_item in work_items:
        totals[work_item.candidate.model_id] += derive_pair_forecast(
            work_item, policy=EXECUTION_POLICY
        ).forecast_usd
    total = sum(totals.values(), Decimal())
    ceiling = CAP_USD * ADMISSION_FRACTION
    if total > ceiling:
        raise FrontierRefreshError(
            f"exact runner forecast ${_decimal_text(total)} exceeds admission ceiling "
            f"${_decimal_text(ceiling)}"
        )
    updated = copy.deepcopy(manifest)
    updated.pop("content_address", None)
    for entry in updated["models"]:
        model_id = str(entry["model"]["id"])
        entry["forecast"] = {
            "pairs": ASSIGNMENTS_PER_MODEL,
            "conditions_per_pair": 2,
            "model_block_worst_case_usd": _decimal_text(totals[model_id]),
        }
    updated["budget"].update(
        {
            "bounded_forecast_usd": _decimal_text(total),
            "headroom_to_admission_ceiling_usd": _decimal_text(ceiling - total),
            "per_model_worst_case_usd": {
                model_id: _decimal_text(value) for model_id, value in totals.items()
            },
        }
    )
    updated["run_design"]["task_registry_sha256"] = registry_sha
    return _address(updated), total


def write_manifest(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_manifest_content_address(document):
        raise FrontierRefreshError("refusing to write an invalid refresh manifest")
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = output_directory / f"flavourbench-frontier-refresh-20-{digest}.json"
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise FrontierRefreshError("content-addressed manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.link(temporary, destination)
    destination.chmod(0o644)
    temporary.unlink()
    return destination


async def build(
    *,
    api_key: str,
    task_artifact: Path,
    routed_manifest_path: Path,
    cohere_manifest_path: Path,
) -> tuple[dict[str, Any], Decimal]:
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    discovery = await discover_candidate_manifest(
        api_key=api_key,
        cap_usd=CAP_USD,
        forecast_policy=ForecastPolicy(
            arms_per_model=1,
            max_generations_per_arm=1,
            max_prompt_tokens_per_generation=1_000,
            max_completion_tokens_per_generation=2_048,
            max_reasoning_tokens_per_generation=2_048,
        ),
        panel=OPENROUTER_PANEL,
        requested_names=(),
        endpoint_exclusions={
            "moonshotai/kimi-k3": KIMI_OPENROUTER_ROUTE_EXCLUSIONS,
            "z-ai/glm-5.2": GLM_OPENROUTER_ROUTE_EXCLUSIONS,
            "deepseek/deepseek-v4-flash-0731": (DEEPSEEK_FLASH_OPENROUTER_ROUTE_EXCLUSIONS),
        },
        observed_at=observed_at,
    )
    routed = _read_manifest(routed_manifest_path)
    cohere = _read_manifest(cohere_manifest_path)
    _, task_source = load_epicure_native_task_inventory(task_artifact)
    openrouter = _openrouter_entries(discovery)
    openrouter = [entry for entry in openrouter if entry["model"]["id"] != "moonshotai/kimi-k3"]
    required_routes = {
        "z-ai/glm-5.2": GLM_OPENROUTER_ROUTE,
        "deepseek/deepseek-v4-flash-0731": DEEPSEEK_FLASH_OPENROUTER_ROUTE,
    }
    resolved_routes = {entry["model"]["id"]: entry["endpoint"]["tag"] for entry in openrouter}
    for model_id, required_route in required_routes.items():
        if resolved_routes.get(model_id) != required_route:
            raise FrontierRefreshError(
                f"{model_id} did not resolve to required route {required_route}"
            )
    direct = _direct_entries(routed, cohere)
    by_slot = {entry["slot"]["slot_id"]: entry for entry in [*openrouter, *direct]}
    expected_slots = {f"frontier-refresh-{index:02d}" for index in range(1, 21)}
    if set(by_slot) != expected_slots:
        raise FrontierRefreshError("the combined panel does not contain exact slots 01-20")
    entries = [by_slot[f"frontier-refresh-{index:02d}"] for index in range(1, 21)]
    preliminary = _address(
        _base_payload(
            entries=entries,
            discovery=discovery,
            task_source=task_source,
            routed_path=routed_manifest_path,
            routed_manifest=routed,
            cohere_path=cohere_manifest_path,
            cohere_manifest=cohere,
            observed_at=observed_at,
        )
    )
    final, total = _apply_exact_forecast(preliminary, task_artifact)
    check, check_total = _apply_exact_forecast(final, task_artifact)
    if check_total != total or check["budget"] != final["budget"]:
        raise FrontierRefreshError("exact forecast is not stable after final content addressing")
    return final, total


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-artifact", type=Path, required=True)
    parser.add_argument("--routed-manifest", type=Path, required=True)
    parser.add_argument("--cohere-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for read-only endpoint discovery")
    document, total = asyncio.run(
        build(
            api_key=api_key,
            task_artifact=args.task_artifact,
            routed_manifest_path=args.routed_manifest,
            cohere_manifest_path=args.cohere_manifest,
        )
    )
    path = write_manifest(document, args.output_directory)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_sha256": document["content_address"]["digest"],
                "model_count": len(document["models"]),
                "pair_count": document["run_design"]["expected_pairs"],
                "arm_count": document["run_design"]["expected_arms"],
                "bounded_forecast_usd": _decimal_text(total),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
