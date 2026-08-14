"""Freeze the 26-model dated frontier successor before score collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v23 import _roster_row
from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v35 import verify_plan as verify_plan_v35
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26 import (
    DIRECT_COHERE_MODEL_IDS,
    NEW_MODEL_IDS,
    PANEL_ORDER,
    RETAINED_MODEL_IDS,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v36"
PLAN_VERSION = "flavourbench-selection-26x640-v36"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PRIMARY_CELLS = MODEL_COUNT * PRIMARY_TASKS
REPEAT_CELLS = MODEL_COUNT * REPEAT_TASKS
NEW_PROVIDER_CALLS = len(NEW_MODEL_IDS) * (PRIMARY_TASKS + REPEAT_TASKS)
PAIR_COUNT = MODEL_COUNT * (MODEL_COUNT - 1) // 2
RUN_CAP_USD = "200"
MAX_OUTPUT_TOKENS_BY_MODEL = {
    "x-ai/grok-4.6": 2_048,
    "deepseek/deepseek-v4-pro-0813": 2_048,
    "cohere/command-a-plus-05-2026": 1_800,
    "cohere/command-a-reasoning-08-2025": 1_800,
    "meta/muse-spark-1.2": 4_096,
    "meta/muse-glimmer-30b": 2_048,
    "anthropic/claude-fable-5": 2_048,
    "qwen/qwen3.8-2.4t-a95b": 4_096,
    "bytedance-seed/seed-2-1-turbo": 2_048,
    "thinkingmachines/inkling": 2_048,
}
REASONING_EFFORT_BY_MODEL = {
    "x-ai/grok-4.6": "minimal",
    "deepseek/deepseek-v4-pro-0813": "minimal",
    "cohere/command-a-plus-05-2026": "provider_fixed",
    "cohere/command-a-reasoning-08-2025": "provider_fixed",
    "meta/muse-spark-1.2": "minimal",
    "meta/muse-glimmer-30b": "minimal",
    "anthropic/claude-fable-5": "minimal",
    "qwen/qwen3.8-2.4t-a95b": "minimal",
    "bytedance-seed/seed-2-1-turbo": "provider_fixed",
    "thinkingmachines/inkling": "minimal",
}


class SelectionPoweredPlanV36Error(RuntimeError):
    """The 26-model successor plan failed verification."""


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


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV36Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV36Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    retained_source_plan: Mapping[str, Any],
    retained_source_plan_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v35(predecessor):
        raise SelectionPoweredPlanV36Error("v36 requires the exact powered-v35 predecessor")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != PANEL_ORDER:
        raise SelectionPoweredPlanV36Error("manifest roster/order differs from v36")
    if retained_source_plan.get("artifact_sha256") != (
        predecessor.get("inputs", {}).get("plan_v31_predecessor", {}).get("semantic_sha256")
    ):
        raise SelectionPoweredPlanV36Error("retained response plan is not the v31 source")

    predecessor_rows = {str(row["model_id"]): dict(row) for row in predecessor["roster"]["models"]}
    retained_source_rows = {
        str(row["model_id"]): dict(row)
        for row in retained_source_plan.get("roster", {}).get("models", [])
    }
    if any(
        predecessor_rows.get(model_id) != retained_source_rows.get(model_id)
        for model_id in RETAINED_MODEL_IDS
    ):
        raise SelectionPoweredPlanV36Error("retained source roster binding drifted")

    roster: list[dict[str, Any]] = []
    for candidate in candidates:
        model_id = candidate.model_id
        if model_id in RETAINED_MODEL_IDS:
            roster.append(dict(predecessor_rows[model_id]))
            continue
        row = _roster_row(candidate, REASONING_EFFORT_BY_MODEL[model_id])
        row["final_max_output_tokens"] = MAX_OUTPUT_TOKENS_BY_MODEL[model_id]
        roster.append(row)

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_26_model_frontier_successor_before_complete_blocks"
    document["inputs"]["plan_v35_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["retained_response_source_plan"] = {
        "semantic_sha256": retained_source_plan["artifact_sha256"],
        "physical_sha256": retained_source_plan_physical_sha256,
        "model_ids": list(RETAINED_MODEL_IDS),
        "responses_reused_without_modification": True,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["roster"] = {
        "model_count": MODEL_COUNT,
        "fallbacks": "disabled",
        "models": roster,
    }
    document["design"].update(
        {
            "primary_provider_calls": PRIMARY_CELLS,
            "repeat_provider_calls": REPEAT_CELLS,
            "total_provider_calls": PRIMARY_CELLS + REPEAT_CELLS,
        }
    )
    document["inference"].update(
        {
            "paired_tests": (
                f"all {PAIR_COUNT} two-sided paired sign-flip permutation tests with Holm "
                "correction"
            ),
            "rank_display": "statistical rank groups; no forced ordering inside unresolved groups",
        }
    )
    document["execution"]["pilot"].update({"cells": MODEL_COUNT * 4})
    document["execution"]["collection_concurrency"] = {
        "global": 32,
        "per_model_default": 4,
        "per_model_by_backend": {"openrouter": 4, "cohere_direct": 1},
        "per_model_by_model_id": {
            "openai/gpt-5.6-sol-pro": 1,
            "openai/gpt-5.6-terra-pro": 1,
            "openai/gpt-5.6-luna-pro": 1,
            "meta-llama/llama-4-maverick": 1,
            "moonshotai/kimi-k3": 1,
            "nvidia/nemotron-3.5-lightning": 1,
            "anthropic/claude-fable-5": 2,
            "cohere/command-a-plus-05-2026": 1,
            "cohere/command-a-reasoning-08-2025": 1,
        },
        "reason": (
            "four lanes on independent exact OpenRouter routes; single-flight retained fragile "
            "routes and direct Cohere; two lanes for Fable on Bedrock"
        ),
    }
    document["execution"]["minimum_request_interval_seconds_by_backend"] = {"cohere_direct": 6.5}
    controlled = sorted(
        {
            *document["execution"].get("reasoning_control_model_ids", []),
            *(
                model_id
                for model_id, effort in REASONING_EFFORT_BY_MODEL.items()
                if effort != "provider_fixed"
            ),
        }
        - {"x-ai/grok-4.5", "deepseek/deepseek-v4-pro"}
    )
    document["execution"]["reasoning_control_model_ids"] = controlled
    document["execution"]["reasoning_control"] = (
        "minimal hidden reasoning on controllable successor routes; provider-fixed on Seed and "
        "the two exact direct Cohere routes; retained blocks keep their original controls"
    )
    document["execution"]["frontier_refresh_successor"] = {
        "release_date": "2026-08-14",
        "retained_model_ids": list(RETAINED_MODEL_IDS),
        "new_model_ids": list(NEW_MODEL_IDS),
        "replaced_predecessor_model_ids": [
            "x-ai/grok-4.5",
            "deepseek/deepseek-v4-pro",
            "cohere/command-a",
            "cohere/command-r-plus-08-2024",
        ],
        "new_primary_cells_per_model": PRIMARY_TASKS,
        "new_repeat_cells_per_model": REPEAT_TASKS,
        "new_provider_calls": NEW_PROVIDER_CALLS,
        "reuse_only_unchanged_exact_route_blocks": True,
        "cross_provider_response_pooling_within_model": False,
        "score_or_result_adaptive_selection": False,
    }
    document["budget"].update(
        {
            "hard_cap": RUN_CAP_USD,
            "program_cap": RUN_CAP_USD,
            "successor_scope": "ten new or replacement complete model blocks",
        }
    )
    document["claim_boundary"] = (
        "Epicure alignment on 640 frozen combinatorial culinary tasks; not universal model quality"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV36Error("constructed v36 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        execution = document["execution"]
        successor = execution["frontier_refresh_successor"]
        manifest = document["inputs"]["route_manifest"]
        predecessor = document["inputs"]["plan_v35_predecessor"]
        retained = document["inputs"]["retained_response_source_plan"]
        policy_document = execution["execution_policy"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(roster) == MODEL_COUNT
        and tuple(row["model_id"] for row in roster) == PANEL_ORDER
        and len(rows) == MODEL_COUNT
        and set(successor.get("retained_model_ids") or []) == set(RETAINED_MODEL_IDS)
        and tuple(successor.get("new_model_ids") or []) == NEW_MODEL_IDS
        and successor.get("new_provider_calls") == NEW_PROVIDER_CALLS
        and successor.get("reuse_only_unchanged_exact_route_blocks") is True
        and successor.get("cross_provider_response_pooling_within_model") is False
        and successor.get("score_or_result_adaptive_selection") is False
        and set(retained.get("model_ids") or []) == set(RETAINED_MODEL_IDS)
        and retained.get("responses_reused_without_modification") is True
        and all(
            rows[model_id].get("final_reasoning_effort") == REASONING_EFFORT_BY_MODEL[model_id]
            and rows[model_id].get("final_max_output_tokens")
            == MAX_OUTPUT_TOKENS_BY_MODEL[model_id]
            for model_id in NEW_MODEL_IDS
        )
        and all(
            rows[model_id].get("execution_backend") == "cohere_direct"
            for model_id in DIRECT_COHERE_MODEL_IDS
        )
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and execution["pilot"].get("cells") == MODEL_COUNT * 4
        and execution["collection_concurrency"].get("global") == 32
        and execution["collection_concurrency"].get("per_model_default") == 4
        and document["design"].get("primary_provider_calls") == PRIMARY_CELLS
        and document["design"].get("repeat_provider_calls") == REPEAT_CELLS
        and document["design"].get("total_provider_calls") == PRIMARY_CELLS + REPEAT_CELLS
        and document["inference"].get("bootstrap_resamples") == 50_000
        and document["inference"].get("permutation_resamples") == 100_000
        and document["budget"].get("hard_cap") == RUN_CAP_USD
        and document["budget"].get("program_cap") == RUN_CAP_USD
        and isinstance(manifest.get("semantic_sha256"), str)
        and len(manifest["semantic_sha256"]) == 64
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(retained.get("semantic_sha256"), str)
        and len(retained["semantic_sha256"]) == 64
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV36Error("content-addressed plan conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
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
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--retained-source-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    retained = _load(args.retained_source_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        retained_source_plan=retained,
        retained_source_plan_physical_sha256=_sha256_file(args.retained_source_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
