"""Freeze the clean eight-block 26-model successor after transport repair."""

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
from .epicure_selection_powered_plan_v36 import verify_plan as verify_plan_v36
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import (
    FINAL_PANEL_ORDER,
    MAX_OUTPUT_TOKENS_BY_MODEL,
    NEW_MODEL_IDS,
    RETAINED_BASE_MODEL_IDS,
    RETAINED_COHERE_MODEL_IDS,
)

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v37"
PLAN_VERSION = "flavourbench-selection-26x640-v37"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
NEW_PROVIDER_CALLS = len(NEW_MODEL_IDS) * (PRIMARY_TASKS + REPEAT_TASKS)
PAIR_COUNT = MODEL_COUNT * (MODEL_COUNT - 1) // 2
RUN_CAP_USD = "200"
REASONING_EFFORT_BY_MODEL = {
    "x-ai/grok-4.6": "minimal",
    "deepseek/deepseek-v4-pro-0813": "minimal",
    "meta/muse-spark-1.2": "minimal",
    "meta/muse-glimmer-30b": "minimal",
    "anthropic/claude-fable-5": "minimal",
    "qwen/qwen3.8-2.4t-a95b": "none",
    "bytedance-seed/seed-2-1-turbo": "none",
    "thinkingmachines/inkling": "minimal",
}


class SelectionPoweredPlanV37Error(RuntimeError):
    """The clean eight-block successor plan failed verification."""


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
        raise SelectionPoweredPlanV37Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV37Error("input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    base_plan: Mapping[str, Any],
    base_plan_physical_sha256: str,
    cohere_plan: Mapping[str, Any],
    cohere_plan_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v36(predecessor):
        raise SelectionPoweredPlanV37Error("v37 requires the exact v36 predecessor")
    if not verify_plan_v35(cohere_plan):
        raise SelectionPoweredPlanV37Error("v37 Cohere source is not exact powered-v35")
    expected_base_sha = predecessor["inputs"]["retained_response_source_plan"]["semantic_sha256"]
    if base_plan.get("artifact_sha256") != expected_base_sha:
        raise SelectionPoweredPlanV37Error("v37 base response plan is not exact v31")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV37Error("v37 manifest roster/order changed")
    base_rows = {str(row["model_id"]): dict(row) for row in base_plan["roster"]["models"]}
    cohere_rows = {str(row["model_id"]): dict(row) for row in cohere_plan["roster"]["models"]}
    roster: list[dict[str, Any]] = []
    for candidate in candidates:
        model_id = candidate.model_id
        if model_id in RETAINED_BASE_MODEL_IDS:
            roster.append(base_rows[model_id])
        elif model_id in RETAINED_COHERE_MODEL_IDS:
            roster.append(cohere_rows[model_id])
        else:
            row = _roster_row(candidate, REASONING_EFFORT_BY_MODEL[model_id])
            row["final_max_output_tokens"] = MAX_OUTPUT_TOKENS_BY_MODEL[model_id]
            roster.append(row)

    document = json.loads(json.dumps(cohere_plan))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "preregistered_after_v36_transport_repair_before_clean_blocks"
    document["inputs"]["plan_v36_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
        "pilot_responses_used_as_score_data": False,
    }
    document["inputs"]["retained_base_response_source_plan"] = {
        "semantic_sha256": base_plan["artifact_sha256"],
        "physical_sha256": base_plan_physical_sha256,
        "model_ids": list(RETAINED_BASE_MODEL_IDS),
    }
    document["inputs"]["retained_cohere_response_source_plan"] = {
        "semantic_sha256": cohere_plan["artifact_sha256"],
        "physical_sha256": cohere_plan_physical_sha256,
        "model_ids": list(RETAINED_COHERE_MODEL_IDS),
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
            "primary_provider_calls": MODEL_COUNT * PRIMARY_TASKS,
            "repeat_provider_calls": MODEL_COUNT * REPEAT_TASKS,
            "total_provider_calls": MODEL_COUNT * (PRIMARY_TASKS + REPEAT_TASKS),
        }
    )
    document["inference"]["paired_tests"] = (
        f"all {PAIR_COUNT} two-sided paired sign-flip permutation tests with Holm correction"
    )
    document["execution"]["pilot"]["cells"] = MODEL_COUNT * 4
    document["execution"]["collection_concurrency"] = {
        "global": 32,
        "per_model_default": 4,
        "per_model_by_backend": {"openrouter": 4},
        "per_model_by_model_id": {
            "openai/gpt-5.6-sol-pro": 1,
            "openai/gpt-5.6-terra-pro": 1,
            "openai/gpt-5.6-luna-pro": 1,
            "meta-llama/llama-4-maverick": 1,
            "moonshotai/kimi-k3": 1,
            "nvidia/nemotron-3.5-lightning": 1,
            "anthropic/claude-fable-5": 2,
        },
        "reason": "exact route-specific transport limits; all eight successor blocks are clean",
    }
    document["execution"]["minimum_request_interval_seconds_by_backend"] = {}
    document["execution"]["frontier_refresh_successor"] = {
        "release_date": "2026-08-14",
        "retained_base_model_ids": list(RETAINED_BASE_MODEL_IDS),
        "retained_cohere_model_ids": list(RETAINED_COHERE_MODEL_IDS),
        "new_model_ids": list(NEW_MODEL_IDS),
        "new_primary_cells_per_model": PRIMARY_TASKS,
        "new_repeat_cells_per_model": REPEAT_TASKS,
        "new_provider_calls": NEW_PROVIDER_CALLS,
        "reuse_only_unchanged_exact_route_blocks": True,
        "cross_provider_response_pooling_within_model": False,
        "score_or_result_adaptive_selection": False,
        "v36_pilot_responses_used_as_score_data": False,
    }
    document["execution"]["reasoning_control_model_ids"] = sorted(
        {
            *document["execution"].get("reasoning_control_model_ids", []),
            *NEW_MODEL_IDS,
        }
        - {
            "x-ai/grok-4.5",
            "deepseek/deepseek-v4-pro",
            "cohere/command-a",
            "cohere/command-r-plus-08-2024",
        }
    )
    document["execution"]["reasoning_control"] = (
        "reasoning disabled for Seed 2.1 Turbo and Qwen3.8 2.4T after transport-only length "
        "findings; minimal on the other controllable successor routes"
    )
    document["budget"].update(
        {
            "hard_cap": RUN_CAP_USD,
            "program_cap": RUN_CAP_USD,
            "successor_scope": "eight complete new or replacement blocks",
        }
    )
    document["claim_boundary"] = (
        "Epicure alignment on 640 frozen combinatorial culinary tasks; not universal model quality"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV37Error("constructed v37 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = list(document["roster"]["models"])
        rows = {str(row["model_id"]): row for row in roster}
        execution = document["execution"]
        successor = execution["frontier_refresh_successor"]
        policy_document = execution["execution_policy"]
        base = document["inputs"]["retained_base_response_source_plan"]
        cohere = document["inputs"]["retained_cohere_response_source_plan"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == MODEL_COUNT
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and len(rows) == MODEL_COUNT
        and set(successor.get("retained_base_model_ids") or []) == set(RETAINED_BASE_MODEL_IDS)
        and set(successor.get("retained_cohere_model_ids") or []) == set(RETAINED_COHERE_MODEL_IDS)
        and tuple(successor.get("new_model_ids") or []) == NEW_MODEL_IDS
        and successor.get("new_provider_calls") == NEW_PROVIDER_CALLS
        and successor.get("reuse_only_unchanged_exact_route_blocks") is True
        and successor.get("cross_provider_response_pooling_within_model") is False
        and successor.get("score_or_result_adaptive_selection") is False
        and successor.get("v36_pilot_responses_used_as_score_data") is False
        and set(base.get("model_ids") or []) == set(RETAINED_BASE_MODEL_IDS)
        and set(cohere.get("model_ids") or []) == set(RETAINED_COHERE_MODEL_IDS)
        and all(
            rows[model_id].get("final_reasoning_effort") == REASONING_EFFORT_BY_MODEL[model_id]
            and rows[model_id].get("final_max_output_tokens")
            == MAX_OUTPUT_TOKENS_BY_MODEL[model_id]
            for model_id in NEW_MODEL_IDS
        )
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and execution["pilot"].get("cells") == MODEL_COUNT * 4
        and document["design"].get("primary_provider_calls") == MODEL_COUNT * PRIMARY_TASKS
        and document["design"].get("repeat_provider_calls") == MODEL_COUNT * REPEAT_TASKS
        and document["budget"].get("hard_cap") == RUN_CAP_USD
        and document["budget"].get("program_cap") == RUN_CAP_USD
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV37Error("content-addressed plan conflict")
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
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--cohere-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    base = _load(args.base_plan)
    cohere = _load(args.cohere_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        base_plan=base,
        base_plan_physical_sha256=_sha256_file(args.base_plan),
        cohere_plan=cohere,
        cohere_plan_physical_sha256=_sha256_file(args.cohere_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
