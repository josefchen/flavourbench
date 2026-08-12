"""Freeze bounded transport checks for clean OpenRouter Cohere successors."""

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
from .epicure_selection_powered_plan_v31 import (
    MAX_OUTPUT_TOKENS,
    selection_execution_policy_v31,
)
from .epicure_selection_powered_plan_v33 import verify_plan as verify_v33_plan
from .epicure_selection_route_manifest_v32 import (
    DIRECT_BACKEND,
    DIRECT_PROVIDER_TAG,
    OPENROUTER_PROVIDER_NAME,
    OPENROUTER_PROVIDER_TAG,
    REPLACEMENTS,
    ROUTE_MAX_OUTPUT_TOKENS,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v31"
PLAN_VERSION = "flavourbench-selection-20x640-v31"
SUCCESSOR_RUN_CAP_USD = "100"
PRIMARY_CELLS = 640
REPEAT_CELLS = 64
TRANSPORT_TASK_IDS = (
    "fb-executable-substitution-001",
    "fb-executable-substitution-079",
    "fb-executable-pairing-001",
    "fb-executable-pairing-063",
    "fb-executable-constraint-001",
    "fb-executable-constraint-031",
    "fb-executable-cultural_composition-001",
    "fb-executable-cultural_composition-085",
)
TRANSPORT_CELLS_PER_MODEL = len(TRANSPORT_TASK_IDS)
SUCCESSOR_MODEL_IDS = tuple(str(spec["model_id"]) for spec in REPLACEMENTS.values())
PREDECESSOR_MODEL_IDS = tuple(REPLACEMENTS)


class SelectionPoweredPlanV34Error(RuntimeError):
    """The v34 Cohere transport plan failed verification."""


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
        raise SelectionPoweredPlanV34Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV34Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_v33_plan(predecessor):
        raise SelectionPoweredPlanV34Error("v34 requires the exact v33 predecessor")
    candidates = select_candidates(manifest)
    if len(candidates) != 20 or len({candidate.model_id for candidate in candidates}) != 20:
        raise SelectionPoweredPlanV34Error("v34 requires exactly 20 unique routes")
    predecessor_rows = {row["slot_id"]: row for row in predecessor["roster"]["models"]}
    if [candidate.slot_id for candidate in candidates] != list(predecessor_rows):
        raise SelectionPoweredPlanV34Error("v34 slot order changed")
    final_ids = [candidate.model_id for candidate in candidates]
    for old_id in PREDECESSOR_MODEL_IDS:
        if old_id in final_ids:
            raise SelectionPoweredPlanV34Error("direct Cohere model remains in successor roster")
    if not set(SUCCESSOR_MODEL_IDS) <= set(final_ids):
        raise SelectionPoweredPlanV34Error("OpenRouter Cohere successors are incomplete")
    refresh = manifest.get("route_refresh") or {}
    failure_routes = refresh.get("direct_quota_failure_observations") or {}
    if (
        refresh.get("direct_responses_excluded") is not True
        or refresh.get("cross_provider_score_pooling") is not False
        or set(failure_routes) != set(PREDECESSOR_MODEL_IDS)
    ):
        raise SelectionPoweredPlanV34Error("direct Cohere exclusion lineage is incomplete")
    failures = {
        model_id: dict(failure_routes[model_id]["response_commitment"])
        for model_id in PREDECESSOR_MODEL_IDS
    }

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = (
        "preregistered_after_direct_cohere_quota_failure_before_openrouter_successor_calls"
    )
    document["inputs"]["plan_v33_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["direct_cohere_quota_failures"] = failures
    roster = []
    for candidate in candidates:
        prior = predecessor_rows[candidate.slot_id]
        effort = (
            "provider_fixed"
            if candidate.model_id in SUCCESSOR_MODEL_IDS
            else str(prior["final_reasoning_effort"])
        )
        row = _roster_row(candidate, effort)
        if candidate.model_id in SUCCESSOR_MODEL_IDS:
            row["final_max_output_tokens"] = ROUTE_MAX_OUTPUT_TOKENS
        roster.append(row)
    document["roster"]["models"] = roster
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    for old_id in PREDECESSOR_MODEL_IDS:
        overrides.pop(old_id, None)
    for model_id in SUCCESSOR_MODEL_IDS:
        overrides[model_id] = 1
    document["execution"]["cohere_route_successor"] = {
        "predecessor_model_ids": list(PREDECESSOR_MODEL_IDS),
        "successor_model_ids": list(SUCCESSOR_MODEL_IDS),
        "predecessor_provider_tag": DIRECT_PROVIDER_TAG,
        "predecessor_execution_backend": DIRECT_BACKEND,
        "successor_provider_tag": OPENROUTER_PROVIDER_TAG,
        "successor_provider_name": OPENROUTER_PROVIDER_NAME,
        "successor_execution_backend": "openrouter",
        "route_max_output_tokens": ROUTE_MAX_OUTPUT_TOKENS,
        "transport_check_cells_per_model": TRANSPORT_CELLS_PER_MODEL,
        "transport_check_task_ids": list(TRANSPORT_TASK_IDS),
        "successor_primary_cells_per_model": PRIMARY_CELLS,
        "successor_repeat_cells_per_model": REPEAT_CELLS,
        "rerun_entire_model_blocks": True,
        "reuse_direct_responses": False,
        "cross_provider_score_pooling": False,
        "all_other_model_response_blocks_remain_bound_to_existing_sources": True,
        "prompt_task_scoring_and_inference_change": False,
        "successor_execution_order": [
            "complete_predetermined_transport_checks_for_each_successor",
            "verify_normal_responses_and_exact_provider_identity_without_inspecting_scores",
            "freeze_full_block_successor_plan",
            "rerun_each_full_primary_and_repeat_block",
        ],
    }
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["budget"]["program_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV34Error("constructed v34 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        execution = document["execution"]
        recovery = execution["cohere_route_successor"]
        failures = document["inputs"]["direct_cohere_quota_failures"]
        predecessor = document["inputs"]["plan_v33_predecessor"]
        route = document["inputs"]["route_manifest"]
        policy_document = execution["execution_policy"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and not set(PREDECESSOR_MODEL_IDS) & set(rows)
        and set(SUCCESSOR_MODEL_IDS) <= set(rows)
        and all(
            rows[model_id].get("provider_tag") == OPENROUTER_PROVIDER_TAG
            and rows[model_id].get("provider_name") == OPENROUTER_PROVIDER_NAME
            and rows[model_id].get("execution_backend") == "openrouter"
            and rows[model_id].get("final_reasoning_effort") == "provider_fixed"
            and rows[model_id].get("final_max_output_tokens") == ROUTE_MAX_OUTPUT_TOKENS
            for model_id in SUCCESSOR_MODEL_IDS
        )
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and policy_document["limits"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and recovery["predecessor_model_ids"] == list(PREDECESSOR_MODEL_IDS)
        and recovery["successor_model_ids"] == list(SUCCESSOR_MODEL_IDS)
        and recovery["transport_check_cells_per_model"] == TRANSPORT_CELLS_PER_MODEL
        and recovery["transport_check_task_ids"] == list(TRANSPORT_TASK_IDS)
        and recovery["route_max_output_tokens"] == ROUTE_MAX_OUTPUT_TOKENS
        and recovery["rerun_entire_model_blocks"] is True
        and recovery["reuse_direct_responses"] is False
        and recovery["cross_provider_score_pooling"] is False
        and set(failures) == set(PREDECESSOR_MODEL_IDS)
        and all(
            failures[model_id].get("excluded_from_all_model_score_and_rank_inference") is True
            and failures[model_id].get("used_as_successor_score_data") is False
            for model_id in PREDECESSOR_MODEL_IDS
        )
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(route.get("semantic_sha256"), str)
        and len(route["semantic_sha256"]) == 64
        and document["design"]["primary_provider_calls"] == 12_800
        and document["inference"]["bootstrap_resamples"] == 50_000
        and document["inference"]["permutation_resamples"] == 100_000
        and document["budget"]["hard_cap"] == SUCCESSOR_RUN_CAP_USD
        and document["budget"]["program_cap"] == SUCCESSOR_RUN_CAP_USD
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV34Error("content-addressed plan conflict")
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
