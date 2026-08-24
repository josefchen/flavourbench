"""Freeze a full-block DeepSeek rerun on exact CoreWeave FP8."""

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
from .epicure_selection_powered_plan_v31 import verify_plan as verify_v31_plan
from .epicure_selection_route_manifest_v26 import (
    DEEPSEEK_PRO_MODEL_ID,
    EXPECTED_ACTUAL_MODEL_ID,
)
from .epicure_selection_route_manifest_v31 import (
    EXPECTED_PREDECESSOR_RESPONSES,
    PREDECESSOR_PROVIDER,
    PREDECESSOR_TAG,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v29"
PLAN_VERSION = "flavourbench-selection-20x640-v29"
SUCCESSOR_RUN_CAP_USD = "72"
PRIMARY_CELLS = 640
REPEAT_CELLS = 64
TRANSPORT_CHECK_CELLS = 8
TRANSPORT_CHECK_TASK_IDS = (
    "fb-executable-substitution-001",
    "fb-executable-substitution-079",
    "fb-executable-pairing-001",
    "fb-executable-pairing-063",
    "fb-executable-constraint-001",
    "fb-executable-constraint-031",
    "fb-executable-cultural_composition-001",
    "fb-executable-cultural_composition-085",
)


class SelectionPoweredPlanV32Error(RuntimeError):
    """The v32 CoreWeave DeepSeek successor failed verification."""


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
        raise SelectionPoweredPlanV32Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV32Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_v31_plan(predecessor):
        raise SelectionPoweredPlanV32Error("v32 requires the exact v31 predecessor")
    candidates = select_candidates(manifest)
    current_roster = [candidate.model_id for candidate in candidates]
    predecessor_roster = [row["model_id"] for row in predecessor["roster"]["models"]]
    if current_roster != predecessor_roster or len(current_roster) != 20:
        raise SelectionPoweredPlanV32Error("v32 must retain the exact model roster and order")
    refresh = manifest.get("route_refresh") or {}
    failure = refresh.get("failed_route_observation") or {}
    if (
        refresh.get("model_id") != DEEPSEEK_PRO_MODEL_ID
        or refresh.get("prior_route", {}).get("tag") != PREDECESSOR_TAG
        or refresh.get("replacement_route", {}).get("tag") != REPLACEMENT_TAG
        or failure.get("response_count") != EXPECTED_PREDECESSOR_RESPONSES
        or failure.get("completed_count") != 546
        or failure.get("failed_count") != 14
        or failure.get("excluded_from_all_model_score_and_rank_inference") is not True
    ):
        raise SelectionPoweredPlanV32Error("v32 DeepSeek route lineage is incomplete")

    effort = {
        row["model_id"]: row["final_reasoning_effort"] for row in predecessor["roster"]["models"]
    }
    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = (
        "preregistered_after_v31_novita_outage_before_any_coreweave_successor_response"
    )
    document["inputs"]["plan_v31_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["deepseek_novita_failure_observation"] = dict(failure)
    document["roster"]["models"] = [
        _roster_row(candidate, effort[candidate.model_id]) for candidate in candidates
    ]
    document["execution"]["deepseek_route_recovery"] = {
        "model_id": DEEPSEEK_PRO_MODEL_ID,
        "predecessor_provider_tag": PREDECESSOR_TAG,
        "predecessor_provider_name": PREDECESSOR_PROVIDER,
        "successor_provider_tag": REPLACEMENT_TAG,
        "successor_provider_name": REPLACEMENT_PROVIDER,
        "transport_check_cells": TRANSPORT_CHECK_CELLS,
        "transport_check_task_ids": list(TRANSPORT_CHECK_TASK_IDS),
        "successor_primary_cells": PRIMARY_CELLS,
        "successor_repeat_cells": REPEAT_CELLS,
        "rerun_entire_model_block": True,
        "reuse_predecessor_responses": False,
        "cross_provider_score_pooling": False,
        "all_other_model_response_blocks_remain_bound_to_v31": True,
        "prompt_task_scoring_and_inference_change": False,
        "successor_execution_order": [
            "complete_eight_predetermined_transport_check_cells",
            "verify_normal_responses_and_exact_provider_identity_without_inspecting_scores",
            "complete_all_remaining_primary_cells",
            "complete_all_repeat_cells",
        ],
    }
    document["budget"]["superseded_deepseek_primary_spend_micros"] = int(failure["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV32Error("constructed v32 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        rows = {row["model_id"]: row for row in roster}
        execution = document["execution"]
        policy_document = execution["execution_policy"]
        recovery = execution["deepseek_route_recovery"]
        failure = document["inputs"]["deepseek_novita_failure_observation"]
        predecessor = document["inputs"]["plan_v31_predecessor"]
        route = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    deepseek = rows.get(DEEPSEEK_PRO_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and deepseek.get("canonical_model_slug") == EXPECTED_ACTUAL_MODEL_ID
        and deepseek.get("execution_backend") == "openrouter"
        and deepseek.get("provider_tag") == REPLACEMENT_TAG
        and deepseek.get("provider_name") == REPLACEMENT_PROVIDER
        and deepseek.get("final_reasoning_effort") == "minimal"
        and execution["collection_concurrency"]["per_model_by_model_id"].get(DEEPSEEK_PRO_MODEL_ID)
        == 1
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and policy_document["limits"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and recovery
        == {
            "model_id": DEEPSEEK_PRO_MODEL_ID,
            "predecessor_provider_tag": PREDECESSOR_TAG,
            "predecessor_provider_name": PREDECESSOR_PROVIDER,
            "successor_provider_tag": REPLACEMENT_TAG,
            "successor_provider_name": REPLACEMENT_PROVIDER,
            "transport_check_cells": TRANSPORT_CHECK_CELLS,
            "transport_check_task_ids": list(TRANSPORT_CHECK_TASK_IDS),
            "successor_primary_cells": PRIMARY_CELLS,
            "successor_repeat_cells": REPEAT_CELLS,
            "rerun_entire_model_block": True,
            "reuse_predecessor_responses": False,
            "cross_provider_score_pooling": False,
            "all_other_model_response_blocks_remain_bound_to_v31": True,
            "prompt_task_scoring_and_inference_change": False,
            "successor_execution_order": [
                "complete_eight_predetermined_transport_check_cells",
                "verify_normal_responses_and_exact_provider_identity_without_inspecting_scores",
                "complete_all_remaining_primary_cells",
                "complete_all_repeat_cells",
            ],
        }
        and failure.get("model_id") == DEEPSEEK_PRO_MODEL_ID
        and failure.get("response_count") == EXPECTED_PREDECESSOR_RESPONSES
        and failure.get("completed_count") == 546
        and failure.get("failed_count") == 14
        and len(failure.get("failed_task_ids") or []) == 14
        and failure.get("used_as_successor_score_data") is False
        and failure.get("excluded_from_all_model_score_and_rank_inference") is True
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(route.get("semantic_sha256"), str)
        and len(route["semantic_sha256"]) == 64
        and document["design"]["primary_provider_calls"] == 12_800
        and document["inference"]["bootstrap_resamples"] == 50_000
        and document["inference"]["permutation_resamples"] == 100_000
        and document["budget"]["hard_cap"] == SUCCESSOR_RUN_CAP_USD
        and document["budget"]["superseded_deepseek_primary_spend_micros"]
        == failure["spend_micros"]
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV32Error("content-addressed plan conflict")
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
