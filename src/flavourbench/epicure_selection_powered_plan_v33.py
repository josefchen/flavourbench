"""Freeze the full DeepSeek CoreWeave block after an eight-cell route check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan import run_commitment
from .epicure_selection_powered_plan_v31 import (
    MAX_OUTPUT_TOKENS,
    selection_execution_policy_v31,
)
from .epicure_selection_powered_plan_v32 import (
    PRIMARY_CELLS,
    REPEAT_CELLS,
    TRANSPORT_CHECK_CELLS,
    TRANSPORT_CHECK_TASK_IDS,
)
from .epicure_selection_powered_plan_v32 import verify_plan as verify_v32_plan
from .epicure_selection_route_manifest_v26 import DEEPSEEK_PRO_MODEL_ID
from .epicure_selection_route_manifest_v31 import (
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v30"
PLAN_VERSION = "flavourbench-selection-20x640-v30"
SUCCESSOR_RUN_CAP_USD = "72"
DEEPSEEK_CONCURRENCY = 4


class SelectionPoweredPlanV33Error(RuntimeError):
    """The v33 full-block DeepSeek plan failed verification."""


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
        raise SelectionPoweredPlanV33Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV33Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    calibration_v32: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_v32_plan(predecessor):
        raise SelectionPoweredPlanV33Error("v33 requires the exact v32 predecessor")
    candidates = select_candidates(manifest)
    if [candidate.model_id for candidate in candidates] != [
        row["model_id"] for row in predecessor["roster"]["models"]
    ]:
        raise SelectionPoweredPlanV33Error("v33 model roster or order changed")
    predecessor_route = predecessor["inputs"]["route_manifest"]
    if (
        manifest["content_address"]["digest"] != predecessor_route["semantic_sha256"]
        or manifest_physical_sha256 != predecessor_route["physical_sha256"]
    ):
        raise SelectionPoweredPlanV33Error("v33 must retain the exact v32 route manifest")
    if calibration_v32.get("response_count") != TRANSPORT_CHECK_CELLS:
        raise SelectionPoweredPlanV33Error("v32 transport check is incomplete")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = (
        "preregistered_after_eight_normal_coreweave_checks_before_full_deepseek_block"
    )
    document["inputs"]["plan_v32_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["calibration_v32"] = {
        **dict(calibration_v32),
        "task_ids": list(TRANSPORT_CHECK_TASK_IDS),
        "interface_finding": (
            "all eight responses completed normally on exact CoreWeave FP8 with the dated "
            "DeepSeek V4 Pro identity and stop finish reason"
        ),
        "scores_or_selections_inspected_before_successor_freeze": False,
        "captured_responses_remain_calibration_only": True,
    }
    recovery = document["execution"]["deepseek_route_recovery"]
    recovery.update(
        {
            "transport_check_source_plan_sha256": predecessor["artifact_sha256"],
            "transport_check_reused_as_primary": False,
            "successor_primary_cells": PRIMARY_CELLS,
            "successor_repeat_cells": REPEAT_CELLS,
            "successor_execution_order": [
                "complete_all_primary_cells_under_v33",
                "verify_complete_exact_provider_block_without_inspecting_scores",
                "complete_all_repeat_cells_under_v33",
            ],
        }
    )
    overrides = document["execution"]["collection_concurrency"]["per_model_by_model_id"]
    overrides[DEEPSEEK_PRO_MODEL_ID] = DEEPSEEK_CONCURRENCY
    document["execution"]["deepseek_concurrency_successor"] = {
        "model_id": DEEPSEEK_PRO_MODEL_ID,
        "predecessor_concurrency": 1,
        "successor_concurrency": DEEPSEEK_CONCURRENCY,
        "basis": "eight_normal_exact_route_responses",
        "changes_scoring_or_inference": False,
        "changes_prompt_or_decoding": False,
        "changes_provider_or_model_identity": False,
    }
    document["budget"]["calibration_spend_micros"] = int(
        predecessor["budget"]["calibration_spend_micros"]
    ) + int(calibration_v32["spend_micros"])
    document["budget"]["hard_cap"] = SUCCESSOR_RUN_CAP_USD
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV33Error("constructed v33 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        execution = document["execution"]
        recovery = execution["deepseek_route_recovery"]
        concurrency = execution["deepseek_concurrency_successor"]
        calibration = document["inputs"]["calibration_v32"]
        predecessor = document["inputs"]["plan_v32_predecessor"]
        route = document["inputs"]["route_manifest"]
        policy_document = execution["execution_policy"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v31()
    rows = {row["model_id"]: row for row in roster}
    deepseek = rows.get(DEEPSEEK_PRO_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and len(roster) == 20
        and len(rows) == 20
        and deepseek.get("provider_tag") == REPLACEMENT_TAG
        and deepseek.get("provider_name") == REPLACEMENT_PROVIDER
        and execution["collection_concurrency"]["per_model_by_model_id"].get(DEEPSEEK_PRO_MODEL_ID)
        == DEEPSEEK_CONCURRENCY
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and execution.get("execution_policy_sha256") == policy.sha256
        and policy_document["limits"]["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and concurrency
        == {
            "model_id": DEEPSEEK_PRO_MODEL_ID,
            "predecessor_concurrency": 1,
            "successor_concurrency": DEEPSEEK_CONCURRENCY,
            "basis": "eight_normal_exact_route_responses",
            "changes_scoring_or_inference": False,
            "changes_prompt_or_decoding": False,
            "changes_provider_or_model_identity": False,
        }
        and recovery.get("transport_check_cells") == TRANSPORT_CHECK_CELLS
        and recovery.get("transport_check_task_ids") == list(TRANSPORT_CHECK_TASK_IDS)
        and recovery.get("transport_check_source_plan_sha256") == predecessor["semantic_sha256"]
        and recovery.get("transport_check_reused_as_primary") is False
        and recovery.get("successor_primary_cells") == PRIMARY_CELLS
        and recovery.get("successor_repeat_cells") == REPEAT_CELLS
        and recovery.get("reuse_predecessor_responses") is False
        and recovery.get("cross_provider_score_pooling") is False
        and calibration.get("response_count") == TRANSPORT_CHECK_CELLS
        and calibration.get("task_ids") == list(TRANSPORT_CHECK_TASK_IDS)
        and calibration.get("used_as_primary_data") is False
        and calibration.get("captured_responses_remain_calibration_only") is True
        and calibration.get("scores_or_selections_inspected_before_successor_freeze") is False
        and isinstance(predecessor.get("semantic_sha256"), str)
        and len(predecessor["semantic_sha256"]) == 64
        and isinstance(route.get("semantic_sha256"), str)
        and len(route["semantic_sha256"]) == 64
        and document["design"]["primary_provider_calls"] == 12_800
        and document["inference"]["bootstrap_resamples"] == 50_000
        and document["inference"]["permutation_resamples"] == 100_000
        and document["budget"]["hard_cap"] == SUCCESSOR_RUN_CAP_USD
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV33Error("content-addressed plan conflict")
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
    parser.add_argument("--calibration-v32-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    calibration = run_commitment(
        args.calibration_v32_directory,
        expected_responses=TRANSPORT_CHECK_CELLS,
    )
    plan = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        calibration_v32=calibration,
    )
    print(_write(plan, args.output_directory))


if __name__ == "__main__":
    run()
