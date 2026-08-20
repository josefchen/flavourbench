"""Freeze the panel-2 DeepSeek all-cell BaseTen replacement."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v23 import _roster_row
from .epicure_selection_powered_plan_v44 import selection_execution_policy_v44
from .epicure_selection_powered_plan_v54 import _sha256, _sha256_file
from .epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from .epicure_selection_route_manifest_v57 import (
    DEEPSEEK_PRO_MODEL_ID,
    PROVIDER_NAME,
    ROUTE_TAG,
    SUPERSEDED_TAG,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v59"
PLAN_VERSION = "flavourbench-selection-26x640-panel-2-deepseek-complete-block-repair-v59"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV59Error(RuntimeError):
    """The panel-2 DeepSeek replacement plan failed verification."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanV59Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV59Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v54(predecessor) or not verify_manifest(manifest):
        raise SelectionPoweredPlanV59Error("v59 predecessor or manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = {candidate.model_id: candidate for candidate in select_candidates(manifest)}
    if list(prior_rows) != list(candidates):
        raise SelectionPoweredPlanV59Error("v59 roster differs from panel 2")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "panel_2_deepseek_complete_block_repair_frozen_before_execution"
    document["inputs"]["plan_v54_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["superseded_route_manifest_v54"] = document["inputs"]["route_manifest"]
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }

    final_rows = json.loads(json.dumps(predecessor["roster"]["models"]))
    index = next(i for i, row in enumerate(final_rows) if row["model_id"] == DEEPSEEK_PRO_MODEL_ID)
    prior = prior_rows[DEEPSEEK_PRO_MODEL_ID]
    replacement = _roster_row(
        candidates[DEEPSEEK_PRO_MODEL_ID], str(prior["final_reasoning_effort"])
    )
    if "final_max_output_tokens" in prior:
        replacement["final_max_output_tokens"] = int(prior["final_max_output_tokens"])
    final_rows[index] = replacement
    document["roster"]["models"] = final_rows
    document["execution"]["deepseek_complete_block_replacement_v59"] = {
        "schema_version": "flavourbench-score-blind-complete-block-route-replacement-v3",
        "source_plan_sha256": predecessor["artifact_sha256"],
        "replacement_model_ids": [DEEPSEEK_PRO_MODEL_ID],
        "replacement_provider_tags": {DEEPSEEK_PRO_MODEL_ID: ROUTE_TAG},
        "replacement_provider_names": {DEEPSEEK_PRO_MODEL_ID: PROVIDER_NAME},
        "superseded_provider_tags": {DEEPSEEK_PRO_MODEL_ID: [SUPERSEDED_TAG]},
        "replacement_primary_cells_per_model": PRIMARY_TASKS,
        "replacement_repeat_cells_per_model": REPEAT_TASKS,
        "replacement_blocks_must_be_complete": True,
        "superseded_responses_used": False,
        "cross_route_response_pooling": False,
        "selective_failed_cell_retry": False,
        "selection_uses_scores_or_selections": False,
        "selection_uses_transport_status_only": True,
        "automatic_fallback": False,
        "quality_score_definition": "successful_and_parseable_only",
    }
    document["execution"]["reasoning_control"] = (
        "replace all 640 primary and 64 repeat DeepSeek cells through exact BaseTen FP4; "
        "preserve tasks, prompts, decoding, scoring, and inference"
    )
    document["execution"]["collection_concurrency"] = {
        "global": 2,
        "per_model_default": 2,
        "per_model_by_backend": {"openrouter": 2},
        "per_model_by_model_id": {DEEPSEEK_PRO_MODEL_ID: 2},
        "reason": "single exact-route complete block with two-call concurrency",
    }
    document["budget"].update(
        {
            "aggregate_program_cap": "800",
            "program_cap": "800",
            "hard_cap": "800",
            "successor_scope": "one complete 640+64 panel-2 DeepSeek replacement block",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV59Error("constructed v59 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        replacement = document["execution"]["deepseek_complete_block_replacement_v59"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        policy_document = document["execution"]["execution_policy"]
        predecessor = document["inputs"]["plan_v54_predecessor"]
        manifest = document["inputs"]["route_manifest"]
        superseded_manifest = document["inputs"]["superseded_route_manifest_v54"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    row = rows.get(DEEPSEEK_PRO_MODEL_ID) or {}
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status")
        == "panel_2_deepseek_complete_block_repair_frozen_before_execution"
        and len(rows) == 26
        and row.get("provider_tag") == ROUTE_TAG
        and row.get("provider_name") == PROVIDER_NAME
        and replacement.get("replacement_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and replacement.get("replacement_provider_tags") == {DEEPSEEK_PRO_MODEL_ID: ROUTE_TAG}
        and replacement.get("replacement_primary_cells_per_model") == PRIMARY_TASKS
        and replacement.get("replacement_repeat_cells_per_model") == REPEAT_TASKS
        and replacement.get("replacement_blocks_must_be_complete") is True
        and replacement.get("superseded_responses_used") is False
        and replacement.get("cross_route_response_pooling") is False
        and replacement.get("selective_failed_cell_retry") is False
        and replacement.get("selection_uses_scores_or_selections") is False
        and replacement.get("selection_uses_transport_status_only") is True
        and replacement.get("automatic_fallback") is False
        and document["outcomes"].get("failed_content_filtered_or_unparseable")
        == "excluded_from_quality_score"
        and policy_document == policy.document()
        and verify_policy_document(policy_document)
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and all(
            isinstance(pin.get("semantic_sha256"), str)
            and len(pin["semantic_sha256"]) == 64
            and isinstance(pin.get("physical_sha256"), str)
            and len(pin["physical_sha256"]) == 64
            for pin in (predecessor, manifest, superseded_manifest)
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV59Error("content-addressed plan conflict")
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
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    manifest = _load(args.manifest)
    document = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
