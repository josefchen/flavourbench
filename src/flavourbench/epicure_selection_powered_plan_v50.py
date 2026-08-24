"""Freeze panel 1 with complete Qwen and Fable replacement blocks."""

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
from .epicure_selection_powered_plan_v44 import selection_execution_policy_v44
from .epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID, QWEN_MODEL_ID
from .epicure_selection_route_manifest_v49 import (
    FABLE_BEDROCK_SPEC,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v50"
PLAN_VERSION = "flavourbench-selection-26x640-panel-1-qwen-fable-composite-v50"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV50Error(RuntimeError):
    """The panel-1 Qwen/Fable composite plan failed verification."""


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
        raise SelectionPoweredPlanV50Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV50Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v47(predecessor) or not verify_manifest(manifest):
        raise SelectionPoweredPlanV50Error("v50 predecessor or route manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV50Error("v50 roster differs from panel 1")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "panel_1_composite_frozen_before_fable_bedrock_pilot"
    document["inputs"]["plan_v47_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    final_rows = json.loads(json.dumps(predecessor["roster"]["models"]))
    fable_index = next(
        index for index, row in enumerate(final_rows) if row["model_id"] == FABLE_MODEL_ID
    )
    candidate = next(candidate for candidate in candidates if candidate.model_id == FABLE_MODEL_ID)
    fable = _roster_row(candidate, str(FABLE_BEDROCK_SPEC["reasoning_effort"]))
    fable["final_max_output_tokens"] = int(FABLE_BEDROCK_SPEC["max_output_tokens"])
    final_rows[fable_index] = fable
    document["roster"]["models"] = final_rows
    prior_composite = predecessor["execution"]["panel_1_composite"]
    document["execution"]["panel_1_composite_v2"] = {
        "schema_version": "flavourbench-score-blind-panel-composite-v2",
        "base_plan_sha256": prior_composite["base_plan_sha256"],
        "qwen_replacement_plan_sha256": prior_composite["replacement_plan_sha256"],
        "fable_replacement_plan_sha256": "self",
        "replacement_model_ids": [QWEN_MODEL_ID, FABLE_MODEL_ID],
        "base_model_ids": [
            model_id for model_id in prior_rows if model_id not in {QWEN_MODEL_ID, FABLE_MODEL_ID}
        ],
        "replacement_primary_cells_per_model": PRIMARY_TASKS,
        "replacement_repeat_cells_per_model": REPEAT_TASKS,
        "replacement_blocks_must_be_complete": True,
        "superseded_qwen_responses_used": False,
        "superseded_fable_responses_used": False,
        "cross_route_response_pooling": False,
        "selective_failed_cell_retry": False,
        "selection_uses_scores_or_selections": False,
        "selection_uses_completion_identity_and_finish_metadata_only": True,
        "fable_provider_tag": FABLE_BEDROCK_SPEC["tag"],
        "quality_score_definition": "successful_and_parseable_only",
    }
    document["execution"]["reasoning_control"] = (
        "analyze the v44 panel, replacing the entire Qwen block with the complete Alibaba "
        "block and the entire Fable block with a separately completed Bedrock block; no cells "
        "are pooled across routes"
    )
    document["budget"].update(
        {
            "aggregate_program_cap": "750",
            "program_cap": "750",
            "hard_cap": "250",
            "successor_scope": "one Fable pilot and at most one complete 640+64 Fable block",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV50Error("constructed v50 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        composite = document["execution"]["panel_1_composite_v2"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        fable = rows[FABLE_MODEL_ID]
        policy_document = document["execution"]["execution_policy"]
        predecessor = document["inputs"]["plan_v47_predecessor"]
        manifest = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status") == "panel_1_composite_frozen_before_fable_bedrock_pilot"
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(rows) == MODEL_COUNT
        and fable.get("provider_tag") == FABLE_BEDROCK_SPEC["tag"]
        and fable.get("provider_name") == FABLE_BEDROCK_SPEC["provider"]
        and fable.get("final_reasoning_effort") == FABLE_BEDROCK_SPEC["reasoning_effort"]
        and fable.get("final_max_output_tokens") == FABLE_BEDROCK_SPEC["max_output_tokens"]
        and composite.get("replacement_model_ids") == [QWEN_MODEL_ID, FABLE_MODEL_ID]
        and len(composite.get("base_model_ids") or []) == MODEL_COUNT - 2
        and not ({QWEN_MODEL_ID, FABLE_MODEL_ID} & set(composite.get("base_model_ids") or []))
        and composite.get("replacement_primary_cells_per_model") == PRIMARY_TASKS
        and composite.get("replacement_repeat_cells_per_model") == REPEAT_TASKS
        and composite.get("replacement_blocks_must_be_complete") is True
        and composite.get("superseded_qwen_responses_used") is False
        and composite.get("superseded_fable_responses_used") is False
        and composite.get("cross_route_response_pooling") is False
        and composite.get("selective_failed_cell_retry") is False
        and composite.get("selection_uses_scores_or_selections") is False
        and composite.get("selection_uses_completion_identity_and_finish_metadata_only") is True
        and composite.get("fable_provider_tag") == FABLE_BEDROCK_SPEC["tag"]
        and document["outcomes"].get("failed_content_filtered_or_unparseable")
        == "excluded_from_quality_score"
        and document["outcomes"].get("dnf_classification") is False
        and policy_document == policy.document()
        and verify_policy_document(policy_document)
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and isinstance(predecessor.get("semantic_sha256"), str)
        and isinstance(predecessor.get("physical_sha256"), str)
        and isinstance(manifest.get("semantic_sha256"), str)
        and isinstance(manifest.get("physical_sha256"), str)
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV50Error("content-addressed plan conflict")
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
