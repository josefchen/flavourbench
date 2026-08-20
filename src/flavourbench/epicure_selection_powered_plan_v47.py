"""Freeze the score-blind panel-1 composite after the complete Qwen replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v44 import (
    selection_execution_policy_v44,
)
from .epicure_selection_powered_plan_v44 import (
    verify_plan as verify_plan_v44,
)
from .epicure_selection_powered_plan_v45 import (
    verify_plan as verify_plan_v45,
)
from .epicure_selection_route_manifest_v45 import (
    FABLE_MODEL_ID,
    QWEN_MODEL_ID,
    ROUTE_SPECS,
)
from .epicure_selection_route_manifest_v46 import verify_manifest
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v47"
PLAN_VERSION = "flavourbench-selection-26x640-panel-1-composite-v47"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV47Error(RuntimeError):
    """The panel-1 composite plan failed verification."""


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
        raise SelectionPoweredPlanV47Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV47Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    recovery: Mapping[str, Any],
    recovery_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v44(predecessor) or not verify_plan_v45(recovery):
        raise SelectionPoweredPlanV47Error("panel-1 source plans failed verification")
    if not verify_manifest(manifest):
        raise SelectionPoweredPlanV47Error("Qwen-only route manifest failed verification")
    recovery_predecessor = recovery["inputs"]["plan_v44_predecessor"]
    if recovery_predecessor != {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }:
        raise SelectionPoweredPlanV47Error("Qwen recovery does not descend from this v44 plan")

    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    recovery_rows = {str(row["model_id"]): row for row in recovery["roster"]["models"]}
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV47Error("composite roster order differs from v44")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "panel_1_composite_frozen_before_quality_analysis"
    document["inputs"]["plan_v44_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["plan_v45_qwen_source"] = {
        "semantic_sha256": recovery["artifact_sha256"],
        "physical_sha256": recovery_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }

    final_rows = json.loads(json.dumps(predecessor["roster"]["models"]))
    qwen_index = next(
        index for index, row in enumerate(final_rows) if row["model_id"] == QWEN_MODEL_ID
    )
    final_rows[qwen_index] = json.loads(json.dumps(recovery_rows[QWEN_MODEL_ID]))
    document["roster"]["models"] = final_rows
    document["execution"]["panel_1_composite"] = {
        "schema_version": "flavourbench-score-blind-panel-composite-v1",
        "base_plan_sha256": predecessor["artifact_sha256"],
        "replacement_plan_sha256": recovery["artifact_sha256"],
        "replacement_model_ids": [QWEN_MODEL_ID],
        "base_model_ids": [model_id for model_id in prior_rows if model_id != QWEN_MODEL_ID],
        "replacement_primary_cells": PRIMARY_TASKS,
        "replacement_repeat_cells": REPEAT_TASKS,
        "replacement_is_complete_model_block": True,
        "superseded_qwen_responses_used": False,
        "cross_route_response_pooling": False,
        "selective_failed_cell_retry": False,
        "selection_uses_scores_or_selections": False,
        "selection_uses_completion_identity_and_finish_metadata_only": True,
        "fable_alternate_pilot": {
            "provider_tag": ROUTE_SPECS[FABLE_MODEL_ID]["tag"],
            "scheduled_cells": 4,
            "normal_completions": 1,
            "content_filtered": 3,
            "selected_for_full_replacement": False,
            "reason": "alternate route did not improve score-blind completion reliability",
        },
        "fable_source_plan_sha256": predecessor["artifact_sha256"],
        "qwen_source_plan_sha256": recovery["artifact_sha256"],
        "quality_score_definition": "successful_and_parseable_only",
    }
    document["execution"]["reasoning_control"] = (
        "analyze the complete v44 panel-1 blocks, replacing only the entire Qwen A95B block "
        "with the complete v45 Alibaba block selected without inspecting scores or selections"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV47Error("constructed panel-1 composite failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        composite = document["execution"]["panel_1_composite"]
        policy_document = document["execution"]["execution_policy"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        qwen = rows[QWEN_MODEL_ID]
        fable = composite["fable_alternate_pilot"]
    except (KeyError, TypeError):
        return False
    qwen_spec = ROUTE_SPECS[QWEN_MODEL_ID]
    policy = selection_execution_policy_v44()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(rows) == MODEL_COUNT
        and qwen.get("provider_tag") == qwen_spec["tag"]
        and qwen.get("provider_name") == qwen_spec["provider"]
        and qwen.get("final_reasoning_effort") == qwen_spec["reasoning_effort"]
        and qwen.get("final_max_output_tokens") == qwen_spec["max_output_tokens"]
        and composite.get("replacement_model_ids") == [QWEN_MODEL_ID]
        and len(composite.get("base_model_ids") or []) == MODEL_COUNT - 1
        and QWEN_MODEL_ID not in composite.get("base_model_ids", [])
        and composite.get("replacement_primary_cells") == PRIMARY_TASKS
        and composite.get("replacement_repeat_cells") == REPEAT_TASKS
        and composite.get("replacement_is_complete_model_block") is True
        and composite.get("superseded_qwen_responses_used") is False
        and composite.get("cross_route_response_pooling") is False
        and composite.get("selective_failed_cell_retry") is False
        and composite.get("selection_uses_scores_or_selections") is False
        and composite.get("selection_uses_completion_identity_and_finish_metadata_only") is True
        and fable.get("provider_tag") == ROUTE_SPECS[FABLE_MODEL_ID]["tag"]
        and fable.get("scheduled_cells") == 4
        and fable.get("normal_completions") == 1
        and fable.get("content_filtered") == 3
        and fable.get("selected_for_full_replacement") is False
        and document["outcomes"].get("failed_content_filtered_or_unparseable")
        == "excluded_from_quality_score"
        and document["outcomes"].get("coverage_reported_for_every_model") is True
        and document["outcomes"].get("dnf_classification") is False
        and document["outcomes"].get("minimum_coverage_for_score") is None
        and policy_document == policy.document()
        and verify_policy_document(policy_document)
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and all(
            isinstance((document["inputs"].get(label) or {}).get("semantic_sha256"), str)
            and isinstance((document["inputs"].get(label) or {}).get("physical_sha256"), str)
            for label in ("plan_v44_predecessor", "plan_v45_qwen_source", "route_manifest")
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV47Error("content-addressed plan conflict")
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
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    recovery = _load(args.recovery_plan)
    manifest = _load(args.manifest)
    document = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor),
        recovery=recovery,
        recovery_physical_sha256=_sha256_file(args.recovery_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
