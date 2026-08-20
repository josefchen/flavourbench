"""Freeze replication 2 with Fable on the score-blind Bedrock route."""

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
from .epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID
from .epicure_selection_route_manifest_v49 import (
    FABLE_BEDROCK_SPEC,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v49"
PLAN_VERSION = "flavourbench-selection-26x640-replication-2-fable-bedrock-v49"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV49Error(RuntimeError):
    """The Fable/Bedrock replication plan failed verification."""


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
        raise SelectionPoweredPlanV49Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV49Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v46(predecessor) or not verify_manifest(manifest):
        raise SelectionPoweredPlanV49Error("v49 predecessor or route manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV49Error("v49 roster differs from replication 2")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "replication_2_fable_bedrock_route_frozen_before_execution"
    document["inputs"]["plan_v46_predecessor"] = {
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
    document["execution"]["fable_bedrock_route"] = {
        "schema_version": "flavourbench-score-blind-full-block-route-v1",
        "model_id": FABLE_MODEL_ID,
        "provider_tag": FABLE_BEDROCK_SPEC["tag"],
        "primary_cells": PRIMARY_TASKS,
        "repeat_cells": REPEAT_TASKS,
        "route_frozen_before_any_replication_2_call": True,
        "selection_uses_status_and_finish_metadata_only": True,
        "quality_scores_or_selections_used": False,
        "automatic_fallback": False,
        "selective_failed_cell_retry": False,
    }
    document["execution"]["reasoning_control"] = (
        "retain the v46 response-blind replication contract with Fable fixed to the exact "
        "OpenRouter Amazon Bedrock endpoint before any replication-2 call"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV49Error("constructed v49 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        route = document["execution"]["fable_bedrock_route"]
        replication = document["execution"]["replication_2"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        fable = rows[FABLE_MODEL_ID]
        policy_document = document["execution"]["execution_policy"]
        predecessor = document["inputs"]["plan_v46_predecessor"]
        manifest = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status") == "replication_2_fable_bedrock_route_frozen_before_execution"
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(rows) == MODEL_COUNT
        and fable.get("provider_tag") == FABLE_BEDROCK_SPEC["tag"]
        and fable.get("provider_name") == FABLE_BEDROCK_SPEC["provider"]
        and fable.get("final_reasoning_effort") == FABLE_BEDROCK_SPEC["reasoning_effort"]
        and fable.get("final_max_output_tokens") == FABLE_BEDROCK_SPEC["max_output_tokens"]
        and route.get("model_id") == FABLE_MODEL_ID
        and route.get("provider_tag") == FABLE_BEDROCK_SPEC["tag"]
        and route.get("primary_cells") == PRIMARY_TASKS
        and route.get("repeat_cells") == REPEAT_TASKS
        and route.get("route_frozen_before_any_replication_2_call") is True
        and route.get("selection_uses_status_and_finish_metadata_only") is True
        and route.get("quality_scores_or_selections_used") is False
        and route.get("automatic_fallback") is False
        and route.get("selective_failed_cell_retry") is False
        and replication.get("replication_index") == 2
        and replication.get("primary_tasks_per_model") == PRIMARY_TASKS
        and replication.get("repeat_tasks_per_model") == REPEAT_TASKS
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
            raise SelectionPoweredPlanV49Error("content-addressed plan conflict")
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
