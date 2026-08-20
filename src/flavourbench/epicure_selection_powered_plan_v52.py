"""Freeze complete Luna and DeepSeek Flash replacement blocks for panel 2."""

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
from .epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from .epicure_selection_route_manifest_v52 import (
    REPLACEMENT_MODEL_IDS,
    ROUTE_SPECS,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v52"
PLAN_VERSION = "flavourbench-selection-26x640-replication-2-route-repair-v52"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV52Error(RuntimeError):
    """The replication-2 complete-block replacement plan failed verification."""


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
        raise SelectionPoweredPlanV52Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV52Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v49(predecessor) or not verify_manifest(manifest):
        raise SelectionPoweredPlanV52Error("v52 predecessor or route manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV52Error("v52 roster differs from replication 2")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "replication_2_complete_route_replacements_frozen_before_execution"
    document["inputs"]["plan_v49_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }

    final_rows = json.loads(json.dumps(predecessor["roster"]["models"]))
    final_by_id = {str(row["model_id"]): index for index, row in enumerate(final_rows)}
    candidates_by_id = {candidate.model_id: candidate for candidate in candidates}
    for model_id in REPLACEMENT_MODEL_IDS:
        spec = ROUTE_SPECS[model_id]
        row = _roster_row(candidates_by_id[model_id], str(spec["reasoning_effort"]))
        row["final_max_output_tokens"] = int(spec["max_output_tokens"])
        final_rows[final_by_id[model_id]] = row
    document["roster"]["models"] = final_rows
    document["execution"]["panel_2_route_replacements_v52"] = {
        "schema_version": "flavourbench-score-blind-complete-block-route-replacement-v1",
        "source_plan_sha256": predecessor["artifact_sha256"],
        "replacement_model_ids": REPLACEMENT_MODEL_IDS,
        "replacement_provider_tags": {
            model_id: ROUTE_SPECS[model_id]["tag"] for model_id in REPLACEMENT_MODEL_IDS
        },
        "superseded_provider_tags": {
            model_id: ROUTE_SPECS[model_id]["superseded_tag"] for model_id in REPLACEMENT_MODEL_IDS
        },
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
        "replace the entire Luna Pro and DeepSeek Flash replication-2 blocks after "
        "score-blind transport failures; do not reuse or pool any superseded-route cell"
    )
    document["budget"].update(
        {
            "aggregate_program_cap": "750",
            "program_cap": "750",
            "hard_cap": "250",
            "successor_scope": "two complete 640+64 panel-2 replacement blocks",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV52Error("constructed v52 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        replacement = document["execution"]["panel_2_route_replacements_v52"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        policy_document = document["execution"]["execution_policy"]
        predecessor = document["inputs"]["plan_v49_predecessor"]
        manifest = document["inputs"]["route_manifest"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    model_checks = []
    for model_id in REPLACEMENT_MODEL_IDS:
        try:
            row = rows[model_id]
            spec = ROUTE_SPECS[model_id]
            model_checks.append(
                row.get("provider_tag") == spec["tag"]
                and row.get("provider_name") == spec["provider"]
                and row.get("final_reasoning_effort") == spec["reasoning_effort"]
                and row.get("final_max_output_tokens") == spec["max_output_tokens"]
            )
        except (KeyError, TypeError):
            return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status")
        == "replication_2_complete_route_replacements_frozen_before_execution"
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(rows) == MODEL_COUNT
        and all(model_checks)
        and replacement.get("replacement_model_ids") == REPLACEMENT_MODEL_IDS
        and replacement.get("replacement_provider_tags")
        == {model_id: ROUTE_SPECS[model_id]["tag"] for model_id in REPLACEMENT_MODEL_IDS}
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
            raise SelectionPoweredPlanV52Error("content-addressed plan conflict")
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
