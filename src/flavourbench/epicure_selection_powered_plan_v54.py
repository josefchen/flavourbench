"""Freeze complete score-blind coverage-repair blocks for panel 2."""

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
from .epicure_selection_powered_plan_v52 import verify_plan as verify_plan_v52
from .epicure_selection_route_manifest_v54 import (
    REPLACEMENT_MODEL_IDS,
    ROUTE_SPECS,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v54"
PLAN_VERSION = "flavourbench-selection-26x640-panel-2-complete-coverage-repair-v54"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV54Error(RuntimeError):
    """The panel-2 complete-coverage repair plan failed verification."""


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
        raise SelectionPoweredPlanV54Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV54Error("plan input is not a JSON object")
    return value


def _replacement_row(*, prior: Mapping[str, Any], candidate: Any) -> dict[str, Any]:
    reasoning_effort = str(prior["final_reasoning_effort"])
    row = _roster_row(candidate, reasoning_effort)
    if "final_max_output_tokens" in prior:
        row["final_max_output_tokens"] = int(prior["final_max_output_tokens"])
    return row


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v52(predecessor) or not verify_manifest(manifest):
        raise SelectionPoweredPlanV54Error("v54 predecessor or route manifest failed verification")
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV54Error("v54 roster differs from panel 2")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "panel_2_complete_coverage_repairs_frozen_before_execution"
    document["inputs"]["plan_v52_predecessor"] = {
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
        final_rows[final_by_id[model_id]] = _replacement_row(
            prior=prior_rows[model_id],
            candidate=candidates_by_id[model_id],
        )
    document["roster"]["models"] = final_rows
    document["execution"]["complete_coverage_route_replacements_v54"] = {
        "schema_version": "flavourbench-score-blind-complete-block-route-replacement-v2",
        "source_plan_sha256": predecessor["artifact_sha256"],
        "replacement_model_ids": REPLACEMENT_MODEL_IDS,
        "replacement_provider_tags": {
            model_id: ROUTE_SPECS[model_id]["tag"] for model_id in REPLACEMENT_MODEL_IDS
        },
        "superseded_provider_tags": {
            model_id: ROUTE_SPECS[model_id]["superseded_tags"] for model_id in REPLACEMENT_MODEL_IDS
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
        "replace every cell for each incomplete panel-2 route; preserve tasks, prompts, "
        "decoding, scoring, and inference; never pool superseded cells"
    )
    document["execution"]["collection_concurrency"] = {
        "global": 18,
        "per_model_default": 2,
        "per_model_by_backend": {"openrouter": 2},
        "per_model_by_model_id": {},
        "reason": "two calls per exact route limits replacement-induced transport failures",
    }
    document["budget"].update(
        {
            "aggregate_program_cap": "1600",
            "program_cap": "1600",
            "hard_cap": "800",
            "successor_scope": "nine complete 640+64 panel-2 replacement blocks",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV54Error("constructed v54 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        replacement = document["execution"]["complete_coverage_route_replacements_v54"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        policy_document = document["execution"]["execution_policy"]
        predecessor = document["inputs"]["plan_v52_predecessor"]
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
            )
        except (KeyError, TypeError):
            return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status") == "panel_2_complete_coverage_repairs_frozen_before_execution"
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
            raise SelectionPoweredPlanV54Error("content-addressed plan conflict")
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
