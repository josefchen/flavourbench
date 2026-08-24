"""Freeze score-blind Fable/Qwen completion routes before replacement calls."""

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
from .epicure_selection_powered_plan_v44 import (
    selection_execution_policy_v44,
)
from .epicure_selection_powered_plan_v44 import (
    verify_plan as verify_plan_v44,
)
from .epicure_selection_route_manifest_v45 import (
    FABLE_MODEL_ID,
    QWEN_MODEL_ID,
    ROUTE_SPECS,
    verify_manifest,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v45"
PLAN_VERSION = "flavourbench-selection-26x640-completion-routes-v45"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
PAIR_COUNT = MODEL_COUNT * (MODEL_COUNT - 1) // 2


class SelectionPoweredPlanV45Error(RuntimeError):
    """The completion-route plan failed verification."""


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
        raise SelectionPoweredPlanV45Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV45Error("input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v44(predecessor):
        raise SelectionPoweredPlanV45Error("v45 requires exact v44 predecessor")
    if not verify_manifest(manifest):
        raise SelectionPoweredPlanV45Error("v45 route manifest failed verification")
    candidates = select_candidates(manifest)
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV45Error("v45 roster order differs from v44")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "completion_routes_frozen_before_score_blind_transport_pilot"
    document["inputs"]["plan_v44_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    roster: list[dict[str, Any]] = []
    for candidate in candidates:
        prior = prior_rows[candidate.model_id]
        effort = str(prior["final_reasoning_effort"])
        if candidate.model_id in ROUTE_SPECS:
            effort = str(ROUTE_SPECS[candidate.model_id]["reasoning_effort"])
        row = _roster_row(candidate, effort)
        maximum = prior.get("final_max_output_tokens")
        if candidate.model_id in ROUTE_SPECS:
            maximum = int(ROUTE_SPECS[candidate.model_id]["max_output_tokens"])
        if maximum is not None:
            row["final_max_output_tokens"] = int(maximum)
        roster.append(row)
    document["roster"]["models"] = roster
    document["execution"]["completion_route_recovery"] = {
        "schema_version": "flavourbench-score-blind-full-block-completion-v1",
        "model_ids": [FABLE_MODEL_ID, QWEN_MODEL_ID],
        "transport_pilot_task_ids": list(document["execution"]["pilot"]["task_ids"]),
        "transport_pilot_cells_per_model": 4,
        "pilot_selection_uses_scores_or_selections": False,
        "pilot_normal_completion_and_identity_only": True,
        "pilot_responses_may_be_reused_in_same_frozen_block": True,
        "complete_primary_cells_per_model": PRIMARY_TASKS,
        "complete_repeat_cells_per_model": REPEAT_TASKS,
        "selective_failed_cell_retry": False,
        "cross_route_response_pooling": False,
        "source_v44_responses_used_in_v45_score": False,
        "automatic_fallback": False,
        "quality_score_definition": "successful_and_parseable_only",
        "failure_endpoint": "coverage_and_retry_burden_only",
    }
    document["execution"]["reasoning_control"] = (
        "retain the v44 anchor-free task, decoding, and scoring contract; recollect complete "
        "Fable and Qwen A95B blocks on fixed score-blind recovery routes"
    )
    document["budget"].update(
        {
            "aggregate_program_cap": "450",
            "program_cap": "450",
            "hard_cap": "250",
            "successor_scope": (
                "transport pilots followed by at most two complete 640-primary plus 64-repeat "
                "replacement blocks"
            ),
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV45Error("constructed v45 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        recovery = document["execution"]["completion_route_recovery"]
        policy_document = document["execution"]["execution_policy"]
        rows = {row["model_id"]: row for row in document["roster"]["models"]}
        route = document["inputs"]["route_manifest"]
        predecessor = document["inputs"]["plan_v44_predecessor"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == MODEL_COUNT
        and len(rows) == MODEL_COUNT
        and set(ROUTE_SPECS) <= rows.keys()
        and all(
            rows[model_id].get("provider_tag") == spec["tag"]
            and rows[model_id].get("provider_name") == spec["provider"]
            and rows[model_id].get("final_reasoning_effort") == spec["reasoning_effort"]
            and rows[model_id].get("final_max_output_tokens") == spec["max_output_tokens"]
            for model_id, spec in ROUTE_SPECS.items()
        )
        and isinstance(route.get("semantic_sha256"), str)
        and isinstance(route.get("physical_sha256"), str)
        and isinstance(predecessor.get("semantic_sha256"), str)
        and isinstance(predecessor.get("physical_sha256"), str)
        and recovery.get("model_ids") == [FABLE_MODEL_ID, QWEN_MODEL_ID]
        and recovery.get("transport_pilot_cells_per_model") == 4
        and recovery.get("pilot_selection_uses_scores_or_selections") is False
        and recovery.get("pilot_normal_completion_and_identity_only") is True
        and recovery.get("selective_failed_cell_retry") is False
        and recovery.get("cross_route_response_pooling") is False
        and recovery.get("source_v44_responses_used_in_v45_score") is False
        and recovery.get("automatic_fallback") is False
        and recovery.get("quality_score_definition") == "successful_and_parseable_only"
        and document["outcomes"].get("failed_content_filtered_or_unparseable")
        == "excluded_from_quality_score"
        and document["outcomes"].get("dnf_classification") is False
        and document["outcomes"].get("minimum_coverage_for_score") is None
        and policy_document == policy.document()
        and verify_policy_document(policy_document)
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("hard_cap") == "250"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV45Error("content-addressed plan conflict")
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
