"""Freeze the response-blind 640-task replication-2 collection plan."""

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
from .epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel,
)
from .epicure_selection_route_manifest_v45 import QWEN_MODEL_ID, ROUTE_SPECS
from .epicure_selection_route_manifest_v46 import verify_manifest
from .epicure_selection_taskset_replication_v1 import (
    FAMILIES,
    verify_taskset,
)
from .execution_policy import verify_policy_document
from .frontier_contract_runner import select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v46"
PLAN_VERSION = "flavourbench-selection-26x640-replication-2-v46"
MODEL_COUNT = 26
PRIMARY_TASKS = 640
REPEAT_TASKS = 64


class SelectionPoweredPlanV46Error(RuntimeError):
    """The second powered replication plan failed verification."""


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
        raise SelectionPoweredPlanV46Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV46Error("plan input is not a JSON object")
    return value


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    taskset: Mapping[str, Any],
    taskset_physical_sha256: str,
    repeat_panel: Mapping[str, Any],
    repeat_panel_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v44(predecessor):
        raise SelectionPoweredPlanV46Error("v46 requires exact v44 predecessor")
    if not verify_manifest(manifest) or not verify_taskset(taskset):
        raise SelectionPoweredPlanV46Error("replication manifest or task set failed verification")
    if not verify_repeat_panel(repeat_panel, taskset=taskset):
        raise SelectionPoweredPlanV46Error("replication repeat panel failed verification")
    candidates = select_candidates(manifest)
    prior_rows = {str(row["model_id"]): row for row in predecessor["roster"]["models"]}
    if len(candidates) != MODEL_COUNT or [candidate.model_id for candidate in candidates] != list(
        prior_rows
    ):
        raise SelectionPoweredPlanV46Error("replication roster differs from v44")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "replication_2_frozen_before_model_execution"
    document["inputs"]["plan_v44_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["taskset"] = {
        "semantic_sha256": taskset["artifact_sha256"],
        "physical_sha256": taskset_physical_sha256,
    }
    document["inputs"]["repeat_panel"] = {
        "semantic_sha256": repeat_panel["artifact_sha256"],
        "physical_sha256": repeat_panel_physical_sha256,
    }
    document["inputs"]["taskset_v44_predecessor"] = dict(predecessor["inputs"]["taskset"])
    document["inputs"]["repeat_panel_v44_predecessor"] = dict(predecessor["inputs"]["repeat_panel"])

    roster: list[dict[str, Any]] = []
    for candidate in candidates:
        prior = prior_rows[candidate.model_id]
        effort = str(prior["final_reasoning_effort"])
        if candidate.model_id == QWEN_MODEL_ID:
            effort = str(ROUTE_SPECS[QWEN_MODEL_ID]["reasoning_effort"])
        row = _roster_row(candidate, effort)
        maximum = prior.get("final_max_output_tokens")
        if candidate.model_id == QWEN_MODEL_ID:
            maximum = int(ROUTE_SPECS[QWEN_MODEL_ID]["max_output_tokens"])
        if maximum is not None:
            row["final_max_output_tokens"] = int(maximum)
        roster.append(row)
    document["roster"]["models"] = roster

    tasks_by_family = {
        family: sorted(
            str(task["task_id"]) for task in taskset["tasks"] if task["family"] == family
        )
        for family in FAMILIES
    }
    pilot_ids = [tasks_by_family[family][0] for family in FAMILIES]
    document["execution"]["pilot"] = {
        "cells": MODEL_COUNT * len(FAMILIES),
        "tasks_per_model": len(FAMILIES),
        "task_ids": pilot_ids,
        "selection_uses_scores_or_selections": False,
        "normal_completion_and_identity_only": True,
    }
    document["execution"].pop("anchor_free_successor", None)
    document["execution"]["replication_2"] = {
        "schema_version": "flavourbench-response-blind-replication-v1",
        "replication_index": 2,
        "primary_tasks_per_model": PRIMARY_TASKS,
        "repeat_tasks_per_model": REPEAT_TASKS,
        "provider_calls": MODEL_COUNT * (PRIMARY_TASKS + REPEAT_TASKS),
        "selection_is_response_blind": True,
        "first_panel_responses_reused": False,
        "first_panel_quality_scores_or_selections_used_in_design": False,
        "quality_score_definition": "successful_and_parseable_only",
        "failures_retained_for_coverage_and_excluded_from_quality": True,
        "shared_anchor_count": taskset["replication"]["shared_anchor_count"],
        "novel_anchor_count": taskset["replication"]["novel_anchor_count"],
        "joint_analysis_clusters_shared_anchors": True,
    }
    document["execution"]["reasoning_control"] = (
        "retain the v44 anchor-free scoring and decoding contract on the independently "
        "selected replication-2 tasks"
    )
    document["budget"].update(
        {
            "aggregate_program_cap": "700",
            "program_cap": "700",
            "hard_cap": "250",
            "successor_scope": "one complete 26-model replication-2 primary and repeat panel",
        }
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV46Error("constructed replication plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        replication = document["execution"]["replication_2"]
        policy_document = document["execution"]["execution_policy"]
        rows = {str(row["model_id"]): row for row in document["roster"]["models"]}
        qwen = rows[QWEN_MODEL_ID]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    qwen_spec = ROUTE_SPECS[QWEN_MODEL_ID]
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
        and replication.get("replication_index") == 2
        and replication.get("primary_tasks_per_model") == PRIMARY_TASKS
        and replication.get("repeat_tasks_per_model") == REPEAT_TASKS
        and replication.get("provider_calls") == MODEL_COUNT * (PRIMARY_TASKS + REPEAT_TASKS)
        and replication.get("selection_is_response_blind") is True
        and replication.get("first_panel_responses_reused") is False
        and replication.get("first_panel_quality_scores_or_selections_used_in_design") is False
        and replication.get("failures_retained_for_coverage_and_excluded_from_quality") is True
        and replication.get("shared_anchor_count") == 102
        and replication.get("novel_anchor_count") == 538
        and replication.get("joint_analysis_clusters_shared_anchors") is True
        and len(document["execution"]["pilot"].get("task_ids") or []) == len(FAMILIES)
        and document["execution"]["pilot"].get("cells") == MODEL_COUNT * len(FAMILIES)
        and document["outcomes"].get("failed_content_filtered_or_unparseable")
        == "excluded_from_quality_score"
        and document["outcomes"].get("dnf_classification") is False
        and document["outcomes"].get("minimum_coverage_for_score") is None
        and policy_document == policy.document()
        and verify_policy_document(policy_document)
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("hard_cap") == "250"
        and all(
            isinstance((document["inputs"].get(label) or {}).get("semantic_sha256"), str)
            and isinstance((document["inputs"].get(label) or {}).get("physical_sha256"), str)
            for label in (
                "route_manifest",
                "taskset",
                "repeat_panel",
                "plan_v44_predecessor",
                "taskset_v44_predecessor",
                "repeat_panel_v44_predecessor",
            )
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV46Error("content-addressed plan conflict")
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
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor)
    manifest = _load(args.manifest)
    taskset = _load(args.taskset)
    repeat_panel = _load(args.repeat_panel)
    document = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        taskset=taskset,
        taskset_physical_sha256=_sha256_file(args.taskset),
        repeat_panel=repeat_panel,
        repeat_panel_physical_sha256=_sha256_file(args.repeat_panel),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
