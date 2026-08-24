"""Freeze the full anchor-free 26-model FlavourBench rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v43 import verify_plan as verify_plan_v43
from .epicure_selection_repeat_panel_v2 import verify_repeat_panel
from .epicure_selection_route_manifest_v43 import verify_manifest as verify_manifest_v43
from .epicure_selection_taskset_v1 import verify_taskset as verify_predecessor_taskset
from .epicure_selection_taskset_v2 import verify_taskset
from .execution_policy import (
    SELECTION_TEXT_PROTOCOL_V2,
    ExecutionPolicy,
    verify_policy_document,
)
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .frontier_refresh_26_v37 import FINAL_PANEL_ORDER

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v44"
PLAN_VERSION = "flavourbench-selection-26x640-v44-anchor-free"
PRIMARY_TASKS = 640
REPEAT_TASKS = 64
MODEL_COUNT = 26
TOTAL_CALLS = MODEL_COUNT * (PRIMARY_TASKS + REPEAT_TASKS)
PROGRAM_CAP_MICROS = 200_000_000
PREDECESSOR_OBSERVED_SPEND_MICROS = 53_305_115


class SelectionPoweredPlanV44Error(RuntimeError):
    """The full anchor-free successor plan failed verification."""


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
        raise SelectionPoweredPlanV44Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV44Error("plan input is not a JSON object")
    return value


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def selection_execution_policy_v44() -> ExecutionPolicy:
    """Retain v43 decoding/limits while removing the concrete answer exemplar."""

    return replace(
        selection_execution_policy_v31(),
        evidence_protocol=SELECTION_TEXT_PROTOCOL_V2,
    )


def _release_commitment(
    release: Mapping[str, Any], *, physical_sha256: str, predecessor_plan: Mapping[str, Any]
) -> dict[str, Any]:
    if not _semantic_valid(release):
        raise SelectionPoweredPlanV44Error("predecessor release failed semantic verification")
    inputs = release.get("inputs") or {}
    plan = inputs.get("plan") or {}
    primary = inputs.get("primary_responses") or {}
    repeat = inputs.get("repeat_responses") or {}
    spend = int(primary.get("spend_micros", -1)) + int(repeat.get("spend_micros", -1))
    if (
        plan.get("semantic_sha256") != predecessor_plan.get("artifact_sha256")
        or primary.get("count") != MODEL_COUNT * PRIMARY_TASKS
        or repeat.get("count") != MODEL_COUNT * REPEAT_TASKS
        or spend != PREDECESSOR_OBSERVED_SPEND_MICROS
    ):
        raise SelectionPoweredPlanV44Error("predecessor release boundary changed")
    return {
        "semantic_sha256": release["artifact_sha256"],
        "physical_sha256": physical_sha256,
        "primary_response_count": primary["count"],
        "repeat_response_count": repeat["count"],
        "observed_spend_micros": spend,
        "quality_scores_reusable": False,
        "reason": "all responses used prompts containing one concrete valid answer exemplar",
    }


def build_plan(
    *,
    predecessor: Mapping[str, Any],
    predecessor_physical_sha256: str,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    predecessor_taskset: Mapping[str, Any],
    predecessor_taskset_physical_sha256: str,
    taskset: Mapping[str, Any],
    taskset_physical_sha256: str,
    predecessor_repeat: Mapping[str, Any],
    predecessor_repeat_physical_sha256: str,
    repeat: Mapping[str, Any],
    repeat_physical_sha256: str,
    predecessor_release: Mapping[str, Any],
    predecessor_release_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_plan_v43(predecessor) or not verify_manifest_v43(manifest):
        raise SelectionPoweredPlanV44Error("v44 requires exact v43 plan and route manifest")
    if not verify_predecessor_taskset(predecessor_taskset):
        raise SelectionPoweredPlanV44Error("v44 predecessor task set failed verification")
    if not verify_taskset(taskset, predecessor=predecessor_taskset):
        raise SelectionPoweredPlanV44Error("v44 anchor-free task set failed verification")
    if not verify_repeat_panel(
        repeat,
        taskset=taskset,
        predecessor=predecessor_repeat,
        predecessor_taskset=predecessor_taskset,
    ):
        raise SelectionPoweredPlanV44Error("v44 anchor-free repeat panel failed verification")
    candidates = select_candidates(manifest)
    if tuple(candidate.model_id for candidate in candidates) != FINAL_PANEL_ORDER:
        raise SelectionPoweredPlanV44Error("v44 route roster/order changed")
    release = _release_commitment(
        predecessor_release,
        physical_sha256=predecessor_release_physical_sha256,
        predecessor_plan=predecessor,
    )
    if (
        predecessor["inputs"]["taskset"]["semantic_sha256"]
        != predecessor_taskset["artifact_sha256"]
        or predecessor["inputs"]["repeat_panel"]["semantic_sha256"]
        != predecessor_repeat["artifact_sha256"]
    ):
        raise SelectionPoweredPlanV44Error("v43 prompt-bearing input lineage changed")

    document = json.loads(json.dumps(predecessor))
    document.pop("artifact_sha256")
    document["schema_version"] = PLAN_SCHEMA_VERSION
    document["plan_version"] = PLAN_VERSION
    document["status"] = "frozen_before_full_anchor_free_26_model_rerun"
    document["frozen_date"] = "2026-08-15"
    document["inputs"]["plan_v43_predecessor"] = {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": predecessor_physical_sha256,
    }
    document["inputs"]["route_manifest"] = {
        "semantic_sha256": manifest["content_address"]["digest"],
        "physical_sha256": manifest_physical_sha256,
    }
    document["inputs"]["predecessor_taskset"] = {
        "semantic_sha256": predecessor_taskset["artifact_sha256"],
        "physical_sha256": predecessor_taskset_physical_sha256,
    }
    document["inputs"]["taskset"] = {
        "semantic_sha256": taskset["artifact_sha256"],
        "task_set_sha256": taskset["task_set_sha256"],
        "physical_sha256": taskset_physical_sha256,
    }
    document["inputs"]["predecessor_repeat_panel"] = {
        "semantic_sha256": predecessor_repeat["artifact_sha256"],
        "physical_sha256": predecessor_repeat_physical_sha256,
    }
    document["inputs"]["repeat_panel"] = {
        "semantic_sha256": repeat["artifact_sha256"],
        "physical_sha256": repeat_physical_sha256,
    }
    document["inputs"]["superseded_anchor_contaminated_release"] = release

    document["outcomes"] = {
        "primary_name": "FlavourBench Score",
        "primary_definition": (
            "equal-family macro mean Epicure selection score over successful parseable responses"
        ),
        "task_score": "prefrozen continuous lookup over all three-of-eight selections",
        "chance": "exact task-specific mean across all 56 possible selections",
        "failed_content_filtered_or_unparseable": "excluded_from_quality_score",
        "coverage_name": "Coverage",
        "coverage_definition": "successful parseable responses divided by scheduled responses",
        "coverage_reported_for_every_model": True,
        "dnf_classification": False,
        "minimum_coverage_for_score": None,
        "epicure_is_judge_not_a_competing_model": True,
    }
    document.pop("eligibility", None)
    document["inference"].update(
        {
            "score_intervals": (
                "family-stratified task bootstrap over successful parseable responses"
            ),
            "paired_tests": (
                "all 325 two-sided paired sign-flip tests on shared successful parseable tasks "
                "with equal-family weights and Holm correction"
            ),
            "rank_display": "point ranks plus simultaneous uncertainty; no DNF category",
            "missing_response_assumption": (
                "quality conditional on a successful parseable response; coverage is a separate "
                "reported endpoint"
            ),
        }
    )
    policy = selection_execution_policy_v44()
    document["execution"]["execution_policy"] = policy.document()
    document["execution"]["execution_policy_sha256"] = policy.sha256
    document["execution"]["response_contract"] = (
        "FINAL_SELECTION marker plus three distinct comma-separated A-H labels; no concrete "
        "answer exemplar appears in any prompt layer"
    )
    document["execution"]["anchor_free_successor"] = {
        "prompt_protocol": SELECTION_TEXT_PROTOCOL_V2,
        "rerun_model_ids": list(FINAL_PANEL_ORDER),
        "primary_cells_per_model": PRIMARY_TASKS,
        "repeat_cells_per_model": REPEAT_TASKS,
        "provider_calls": TOTAL_CALLS,
        "reuse_any_predecessor_response": False,
        "selective_retry": False,
        "score_or_result_adaptive_selection": False,
        "pilot_is_score_blind": True,
        "pilot_responses_reused_within_v44_primary": True,
        "old_literal_exemplar": "A,C,F",
        "concrete_answer_examples_in_v44_prompts": 0,
    }
    document["execution"]["reasoning_control"] = (
        "retain the exact v43 roster, routes, task choices, Epicure score maps, decoding, "
        "reasoning controls, and concurrency; remove only the concrete answer exemplar and rerun "
        "every primary and repeat cell"
    )
    document["budget"]["prior_observed_program_spend_micros"] = release["observed_spend_micros"]
    document["budget"]["hard_cap"] = (
        f"{(PROGRAM_CAP_MICROS - PREDECESSOR_OBSERVED_SPEND_MICROS) / 1_000_000:.6f}"
    )
    document["budget"]["successor_scope"] = (
        "one fresh anchor-free 26-model panel: 16640 primary plus 1664 repeat calls"
    )
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV44Error("constructed v44 plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        taskset = document["inputs"]["taskset"]
        repeat = document["inputs"]["repeat_panel"]
        release = document["inputs"]["superseded_anchor_contaminated_release"]
        successor = document["execution"]["anchor_free_successor"]
        policy_document = document["execution"]["execution_policy"]
        outcomes = document["outcomes"]
    except (KeyError, TypeError):
        return False
    policy = selection_execution_policy_v44()
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document["roster"].get("model_count") == MODEL_COUNT
        and tuple(row["model_id"] for row in roster) == FINAL_PANEL_ORDER
        and document["design"].get("primary_provider_calls") == MODEL_COUNT * PRIMARY_TASKS
        and document["design"].get("repeat_provider_calls") == MODEL_COUNT * REPEAT_TASKS
        and taskset.get("semantic_sha256") == document["inputs"]["taskset"].get("semantic_sha256")
        and isinstance(taskset.get("physical_sha256"), str)
        and isinstance(repeat.get("physical_sha256"), str)
        and release.get("observed_spend_micros") == PREDECESSOR_OBSERVED_SPEND_MICROS
        and release.get("quality_scores_reusable") is False
        and outcomes.get("failed_content_filtered_or_unparseable") == "excluded_from_quality_score"
        and outcomes.get("coverage_reported_for_every_model") is True
        and outcomes.get("dnf_classification") is False
        and outcomes.get("minimum_coverage_for_score") is None
        and "eligibility" not in document
        and successor.get("prompt_protocol") == SELECTION_TEXT_PROTOCOL_V2
        and successor.get("rerun_model_ids") == list(FINAL_PANEL_ORDER)
        and successor.get("provider_calls") == TOTAL_CALLS
        and successor.get("reuse_any_predecessor_response") is False
        and successor.get("selective_retry") is False
        and successor.get("concrete_answer_examples_in_v44_prompts") == 0
        and verify_policy_document(policy_document)
        and policy_document == policy.document()
        and document["execution"].get("execution_policy_sha256") == policy.sha256
        and document["budget"].get("aggregate_program_cap") == "200"
        and document["budget"].get("hard_cap") == "146.694885"
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-analysis-plan-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV44Error("content-addressed plan conflict")
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
    parser.add_argument("--predecessor-taskset", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--predecessor-repeat-panel", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    predecessor = _load(args.predecessor_plan)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    predecessor_taskset = _load(args.predecessor_taskset)
    taskset = _load(args.taskset)
    predecessor_repeat = _load(args.predecessor_repeat_panel)
    repeat = _load(args.repeat_panel)
    predecessor_release = _load(args.predecessor_release)
    document = build_plan(
        predecessor=predecessor,
        predecessor_physical_sha256=_sha256_file(args.predecessor_plan),
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        predecessor_taskset=predecessor_taskset,
        predecessor_taskset_physical_sha256=_sha256_file(args.predecessor_taskset),
        taskset=taskset,
        taskset_physical_sha256=_sha256_file(args.taskset),
        predecessor_repeat=predecessor_repeat,
        predecessor_repeat_physical_sha256=_sha256_file(args.predecessor_repeat_panel),
        repeat=repeat,
        repeat_physical_sha256=_sha256_file(args.repeat_panel),
        predecessor_release=predecessor_release,
        predecessor_release_physical_sha256=_sha256_file(args.predecessor_release),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
