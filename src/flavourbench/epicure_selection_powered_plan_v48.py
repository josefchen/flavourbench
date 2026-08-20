"""Freeze the response-blind two-panel, anchor-clustered analysis contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from .epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from .epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from .epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50
from .epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel as verify_repeat_panel_replication_2,
)
from .epicure_selection_repeat_panel_v2 import verify_repeat_panel
from .epicure_selection_taskset_replication_v1 import (
    verify_taskset as verify_taskset_replication_2,
)
from .epicure_selection_taskset_v1 import FAMILIES
from .epicure_selection_taskset_v2 import verify_taskset

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-plan-v48"
PLAN_VERSION = "flavourbench-selection-26x1280-two-panel-anchor-clustered-v48"
MODEL_COUNT = 26
PANEL_TASKS = 640
PANEL_REPEAT_TASKS = 64
TOTAL_TASKS = PANEL_TASKS * 2
TOTAL_REPEAT_TASKS = PANEL_REPEAT_TASKS * 2
PAIRWISE_HYPOTHESES = MODEL_COUNT * (MODEL_COUNT - 1) // 2


class SelectionPoweredPlanV48Error(RuntimeError):
    """The two-panel joint analysis plan failed verification."""


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
        raise SelectionPoweredPlanV48Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanV48Error("joint-plan input is not a JSON object")
    return value


def _pin(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "semantic_sha256": str(document["artifact_sha256"]),
        "physical_sha256": _sha256_file(path),
    }


def _task_projection(taskset: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        str(task["anchor_ingredient"]): (str(task["family"]), str(task["task_id"]))
        for task in taskset["tasks"]
    }


def build_plan(
    *,
    panel_1_plan: Mapping[str, Any],
    panel_1_plan_path: Path,
    panel_1_taskset: Mapping[str, Any],
    panel_1_taskset_path: Path,
    panel_1_repeat: Mapping[str, Any],
    panel_1_repeat_path: Path,
    panel_2_plan: Mapping[str, Any],
    panel_2_plan_path: Path,
    panel_2_taskset: Mapping[str, Any],
    panel_2_taskset_path: Path,
    panel_2_repeat: Mapping[str, Any],
    panel_2_repeat_path: Path,
) -> dict[str, Any]:
    if not (verify_plan_v47(panel_1_plan) or verify_plan_v50(panel_1_plan)) or not (
        verify_plan_v46(panel_2_plan) or verify_plan_v49(panel_2_plan)
    ):
        raise SelectionPoweredPlanV48Error("panel source plan failed verification")
    if not verify_taskset(panel_1_taskset) or not verify_repeat_panel(
        panel_1_repeat, taskset=panel_1_taskset
    ):
        raise SelectionPoweredPlanV48Error("panel 1 task inputs failed verification")
    if not verify_taskset_replication_2(panel_2_taskset) or not verify_repeat_panel_replication_2(
        panel_2_repeat, taskset=panel_2_taskset
    ):
        raise SelectionPoweredPlanV48Error("panel 2 task inputs failed verification")
    roster_1 = list(panel_1_plan["roster"]["models"])
    roster_2 = list(panel_2_plan["roster"]["models"])
    if roster_1 != roster_2 or len(roster_1) != MODEL_COUNT:
        raise SelectionPoweredPlanV48Error("joint panels do not have one exact roster")

    first = _task_projection(panel_1_taskset)
    second = _task_projection(panel_2_taskset)
    if len(first) != PANEL_TASKS or len(second) != PANEL_TASKS:
        raise SelectionPoweredPlanV48Error("each panel must use 640 unique anchors")
    shared = sorted(set(first) & set(second))
    same_family = sum(first[anchor][0] == second[anchor][0] for anchor in shared)
    family_counts = Counter(
        str(task["family"])
        for taskset in (panel_1_taskset, panel_2_taskset)
        for task in taskset["tasks"]
    )
    if family_counts != Counter({family: TOTAL_TASKS // len(FAMILIES) for family in FAMILIES}):
        raise SelectionPoweredPlanV48Error("joint tasks are not exactly family balanced")

    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "status": "joint_analysis_frozen_before_any_quality_score_inspection",
        "frozen_date": "2026-08-15",
        "inputs": {
            "panel_1_plan": _pin(panel_1_plan, panel_1_plan_path),
            "panel_1_taskset": _pin(panel_1_taskset, panel_1_taskset_path),
            "panel_1_repeat_panel": _pin(panel_1_repeat, panel_1_repeat_path),
            "panel_2_plan": _pin(panel_2_plan, panel_2_plan_path),
            "panel_2_taskset": _pin(panel_2_taskset, panel_2_taskset_path),
            "panel_2_repeat_panel": _pin(panel_2_repeat, panel_2_repeat_path),
        },
        "roster": {
            "model_count": MODEL_COUNT,
            "models": roster_1,
            "pairwise_hypotheses": PAIRWISE_HYPOTHESES,
        },
        "design": {
            "panel_count": 2,
            "primary_tasks_per_panel": PANEL_TASKS,
            "primary_model_task_cells": MODEL_COUNT * TOTAL_TASKS,
            "repeat_tasks_per_panel": PANEL_REPEAT_TASKS,
            "repeat_model_task_cells": MODEL_COUNT * TOTAL_REPEAT_TASKS,
            "scheduled_primary_tasks_per_model": TOTAL_TASKS,
            "scheduled_repeat_tasks_per_model": TOTAL_REPEAT_TASKS,
            "unique_anchor_clusters": len(set(first) | set(second)),
            "shared_anchor_clusters": len(shared),
            "same_family_shared_anchors": same_family,
            "cross_family_shared_anchors": len(shared) - same_family,
            "family_task_counts": dict(sorted(family_counts.items())),
            "independence_unit": "anchor_ingredient",
            "shared_anchors_count_once_as_clusters_not_twice_as_independent_tasks": True,
        },
        "outcomes": {
            "primary_name": "FlavourBench Score",
            "primary_definition": (
                "equal-family macro mean Epicure selection score over successful parseable "
                "responses across both frozen panels"
            ),
            "task_score": "prefrozen continuous lookup over all three-of-eight selections",
            "coverage_name": "Coverage",
            "coverage_definition": (
                "successful parseable responses divided by all scheduled responses"
            ),
            "failed_content_filtered_or_unparseable": "excluded_from_quality_score",
            "failed_cells_retained_in_coverage_and_provenance": True,
            "minimum_coverage_for_score": None,
            "dnf_classification": False,
            "chance": "exact task-specific mean across all 56 possible selections",
            "epicure_is_judge_not_a_competing_model": True,
        },
        "inference": {
            "seed": 20260815,
            "bootstrap_resamples": 50000,
            "permutation_resamples": 100000,
            "familywise_alpha": 0.05,
            "score_intervals": (
                "anchor-cluster bootstrap; all tasks sharing an anchor move together; "
                "the statistic remains an equal-family available-case macro mean"
            ),
            "paired_tests": (
                "all 325 two-sided anchor-cluster sign-flip tests on shared successful "
                "parseable tasks, with equal-family weights and Holm correction"
            ),
            "chance_tests": (
                "anchor-cluster sign flips of taskwise score minus exact chance, with Holm "
                "correction across 26 models"
            ),
            "score_uncertainty": "pointwise percentile and simultaneous max-t 95% intervals",
            "rank_display": "point ranks, bootstrap rank intervals, and statistical rank groups",
            "pairwise_missingness": "shared successful parseable tasks only",
            "no_result_dependent_test_selection": True,
            "panel_replication_summary": (
                "panel-specific scores and cross-panel Pearson/Spearman correlations are "
                "descriptive stability diagnostics"
            ),
        },
        "repeatability": {
            "repeat_tasks_per_panel": PANEL_REPEAT_TASKS,
            "total_repeat_tasks_per_model": TOTAL_REPEAT_TASKS,
            "primary_metric": "mean ingredient-set Jaccard similarity",
            "secondary_metrics": [
                "exact ingredient-set match rate",
                "mean absolute Epicure score difference",
            ],
            "acceptance_floor": 0.8,
            "anchor_clustered_across_panels": True,
        },
        "power": {
            "method": (
                "response-blind normal approximation using the conservative 1178 unique "
                "anchor-cluster count, paired SD 20, and Bonferroni alpha over 325 pairs"
            ),
            "unique_anchor_clusters": len(set(first) | set(second)),
            "assumed_paired_sd_points": 20,
            "familywise_comparisons": PAIRWISE_HYPOTHESES,
            "two_point_difference_power": 0.362202774294,
            "three_point_difference_power": 0.913641158605,
            "four_point_difference_power": 0.998963687602,
            "five_point_difference_power": 0.999999189557,
            "target": 0.8,
            "primary_target_difference_points": 3,
            "primary_target_meets_power": True,
            "final_observed_precision_and_missingness_sensitivity_required": True,
        },
        "source_rules": {
            "panel_1_uses_complete_qwen_replacement_block": True,
            "panel_2_is_a_fresh_response_collection": True,
            "cross_route_response_pooling": False,
            "selective_failed_cell_retry": False,
            "score_or_selection_inspected_before_freeze": False,
            "route_changes_require_complete_model_block_replacement": True,
        },
        "claim_boundary": (
            "Epicure alignment on two frozen combinatorial culinary panels; not universal "
            "model quality"
        ),
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanV48Error("constructed joint plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        design = document["design"]
        outcomes = document["outcomes"]
        inference = document["inference"]
        repeatability = document["repeatability"]
        power = document["power"]
        source = document["source_rules"]
        inputs = document["inputs"]
        roster = document["roster"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and document.get("plan_version") == PLAN_VERSION
        and recorded == _sha256(payload)
        and document.get("status") == "joint_analysis_frozen_before_any_quality_score_inspection"
        and roster.get("model_count") == MODEL_COUNT
        and len(roster.get("models") or []) == MODEL_COUNT
        and roster.get("pairwise_hypotheses") == PAIRWISE_HYPOTHESES
        and design.get("panel_count") == 2
        and design.get("primary_tasks_per_panel") == PANEL_TASKS
        and design.get("scheduled_primary_tasks_per_model") == TOTAL_TASKS
        and design.get("scheduled_repeat_tasks_per_model") == TOTAL_REPEAT_TASKS
        and design.get("unique_anchor_clusters") == 1178
        and design.get("shared_anchor_clusters") == 102
        and design.get("same_family_shared_anchors") == 35
        and design.get("cross_family_shared_anchors") == 67
        and design.get("family_task_counts")
        == {family: TOTAL_TASKS // len(FAMILIES) for family in FAMILIES}
        and design.get("independence_unit") == "anchor_ingredient"
        and design.get("shared_anchors_count_once_as_clusters_not_twice_as_independent_tasks")
        is True
        and outcomes.get("failed_content_filtered_or_unparseable") == "excluded_from_quality_score"
        and outcomes.get("failed_cells_retained_in_coverage_and_provenance") is True
        and outcomes.get("minimum_coverage_for_score") is None
        and outcomes.get("dnf_classification") is False
        and outcomes.get("epicure_is_judge_not_a_competing_model") is True
        and inference.get("bootstrap_resamples") == 50000
        and inference.get("permutation_resamples") == 100000
        and inference.get("familywise_alpha") == 0.05
        and inference.get("no_result_dependent_test_selection") is True
        and repeatability.get("total_repeat_tasks_per_model") == TOTAL_REPEAT_TASKS
        and repeatability.get("acceptance_floor") == 0.8
        and repeatability.get("anchor_clustered_across_panels") is True
        and power.get("unique_anchor_clusters") == 1178
        and power.get("familywise_comparisons") == PAIRWISE_HYPOTHESES
        and power.get("primary_target_difference_points") == 3
        and power.get("primary_target_meets_power") is True
        and source.get("panel_1_uses_complete_qwen_replacement_block") is True
        and source.get("panel_2_is_a_fresh_response_collection") is True
        and source.get("cross_route_response_pooling") is False
        and source.get("selective_failed_cell_retry") is False
        and source.get("score_or_selection_inspected_before_freeze") is False
        and source.get("route_changes_require_complete_model_block_replacement") is True
        and all(
            isinstance((inputs.get(label) or {}).get("semantic_sha256"), str)
            and len(inputs[label]["semantic_sha256"]) == 64
            and isinstance((inputs.get(label) or {}).get("physical_sha256"), str)
            and len(inputs[label]["physical_sha256"]) == 64
            for label in (
                "panel_1_plan",
                "panel_1_taskset",
                "panel_1_repeat_panel",
                "panel_2_plan",
                "panel_2_taskset",
                "panel_2_repeat_panel",
            )
        )
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = (
        directory / f"epicure-selection-joint-analysis-plan-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanV48Error("content-addressed joint-plan conflict")
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
    parser.add_argument("--panel-1-plan", type=Path, required=True)
    parser.add_argument("--panel-1-taskset", type=Path, required=True)
    parser.add_argument("--panel-1-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--panel-2-taskset", type=Path, required=True)
    parser.add_argument("--panel-2-repeat-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "panel_1_plan": args.panel_1_plan,
        "panel_1_taskset": args.panel_1_taskset,
        "panel_1_repeat": args.panel_1_repeat_panel,
        "panel_2_plan": args.panel_2_plan,
        "panel_2_taskset": args.panel_2_taskset,
        "panel_2_repeat": args.panel_2_repeat_panel,
    }
    documents = {label: _load(path) for label, path in paths.items()}
    document = build_plan(
        panel_1_plan=documents["panel_1_plan"],
        panel_1_plan_path=paths["panel_1_plan"],
        panel_1_taskset=documents["panel_1_taskset"],
        panel_1_taskset_path=paths["panel_1_taskset"],
        panel_1_repeat=documents["panel_1_repeat"],
        panel_1_repeat_path=paths["panel_1_repeat"],
        panel_2_plan=documents["panel_2_plan"],
        panel_2_plan_path=paths["panel_2_plan"],
        panel_2_taskset=documents["panel_2_taskset"],
        panel_2_taskset_path=paths["panel_2_taskset"],
        panel_2_repeat=documents["panel_2_repeat"],
        panel_2_repeat_path=paths["panel_2_repeat"],
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
