"""Build the final complete-response FlavourBench analysis release."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_selection_common_core_analysis_v1 import analyze_complete_common_core
from .epicure_selection_complete_core_plan_v84 import (
    CORE_FAMILIES,
    PRIMARY_TASKS,
    selected_task_ids,
    verify_plan,
)
from .epicure_selection_powered_analysis import (
    PanelData,
    SelectionPoweredAnalysisError,
    _sha256,
    _sha256_file,
)
from .epicure_selection_powered_joint_analysis_v1 import combine_panel_data

RELEASE_SCHEMA_VERSION = "flavourbench-complete-common-core-release-v1"


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def leaderboard_csv(analysis: Mapping[str, Any]) -> bytes:
    rows: list[dict[str, Any]] = []
    for model in analysis["models"]:
        coverage = model["coverage"]
        rows.append(
            {
                "rank": model["point_estimate_rank"],
                "rank_group": model["statistical_rank_group"],
                "rank_ci_low": model["bootstrap_rank_95_interval"][0],
                "rank_ci_high": model["bootstrap_rank_95_interval"][1],
                "model": model["model_name"],
                "model_id": model["model_id"],
                "flavourbench_score": f"{model['flavourbench_score']:.6f}",
                "pointwise_ci_low": f"{model['score_pointwise_95_ci'][0]:.6f}",
                "pointwise_ci_high": f"{model['score_pointwise_95_ci'][1]:.6f}",
                "simultaneous_ci_low": f"{model['score_simultaneous_95_ci'][0]:.6f}",
                "simultaneous_ci_high": f"{model['score_simultaneous_95_ci'][1]:.6f}",
                "valid_scored": coverage["valid_scored"],
                "scheduled": coverage["scheduled"],
                "coverage": f"{coverage['valid_scored_rate']:.6f}",
                "above_exact_chance_holm": str(
                    model["chance_comparison"]["holm_significant_above_chance"]
                ).lower(),
            }
        )
    rows.sort(key=lambda row: (row["rank"], row["model_id"]))
    return _csv_bytes(rows, tuple(rows[0]))


def pairwise_csv(analysis: Mapping[str, Any]) -> bytes:
    rows = [
        {
            "left_model_id": row["left_model_id"],
            "right_model_id": row["right_model_id"],
            "shared_valid_tasks": row["shared_valid_tasks"],
            "mean_difference": f"{row['mean_difference']:.6f}",
            "ci_low": f"{row['bootstrap_95_ci'][0]:.6f}",
            "ci_high": f"{row['bootstrap_95_ci'][1]:.6f}",
            "cohen_dz": "" if row["cohen_dz"] is None else f"{row['cohen_dz']:.8f}",
            "sign_flip_p": f"{row['sign_flip_p']:.10g}",
            "holm_p": f"{row['holm_p']:.10g}",
            "holm_significant": str(row["holm_significant"]).lower(),
            "direction": row["direction"],
        }
        for row in analysis["pairwise_comparisons"]
    ]
    return _csv_bytes(rows, tuple(rows[0]))


def build_release(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    panel_1: PanelData,
    panel_1_taskset: Mapping[str, Any],
    panel_1_taskset_path: Path,
    panel_2: PanelData,
    panel_2_taskset: Mapping[str, Any],
    panel_2_taskset_path: Path,
    bootstrap_resamples: int | None = None,
    permutation_resamples: int | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Analyze and package the exact task set named by a verified v84 plan."""

    if not verify_plan(plan):
        raise SelectionPoweredAnalysisError("complete-core release requires a valid v84 plan")
    expected_1, expected_2 = selected_task_ids(plan)
    if panel_1.task_ids != expected_1 or panel_2.task_ids != expected_2:
        raise SelectionPoweredAnalysisError("selected response task order differs from v84")
    if panel_1.model_ids != panel_2.model_ids:
        raise SelectionPoweredAnalysisError("selected panels have different model rosters")
    if len(panel_1.task_ids) + len(panel_2.task_ids) != PRIMARY_TASKS:
        raise SelectionPoweredAnalysisError("selected common-core task count differs")

    primary = combine_panel_data(panel_1, panel_2, panel="joint_primary_complete_core")
    task_by_id = {
        str(task["task_id"]): task
        for task in [*panel_1_taskset["tasks"], *panel_2_taskset["tasks"]]
    }
    selected_tasks = [task_by_id[task_id] for task_id in primary.task_ids]
    clusters = tuple(str(task["anchor_ingredient"]) for task in selected_tasks)
    panel_ids = tuple(["panel_1"] * len(panel_1.task_ids) + ["panel_2"] * len(panel_2.task_ids))
    analysis = analyze_complete_common_core(
        primary=primary,
        taskset={"tasks": selected_tasks},
        plan=plan,
        family_order=CORE_FAMILIES,
        cluster_ids=clusters,
        panel_ids=panel_ids,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
    )
    leaderboard = leaderboard_csv(analysis)
    pairwise = pairwise_csv(analysis)
    leaderboard_sha256 = _sha256_file_bytes(leaderboard)
    pairwise_sha256 = _sha256_file_bytes(pairwise)
    release: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "final_complete_common_core",
        "benchmark": "FlavourBench",
        "track": "Epicure-scored combinatorial culinary decisions",
        "claim_boundary": plan["claim_boundary"],
        "inputs": {
            "complete_core_plan": {
                "semantic_sha256": plan["artifact_sha256"],
                "physical_sha256": _sha256_file(plan_path),
            },
            "panel_1_taskset": {
                "semantic_sha256": panel_1_taskset["artifact_sha256"],
                "physical_sha256": _sha256_file(panel_1_taskset_path),
            },
            "panel_2_taskset": {
                "semantic_sha256": panel_2_taskset["artifact_sha256"],
                "physical_sha256": _sha256_file(panel_2_taskset_path),
            },
            "panel_1_responses": {
                "count": len(panel_1.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(panel_1.response_artifact_sha256s)),
            },
            "panel_2_responses": {
                "count": len(panel_2.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(panel_2.response_artifact_sha256s)),
            },
        },
        "design": plan["design"],
        "analysis": analysis,
        "tables": {
            "leaderboard": {
                "filename": f"flavourbench-complete-core-leaderboard-{leaderboard_sha256}.csv",
                "sha256": leaderboard_sha256,
            },
            "pairwise": {
                "filename": f"flavourbench-complete-core-pairwise-{pairwise_sha256}.csv",
                "sha256": pairwise_sha256,
            },
        },
        "all_ranked_models_have_identical_task_count": True,
        "failed_or_unparseable_cells_scored_as_zero": False,
        "dnf_rows_emitted": False,
    }
    release["artifact_sha256"] = _sha256(release)
    return release, leaderboard, pairwise


def _sha256_file_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_new(path: Path, value: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != value:
            raise SelectionPoweredAnalysisError(f"content-addressed output conflict: {path}")
        return
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
        path.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def write_release(
    *,
    release: Mapping[str, Any],
    leaderboard: bytes,
    pairwise: bytes,
    directory: Path,
) -> Path:
    """Publish the release and its two content-addressed tables without replacement."""

    payload = dict(release)
    recorded = str(payload.pop("artifact_sha256", ""))
    if not recorded or recorded != _sha256(payload):
        raise SelectionPoweredAnalysisError("complete-core release semantic hash differs")
    if _sha256_file_bytes(leaderboard) != release["tables"]["leaderboard"]["sha256"]:
        raise SelectionPoweredAnalysisError("leaderboard table hash differs")
    if _sha256_file_bytes(pairwise) != release["tables"]["pairwise"]["sha256"]:
        raise SelectionPoweredAnalysisError("pairwise table hash differs")
    directory.mkdir(parents=True, exist_ok=True)
    _write_new(directory / release["tables"]["leaderboard"]["filename"], leaderboard)
    _write_new(directory / release["tables"]["pairwise"]["filename"], pairwise)
    destination = directory / f"flavourbench-complete-core-release-{recorded}.json"
    data = (json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _write_new(destination, data)
    return destination
