from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from flavourbench.epicure_selection_powered_analysis import PanelData
from flavourbench.epicure_selection_powered_analysis_v2 import analyze_panels
from flavourbench.epicure_selection_powered_plan_v44 import (
    PLAN_SCHEMA_VERSION,
    selection_execution_policy_v44,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_repeat_panel_v2 import verify_repeat_panel
from flavourbench.epicure_selection_taskset_v1 import verify_taskset as verify_taskset_v1
from flavourbench.epicure_selection_taskset_v2 import (
    CONCRETE_SELECTION_EXAMPLE,
    verify_taskset,
)
from flavourbench.frontier_contract_runner import load_candidate_manifest, select_candidates
from flavourbench.provider import (
    ANCHOR_FREE_SELECTION_TEXT_FINAL_INSTRUCTION,
    system_prompt_text,
)

ROOT = Path(__file__).resolve().parents[1]
TASKSET_V1 = ROOT / (
    "benchmark/powered-v16/taskset/"
    "epicure-selection-taskset-99932ef2e72d34b61641270670e6d56233f167273a8877d03ac214115c084ff7.json"
)
REPEAT_V1 = ROOT / (
    "benchmark/powered-v17/plan/"
    "epicure-selection-repeat-panel-c3829d4cdb7039f920411c6edde13691237f42cafc20e463ac326a06895c97fb.json"
)
TASKSET_V2 = ROOT / (
    "benchmark/powered-v44/taskset/"
    "epicure-selection-taskset-a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
REPEAT_V2 = ROOT / (
    "benchmark/powered-v44/plan/"
    "epicure-selection-repeat-panel-96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
)
PLAN_V44 = ROOT / (
    "benchmark/powered-v44/plan/"
    "epicure-selection-analysis-plan-dd74a82d4a34500f22ed91178f63497486fd957e67ebc0136bfa3350d3f6d57e.json"
)
PLAN_V43 = ROOT / (
    "benchmark/powered-v43/plan/"
    "epicure-selection-analysis-plan-6385f46243b34f9f5e8211fa765aacdb2ee2c51690b369ac00cf4350d38e47f4.json"
)
MANIFEST = ROOT / (
    "benchmark/powered-v43/manifest/"
    "flavourbench-frontier-refresh-26-33796dd9a0a4580f15fa79ec9cd50179c2b6ddc7c120c03f1814faf8259f8e9d.json"
)
RELEASE = ROOT / (
    "paper/generated/powered/"
    "flavourbench-powered-release-7aeddf27998b0a8ed0b961cab035e4793305ea120f73ce8f3baa47e4db612cf7.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_anchor_free_task_and_repeat_successors_preserve_scoring_geometry() -> None:
    old_taskset = _load(TASKSET_V1)
    old_repeat = _load(REPEAT_V1)
    taskset = _load(TASKSET_V2)
    repeat = _load(REPEAT_V2)
    assert verify_taskset_v1(old_taskset)
    assert verify_taskset(taskset, predecessor=old_taskset)
    assert verify_repeat_panel(
        repeat,
        taskset=taskset,
        predecessor=old_repeat,
        predecessor_taskset=old_taskset,
    )
    assert (
        sum(
            bool(CONCRETE_SELECTION_EXAMPLE.search(task["prompt"])) for task in old_taskset["tasks"]
        )
        == 640
    )
    assert not any(CONCRETE_SELECTION_EXAMPLE.search(task["prompt"]) for task in taskset["tasks"])
    assert not any(CONCRETE_SELECTION_EXAMPLE.search(task["prompt"]) for task in repeat["tasks"])
    old_by_id = {task["task_id"]: task for task in old_taskset["tasks"]}
    for task in taskset["tasks"]:
        old = old_by_id[task["task_id"]]
        assert task["choices"] == old["choices"]
        assert task["selection_scores_bps"] == old["selection_scores_bps"]
        assert task["optimal_selection"] == old["optimal_selection"]
        assert task["oracle_reference_sha256"] == old["oracle_reference_sha256"]
        assert task["prompt_sha256"] != old["prompt_sha256"]


def test_v44_protocol_has_no_concrete_answer_in_any_prompt_layer() -> None:
    policy = selection_execution_policy_v44()
    system = system_prompt_text(
        "epicure_off",
        final_response_mode=policy.final_response_mode,
        evidence_protocol=policy.evidence_protocol,
    )
    assert policy.evidence_protocol == "selection_text_v2_anchor_free"
    assert policy.document()["schema_version"] == "flavourbench-real-execution-policy-v12"
    assert not CONCRETE_SELECTION_EXAMPLE.search(system)
    assert not CONCRETE_SELECTION_EXAMPLE.search(ANCHOR_FREE_SELECTION_TEXT_FINAL_INSTRUCTION)


def test_v44_plan_removes_dnf_and_requires_a_complete_fresh_panel() -> None:
    plan = _load(PLAN_V44)
    assert verify_plan(plan)
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert "eligibility" not in plan
    assert plan["outcomes"]["dnf_classification"] is False
    assert plan["outcomes"]["minimum_coverage_for_score"] is None
    assert (
        plan["outcomes"]["failed_content_filtered_or_unparseable"] == "excluded_from_quality_score"
    )
    successor = plan["execution"]["anchor_free_successor"]
    assert len(successor["rerun_model_ids"]) == 26
    assert successor["provider_calls"] == 18_304
    assert successor["reuse_any_predecessor_response"] is False
    assert successor["concrete_answer_examples_in_v44_prompts"] == 0


def test_v44_inputs_and_schedule_replay_exactly() -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256="33796dd9a0a4580f15fa79ec9cd50179c2b6ddc7c120c03f1814faf8259f8e9d",
        taskset_path=TASKSET_V2,
        repeat_panel_path=REPEAT_V2,
        plan_path=PLAN_V44,
        predecessor_release_path=RELEASE,
    )
    assert (
        predecessor["artifact_sha256"]
        == "7aeddf27998b0a8ed0b961cab035e4793305ea120f73ce8f3baa47e4db612cf7"
    )
    assert (
        manifest["content_address"]["digest"] == plan["inputs"]["route_manifest"]["semantic_sha256"]
    )
    assert taskset["artifact_sha256"] == plan["inputs"]["taskset"]["semantic_sha256"]
    assert repeat["artifact_sha256"] == plan["inputs"]["repeat_panel"]["semantic_sha256"]
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="all",
    )
    assert len(cells) == 18_304
    assert len({cell.cell_id for cell in cells}) == 18_304
    assert all(not CONCRETE_SELECTION_EXAMPLE.search(cell.task["prompt"]) for cell in cells)


def test_manifest_and_artifact_files_are_physically_bound() -> None:
    manifest = load_candidate_manifest(
        MANIFEST,
        expected_digest="33796dd9a0a4580f15fa79ec9cd50179c2b6ddc7c120c03f1814faf8259f8e9d",
    )
    assert len(select_candidates(manifest)) == 26
    plan = _load(PLAN_V44)
    assert (
        hashlib.sha256(TASKSET_V2.read_bytes()).hexdigest()
        == plan["inputs"]["taskset"]["physical_sha256"]
    )
    assert (
        hashlib.sha256(REPEAT_V2.read_bytes()).hexdigest()
        == plan["inputs"]["repeat_panel"]["physical_sha256"]
    )
    assert (
        hashlib.sha256(PLAN_V43.read_bytes()).hexdigest()
        == plan["inputs"]["plan_v43_predecessor"]["physical_sha256"]
    )


def test_success_only_score_excludes_failed_and_unparseable_cells_without_dnf() -> None:
    families = (
        "substitution",
        "substitution",
        "pairing",
        "pairing",
        "constraint",
        "constraint",
        "cultural_composition",
        "cultural_composition",
    )
    scores = np.asarray(
        [
            [80, 80, 80, 80, 80, 80, 80, 80],
            [70, 0, 70, 0, 70, 0, 70, 0],
            [40, 40, 40, 40, 40, 40, 40, 40],
        ],
        dtype=float,
    )
    completed = np.asarray(
        [
            [True] * 8,
            [True, False, True, False, True, False, True, False],
            [True] * 8,
        ]
    )
    parseable = completed.copy()
    primary = PanelData(
        panel="primary",
        model_ids=("high", "partial", "low"),
        model_names=("High", "Partial", "Low"),
        slot_ids=("a", "b", "c"),
        task_ids=tuple(f"task-{index}" for index in range(8)),
        families=families,
        scores=scores,
        completed=completed,
        parseable=parseable,
        selections=tuple(tuple("ABC" if value else None for value in row) for row in completed),
        response_artifact_sha256s=(),
        spend_micros=0,
    )
    taskset = {
        "tasks": [
            {
                "task_id": f"task-{index}",
                "chance_score_bps": 3000,
            }
            for index in range(8)
        ]
    }
    plan = {
        "artifact_sha256": "0" * 64,
        "outcomes": {
            "primary_definition": "equal-family success-only score",
        },
        "inference": {
            "bootstrap_resamples": 200,
            "permutation_resamples": 200,
            "seed": 7,
        },
        "repeatability": {"acceptance_floor": 0.8},
    }
    analysis = analyze_panels(primary=primary, taskset=taskset, plan=plan)
    rows = {row["model_id"]: row for row in analysis["models"]}
    assert rows["high"]["flavourbench_score"] == 80
    assert rows["partial"]["flavourbench_score"] == 70
    assert rows["low"]["flavourbench_score"] == 40
    assert rows["partial"]["coverage"]["valid_scored"] == 4
    assert rows["partial"]["coverage"]["valid_scored_rate"] == 0.5
    assert rows["partial"]["score_status"] == "scored"
    assert rows["high"]["failure_exclusion_sensitivity"]["scheduled_panel_worst_best_bounds"] == [
        80,
        80,
    ]
    assert rows["partial"]["failure_exclusion_sensitivity"][
        "scheduled_panel_worst_best_bounds"
    ] == [35, 85]
    assert (
        rows["partial"]["failure_exclusion_sensitivity"]["primary_score_uses_either_endpoint"]
        is False
    )
    assert analysis["dnf_classification"] is False
    partial_vs_low = next(
        row
        for row in analysis["pairwise_comparisons"]
        if {row["left_model_id"], row["right_model_id"]} == {"partial", "low"}
    )
    assert partial_vs_low["shared_valid_tasks"] == 4
