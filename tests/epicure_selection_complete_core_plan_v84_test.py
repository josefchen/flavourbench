from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flavourbench.epicure_selection_complete_core_plan_v84 import (
    CORE_FAMILIES,
    MODEL_COUNT,
    PRIMARY_TASKS,
    SelectionCompleteCorePlanV84Error,
    build_plan,
    selected_task_ids,
    verify_plan,
)
from flavourbench.epicure_selection_complete_core_release_v1 import build_release
from flavourbench.epicure_selection_powered_analysis import PanelData

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / (
    "benchmark/powered-v83/plan/epicure-selection-joint-analysis-plan-"
    "31f45aaf447b9337e07b9b27a75c9706bb6523efec3ad7e2738f76b9fc9d798b.json"
)
TASKSET_1 = ROOT / (
    "benchmark/powered-v44/taskset/epicure-selection-taskset-"
    "a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
TASKSET_2 = ROOT / (
    "benchmark/powered-v45/taskset/epicure-selection-taskset-"
    "925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
PARSER = ROOT / "src/flavourbench/selection_response_parser_v3.py"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel(taskset: dict[str, object]) -> PanelData:
    predecessor = _load(PREDECESSOR)
    roster = predecessor["roster"]["models"]
    task_ids = tuple(str(task["task_id"]) for task in taskset["tasks"])
    families = tuple(str(task["family"]) for task in taskset["tasks"])
    model_ids = tuple(str(row["model_id"]) for row in roster)
    shape = (len(model_ids), len(task_ids))
    artifacts = tuple(
        hashlib.sha256(f"{model_id}\0{task_id}".encode()).hexdigest()
        for model_id in model_ids
        for task_id in task_ids
    )
    return PanelData(
        panel="primary",
        model_ids=model_ids,
        model_names=tuple(str(row["model_name"]) for row in roster),
        slot_ids=tuple(str(row["slot_id"]) for row in roster),
        task_ids=task_ids,
        families=families,
        scores=np.zeros(shape, dtype=np.float64),
        completed=np.ones(shape, dtype=bool),
        parseable=np.ones(shape, dtype=bool),
        selections=tuple(tuple(None for _ in task_ids) for _ in model_ids),
        response_artifact_sha256s=artifacts,
        spend_micros=0,
    )


def _build(*, panel_1: PanelData | None = None) -> dict[str, object]:
    taskset_1 = _load(TASKSET_1)
    taskset_2 = _load(TASKSET_2)
    return build_plan(
        predecessor=_load(PREDECESSOR),
        predecessor_path=PREDECESSOR,
        panel_1_data=panel_1 or _panel(taskset_1),
        panel_1_taskset=taskset_1,
        panel_1_taskset_path=TASKSET_1,
        panel_2_data=_panel(taskset_2),
        panel_2_taskset=taskset_2,
        panel_2_taskset_path=TASKSET_2,
        parser_path=PARSER,
    )


def test_v84_freezes_balanced_complete_core_without_scores() -> None:
    document = _build()
    assert verify_plan(document)
    left, right = selected_task_ids(document)
    assert len(left) == len(right) == PRIMARY_TASKS // 2
    assert len(set(left)) == len(left)
    assert len(set(right)) == len(right)
    assert document["common_core"]["ranked_models"] == MODEL_COUNT
    assert document["common_core"]["included_families"] == list(CORE_FAMILIES)
    assert document["common_core"]["model_task_cells"] == MODEL_COUNT * PRIMARY_TASKS
    assert (
        document["common_core"]["quality_scores_or_observed_selections_used_for_task_selection"]
        is False
    )
    assert document["eligibility"]["missing_or_invalid_ranked_cells"] == 0


def test_v84_rejects_insufficient_all_model_intersection_and_claim_drift() -> None:
    taskset = _load(TASKSET_1)
    panel = _panel(taskset)
    pairing = [index for index, family in enumerate(panel.families) if family == "pairing"]
    panel.parseable[0, pairing[:72]] = False
    with pytest.raises(SelectionCompleteCorePlanV84Error, match="complete pairing tasks"):
        _build(panel_1=panel)

    document = _build()
    drifted = copy.deepcopy(document)
    drifted["common_core"]["failures_and_unparseable_responses_scored_as_zero"] = True
    assert not verify_plan(drifted)


def _selected_panel(data: PanelData, task_ids: tuple[str, ...]) -> PanelData:
    indices = [data.task_ids.index(task_id) for task_id in task_ids]
    scores = np.empty((len(data.model_ids), len(indices)), dtype=np.float64)
    for model_index in range(len(data.model_ids)):
        scores[model_index] = (
            np.arange(len(indices), dtype=np.float64) * 7 + model_index * 13
        ) % 101
    artifacts = tuple(
        hashlib.sha256(f"selected\0{model_id}\0{task_id}".encode()).hexdigest()
        for model_id in data.model_ids
        for task_id in task_ids
    )
    return PanelData(
        panel="primary",
        model_ids=data.model_ids,
        model_names=data.model_names,
        slot_ids=data.slot_ids,
        task_ids=task_ids,
        families=tuple(data.families[index] for index in indices),
        scores=scores,
        completed=np.ones_like(scores, dtype=bool),
        parseable=np.ones_like(scores, dtype=bool),
        selections=tuple(tuple("ABC" for _ in indices) for _ in data.model_ids),
        response_artifact_sha256s=artifacts,
        spend_micros=0,
    )


def test_complete_core_release_emits_27_ranked_rows_and_351_pairs(tmp_path: Path) -> None:
    taskset_1 = _load(TASKSET_1)
    taskset_2 = _load(TASKSET_2)
    full_1 = _panel(taskset_1)
    full_2 = _panel(taskset_2)
    plan = _build(panel_1=full_1)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    task_ids_1, task_ids_2 = selected_task_ids(plan)
    release, leaderboard, pairwise = build_release(
        plan=plan,
        plan_path=plan_path,
        panel_1=_selected_panel(full_1, task_ids_1),
        panel_1_taskset=taskset_1,
        panel_1_taskset_path=TASKSET_1,
        panel_2=_selected_panel(full_2, task_ids_2),
        panel_2_taskset=taskset_2,
        panel_2_taskset_path=TASKSET_2,
        bootstrap_resamples=50,
        permutation_resamples=100,
    )
    assert len(release["analysis"]["models"]) == MODEL_COUNT
    assert len(release["analysis"]["pairwise_comparisons"]) == 351
    assert len(leaderboard.decode().splitlines()) == MODEL_COUNT + 1
    assert len(pairwise.decode().splitlines()) == 352
    assert release["failed_or_unparseable_cells_scored_as_zero"] is False
