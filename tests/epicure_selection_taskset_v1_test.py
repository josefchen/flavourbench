from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from flavourbench.epicure_selection_taskset_v1 import (
    ALL_SELECTION_KEYS,
    FAMILIES,
    score_answer,
    verify_taskset,
)

ROOT = Path(__file__).resolve().parents[1]
TASKSET_DIR = ROOT / "benchmark/powered-v16/taskset"


def _taskset() -> dict:
    path = next(TASKSET_DIR.glob("epicure-selection-taskset-*.json"))
    return json.loads(path.read_text())


def test_materialized_taskset_is_complete_and_exactly_scored() -> None:
    document = _taskset()
    assert verify_taskset(document)
    assert len(document["tasks"]) == 640
    assert Counter(task["family"] for task in document["tasks"]) == Counter(
        {family: 160 for family in FAMILIES}
    )
    assert all(
        set(task["selection_scores_bps"]) == set(ALL_SELECTION_KEYS) for task in document["tasks"]
    )
    assert all(
        sum(value == 10_000 for value in task["selection_scores_bps"].values()) == 1
        for task in document["tasks"]
    )


def test_scoring_is_continuous_and_fail_closed() -> None:
    task = _taskset()["tasks"][0]
    optimum = task["optimal_selection"]
    rendered = ",".join(optimum)
    assert score_answer(task, f"FINAL_SELECTION: {rendered}")["score_bps"] == 10_000
    assert score_answer(task, f"**FINAL_SELECTION: {rendered}**\nExplanation.")["score_bps"] == (
        10_000
    )
    assert score_answer(task, f"`FINAL_SELECTION: {rendered}`")["score_bps"] == 10_000
    assert score_answer(task, "FINAL_SELECTION: A,A,B")["score_bps"] == 0
    assert score_answer(task, "FINAL_SELECTION: C,B,A")["observed_selection"] == "ABC"
    assert score_answer(task, "The answer is A, B, C")["score_bps"] == 0


def test_calibration_is_excluded_and_chance_is_task_specific() -> None:
    document = _taskset()
    assert all(
        predecessor["used_as_primary_data"] is False
        for predecessor in document["calibration_predecessors"].values()
    )
    chance = [task["chance_score_bps"] for task in document["tasks"]]
    assert min(chance) >= 0
    assert max(chance) <= 10_000
    assert len(set(chance)) > 20


def test_cultural_prompts_use_neutral_operational_language() -> None:
    document = _taskset()
    cultural = [task for task in document["tasks"] if task["family"] == "cultural_composition"]
    assert len(cultural) == 160
    assert all("Target cuisine label:" in task["prompt"] for task in cultural)
    assert all("culturally coherent" not in task["prompt"] for task in cultural)
    assert all("Family: cultural composition" not in task["prompt"] for task in cultural)
    assert all("Family: regional cuisine selection" in task["prompt"] for task in cultural)
    assert document["prompt_revision"]["score_maps_changed"] is False
