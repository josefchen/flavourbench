from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_plan_v2 import (
    build_repeat_panel,
    calibration_commitment,
    verify_plan,
    verify_repeat_panel,
)
from flavourbench.epicure_native_taskset_v3 import verify_taskset

ROOT = Path(__file__).resolve().parents[1]
TASKSET = next((ROOT / "benchmark/powered-v3/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v3/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v3/plan").glob("*analysis-plan*.json"))


def test_materialized_successor_plan_and_repeat_panel() -> None:
    taskset = json.loads(TASKSET.read_text())
    repeat = json.loads(REPEAT.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_taskset(taskset)
    assert verify_repeat_panel(repeat, taskset=taskset)
    assert verify_plan(plan)
    assert len(plan["execution"]["pilot"]["task_ids"]) == 4
    assert plan["execution"]["pilot"]["cells"] == 80
    assert plan["power"]["bonferroni_all_190"] > 0.83


def test_repeat_panel_is_deterministic_and_permuted() -> None:
    taskset = json.loads(TASKSET.read_text())
    first = build_repeat_panel(taskset)
    assert first == build_repeat_panel(taskset)
    assert all(task["permutation_shift"] in {1, 2, 3} for task in first["tasks"])


def test_calibration_commitment_binds_exact_twenty_cells() -> None:
    commitment = calibration_commitment(ROOT / "benchmark/powered-v2/run")
    assert commitment["response_count"] == 20
    assert commitment["used_as_primary_data"] is False
