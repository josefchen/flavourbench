from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner_v2 import build_cells, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(
    ROOT / "benchmark/powered-v2/manifest/"
    "flavourbench-frontier-refresh-20-"
    "44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc.json"
)
MANIFEST_SHA = "44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc"
TASKSET = next((ROOT / "benchmark/powered-v3/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v3/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v3/plan").glob("*analysis-plan*.json"))
PREDECESSOR = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def _validated():
    return validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )


def test_exact_successor_inputs_validate() -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = _validated()
    assert manifest["content_address"]["digest"] == MANIFEST_SHA
    assert taskset["artifact_sha256"] == plan["inputs"]["taskset"]["semantic_sha256"]
    assert repeat["artifact_sha256"] == plan["inputs"]["repeat_panel"]["semantic_sha256"]
    assert (
        predecessor["artifact_sha256"] == plan["inputs"]["predecessor_release"]["semantic_sha256"]
    )
    assert len(candidates) == 20


def test_schedule_is_complete_unique_balanced_and_pilot_is_reused() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    schedules = {
        phase: build_cells(
            plan=plan,
            taskset=taskset,
            repeat_panel=repeat,
            candidates=candidates,
            phase=phase,
        )
        for phase in ("pilot", "primary", "repeat", "all")
    }
    assert len(schedules["pilot"]) == 80
    assert len(schedules["primary"]) == 12_800
    assert len(schedules["repeat"]) == 1_280
    assert len(schedules["all"]) == 14_080
    assert {cell.cell_id for cell in schedules["pilot"]} <= {
        cell.cell_id for cell in schedules["primary"]
    }
    assert len({cell.cell_id for cell in schedules["all"]}) == 14_080
    assert len({cell.task["task_id"] for cell in schedules["pilot"]}) == 4
    assert {cell.task["family"] for cell in schedules["pilot"]} == {
        "substitution",
        "pairing",
        "constraint",
        "provenance",
    }


def test_plan_calibration_is_not_primary_data() -> None:
    plan = json.loads(PLAN.read_text())
    calibration = plan["inputs"]["calibration_predecessor"]
    assert calibration["response_count"] == 20
    assert calibration["used_as_primary_data"] is False
