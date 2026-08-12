from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v21 import (
    NO_REASONING_MODEL_IDS,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v19/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v21/plan").glob("*analysis-plan*.json"))
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


def test_v21_plan_binds_v20_calibration_and_exact_disabled_routes() -> None:
    plan = json.loads(PLAN.read_text())
    assert verify_plan(plan)
    assert set(plan["execution"]["reasoning_disabled_model_ids"]) == NO_REASONING_MODEL_IDS
    assert plan["inputs"]["calibration_v20"]["response_count"] == 579
    assert plan["inputs"]["calibration_v20"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v20"]["captured_responses_remain_calibration_only"]


def test_v21_disabled_routes_build_reasoning_none_requests() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    for candidate in candidates:
        if candidate.model_id not in NO_REASONING_MODEL_IDS:
            continue
        cell = build_cells(
            plan=plan,
            taskset=taskset,
            repeat_panel=repeat,
            candidates=[candidate],
            phase="pilot",
        )[0]
        spec, bundle = build_generation_spec(
            cell=cell,
            plan=plan,
            manifest_sha256=MANIFEST_SHA,
            taskset=taskset,
            reserve_micros=100_000,
            execution_policy=selection_execution_policy(),
        )
        assert spec.final_reasoning_effort == "none"
        assert bundle["run_binding"]["final_reasoning_effort"] == "none"


def test_v21_other_reasoning_routes_remain_minimal() -> None:
    plan = json.loads(PLAN.read_text())
    controlled = set(plan["execution"]["reasoning_control_model_ids"])
    efforts = {row["model_id"]: row["final_reasoning_effort"] for row in plan["roster"]["models"]}
    assert all(efforts[model_id] == "minimal" for model_id in controlled - NO_REASONING_MODEL_IDS)
