from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v18 import (
    NEMOTRON_MODEL_ID,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v16/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v18/plan").glob("*analysis-plan*.json"))
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


def test_v18_plan_is_exact_and_transport_repaired() -> None:
    plan = json.loads(PLAN.read_text())
    assert verify_plan(plan)
    assert plan["inputs"]["calibration_v17"]["response_count"] == 835
    assert plan["inputs"]["calibration_v17"]["used_as_primary_data"] is False
    assert plan["execution"]["collection_concurrency"]["per_model_by_model_id"] == {
        NEMOTRON_MODEL_ID: 1
    }
    reasoning = {row["model_id"]: row["final_reasoning_effort"] for row in plan["roster"]["models"]}
    assert reasoning[NEMOTRON_MODEL_ID] == "minimal"


def test_nemotron_generation_uses_minimal_reasoning() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    nemotron = next(value for value in candidates if value.model_id == NEMOTRON_MODEL_ID)
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[nemotron],
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
    assert spec.final_reasoning_effort == "minimal"
    assert bundle["run_binding"]["final_reasoning_effort"] == "minimal"
