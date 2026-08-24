from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v20 import verify_plan
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v19/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v20/plan").glob("*analysis-plan*.json"))
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


def test_v20_plan_controls_every_route_that_advertises_reasoning_effort() -> None:
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_plan(plan)
    supported = {
        row["model"]["id"]
        for row in manifest["models"]
        if "reasoning_effort" in row["endpoint"].get("supported_parameters", [])
    }
    controlled = set(plan["execution"]["reasoning_control_model_ids"])
    assert controlled == supported
    assert len(controlled) == 14
    assert plan["inputs"]["calibration_v19"]["response_count"] == 934
    assert plan["inputs"]["calibration_v19"]["used_as_primary_data"] is False


def test_v20_affected_route_specs_use_minimal_reasoning() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    affected = {
        "google/gemini-3.1-pro-preview",
        "z-ai/glm-5.2",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash-0731",
    }
    for candidate in candidates:
        if candidate.model_id not in affected:
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
        assert spec.final_reasoning_effort == "minimal"
        assert bundle["run_binding"]["final_reasoning_effort"] == "minimal"


def test_v20_provider_fixed_routes_remain_explicit() -> None:
    plan = json.loads(PLAN.read_text())
    controlled = set(plan["execution"]["reasoning_control_model_ids"])
    fixed = {
        row["model_id"]
        for row in plan["roster"]["models"]
        if row["final_reasoning_effort"] == "provider_fixed"
    }
    assert len(fixed) == 6
    assert fixed.isdisjoint(controlled)
    assert fixed | controlled == {row["model_id"] for row in plan["roster"]["models"]}
