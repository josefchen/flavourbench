from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v18 import NEMOTRON_MODEL_ID
from flavourbench.epicure_selection_powered_plan_v19 import verify_plan
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v19/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v19/plan").glob("*analysis-plan*.json"))
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


def test_v19_plan_and_manifest_bind_together_nemotron() -> None:
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    selected = next(row for row in manifest["models"] if row["model"]["id"] == NEMOTRON_MODEL_ID)
    assert selected["endpoint"]["tag"] == "together"
    assert selected["request_policy"]["provider"]["only"] == ["together"]
    assert selected["request_policy"]["provider"]["allow_fallbacks"] is False
    assert plan["inputs"]["calibration_v18"]["response_count"] == 8
    assert plan["inputs"]["calibration_v18"]["used_as_primary_data"] is False


def test_v19_nemotron_spec_keeps_exact_route_and_minimal_reasoning() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    nemotron = next(value for value in candidates if value.model_id == NEMOTRON_MODEL_ID)
    assert nemotron.provider_tag == "together"
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
    assert spec.provider_slug == "together"
    assert spec.expected_actual_provider_slug == "Together"
    assert bundle["run_binding"]["final_reasoning_effort"] == "minimal"
