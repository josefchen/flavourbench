from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v23 import (
    KIMI_MODEL_ID,
    KIMI_ROUTE,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v23/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v23/plan").glob("*analysis-plan*.json"))
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


def test_v23_manifest_freezes_exact_openrouter_kimi_route() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert verify_manifest_content_address(manifest)
    kimi = next(row for row in manifest["models"] if row["model"]["id"] == KIMI_MODEL_ID)
    assert kimi["execution_route"]["selected_backend"] == "openrouter"
    assert kimi["endpoint"]["tag"] == KIMI_ROUTE
    assert kimi["request_policy"]["provider"]["only"] == [KIMI_ROUTE]
    assert kimi["request_policy"]["provider"]["allow_fallbacks"] is False


def test_v23_plan_binds_route_calibration_and_single_flight() -> None:
    plan = json.loads(PLAN.read_text())
    assert verify_plan(plan)
    kimi = next(row for row in plan["roster"]["models"] if row["model_id"] == KIMI_MODEL_ID)
    assert kimi["execution_backend"] == "openrouter"
    assert kimi["provider_tag"] == KIMI_ROUTE
    assert kimi["final_reasoning_effort"] == "low"
    assert plan["inputs"]["calibration_v22"]["response_count"] == 186
    assert plan["execution"]["collection_concurrency"]["per_model_by_model_id"][KIMI_MODEL_ID] == 1


def test_v23_kimi_generation_spec_is_exact_and_low_reasoning() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    kimi = next(candidate for candidate in candidates if candidate.model_id == KIMI_MODEL_ID)
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[kimi],
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
    assert spec.execution_backend == "openrouter"
    assert spec.provider_slug == KIMI_ROUTE
    assert spec.final_reasoning_effort == "low"
    assert bundle["run_binding"]["final_reasoning_effort"] == "low"
