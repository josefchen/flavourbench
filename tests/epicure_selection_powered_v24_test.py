from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v23 import KIMI_MODEL_ID
from flavourbench.epicure_selection_powered_plan_v24 import KIMI_ROUTE, verify_plan
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v24/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v24/plan").glob("*analysis-plan*.json"))
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


def test_v24_manifest_and_plan_freeze_morph_kimi() -> None:
    manifest = json.loads(MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    entry = next(row for row in manifest["models"] if row["model"]["id"] == KIMI_MODEL_ID)
    roster = next(row for row in plan["roster"]["models"] if row["model_id"] == KIMI_MODEL_ID)
    assert entry["endpoint"]["tag"] == KIMI_ROUTE
    assert entry["request_policy"]["provider"]["only"] == [KIMI_ROUTE]
    assert roster["provider_tag"] == KIMI_ROUTE
    assert roster["execution_backend"] == "openrouter"
    assert roster["final_reasoning_effort"] == "low"
    assert plan["inputs"]["calibration_v23"]["response_count"] == 2


def test_v24_kimi_spec_uses_exact_morph_route() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    kimi = next(candidate for candidate in candidates if candidate.model_id == KIMI_MODEL_ID)
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[kimi],
        phase="pilot",
    )[0]
    spec, _ = build_generation_spec(
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
