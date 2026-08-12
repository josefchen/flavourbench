from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v22 import GPT_MODEL_IDS, verify_plan
from flavourbench.epicure_selection_powered_runner import validate_inputs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v19/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v22/plan").glob("*analysis-plan*.json"))
PREDECESSOR = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def test_v22_plan_binds_v21_calibration_and_single_flights_gpt_routes() -> None:
    plan = json.loads(PLAN.read_text())
    assert verify_plan(plan)
    overrides = plan["execution"]["collection_concurrency"]["per_model_by_model_id"]
    assert all(overrides[model_id] == 1 for model_id in GPT_MODEL_IDS)
    assert overrides["nvidia/nemotron-3-ultra-550b-a55b"] == 1
    assert plan["inputs"]["calibration_v21"]["response_count"] == 181
    assert plan["inputs"]["calibration_v21"]["used_as_primary_data"] is False


def test_v22_complete_input_bundle_validates() -> None:
    _, _, _, plan, _, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )
    assert verify_plan(plan)
    assert len(candidates) == 20
