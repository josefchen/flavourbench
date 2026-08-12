from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v28 import LLAMA_MODEL_ID
from flavourbench.epicure_selection_powered_plan_v29 import NEMOTRON_CHECK_TASK_IDS
from flavourbench.epicure_selection_powered_plan_v30 import verify_plan
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v29 import NEMOTRON_MODEL_ID as ULTRA_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v30 import (
    LIGHTNING_CANONICAL_ID,
    LIGHTNING_MODEL_ID,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v30/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
PREDECESSOR_MANIFEST = next((ROOT / "benchmark/powered-v29/manifest").glob("*.json"))
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v30/plan").glob("*analysis-plan*.json"))
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


def test_v30_replaces_only_ultra_with_lightning() -> None:
    manifest = json.loads(MANIFEST.read_text())
    predecessor = json.loads(PREDECESSOR_MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    current = {row["model"]["id"]: row for row in manifest["models"]}
    prior = {row["model"]["id"]: row for row in predecessor["models"]}
    for model_id in current.keys() - {LIGHTNING_MODEL_ID}:
        assert current[model_id] == prior[model_id]
    assert ULTRA_MODEL_ID not in current
    lightning = current[LIGHTNING_MODEL_ID]
    assert lightning["model"]["canonical_slug"] == LIGHTNING_CANONICAL_ID
    assert lightning["endpoint"]["tag"] == REPLACEMENT_TAG
    assert lightning["endpoint"]["provider_name"] == REPLACEMENT_PROVIDER
    assert lightning["request_policy"]["provider"]["only"] == [REPLACEMENT_TAG]
    assert plan["inputs"]["calibration_v29"]["response_count"] == 6
    assert set(plan["execution"]["nemotron_lightning_requalification"]["task_ids"]) == (
        NEMOTRON_CHECK_TASK_IDS
    )
    overrides = plan["execution"]["collection_concurrency"]["per_model_by_model_id"]
    assert overrides[LIGHTNING_MODEL_ID] == 1
    assert overrides[LLAMA_MODEL_ID] == 1


def test_v30_lightning_spec_expects_exact_coreweave_route() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    candidate = next(item for item in candidates if item.model_id == LIGHTNING_MODEL_ID)
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[candidate],
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
    assert spec.expected_actual_model_id == LIGHTNING_CANONICAL_ID
    assert spec.expected_actual_provider_slug == REPLACEMENT_PROVIDER
    assert spec.provider_slug == REPLACEMENT_TAG
