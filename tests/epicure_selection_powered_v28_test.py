from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_plan_v28 import (
    FABLE_CHECK_TASK_IDS,
    LLAMA_MODEL_ID,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v28 import (
    FABLE_CANONICAL_ID,
    FABLE_MODEL_ID,
    NEMOTRON_MODEL_ID,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v28/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
PREDECESSOR_MANIFEST = next((ROOT / "benchmark/powered-v27/manifest").glob("*.json"))
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v28/plan").glob("*analysis-plan*.json"))
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


def test_v28_replaces_only_nemotron_with_fable() -> None:
    manifest = json.loads(MANIFEST.read_text())
    predecessor = json.loads(PREDECESSOR_MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    current = {row["model"]["id"]: row for row in manifest["models"]}
    prior = {row["model"]["id"]: row for row in predecessor["models"]}
    for model_id in current.keys() - {FABLE_MODEL_ID}:
        assert current[model_id] == prior[model_id]
    assert NEMOTRON_MODEL_ID not in current
    fable = current[FABLE_MODEL_ID]
    assert fable["model"]["canonical_slug"] == FABLE_CANONICAL_ID
    assert fable["endpoint"]["tag"] == REPLACEMENT_TAG
    assert fable["endpoint"]["provider_name"] == REPLACEMENT_PROVIDER
    assert fable["request_policy"]["provider"]["only"] == [REPLACEMENT_TAG]
    assert plan["inputs"]["calibration_v27"]["response_count"] == 639
    assert set(plan["execution"]["fable_requalification"]["task_ids"]) == FABLE_CHECK_TASK_IDS
    overrides = plan["execution"]["collection_concurrency"]["per_model_by_model_id"]
    assert overrides[FABLE_MODEL_ID] == 1
    assert overrides[LLAMA_MODEL_ID] == 1


def test_v28_fable_spec_expects_exact_bedrock_route() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    candidate = next(item for item in candidates if item.model_id == FABLE_MODEL_ID)
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
        reserve_micros=500_000,
        execution_policy=selection_execution_policy(),
    )
    assert spec.expected_actual_model_id == FABLE_CANONICAL_ID
    assert spec.expected_actual_provider_slug == REPLACEMENT_PROVIDER
    assert spec.provider_slug == REPLACEMENT_TAG
