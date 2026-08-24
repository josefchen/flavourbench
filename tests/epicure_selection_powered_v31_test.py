from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan_v29 import NEMOTRON_CHECK_TASK_IDS
from flavourbench.epicure_selection_powered_plan_v31 import (
    MAX_OUTPUT_TOKENS,
    selection_execution_policy_v31,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v30 import (
    LIGHTNING_CANONICAL_ID,
    LIGHTNING_MODEL_ID,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v30/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v31/plan").glob("*analysis-plan*.json"))
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


def test_v31_changes_only_the_uniform_output_ceiling() -> None:
    plan = json.loads(PLAN.read_text())
    predecessor = json.loads(
        next((ROOT / "benchmark/powered-v30/plan").glob("*analysis-plan*.json")).read_text()
    )
    assert verify_plan(plan)
    assert plan["roster"] == predecessor["roster"]
    assert plan["inputs"]["route_manifest"] == predecessor["inputs"]["route_manifest"]
    assert plan["execution"]["execution_policy"]["limits"]["max_output_tokens"] == (
        MAX_OUTPUT_TOKENS
    )
    assert plan["execution"]["execution_policy_sha256"] == (selection_execution_policy_v31().sha256)
    change = plan["execution"]["uniform_output_ceiling_successor"]
    assert change["applies_to_all_models"] is True
    assert change["prompt_or_scoring_change"] is False
    assert change["route_or_roster_change"] is False
    assert plan["inputs"]["calibration_v30"]["response_count"] == 5
    assert plan["inputs"]["calibration_v30"]["used_as_primary_data"] is False
    assert set(plan["execution"]["nemotron_lightning_requalification"]["task_ids"]) == (
        NEMOTRON_CHECK_TASK_IDS
    )


def test_v31_generation_spec_binds_16384_and_exact_lightning_route() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    candidate = next(item for item in candidates if item.model_id == LIGHTNING_MODEL_ID)
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[candidate],
        phase="pilot",
    )[0]
    policy = selection_execution_policy_v31()
    spec, _ = build_generation_spec(
        cell=cell,
        plan=plan,
        manifest_sha256=MANIFEST_SHA,
        taskset=taskset,
        reserve_micros=100_000,
        execution_policy=policy,
    )
    assert policy.max_output_tokens == MAX_OUTPUT_TOKENS
    assert spec.decoding_parameters["max_tokens"] == MAX_OUTPUT_TOKENS
    assert spec.expected_actual_model_id == LIGHTNING_CANONICAL_ID
    assert spec.expected_actual_provider_slug == REPLACEMENT_PROVIDER
    assert spec.provider_slug == REPLACEMENT_TAG
