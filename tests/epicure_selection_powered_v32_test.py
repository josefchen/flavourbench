from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_native_powered_runner import build_generation_spec
from flavourbench.epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from flavourbench.epicure_selection_powered_plan_v32 import (
    TRANSPORT_CHECK_TASK_IDS,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v26 import (
    DEEPSEEK_PRO_MODEL_ID,
    EXPECTED_ACTUAL_MODEL_ID,
)
from flavourbench.epicure_selection_route_manifest_v31 import (
    EXPECTED_PREDECESSOR_RESPONSES,
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
    deepseek_failure_commitment,
)
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v32/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
PREDECESSOR_MANIFEST = next((ROOT / "benchmark/powered-v30/manifest").glob("*.json"))
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = (
    ROOT / "benchmark/powered-v32/plan/epicure-selection-analysis-plan-"
    "ba39add4966e6991e86c591209ae416cfd232d73d8fed9d266cde59193bffed2.json"
)
PREDECESSOR_PLAN = next((ROOT / "benchmark/powered-v31/plan").glob("*analysis-plan*.json"))
PREDECESSOR_RUN = ROOT / "benchmark/powered-v31/run"
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


def test_v32_changes_only_the_deepseek_route() -> None:
    manifest = json.loads(MANIFEST.read_text())
    predecessor = json.loads(PREDECESSOR_MANIFEST.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    current = {row["model"]["id"]: row for row in manifest["models"]}
    prior = {row["model"]["id"]: row for row in predecessor["models"]}
    for model_id in current.keys() - {DEEPSEEK_PRO_MODEL_ID}:
        assert current[model_id] == prior[model_id]
    deepseek = current[DEEPSEEK_PRO_MODEL_ID]
    assert deepseek["model"]["canonical_slug"] == EXPECTED_ACTUAL_MODEL_ID
    assert deepseek["endpoint"]["tag"] == REPLACEMENT_TAG
    assert deepseek["endpoint"]["provider_name"] == REPLACEMENT_PROVIDER
    assert deepseek["request_policy"]["provider"]["only"] == [REPLACEMENT_TAG]
    recovery = plan["execution"]["deepseek_route_recovery"]
    assert recovery["successor_primary_cells"] == 640
    assert recovery["successor_repeat_cells"] == 64
    assert recovery["transport_check_task_ids"] == list(TRANSPORT_CHECK_TASK_IDS)
    assert recovery["rerun_entire_model_block"] is True
    assert recovery["reuse_predecessor_responses"] is False
    assert recovery["cross_provider_score_pooling"] is False


def test_v32_binds_and_excludes_the_complete_novita_observation() -> None:
    predecessor_plan = json.loads(PREDECESSOR_PLAN.read_text())
    observed = deepseek_failure_commitment(
        PREDECESSOR_RUN,
        expected_plan_sha256=predecessor_plan["artifact_sha256"],
    )
    manifest = json.loads(MANIFEST.read_text())
    frozen = manifest["route_refresh"]["failed_route_observation"]
    assert observed == frozen
    assert frozen["response_count"] == EXPECTED_PREDECESSOR_RESPONSES
    assert frozen["completed_count"] == 546
    assert frozen["failed_count"] == 14
    assert len(frozen["failed_task_ids"]) == 14
    assert frozen["used_as_successor_score_data"] is False
    assert frozen["excluded_from_all_model_score_and_rank_inference"] is True


def test_v32_deepseek_generation_spec_is_exact_coreweave() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    candidate = next(item for item in candidates if item.model_id == DEEPSEEK_PRO_MODEL_ID)
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
        execution_policy=selection_execution_policy_v31(),
    )
    assert spec.expected_actual_model_id == EXPECTED_ACTUAL_MODEL_ID
    assert spec.expected_actual_provider_slug == REPLACEMENT_PROVIDER
    assert spec.provider_slug == REPLACEMENT_TAG
    assert spec.decoding_parameters is not None
    assert spec.decoding_parameters["max_tokens"] == 16_384
