from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.epicure_selection_powered_plan import run_commitment
from flavourbench.epicure_selection_powered_plan_v32 import TRANSPORT_CHECK_TASK_IDS
from flavourbench.epicure_selection_powered_plan_v33 import (
    DEEPSEEK_CONCURRENCY,
    verify_plan,
)
from flavourbench.epicure_selection_powered_runner import validate_inputs
from flavourbench.epicure_selection_route_manifest_v26 import DEEPSEEK_PRO_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v31 import (
    REPLACEMENT_PROVIDER,
    REPLACEMENT_TAG,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v32/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v33/plan").glob("*analysis-plan*.json"))
CALIBRATION = ROOT / "benchmark/powered-v32/run"
PREDECESSOR = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def test_v33_freezes_full_coreweave_block_after_eight_checks() -> None:
    if not (CALIBRATION / "attempts/provider-attempts.jsonl").is_file():
        pytest.skip("raw pre-release calibration is outside the compact public checkout")
    document = json.loads(PLAN.read_text())
    assert verify_plan(document)
    calibration = document["inputs"]["calibration_v32"]
    assert calibration["response_count"] == 8
    assert calibration["task_ids"] == list(TRANSPORT_CHECK_TASK_IDS)
    assert calibration["scores_or_selections_inspected_before_successor_freeze"] is False
    assert calibration["captured_responses_remain_calibration_only"] is True
    assert calibration["used_as_primary_data"] is False
    assert (
        run_commitment(CALIBRATION, expected_responses=8)["response_artifact_set_sha256"]
        == calibration["response_artifact_set_sha256"]
    )
    recovery = document["execution"]["deepseek_route_recovery"]
    assert recovery["transport_check_reused_as_primary"] is False
    assert recovery["successor_primary_cells"] == 640
    assert recovery["successor_repeat_cells"] == 64
    concurrency = document["execution"]["deepseek_concurrency_successor"]
    assert concurrency["successor_concurrency"] == DEEPSEEK_CONCURRENCY
    assert (
        document["execution"]["collection_concurrency"]["per_model_by_model_id"][
            DEEPSEEK_PRO_MODEL_ID
        ]
        == DEEPSEEK_CONCURRENCY
    )


def test_v33_inputs_resolve_to_exact_coreweave_route() -> None:
    _, _, _, plan, _, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )
    candidate = next(item for item in candidates if item.model_id == DEEPSEEK_PRO_MODEL_ID)
    assert candidate.provider_tag == REPLACEMENT_TAG
    assert candidate.provider_name == REPLACEMENT_PROVIDER
    row = next(
        item for item in plan["roster"]["models"] if item["model_id"] == DEEPSEEK_PRO_MODEL_ID
    )
    assert row["provider_tag"] == REPLACEMENT_TAG
    assert row["provider_name"] == REPLACEMENT_PROVIDER
