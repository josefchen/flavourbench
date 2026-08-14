from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.epicure_selection_powered_plan import verify_repeat_panel
from flavourbench.epicure_selection_powered_plan_v36 import (
    NEW_PROVIDER_CALLS,
    PANEL_ORDER,
    verify_plan,
)
from flavourbench.epicure_selection_powered_plan_v38 import verify_plan as verify_plan_v38
from flavourbench.epicure_selection_powered_plan_v39 import (
    DEEPSEEK_ID,
    FINAL_MAX_OUTPUT_TOKENS,
    SelectionPoweredPlanV39Error,
    _verified_attempt_event,
)
from flavourbench.epicure_selection_powered_plan_v39 import (
    build_plan as build_plan_v39,
)
from flavourbench.epicure_selection_powered_plan_v39 import (
    verify_plan as verify_plan_v39,
)
from flavourbench.epicure_selection_powered_runner import build_cells
from flavourbench.epicure_selection_taskset_v1 import verify_taskset
from flavourbench.frontier_contract_runner import load_candidate_manifest, select_candidates
from flavourbench.frontier_manifest import verify_manifest_content_address

ROOT = Path(__file__).resolve().parents[1]


def _sole(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    assert len(matches) == 1
    return matches[0]


def _physical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def test_26_model_successor_artifacts_bind_complete_schedule() -> None:
    manifest_path = _sole("benchmark/powered-v36/manifest/*.json")
    plan_path = _sole("benchmark/powered-v36/plan/*.json")
    taskset_path = _sole("benchmark/powered-v16/taskset/*.json")
    repeat_path = ROOT / (
        "benchmark/powered-v16/plan/"
        "epicure-selection-repeat-panel-"
        "c3829d4cdb7039f920411c6edde13691237f42cafc20e463ac326a06895c97fb.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    taskset = json.loads(taskset_path.read_text(encoding="utf-8"))
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))

    assert verify_manifest_content_address(manifest)
    assert verify_plan(plan)
    assert verify_taskset(taskset)
    assert verify_repeat_panel(repeat, taskset=taskset)
    candidates = select_candidates(
        load_candidate_manifest(
            manifest_path,
            expected_digest=manifest["content_address"]["digest"],
        )
    )
    assert tuple(candidate.model_id for candidate in candidates) == PANEL_ORDER
    assert len(candidates) == 26
    assert plan["execution"]["frontier_refresh_successor"]["new_provider_calls"] == (
        NEW_PROVIDER_CALLS
    )

    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="all",
    )
    assert len(cells) == 26 * (640 + 64)
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_v39_replaces_only_the_deepseek_transport_block() -> None:
    predecessor_path = _sole("benchmark/powered-v38/plan/*.json")
    manifest_path = _sole("benchmark/powered-v37/manifest/*.json")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_plan_v38(predecessor)

    transport = {
        "plan_sha256": predecessor["artifact_sha256"],
        "response_count": 8 * 640,
        "per_model_response_count": {
            model_id: 640
            for model_id in predecessor["execution"]["frontier_refresh_successor"]["new_model_ids"]
        },
        "deepseek_status_counts": {"completed": 580, "failed": 60},
        "deepseek_finish_reason_counts": {"stop": 580, "missing": 60},
        "deepseek_provider_finish_reason_counts": {"stop": 580, "length": 60},
        "response_artifact_set_sha256": "1" * 64,
        "attempt_journal_physical_sha256": "2" * 64,
        "previous_run_spend_micros": 12_500_000,
        "scores_or_selections_used": False,
        "responses_used_as_final_deepseek_score_data": False,
    }
    plan = build_plan_v39(
        predecessor=predecessor,
        predecessor_physical_sha256=_physical_sha256(predecessor_path),
        manifest=manifest,
        manifest_physical_sha256=_physical_sha256(manifest_path),
        transport=transport,
    )

    assert verify_plan_v39(plan)
    predecessor_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
    final_rows = {row["model_id"]: row for row in plan["roster"]["models"]}
    assert final_rows[DEEPSEEK_ID]["final_max_output_tokens"] == FINAL_MAX_OUTPUT_TOKENS
    assert {model_id: row for model_id, row in final_rows.items() if model_id != DEEPSEEK_ID} == {
        model_id: row for model_id, row in predecessor_rows.items() if model_id != DEEPSEEK_ID
    }
    assert plan["budget"]["hard_cap"] == "187.500000"
    assert (
        plan["execution"]["frontier_refresh_successor"]["v38_deepseek_responses_used_as_score_data"]
        is False
    )

    plan["inputs"]["calibration_v38_primary_transport"]["scores_or_selections_used"] = True
    assert not verify_plan_v39(plan)


def test_v39_attempt_integrity_covers_the_complete_envelope() -> None:
    plan_sha256 = "a" * 64
    event = {"arm_id": "arm-1", "event_type": "response_received", "metadata": {}}
    document = {
        "schema_version": "flavourbench-powered-attempt-event-v1",
        "plan_sha256": plan_sha256,
        "recorded_at": "2026-08-14T00:00:00Z",
        "event": event,
    }
    document["event_sha256"] = _semantic_sha256(document)

    assert _verified_attempt_event(document, plan_sha256=plan_sha256) == event
    document["recorded_at"] = "2026-08-14T00:00:01Z"
    with pytest.raises(SelectionPoweredPlanV39Error, match="journal failed integrity"):
        _verified_attempt_event(document, plan_sha256=plan_sha256)
