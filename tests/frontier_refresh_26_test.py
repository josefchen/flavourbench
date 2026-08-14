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
from flavourbench.epicure_selection_powered_plan_v40 import FABLE_ID
from flavourbench.epicure_selection_powered_plan_v40 import (
    build_plan as build_plan_v40,
)
from flavourbench.epicure_selection_powered_plan_v40 import (
    verify_plan as verify_plan_v40,
)
from flavourbench.epicure_selection_powered_plan_v41 import (
    verify_plan as verify_plan_v41,
)
from flavourbench.epicure_selection_powered_plan_v42 import (
    verify_plan as verify_plan_v42,
)
from flavourbench.epicure_selection_powered_plan_v43 import (
    verify_plan as verify_plan_v43,
)
from flavourbench.epicure_selection_powered_runner import build_cells
from flavourbench.epicure_selection_route_manifest_v41 import SELECTED_PROVIDER, SELECTED_TAG
from flavourbench.epicure_selection_route_manifest_v41 import (
    verify_manifest as verify_manifest_v41,
)
from flavourbench.epicure_selection_route_manifest_v42 import (
    verify_manifest as verify_manifest_v42,
)
from flavourbench.epicure_selection_route_manifest_v43 import (
    verify_manifest as verify_manifest_v43,
)
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


def test_v40_replaces_the_complete_fable_block_without_changing_its_contract() -> None:
    predecessor_path = _sole("benchmark/powered-v39/plan/*.json")
    manifest_path = _sole("benchmark/powered-v37/manifest/*.json")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_plan_v39(predecessor)

    fable_transport = {
        "plan_sha256": predecessor["inputs"]["plan_v38_predecessor"]["semantic_sha256"],
        "primary_response_count": 640,
        "repeat_response_count": 64,
        "primary_status_counts": {"completed": 244, "failed": 396},
        "repeat_status_counts": {"completed": 27, "failed": 37},
        "primary_completion_rate": 244 / 640,
        "repeat_completion_rate": 27 / 64,
        "failed_error_type_counts": {"ProviderError": 433},
        "failed_error_message_sha256_counts": {"1" * 64: 433},
        "response_artifact_set_sha256": "2" * 64,
        "attempt_journal_physical_sha256": "3" * 64,
        "spend_micros": 2_110_682,
        "aggregate_fable_score_was_inspected_before_repair": True,
        "task_level_scores_or_selections_used_to_change_execution_contract": False,
        "execution_contract_changed": False,
        "complete_old_fable_block_used_as_final_score_data": False,
    }
    deepseek_source = {
        "plan_sha256": predecessor["artifact_sha256"],
        "primary_response_count": 640,
        "repeat_response_count": 64,
        "primary_status_counts": {"completed": 629, "failed": 11},
        "repeat_status_counts": {"completed": 55, "failed": 9},
        "response_artifact_set_sha256": "4" * 64,
        "spend_micros": 6_307_482,
        "responses_used_as_final_deepseek_score_data": True,
    }
    plan = build_plan_v40(
        predecessor=predecessor,
        predecessor_physical_sha256=_physical_sha256(predecessor_path),
        manifest=manifest,
        manifest_physical_sha256=_physical_sha256(manifest_path),
        fable_transport=fable_transport,
        deepseek_source=deepseek_source,
    )

    assert verify_plan_v40(plan)
    predecessor_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
    final_rows = {row["model_id"]: row for row in plan["roster"]["models"]}
    assert final_rows == predecessor_rows
    successor = plan["execution"]["frontier_refresh_successor"]
    assert successor["rerun_model_ids"] == [FABLE_ID]
    assert successor["retained_v39_new_model_ids"] == [DEEPSEEK_ID]
    assert successor["selective_failed_cell_retry"] is False
    assert successor["v38_fable_responses_used_as_score_data"] is False
    assert plan["budget"]["hard_cap"] == "170.457085"

    plan["inputs"]["calibration_v38_fable_transport"]["execution_contract_changed"] = True
    assert not verify_plan_v40(plan)


def test_v41_changes_only_fables_provider_route_after_content_filter_failure() -> None:
    source_manifest = json.loads(_sole("benchmark/powered-v37/manifest/*.json").read_text())
    manifest = json.loads(_sole("benchmark/powered-v41/manifest/*.json").read_text())
    predecessor = json.loads(_sole("benchmark/powered-v40/plan/*.json").read_text())
    plan = json.loads(_sole("benchmark/powered-v41/plan/*.json").read_text())

    assert verify_manifest_v41(manifest)
    assert verify_plan_v41(plan)
    source_rows = {row["model"]["id"]: row for row in source_manifest["models"]}
    final_rows = {row["model"]["id"]: row for row in manifest["models"]}
    assert {key: value for key, value in source_rows.items() if key != FABLE_ID} == {
        key: value for key, value in final_rows.items() if key != FABLE_ID
    }
    assert final_rows[FABLE_ID]["endpoint"]["tag"] == SELECTED_TAG
    assert final_rows[FABLE_ID]["endpoint"]["provider_name"] == SELECTED_PROVIDER
    assert (
        final_rows[FABLE_ID]["model"]["canonical_slug"]
        == source_rows[FABLE_ID]["model"]["canonical_slug"]
    )

    predecessor_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
    plan_rows = {row["model_id"]: row for row in plan["roster"]["models"]}
    assert {key: value for key, value in predecessor_rows.items() if key != FABLE_ID} == {
        key: value for key, value in plan_rows.items() if key != FABLE_ID
    }
    assert plan_rows[FABLE_ID]["provider_tag"] == SELECTED_TAG
    assert plan_rows[FABLE_ID]["provider_name"] == SELECTED_PROVIDER
    transport = plan["inputs"]["calibration_v40_fable_transport"]
    assert transport["response_count"] == 26
    assert transport["status_counts"] == {"completed": 11, "failed": 15}
    assert transport["provider_finish_reason_counts"] == {"content_filter": 16, "stop": 12}
    assert transport["response_received_count"] == 28
    assert transport["artifactless_response_received_count"] == 2
    assert transport["bounded_exposure_micros"] == 1_804_536
    assert transport["task_scores_or_selections_used_for_route_choice"] is False
    assert (
        plan["execution"]["frontier_refresh_successor"]["v40_fable_responses_used_as_score_data"]
        is False
    )


def test_v42_freezes_google_global_before_the_complete_fable_block() -> None:
    source_manifest = json.loads(_sole("benchmark/powered-v41/manifest/*.json").read_text())
    manifest = json.loads(_sole("benchmark/powered-v42/manifest/*.json").read_text())
    predecessor = json.loads(_sole("benchmark/powered-v41/plan/*.json").read_text())
    plan = json.loads(_sole("benchmark/powered-v42/plan/*.json").read_text())

    assert verify_manifest_v42(manifest)
    assert verify_plan_v42(plan)
    source_rows = {row["model"]["id"]: row for row in source_manifest["models"]}
    final_rows = {row["model"]["id"]: row for row in manifest["models"]}
    assert {key: value for key, value in source_rows.items() if key != FABLE_ID} == {
        key: value for key, value in final_rows.items() if key != FABLE_ID
    }
    assert final_rows[FABLE_ID]["endpoint"]["tag"] == "google-vertex/global"
    assert final_rows[FABLE_ID]["endpoint"]["provider_name"] == "Google"

    predecessor_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
    plan_rows = {row["model_id"]: row for row in plan["roster"]["models"]}
    assert {key: value for key, value in predecessor_rows.items() if key != FABLE_ID} == {
        key: value for key, value in plan_rows.items() if key != FABLE_ID
    }
    assert plan_rows[FABLE_ID]["provider_tag"] == "google-vertex/global"
    assert plan_rows[FABLE_ID]["provider_name"] == "Google"
    probe = plan["inputs"]["bounded_fable_route_probe"]
    assert probe["answers_or_scores_used"] is False
    assert probe["selected_provider_tag"] == "google-vertex/global"
    assert probe["provider_calls"] == 16
    assert plan["budget"]["hard_cap"] == "154.884415"


def test_v43_replaces_the_entire_fable_block_on_public_openrouter_anthropic() -> None:
    source_manifest = json.loads(_sole("benchmark/powered-v42/manifest/*.json").read_text())
    manifest = json.loads(_sole("benchmark/powered-v43/manifest/*.json").read_text())
    predecessor = json.loads(_sole("benchmark/powered-v42/plan/*.json").read_text())
    plan = json.loads(_sole("benchmark/powered-v43/plan/*.json").read_text())

    assert verify_manifest_v43(manifest)
    assert verify_plan_v43(plan)
    source_rows = {row["model"]["id"]: row for row in source_manifest["models"]}
    final_rows = {row["model"]["id"]: row for row in manifest["models"]}
    assert {key: value for key, value in source_rows.items() if key != FABLE_ID} == {
        key: value for key, value in final_rows.items() if key != FABLE_ID
    }
    assert final_rows[FABLE_ID]["endpoint"]["tag"] == "anthropic"
    assert final_rows[FABLE_ID]["endpoint"]["provider_name"] == "Anthropic"
    assert final_rows[FABLE_ID]["request_policy"]["provider"] == {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "only": ["anthropic"],
        "require_parameters": True,
    }

    predecessor_rows = {row["model_id"]: row for row in predecessor["roster"]["models"]}
    plan_rows = {row["model_id"]: row for row in plan["roster"]["models"]}
    assert {key: value for key, value in predecessor_rows.items() if key != FABLE_ID} == {
        key: value for key, value in plan_rows.items() if key != FABLE_ID
    }
    assert plan_rows[FABLE_ID]["provider_tag"] == "anthropic"
    assert plan_rows[FABLE_ID]["provider_name"] == "Anthropic"
    transport = plan["inputs"]["complete_v42_fable_transport"]
    assert transport["primary_response_count"] == 640
    assert transport["repeat_response_count"] == 64
    assert transport["primary_status_counts"] == {"completed": 413, "failed": 227}
    assert transport["repeat_status_counts"] == {"completed": 43, "failed": 21}
    assert transport["complete_old_block_used_as_final_score_data"] is False
    successor = plan["execution"]["frontier_refresh_successor"]
    assert successor["full_fable_block_replacement"] is True
    assert successor["selective_failed_cell_retry"] is False
    assert successor["new_provider_calls"] == 704
    assert plan["budget"]["prior_v42_fable_spend_micros"] == 2_911_055
    assert plan["budget"]["hard_cap"] == "151.973360"
