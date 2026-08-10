from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from flavourbench.frontier_contract_runner import AdmissionDenied
from flavourbench.frontier_coverage_continuation import (
    COHERE_WORK_ITEM_ID,
    GLM_WORK_ITEM_ID,
    MISTRAL_WORK_ITEM_ID,
    append_guarded_continuation_reservation,
    build_stopped_run_audit,
    require_prefixed_credential_before_reservation,
    verify_orphan_closure,
)
from flavourbench.real_dataset_runner import load_dataset_ledger
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
EXECUTION = CURRENT / "frontier-coverage-repair-execution-v1"
OUTPUT = CURRENT / "frontier-coverage-continuation-v2"
AUDIT = OUTPUT / (
    "frontier-coverage-stopped-run-audit-"
    "b0990b3b8869325771433cccd8a390a0e48038cf07637cac7ee244a39e9ca4d5.json"
)
CLOSURE = OUTPUT / (
    "frontier-coverage-orphan-closure-"
    "3cb144abd1162447e3e64ba0b703ea09d9ead595d141e2dbf1ffb0103d27e370.json"
)
CONTINUATION = OUTPUT / (
    "frontier-coverage-continuation-plan-"
    "e9f4375f8976ec7468d436ff1ade21642d6746a6eca1722f4355cdd96be19646.json"
)
REPLACEMENTS = OUTPUT / (
    "frontier-coverage-replacement-plan-"
    "3baff4ae405b0dbe4eb5168a5a088b29cb9438c86b01ce3d5a5be670839d14ee.json"
)


def _artifact(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("artifact_sha256")
    assert sha256_json(value) == digest
    return {**value, "artifact_sha256": digest}


def test_stopped_run_audit_reconstructs_three_distinct_failures() -> None:
    rebuilt = build_stopped_run_audit(
        source_directory=EXECUTION / "source",
        response_directory=EXECUTION / "responses",
        ledger_path=EXECUTION / "ledger.jsonl",
        code_directory=ROOT / "src/flavourbench",
    )
    failures = {item["work_item_id"]: item for item in rebuilt["evidence"]["failures"]}
    assert failures[GLM_WORK_ITEM_ID]["failure_class"] == "local_tool_fanout_safety_guard"
    assert failures[GLM_WORK_ITEM_ID]["actual_cost_usd"] == "0.018877"
    assert failures[MISTRAL_WORK_ITEM_ID]["failure_class"] == (
        "provider_declared_length_stop_before_tool_execution"
    )
    assert failures[MISTRAL_WORK_ITEM_ID]["actual_cost_usd"] == "0.128927"
    assert failures[COHERE_WORK_ITEM_ID]["failure_class"] == "pre_request_credential_gate"
    assert failures[COHERE_WORK_ITEM_ID]["provider_calls_verified"] == 0
    assert all(item["safe_to_replay_original_work_item"] is False for item in failures.values())


def test_append_only_orphan_closure_verifies_and_v1_stays_visible() -> None:
    closure = verify_orphan_closure(
        closure_path=CLOSURE,
        audit_path=AUDIT,
        ledger_path=EXECUTION / "ledger.jsonl",
    )
    assert closure["v1_ledger_remains_blocked_by_design"] is True
    ledger = load_dataset_ledger(EXECUTION / "ledger.jsonl")
    assert len(ledger) == 15
    assert ledger[12]["work_item_id"] == COHERE_WORK_ITEM_ID
    assert ledger[13]["work_item_id"] == COHERE_WORK_ITEM_ID
    assert ledger[14]["work_item_id"] == COHERE_WORK_ITEM_ID
    assert ledger[14]["safe_to_replay"] is False


def test_orphan_closure_matches_published_schema() -> None:
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "contracts/season1/frontier-coverage-orphan-closure-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(closure)


def test_prefixed_cohere_credential_is_required_before_reservation(tmp_path: Path) -> None:
    continuation = _artifact(CONTINUATION)
    cohere_cell = {
        "schema_version": "flavourbench-frontier-coverage-replacement-v3",
        "execution_backend": "cohere_direct",
        "work_item_id": "f" * 64,
    }
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(AdmissionDenied, match="FLAVOURBENCH_COHERE_API_KEY"):
        append_guarded_continuation_reservation(
            ledger_path=ledger,
            runner_run_id="test",
            cell=cohere_cell,
            reserved_usd="0",
            environment={"COHERE_API_KEY": "unprefixed-is-not-accepted"},
        )
    assert not ledger.exists()
    require_prefixed_credential_before_reservation(
        "cohere_direct", {"FLAVOURBENCH_COHERE_API_KEY": "configured"}
    )
    assert continuation["credential_gate"]["checked_before_reservation"] is True


def test_v2_migrates_only_untouched_non_cohere_cells_with_fresh_ids() -> None:
    plan = _artifact(CONTINUATION)
    assert plan["counts"]["migrated_untouched_non_cohere_cells"] == 6
    assert plan["counts"]["new_arm_ids"] == 12
    assert plan["v1_disposition"]["retired_work_items_replayed"] == 0
    assert all(cell["same_task_as_v1"] is False for cell in plan["fresh_non_cohere_cells"])
    assert all(
        cell["execution_backend"] != "cohere_direct"
        for cell in plan["fresh_non_cohere_cells"]
    )


def test_v3_replacements_are_fresh_blocked_and_separate_from_official_fit() -> None:
    plan = _artifact(REPLACEMENTS)
    assert plan["status"] == "blocked_missing_prefixed_cohere_credential"
    assert plan["counts"]["replacement_cells"] == 3
    assert plan["counts"]["new_real_arms_planned"] == 6
    assert {cell["failed_work_item_id"] for cell in plan["replacement_cells"]} == {
        GLM_WORK_ITEM_ID,
        MISTRAL_WORK_ITEM_ID,
        COHERE_WORK_ITEM_ID,
    }
    assert all(
        cell["alternate_task_not_previously_exposed_to_model"] is True
        and cell["fresh_identifiers_disjoint_from_v1_v2"] is True
        and cell["execution_gate"] == "frozen_not_automatically_executable"
        for cell in plan["replacement_cells"]
    )
    boundary = plan["methodological_boundary"]
    assert boundary["original_failures_remain_in_reliability_denominator"] is True
    assert boundary["official_preference_or_uplift_fit_eligible"] is False
    assert boundary["same_work_or_arm_replay_permitted"] is False
