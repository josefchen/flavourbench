from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from flavourbench.frontier_contract_runner import AdmissionDenied, IntegrityError
from flavourbench.frontier_coverage_repair_executor import (
    EXPECTED_ARM_COUNT,
    EXPECTED_CELL_COUNT,
    RESPONSE_ENVELOPE_CLASSIFIER_SHA256,
    SupplementalRun,
    _budget_and_plan,
    _run_accounting,
    build_materialization,
    run_coverage_repair,
)
from flavourbench.real_dataset_runner import (
    _subprocess_command,
    append_dataset_ledger_event,
    derive_conditions_forecast,
    derive_pair_forecast,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
SCHEDULE = (
    CURRENT
    / "frontier-coverage-repair-v1"
    / (
        "frontier-coverage-repair-"
        "45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json"
    )
)
ARENA = (
    CURRENT
    / "frontier-model-arena-review-pool-quarantine-v1"
    / (
        "frontier-model-arena-review-pool-"
        "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json"
    )
)
TASKS = (
    ROOT
    / "artifacts/season1/task-validity/development-v2"
    / (
        "development-task-validity-v2-"
        "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
    )
)
ROUTES = (
    CURRENT
    / "manifest-v29-high-resource"
    / (
        "flavourbench-routed-unranked-"
        "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json"
    ),
    CURRENT
    / "manifest-v42-high-resource-cohere-direct"
    / (
        "flavourbench-cohere-unranked-"
        "fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json"
    ),
)
BUDGET = (
    CURRENT
    / "frontier-budget-audits"
    / "frontier-global-budget-5c37a79515a4a0a69bdc0a6d19f21a68f7f2cc24b697088c96878a1eb5da528c.json"
)
V3_ROUTE_PLAN = (
    CURRENT
    / "reasoning-effort-sensitivity-v3-route-validation"
    / (
        "reasoning-effort-v3-route-validation-plan-"
        "be2f9d19c2565df76988318b91aa8963d216ec24691446aee8c49b8737f57a56.json"
    )
)
V3_FAILED_ROUTE_AUDIT = (
    CURRENT
    / "reasoning-effort-sensitivity-v3-route-validation/final-be2f9d19/audits"
    / (
        "reasoning-effort-v3-route-validation-audit-"
        "aa66b52d784d813251f7506bbff3eff287f6a94c206fe0550b081ad34a37fb78.json"
    )
)
SENSITIVITY_ROOT = CURRENT / "reasoning-effort-sensitivity-v1/runs"
SUPPLEMENTAL = tuple(
    SupplementalRun(
        source_directory=SENSITIVITY_ROOT / variant / "source",
        ledger_path=SENSITIVITY_ROOT / variant / "ledger.jsonl",
    )
    for variant in ("explicit_low", "provider_default", "explicit_high")
)


def _materialization():
    return build_materialization(
        schedule_path=SCHEDULE,
        arena_path=ARENA,
        task_validity_path=TASKS,
        route_manifest_paths=ROUTES,
    )


def test_materialization_is_exact_real_high_resource_repair() -> None:
    materialization = _materialization()
    document = materialization.document

    assert document["counts"]["endpoint_task_cells"] == EXPECTED_CELL_COUNT
    assert document["counts"]["new_real_arms"] == EXPECTED_ARM_COUNT
    assert document["counts"]["planned_provider_work_items"] == EXPECTED_CELL_COUNT
    assert document["counts"]["provider_calls_executed_by_materialization"] == 0
    assert "provider_invocations" not in document["counts"]
    assert document["counts"]["full_pair_cells"] == 12
    assert document["counts"]["partial_condition_cells"] == 1
    assert document["counts"]["synthetic_arms"] == 0
    assert document["counts"]["current_missing_model_pair_family_cells_by_family"] == {
        "composition": 17,
        "cookability": 27,
        "evidence": 27,
        "substitution": 23,
    }
    assert document["counts"]["current_missing_model_pair_family_cells"] == 94
    assert document["counts"]["projected_missing_model_pair_family_cells_after_repair"] == 0
    assert document["execution_policy_sha256"] == (
        "579bef8dee7495d1b695c7d59365a218afebedaeb71cbad136eaab9e28d5916d"
    )
    assert document["reasoning_effort_disclosure"] == {
        "intermediate": "low",
        "final": "low",
    }
    assert document["epicure"] == {
        "release_id": "exploratory-unmatched-1790-runtime",
        "bundle_sha256": "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1",
        "application_sha256": "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313",
        "tool_schema_sha256": "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd",
    }
    assert document["response_envelope_classifier"]["contract_sha256"] == (
        RESPONSE_ENVELOPE_CLASSIFIER_SHA256
    )
    assert document["response_envelope_classifier"]["behavioral_self_test_passed"] is True
    assert document["worst_case_budget"]["total_usd"] == ("7.120075866666666666666666667")
    partial = [cell for cell in materialization.cells if len(cell.conditions) == 1]
    assert len(partial) == 1
    assert partial[0].work_item.candidate.model_id == "deepseek/deepseek-v4-pro"
    assert partial[0].work_item.task.public_id == "fb-s0-composition-004"
    assert partial[0].conditions == ("epicure_off",)
    assert partial[0].existing_conditions == ("epicure_on",)


def test_partial_forecast_and_command_never_replay_existing_on_arm(tmp_path: Path) -> None:
    materialization = _materialization()
    cell = next(cell for cell in materialization.cells if len(cell.conditions) == 1)
    on = derive_conditions_forecast(
        cell.work_item,
        policy=materialization.policy,
        conditions=("epicure_on",),
    )
    pair = derive_pair_forecast(cell.work_item, policy=materialization.policy)
    assert cell.forecast.forecast_usd + on.forecast_usd == pair.forecast_usd
    command = _subprocess_command(
        cell.work_item,
        forecast=cell.forecast,
        source_directory=tmp_path,
        manifest_path=cell.route_manifest_path,
        conditions=cell.conditions,
        expected_epicure=materialization.epicure,
    )
    condition_positions = [index for index, value in enumerate(command) if value == "--condition"]
    assert len(condition_positions) == 1
    assert command[condition_positions[0] + 1] == "epicure_off"
    assert "epicure_on" not in command
    assert "--expected-epicure-bundle-sha256" in command


def test_route_manifest_duplication_fails_closed() -> None:
    with pytest.raises(IntegrityError, match="more than one route manifest"):
        build_materialization(
            schedule_path=SCHEDULE,
            arena_path=ARENA,
            task_validity_path=TASKS,
            route_manifest_paths=(ROUTES[0], ROUTES[0], ROUTES[1]),
        )


def test_budget_reconciliation_counts_source_backed_sensitivity_reserves_once(
    tmp_path: Path,
) -> None:
    materialization = _materialization()
    coverage, budget, plan = _budget_and_plan(
        materialization,
        budget_audit_path=BUDGET,
        project_root=ROOT,
        supplemental_runs=SUPPLEMENTAL,
        source_directory=tmp_path / "source",
        corrections_directory=tmp_path / "corrections",
        response_directory=tmp_path / "responses",
        ledger_path=tmp_path / "ledger.jsonl",
        global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
        global_artifact_directory=ROOT / "artifacts/live-smoke",
        global_corrections_directory=ROOT / "artifacts/corrections",
        global_reconciliation_directory=ROOT / "artifacts/frontier-contract/reconciliations",
        cap_usd=Decimal("100"),
        admission_fraction=Decimal("0.85"),
        response_envelope_route_plan_path=None,
        response_envelope_route_audit_path=None,
    )
    assert coverage.accounting.source_count == 0
    assert budget.baseline_exposure_usd == Decimal("41.36751432666666666666666666")
    assert budget.supplemental_actual_cost_usd == Decimal("0.140822")
    assert budget.supplemental_exposure_usd == Decimal("4.785679999999999999999999998")
    assert budget.supplemental_orphan_reservation_usd == 0
    assert budget.current_total_exposure_usd == Decimal("46.15319432666666666666666666")
    assert budget.outstanding_repair_forecast_usd == Decimal("7.120075866666666666666666667")
    assert budget.projected_total_exposure_usd == Decimal("53.27327019333333333333333333")
    assert budget.budget_within_limits is True
    assert budget.admission_allowed is False
    assert plan["response_envelope_route_gate"]["status"] == ("blocked_pending_v3_route_validation")
    assert plan["blockers"][0]["gate"] == "response_envelope_route_acceptance"
    assert plan["provider_calls_made_by_plan"] == 0
    assert plan["epicure_calls_made_by_plan"] == 0


def test_failed_v3_route_audit_blocks_coverage_without_provider_call(
    tmp_path: Path,
) -> None:
    materialization = _materialization()
    with patch("flavourbench.frontier_coverage_repair_executor.subprocess.run") as run_mock:
        _, _, plan = _budget_and_plan(
            materialization,
            budget_audit_path=BUDGET,
            project_root=ROOT,
            supplemental_runs=SUPPLEMENTAL,
            source_directory=tmp_path / "source",
            corrections_directory=tmp_path / "corrections",
            response_directory=tmp_path / "responses",
            ledger_path=tmp_path / "ledger.jsonl",
            global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
            global_artifact_directory=ROOT / "artifacts/live-smoke",
            global_corrections_directory=ROOT / "artifacts/corrections",
            global_reconciliation_directory=ROOT / "artifacts/frontier-contract/reconciliations",
            cap_usd=Decimal("100"),
            admission_fraction=Decimal("0.85"),
            response_envelope_route_plan_path=V3_ROUTE_PLAN,
            response_envelope_route_audit_path=V3_FAILED_ROUTE_AUDIT,
        )
    run_mock.assert_not_called()
    gate = plan["response_envelope_route_gate"]
    assert gate["status"] == "blocked_failed_v3_route_validation"
    assert gate["route_audit_decision"] == "failed_one_or_more_predicates"
    assert plan["provider_calls_made_by_plan"] == 0
    assert plan["epicure_calls_made_by_plan"] == 0
    assert any(
        blocker["gate"] == "response_envelope_route_acceptance"
        for blocker in plan["blockers"]
    )


def test_supplemental_reservation_without_source_is_a_blocker(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_dataset_ledger_event(
        ledger,
        {
            "event_type": "reservation_created",
            "runner_run_id": "test",
            "work_item_id": "a" * 64,
            "reserved_usd": "2.5",
        },
        recorded_at="2026-08-03T00:00:00Z",
    )
    accounting = _run_accounting(
        SupplementalRun(source_directory=tmp_path / "source", ledger_path=ledger),
        label="fixture",
    )
    assert accounting.orphan_reservation_usd == Decimal("2.5")
    assert accounting.blockers[0]["gate"] == "active_reservation_without_source"


def test_dry_run_writes_content_addressed_plan_without_subprocess(
    tmp_path: Path,
) -> None:
    with patch("flavourbench.frontier_coverage_repair_executor.subprocess.run") as run_mock:
        result, materialization_path, plan_path = run_coverage_repair(
            schedule_path=SCHEDULE,
            arena_path=ARENA,
            task_validity_path=TASKS,
            route_manifest_paths=ROUTES,
            budget_audit_path=BUDGET,
            supplemental_runs=SUPPLEMENTAL,
            project_root=ROOT,
            source_directory=tmp_path / "source",
            corrections_directory=tmp_path / "corrections",
            response_directory=tmp_path / "responses",
            ledger_path=tmp_path / "ledger.jsonl",
            global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
            global_artifact_directory=ROOT / "artifacts/live-smoke",
            global_corrections_directory=ROOT / "artifacts/corrections",
            global_reconciliation_directory=ROOT / "artifacts/frontier-contract/reconciliations",
            output_directory=tmp_path / "plans",
        )
    run_mock.assert_not_called()
    assert result["subprocesses_started"] == 0
    assert result["status"] == "blocked_dry_run"
    for path in (materialization_path, plan_path):
        document = json.loads(path.read_text(encoding="utf-8"))
        digest = document.pop("artifact_sha256")
        from flavourbench.real_task_bank import sha256_json

        assert sha256_json(document) == digest
        assert digest in path.name


def test_live_execution_is_blocked_without_verified_envelope_route_pass(
    tmp_path: Path,
) -> None:
    with (
        patch("flavourbench.frontier_coverage_repair_executor.subprocess.run") as run_mock,
        pytest.raises(AdmissionDenied, match="shared budget state"),
    ):
        run_coverage_repair(
            schedule_path=SCHEDULE,
            arena_path=ARENA,
            task_validity_path=TASKS,
            route_manifest_paths=ROUTES,
            budget_audit_path=BUDGET,
            supplemental_runs=SUPPLEMENTAL,
            project_root=ROOT,
            source_directory=tmp_path / "source",
            corrections_directory=tmp_path / "corrections",
            response_directory=tmp_path / "responses",
            ledger_path=tmp_path / "ledger.jsonl",
            global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
            global_artifact_directory=ROOT / "artifacts/live-smoke",
            global_corrections_directory=ROOT / "artifacts/corrections",
            global_reconciliation_directory=ROOT / "artifacts/frontier-contract/reconciliations",
            output_directory=tmp_path / "plans",
            execute=True,
            confirmation="RUN_EXACT_COVERAGE_REPAIR_25_REAL_ARMS",
        )
    run_mock.assert_not_called()
