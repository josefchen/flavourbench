from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from flavourbench.frontier_contract_runner import AdmissionDenied, IntegrityError
from flavourbench.frontier_coverage_continuation_executor import (
    build_postrun_audit as build_parent_postrun_audit,
)
from flavourbench.frontier_coverage_recovery_v4 import (
    CELL_SPECS,
    GLM_PHASE,
    HISTORICAL_EXPOSURE_RECORD_COUNT,
    HISTORICAL_EXPOSURE_VIEW_SHA256,
    PARENT_AUDIT_SHA256,
    PARENT_CLOSURE_SHA256,
    PARENT_INCOMPLETE_GLM_WORK_ITEM,
    PARENT_PREFLIGHT_SHA256,
    PARENT_RECEIPT_SHA256,
    PARENT_SOURCE_MIGRATED_BYTES,
    PARENT_SOURCE_MIGRATED_PATH,
    PARENT_SOURCE_MIGRATED_SHA256,
    PARENT_SOURCE_MIGRATION_SHA256,
    PLAN_SCHEMA_VERSION,
    QUARANTINED_TASK_IDS,
    REASONING_V4_AUDIT_SHA256,
    REASONING_V4_CLOSURE_SHA256,
    REASONING_V4_GEMINI_RESERVATION_SHA256,
    REASONING_V4_RECEIPT_SHA256,
    REASONING_V4_ROUTE_PLAN_SHA256,
    REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD,
    RECOVERY_PHASE,
    _collect_independent_dispositions,
    _assert_no_model_task_overlap,
    _historical_exposure_view_path,
    _load_addressed,
    _parent_source_migration_path,
    _reservation_fields,
    _runtime_cells,
    _scan_model_task_exposure,
    _terminal_orphan_blocker_matches,
    _verify_parent_audit_source_migration,
    _verify_parent_historical_source_view,
    _verify_historical_exposure_view,
    _verify_reasoning_v5_resolution_from_preflight,
    _verify_terminal_resolution_from_preflight,
    build_plan,
    build_preflight,
    execute_phase,
    reconstruct_parent,
    verify_reasoning_v4_terminal_orphan,
    verify_reasoning_v5_terminal_endpoints,
)
from flavourbench.real_dataset_runner import append_dataset_ledger_event

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
PARENT_ROOT = CURRENT / "frontier-coverage-continuation-execution-v1"
PARENT_PREFLIGHT = PARENT_ROOT / (
    f"frontier-coverage-continuation-preflight-{PARENT_PREFLIGHT_SHA256}.json"
)
PARENT_RECEIPT = PARENT_ROOT / (
    f"frontier-coverage-continuation-receipt-{PARENT_RECEIPT_SHA256}.json"
)
PARENT_CLOSURE = PARENT_ROOT / (
    f"frontier-coverage-continuation-closure-{PARENT_CLOSURE_SHA256}.json"
)
PARENT_AUDIT = PARENT_ROOT / (
    f"frontier-coverage-continuation-postrun-audit-{PARENT_AUDIT_SHA256}.json"
)
OUTPUT_ROOT = CURRENT / "frontier-coverage-recovery-v4"
QUARANTINE = (
    CURRENT
    / "task-quarantine-v1"
    / (
        "current-frontier-task-quarantine-"
        "e095c45ed27b0639a8eefae13a028c653fdea493999e095c2a757818ebbb7a15.json"
    )
)
REASONING_V4_ROOT = CURRENT / "reasoning-effort-sensitivity-v4"
REASONING_V4_GATE = REASONING_V4_ROOT / "route-gate"
REASONING_V4_ROUTE_PLAN = REASONING_V4_ROOT / (
    f"reasoning-effort-v4-route-gate-plan-{REASONING_V4_ROUTE_PLAN_SHA256}.json"
)
REASONING_V4_RECEIPT = REASONING_V4_GATE / (
    f"reasoning-effort-v4-route-gate-execution-receipt-{REASONING_V4_RECEIPT_SHA256}.json"
)
REASONING_V4_AUDIT = REASONING_V4_GATE / (
    f"reasoning-effort-v4-route-gate-audit-{REASONING_V4_AUDIT_SHA256}.json"
)
REASONING_V4_CLOSURE = REASONING_V4_GATE / (
    f"reasoning-effort-v4-route-gate-closure-{REASONING_V4_CLOSURE_SHA256}.json"
)
REASONING_V4_LEDGER = REASONING_V4_GATE / "ledger.jsonl"
REASONING_V4_SOURCE = REASONING_V4_GATE / "source"
REASONING_V4_JOURNAL = REASONING_V4_SOURCE / (
    ".flavourbench-live-smoke-journal-19125098-99b0-58af-b87b-a6260a9c5bd3.inprogress.jsonl"
)
BUDGET_AUDIT = (
    CURRENT
    / "frontier-budget-audits"
    / (
        "frontier-global-budget-"
        "ec179b7889834d2c6c92343acfb332e907a22600531333e9f0e1f7d7708a241d.json"
    )
)
REASONING_V5_ROOT = CURRENT / "reasoning-effort-sensitivity-v5"
REASONING_V5_ROUTE_PLAN = next((REASONING_V5_ROOT / "route-gate").glob("*route-gate-plan*.json"))
REASONING_V5_ENDPOINT_SNAPSHOT = next((REASONING_V5_ROOT / "endpoint-snapshot").glob("*.json"))
REASONING_V5_SONNET_ROOT = REASONING_V5_ROOT / "sonnet"
REASONING_V5_GEMINI_ROOT = REASONING_V5_ROOT / "gemini"
REASONING_V5_SONNET_RECEIPT = next((REASONING_V5_SONNET_ROOT / "receipts").glob("*.json"))
REASONING_V5_SONNET_AUDIT = next((REASONING_V5_SONNET_ROOT / "audits").glob("*.json"))
REASONING_V5_SONNET_CLOSURE = next((REASONING_V5_SONNET_ROOT / "closures").glob("*.json"))
REASONING_V5_GEMINI_RECEIPT = next((REASONING_V5_GEMINI_ROOT / "receipts").glob("*.json"))
REASONING_V5_GEMINI_AUDIT = next((REASONING_V5_GEMINI_ROOT / "audits").glob("*.json"))
REASONING_V5_GEMINI_CLOSURE = next((REASONING_V5_GEMINI_ROOT / "closures").glob("*.json"))
REASONING_V5_AGGREGATE_AUDIT = next(
    (REASONING_V5_ROOT / "aggregate").glob("*aggregate-audit*.json")
)
REASONING_V5_AGGREGATE_CLOSURE = next(
    (REASONING_V5_ROOT / "aggregate").glob("*aggregate-closure*.json")
)


def _terminal_resolution(*, journal_path: Path = REASONING_V4_JOURNAL) -> dict[str, object]:
    return verify_reasoning_v4_terminal_orphan(
        project_root=ROOT,
        route_plan_path=REASONING_V4_ROUTE_PLAN,
        receipt_path=REASONING_V4_RECEIPT,
        audit_path=REASONING_V4_AUDIT,
        closure_path=REASONING_V4_CLOSURE,
        ledger_path=REASONING_V4_LEDGER,
        journal_path=journal_path,
        source_directory=REASONING_V4_SOURCE,
    )


def _current_preflight(*, output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    return build_preflight(
        project_root=ROOT,
        plan_path=_plan_path(),
        parent_root=PARENT_ROOT,
        quarantine_path=QUARANTINE,
        exposure_root=CURRENT,
        output_root=output_root,
        budget_audit_path=BUDGET_AUDIT,
        supplemental_roots=[
            CURRENT / "reasoning-effort-sensitivity-v1/runs/explicit_low",
            CURRENT / "reasoning-effort-sensitivity-v1/runs/provider_default",
            CURRENT / "reasoning-effort-sensitivity-v1/runs/explicit_high",
            REASONING_V4_GATE,
        ],
        global_ledger_path=ROOT / "artifacts/frontier-contract/ledger.jsonl",
        global_artifact_directory=ROOT / "artifacts/live-smoke",
        global_corrections_directory=ROOT / "artifacts/corrections",
        global_reconciliation_directory=ROOT / "artifacts/frontier-contract/reconciliations",
        reasoning_v4_route_plan_path=REASONING_V4_ROUTE_PLAN,
        reasoning_v4_receipt_path=REASONING_V4_RECEIPT,
        reasoning_v4_audit_path=REASONING_V4_AUDIT,
        reasoning_v4_closure_path=REASONING_V4_CLOSURE,
        reasoning_v4_ledger_path=REASONING_V4_LEDGER,
        reasoning_v4_journal_path=REASONING_V4_JOURNAL,
        reasoning_v4_source_directory=REASONING_V4_SOURCE,
        reasoning_v5_route_plan_path=REASONING_V5_ROUTE_PLAN,
        reasoning_v5_endpoint_snapshot_path=REASONING_V5_ENDPOINT_SNAPSHOT,
        reasoning_v5_sonnet_root=REASONING_V5_SONNET_ROOT,
        reasoning_v5_sonnet_receipt_path=REASONING_V5_SONNET_RECEIPT,
        reasoning_v5_sonnet_audit_path=REASONING_V5_SONNET_AUDIT,
        reasoning_v5_sonnet_closure_path=REASONING_V5_SONNET_CLOSURE,
        reasoning_v5_gemini_root=REASONING_V5_GEMINI_ROOT,
        reasoning_v5_gemini_receipt_path=REASONING_V5_GEMINI_RECEIPT,
        reasoning_v5_gemini_audit_path=REASONING_V5_GEMINI_AUDIT,
        reasoning_v5_gemini_closure_path=REASONING_V5_GEMINI_CLOSURE,
        reasoning_v5_aggregate_audit_path=REASONING_V5_AGGREGATE_AUDIT,
        reasoning_v5_aggregate_closure_path=REASONING_V5_AGGREGATE_CLOSURE,
        environment={
            "FLAVOURBENCH_OPENROUTER_API_KEY": "present-not-contacted",
            "FLAVOURBENCH_COHERE_API_KEY": "present-not-contacted",
            "FLAVOURBENCH_MCP_URL": "http://127.0.0.1:1/mcp",
            "FLAVOURBENCH_MCP_TOKEN": "present-not-contacted",
        },
    )


def _plan_path() -> Path:
    paths = sorted(OUTPUT_ROOT.glob("frontier-coverage-recovery-v4-plan-*.json"))
    assert len(paths) == 1
    return paths[0]


def test_parent_terminal_audit_reconstructs_exactly() -> None:
    parent = reconstruct_parent(project_root=ROOT, parent_root=PARENT_ROOT)
    assert parent.audit["artifact_sha256"] == PARENT_AUDIT_SHA256
    assert parent.audit["counts"] == {
        "planned_cells": 9,
        "usable_cells": 1,
        "planned_real_arms": 18,
        "usable_real_arms": 2,
        "provider_generations": 11,
        "successful_epicure_tool_calls": 1,
        "synthetic_arms": 0,
    }
    failures = parent.audit["failures"]
    assert len(failures) == 2
    assert all(value.startswith(PARENT_INCOMPLETE_GLM_WORK_ITEM) for value in failures)


def test_historical_source_migration_is_exact_content_addressed_and_execution_isolated() -> None:
    preflight = _load_addressed(
        PARENT_PREFLIGHT,
        label="parent preflight",
        expected_schema="flavourbench-frontier-coverage-continuation-preflight-v1",
        expected_digest=PARENT_PREFLIGHT_SHA256,
    )
    migration = _verify_parent_historical_source_view(
        project_root=ROOT,
        preflight=preflight,
    )
    assert migration["artifact_sha256"] == PARENT_SOURCE_MIGRATION_SHA256
    archive = ROOT / migration["archived_source"]["path"]
    payload = archive.read_bytes()
    assert len(payload) == PARENT_SOURCE_MIGRATED_BYTES
    assert hashlib.sha256(payload).hexdigest() == PARENT_SOURCE_MIGRATED_SHA256
    assert migration["prospective_execution_source_override"] is False
    assert migration["same_archive_permitted_for_new_generations"] is False
    assert hashlib.sha256((ROOT / PARENT_SOURCE_MIGRATED_PATH).read_bytes()).hexdigest() != (
        PARENT_SOURCE_MIGRATED_SHA256
    )


def test_historical_source_migration_rejects_archive_and_record_tampering(
    tmp_path: Path,
) -> None:
    preflight = _load_addressed(
        PARENT_PREFLIGHT,
        label="parent preflight",
        expected_schema="flavourbench-frontier-coverage-continuation-preflight-v1",
        expected_digest=PARENT_PREFLIGHT_SHA256,
    )
    migration_source = _parent_source_migration_path(ROOT)
    migration = json.loads(migration_source.read_text(encoding="utf-8"))
    project = tmp_path / "historical-project"
    archive = project / migration["archived_source"]["path"]
    archive.parent.mkdir(parents=True)
    shutil.copy2(ROOT / migration["archived_source"]["path"], archive)
    migration_copy = project / migration_source.relative_to(ROOT)
    migration_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(migration_source, migration_copy)
    _verify_parent_historical_source_view(
        project_root=project,
        preflight=preflight,
        migration_path=migration_copy,
        current_source_bundle=preflight["source_code"],
    )

    archive.write_bytes(archive.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="archive is missing or tampered"):
        _verify_parent_historical_source_view(
            project_root=project,
            preflight=preflight,
            migration_path=migration_copy,
            current_source_bundle=preflight["source_code"],
        )
    shutil.copy2(ROOT / migration["archived_source"]["path"], archive)
    tampered = json.loads(migration_copy.read_text(encoding="utf-8"))
    tampered["prospective_execution_source_override"] = True
    migration_copy.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(IntegrityError, match="content address does not verify"):
        _verify_parent_historical_source_view(
            project_root=project,
            preflight=preflight,
            migration_path=migration_copy,
            current_source_bundle=preflight["source_code"],
        )


def test_historical_source_view_is_independent_only_of_the_migrated_live_file() -> None:
    preflight = _load_addressed(
        PARENT_PREFLIGHT,
        label="parent preflight",
        expected_schema="flavourbench-frontier-coverage-continuation-preflight-v1",
        expected_digest=PARENT_PREFLIGHT_SHA256,
    )
    future_bundle = copy.deepcopy(preflight["source_code"])
    migrated = next(
        row for row in future_bundle["files"] if row["path"] == PARENT_SOURCE_MIGRATED_PATH
    )
    migrated.update({"sha256": "a" * 64, "bytes": 1})
    _verify_parent_historical_source_view(
        project_root=ROOT,
        preflight=preflight,
        current_source_bundle=future_bundle,
    )

    unmigrated = next(
        row for row in future_bundle["files"] if row["path"] != PARENT_SOURCE_MIGRATED_PATH
    )
    unmigrated["sha256"] = "b" * 64
    with pytest.raises(IntegrityError, match="unmigrated historical source differs"):
        _verify_parent_historical_source_view(
            project_root=ROOT,
            preflight=preflight,
            current_source_bundle=future_bundle,
        )


def test_historical_audit_migration_rejects_any_second_delta() -> None:
    preflight = _load_addressed(
        PARENT_PREFLIGHT,
        label="parent preflight",
        expected_schema="flavourbench-frontier-coverage-continuation-preflight-v1",
        expected_digest=PARENT_PREFLIGHT_SHA256,
    )
    frozen = _load_addressed(
        PARENT_AUDIT,
        label="parent audit",
        expected_schema="flavourbench-frontier-coverage-continuation-postrun-audit-v1",
        expected_digest=PARENT_AUDIT_SHA256,
    )
    rebuilt = build_parent_postrun_audit(
        preflight_path=PARENT_PREFLIGHT,
        receipt_path=PARENT_RECEIPT,
        closure_path=PARENT_CLOSURE,
        project_root=ROOT,
        output_root=PARENT_ROOT,
    )
    _verify_parent_audit_source_migration(
        project_root=ROOT,
        preflight=preflight,
        frozen_audit=frozen,
        rebuilt_audit=rebuilt,
    )
    altered = copy.deepcopy(rebuilt)
    altered["counts"]["provider_generations"] += 1
    with pytest.raises(IntegrityError, match="does not recover the frozen audit"):
        _verify_parent_audit_source_migration(
            project_root=ROOT,
            preflight=preflight,
            frozen_audit=frozen,
            rebuilt_audit=altered,
        )


def test_frozen_plan_is_fresh_ordered_unexposed_and_unranked() -> None:
    frozen = _load_addressed(_plan_path(), label="v4 plan", expected_schema=PLAN_SCHEMA_VERSION)
    rebuilt = build_plan(
        project_root=ROOT,
        parent_root=PARENT_ROOT,
        quarantine_path=QUARANTINE,
        exposure_root=CURRENT,
        historical_plan=frozen,
    )
    assert rebuilt == frozen
    assert [cell["phase"] for cell in frozen["cells"]] == [RECOVERY_PHASE] * 7 + [GLM_PHASE]
    assert [cell["source_closed_work_item_id"] for cell in frozen["cells"]] == [
        source_id for _, source_id, _ in CELL_SPECS
    ]
    assert all(cell["model_id"] != "z-ai/glm-5.2" for cell in frozen["cells"][:6])
    assert frozen["cells"][6]["model_id"] == "z-ai/glm-5.2"
    assert frozen["cells"][7]["source_closed_work_item_id"] == (PARENT_INCOMPLETE_GLM_WORK_ITEM)
    assert not ({cell["task_id"] for cell in frozen["cells"]} & QUARANTINED_TASK_IDS)
    assert all(cell["no_prior_model_task_exposure_at_freeze"] for cell in frozen["cells"])
    assert frozen["counts"] == {
        "recovery_cells": 7,
        "glm_specific_cells": 1,
        "planned_real_arms": 16,
        "planned_synthetic_arms": 0,
        "provider_calls_by_freeze": 0,
        "epicure_calls_by_freeze": 0,
        "fresh_work_item_ids": 8,
    }
    assert frozen["claim_boundary"]["official"] is False
    assert frozen["claim_boundary"]["official_preference_or_uplift_fit_eligible"] is False


def test_identifier_namespaces_are_disjoint_and_budget_is_exact() -> None:
    parent = reconstruct_parent(project_root=ROOT, parent_root=PARENT_ROOT)
    frozen = _load_addressed(_plan_path(), label="v4 plan", expected_schema=PLAN_SCHEMA_VERSION)
    old = {
        value
        for cell in parent.bundle.cells
        for value in (
            cell.cell_id,
            cell.run_id,
            cell.work_item.work_item_id,
            *cell.arm_ids.values(),
            *(str(slot["attempt_id"]) for slot in cell.attempt_slots),
        )
    }
    new = {
        value
        for cell in frozen["cells"]
        for value in (
            cell["cell_id"],
            cell["run_id"],
            cell["work_item_id"],
            *cell["arm_ids"].values(),
            *(str(slot["attempt_id"]) for slot in cell["attempt_slots"]),
        )
    }
    assert not (old & new)
    assert len(new) == sum(5 + len(cell["attempt_slots"]) for cell in frozen["cells"])
    assert frozen["budget"] == {
        "currency": "USD",
        "recovery_phase_worst_case_usd": "3.763091066666666666666666667",
        "glm_phase_worst_case_usd": "0.3419004",
        "total_worst_case_usd": "4.104991466666666666666666667",
    }


def test_content_address_tampering_fails_closed(tmp_path: Path) -> None:
    document = json.loads(_plan_path().read_text(encoding="utf-8"))
    document["status"] = "admissible"
    tampered = tmp_path / _plan_path().name
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IntegrityError, match="content address"):
        _load_addressed(tampered, label="tampered plan", expected_schema=PLAN_SCHEMA_VERSION)


def test_wrong_confirmation_fails_before_any_execution(tmp_path: Path) -> None:
    with pytest.raises(AdmissionDenied, match="requires --confirm"):
        execute_phase(
            preflight_path=tmp_path / "absent.json",
            project_root=ROOT,
            output_root=tmp_path,
            phase=RECOVERY_PHASE,
            confirmation="wrong",
            process_timeout_seconds=1,
        )


def test_cell_failure_does_not_stop_later_unrelated_dispositions() -> None:
    cells = [object() for _ in range(7)]
    seen: list[int] = []

    def execute_one(cell: object) -> tuple[dict[str, object], bool]:
        index = cells.index(cell)
        seen.append(index)
        return {
            "work_item_id": str(index),
            "decision": "failed_closed" if index == 0 else "complete",
        }, True

    outcomes, started = _collect_independent_dispositions(cells, execute_one)  # type: ignore[arg-type]
    assert seen == list(range(7))
    assert len(outcomes) == 7
    assert outcomes[0]["decision"] == "failed_closed"
    assert outcomes[-1]["decision"] == "complete"
    assert started == 7


def test_reasoning_v4_orphan_is_terminal_no_replay_but_still_charged() -> None:
    resolution = _terminal_resolution()
    assert resolution["classification"] == ("verified_pre_request_terminal_no_replay_reservation")
    assert resolution["terminal_no_replay"] is True
    assert resolution["reservation_released"] is False
    assert resolution["reserved_usd_retained_as_conservative_exposure"] == "0.6765315"
    assert resolution["request_boundary"] == {
        "provider_completion_request_events_for_orphan": 0,
        "provider_generation_ids_for_orphan": 0,
        "mcp_sessions_for_orphan": 0,
        "mcp_tool_calls_for_orphan": 0,
        "account_status_events": 1,
    }
    assert resolution["verification_sha256"] == (
        "257ccd4b95ba8498bc4659d83a66595b9890d6dba7301f96145b878cd980a4c0"
    )
    exact_blocker = {
        "gate": "active_reservation_without_source",
        "reservation_entry_sha256": REASONING_V4_GEMINI_RESERVATION_SHA256,
        "reserved_usd": "0.6765315",
        "work_item_id": resolution["work_item_id"],
    }
    assert _terminal_orphan_blocker_matches(exact_blocker, resolution)
    assert not _terminal_orphan_blocker_matches(
        {**exact_blocker, "reserved_usd": "0.6765314"}, resolution
    )


def test_reasoning_v4_journal_tampering_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / REASONING_V4_JOURNAL.name
    tampered.write_bytes(REASONING_V4_JOURNAL.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="journal is not the exact pre-request stop"):
        _terminal_resolution(journal_path=tampered)


def test_fresh_preflight_resolves_only_terminal_blocker_and_retains_exposure() -> None:
    preflight = _current_preflight()
    assert preflight["status"] == "admissible_zero_call_preflight"
    assert preflight["blockers"] == []
    assert preflight["calls"] == {"provider": 0, "epicure": 0}
    assert preflight["budget"] == {
        "currency": "USD",
        "baseline_exposure_usd": "41.37041332666666666666666666",
        "global_active_reservation_usd": "0",
        "supplemental_actual_cost_usd": "0.634367",
        "supplemental_exposure_usd": "5.955756499999999999999999998",
        "terminal_no_replay_reservation_exposure_usd": "0.6765315",
        "terminal_no_replay_blockers_resolved": 1,
        "terminal_reservation_released": False,
        "reasoning_v5_source_conservative_exposure_usd": "0.128245",
        "reasoning_v5_orphan_reservation_usd": "0",
        "current_total_exposure_usd": "47.32616982666666666666666666",
        "recovery_phase_worst_case_usd": "3.763091066666666666666666667",
        "glm_phase_worst_case_usd": "0.3419004",
        "outstanding_worst_case_usd": "4.104991466666666666666666667",
        "projected_total_exposure_usd": "51.43116129333333333333333333",
        "admission_ceiling_usd": "85",
        "hard_cap_usd": "100",
        "admission_allowed": True,
    }
    _verify_terminal_resolution_from_preflight(ROOT, preflight)
    _verify_reasoning_v5_resolution_from_preflight(ROOT, preflight)


def test_reasoning_v5_endpoints_are_closed_and_full_source_cost_is_charged() -> None:
    resolution = verify_reasoning_v5_terminal_endpoints(
        project_root=ROOT,
        route_plan_path=REASONING_V5_ROUTE_PLAN,
        endpoint_snapshot_path=REASONING_V5_ENDPOINT_SNAPSHOT,
        sonnet_root=REASONING_V5_SONNET_ROOT,
        sonnet_receipt_path=REASONING_V5_SONNET_RECEIPT,
        sonnet_audit_path=REASONING_V5_SONNET_AUDIT,
        sonnet_closure_path=REASONING_V5_SONNET_CLOSURE,
        gemini_root=REASONING_V5_GEMINI_ROOT,
        gemini_receipt_path=REASONING_V5_GEMINI_RECEIPT,
        gemini_audit_path=REASONING_V5_GEMINI_AUDIT,
        gemini_closure_path=REASONING_V5_GEMINI_CLOSURE,
        aggregate_audit_path=REASONING_V5_AGGREGATE_AUDIT,
        aggregate_closure_path=REASONING_V5_AGGREGATE_CLOSURE,
    )
    assert resolution["all_v5_identifiers_closed"] is True
    assert resolution["replay_permitted"] is False
    assert resolution["coverage_recovery_blocked"] is False
    assert resolution["accounting"] == {
        "source_conservative_exposure_usd": str(REASONING_V5_TOTAL_SOURCE_EXPOSURE_USD),
        "aggregate_quality_audit_cost_scope_usd": "0.099978",
        "reconciled_failed_arm_cost_outside_aggregate_scope_usd": "0.028267",
        "orphan_reservation_usd": "0",
        "budget_basis": "complete_source_generation_accounting_not_narrow_quality_audit",
    }
    assert (
        resolution["endpoints"]["sonnet"]["reconciled_cost_outside_endpoint_audit_scope_usd"]
        == "0.028267"
    )
    assert resolution["endpoint_metadata"] == {
        "attestations": 3,
        "catalog_http_gets": 6,
        "provider_completion_requests_by_attestations": 0,
        "epicure_calls_by_attestations": 0,
    }


def test_reservation_payload_defers_hash_chain_fields_to_ledger(tmp_path: Path) -> None:
    parent = reconstruct_parent(project_root=ROOT, parent_root=PARENT_ROOT)
    plan = _load_addressed(_plan_path(), label="v4 plan", expected_schema=PLAN_SCHEMA_VERSION)
    cell = _runtime_cells(project_root=ROOT, parent=parent, plan=plan)[0]
    fields = _reservation_fields(
        cell=cell,
        bundle=parent.bundle,
        namespace_sha256="0" * 64,
    )
    protected = {
        "entry_sha256",
        "previous_entry_sha256",
        "recorded_at",
        "schema_version",
        "sequence",
    }
    assert not protected.intersection(fields)
    recorded = append_dataset_ledger_event(
        tmp_path / "ledger.jsonl",
        {"event_type": "reservation_created", **fields},
    )
    assert recorded["schema_version"] == "flavourbench-real-exploratory-ledger-v1"
    assert recorded["sequence"] == 1


def test_failed_attempt_is_proven_pre_reservation_and_preserves_ids() -> None:
    preflight = _current_preflight()
    attempt = preflight["exact_inputs"]["failed_pre_reservation_attempt"]
    assert attempt["classification"] == ("failed_before_reservation_append_and_subprocess_boundary")
    assert attempt["filesystem"] == {
        "inventory": ["ledger.jsonl.lock"],
        "ledger_exists": False,
        "source_directory_exists": False,
        "response_directory_exists": False,
        "run_journals": 0,
        "phase_receipts": 0,
        "phase_closures": 0,
        "phase_audits": 0,
    }
    assert attempt["calls"] == {"provider": 0, "epicure": 0}
    assert attempt["identifier_disposition"] == {
        "planned_cells": 7,
        "reservations_appended": 0,
        "work_item_ids_started": [],
        "run_ids_started": [],
        "attempt_ids_started": [],
        "fresh_identifiers_may_be_preserved": True,
    }


def test_embedded_terminal_resolution_cannot_be_self_asserted() -> None:
    preflight = _current_preflight()
    resolution = preflight["exact_inputs"]["reasoning_v4_terminal_orphan"]
    resolution["request_boundary"]["provider_completion_request_events_for_orphan"] = 1
    with pytest.raises(IntegrityError, match="resolution changed after preflight"):
        _verify_terminal_resolution_from_preflight(ROOT, preflight)
