from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.frontier_contract_runner import IntegrityError
from flavourbench.frontier_coverage_continuation_executor import (
    EXECUTION_CONFIRMATION,
    EXPECTED_ARMS,
    PREFLIGHT_SCHEMA_VERSION,
    V1_MATERIALIZATION_SHA256,
    V2_PLAN_SHA256,
    V3_PLAN_SHA256,
    RunPaths,
    _load_addressed,
    _recovery_evidence,
    _write_addressed,
    build_runtime_bundle,
    execute_preflight,
)
from flavourbench.run_journal import RunJournal

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
V1_ROOT = CURRENT / "frontier-coverage-repair-execution-v1"
CONTINUATION_ROOT = CURRENT / "frontier-coverage-continuation-v2"
V2_PLAN = CONTINUATION_ROOT / f"frontier-coverage-continuation-plan-{V2_PLAN_SHA256}.json"
V3_PLAN = CONTINUATION_ROOT / f"frontier-coverage-replacement-plan-{V3_PLAN_SHA256}.json"
V1_MATERIALIZATION = V1_ROOT / (
    f"frontier-coverage-materialization-{V1_MATERIALIZATION_SHA256}.json"
)
TASK_VALIDITY = ROOT / "artifacts/season1/task-validity/development-v2" / (
    "development-task-validity-v2-"
    "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
)
ROUTE_MANIFESTS = (
    CURRENT
    / "manifest-v29-high-resource"
    / "flavourbench-routed-unranked-"
    "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json",
    CURRENT
    / "manifest-v42-high-resource-cohere-direct"
    / "flavourbench-cohere-unranked-"
    "fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json",
)
STOPPED_AUDIT = CONTINUATION_ROOT / (
    "frontier-coverage-stopped-run-audit-"
    "b0990b3b8869325771433cccd8a390a0e48038cf07637cac7ee244a39e9ca4d5.json"
)
ORPHAN_CLOSURE = CONTINUATION_ROOT / (
    "frontier-coverage-orphan-closure-"
    "3cb144abd1162447e3e64ba0b703ea09d9ead595d141e2dbf1ffb0103d27e370.json"
)


def _bundle():  # type: ignore[no-untyped-def]
    return build_runtime_bundle(
        project_root=ROOT,
        v2_plan_path=V2_PLAN,
        v3_plan_path=V3_PLAN,
        v1_materialization_path=V1_MATERIALIZATION,
        task_validity_path=TASK_VALIDITY,
        route_manifest_paths=ROUTE_MANIFESTS,
        stopped_audit_path=STOPPED_AUDIT,
        orphan_closure_path=ORPHAN_CLOSURE,
        v1_ledger_path=V1_ROOT / "ledger.jsonl",
        v1_source_directory=V1_ROOT / "source",
        v1_response_directory=V1_ROOT / "responses",
    )


def test_exact_v2_v3_bundle_is_fresh_unranked_and_graph_only() -> None:
    bundle = _bundle()
    assert len(bundle.cells) == 9
    assert sum(len(cell.conditions) for cell in bundle.cells) == EXPECTED_ARMS
    assert {cell.plan_kind for cell in bundle.cells} == {
        "v2_continuation",
        "v3_replacement",
    }
    work_ids = {cell.work_item.work_item_id for cell in bundle.cells}
    attempt_ids = {
        str(slot["attempt_id"]) for cell in bundle.cells for slot in cell.attempt_slots
    }
    assert len(work_ids) == 9
    assert len(attempt_ids) == sum(len(cell.attempt_slots) for cell in bundle.cells)
    assert bundle.document["claim_boundary"] == {
        "development_only": True,
        "official": False,
        "rank_eligible": False,
        "official_preference_or_uplift_fit_eligible": False,
        "permitted_analysis": "comparison_graph_diagnostics_only",
        "replacement_observations_are_not_missing_at_random": True,
        "quality_judgments": 0,
    }
    assert bundle.document["v1_disposition"]["failure_records_preserved"] is True
    assert bundle.document["v1_disposition"]["retired_work_items_replayed"] == 0


def test_content_address_tampering_and_filename_substitution_fail_closed(
    tmp_path: Path,
) -> None:
    document = json.loads(V2_PLAN.read_text(encoding="utf-8"))
    document["status"] = "admissible"
    tampered = tmp_path / V2_PLAN.name
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IntegrityError, match="content address"):
        _load_addressed(
            tampered,
            label="tampered v2",
            expected_schema=document["schema_version"],
            expected_digest=V2_PLAN_SHA256,
        )
    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(V2_PLAN.read_bytes())
    with pytest.raises(IntegrityError, match="content address"):
        _load_addressed(
            renamed,
            label="renamed v2",
            expected_schema=document["schema_version"],
            expected_digest=V2_PLAN_SHA256,
        )


def test_recovery_classifier_distinguishes_pre_request_from_uncertain_delivery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    paths = RunPaths(tmp_path, source, tmp_path / "responses", tmp_path / "ledger.jsonl")
    pre_work = "a" * 64
    pre = RunJournal.create(
        source,
        run_id="00000000-0000-4000-8000-000000000001",
        metadata={"dataset_work_item_id": pre_work},
    )
    pre.finalize({"status": "pre_provider_failure"})
    assert _recovery_evidence(paths, pre_work)["delivery_classification"] == (
        "pre_request_or_safe_provider_rejection_no_generation"
    )

    uncertain_work = "b" * 64
    uncertain = RunJournal.create(
        source,
        run_id="00000000-0000-4000-8000-000000000002",
        metadata={"dataset_work_item_id": uncertain_work},
    )
    uncertain.append(
        "provider_attempt",
        {
            "attempt_id": "attempt-1",
            "arm_id": "arm-1",
            "phase": "planning",
            "attempt_index": 0,
            "event_type": "request_started",
        },
    )
    evidence = _recovery_evidence(paths, uncertain_work)
    assert evidence["delivery_classification"] == (
        "uncertain_delivery_or_unreconciled_generation"
    )
    assert evidence["parent_policy_safe_to_replay"] is False


def test_execute_rejects_source_drift_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "admissible_zero_call_preflight",
        "source_code": {"files": [], "bundle_sha256": "0" * 64},
    }
    path = _write_addressed(preflight, directory=tmp_path, prefix="preflight")
    called = False

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(IntegrityError, match="execution source changed"):
        execute_preflight(
            preflight_path=path,
            project_root=ROOT,
            output_root=tmp_path / "execution",
            confirmation=EXECUTION_CONFIRMATION,
            process_timeout_seconds=1,
        )
    assert called is False
