"""Offline and crash-recovery tests for the V8 incident-recovery V9."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier
import flavourbench.reasoning_effort_v8_incident_recovery_v9 as recovery

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single(pattern: str) -> Path:
    matches = sorted((REPO_ROOT / recovery.V9_ROOT / "offline").glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def _bundle() -> Path:
    return _single("v9-recovery-review-bundle-*.json")


def _incident() -> dict[str, Any]:
    return _json(_single("v8-live-audit-incident-*.json"))


def _protected_snapshot(root: Path) -> dict[str, bytes]:
    paths = [
        root / recovery.GLOBAL_LEDGER_PATH,
        root / recovery.SOURCE_ROOT / recovery.SOURCE_FILENAME,
        root / recovery.SOURCE_ROOT / recovery.JOURNAL_FILENAME,
        *sorted((root / recovery.V8_ROOT).rglob("*")),
    ]
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def _copy_state(tmp_path: Path, *, include_offline: bool) -> Path:
    state = tmp_path / "state"
    directories = [
        recovery.V8_ROOT,
        recovery.SOURCE_ROOT,
        "flavourbench/artifacts/frontier-contract",
    ]
    if include_offline:
        directories.append(f"{recovery.V9_ROOT}/offline")
    for relative in directories:
        source = REPO_ROOT / relative
        destination = state / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    return state


def _patch_pair_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    audit = copy.deepcopy(_incident()["corrected_offline_pair_audit"])
    monkeypatch.setattr(
        recovery,
        "pair_audit_v9",
        lambda **_kwargs: copy.deepcopy(audit),
    )


def _go(plan: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    return recovery._with_hash(
        {
            "schema_version": recovery.GOVERNANCE_GO_SCHEMA,
            "record_role": "independent_v9_zero_call_recovery_go",
            "decision": "go_for_one_source_import_and_27_no_delivery_releases",
            "recovery_plan_sha256": plan["artifact_sha256"],
            "reviewed_bundle_sha256": bundle["artifact_sha256"],
            "reviewed_incident_sha256": plan["v8_incident_sha256"],
            "reviewed_source_closure_sha256": plan["source_closure_sha256"],
            "reviewed_no_delivery_proof_count": 27,
            "reviewed_fresh_identifier_count": 27,
            "maximum_canonical_ledger_events": 28,
            "new_provider_requests": 0,
            "new_epicure_calls": 0,
            "catalog_requests": 0,
            "new_reservations": 0,
            "continuation_authorized": False,
            "reviewer_is_executor": False,
            "reviewer_is_v9_builder": False,
            "independent_technical_review_completed": True,
            "reviewer_identity_commitment_sha256": "7" * 64,
            "reviewed_at": "2026-08-08T20:00:00Z",
            "reviewer_role": "independent technical reviewer",
        }
    )


def test_actual_v9_bundle_rederives_offline_without_mutation() -> None:
    before = _protected_snapshot(REPO_ROOT)
    result = recovery.verify_bundle(
        bundle_path=_bundle(),
        state_root=REPO_ROOT,
        code_root=REPO_ROOT,
    )
    assert result == {
        "decision": "offline_v9_bundle_verified_no_apply_authority",
        "bundle_sha256": _json(_bundle())["artifact_sha256"],
        "recovery_plan_sha256": result["recovery_plan_sha256"],
        "completed_source_pairs": 1,
        "continuation_pairs": 27,
        "provider_or_epicure_calls": 0,
        "ledger_writes": 0,
    }
    assert recovery._verify_live_artifact is frontier._verify_live_artifact
    assert _protected_snapshot(REPO_ROOT) == before


def test_independent_go_requires_a_parseable_utc_review_timestamp() -> None:
    bundle = _json(_bundle())
    plan = _json(REPO_ROOT / bundle["recovery_plan"]["path"])
    governance_go = _go(plan, bundle)
    governance_go["reviewed_at"] = "not-a-timestamp"
    governance_go = recovery._with_hash(governance_go)
    with pytest.raises(recovery.RecoveryError, match="exact independent V9"):
        recovery._verify_go(
            governance_go=governance_go,
            recovery_plan=plan,
            bundle=bundle,
        )


def test_actual_forensics_correct_only_the_two_legacy_audit_defects() -> None:
    forensic = recovery.verify_forensic_state(
        state_root=REPO_ROOT,
        exact_global_length=True,
    )
    audit = forensic.pair_audit
    assert audit["decision"] == "passed_all_predicates"
    assert audit["failures"] == []
    assert audit["request_semantics"] == "reasoning_effort_explicit_low"
    assert audit["legacy_request_semantics_label"] == "reasoning_effort_explicit_high"
    assert audit["execution_policy_v9"] == {
        "decision": "exact_v1_v8_concurrent_policy_verified",
        "frozen_execution_policy_sha256": recovery.FROZEN_CONCURRENT_POLICY_SHA256,
        "source_execution_policy_sha256": recovery.FROZEN_CONCURRENT_POLICY_SHA256,
        "manifest_execution_policy_sha256": recovery.FROZEN_CONCURRENT_POLICY_SHA256,
        "pair_arm_scheduling": "concurrent",
        "legacy_v4_derived_pair_arm_scheduling": "sequential",
        "legacy_v4_derived_execution_policy_sha256": (
            recovery.LEGACY_SEQUENTIAL_POLICY_SHA256
        ),
        "legacy_failure_removed_only_after_exact_policy_equality": (
            "variant_execution_policy_mismatch"
        ),
    }
    assert audit["identifiers"]["generation_ids"] == list(recovery.GENERATION_IDS)
    assert audit["accounting"]["actual_cost_micros"] == recovery.ACTUAL_COST_MICROS
    assert len(forensic.continuation_items) == 27
    assert all(
        item["delivery_evidence"]
        == {
            "item_execution_started_events": 0,
            "provider_request_journals": 0,
            "source_artifacts": 0,
            "canonical_finalizations_at_v8_stop": 0,
        }
        for item in forensic.continuation_items
    )
    assert all(
        item["v8_identifier_disposition"]
        == "release_after_content_addressed_v2_no_delivery_proof"
        and item["same_identifier_replay_permitted"] is False
        and item["fresh_reservation_required_for_any_future_delivery"] is True
        for item in forensic.continuation_items
    )


def test_offline_freeze_is_byte_deterministic_and_zero_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pair_audit(monkeypatch)
    state = _copy_state(tmp_path, include_offline=False)
    output = state / recovery.V9_ROOT / "offline"
    before = _protected_snapshot(state)
    first = recovery.freeze(
        state_root=state,
        code_root=REPO_ROOT,
        output_dir=output,
    )
    first_files = {
        str(path.relative_to(output)): path.read_bytes()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    assert len(list((output / "reconciliations").glob("*.json"))) == 27
    assert _protected_snapshot(state) == before
    shutil.rmtree(output)
    second = recovery.freeze(
        state_root=state,
        code_root=REPO_ROOT,
        output_dir=output,
    )
    second_files = {
        str(path.relative_to(output)): path.read_bytes()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    assert set(first) == set(second)
    assert first_files == second_files
    assert _protected_snapshot(state) == before


def test_every_durable_v9_recovery_cut_replays_without_duplicate_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_pair_audit(monkeypatch)
    monkeypatch.setattr(
        recovery.source_closure_v9,
        "verify_source_closure",
        lambda **_kwargs: None,
    )
    state = _copy_state(tmp_path, include_offline=True)
    before_v8 = recovery._tree_snapshot(state, state / recovery.V8_ROOT)
    bundle = next((state / recovery.V9_ROOT / "offline").glob(
        "v9-recovery-review-bundle-*.json"
    ))
    bundle_document = _json(bundle)
    plan_path = state / bundle_document["recovery_plan"]["path"]
    plan = _json(plan_path)
    go = _go(plan, bundle_document)
    go_path = state / recovery.V9_ROOT / (
        f"v9-recovery-independent-go-{go['artifact_sha256']}.json"
    )
    go_path.parent.mkdir(parents=True, exist_ok=True)
    go_path.write_text(json.dumps(go), encoding="utf-8")
    output = state / recovery.V9_ROOT / "runs/recovery"

    def crash_at(point: str) -> None:
        tripped = False

        def inject(observed: str) -> None:
            nonlocal tripped
            if not tripped and observed == point:
                tripped = True
                raise recovery.SimulatedCrash(point)

        with pytest.raises(recovery.SimulatedCrash, match=point):
            recovery.apply_recovery(
                state_root=state,
                code_root=REPO_ROOT,
                bundle_path=bundle,
                governance_go_path=go_path,
                output_dir=output,
                confirmation=recovery.CONFIRMATION,
                failure_injector=inject,
            )
        assert tripped

    for cut in (
        "before_global_lock",
        "after_locks_before_revalidation",
        "before_canonical_artifact_append",
        "after_canonical_artifact_append",
    ):
        crash_at(cut)
    for ordinal in range(1, 28):
        crash_at(f"before_no_delivery_reconciliation_append_{ordinal:02d}")
        crash_at(f"after_no_delivery_reconciliation_append_{ordinal:02d}")
    for cut in (
        "before_local_terminal_append",
        "after_local_terminal_append",
        "before_receipt_write",
        "after_receipt_write",
    ):
        crash_at(cut)
    receipt = recovery.apply_recovery(
        state_root=state,
        code_root=REPO_ROOT,
        bundle_path=bundle,
        governance_go_path=go_path,
        output_dir=output,
        confirmation=recovery.CONFIRMATION,
    )
    receipt_document = _json(receipt)
    assert receipt_document["canonical_artifact_events_for_completed_item"] == 1
    assert receipt_document["canonical_no_delivery_reconciliation_events"] == 27
    assert receipt_document["released_never_started_reservation_usd"] == (
        recovery.UNUSED_RESERVATION_USD
    )
    assert receipt_document["post_recovery_total_exposure_usd"] == (
        recovery.POST_RECOVERY_EXPOSURE_USD
    )
    entries = frontier.load_ledger(state / recovery.GLOBAL_LEDGER_PATH)
    v8_reservations = recovery.v8_executor._campaign_global_reservations(
        plan=_json(state / recovery.V8_PLAN_PATH), entries=entries
    )
    v8_ids = {entry["entry_sha256"] for entry in v8_reservations.values()}
    dispositions = [
        entry
        for entry in entries
        if entry.get("event_type")
        in {"artifact_recorded", "no_artifact_reconciliation_recorded"}
        and entry.get("reservation_entry_sha256") in v8_ids
    ]
    assert len(dispositions) == 28
    assert sum(entry["event_type"] == "artifact_recorded" for entry in dispositions) == 1
    assert (
        sum(
            entry["event_type"] == "no_artifact_reconciliation_recorded"
            for entry in dispositions
        )
        == 27
    )
    assert not v8_ids.intersection(frontier.active_ledger_reservations(entries))
    assert len(recovery._load_local_ledger(output / "ledger.jsonl")) == 1
    assert recovery._tree_snapshot(state, state / recovery.V8_ROOT) == before_v8
    assert recovery._file_sha256(
        state / recovery.SOURCE_ROOT / recovery.SOURCE_FILENAME
    ) == recovery.SOURCE_FILE_SHA256
    assert recovery._file_sha256(
        state / recovery.SOURCE_ROOT / recovery.JOURNAL_FILENAME
    ) == recovery.JOURNAL_SHA256


def test_fresh_v9_continuation_reuses_no_v8_identifier_or_reservation() -> None:
    plan = _json(_single("v9-recovery-plan-*.json"))
    assert plan["record_role"] == (
        "append_only_v8_incident_recovery_and_fresh_unreserved_continuation_plan"
    )
    old = plan["continuation"]["retired_v8_items"]
    fresh = plan["continuation"]["fresh_v9_items"]
    old_ids = {
        value
        for item in old
        for value in (
            item["work_item_id"],
            item["run_id"],
            item["canonical_reservation_entry_sha256"],
        )
    }
    fresh_ids = {
        value
        for item in fresh
        for value in (
            item["work_item_id"],
            item["run_id"],
            *item["arm_ids"],
            *[slot["attempt_id"] for slot in item["attempt_slots"]],
        )
    }
    assert len(fresh) == 27
    assert not old_ids.intersection(fresh_ids)
    assert all(item["reservation_status"] == "not_created" for item in fresh)
    assert all(item["live_execution_authorized"] is False for item in fresh)
    assert plan["continuation"]["future_worst_case_reserve_usd"] == (
        recovery.UNUSED_RESERVATION_USD
    )
