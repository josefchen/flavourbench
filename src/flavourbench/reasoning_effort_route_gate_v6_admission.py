"""Transactional budget rebase for the frozen Sonnet v6 route gate.

The v6 route plan was frozen before coverage-recovery v4 ran.  This module
binds both terminal coverage phases, reconstructs their source cost, and
supersedes only the historical budget admission.  It never changes the v6
protocol or identifiers.

``freeze`` is network-free.  ``execute`` is the sole provider/MCP-capable
command.  It rebuilds the admission while holding the real global budget lock,
then runs the frozen v6 executor under a private nested lock so the global lock
remains held for the complete two-pair transaction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import reasoning_effort_route_gate_v5 as v5
from . import reasoning_effort_route_gate_v6 as v6

ADMISSION_SCHEMA = "flavourbench-reasoning-effort-sonnet-v6-rebased-admission-v1"
RECEIPT_SCHEMA = "flavourbench-reasoning-effort-sonnet-v6-rebased-receipt-v1"
CONFIRMATION = "RUN_REBASED_REASONING_EFFORT_V6_SONNET_2_PAIRS"

V6_PLAN_SHA = "905f41ba1cd50915d6aa8fc11f5f582930e045e9dc0586dae98549ad21fa6a2c"
V6_HISTORICAL_PREFLIGHT_SHA = (
    "4091db95f115d79aa454821aa0700941284dd1a8795e787c5db7d2a405121d54"
)
V6_EXECUTOR_SHA = "0542db08d466f7a9a8276910bfca91d0fde5b51bbdd1add1c73454d221632291"
COVERAGE_PLAN_SHA = "730b426cfa5b7481446b4618166a2e6f75107c52ec26243283ef10ccbe01c0b8"
COVERAGE_PREFLIGHT_SHA = (
    "c2cf6aa4d6397f6034114dfb9ead0b446895256a7a84705fbd3a55c70d742268"
)
PHASE1_RECEIPT_SHA = "f6d0babc3b6275a5067c8446c4da2783a4fa6a0afea3c725c944eca2838cfcda"
PHASE1_CLOSURE_SHA = "d0dd8c0075f8e2f8dc7a8988a3d2d1ccec6400b3a8b2198c5644a741bc66d287"
PHASE1_AUDIT_SHA = "6166f1451c9b3fad14aed6838a7aa55a6cf577c7c310ee45d758a5ebcccafe15"
PHASE2_RECEIPT_SHA = "be0fce2a5fea950e77a8619124858acdb7fdb3e4552518a9cbe29f9f31c9b910"
PHASE2_CLOSURE_SHA = "3630d08f5d1734a0a0c67f585380b82f3a1fa96beba2502be970d767ba969cab"
PHASE2_AUDIT_SHA = "0698fead0375847f3b72f4198584cce1332afc993ccc98f77c39193861292ab7"

HISTORICAL_BASELINE_USD = Decimal("47.32616982666666666666666666")
PHASE1_ACTUAL_USD = Decimal("0.424886")
PHASE2_ACTUAL_USD = Decimal("0.024079")
V6_RESERVE_USD = Decimal("2.297448")
ADMISSION_CEILING_USD = Decimal("85")
HARD_CAP_USD = Decimal("100")


class RebasedAdmissionError(RuntimeError):
    """A terminal input, source inventory, budget, or lock predicate failed."""


def _artifact(path: Path, digest: str, schema: str) -> dict[str, Any]:
    document = v5._regular_json(path)
    if (
        document.get("artifact_sha256") != digest
        or not v5._artifact_verifies(document, schema)
    ):
        raise RebasedAdmissionError(f"artifact identity or schema failed: {path}")
    return document


def _relative(repo_root: Path, path: Path) -> str:
    return v5._relative(repo_root, path)


def _file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RebasedAdmissionError(f"bound input is not a regular file: {path}")
    return {
        "path": _relative(repo_root, path),
        "bytes": path.stat().st_size,
        "sha256": v5._file_sha256(path),
    }


def _inventory(repo_root: Path, directory: Path) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise RebasedAdmissionError(f"source inventory is not a directory: {directory}")
    records = [
        _file_record(repo_root, path)
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.name.endswith(".lock")
    ]
    if not records or any("inprogress" in record["path"] for record in records):
        raise RebasedAdmissionError("terminal source inventory is empty or in progress")
    return {
        "directory": _relative(repo_root, directory),
        "files": records,
        "inventory_sha256": v5._sha256(records),
    }


def _source_code(repo_root: Path) -> dict[str, Any]:
    wrapper = Path(__file__).resolve()
    executor = Path(v6.__file__).resolve()
    predecessor = Path(v5.__file__).resolve()
    if v5._file_sha256(executor) != V6_EXECUTOR_SHA:
        raise RebasedAdmissionError("frozen v6 executor source differs")
    return {
        "admission_wrapper": _file_record(repo_root, wrapper),
        "frozen_v6_executor": _file_record(repo_root, executor),
        "frozen_v5_adapter": _file_record(repo_root, predecessor),
    }


def _phase_evidence(
    *,
    repo_root: Path,
    phase: str,
    root: Path,
    receipt_path: Path,
    receipt_sha: str,
    closure_path: Path,
    closure_sha: str,
    audit_path: Path,
    audit_sha: str,
    expected_cells: int,
    expected_actual: Decimal,
) -> dict[str, Any]:
    from .frontier_coverage_recovery_v4 import (
        AUDIT_SCHEMA_VERSION,
        CLOSURE_SCHEMA_VERSION,
        RECEIPT_SCHEMA_VERSION,
    )
    from .frontier_coverage_repair_executor import SupplementalRun, _run_accounting
    from .real_dataset_runner import dataset_ledger_state, load_dataset_ledger

    receipt = _artifact(receipt_path, receipt_sha, RECEIPT_SCHEMA_VERSION)
    closure = _artifact(closure_path, closure_sha, CLOSURE_SCHEMA_VERSION)
    audit = _artifact(audit_path, audit_sha, AUDIT_SCHEMA_VERSION)
    ledger_path = root / "ledger.jsonl"
    source_directory = root / "source"
    ledger = load_dataset_ledger(ledger_path)
    reservations, finalizations = dataset_ledger_state(ledger)
    accounting = _run_accounting(
        SupplementalRun(source_directory=source_directory, ledger_path=ledger_path),
        label=f"terminal_coverage_{phase}",
    )
    if (
        receipt.get("status") != "all_phase_cells_received_terminal_disposition"
        or receipt.get("phase") != phase
        or receipt.get("same_identifier_replays") != 0
        or receipt.get("manual_retries") != 0
        or receipt.get("plan_sha256") != COVERAGE_PLAN_SHA
        or receipt.get("preflight_sha256") != COVERAGE_PREFLIGHT_SHA
        or audit.get("decision") != "passed_complete_phase_disposition"
        or audit.get("integrity_failures") != []
        or audit.get("phase") != phase
        or audit.get("receipt_sha256") != receipt_sha
        or audit.get("closure_sha256") != closure_sha
        or audit.get("counts", {}).get("planned_cells") != expected_cells
        or audit.get("counts", {}).get("terminally_dispositioned_cells")
        != expected_cells
        or Decimal(str(audit.get("accounting", {}).get("actual_cost_usd")))
        != expected_actual
        or closure.get("status") != "permanently_closed_all_phase_identifiers"
        or closure.get("phase") != phase
        or closure.get("plan_sha256") != COVERAGE_PLAN_SHA
        or closure.get("preflight_sha256") != COVERAGE_PREFLIGHT_SHA
        or closure.get("receipt_sha256") != receipt_sha
        or closure.get("all_cells_received_terminal_disposition") is not True
        or closure.get("future_execution_with_same_phase_permitted") is not False
        or closure.get("safe_to_replay_any_parent_or_v4_work") is not False
        or len(closure.get("closed_work_item_ids") or []) != expected_cells
        or len(closure.get("closed_run_ids") or []) != expected_cells
        or len(reservations) != expected_cells
        or len(finalizations) != expected_cells
        or accounting.source_count != expected_cells
        or accounting.actual_cost_usd != expected_actual
        or accounting.exposure_usd != expected_actual
        or accounting.orphan_reservation_usd != 0
        or accounting.blockers
        or (ledger[-1]["entry_sha256"] if ledger else None)
        != closure.get("ledger_head_sha256")
    ):
        raise RebasedAdmissionError(f"terminal coverage evidence failed: {phase}")
    receipt_sources = {
        str(item.get("source_artifact_sha256") or "")
        for item in receipt.get("outcomes") or []
    }
    if receipt_sources != set(accounting.artifact_sha256s):
        raise RebasedAdmissionError(f"receipt/source inventory differs: {phase}")
    return {
        "phase": phase,
        "receipt": {
            **_file_record(repo_root, receipt_path),
            "artifact_sha256": receipt_sha,
        },
        "closure": {
            **_file_record(repo_root, closure_path),
            "artifact_sha256": closure_sha,
        },
        "audit": {
            **_file_record(repo_root, audit_path),
            "artifact_sha256": audit_sha,
        },
        "ledger": {
            **_file_record(repo_root, ledger_path),
            "entry_count": len(ledger),
            "head_entry_sha256": ledger[-1]["entry_sha256"],
        },
        "source_inventory": _inventory(repo_root, source_directory),
        "accounting": {
            "source_count": accounting.source_count,
            "actual_cost_usd": v5._decimal_text(accounting.actual_cost_usd),
            "conservative_exposure_usd": v5._decimal_text(accounting.exposure_usd),
            "orphan_reservation_usd": "0",
            "all_reservations_finalized": True,
        },
        "closed_identifiers": {
            "work_item_ids": closure["closed_work_item_ids"],
            "run_ids": closure["closed_run_ids"],
            "attempt_ids_sha256": closure["closed_attempt_ids_sha256"],
            "replay_permitted": False,
        },
    }


def _global_state(
    *,
    repo_root: Path,
    ledger_path: Path,
    artifact_directory: Path,
    corrections_directory: Path,
    reconciliation_directory: Path,
) -> dict[str, Any]:
    from .frontier_contract_runner import load_ledger
    from .frontier_coverage_repair_executor import _global_ledger_state

    active, blockers = _global_ledger_state(
        ledger_path=ledger_path,
        artifact_directory=artifact_directory,
        corrections_directory=corrections_directory,
        reconciliation_directory=reconciliation_directory,
    )
    ledger = load_ledger(ledger_path)
    if active != 0 or blockers:
        raise RebasedAdmissionError("global frontier ledger has active exposure or blockers")
    return {
        "ledger": {
            **_file_record(repo_root, ledger_path),
            "entry_count": len(ledger),
            "head_entry_sha256": ledger[-1]["entry_sha256"] if ledger else None,
        },
        "active_reservation_usd": "0",
        "blockers": [],
        "artifact_directory": _relative(repo_root, artifact_directory),
        "corrections_directory": _relative(repo_root, corrections_directory),
        "reconciliation_directory": _relative(repo_root, reconciliation_directory),
    }


def _verify_generation_core(repo_root: Path, v5_plan: Mapping[str, Any]) -> dict[str, Any]:
    core = (v5_plan.get("source_code") or {}).get("generation_core") or {}
    records = core.get("files") or []
    observed: list[dict[str, Any]] = []
    for record in records:
        path = repo_root / str(record.get("path") or "")
        current = _file_record(repo_root, path)
        if current != dict(record):
            raise RebasedAdmissionError("frozen generation-core source differs")
        observed.append(current)
    if not observed or v5._sha256(observed) != core.get("bundle_sha256"):
        raise RebasedAdmissionError("frozen generation-core bundle does not rederive")
    return {"bundle_sha256": core["bundle_sha256"], "files": observed}


def build_admission(
    *,
    repo_root: Path,
    v6_plan_path: Path,
    historical_preflight_path: Path,
    v5_plan_path: Path,
    v5_baseline_receipt_path: Path,
    coverage_plan_path: Path,
    coverage_preflight_path: Path,
    phase1_root: Path,
    phase1_receipt_path: Path,
    phase1_closure_path: Path,
    phase1_audit_path: Path,
    phase2_root: Path,
    phase2_receipt_path: Path,
    phase2_closure_path: Path,
    phase2_audit_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path,
    global_reconciliation_directory: Path,
    v6_execution_root: Path,
) -> dict[str, Any]:
    from .frontier_coverage_recovery_v4 import (
        PLAN_SCHEMA_VERSION,
        PREFLIGHT_SCHEMA_VERSION,
    )

    plan = _artifact(v6_plan_path, V6_PLAN_SHA, v6.ROUTE_PLAN_SCHEMA)
    v6.validate_route_plan(plan, repo_root=repo_root)
    historical = _artifact(
        historical_preflight_path,
        V6_HISTORICAL_PREFLIGHT_SHA,
        v6.EXECUTION_PLAN_SCHEMA,
    )
    if (
        historical.get("route_plan_sha256") != V6_PLAN_SHA
        or historical.get("budget", {}).get("current_total_exposure_usd")
        != v5._decimal_text(HISTORICAL_BASELINE_USD)
    ):
        raise RebasedAdmissionError("historical v6 preflight boundary differs")
    v5_plan = _artifact(v5_plan_path, v6.V5_PLAN_SHA, v5.ROUTE_PLAN_SCHEMA)
    baseline = _artifact(
        v5_baseline_receipt_path,
        v6.V5_GEMINI_RECEIPT_SHA,
        v5.ENDPOINT_RECEIPT_SCHEMA,
    )
    if Decimal(
        str(baseline.get("final_budget", {}).get("current_total_exposure_usd"))
    ) != HISTORICAL_BASELINE_USD:
        raise RebasedAdmissionError("historical exposure does not rederive from v5 receipt")
    coverage_plan = _artifact(coverage_plan_path, COVERAGE_PLAN_SHA, PLAN_SCHEMA_VERSION)
    coverage_preflight = _artifact(
        coverage_preflight_path, COVERAGE_PREFLIGHT_SHA, PREFLIGHT_SCHEMA_VERSION
    )
    if (
        coverage_preflight.get("plan", {}).get("sha256") != COVERAGE_PLAN_SHA
        or Decimal(
            str(coverage_preflight.get("budget", {}).get("current_total_exposure_usd"))
        )
        != HISTORICAL_BASELINE_USD
        or coverage_preflight.get("calls") != {"epicure": 0, "provider": 0}
    ):
        raise RebasedAdmissionError("coverage preflight does not share the v6 baseline")
    phase1 = _phase_evidence(
        repo_root=repo_root,
        phase="untouched_recovery",
        root=phase1_root,
        receipt_path=phase1_receipt_path,
        receipt_sha=PHASE1_RECEIPT_SHA,
        closure_path=phase1_closure_path,
        closure_sha=PHASE1_CLOSURE_SHA,
        audit_path=phase1_audit_path,
        audit_sha=PHASE1_AUDIT_SHA,
        expected_cells=7,
        expected_actual=PHASE1_ACTUAL_USD,
    )
    phase2 = _phase_evidence(
        repo_root=repo_root,
        phase="glm_specific_replacement",
        root=phase2_root,
        receipt_path=phase2_receipt_path,
        receipt_sha=PHASE2_RECEIPT_SHA,
        closure_path=phase2_closure_path,
        closure_sha=PHASE2_CLOSURE_SHA,
        audit_path=phase2_audit_path,
        audit_sha=PHASE2_AUDIT_SHA,
        expected_cells=1,
        expected_actual=PHASE2_ACTUAL_USD,
    )
    global_state = _global_state(
        repo_root=repo_root,
        ledger_path=global_ledger_path,
        artifact_directory=global_artifact_directory,
        corrections_directory=global_corrections_directory,
        reconciliation_directory=global_reconciliation_directory,
    )
    if v6_execution_root.exists() and any(
        path.is_file() and not path.name.endswith(".lock")
        for path in v6_execution_root.rglob("*")
    ):
        raise RebasedAdmissionError("v6 execution root is not fresh")
    current = HISTORICAL_BASELINE_USD + PHASE1_ACTUAL_USD + PHASE2_ACTUAL_USD
    projected = current + V6_RESERVE_USD
    if projected > ADMISSION_CEILING_USD or projected > HARD_CAP_USD:
        raise RebasedAdmissionError("rebased v6 projection exceeds a budget threshold")
    source_code = _source_code(repo_root)
    generation_core = _verify_generation_core(repo_root, v5_plan)
    return {
        "schema_version": ADMISSION_SCHEMA,
        "record_role": "terminal_coverage_rebased_transactional_v6_admission",
        "status": "admissible_zero_call_rebased_preflight",
        "v6": {
            "route_plan": {
                **_file_record(repo_root, v6_plan_path),
                "artifact_sha256": plan["artifact_sha256"],
            },
            "historical_preflight": {
                **_file_record(repo_root, historical_preflight_path),
                "artifact_sha256": historical["artifact_sha256"],
                "execution_authority_superseded": True,
            },
            "execution_root": _relative(repo_root, v6_execution_root),
            "work_item_ids": [item["work_item_id"] for item in plan["work_items"]],
            "run_ids": [item["run_id"] for item in plan["work_items"]],
            "identifiers_started_before_rebase": False,
            "identifiers_preserved_without_replay": True,
            "protocol_or_source_contract_changed": False,
        },
        "coverage": {
            "plan": {
                **_file_record(repo_root, coverage_plan_path),
                "artifact_sha256": coverage_plan["artifact_sha256"],
            },
            "preflight": {
                **_file_record(repo_root, coverage_preflight_path),
                "artifact_sha256": coverage_preflight["artifact_sha256"],
            },
            "phase1": phase1,
            "phase2": phase2,
            "all_eight_cells_terminally_dispositioned": True,
            "all_phase_identifiers_closed_no_replay": True,
        },
        "global_frontier_state": global_state,
        "budget": {
            "currency": "USD",
            "historical_v5_total_exposure_usd": v5._decimal_text(
                HISTORICAL_BASELINE_USD
            ),
            "terminal_coverage_phase1_exposure_usd": v5._decimal_text(
                PHASE1_ACTUAL_USD
            ),
            "terminal_coverage_phase2_exposure_usd": v5._decimal_text(
                PHASE2_ACTUAL_USD
            ),
            "current_rebased_total_exposure_usd": v5._decimal_text(current),
            "v6_two_pair_worst_case_usd": v5._decimal_text(V6_RESERVE_USD),
            "projected_total_exposure_usd": v5._decimal_text(projected),
            "admission_ceiling_usd": v5._decimal_text(ADMISSION_CEILING_USD),
            "hard_cap_usd": v5._decimal_text(HARD_CAP_USD),
            "admission_allowed": True,
        },
        "transaction": {
            "confirmation": CONFIRMATION,
            "real_global_lock_held_across_both_v6_pairs": True,
            "admission_rebuilt_under_real_global_lock": True,
            "frozen_v6_executor_uses_private_nested_lock": True,
            "all_two_pair_worst_case_reserved_by_outer_admission": True,
            "provider_calls_made_by_freeze": 0,
            "epicure_calls_made_by_freeze": 0,
        },
        "source_code": source_code,
        "generation_core": generation_core,
        "claim_boundary": plan["claim_boundary"],
    }


def _write_artifact(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = v5._sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RebasedAdmissionError(f"content-addressed conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _default_paths(repo_root: Path) -> dict[str, Path]:
    project = repo_root / "flavourbench"
    current = project / "artifacts/season1/current-quality-run"
    coverage = current / "frontier-coverage-recovery-v4"
    v6_root = current / "reasoning-effort-sensitivity-v6"
    return {
        "project": project,
        "output": v6_root / "admission",
        "v6_execution_root": v6_root / "sonnet",
        "v6_plan": v6_root
        / "route-gate/reasoning-effort-v6-sonnet-route-gate-plan-"
        f"{V6_PLAN_SHA}.json",
        "historical_preflight": v6_root
        / "route-gate/reasoning-effort-v6-sonnet-execution-plan-"
        f"{V6_HISTORICAL_PREFLIGHT_SHA}.json",
        "v5_plan": current
        / "reasoning-effort-sensitivity-v5/route-gate/"
        "reasoning-effort-v5-route-gate-plan-"
        f"{v6.V5_PLAN_SHA}.json",
        "v5_baseline_receipt": current
        / "reasoning-effort-sensitivity-v5/gemini/receipts/"
        "reasoning-effort-v5-gemini-receipt-"
        f"{v6.V5_GEMINI_RECEIPT_SHA}.json",
        "coverage_plan": coverage
        / f"frontier-coverage-recovery-v4-plan-{COVERAGE_PLAN_SHA}.json",
        "coverage_preflight": coverage
        / f"frontier-coverage-recovery-v4-preflight-{COVERAGE_PREFLIGHT_SHA}.json",
        "phase1_root": coverage / "untouched-recovery",
        "phase1_receipt": coverage
        / "frontier-coverage-recovery-v4-untouched_recovery-receipt-"
        f"{PHASE1_RECEIPT_SHA}.json",
        "phase1_closure": coverage
        / "frontier-coverage-recovery-v4-untouched_recovery-closure-"
        f"{PHASE1_CLOSURE_SHA}.json",
        "phase1_audit": coverage
        / "frontier-coverage-recovery-v4-untouched_recovery-audit-"
        f"{PHASE1_AUDIT_SHA}.json",
        "phase2_root": coverage / "glm-specific-replacement",
        "phase2_receipt": coverage
        / "frontier-coverage-recovery-v4-glm_specific_replacement-receipt-"
        f"{PHASE2_RECEIPT_SHA}.json",
        "phase2_closure": coverage
        / "frontier-coverage-recovery-v4-glm_specific_replacement-closure-"
        f"{PHASE2_CLOSURE_SHA}.json",
        "phase2_audit": coverage
        / "frontier-coverage-recovery-v4-glm_specific_replacement-audit-"
        f"{PHASE2_AUDIT_SHA}.json",
        "global_ledger": project / "artifacts/frontier-contract/ledger.jsonl",
        "global_artifacts": project / "artifacts/live-smoke",
        "global_corrections": project / "artifacts/corrections",
        "global_reconciliations": project
        / "artifacts/frontier-contract/reconciliations",
    }


def _build_default(repo_root: Path) -> dict[str, Any]:
    paths = _default_paths(repo_root)
    return build_admission(
        repo_root=repo_root,
        v6_plan_path=paths["v6_plan"],
        historical_preflight_path=paths["historical_preflight"],
        v5_plan_path=paths["v5_plan"],
        v5_baseline_receipt_path=paths["v5_baseline_receipt"],
        coverage_plan_path=paths["coverage_plan"],
        coverage_preflight_path=paths["coverage_preflight"],
        phase1_root=paths["phase1_root"],
        phase1_receipt_path=paths["phase1_receipt"],
        phase1_closure_path=paths["phase1_closure"],
        phase1_audit_path=paths["phase1_audit"],
        phase2_root=paths["phase2_root"],
        phase2_receipt_path=paths["phase2_receipt"],
        phase2_closure_path=paths["phase2_closure"],
        phase2_audit_path=paths["phase2_audit"],
        global_ledger_path=paths["global_ledger"],
        global_artifact_directory=paths["global_artifacts"],
        global_corrections_directory=paths["global_corrections"],
        global_reconciliation_directory=paths["global_reconciliations"],
        v6_execution_root=paths["v6_execution_root"],
    )


async def execute_rebased(
    *,
    repo_root: Path,
    admission_path: Path,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    from .frontier_contract_runner import _exclusive_runner_lock
    from .frontier_coverage_repair_executor import SupplementalRun, _run_accounting

    admission = v5._regular_json(admission_path)
    if not v5._artifact_verifies(admission, ADMISSION_SCHEMA):
        raise RebasedAdmissionError("rebased admission content address failed")
    paths = _default_paths(repo_root)
    with _exclusive_runner_lock(paths["global_ledger"]):
        rebuilt = _build_default(repo_root)
        if v5._sha256(rebuilt) != admission["artifact_sha256"]:
            raise RebasedAdmissionError(
                "rebased admission changed before the transactional lock"
            )
        baseline = _artifact(
            paths["v5_baseline_receipt"],
            v6.V5_GEMINI_RECEIPT_SHA,
            v5.ENDPOINT_RECEIPT_SCHEMA,
        )
        plan = _artifact(paths["v6_plan"], V6_PLAN_SHA, v6.ROUTE_PLAN_SCHEMA)
        result = await v6.execute(
            plan=plan,
            root=paths["v6_execution_root"],
            baseline_receipt=baseline,
            repo_root=repo_root,
            global_budget_lock_path=paths["output"] / "private-executor-lock.jsonl",
            api_base=api_base,
            api_key=api_key,
        )
        accounting = _run_accounting(
            SupplementalRun(
                source_directory=paths["v6_execution_root"] / "source",
                ledger_path=paths["v6_execution_root"] / "ledger.jsonl",
            ),
            label="reasoning_effort_v6_rebased_post_execution",
        )
        current = (
            HISTORICAL_BASELINE_USD
            + PHASE1_ACTUAL_USD
            + PHASE2_ACTUAL_USD
            + accounting.exposure_usd
            + accounting.orphan_reservation_usd
        )
        if current > HARD_CAP_USD:
            raise RebasedAdmissionError("post-execution exposure exceeded the hard cap")
        wrapper_receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "record_role": "transactionally_rebased_v6_execution_receipt",
            "admission_sha256": admission["artifact_sha256"],
            "v6_route_plan_sha256": V6_PLAN_SHA,
            "v6_execution_receipt": {
                "path": _relative(repo_root, Path(result["path"])),
                "artifact_sha256": result["document"]["artifact_sha256"],
                "status": result["document"]["status"],
            },
            "budget": {
                "historical_v5_total_exposure_usd": v5._decimal_text(
                    HISTORICAL_BASELINE_USD
                ),
                "terminal_coverage_exposure_usd": v5._decimal_text(
                    PHASE1_ACTUAL_USD + PHASE2_ACTUAL_USD
                ),
                "v6_source_exposure_usd": v5._decimal_text(accounting.exposure_usd),
                "v6_orphan_reservation_usd": v5._decimal_text(
                    accounting.orphan_reservation_usd
                ),
                "complete_rebased_total_exposure_usd": v5._decimal_text(current),
                "hard_cap_usd": v5._decimal_text(HARD_CAP_USD),
                "hard_cap_respected": True,
            },
            "transaction": {
                "real_global_lock_held_across_both_pairs": True,
                "admission_rebuilt_under_lock": True,
                "uncertain_delivery_replayed": False,
            },
            "quality_observations": 0,
            "rank_eligible": False,
        }
        receipt_path = _write_artifact(
            paths["output"] / "receipts",
            "reasoning-effort-v6-rebased-receipt",
            wrapper_receipt,
        )
    return {
        "rebased_receipt": str(receipt_path.resolve()),
        "rebased_receipt_sha256": v5._regular_json(receipt_path)["artifact_sha256"],
        "v6_receipt": str(Path(result["path"]).resolve()),
        "v6_receipt_sha256": result["document"]["artifact_sha256"],
        "v6_status": result["document"]["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--admission", type=Path, required=True)
    execute.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    execute.add_argument("--confirm", required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    repo_root = arguments.repo_root.resolve()
    paths = _default_paths(repo_root)
    if arguments.command == "freeze":
        payload = _build_default(repo_root)
        path = _write_artifact(
            paths["output"], "reasoning-effort-v6-rebased-admission", payload
        )
        document = v5._regular_json(path)
        print(
            json.dumps(
                {
                    "status": document["status"],
                    "output": str(path.resolve()),
                    "artifact_sha256": document["artifact_sha256"],
                    "budget": document["budget"],
                    "provider_calls": 0,
                    "epicure_calls": 0,
                    "confirmation": CONFIRMATION,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.confirm != CONFIRMATION:
        raise RebasedAdmissionError(f"execution requires --confirm {CONFIRMATION}")
    result = asyncio.run(
        execute_rebased(
            repo_root=repo_root,
            admission_path=arguments.admission,
            api_base=arguments.api_base,
            api_key=v5._api_key(),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
