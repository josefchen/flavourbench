from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import flavourbench.reasoning_effort_pregen_recovery_v1 as recovery
from flavourbench import reasoning_effort_full_study_v1 as study

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PLAN = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/plan"
    / (
        "reasoning-effort-task-wave-plan-v2-"
        "03731cb5e509bc40ec733bc5c55ee91ad035b04e1c4adaf64684437751fb1f0c.json"
    )
)
HUMAN = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-human-protocol-v1"
    / (
        "reasoning-effort-human-protocol-"
        "42fb1b5ea606034d4eb62eb813c957b87ffee44392e1c8f11322bf61fe7002ea.json"
    )
)
BOUND = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2/bound-preflight"
    / (
        "reasoning-effort-bound-admission-preflight-v2-"
        "9c5cb664b5708fccfa49e20f8c362736786e705b06444ed7c59f5013181e8d8e.json"
    )
)
RECOVERY_ROOT = (
    ROOT
    / "artifacts/season1/current-quality-run/reasoning-effort-task-waves-v2"
    / "pregen-recovery-v1"
)
INCIDENT = RECOVERY_ROOT / (
    "reasoning-effort-pregen-pipeline-incident-"
    "5457d837103a165cfab969ea9f50a9b640e80c4b6a73c4775225b49539942402.json"
)
CONTRACT = RECOVERY_ROOT / (
    "reasoning-effort-pregen-recovery-contract-"
    "40c5d1dfec92bf50250cd144999263f11a6c803b9591afd0e8916bc7acb36039.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_zero_call_incident_and_recovery_contract_are_exactly_bound() -> None:
    plan, human, bound = _json(PLAN), _json(HUMAN), _json(BOUND)
    assert plan["artifact_sha256"] == recovery.PLAN_SHA256
    assert human["artifact_sha256"] == recovery.HUMAN_PROTOCOL_SHA256
    assert bound["artifact_sha256"] == recovery.BOUND_PREFLIGHT_SHA256
    assert (
        study._sha256({key: value for key, value in plan.items() if key != "artifact_sha256"})
        == plan["artifact_sha256"]
    )
    assert (
        study._sha256({key: value for key, value in human.items() if key != "artifact_sha256"})
        == human["artifact_sha256"]
    )
    assert (
        study._sha256({key: value for key, value in bound.items() if key != "artifact_sha256"})
        == bound["artifact_sha256"]
    )
    incident = _json(INCIDENT)
    contract = _json(CONTRACT)
    assert study._artifact_ok(incident, recovery.INCIDENT_SCHEMA)
    assert study._artifact_ok(contract, recovery.CONTRACT_SCHEMA)
    recovery._verify_contract(
        contract=contract,
        contract_path=CONTRACT,
        incident=incident,
        incident_path=INCIDENT,
        repo_root=REPO_ROOT,
    )
    assert incident["impact"] == {
        "scheduled_pairs": 28,
        "generated_pairs": 0,
        "provider_completion_requests": 0,
        "epicure_calls": 0,
        "actual_cost_usd": "0",
        "model_reliability_eligible": False,
        "preference_eligible": False,
        "rank_eligible": False,
    }
    assert contract["action"]["new_provider_requests"] == 0
    assert contract["action"]["new_epicure_calls"] == 0
    assert contract["action"]["replay_permitted"] is False


def test_frozen_incident_records_the_fixed_mismatch_and_live_block_is_closed() -> None:
    plan = _json(PLAN)
    defect = _json(INCIDENT)["defect"]
    assert defect["frozen_pair_arm_scheduling"] == "concurrent"
    assert defect["parsed_pair_arm_scheduling"] == "sequential"
    prefix = recovery._verify_recovery_prefix(
        plan=plan,
        contract=_json(CONTRACT),
        incident_path=INCIDENT,
        repo_root=REPO_ROOT,
    )
    assert prefix["terminal_count"] == 28
    assert prefix["completed"] is True


def test_every_fsynced_recovery_boundary_resumes_without_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN)
    contract = _json(CONTRACT)
    live_coordinator, live_endpoints = recovery.executor._roots(plan, REPO_ROOT)
    coordinator = tmp_path / "coordinator"
    endpoints = {
        endpoint_id: tmp_path / endpoint_id for endpoint_id in ("deepseek", "gemini", "sonnet")
    }
    (coordinator / "endpoint-attestations").mkdir(parents=True)
    coordinator_line = (
        (live_coordinator / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    (coordinator / "ledger.jsonl").write_text(coordinator_line + "\n", encoding="utf-8")
    shutil.copy2(
        live_coordinator
        / "endpoint-attestations"
        / (
            "reasoning-effort-family-block-01-attestations-"
            f"{recovery.ATTESTATION_SEMANTIC_SHA256}.json"
        ),
        coordinator / "endpoint-attestations",
    )
    endpoints["sonnet"].mkdir(parents=True)
    sonnet_line = (
        (live_endpoints["sonnet"] / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    (endpoints["sonnet"] / "ledger.jsonl").write_text(sonnet_line + "\n", encoding="utf-8")
    monkeypatch.setattr(
        recovery.executor,
        "_roots",
        lambda plan, repo_root: (coordinator, endpoints),
    )
    attestation = recovery._attestation(plan, REPO_ROOT)
    attestation_by_endpoint = {row["endpoint_id"]: row for row in attestation["records"]}
    block = recovery.executor._block_map(plan)[recovery.BLOCK_ID]
    items = recovery.executor._item_map(plan)
    reservation = recovery.executor._coordinator_state(
        plan,
        recovery.executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator"),
    )["reservations"][recovery.BLOCK_ID]
    original_append = recovery.executor._append_ledger

    class SimulatedCrash(RuntimeError):
        pass

    def crash_after(target: int):
        counter = 0

        def append(*args, **kwargs):
            nonlocal counter
            row = original_append(*args, **kwargs)
            counter += 1
            if counter == target:
                raise SimulatedCrash
            return row

        return append

    def run_item(item_id: str) -> None:
        item = items[item_id]
        endpoint_id = item["route_coordinate"]["endpoint_id"]
        recovery._recover_item(
            plan=plan,
            block=block,
            item=item,
            reservation=reservation,
            coordinator_ledger=coordinator / "ledger.jsonl",
            endpoint_root=endpoints[endpoint_id],
            raw_endpoint_execution_contract_sha256=attestation_by_endpoint[endpoint_id][
                "raw_execution_contract_sha256"
            ],
            contract=contract,
            incident_path=INCIDENT,
            repo_root=REPO_ROOT,
        )

    ordered = list(block["work_item_ids"])

    # Existing start -> endpoint terminal persisted -> crash -> coordinator-only resume.
    monkeypatch.setattr(recovery.executor, "_append_ledger", crash_after(1))
    with pytest.raises(SimulatedCrash):
        run_item(ordered[0])
    monkeypatch.setattr(recovery.executor, "_append_ledger", original_append)
    assert (
        recovery._verify_recovery_prefix(
            plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
        )["terminal_count"]
        == 0
    )
    run_item(ordered[0])

    # Recovery start persisted -> crash; then endpoint terminal persisted -> crash.
    monkeypatch.setattr(recovery.executor, "_append_ledger", crash_after(1))
    with pytest.raises(SimulatedCrash):
        run_item(ordered[1])
    monkeypatch.setattr(recovery.executor, "_append_ledger", original_append)
    recovery._verify_recovery_prefix(
        plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
    )
    monkeypatch.setattr(recovery.executor, "_append_ledger", crash_after(1))
    with pytest.raises(SimulatedCrash):
        run_item(ordered[1])
    monkeypatch.setattr(recovery.executor, "_append_ledger", original_append)
    recovery._verify_recovery_prefix(
        plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
    )
    run_item(ordered[1])

    # Start + endpoint + coordinator persisted; crash after the coordinator append.
    monkeypatch.setattr(recovery.executor, "_append_ledger", crash_after(3))
    with pytest.raises(SimulatedCrash):
        run_item(ordered[2])
    monkeypatch.setattr(recovery.executor, "_append_ledger", original_append)
    prefix = recovery._verify_recovery_prefix(
        plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
    )
    assert prefix["terminal_count"] == 3
    run_item(ordered[2])

    for item_id in ordered[3:]:
        run_item(item_id)
    before_block_terminal = recovery._verify_recovery_prefix(
        plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
    )
    assert before_block_terminal["terminal_count"] == 28
    assert before_block_terminal["completed"] is False
    terminal = recovery.executor._terminalize_block(
        plan=plan, block=block, coordinator_ledger=coordinator / "ledger.jsonl"
    )
    before_receipt = recovery._verify_recovery_prefix(
        plan=plan, contract=contract, incident_path=INCIDENT, repo_root=REPO_ROOT
    )
    assert before_receipt["completed"] is True
    receipt = recovery._build_receipt(
        contract=contract,
        incident_ref=study._file_ref(REPO_ROOT, INCIDENT),
        terminal=terminal,
        coordinator_ledger=coordinator / "ledger.jsonl",
        endpoints=endpoints,
    )
    first = recovery._write(tmp_path / "receipts", "receipt", receipt)
    second = recovery._write(tmp_path / "receipts", "receipt", receipt)
    assert first == second
    assert (
        len(recovery.executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")) == 30
    )
    assert {
        endpoint: len(recovery.executor._load_ledger(root / "ledger.jsonl", role="endpoint"))
        for endpoint, root in endpoints.items()
    } == {"deepseek": 16, "gemini": 24, "sonnet": 16}
