"""Independent, zero-call reproducers for the V5 execution NO-GO.

These tests intentionally describe defects in the frozen V5 successor.  They
do not grant live authority and mutate only pytest temporary directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier
import flavourbench.reasoning_effort_full_study_executor_v5 as executor
import flavourbench.reasoning_effort_full_study_v5 as study
import flavourbench.reasoning_effort_route_gate_v4 as route_v4
import flavourbench.reasoning_effort_route_gate_v5 as route_v5

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
FROZEN_ROOT = (
    ROOT / "artifacts/season1/current-quality-run/"
    "reasoning-effort-task-waves-v5-shared-ledger-crash-safe"
)
LIVE_GLOBAL_LEDGER = REPO_ROOT / study.GLOBAL_LEDGER_PATH
LIVE_SOURCE_ROOT = REPO_ROOT / study.GLOBAL_SOURCE_PATH

EXPECTED = {
    "plan": {
        "semantic": "2b07db3988828b1f7f50e5f2004fa3f461cb4defdcd3e12b2b744e5a65570e3d",
        "physical": "9ae584c7f14158294e98d2ba590e396b5837770f5de5c5550a7d90abf39dc4c0",
    },
    "human": {
        "semantic": "d5f7e78a7a635070a727fe8549499de9d07da05371b8b242983874877d9b1eb9",
        "physical": "82c3f58137888d47b3ec1a0318866cd15e7607b235d7340d408d31880dee01a3",
    },
    "preflight": {
        "semantic": "3a552e2674d305e6b89869e37a7cdf0f21db754c2c2a632d4fa89678dc9a23a7",
        "physical": "9045e49d3d85b57400d892a90aeefd4be3a1a6f6bbe9ab064e3037f8351ad3e4",
    },
    "bound": {
        "semantic": "208321ef17688a66e1c72d6205b122e53b0bd7979f8e835c167aabcf6450a509",
        "physical": "2a4753348087456b6ea65f6dc60082fe37704b6a59664f1d035fa1608e9d2c00",
    },
}

# The reviewed pre-V5 source subset is explicit so later, legitimate additions
# to the canonical root cannot turn this historical reproducer into a time bomb.
REVIEWED_SOURCE_MEMBERS = (
    "20260715T153842Z-61936670dc81.json",
    "20260715T154025Z-865d38781d6b.json",
    "20260715T154241Z-ee0010734bab.json",
    "20260715T154330Z-32f0e2921c12.json",
    "20260715T154608Z-37ef5c11a814.json",
    "20260715T154741Z-1b198a8afc60.json",
    "20260715T161257Z-bea82a68f6dc.json",
    "20260715T161521Z-523ee0033edb.json",
    "20260715T161521Z-95c89438ef2d.json",
    "20260715T161521Z-a3c8f02b84dd.json",
    "20260715T161933Z-a608f122e9ac.json",
    "20260715T161933Z-b26f9dc8f0e0.json",
    "20260715T162406Z-391594d20273.json",
    "20260715T162406Z-fda1ed039761.json",
    "20260715T162704Z-80bda238dfce.json",
    "20260715T162704Z-a10e40442d8b.json",
    "20260715T163130Z-281424a13434.json",
    "20260715T163130Z-dfdfc62dcddd.json",
    "20260715T163408Z-861daebfc805.json",
    "20260715T163408Z-caedcef9b463.json",
    "20260715T163408Z-fb07c83a47b5.json",
    "20260715T163634Z-6f592dcb7297.json",
    "20260715T163654Z-019e5ffe9e30.json",
    "20260715T163654Z-2318c5ed7139.json",
    "20260715T163855Z-3837be3d912e.json",
    "20260715T163855Z-ccba8973ea6f.json",
    "20260715T164130Z-9df17b2270a5.json",
    "20260715T165211Z-abaca8692188.json",
    "20260715T165212Z-4c8fc4c1a177.json",
    "20260715T165214Z-3223b550f007.json",
    "20260715T165215Z-071bbb7c9624.json",
    "20260715T165217Z-ac3ca07cd37d.json",
    "20260715T165218Z-7e62313d1659.json",
    "20260715T165219Z-392bcd380611.json",
    "20260715T170428Z-90b6a0a60de6.json",
    "20260715T170526Z-b1c59b2aa7b0.json",
    "20260715T170606Z-2c6999285a50.json",
    "20260715T170607Z-772b33767d27.json",
    "20260715T170704Z-a09b79afc669.json",
)
REVIEWED_SOURCE_INVENTORY_SHA256 = (
    "7bfd5c9e343e300199914da791e5dec89db9a3c91545a4e34ed243f2086424c1"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _document(directory: str) -> tuple[Path, dict[str, Any]]:
    path = next((FROZEN_ROOT / directory).glob("*.json"))
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen() -> dict[str, dict[str, Any]]:
    return {
        "plan": _document("plan")[1],
        "human": _document("human-protocol")[1],
        "preflight": _document("preflight")[1],
        "bound": _document("bound-preflight")[1],
    }


def _go(frozen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plan = frozen["plan"]
    payload = {
        "schema_version": study.GOVERNANCE_GO_SCHEMA,
        "record_role": "independent_test_go_not_live_authority",
        "decision": "go_for_exactly_one_family_block",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": frozen["human"]["artifact_sha256"],
        "bound_preflight_sha256": frozen["bound"]["artifact_sha256"],
        "authorized_admission_block_id": plan["admission_blocks"][0]["admission_block_id"],
        "reviewer_is_executor": False,
        "provider_or_epicure_calls_made_by_review": False,
        "maximum_family_blocks": 1,
    }
    return {**payload, "artifact_sha256": study._sha256(payload)}


def _clone_global_ledger(tmp_path: Path) -> Path:
    target = tmp_path / "frontier-contract/ledger.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LIVE_GLOBAL_LEDGER, target)
    return target


def _global_reservations(
    *, plan: dict[str, Any], block: dict[str, Any], global_ledger: Path
) -> dict[str, dict[str, Any]]:
    items = executor._item_map(plan)
    return {
        item_id: frontier.append_ledger_event(
            global_ledger,
            executor._canonical_reservation_identity(plan=plan, block=block, item=items[item_id]),
            recorded_at=f"2026-08-04T00:00:{index:02d}Z",
        )
        for index, item_id in enumerate(block["work_item_ids"])
    }


def _local_reservation(
    *,
    plan: dict[str, Any],
    block: dict[str, Any],
    coordinator_ledger: Path,
    canonical: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mapping = {item_id: canonical[item_id]["entry_sha256"] for item_id in block["work_item_ids"]}
    return executor._append_ledger(
        coordinator_ledger,
        role="coordinator",
        event={
            "event_type": "family_block_reservation_created",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "block_ordinal": block["block_ordinal"],
            "wave_ids": block["wave_ids"],
            "task_ids": block["task_ids"],
            "task_families": block["task_families"],
            "work_item_ids": block["work_item_ids"],
            "reserved_usd": block["worst_case_reserve_usd"],
            "canonical_reservation_entry_sha256_by_work_item": mapping,
            "canonical_reservation_entry_sha256s": list(mapping.values()),
            "endpoint_attestation": {"semantic_sha256": "a" * 64},
            "global_accounting_at_admission": {},
            "replay_permitted": False,
        },
    )


def _start_item(
    *,
    plan: dict[str, Any],
    block: dict[str, Any],
    item: dict[str, Any],
    local_reservation: dict[str, Any],
    canonical_reservation: dict[str, Any],
    endpoint_ledger: Path,
) -> dict[str, Any]:
    return executor._append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": "item_execution_started",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": executor._item_wave_id(plan, item["work_item_id"]),
            "work_item_id": item["work_item_id"],
            "run_id": item["run_id"],
            "endpoint_id": item["route_coordinate"]["endpoint_id"],
            "variant_id": item["route_coordinate"]["variant_id"],
            "block_reservation_entry_sha256": local_reservation["entry_sha256"],
            "canonical_reservation_entry_sha256": canonical_reservation["entry_sha256"],
            "raw_endpoint_execution_sha256": "b" * 64,
            "replay_permitted": False,
        },
    )


def _fake_accounting(
    *, plan: dict[str, Any], repo_root: Path, global_ledger: Path
) -> dict[str, Any]:
    del repo_root
    entries = frontier.load_ledger(global_ledger)
    executor._verify_global_anchor(plan=plan, entries=entries)
    active = frontier.active_ledger_reservations(entries)
    active_total = sum(active.values(), Decimal(0))
    current = study.CURRENT_EXPOSURE_USD + active_total
    return {
        "entries": entries,
        "active": active,
        "baseline_exposure_usd": study._decimal_text(study.CURRENT_EXPOSURE_USD),
        "post_anchor_finalized_exposure_usd": "0",
        "canonical_active_reservation_usd": study._decimal_text(active_total),
        "current_total_exposure_usd": study._decimal_text(current),
        "active_incidents": [],
    }


def _fake_attestations() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for endpoint_id in study.ENDPOINTS:
        raw = {
            "tag": endpoint_id,
            "provider_name": endpoint_id,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
        values.append(
            {
                "endpoint_id": endpoint_id,
                "raw_execution_contract": raw,
                "raw_execution_contract_sha256": study._sha256(raw),
            }
        )
    return values


def _external_file_ref(repo_root: Path, path: Path) -> dict[str, Any]:
    del repo_root
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "semantic_sha256": document["artifact_sha256"],
    }


def test_frozen_hashes_and_heterogeneous_canonical_source_root_prove_no_go(
    frozen: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    directories = {
        "plan": "plan",
        "human": "human-protocol",
        "preflight": "preflight",
        "bound": "bound-preflight",
    }
    for label, directory in directories.items():
        path, document = _document(directory)
        body = {key: value for key, value in document.items() if key != "artifact_sha256"}
        assert document["artifact_sha256"] == EXPECTED[label]["semantic"]
        assert hashlib.sha256(_canonical(body)).hexdigest() == EXPECTED[label]["semantic"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED[label]["physical"]

    reviewed_subset = tmp_path / "reviewed-live-smoke-subset"
    reviewed_subset.mkdir()
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, str]] = []
    for filename in REVIEWED_SOURCE_MEMBERS:
        path = LIVE_SOURCE_ROOT / filename
        document, _ = frontier._verify_live_artifact(path)
        inventory.append({"filename": filename, "artifact_sha256": document["artifact_sha256"]})
        records.append(document)
        shutil.copyfile(path, reviewed_subset / filename)
    assert hashlib.sha256(_canonical(inventory)).hexdigest() == (REVIEWED_SOURCE_INVENTORY_SHA256)
    assert len(records) == 39
    assert sum("dataset_work_item_id" not in row for row in records) == 27
    assert (
        sum(
            "dataset_work_item_id" in row and row["dataset_work_item_id"] is None for row in records
        )
        == 12
    )
    assert not any(
        isinstance(row.get("dataset_work_item_id"), str) and len(row["dataset_work_item_id"]) == 64
        for row in records
    )
    with pytest.raises(route_v5.RouteGateV5Error, match="absent or duplicate"):
        route_v5._source_map(reviewed_subset)


def test_attestation_wrapper_has_no_top_level_pricing_and_cannot_invoke(
    frozen: dict[str, dict[str, Any]],
) -> None:
    plan = frozen["plan"]
    item_id = plan["admission_blocks"][0]["work_item_ids"][0]
    policy = executor.prepare_all_runtime_items(plan=plan, repo_root=REPO_ROOT)[item_id][0]
    raw = {
        "tag": "deepinfra/fp4",
        "provider_name": "DeepInfra",
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    }
    wrapper = {
        "endpoint_id": "deepseek",
        "raw_execution_contract": raw,
        "raw_execution_contract_sha256": study._sha256(raw),
    }
    with pytest.raises(route_v4.RouteGateError, match="prompt price"):
        with route_v4._policy_environment(policy=policy, endpoint=wrapper):
            pass
    with route_v4._policy_environment(policy=policy, endpoint=raw):
        pass


@pytest.mark.asyncio
async def test_post_start_lookup_failure_escapes_without_durable_incident(
    frozen: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    coordinator = tmp_path / "coordinator"
    endpoints = {key: tmp_path / key for key in study.ENDPOINTS}
    global_ledger = _clone_global_ledger(tmp_path)

    monkeypatch.setattr(executor, "prepare_all_runtime_items", lambda **_: {})
    monkeypatch.setattr(executor, "_require_live_environment_before_reservation", lambda: None)
    monkeypatch.setattr(executor, "_roots", lambda *_args, **_kwargs: (coordinator, endpoints))
    monkeypatch.setattr(executor, "_global_ledger_path", lambda *_args, **_kwargs: global_ledger)
    monkeypatch.setattr(executor, "_global_accounting_locked", _fake_accounting)
    monkeypatch.setattr(study, "_file_ref", _external_file_ref)

    async def attest(**_: Any) -> list[dict[str, Any]]:
        return _fake_attestations()

    monkeypatch.setattr(executor, "_attest_all_endpoints", attest)
    monkeypatch.setattr(executor, "_bind_block_runtime_after_attestation", lambda **_: {})

    with pytest.raises(KeyError, match=block["work_item_ids"][0]):
        await executor.execute_one_block(
            plan=plan,
            human_protocol=frozen["human"],
            bound_preflight=frozen["bound"],
            governance_go=_go(frozen),
            repo_root=REPO_ROOT,
            api_base="https://invalid.example",
            api_key="not-used",
        )
    coordinator_state = executor._coordinator_state(
        plan,
        executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator"),
    )
    assert coordinator_state["active_block_id"] == block["admission_block_id"]
    assert coordinator_state["incidents"] == {}
    first_item = executor._item_map(plan)[block["work_item_ids"][0]]
    endpoint_state = executor._endpoint_state(
        executor._load_ledger(
            endpoints[first_item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl",
            role="endpoint",
        )
    )
    assert block["work_item_ids"][0] in endpoint_state["started"]
    assert endpoint_state["incidents"] == {}
    assert len(frontier.active_ledger_reservations(frontier.load_ledger(global_ledger))) == 28


@pytest.mark.asyncio
async def test_normal_failure_after_global_artifact_falsely_claims_retained_reserve(
    frozen: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    item = executor._item_map(plan)[block["work_item_ids"][0]]
    global_ledger = _clone_global_ledger(tmp_path)
    canonical = _global_reservations(plan=plan, block=block, global_ledger=global_ledger)
    coordinator_ledger = tmp_path / "coordinator/ledger.jsonl"
    local_reservation = _local_reservation(
        plan=plan,
        block=block,
        coordinator_ledger=coordinator_ledger,
        canonical=canonical,
    )
    endpoint_root = tmp_path / item["route_coordinate"]["endpoint_id"]
    endpoint_ledger = endpoint_root / "ledger.jsonl"
    _start_item(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        endpoint_ledger=endpoint_ledger,
    )
    source_digest = hashlib.sha256(b"review-source").hexdigest()
    source = (
        tmp_path / "review-source.json",
        {"artifact_sha256": source_digest},
        source_digest,
    )
    monkeypatch.setattr(executor, "_canonical_source_for_item", lambda *_: source)
    monkeypatch.setattr(
        executor,
        "_source_terminal_payload",
        lambda **_: {
            "disposition": "source_usable",
            "source_path": "review-source.json",
            "source_artifact_sha256": source_digest,
            "pair_audit_path": "review-audit.json",
            "pair_audit_sha256": "c" * 64,
            "actual_cost_usd": "0.01",
            "audit_failures": [],
        },
    )
    monkeypatch.setattr(
        executor,
        "_journal_evidence",
        lambda *_: {"journal_count": 0, "request_started_count": 0, "journals": []},
    )

    def fail_after_global_artifact(stage: str, _item_id: str | None) -> None:
        if stage == "after_global_artifact_finalization":
            raise RuntimeError("review normal failure after canonical finalization")

    outcome = await executor._process_started_item(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        policy=None,
        args=argparse.Namespace(),
        endpoint_attestation={},
        repo_root=REPO_ROOT,
        source_root=tmp_path,
        endpoint_root=endpoint_root,
        endpoint_ledger=endpoint_ledger,
        coordinator_ledger=coordinator_ledger,
        global_ledger=global_ledger,
        failure_injector=fail_after_global_artifact,
    )
    assert outcome["decision"] == "durable_incident_reservation_retained"
    active = frontier.active_ledger_reservations(frontier.load_ledger(global_ledger))
    reservation_sha = canonical[item["work_item_id"]]["entry_sha256"]
    assert reservation_sha not in active
    incident = executor._coordinator_state(
        plan,
        executor._load_ledger(coordinator_ledger, role="coordinator"),
    )["incidents"][block["admission_block_id"]][0]
    assert incident["canonical_reservation_retained"] is True
    assert incident["work_item_reserve_retained_usd"] == item["worst_case_reserve_usd"]


@pytest.mark.asyncio
async def test_completed_block_without_receipt_cannot_recover_under_same_go(
    frozen: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    coordinator = tmp_path / "coordinator"
    coordinator_ledger = coordinator / "ledger.jsonl"
    canonical = {
        item_id: {"entry_sha256": f"{index + 1:064x}"}
        for index, item_id in enumerate(block["work_item_ids"])
    }
    local_reservation = _local_reservation(
        plan=plan,
        block=block,
        coordinator_ledger=coordinator_ledger,
        canonical=canonical,
    )
    for index, item_id in enumerate(block["work_item_ids"]):
        item = executor._item_map(plan)[item_id]
        executor._append_ledger(
            coordinator_ledger,
            role="coordinator",
            event={
                "event_type": "family_block_item_terminalized",
                "study_plan_sha256": plan["artifact_sha256"],
                "admission_block_id": block["admission_block_id"],
                "task_wave_id": executor._item_wave_id(plan, item_id),
                "work_item_id": item_id,
                "block_reservation_entry_sha256": local_reservation["entry_sha256"],
                "canonical_reservation_entry_sha256": canonical[item_id]["entry_sha256"],
                "canonical_artifact_record_entry_sha256": f"{index + 101:064x}",
                "source_artifact_sha256": f"{index + 201:064x}",
                "disposition": "source_usable",
                "source_path": f"source-{index}.json",
                "pair_audit_path": f"audit-{index}.json",
                "pair_audit_sha256": f"{index + 301:064x}",
                "actual_cost_usd": "0.01",
                "audit_failures": [],
                "endpoint_id": item["route_coordinate"]["endpoint_id"],
                "variant_id": item["route_coordinate"]["variant_id"],
                "endpoint_terminal_entry_sha256": f"{index + 401:064x}",
                "replay_permitted": False,
                "rank_eligible": False,
            },
        )
    executor._terminalize_block(plan=plan, block=block, coordinator_ledger=coordinator_ledger)
    assert not (coordinator / "receipts").exists()

    monkeypatch.setattr(executor, "prepare_all_runtime_items", lambda **_: {})
    monkeypatch.setattr(executor, "_require_live_environment_before_reservation", lambda: None)
    monkeypatch.setattr(
        executor,
        "_roots",
        lambda *_args, **_kwargs: (
            coordinator,
            {key: tmp_path / key for key in study.ENDPOINTS},
        ),
    )
    with pytest.raises(executor.FullStudyExecutionError, match="does not authorize"):
        await executor.execute_one_block(
            plan=plan,
            human_protocol=frozen["human"],
            bound_preflight=frozen["bound"],
            governance_go=_go(frozen),
            repo_root=REPO_ROOT,
            api_base="https://invalid.example",
            api_key="not-used",
        )


def test_incident_restart_does_not_reuse_exact_endpoint_evidence(
    frozen: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    item = executor._item_map(plan)[block["work_item_ids"][0]]
    global_ledger = _clone_global_ledger(tmp_path)
    canonical = _global_reservations(plan=plan, block=block, global_ledger=global_ledger)
    coordinator_ledger = tmp_path / "coordinator/ledger.jsonl"
    local_reservation = _local_reservation(
        plan=plan,
        block=block,
        coordinator_ledger=coordinator_ledger,
        canonical=canonical,
    )
    endpoint_ledger = tmp_path / "endpoint/ledger.jsonl"
    _start_item(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        endpoint_ledger=endpoint_ledger,
    )
    original_evidence = {
        "journal_count": 1,
        "request_started_count": 1,
        "journals": [{"sha256": "d" * 64}],
    }
    original_error = RuntimeError("original endpoint incident")
    endpoint_incident = executor._append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": "uncertain_execution_incident",
            "study_plan_sha256": plan["artifact_sha256"],
            "admission_block_id": block["admission_block_id"],
            "task_wave_id": executor._item_wave_id(plan, item["work_item_id"]),
            "work_item_id": item["work_item_id"],
            "block_reservation_entry_sha256": local_reservation["entry_sha256"],
            "canonical_reservation_entry_sha256": canonical[item["work_item_id"]]["entry_sha256"],
            "incident": "durable_post_start_without_finalizable_canonical_source",
            "journal_evidence": original_evidence,
            **executor._error_record(original_error),
            "work_item_reserve_retained_usd": item["worst_case_reserve_usd"],
            "canonical_reservation_retained": True,
            "replay_permitted": False,
        },
    )
    recovered = executor._append_incident_idempotent(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        endpoint_ledger=endpoint_ledger,
        coordinator_ledger=coordinator_ledger,
        evidence={},
        error=executor.MissingCanonicalSource("recovered endpoint incident"),
    )
    assert recovered["endpoint_incident_entry_sha256"] == endpoint_incident["entry_sha256"]
    assert recovered["journal_evidence"] != endpoint_incident["journal_evidence"]
    assert recovered["error_sha256"] != endpoint_incident["error_sha256"]


def test_normal_failure_after_endpoint_terminal_corrupts_endpoint_state(
    frozen: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    item = executor._item_map(plan)[block["work_item_ids"][0]]
    global_ledger = _clone_global_ledger(tmp_path)
    canonical = _global_reservations(plan=plan, block=block, global_ledger=global_ledger)
    coordinator_ledger = tmp_path / "coordinator/ledger.jsonl"
    local_reservation = _local_reservation(
        plan=plan,
        block=block,
        coordinator_ledger=coordinator_ledger,
        canonical=canonical,
    )
    endpoint_ledger = tmp_path / "endpoint/ledger.jsonl"
    _start_item(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        endpoint_ledger=endpoint_ledger,
    )
    executor._append_ledger(
        endpoint_ledger,
        role="endpoint",
        event={
            "event_type": "source_terminalized",
            "work_item_id": item["work_item_id"],
        },
    )
    executor._append_incident_idempotent(
        plan=plan,
        block=block,
        item=item,
        local_reservation=local_reservation,
        canonical_reservation=canonical[item["work_item_id"]],
        endpoint_ledger=endpoint_ledger,
        coordinator_ledger=coordinator_ledger,
        evidence={},
        error=OSError("coordinator terminal append failed"),
    )
    with pytest.raises(executor.FullStudyExecutionError, match="order is invalid"):
        executor._endpoint_state(executor._load_ledger(endpoint_ledger, role="endpoint"))


def test_exact_decimal_budget_display_is_internally_inconsistent(
    frozen: dict[str, dict[str, Any]],
) -> None:
    plan = frozen["plan"]
    block = plan["admission_blocks"][0]
    items = executor._item_map(plan)
    exact = sum(
        (Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in block["work_item_ids"]),
        Decimal(0),
    )
    assert exact == Decimal("17.58085589333333333333333334")
    assert Decimal(block["worst_case_reserve_usd"]) == exact
    assert Decimal(frozen["preflight"]["checks"]["first_block_reserve_usd"]) == exact
    assert Decimal(plan["budget"]["first_block_worst_case_usd"]) != exact
    projected = Decimal(plan["canonical_global_ledger_anchor"]["baseline_exposure_usd"]) + exact
    assert projected == Decimal("65.60030272000000000000000000")
    assert Decimal(frozen["preflight"]["checks"]["first_block_projected_usd"]) == projected
