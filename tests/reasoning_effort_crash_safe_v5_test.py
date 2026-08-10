from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier
import flavourbench.live_smoke as live_smoke_module
import flavourbench.reasoning_effort_full_study_executor_v5 as executor
import flavourbench.reasoning_effort_full_study_v5 as study

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
LIVE_GLOBAL_LEDGER = REPO_ROOT / study.GLOBAL_LEDGER_PATH


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return study.build_plan(repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def human_protocol(plan: dict[str, Any]) -> dict[str, Any]:
    return study.build_human_protocol(plan=plan, repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def bound_preflight(
    plan: dict[str, Any], human_protocol: dict[str, Any]
) -> dict[str, Any]:
    return study.build_bound_preflight(
        plan=plan, human_protocol=human_protocol, repo_root=REPO_ROOT
    )


@pytest.fixture(scope="module")
def governance_go(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": study.GOVERNANCE_GO_SCHEMA,
        "record_role": "independent_test_go_not_live_authority",
        "decision": "go_for_exactly_one_family_block",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human_protocol["artifact_sha256"],
        "bound_preflight_sha256": bound_preflight["artifact_sha256"],
        "authorized_admission_block_id": plan["admission_blocks"][0][
            "admission_block_id"
        ],
        "reviewer_is_executor": False,
        "provider_or_epicure_calls_made_by_review": False,
        "maximum_family_blocks": 1,
    }
    return {**payload, "artifact_sha256": study._sha256(payload)}


def _clone_global_ledger(tmp_path: Path) -> Path:
    destination = tmp_path / "frontier-contract" / "ledger.jsonl"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(LIVE_GLOBAL_LEDGER, destination)
    return destination


def _fake_attestations() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for endpoint_id in study.ENDPOINTS:
        raw = {"tag": endpoint_id, "provider_name": endpoint_id}
        records.append(
            {
                "endpoint_id": endpoint_id,
                "raw_execution_contract": raw,
                "raw_execution_contract_sha256": study._sha256(raw),
            }
        )
    return records


def _fake_global_accounting(
    *, plan: dict[str, Any], repo_root: Path, global_ledger: Path
) -> dict[str, Any]:
    del repo_root
    entries = frontier.load_ledger(global_ledger)
    executor._verify_global_anchor(plan=plan, entries=entries)
    active = frontier.active_ledger_reservations(entries)
    current = study.CURRENT_EXPOSURE_USD + sum(active.values(), Decimal(0))
    return {
        "entries": entries,
        "active": active,
        "baseline_exposure_usd": study._decimal_text(study.CURRENT_EXPOSURE_USD),
        "post_anchor_finalized_exposure_usd": "0",
        "canonical_active_reservation_usd": study._decimal_text(
            sum(active.values(), Decimal(0))
        ),
        "current_total_exposure_usd": study._decimal_text(current),
        "active_incidents": [],
    }


def _configure_fake_execution(
    *,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    tmp_path: Path,
    produce_source: bool = True,
) -> tuple[Path, dict[str, Path], Path, set[str], dict[str, int]]:
    coordinator = tmp_path / study.ROOT_ID / "coordinator"
    endpoints = {
        endpoint: tmp_path / study.ROOT_ID / endpoint for endpoint in study.ENDPOINTS
    }
    global_ledger = _clone_global_ledger(tmp_path)
    available: set[str] = set()
    invocations: dict[str, int] = {}
    monkeypatch.setattr(
        executor, "_roots", lambda plan, repo_root: (coordinator, endpoints)
    )
    monkeypatch.setattr(
        executor, "_require_live_environment_before_reservation", lambda: None
    )

    async def fake_attest(**_: Any) -> list[dict[str, Any]]:
        return _fake_attestations()

    async def fake_invoke(*, args: Any, **_: Any) -> None:
        item_id = str(args.dataset_work_item_id)
        invocations[item_id] = invocations.get(item_id, 0) + 1
        if produce_source:
            available.add(item_id)

    def fake_source(_: Path, item_id: str):
        if item_id not in available:
            return None
        digest = hashlib.sha256(f"source:{item_id}".encode()).hexdigest()
        return (
            tmp_path / f"source-{item_id}.json",
            {"artifact_sha256": digest},
            digest,
        )

    def fake_payload(**kwargs: Any) -> dict[str, Any]:
        item_id = str(kwargs["item"]["work_item_id"])
        digest = hashlib.sha256(f"source:{item_id}".encode()).hexdigest()
        return {
            "disposition": "source_usable",
            "source_path": f"fake/{item_id}.json",
            "source_artifact_sha256": digest,
            "pair_audit_path": f"fake/{item_id}-audit.json",
            "pair_audit_sha256": hashlib.sha256(f"audit:{item_id}".encode()).hexdigest(),
            "actual_cost_usd": "0.01",
            "audit_failures": [],
        }

    original_file_ref = study._file_ref

    def fake_file_ref(repo_root: Path, path: Path) -> dict[str, Any]:
        try:
            return original_file_ref(repo_root, path)
        except ValueError:
            document = _json(path)
            return {
                "path": str(path),
                "bytes": path.stat().st_size,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "semantic_sha256": document["artifact_sha256"],
            }

    monkeypatch.setattr(executor, "_attest_all_endpoints", fake_attest)
    monkeypatch.setattr(executor, "_invoke_live_pair", fake_invoke)
    monkeypatch.setattr(executor, "_canonical_source_for_item", fake_source)
    monkeypatch.setattr(executor, "_source_terminal_payload", fake_payload)
    monkeypatch.setattr(executor, "_global_accounting_locked", _fake_global_accounting)
    monkeypatch.setattr(
        executor, "_global_ledger_path", lambda plan, repo_root: global_ledger
    )
    monkeypatch.setattr(study, "_file_ref", fake_file_ref)
    monkeypatch.setattr(
        executor,
        "_journal_evidence",
        lambda source_root, work_item_id: {
            "journal_count": 0,
            "request_started_count": 0,
            "journals": [],
        },
    )
    return coordinator, endpoints, global_ledger, available, invocations


def test_estimand_fresh_ids_and_retired_v4_binding(
    plan: dict[str, Any], human_protocol: dict[str, Any]
) -> None:
    study.validate_plan(plan, repo_root=REPO_ROOT)
    assert plan["supersedes"]["retired_v4_plan_sha256"] == study.V4_PLAN_SHA256
    assert plan["supersedes"]["retired_v4_decision"] == "no_go"
    assert plan["supersedes"]["v3_import_pipeline_incident_sha256"] == (
        study.V3_INCIDENT_SHA256
    )
    assert plan["supersedes"]["v3_zero_call_recovery_sha256"] == (
        study.V3_RECOVERY_SHA256
    )
    identities = study._identity_sets(plan)
    assert {key: len(value) for key, value in identities.items()} == {
        "work_item_ids": 168,
        "run_ids": 168,
        "arm_ids": 336,
        "attempt_ids": 9408,
        "wave_ids": 24,
        "block_ids": 6,
        "presentation_ids": 240,
    }
    for relative in (study.V4_PLAN_PATH, study.V3_PLAN_PATH, study.V2_PLAN_PATH):
        retired = study._identity_sets(_json(REPO_ROOT / relative))
        assert all(identities[key].isdisjoint(retired[key]) for key in identities)
    canonical = plan["human_evaluation"]["presentations"]
    assert sum(row["contrast"] == "primary_low_high" for row in canonical) == 144
    assert sum(row["contrast"] != "primary_low_high" for row in canonical) == 96
    assert len(human_protocol["arm_coordinates"]) == 336
    assert len(human_protocol["comparison_cells"]) == 240
    assert len(human_protocol["presentations"]) == 1584
    assert human_protocol["counts"]["original_presentations"] == 1440
    assert human_protocol["counts"]["position_swapped_repeats"] == 144
    previous_human = _json(REPO_ROOT / study.V4_HUMAN_PATH)
    assert {
        row["presentation_id"] for row in human_protocol["presentations"]
    }.isdisjoint({row["presentation_id"] for row in previous_human["presentations"]})


def test_double_freeze_is_deterministic(
    plan: dict[str, Any], tmp_path: Path
) -> None:
    output = ROOT / ".pytest_cache" / f"v5-freeze-{tmp_path.name}"
    shutil.rmtree(output, ignore_errors=True)
    try:
        first = study.freeze(repo_root=REPO_ROOT, output_dir=output)
        first_bytes = {key: path.read_bytes() for key, path in first.items()}
        second = study.freeze(repo_root=REPO_ROOT, output_dir=output)
        assert _json(first["plan"]) == plan
        assert first == second
        assert first_bytes == {key: path.read_bytes() for key, path in second.items()}
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_all_runtime_args_and_attestation_binding_are_side_effect_free(
    plan: dict[str, Any]
) -> None:
    prepared = executor.prepare_all_runtime_items(plan=plan, repo_root=REPO_ROOT)
    assert len(prepared) == 168
    block = plan["admission_blocks"][0]
    rebound = executor._bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=_fake_attestations(),
        prepared_all=prepared,
        repo_root=REPO_ROOT,
    )
    assert list(rebound) == block["work_item_ids"]
    assert all(
        len(args.expected_endpoint_execution_sha256) == 64
        for _, args in rebound.values()
    )


def test_partial_shared_reservation_is_visible_and_resumable(
    plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    global_ledger = _clone_global_ledger(tmp_path)
    monkeypatch.setattr(executor, "_global_accounting_locked", _fake_global_accounting)
    block = plan["admission_blocks"][0]
    seen = 0

    def crash(stage: str, item_id: str | None) -> None:
        nonlocal seen
        if stage == "after_global_reservation":
            seen += 1
            if seen == 3:
                raise executor.SimulatedCrash("third canonical reservation")

    with pytest.raises(executor.SimulatedCrash):
        executor._ensure_canonical_reservations(
            plan=plan,
            block=block,
            repo_root=REPO_ROOT,
            global_ledger=global_ledger,
            failure_injector=crash,
        )
    active = frontier.active_ledger_reservations(frontier.load_ledger(global_ledger))
    assert len(active) == 3
    reservations, _ = executor._ensure_canonical_reservations(
        plan=plan,
        block=block,
        repo_root=REPO_ROOT,
        global_ledger=global_ledger,
        failure_injector=None,
    )
    assert len(reservations) == 28
    active = frontier.active_ledger_reservations(frontier.load_ledger(global_ledger))
    assert {row["entry_sha256"] for row in reservations} <= set(active)
    assert _fake_global_accounting(
        plan=plan, repo_root=REPO_ROOT, global_ledger=global_ledger
    )["current_total_exposure_usd"] == study._decimal_text(
        study.CURRENT_EXPOSURE_USD + Decimal(block["worst_case_reserve_usd"])
    )
    assert len(
        executor._campaign_global_reservations(
            plan=plan, entries=frontier.load_ledger(global_ledger)
        )
    ) == 28


def test_forged_or_missing_global_binding_is_rejected(
    plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    global_ledger = _clone_global_ledger(tmp_path)
    monkeypatch.setattr(executor, "_global_accounting_locked", _fake_global_accounting)
    block = plan["admission_blocks"][0]
    reservations, _ = executor._ensure_canonical_reservations(
        plan=plan,
        block=block,
        repo_root=REPO_ROOT,
        global_ledger=global_ledger,
        failure_injector=None,
    )
    mapping = {
        item_id: row["entry_sha256"]
        for item_id, row in zip(block["work_item_ids"], reservations, strict=True)
    }
    local_reservation = {
        "canonical_reservation_entry_sha256_by_work_item": mapping,
        "canonical_reservation_entry_sha256s": list(mapping.values()),
    }
    executor._verify_local_global_binding(
        plan=plan,
        block=block,
        local_reservation=local_reservation,
        global_entries=frontier.load_ledger(global_ledger),
    )
    forged = {**local_reservation}
    forged_mapping = dict(mapping)
    forged_mapping[block["work_item_ids"][0]] = "f" * 64
    forged["canonical_reservation_entry_sha256_by_work_item"] = forged_mapping
    forged["canonical_reservation_entry_sha256s"] = list(forged_mapping.values())
    with pytest.raises(executor.FullStudyExecutionError, match="binding differs"):
        executor._verify_local_global_binding(
            plan=plan,
            block=block,
            local_reservation=forged,
            global_entries=frontier.load_ledger(global_ledger),
        )
    item = next(
        row
        for row in plan["work_items"]
        if row["work_item_id"] == block["work_item_ids"][0]
    )
    wrong = executor._canonical_reservation_identity(plan=plan, block=block, item=item)
    wrong["study_plan_sha256"] = "0" * 64
    frontier.append_ledger_event(global_ledger, wrong)
    with pytest.raises(executor.FullStudyExecutionError, match="forged"):
        executor._campaign_global_reservations(
            plan=plan, entries=frontier.load_ledger(global_ledger)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cut",
    [
        "after_global_lock_before_reservations",
        "before_global_reservation",
        "after_global_reservation",
        "before_local_block_reservation",
        "after_local_block_reservation",
        "before_item_start",
        "after_item_start",
        "before_source_lookup",
        "after_provider_invocation_before_source_lookup",
        "before_source_classification",
        "after_source_classification",
        "before_global_artifact_finalization",
        "after_global_artifact_finalization",
        "before_local_terminal",
        "after_endpoint_terminal_before_coordinator_terminal",
        "after_coordinator_terminal",
    ],
)
async def test_every_crash_cut_recovers_without_provider_replay(
    cut: str,
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    governance_go: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_plan = json.loads(json.dumps(plan))
    coordinator, _, global_ledger, _, invocations = _configure_fake_execution(
        monkeypatch=monkeypatch, plan=local_plan, tmp_path=tmp_path
    )
    first = local_plan["admission_blocks"][0]["work_item_ids"][0]
    fired = False

    def crash(stage: str, item_id: str | None) -> None:
        nonlocal fired
        if fired or stage != cut:
            return
        if item_id is not None and item_id != first:
            return
        fired = True
        raise executor.SimulatedCrash(cut)

    with pytest.raises(executor.SimulatedCrash):
        await executor.execute_one_block(
            plan=local_plan,
            human_protocol=human_protocol,
            bound_preflight=bound_preflight,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://invalid.example",
            api_key="not-used",
            failure_injector=crash,
        )
    assert fired is True
    result = await executor.execute_one_block(
        plan=local_plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
    )
    assert result["decision"] == "block_terminal"
    assert result["document"]["canonical_reservation_count"] == 28
    state = executor._coordinator_state(
        local_plan,
        executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator"),
    )
    assert len(state["terminals"]) == 28
    assert len(state["completed"]) == 1
    assert max(invocations.values(), default=0) == 1
    entries = frontier.load_ledger(global_ledger)
    campaign = executor._campaign_global_reservations(plan=local_plan, entries=entries)
    assert len(campaign) == 28
    assert not ({row["entry_sha256"] for row in campaign.values()} & set(
        frontier.active_ledger_reservations(entries)
    ))
    artifact_events = {
        row["entry_sha256"]: row
        for row in entries
        if row.get("event_type") == "artifact_recorded"
        and row.get("campaign_id") == study.STUDY_ID
    }
    assert len(artifact_events) == 28
    assert {
        terminal["canonical_artifact_record_entry_sha256"]
        for terminal in state["terminals"].values()
    } == set(artifact_events)
    assert all(
        artifact_events[terminal["canonical_artifact_record_entry_sha256"]][
            "reservation_entry_sha256"
        ]
        == terminal["canonical_reservation_entry_sha256"]
        for terminal in state["terminals"].values()
    )


@pytest.mark.asyncio
async def test_post_start_without_source_is_incident_and_reserve_is_retained(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    governance_go: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_plan = json.loads(json.dumps(plan))
    coordinator, _, global_ledger, _, _ = _configure_fake_execution(
        monkeypatch=monkeypatch,
        plan=local_plan,
        tmp_path=tmp_path,
        produce_source=False,
    )
    result = await executor.execute_one_block(
        plan=local_plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        governance_go=governance_go,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
    )
    assert result["decision"] == "durable_incident_stop"
    assert result["outcomes"][0]["decision"] == (
        "durable_incident_reservation_retained"
    )
    state = executor._coordinator_state(
        local_plan,
        executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator"),
    )
    assert state["incidents"][state["active_block_id"]]
    entries = frontier.load_ledger(global_ledger)
    campaign = executor._campaign_global_reservations(plan=local_plan, entries=entries)
    active = frontier.active_ledger_reservations(entries)
    assert len(campaign) == 28
    assert {row["entry_sha256"] for row in campaign.values()} <= set(active)
    assert not any(
        row.get("event_type") == "artifact_recorded"
        and row.get("campaign_id") == study.STUDY_ID
        for row in entries
    )


def test_missing_symbol_and_retired_confirmation_have_no_side_effects(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    governance_go: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    roots = study.successor_roots(plan=plan, repo_root=REPO_ROOT)
    assert all(not root.exists() for root in roots)
    observed: list[str] = []

    async def forbidden_attest(**_: Any) -> list[dict[str, Any]]:
        observed.append("catalog")
        return []

    @contextmanager
    def forbidden_lock(_: Path):
        observed.append("lock")
        yield

    monkeypatch.delattr(live_smoke_module, "CONFIRMATION")
    monkeypatch.setattr(executor, "_attest_all_endpoints", forbidden_attest)
    monkeypatch.setattr(frontier, "_exclusive_runner_lock", forbidden_lock)
    with pytest.raises(ImportError):
        import asyncio

        asyncio.run(
            executor.execute_one_block(
                plan=plan,
                human_protocol=human_protocol,
                bound_preflight=bound_preflight,
                governance_go=governance_go,
                repo_root=REPO_ROOT,
                api_base="https://invalid.example",
                api_key="not-used",
            )
        )
    assert observed == []
    assert all(not root.exists() for root in roots)
    absent = tmp_path / "absent.json"
    with pytest.raises(SystemExit, match="exact V5 crash-safe"):
        executor.run(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--plan",
                str(absent),
                "--human-protocol",
                str(absent),
                "--bound-preflight",
                str(absent),
                "--governance-go",
                str(absent),
                "--confirm",
                study.v4.CONFIRMATION,
            ]
        )
    assert not absent.exists()
    assert observed == []


def test_historical_sources_artifacts_and_live_ledger_are_unchanged() -> None:
    expected = {
        "flavourbench/src/flavourbench/reasoning_effort_full_study_v1.py": (
            "ad816b5179ffc091a1da4acfe303a7b6f23eb865dfea6c96b4973fb6de7046fc"
        ),
        "flavourbench/src/flavourbench/reasoning_effort_full_study_executor_v1.py": (
            "aa13f05ccaf7b85c91f3918f158fd76f242c7780674d82f8cd872fcb656d6ad4"
        ),
        "flavourbench/src/flavourbench/reasoning_effort_source_closure_v1.py": (
            "5036763df51575eef444924de1f3950c6c745cf3a508bc95cb594c0e6510a73b"
        ),
        "flavourbench/src/flavourbench/reasoning_effort_full_study_v2.py": (
            "35448f942922255adb31bdbd12add2f2296b60aa76cd996fa79fe4c14b18353f"
        ),
        "flavourbench/src/flavourbench/reasoning_effort_full_study_executor_v2.py": (
            "b4bdb581e50acca88f203596c620dd3ee59f3f9463ef640d0daff93b1d629324"
        ),
        "flavourbench/src/flavourbench/reasoning_effort_source_closure_v2.py": (
            "37ab9e7de31ddbc92fd1760332b378d02b8ff3abe4da5f031c9157da4063d171"
        ),
        study.V4_PLAN_PATH: (
            "fb23b6ba2275155c6f299261caccb4c7f235650df1ec47b7e91e1a2ef76df254"
        ),
        study.V4_HUMAN_PATH: (
            "e2f6792377d5a9a9a012839be32bccf1715d33e9b30774241030ac5d9dc0cc8e"
        ),
        study.GLOBAL_LEDGER_PATH: study.GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() == digest


def test_real_locked_global_accounting_rederives_baseline(plan: dict[str, Any]) -> None:
    accounting = executor._global_accounting_locked(
        plan=plan,
        repo_root=REPO_ROOT,
        global_ledger=LIVE_GLOBAL_LEDGER,
    )
    assert accounting["baseline_exposure_usd"] == (
        "48.01944682666666666666666666"
    )
    assert accounting["post_anchor_finalized_exposure_usd"] == "0"
    assert accounting["canonical_active_reservation_usd"] == "0"
    assert accounting["current_total_exposure_usd"] == (
        "48.01944682666666666666666666"
    )
    assert accounting["active_incidents"] == []
