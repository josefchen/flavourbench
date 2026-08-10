from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier_runner
import flavourbench.live_smoke as live_smoke_module
import flavourbench.reasoning_effort_full_study_executor_v2 as executor
import flavourbench.reasoning_effort_full_study_v2 as study
import flavourbench.reasoning_effort_source_closure_v2 as closure

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


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


def _identity_sets(document: dict[str, Any]) -> dict[str, set[str]]:
    return study._identity_sets(document)


def test_successor_binds_exact_incident_recovery_and_new_identity(
    plan: dict[str, Any],
) -> None:
    study.validate_plan(plan, repo_root=REPO_ROOT)
    assert plan["freeze_nonce"] == study.FREEZE_NONCE
    assert plan["study_id"] == study.STUDY_ID
    assert plan["root_id"] == study.ROOT_ID
    assert plan["execution"]["module"] == (
        "flavourbench.reasoning_effort_full_study_executor_v2"
    )
    assert plan["execution"]["confirmation"] == (
        "RUN_REASONING_EFFORT_V4_ONE_COMPLETE_FAMILY_BLOCK"
    )
    assert plan["execution"]["confirmation"] != (
        "RUN_REASONING_EFFORT_V3_ONE_COMPLETE_FAMILY_BLOCK"
    )
    assert plan["supersedes"]["plans"] == [
        study.PREDECESSOR_PLAN_SHA256,
        study.EARLIER_RETIRED_PLAN_SHA256,
    ]
    assert plan["supersedes"]["pipeline_incident_sha256"] == study.INCIDENT_SHA256
    assert (
        plan["supersedes"]["zero_call_recovery_receipt_sha256"]
        == study.RECOVERY_RECEIPT_SHA256
    )
    recovery_ref = plan["source_artifacts"]["import_recovery_receipt"]
    assert recovery_ref["semantic_sha256"] == study.RECOVERY_RECEIPT_SHA256
    assert recovery_ref["file_sha256"] == (
        "d0f14d3cfcdc8ca2047a047f6997ac199faafc90613669d15d15ae4ab85b7706"
    )
    assert recovery_ref["semantic_sha256"] != recovery_ref["file_sha256"]
    receipt = _json(REPO_ROOT / recovery_ref["path"])
    assert receipt["actual_cost_usd"] == "0"
    assert receipt["provider_completion_requests"] == 0
    assert receipt["epicure_calls"] == 0
    assert receipt["reservation_released"] is True


def test_all_execution_and_presentation_identifiers_are_disjoint(
    plan: dict[str, Any],
) -> None:
    current = _identity_sets(plan)
    assert {key: len(value) for key, value in current.items()} == {
        "work_item_ids": 168,
        "run_ids": 168,
        "arm_ids": 336,
        "attempt_ids": 9408,
        "wave_ids": 24,
        "block_ids": 6,
        "presentation_ids": 240,
    }
    for relative in (study.PREDECESSOR_PLAN_PATH, study.EARLIER_RETIRED_PLAN_PATH):
        retired = _identity_sets(_json(REPO_ROOT / relative))
        assert all(current[key].isdisjoint(retired[key]) for key in current)


def test_estimand_graph_and_blinded_assignment_counts_are_not_conflated(
    plan: dict[str, Any], human_protocol: dict[str, Any]
) -> None:
    canonical = plan["human_evaluation"]["presentations"]
    assert len(canonical) == 240
    assert sum(row["contrast"] == "primary_low_high" for row in canonical) == 144
    assert sum(row["contrast"] != "primary_low_high" for row in canonical) == 96
    assert len(plan["human_evaluation"]["position_swapped_repeat_presentation_ids"]) == 24

    assert len(human_protocol["comparison_cells"]) == 240
    assert human_protocol["counts"]["primary_comparison_cells"] == 144
    assert human_protocol["counts"]["secondary_comparison_cells"] == 96
    assert len(human_protocol["presentations"]) == 1584
    assert human_protocol["counts"]["original_presentations"] == 1440
    assert human_protocol["counts"]["position_swapped_repeats"] == 144
    assert len({row["cell_id"] for row in human_protocol["comparison_cells"]}) == 240


def test_successor_source_closure_is_transitive_and_import_safe(
    plan: dict[str, Any],
) -> None:
    frozen = plan["source_code"]
    closure.verify_source_closure(expected=frozen, repo_root=REPO_ROOT)
    modules = {row["module"] for row in frozen["modules"]}
    assert set(closure.REQUIRED_MODULES) <= modules
    assert frozen["entrypoint_modules"] == sorted(closure.ENTRYPOINT_MODULES)
    source = (
        ROOT / "src/flavourbench/reasoning_effort_full_study_executor_v2.py"
    ).read_text(encoding="utf-8")
    assert (
        "from .live_smoke import CONFIRMATION as LIVE_SMOKE_CONFIRMATION" in source
    )
    assert "from .live_smoke import LIVE_SMOKE_CONFIRMATION" not in source


def test_all_168_runtime_arguments_construct_before_any_side_effect(
    plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def forbidden_attestation(**_: Any) -> list[dict[str, Any]]:
        calls.append("catalog")
        raise AssertionError("catalog attestation must not run")

    monkeypatch.setattr(executor, "_attest_all_endpoints", forbidden_attestation)
    prepared = executor.prepare_all_runtime_items(plan=plan, repo_root=REPO_ROOT)
    assert len(prepared) == 168
    assert calls == []
    for item_id, (policy, args) in prepared.items():
        item = next(row for row in plan["work_items"] if row["work_item_id"] == item_id)
        executor._validate_live_args(plan=plan, item=item, args=args, policy=policy)
        assert Path(args.output_dir).exists() is False


def test_missing_live_smoke_symbol_fails_before_attestation_lock_directory_or_reservation(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = study.successor_roots(plan=plan, repo_root=REPO_ROOT)
    assert all(not root.exists() for root in roots)
    observed: list[str] = []

    async def forbidden_attestation(**_: Any) -> list[dict[str, Any]]:
        observed.append("catalog")
        return []

    def forbidden_live_gate() -> None:
        observed.append("live_gate")

    @contextmanager
    def forbidden_lock(_: Path):
        observed.append("lock")
        yield

    monkeypatch.delattr(live_smoke_module, "CONFIRMATION")
    monkeypatch.setattr(executor, "_attest_all_endpoints", forbidden_attestation)
    monkeypatch.setattr(
        executor, "_require_live_environment_before_reservation", forbidden_live_gate
    )
    monkeypatch.setattr(executor, "_ledger_lock", forbidden_lock)
    with pytest.raises(ImportError):
        import asyncio

        asyncio.run(
            executor.execute_one_block(
                plan=plan,
                human_protocol=human_protocol,
                bound_preflight=bound_preflight,
                repo_root=REPO_ROOT,
                api_base="https://invalid.example",
                api_key="not-used",
            )
        )
    assert observed == []
    assert all(not root.exists() for root in roots)


def test_retired_v3_confirmation_is_rejected_before_file_or_root_access(
    plan: dict[str, Any], tmp_path: Path
) -> None:
    roots = study.successor_roots(plan=plan, repo_root=REPO_ROOT)
    assert all(not root.exists() for root in roots)
    absent = tmp_path / "must-not-be-read.json"
    with pytest.raises(SystemExit, match="exact successor one-family-block confirmation"):
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
                "--confirm",
                "RUN_REASONING_EFFORT_V3_ONE_COMPLETE_FAMILY_BLOCK",
            ]
        )
    assert not absent.exists()
    assert all(not root.exists() for root in roots)


def _fake_accounting(
    *,
    plan: dict[str, Any],
    repo_root: Path,
    coordinator_ledger: Path,
    endpoint_roots: dict[str, Path],
) -> dict[str, Any]:
    del repo_root, endpoint_roots
    entries = executor._load_ledger(coordinator_ledger, role="coordinator")
    state = executor._coordinator_state(plan, entries)
    completed = len(state["completed"])
    active_id = state["active_block_id"]
    next_id = plan["block_execution_order"][completed] if completed < 6 else None
    reserve = (
        next(
            Decimal(block["worst_case_reserve_usd"])
            for block in plan["admission_blocks"]
            if block["admission_block_id"] == next_id
        )
        if next_id
        else Decimal(0)
    )
    blockers = [
        incident
        for block_id, incidents in state["incidents"].items()
        if block_id not in state["completed"]
        for incident in incidents
    ]
    return {
        "current_total_exposure_usd": str(study.CURRENT_EXPOSURE_USD),
        "next_block_id": next_id,
        "next_block_projected_total_usd": str(study.CURRENT_EXPOSURE_USD + reserve),
        "new_block_admission_allowed": bool(not active_id and not blockers and next_id),
        "active_block_resume_allowed": bool(active_id and not blockers),
        "active_block_id": active_id,
        "completed_family_blocks": completed,
        "blockers": blockers,
    }


def _fake_attestations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for endpoint_id in study.ENDPOINTS:
        records.append(
            {
                "endpoint_id": endpoint_id,
                "raw_execution_contract": {"tag": endpoint_id, "provider_name": endpoint_id},
                "raw_execution_contract_sha256": hashlib.sha256(
                    endpoint_id.encode("utf-8")
                ).hexdigest(),
            }
        )
    return records


def _configure_fake_execution(
    *,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    tmp_path: Path,
    source_mode: str = "all",
) -> tuple[Path, dict[str, Path]]:
    coordinator = tmp_path / study.ROOT_ID / "coordinator"
    endpoints = {
        endpoint_id: tmp_path / study.ROOT_ID / endpoint_id
        for endpoint_id in study.ENDPOINTS
    }
    monkeypatch.setattr(executor, "_roots", lambda plan, repo_root: (coordinator, endpoints))
    monkeypatch.setattr(executor, "_require_live_environment_before_reservation", lambda: None)

    @contextmanager
    def no_global_lock(_: Path):
        yield

    monkeypatch.setattr(frontier_runner, "_exclusive_runner_lock", no_global_lock)
    monkeypatch.setattr(executor, "_accounting", _fake_accounting)

    async def fake_attest(**_: Any) -> list[dict[str, Any]]:
        return _fake_attestations(plan)

    async def fake_invoke(**_: Any) -> None:
        return None

    monkeypatch.setattr(executor, "_attest_all_endpoints", fake_attest)
    monkeypatch.setattr(executor, "_invoke_live_pair", fake_invoke)
    original_file_ref = study._file_ref

    def file_ref(repo_root: Path, path: Path) -> dict[str, Any]:
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

    monkeypatch.setattr(study, "_file_ref", file_ref)
    first_id = plan["admission_blocks"][0]["work_item_ids"][0]

    def fake_source(endpoint_root: Path, work_item_id: str):
        del endpoint_root
        if source_mode == "none" or (source_mode == "except_first" and work_item_id == first_id):
            return None
        return Path("fake-source.json"), {}, hashlib.sha256(work_item_id.encode()).hexdigest()

    def fake_payload(**kwargs: Any) -> dict[str, Any]:
        item = kwargs["item"]
        return {
            "disposition": "source_usable",
            "actual_cost_usd": "0.01",
            "source_artifact_sha256": hashlib.sha256(
                str(item["work_item_id"]).encode()
            ).hexdigest(),
        }

    monkeypatch.setattr(executor, "_source_for_item", fake_source)
    monkeypatch.setattr(executor, "_source_terminal_payload", fake_payload)
    monkeypatch.setattr(
        executor,
        "_journal_evidence",
        lambda source_root, work_item_id: {
            "journal_count": 0,
            "request_started_count": 0,
            "journals": [],
        },
    )
    return coordinator, endpoints


@pytest.mark.asyncio
async def test_fake_provider_completes_one_atomic_block_without_network(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinator, endpoints = _configure_fake_execution(
        monkeypatch=monkeypatch, plan=plan, tmp_path=tmp_path
    )
    result = await executor.execute_one_block(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
    )
    assert result["document"]["all_runtime_items_prevalidated"] == 168
    assert result["document"]["provider_pair_invocations"] == 28
    assert result["document"]["completed_family_blocks"] == 1
    assert all(row["decision"] == "source_usable" for row in result["document"]["outcomes"])
    state = executor._coordinator_state(
        plan, executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")
    )
    assert len(state["terminals"]) == 28
    assert len(state["completed"]) == 1
    assert sum(
        len(executor._load_ledger(root / "ledger.jsonl", role="endpoint"))
        for root in endpoints.values()
    ) == 56


@pytest.mark.asyncio
async def test_failure_after_attestation_before_reservation_creates_no_reservation(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinator, _ = _configure_fake_execution(
        monkeypatch=monkeypatch, plan=plan, tmp_path=tmp_path
    )

    def fail(stage: str, item_id: str | None) -> None:
        del item_id
        if stage == "after_attestation_before_reservation":
            raise RuntimeError("injected-before-reservation")

    with pytest.raises(RuntimeError, match="injected-before-reservation"):
        await executor.execute_one_block(
            plan=plan,
            human_protocol=human_protocol,
            bound_preflight=bound_preflight,
            repo_root=REPO_ROOT,
            api_base="https://invalid.example",
            api_key="not-used",
            failure_injector=fail,
        )
    assert not (coordinator / "ledger.jsonl").exists()


@pytest.mark.asyncio
async def test_failure_after_reservation_is_durable_and_stops_before_item_start(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinator, endpoints = _configure_fake_execution(
        monkeypatch=monkeypatch, plan=plan, tmp_path=tmp_path
    )

    def fail(stage: str, item_id: str | None) -> None:
        del item_id
        if stage == "after_reservation_before_first_item_start":
            raise RuntimeError("injected-after-reservation")

    result = await executor.execute_one_block(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
        failure_injector=fail,
    )
    assert result["document"]["outcomes"][-1]["decision"] == (
        "durable_pipeline_incident_stop"
    )
    entries = executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")
    assert [row["event_type"] for row in entries] == [
        "family_block_reservation_created",
        "family_block_execution_incident",
    ]
    assert all(not (root / "ledger.jsonl").exists() for root in endpoints.values())
    assert list((coordinator / "operation-incidents").glob("*.json"))


@pytest.mark.asyncio
async def test_failure_after_item_start_before_request_terminalizes_zero_cost_without_replay(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinator, _ = _configure_fake_execution(
        monkeypatch=monkeypatch,
        plan=plan,
        tmp_path=tmp_path,
        source_mode="except_first",
    )
    first = plan["admission_blocks"][0]["work_item_ids"][0]
    fired = False

    def fail(stage: str, item_id: str | None) -> None:
        nonlocal fired
        if stage == "after_item_start_before_provider_request" and item_id == first and not fired:
            fired = True
            raise RuntimeError("injected-before-request")

    result = await executor.execute_one_block(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
        failure_injector=fail,
    )
    decisions = {row["work_item_id"]: row["decision"] for row in result["document"]["outcomes"]}
    assert decisions[first] == "pre_generation_failure_zero_cost"
    assert result["document"]["completed_family_blocks"] == 1
    state = executor._coordinator_state(
        plan, executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")
    )
    assert state["terminals"][first]["disposition"] == (
        "pre_generation_failure_zero_cost"
    )


@pytest.mark.asyncio
async def test_request_started_without_source_retains_reserve_and_stops(
    plan: dict[str, Any],
    human_protocol: dict[str, Any],
    bound_preflight: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinator, _ = _configure_fake_execution(
        monkeypatch=monkeypatch,
        plan=plan,
        tmp_path=tmp_path,
        source_mode="except_first",
    )
    first = plan["admission_blocks"][0]["work_item_ids"][0]

    async def request_then_fail(**_: Any) -> None:
        raise RuntimeError("injected-after-request-start")

    monkeypatch.setattr(executor, "_invoke_live_pair", request_then_fail)
    monkeypatch.setattr(
        executor,
        "_journal_evidence",
        lambda source_root, work_item_id: {
            "journal_count": 1 if work_item_id == first else 0,
            "request_started_count": 1 if work_item_id == first else 0,
            "journals": [],
        },
    )
    result = await executor.execute_one_block(
        plan=plan,
        human_protocol=human_protocol,
        bound_preflight=bound_preflight,
        repo_root=REPO_ROOT,
        api_base="https://invalid.example",
        api_key="not-used",
    )
    assert len(result["document"]["outcomes"]) == 1
    assert result["document"]["outcomes"][0]["work_item_id"] == first
    assert result["document"]["outcomes"][0]["decision"] == (
        "request_started_no_source_stop"
    )
    state = executor._coordinator_state(
        plan, executor._load_ledger(coordinator / "ledger.jsonl", role="coordinator")
    )
    assert state["active_block_id"] == plan["admission_blocks"][0]["admission_block_id"]
    assert state["incidents"][state["active_block_id"]]
    assert not state["completed"]


def test_successor_roots_are_empty_and_historical_files_are_immutable(
    plan: dict[str, Any],
) -> None:
    study.assert_successor_roots_empty(plan=plan, repo_root=REPO_ROOT)
    expected = {
        "src/flavourbench/reasoning_effort_full_study_executor_v1.py": (
            "aa13f05ccaf7b85c91f3918f158fd76f242c7780674d82f8cd872fcb656d6ad4"
        ),
        "src/flavourbench/reasoning_effort_full_study_v1.py": (
            "ad816b5179ffc091a1da4acfe303a7b6f23eb865dfea6c96b4973fb6de7046fc"
        ),
        "src/flavourbench/live_smoke.py": (
            "2cfdb38052df96082e74ad603e06d7d49701416a4cbc7c5b0fae7dafab84a42c"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
