"""Zero-network production-path tests for the V7 reasoning successor."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier
import flavourbench.reasoning_effort_full_study_executor_v7 as executor
import flavourbench.reasoning_effort_full_study_v7 as study
from flavourbench.run_journal import RunJournal

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
LIVE_LEDGER = REPO_ROOT / study.GLOBAL_LEDGER_PATH
LIVE_SOURCES = REPO_ROOT / study.GLOBAL_SOURCE_PATH

CRASH_AND_EXCEPTION_CUTS = (
    "after_all_runtime_validation_before_side_effect",
    "before_endpoint_attestations",
    "after_endpoint_attestations",
    "after_attestation_binding_before_global_lock",
    "after_global_lock_before_reservations",
    "before_global_reservation",
    "after_global_reservation",
    "before_local_block_reservation",
    "after_local_block_reservation",
    "before_item_start",
    "after_item_start",
    "before_source_lookup",
    "before_provider_invocation",
    "after_provider_invocation_before_source_lookup",
    "before_source_classification",
    "after_source_classification",
    "before_global_artifact_finalization",
    "after_global_artifact_finalization",
    "before_local_terminal",
    "after_endpoint_terminal_before_coordinator_terminal",
    "after_coordinator_terminal",
)
PRE_CLASSIFICATION_CUTS = frozenset(CRASH_AND_EXCEPTION_CUTS[:9])
PRESTART_CUTS = frozenset({"before_item_start"})
POSTSTART_INCIDENT_CUTS = frozenset(CRASH_AND_EXCEPTION_CUTS[10:17])
RECOVERED_TERMINAL_CUTS = frozenset(CRASH_AND_EXCEPTION_CUTS[17:])


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return study.build_plan(repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def human(plan: dict[str, Any]) -> dict[str, Any]:
    return study.build_human_protocol(plan=plan, repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def bound(plan: dict[str, Any], human: dict[str, Any]) -> dict[str, Any]:
    return study.build_bound_preflight(plan=plan, human_protocol=human, repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def governance_go(
    plan: dict[str, Any], human: dict[str, Any], bound: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": study.GOVERNANCE_GO_SCHEMA,
        "record_role": "offline_test_go_not_live_authority",
        "decision": "go_for_exactly_one_family_block",
        "study_plan_sha256": plan["artifact_sha256"],
        "human_protocol_sha256": human["artifact_sha256"],
        "bound_preflight_sha256": bound["artifact_sha256"],
        "authorized_admission_block_id": plan["admission_blocks"][0]["admission_block_id"],
        "reviewer_is_executor": False,
        "reviewer_is_v6_independent_reviewer": False,
        "reviewer_is_v7_builder": False,
        "reviewed_v6_no_go_sha256": study.V6_NO_GO_SHA256,
        "provider_or_epicure_calls_made_by_review": False,
        "maximum_family_blocks": 1,
    }
    return {**payload, "artifact_sha256": study._sha256(payload)}


def _attestations() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for endpoint_id in study.ENDPOINTS:
        raw = {
            "tag": endpoint_id,
            "provider_name": endpoint_id,
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
        records.append(
            {
                "endpoint_id": endpoint_id,
                "raw_execution_contract": raw,
                "raw_execution_contract_sha256": study._sha256(raw),
                "catalog_http_gets": 0,
            }
        )
    return records


def _rehash_live_artifact(root: Path, body: dict[str, Any], prefix: str) -> Path:
    body.pop("artifact_sha256", None)
    digest = frontier._live_smoke_sha256(body)
    document = {**body, "artifact_sha256": digest}
    path = root / f"{prefix}-{digest[:12]}.json"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frontier._verify_live_artifact(path)
    return path


def _historical_template() -> dict[str, Any]:
    return _json(sorted(LIVE_SOURCES.glob("*.json"))[0])


def _write_current_source(*, root: Path, plan: dict[str, Any], item: dict[str, Any]) -> Path:
    body = _historical_template()
    body.update(
        {
            "run_id": item["run_id"],
            "dataset_work_item_id": item["work_item_id"],
            "dataset_task_id": item["task"]["task_id"],
            "requested_model_id": item["route_coordinate"]["model_id"],
            "requested_provider": item["route_coordinate"]["provider_endpoint"],
            "candidate_manifest_sha256": item["manifest"]["semantic_sha256"],
            "provider_attempt_events": [],
            "mcp_trace_events": [],
            "official": False,
            "rank_eligible": False,
        }
    )
    journal = RunJournal.create(
        root,
        run_id=item["run_id"],
        metadata={
            "dataset_work_item_id": item["work_item_id"],
            "dataset_task_id": item["task"]["task_id"],
            "candidate_manifest_sha256": item["manifest"]["semantic_sha256"],
        },
    )
    body["run_journal"] = journal.finalize({"status": "complete"}).payload()
    return _rehash_live_artifact(root, body, "flavourbench-live-smoke-v7-test")


def _classifier(**kwargs: Any) -> dict[str, Any]:
    path, _, digest = kwargs["source_record"]
    item_id = str(kwargs["item"]["work_item_id"])
    return {
        "disposition": "source_usable",
        "source_path": str(path),
        "source_artifact_sha256": digest,
        "pair_audit_path": f"offline-audit/{item_id}.json",
        "pair_audit_sha256": hashlib.sha256(f"audit:{item_id}".encode()).hexdigest(),
        "actual_cost_usd": "0",
        "audit_failures": [],
    }


@dataclass
class OfflineRun:
    root: Path
    coordinator: Path
    endpoints: dict[str, Path]
    ledger: Path
    sources: Path
    invocations: dict[str, int]
    attest_calls: list[int]
    adapters: executor.ExecutionAdapters

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _offline_run(
    plan: dict[str, Any],
    *,
    produce_source: bool = True,
    classifier: Any = _classifier,
) -> OfflineRun:
    root = ROOT / ".pytest_cache" / f"v7-production-path-{uuid.uuid4().hex}"
    coordinator = root / "coordinator"
    endpoints = {endpoint: root / endpoint for endpoint in study.ENDPOINTS}
    sources = root / "live-smoke"
    sources.mkdir(parents=True)
    for source in LIVE_SOURCES.glob("*.json"):
        shutil.copyfile(source, sources / source.name)
    ledger = root / "frontier-contract/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    shutil.copyfile(LIVE_LEDGER, ledger)
    items = executor._item_map(plan)
    invocations: dict[str, int] = {}
    attest_calls: list[int] = []

    async def attest(**_: Any) -> list[dict[str, Any]]:
        attest_calls.append(1)
        return _attestations()

    async def invoke(*, args: Any, **_: Any) -> None:
        item_id = str(args.dataset_work_item_id)
        invocations[item_id] = invocations.get(item_id, 0) + 1
        if produce_source:
            _write_current_source(root=sources, plan=plan, item=items[item_id])

    adapters = executor.ExecutionAdapters(
        attest_all=attest,
        invoke_pair=invoke,
        classify_source=classifier,
        require_live_environment=lambda: None,
        roots=(coordinator, endpoints),
        global_ledger_path=ledger,
        source_root=sources,
    )
    return OfflineRun(
        root,
        coordinator,
        endpoints,
        ledger,
        sources,
        invocations,
        attest_calls,
        adapters,
    )


async def _execute(
    *,
    run: OfflineRun,
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
    failure_injector: Any = None,
) -> dict[str, Any]:
    return await executor.execute_one_block(
        plan=plan,
        human_protocol=human,
        bound_preflight=bound,
        governance_go=governance_go,
        repo_root=REPO_ROOT,
        api_base="https://network-forbidden.invalid",
        api_key="not-used",
        failure_injector=failure_injector,
        adapters=run.adapters,
    )


def _validate_all_durable_state(plan: dict[str, Any], run: OfflineRun) -> None:
    entries = frontier.load_ledger(run.ledger)
    executor._verify_global_anchor(plan=plan, entries=entries)
    executor._campaign_global_reservations(plan=plan, entries=entries)
    executor._global_accounting_locked(
        plan=plan,
        repo_root=REPO_ROOT,
        global_ledger=run.ledger,
        source_root=run.sources,
    )
    executor._verified_source_index(
        run.sources,
        current_work_item_ids={item["work_item_id"] for item in plan["work_items"]},
    )
    coordinator_ledger = run.coordinator / "ledger.jsonl"
    if coordinator_ledger.exists():
        executor._coordinator_state(
            plan,
            executor._load_ledger(coordinator_ledger, role="coordinator"),
        )
    for endpoint in run.endpoints.values():
        ledger = endpoint / "ledger.jsonl"
        if ledger.exists():
            executor._endpoint_state(executor._load_ledger(ledger, role="endpoint"))
    for receipt in (run.coordinator / "receipts").glob("*.json"):
        document = _json(receipt)
        body = {key: value for key, value in document.items() if key != "artifact_sha256"}
        assert document["artifact_sha256"] == study._sha256(body)


def test_v7_binds_exact_v6_no_go_and_preserves_design(
    plan: dict[str, Any], human: dict[str, Any], bound: dict[str, Any]
) -> None:
    study.validate_plan(plan, repo_root=REPO_ROOT)
    assert plan["supersedes"]["v6_independent_no_go_sha256"] == study.V6_NO_GO_SHA256
    assert plan["supersedes"]["v6_independent_no_go_file_sha256"] == (study.V6_NO_GO_FILE_SHA256)
    no_go = REPO_ROOT / study.V6_NO_GO_PATH
    assert hashlib.sha256(no_go.read_bytes()).hexdigest() == study.V6_NO_GO_FILE_SHA256
    assert plan["status"] == "frozen_not_executed_independent_go_required"
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
    assert len(human["arm_coordinates"]) == 336
    assert len(human["comparison_cells"]) == 240
    assert len(human["presentations"]) == 1584
    block = plan["admission_blocks"][0]
    items = executor._item_map(plan)
    exact = study._exact_sum(
        [Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in block["work_item_ids"]]
    )
    assert exact == Decimal("17.58085589333333333333333333402")
    assert exact == Decimal(block["worst_case_reserve_usd"])
    assert exact == Decimal(plan["budget"]["first_block_worst_case_usd"])
    assert exact == Decimal(bound["checks"]["first_block_reserve_usd"])


def test_v7_namespaces_are_disjoint_from_v2_through_v6(
    plan: dict[str, Any], human: dict[str, Any]
) -> None:
    current = study._identity_sets(plan)
    historical = (
        (study.V6_PLAN_PATH, study.V6_PLAN_SHA256, study.V6_PLAN_FILE_SHA256),
        (
            study.v6.V5_PLAN_PATH,
            study.v6.V5_PLAN_SHA256,
            study.v6.V5_PLAN_FILE_SHA256,
        ),
        (study.v6.v5.V4_PLAN_PATH, study.v6.v5.V4_PLAN_SHA256, None),
        (study.v6.v5.V3_PLAN_PATH, study.v6.v5.V3_PLAN_SHA256, None),
        (study.v6.v5.V2_PLAN_PATH, study.v6.v5.V2_PLAN_SHA256, None),
    )
    for relative, semantic, physical in historical:
        prior = study._identity_sets(
            study._verified_artifact(REPO_ROOT, relative, semantic, physical)
        )
        assert all(current[key].isdisjoint(prior[key]) for key in current)
    v6_human = _json(REPO_ROOT / study.V6_HUMAN_PATH)
    assert {row["presentation_id"] for row in human["presentations"]}.isdisjoint(
        {row["presentation_id"] for row in v6_human["presentations"]}
    )


def test_every_incident_replay_api_requires_global_ledger() -> None:
    signature = inspect.signature(executor._replay_endpoint_incident_exactly)
    assert "global_ledger" in signature.parameters
    assert signature.parameters["global_ledger"].default is inspect.Parameter.empty
    source = inspect.getsource(executor._classification_fence)
    assert source.count("_replay_endpoint_incident_exactly(") == 2
    assert source.count("global_ledger=global_ledger") >= 4


def test_historical_source_and_nested_raw_attestation_controls(plan: dict[str, Any]) -> None:
    current = {item["work_item_id"] for item in plan["work_items"]}
    assert executor._verified_source_index(LIVE_SOURCES, current_work_item_ids=current) == {}
    prepared = executor.prepare_all_runtime_items(plan=plan, repo_root=REPO_ROOT)
    block = plan["admission_blocks"][0]
    rebound = executor._bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=_attestations(),
        prepared_all=prepared,
        repo_root=REPO_ROOT,
        source_root=LIVE_SOURCES,
    )
    assert list(rebound) == block["work_item_ids"]
    assert all("pricing" in runtime.raw_execution_contract for runtime in rebound.values())
    assert all(
        "raw_execution_contract" not in runtime.raw_execution_contract
        for runtime in rebound.values()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", CRASH_AND_EXCEPTION_CUTS)
async def test_all_21_process_crash_cuts_recover(
    cut: str,
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan)
    first = plan["admission_blocks"][0]["work_item_ids"][0]
    fired = False

    def crash(stage: str, item_id: str | None) -> None:
        nonlocal fired
        if not fired and stage == cut and (item_id is None or item_id == first):
            fired = True
            raise executor.SimulatedCrash(cut)

    try:
        with pytest.raises(executor.SimulatedCrash):
            await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
                failure_injector=crash,
            )
        assert fired
        _validate_all_durable_state(plan, run)
        result = await _execute(
            run=run,
            plan=plan,
            human=human,
            bound=bound,
            governance_go=governance_go,
        )
        assert result["decision"] == "block_terminal"
        _validate_all_durable_state(plan, run)
        assert max(run.invocations.values(), default=0) <= 1
    finally:
        run.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", CRASH_AND_EXCEPTION_CUTS)
async def test_all_21_normal_exception_cuts_are_valid_before_and_after_restart(
    cut: str,
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan)
    first = plan["admission_blocks"][0]["work_item_ids"][0]
    fired = False

    def fail(stage: str, item_id: str | None) -> None:
        nonlocal fired
        if not fired and stage == cut and (item_id is None or item_id == first):
            fired = True
            raise RuntimeError(f"normal-exception:{cut}")

    try:
        if cut in PRE_CLASSIFICATION_CUTS:
            with pytest.raises(RuntimeError, match="normal-exception"):
                await _execute(
                    run=run,
                    plan=plan,
                    human=human,
                    bound=bound,
                    governance_go=governance_go,
                    failure_injector=fail,
                )
            expected_restart = "block_terminal"
        else:
            result = await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
                failure_injector=fail,
            )
            if cut in PRESTART_CUTS:
                assert result["decision"] == "durable_incident_stop"
                assert result["outcomes"][0]["decision"] == ("durable_pre_start_no_delivery")
                expected_restart = "blocked_by_durable_incident"
            elif cut in POSTSTART_INCIDENT_CUTS:
                assert result["decision"] == "durable_incident_stop"
                assert result["outcomes"][0]["decision"] == ("durable_incident_reservation_derived")
                expected_restart = "blocked_by_durable_incident"
            elif cut in RECOVERED_TERMINAL_CUTS:
                assert result["decision"] == "block_terminal"
                expected_restart = "block_terminal_receipt_recovered"
            else:  # pragma: no cover - the frozen cut partition is exhaustive
                raise AssertionError(f"unclassified normal cut: {cut}")
        assert fired
        _validate_all_durable_state(plan, run)

        first_item = executor._item_map(plan)[first]
        endpoint_ledger = (
            run.endpoints[first_item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl"
        )
        if cut in PRESTART_CUTS:
            assert not endpoint_ledger.exists()
            state = executor._coordinator_state(
                plan,
                executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
            )
            incident = state["incidents"][state["active_block_id"]][0]
            assert incident["incident"] == "durable_pre_start_no_delivery"
            assert incident["endpoint_incident_appended"] is False
            assert incident["delivery_evidence"]["endpoint_event_count"] == 0
        if cut in {
            "after_endpoint_terminal_before_coordinator_terminal",
            "after_coordinator_terminal",
        }:
            endpoint_state = executor._endpoint_state(
                executor._load_ledger(endpoint_ledger, role="endpoint")
            )
            assert first in endpoint_state["terminals"]
            assert endpoint_state["incidents"] == {}

        restarted = await _execute(
            run=run,
            plan=plan,
            human=human,
            bound=bound,
            governance_go=governance_go,
        )
        assert restarted["decision"] == expected_restart
        _validate_all_durable_state(plan, run)
        assert max(run.invocations.values(), default=0) <= 1
    finally:
        run.close()


@pytest.mark.asyncio
async def test_independent_prestart_reproducer_creates_no_endpoint_incident(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan)
    first = plan["admission_blocks"][0]["work_item_ids"][0]

    def fail(stage: str, item_id: str | None) -> None:
        if stage == "before_item_start" and item_id == first:
            raise RuntimeError("independent V6-NG-002 reproduction")

    try:
        result = await _execute(
            run=run,
            plan=plan,
            human=human,
            bound=bound,
            governance_go=governance_go,
            failure_injector=fail,
        )
        assert result["outcomes"] == [
            {
                "work_item_id": first,
                "decision": "durable_pre_start_no_delivery",
                "incident_entry_sha256": result["outcomes"][0]["incident_entry_sha256"],
            }
        ]
        item = executor._item_map(plan)[first]
        endpoint_ledger = run.endpoints[item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl"
        assert not endpoint_ledger.exists()
        assert run.invocations == {}
        _validate_all_durable_state(plan, run)
    finally:
        run.close()


@pytest.mark.asyncio
async def test_independent_stale_retained_reproducer_fails_closed(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan, produce_source=False)
    block = plan["admission_blocks"][0]
    first = block["work_item_ids"][0]

    def crash(stage: str, item_id: str | None) -> None:
        if stage == "after_endpoint_incident_before_coordinator_incident" and item_id == first:
            raise executor.SimulatedCrash(stage)

    try:
        with pytest.raises(executor.SimulatedCrash):
            await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
                failure_injector=crash,
            )
        items = executor._item_map(plan)
        item = items[first]
        endpoint_ledger = run.endpoints[item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl"
        endpoint_incident = executor._endpoint_state(
            executor._load_ledger(endpoint_ledger, role="endpoint")
        )["incidents"][first]
        assert endpoint_incident["canonical_reservation_status"] == "active_reservation"
        assert endpoint_incident["canonical_reservation_retained"] is True
        coordinator_state = executor._coordinator_state(
            plan,
            executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
        )
        assert coordinator_state["incidents"] == {}

        source_path = _write_current_source(root=run.sources, plan=plan, item=item)
        source = executor._verified_source_index(
            run.sources,
            current_work_item_ids={value["work_item_id"] for value in plan["work_items"]},
        )[first]
        assert source[0] == source_path
        reservations = executor._campaign_global_reservations(
            plan=plan, entries=frontier.load_ledger(run.ledger)
        )
        artifact_event = executor._record_canonical_artifact(
            plan=plan,
            block=block,
            item=item,
            canonical_reservation=reservations[first],
            source_record=source,
            global_ledger=run.ledger,
        )
        assert artifact_event["event_type"] == "artifact_recorded"
        assert reservations[first]["entry_sha256"] not in frontier.active_ledger_reservations(
            frontier.load_ledger(run.ledger)
        )

        with pytest.raises(
            executor.FullStudyExecutionError,
            match="stale endpoint incident canonical disposition",
        ):
            await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
            )
        coordinator_state = executor._coordinator_state(
            plan,
            executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
        )
        assert coordinator_state["incidents"] == {}
        observed = executor._endpoint_state(
            executor._load_ledger(endpoint_ledger, role="endpoint")
        )["incidents"][first]
        assert observed["entry_sha256"] == endpoint_incident["entry_sha256"]
        _validate_all_durable_state(plan, run)

        with pytest.raises(
            executor.FullStudyExecutionError,
            match="stale endpoint incident canonical disposition",
        ):
            executor._append_incident_idempotent(
                plan=plan,
                block=block,
                item=item,
                local_reservation=coordinator_state["reservations"][block["admission_block_id"]],
                canonical_reservation=reservations[first],
                endpoint_ledger=endpoint_ledger,
                coordinator_ledger=run.coordinator / "ledger.jsonl",
                global_ledger=run.ledger,
                evidence={},
                error=RuntimeError("direct stale replay"),
            )
    finally:
        run.close()


@pytest.mark.asyncio
async def test_exception_recovery_replay_path_rejects_intervening_finalization(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run_holder: dict[str, OfflineRun] = {}
    fired = False

    def adversarial_classifier(**kwargs: Any) -> dict[str, Any]:
        nonlocal fired
        run = run_holder["run"]
        item = kwargs["item"]
        item_id = item["work_item_id"]
        plan_value = kwargs["plan"]
        block = plan_value["admission_blocks"][0]
        coordinator_ledger = run.coordinator / "ledger.jsonl"
        coordinator_state = executor._coordinator_state(
            plan_value,
            executor._load_ledger(coordinator_ledger, role="coordinator"),
        )
        local_reservation = coordinator_state["reservations"][block["admission_block_id"]]
        reservations = executor._campaign_global_reservations(
            plan=plan_value, entries=frontier.load_ledger(run.ledger)
        )
        endpoint_ledger = run.endpoints[item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl"

        def interrupt(stage: str, observed_item: str | None) -> None:
            if (
                stage == "after_endpoint_incident_before_coordinator_incident"
                and observed_item == item_id
            ):
                raise RuntimeError("endpoint-only incident committed")

        with pytest.raises(RuntimeError, match="endpoint-only"):
            executor._append_incident_idempotent(
                plan=plan_value,
                block=block,
                item=item,
                local_reservation=local_reservation,
                canonical_reservation=reservations[item_id],
                endpoint_ledger=endpoint_ledger,
                coordinator_ledger=coordinator_ledger,
                global_ledger=run.ledger,
                evidence={"journal_count": 0},
                error=RuntimeError("classifier incident"),
                failure_injector=interrupt,
            )
        source_record = kwargs["source_record"]
        executor._record_canonical_artifact(
            plan=plan_value,
            block=block,
            item=item,
            canonical_reservation=reservations[item_id],
            source_record=source_record,
            global_ledger=run.ledger,
        )
        fired = True
        raise RuntimeError("classification fails after disposition drift")

    run = _offline_run(plan, classifier=adversarial_classifier)
    run_holder["run"] = run
    try:
        with pytest.raises(
            executor.FullStudyExecutionError,
            match="stale endpoint incident canonical disposition",
        ):
            await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
            )
        assert fired
        state = executor._coordinator_state(
            plan,
            executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
        )
        assert state["incidents"] == {}
        _validate_all_durable_state(plan, run)
    finally:
        run.close()


@pytest.mark.asyncio
async def test_terminal_block_receipt_recovers_before_attestation(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan)

    def crash(stage: str, _item_id: str | None) -> None:
        if stage == "after_block_terminal":
            raise executor.SimulatedCrash(stage)

    try:
        with pytest.raises(executor.SimulatedCrash):
            await _execute(
                run=run,
                plan=plan,
                human=human,
                bound=bound,
                governance_go=governance_go,
                failure_injector=crash,
            )
        assert len(run.attest_calls) == 1
        assert not (run.coordinator / "receipts").exists()
        result = await _execute(
            run=run,
            plan=plan,
            human=human,
            bound=bound,
            governance_go=governance_go,
        )
        assert result["decision"] == "block_terminal_receipt_recovered"
        assert len(run.attest_calls) == 1
        _validate_all_durable_state(plan, run)
    finally:
        run.close()


def test_double_freeze_byte_determinism_and_zero_live_side_effects(tmp_path: Path) -> None:
    output = ROOT / ".pytest_cache" / f"v7-freeze-{tmp_path.name}"
    shutil.rmtree(output, ignore_errors=True)
    ledger_before = LIVE_LEDGER.read_bytes()
    source_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in LIVE_SOURCES.glob("*")
        if path.is_file()
    }
    predecessor_paths = [
        REPO_ROOT / study.V6_PLAN_PATH,
        REPO_ROOT / study.V6_HUMAN_PATH,
        REPO_ROOT / study.V6_PREFLIGHT_PATH,
        REPO_ROOT / study.V6_BOUND_PATH,
        REPO_ROOT / study.V6_NO_GO_PATH,
        ROOT / "src/flavourbench/reasoning_effort_full_study_v6.py",
        ROOT / "src/flavourbench/reasoning_effort_full_study_executor_v6.py",
        ROOT / "src/flavourbench/reasoning_effort_source_closure_v6.py",
        ROOT / "tests/reasoning_effort_crash_safe_v6_test.py",
    ]
    predecessor_before = {path: path.read_bytes() for path in predecessor_paths}
    try:
        first = study.freeze(repo_root=REPO_ROOT, output_dir=output)
        first_bytes = {key: path.read_bytes() for key, path in first.items()}
        second = study.freeze(repo_root=REPO_ROOT, output_dir=output)
        assert first == second
        assert first_bytes == {key: path.read_bytes() for key, path in second.items()}
    finally:
        shutil.rmtree(output, ignore_errors=True)
    assert LIVE_LEDGER.read_bytes() == ledger_before
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in LIVE_SOURCES.glob("*")
        if path.is_file()
    } == source_before
    assert {path: path.read_bytes() for path in predecessor_paths} == predecessor_before


def test_v6_and_canonical_hashes_are_unchanged() -> None:
    expected = {
        ROOT / "src/flavourbench/reasoning_effort_full_study_v6.py": (
            "5e5cfb426594c548f335904cbf2634ed4d61ed73b87245e5541c4ae5ed84583d"
        ),
        ROOT / "src/flavourbench/reasoning_effort_full_study_executor_v6.py": (
            "70fa3408dacece513e34a49bab81756e436a2501f6455ba521cbbb069458c1bc"
        ),
        ROOT / "src/flavourbench/reasoning_effort_source_closure_v6.py": (
            "d1a425e857ec447ac0044b0f58c6347a5d195eee79d1f40244da51239e2a1a74"
        ),
        ROOT / "tests/reasoning_effort_crash_safe_v6_test.py": (
            "8997fd326f30ec6dca177334c9f01889a5719df8922864128a6257e6f6ca3d07"
        ),
        LIVE_LEDGER: study.GLOBAL_LEDGER_ANCHOR_FILE_SHA256,
        REPO_ROOT / study.V6_NO_GO_PATH: study.V6_NO_GO_FILE_SHA256,
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert len(frontier.load_ledger(LIVE_LEDGER)) == 29
    assert frontier.active_ledger_reservations(frontier.load_ledger(LIVE_LEDGER)) == {}
    assert len(list(LIVE_SOURCES.glob("*.json"))) == 39
