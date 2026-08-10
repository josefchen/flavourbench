"""Zero-network production-path tests for the V6 reasoning successor."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import flavourbench.frontier_contract_runner as frontier
import flavourbench.reasoning_effort_full_study_executor_v6 as executor
import flavourbench.reasoning_effort_full_study_v6 as study
from flavourbench.run_journal import RunJournal

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
LIVE_LEDGER = REPO_ROOT / study.GLOBAL_LEDGER_PATH
LIVE_SOURCES = REPO_ROOT / study.GLOBAL_SOURCE_PATH


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
        "reviewer_is_v5_independent_reviewer": False,
        "reviewed_v5_no_go_sha256": study.V5_NO_GO_SHA256,
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
    return _rehash_live_artifact(root, body, "flavourbench-live-smoke-v6-test")


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
    raw_contracts_seen: list[dict[str, Any]]
    adapters: executor.ExecutionAdapters

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _offline_run(plan: dict[str, Any], *, produce_source: bool = True) -> OfflineRun:
    root = ROOT / ".pytest_cache" / f"v6-production-path-{uuid.uuid4().hex}"
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
    raw_contracts_seen: list[dict[str, Any]] = []

    async def attest(**_: Any) -> list[dict[str, Any]]:
        attest_calls.append(1)
        return _attestations()

    async def invoke(*, args: Any, raw_endpoint: Any, **_: Any) -> None:
        item_id = str(args.dataset_work_item_id)
        invocations[item_id] = invocations.get(item_id, 0) + 1
        raw_contracts_seen.append(dict(raw_endpoint))
        if produce_source:
            _write_current_source(root=sources, plan=plan, item=items[item_id])

    adapters = executor.ExecutionAdapters(
        attest_all=attest,
        invoke_pair=invoke,
        classify_source=_classifier,
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
        raw_contracts_seen,
        adapters,
    )


def test_v6_binds_v5_no_go_preserves_estimand_and_uses_exact_decimal(
    plan: dict[str, Any], human: dict[str, Any], bound: dict[str, Any]
) -> None:
    study.validate_plan(plan, repo_root=REPO_ROOT)
    supersedes = plan["supersedes"]
    assert supersedes["retired_v5_plan_sha256"] == study.V5_PLAN_SHA256
    assert supersedes["retired_v5_plan_file_sha256"] == study.V5_PLAN_FILE_SHA256
    assert supersedes["v5_independent_no_go_sha256"] == study.V5_NO_GO_SHA256
    assert supersedes["v5_independent_no_go_file_sha256"] == study.V5_NO_GO_FILE_SHA256
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
    assert len(human["presentations"]) == 1584
    assert len(human["arm_coordinates"]) == 336
    assert len(human["comparison_cells"]) == 240
    items = executor._item_map(plan)
    block = plan["admission_blocks"][0]
    exact = study._exact_sum(
        [Decimal(items[item_id]["worst_case_reserve_usd"]) for item_id in block["work_item_ids"]]
    )
    assert exact == Decimal(block["worst_case_reserve_usd"])
    assert exact == Decimal(plan["budget"]["first_block_worst_case_usd"])
    assert exact == Decimal(bound["checks"]["first_block_reserve_usd"])
    assert study._exact_add(study.CURRENT_EXPOSURE_USD, exact) == Decimal(
        bound["checks"]["first_block_projected_usd"]
    )
    assert block["task_families"] == list(study.TASK_FAMILIES)


def test_all_v6_id_namespaces_are_disjoint_from_v2_through_v5(
    plan: dict[str, Any], human: dict[str, Any]
) -> None:
    current = study._identity_sets(plan)
    for relative, semantic, physical in (
        (study.V5_PLAN_PATH, study.V5_PLAN_SHA256, study.V5_PLAN_FILE_SHA256),
        (study.v5.V4_PLAN_PATH, study.v5.V4_PLAN_SHA256, None),
        (study.v5.V3_PLAN_PATH, study.v5.V3_PLAN_SHA256, None),
        (study.v5.V2_PLAN_PATH, study.v5.V2_PLAN_SHA256, None),
    ):
        historical = study._identity_sets(
            study._verified_artifact(REPO_ROOT, relative, semantic, physical)
        )
        assert all(current[key].isdisjoint(historical[key]) for key in current)
    v5_human = _json(REPO_ROOT / study.V5_HUMAN_PATH)
    assert {row["presentation_id"] for row in human["presentations"]}.isdisjoint(
        {row["presentation_id"] for row in v5_human["presentations"]}
    )


def test_heterogeneous_source_index_verifies_all_and_rejects_ambiguity(
    plan: dict[str, Any], tmp_path: Path
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    for source in LIVE_SOURCES.glob("*.json"):
        shutil.copyfile(source, root / source.name)
    current = {item["work_item_id"] for item in plan["work_items"]}
    assert executor._verified_source_index(root, current_work_item_ids=current) == {}
    item = plan["work_items"][0]
    first = _write_current_source(root=root, plan=plan, item=item)
    indexed = executor._verified_source_index(root, current_work_item_ids=current)
    assert indexed[item["work_item_id"]][0] == first

    duplicate = _json(first)
    duplicate["note"] = "same current ID, different content address"
    _rehash_live_artifact(root, duplicate, "duplicate-current")
    with pytest.raises(executor.FullStudyExecutionError, match="current-ID ambiguity"):
        executor._verified_source_index(root, current_work_item_ids=current)

    first.unlink()
    for path in root.glob("duplicate-current-*.json"):
        path.unlink()
    malformed = _historical_template()
    malformed["dataset_work_item_id"] = "not-a-sha256"
    _rehash_live_artifact(root, malformed, "malformed")
    with pytest.raises(executor.FullStudyExecutionError, match="malformed non-null"):
        executor._verified_source_index(root, current_work_item_ids=current)

    for path in root.glob("malformed-*.json"):
        path.unlink()
    victim = next(root.glob("*.json"))
    victim.write_text("{}", encoding="utf-8")
    with pytest.raises(executor.FullStudyExecutionError, match="verification failed"):
        executor._verified_source_index(root, current_work_item_ids=current)


def test_nested_attestation_unwraps_and_hash_checks_raw_contract(plan: dict[str, Any]) -> None:
    prepared = executor.prepare_all_runtime_items(plan=plan, repo_root=REPO_ROOT)
    block = plan["admission_blocks"][0]
    rebound = executor._bind_block_runtime_after_attestation(
        plan=plan,
        block=block,
        attestations=_attestations(),
        prepared_all=prepared,
        repo_root=REPO_ROOT,
        source_root=REPO_ROOT / study.GLOBAL_SOURCE_PATH,
    )
    assert list(rebound) == block["work_item_ids"]
    assert all("pricing" in runtime.raw_execution_contract for runtime in rebound.values())
    assert all(
        "raw_execution_contract" not in runtime.raw_execution_contract
        for runtime in rebound.values()
    )
    forged = _attestations()
    forged[0]["raw_execution_contract_sha256"] = "0" * 64
    with pytest.raises(executor.FullStudyExecutionError, match="hash binding"):
        executor._bind_block_runtime_after_attestation(
            plan=plan,
            block=block,
            attestations=forged,
            prepared_all=prepared,
            repo_root=REPO_ROOT,
            source_root=REPO_ROOT / study.GLOBAL_SOURCE_PATH,
        )


@pytest.mark.asyncio
async def test_full_source_journal_canonical_and_receipt_lifecycle(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan)
    try:
        result = await executor.execute_one_block(
            plan=plan,
            human_protocol=human,
            bound_preflight=bound,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://network-forbidden.invalid",
            api_key="not-used",
            adapters=run.adapters,
        )
        assert result["decision"] == "block_terminal"
        assert len(result["document"]["canonical_lifecycle"]) == 28
        assert len(run.invocations) == 28
        assert max(run.invocations.values()) == 1
        assert all("pricing" in raw for raw in run.raw_contracts_seen)
        assert all("raw_execution_contract" not in raw for raw in run.raw_contracts_seen)
        source_index = executor._verified_source_index(
            run.sources,
            current_work_item_ids={item["work_item_id"] for item in plan["work_items"]},
        )
        assert len(source_index) == 28
        entries = frontier.load_ledger(run.ledger)
        campaign = executor._campaign_global_reservations(plan=plan, entries=entries)
        assert len(campaign) == 28
        assert not (
            {row["entry_sha256"] for row in campaign.values()}
            & set(frontier.active_ledger_reservations(entries))
        )
        assert (
            sum(
                row.get("event_type") == "artifact_recorded"
                and row.get("campaign_id") == study.STUDY_ID
                for row in entries
            )
            == 28
        )
    finally:
        run.close()


@pytest.mark.asyncio
async def test_missing_prebound_dependency_after_existing_start_is_durably_classified(
    plan: dict[str, Any],
) -> None:
    run = _offline_run(plan)
    block = plan["admission_blocks"][0]
    items = executor._item_map(plan)
    try:
        reservations, _ = executor._ensure_canonical_reservations(
            plan=plan,
            block=block,
            repo_root=REPO_ROOT,
            global_ledger=run.ledger,
            source_root=run.sources,
            failure_injector=None,
        )
        canonical = dict(zip(block["work_item_ids"], reservations, strict=True))
        mapping = {
            item_id: canonical[item_id]["entry_sha256"] for item_id in block["work_item_ids"]
        }
        coordinator_ledger = run.coordinator / "ledger.jsonl"
        local_reservation = executor._append_ledger(
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
        first = block["work_item_ids"][0]
        item = items[first]
        endpoint_root = run.endpoints[item["route_coordinate"]["endpoint_id"]]
        endpoint_ledger = endpoint_root / "ledger.jsonl"
        executor._append_ledger(
            endpoint_ledger,
            role="endpoint",
            event={
                "event_type": "item_execution_started",
                "study_plan_sha256": plan["artifact_sha256"],
                "admission_block_id": block["admission_block_id"],
                "task_wave_id": executor._item_wave_id(plan, first),
                "work_item_id": first,
                "run_id": item["run_id"],
                "endpoint_id": item["route_coordinate"]["endpoint_id"],
                "variant_id": item["route_coordinate"]["variant_id"],
                "block_reservation_entry_sha256": local_reservation["entry_sha256"],
                "canonical_reservation_entry_sha256": canonical[first]["entry_sha256"],
                "raw_endpoint_execution_sha256": "b" * 64,
                "replay_permitted": False,
            },
        )
        outcome = await executor._classification_fence(
            plan=plan,
            block=block,
            item=item,
            local_reservation=local_reservation,
            canonical_reservation=canonical[first],
            prepared_block={},
            repo_root=REPO_ROOT,
            source_root=run.sources,
            endpoint_root=endpoint_root,
            endpoint_ledger=endpoint_ledger,
            coordinator_ledger=coordinator_ledger,
            global_ledger=run.ledger,
            adapters=run.adapters,
            failure_injector=None,
        )
        assert outcome["decision"] == "durable_incident_reservation_derived"
        endpoint_incident = executor._endpoint_state(
            executor._load_ledger(endpoint_ledger, role="endpoint")
        )["incidents"][first]
        assert endpoint_incident["error_type"] == "KeyError"
        assert endpoint_incident["canonical_reservation_retained"] is True
        assert run.invocations == {}
    finally:
        run.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cut",
    [
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
    ],
)
async def test_prior_crash_cuts_recover_without_identifier_or_provider_replay(
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
            await executor.execute_one_block(
                plan=plan,
                human_protocol=human,
                bound_preflight=bound,
                governance_go=governance_go,
                repo_root=REPO_ROOT,
                api_base="https://network-forbidden.invalid",
                api_key="not-used",
                failure_injector=crash,
                adapters=run.adapters,
            )
        assert fired
        result = await executor.execute_one_block(
            plan=plan,
            human_protocol=human,
            bound_preflight=bound,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://network-forbidden.invalid",
            api_key="not-used",
            adapters=run.adapters,
        )
        assert result["decision"] == "block_terminal"
        assert max(run.invocations.values(), default=0) == 1
        entries = frontier.load_ledger(run.ledger)
        assert len(executor._campaign_global_reservations(plan=plan, entries=entries)) == 28
    finally:
        run.close()


@pytest.mark.asyncio
async def test_normal_failures_after_global_or_endpoint_terminal_recover_not_incident(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    for cut in (
        "after_global_artifact_finalization",
        "after_endpoint_terminal_before_coordinator_terminal",
    ):
        run = _offline_run(plan)
        first = plan["admission_blocks"][0]["work_item_ids"][0]
        fired = False

        def fail(
            stage: str,
            item_id: str | None,
            expected_cut: str = cut,
            expected_first: str = first,
        ) -> None:
            nonlocal fired
            if not fired and stage == expected_cut and item_id == expected_first:
                fired = True
                raise RuntimeError(expected_cut)

        try:
            result = await executor.execute_one_block(
                plan=plan,
                human_protocol=human,
                bound_preflight=bound,
                governance_go=governance_go,
                repo_root=REPO_ROOT,
                api_base="https://network-forbidden.invalid",
                api_key="not-used",
                failure_injector=fail,
                adapters=run.adapters,
            )
            assert fired
            assert result["decision"] == "block_terminal"
            coordinator_state = executor._coordinator_state(
                plan,
                executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
            )
            assert coordinator_state["incidents"] == {}
            first_item = executor._item_map(plan)[first]
            endpoint_state = executor._endpoint_state(
                executor._load_ledger(
                    run.endpoints[first_item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl",
                    role="endpoint",
                )
            )
            assert first in endpoint_state["terminals"]
            assert endpoint_state["incidents"] == {}
        finally:
            run.close()


@pytest.mark.asyncio
async def test_missing_source_retains_reserve_and_exact_endpoint_incident_replays(
    plan: dict[str, Any],
    human: dict[str, Any],
    bound: dict[str, Any],
    governance_go: dict[str, Any],
) -> None:
    run = _offline_run(plan, produce_source=False)
    first = plan["admission_blocks"][0]["work_item_ids"][0]
    fired = False

    def crash(stage: str, item_id: str | None) -> None:
        nonlocal fired
        if stage == "after_endpoint_incident_before_coordinator_incident" and item_id == first:
            fired = True
            raise executor.SimulatedCrash(stage)

    try:
        with pytest.raises(executor.SimulatedCrash):
            await executor.execute_one_block(
                plan=plan,
                human_protocol=human,
                bound_preflight=bound,
                governance_go=governance_go,
                repo_root=REPO_ROOT,
                api_base="https://network-forbidden.invalid",
                api_key="not-used",
                failure_injector=crash,
                adapters=run.adapters,
            )
        assert fired
        first_item = executor._item_map(plan)[first]
        endpoint_ledger = (
            run.endpoints[first_item["route_coordinate"]["endpoint_id"]] / "ledger.jsonl"
        )
        endpoint_incident = executor._endpoint_state(
            executor._load_ledger(endpoint_ledger, role="endpoint")
        )["incidents"][first]
        result = await executor.execute_one_block(
            plan=plan,
            human_protocol=human,
            bound_preflight=bound,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://network-forbidden.invalid",
            api_key="not-used",
            adapters=run.adapters,
        )
        assert result["decision"] == "durable_incident_stop"
        state = executor._coordinator_state(
            plan,
            executor._load_ledger(run.coordinator / "ledger.jsonl", role="coordinator"),
        )
        coordinator_incident = state["incidents"][state["active_block_id"]][0]
        for key, value in endpoint_incident.items():
            if key not in executor._LEDGER_PROTECTED:
                assert coordinator_incident[key] == value
        assert (
            coordinator_incident["endpoint_incident_entry_sha256"]
            == (endpoint_incident["entry_sha256"])
        )
        assert coordinator_incident["canonical_reservation_retained"] is True
        assert (
            coordinator_incident["work_item_reserve_retained_usd"]
            == (first_item["worst_case_reserve_usd"])
        )
        active = frontier.active_ledger_reservations(frontier.load_ledger(run.ledger))
        assert len(active) == 28
    finally:
        run.close()


@pytest.mark.asyncio
async def test_terminal_block_missing_receipt_recovers_before_attest_or_next_selection(
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
            await executor.execute_one_block(
                plan=plan,
                human_protocol=human,
                bound_preflight=bound,
                governance_go=governance_go,
                repo_root=REPO_ROOT,
                api_base="https://network-forbidden.invalid",
                api_key="not-used",
                failure_injector=crash,
                adapters=run.adapters,
            )
        assert len(run.attest_calls) == 1
        assert not (run.coordinator / "receipts").exists()
        result = await executor.execute_one_block(
            plan=plan,
            human_protocol=human,
            bound_preflight=bound,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://network-forbidden.invalid",
            api_key="not-used",
            adapters=run.adapters,
        )
        assert result["decision"] == "block_terminal_receipt_recovered"
        assert len(run.attest_calls) == 1
        before = Path(result["receipt_path"]).read_bytes()
        repeated = await executor.execute_one_block(
            plan=plan,
            human_protocol=human,
            bound_preflight=bound,
            governance_go=governance_go,
            repo_root=REPO_ROOT,
            api_base="https://network-forbidden.invalid",
            api_key="not-used",
            adapters=run.adapters,
        )
        assert Path(repeated["receipt_path"]).read_bytes() == before
        assert len(run.attest_calls) == 1
    finally:
        run.close()


def test_double_freeze_byte_determinism_and_no_live_side_effects(tmp_path: Path) -> None:
    output = ROOT / ".pytest_cache" / f"v6-freeze-{tmp_path.name}"
    shutil.rmtree(output, ignore_errors=True)
    ledger_before = LIVE_LEDGER.read_bytes()
    source_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in LIVE_SOURCES.glob("*")
        if path.is_file()
    }
    v5_paths = [
        REPO_ROOT / study.V5_PLAN_PATH,
        REPO_ROOT / study.V5_HUMAN_PATH,
        REPO_ROOT / study.V5_PREFLIGHT_PATH,
        REPO_ROOT / study.V5_BOUND_PATH,
        REPO_ROOT / study.V5_NO_GO_PATH,
    ]
    v5_before = {path: path.read_bytes() for path in v5_paths}
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
    assert {path: path.read_bytes() for path in v5_paths} == v5_before


def test_live_global_anchor_and_all_v1_v5_modules_are_unchanged() -> None:
    assert hashlib.sha256(LIVE_LEDGER.read_bytes()).hexdigest() == (
        study.GLOBAL_LEDGER_ANCHOR_FILE_SHA256
    )
    expected = {
        "reasoning_effort_full_study_v5.py": (
            "5f8db491598d08a54f9b0c0f31bc8c3a334b6e8a30f3ebe1bde0c1855ed1c1e6"
        ),
        "reasoning_effort_full_study_executor_v5.py": (
            "4040047771fb0fe23af32c2a89b5a5f7b110e4b6fde190d4d1bbfd43d318b64d"
        ),
        "reasoning_effort_source_closure_v5.py": (
            "48f209205d04a4a543c0e9f86d8ee677caa6106b18efef4eec623181451ca10e"
        ),
    }
    for filename, digest in expected.items():
        path = ROOT / "src/flavourbench" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
