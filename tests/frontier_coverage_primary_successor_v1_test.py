from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import flavourbench.frontier_coverage_primary_executor_v1 as executor
import flavourbench.frontier_coverage_primary_successor_v1 as successor
from flavourbench.frontier_contract_runner import (
    AdmissionDenied,
    IntegrityError,
    _live_smoke_sha256,
)
from flavourbench.run_journal import RunJournal

REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_governed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "flavourbench/src", repo / "flavourbench/src")
    for relative in (
        "flavourbench/Dockerfile",
        "flavourbench/pyproject.toml",
        "flavourbench/requirements.lock",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    governed_inputs = {
        successor.PREDECESSOR,
        successor.PREDECESSOR_PREFLIGHT,
        successor.COHERE_GATE,
        successor.CORRECTED_ARENA,
        successor.TASK_VALIDITY,
        successor.TASK_QUARANTINE,
        *successor.ROUTE_MANIFESTS,
        *successor.HISTORICAL_UNPRICED_COHERE_LEDGERS.values(),
        *(relative for _, relative, _, _, _ in successor.FAILED_V1_ARTIFACTS),
        *(relative for _, relative, _, _, _ in successor.FAILED_V2_ARTIFACTS),
        *(relative for _, relative, _, _, _ in successor.FAILED_V3_ARTIFACTS),
        successor.FAILED_V3_AUDIT,
        successor.RETIRED_V4_RECEIPT,
        *(
            f"{root}/{suffix}"
            for root in (
                successor.RETIRED_V4_ROOT,
                successor.RETIRED_V4_ALTERNATE_ROOT,
            )
            for _, suffix, _, _, _ in successor.RETIRED_V4_ARTIFACTS
        ),
    }
    for relative in sorted(governed_inputs):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    return repo


@pytest.fixture(scope="module")
def frozen_plan() -> dict:
    plan = successor.build_plan(repo_root=REPO_ROOT)
    successor.validate_plan(plan, repo_root=REPO_ROOT)
    return plan


def _batch(plan: dict, backend: str) -> dict:
    return next(row for row in plan["endpoint_batches"] if row["execution_backend"] == backend)


def _cells(plan: dict) -> dict[str, dict]:
    return {row["work_item_id"]: row for row in plan["cells"]}


def _identifiers(value: object, key: str = "") -> set[str]:
    singular = {"batch_id", "cell_id", "work_item_id", "run_id", "arm_id", "attempt_id"}
    plural = {"batch_ids", "cell_ids", "work_item_ids", "run_ids", "arm_ids", "attempt_ids"}
    found: set[str] = set()
    if key in singular and isinstance(value, str):
        found.add(value)
    elif key in plural and isinstance(value, dict):
        found.update(item for item in value.values() if isinstance(item, str))
    elif key in plural and isinstance(value, list):
        found.update(item for item in value if isinstance(item, str))
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(_identifiers(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(_identifiers(child, key))
    return found


def _write_chain(path: Path, rows: list[dict]) -> None:
    previous = None
    rendered = []
    for sequence, row in enumerate(rows, 1):
        entry = {
            "schema_version": row.pop("schema_version"),
            "sequence": sequence,
            "previous_entry_sha256": previous,
            **row,
        }
        entry["entry_sha256"] = executor._ledger_digest(entry)
        previous = entry["entry_sha256"]
        rendered.append(json.dumps(entry, separators=(",", ":"), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def _write_rehashed_successor_chain(path: Path, rows: list[dict]) -> None:
    previous = None
    terminal_digests: dict[str, str] = {}
    reservation_digests: dict[str, str] = {}
    rendered = []
    for sequence, original in enumerate(copy.deepcopy(rows), 1):
        entry = dict(original)
        entry["sequence"] = sequence
        entry["previous_entry_sha256"] = previous
        batch_id = str(entry.get("batch_id") or "")
        if (
            entry.get("event_type") != "endpoint_batch_reserved"
            and "batch_reservation_entry_sha256" in entry
        ):
            entry["batch_reservation_entry_sha256"] = reservation_digests[batch_id]
        if entry.get("event_type") == "endpoint_batch_terminalized":
            entry["item_terminal_entry_sha256s"] = [
                terminal_digests[work_id] for work_id in entry["work_item_ids"]
            ]
        entry["entry_sha256"] = executor._ledger_digest(entry)
        if entry.get("event_type") == "endpoint_batch_reserved":
            reservation_digests[batch_id] = str(entry["entry_sha256"])
        if entry.get("event_type") == "item_terminalized":
            terminal_digests[str(entry["work_item_id"])] = str(entry["entry_sha256"])
        previous = str(entry["entry_sha256"])
        rendered.append(json.dumps(entry, separators=(",", ":"), sort_keys=True))
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def _priced_allowances(plan: dict, batch: dict) -> dict[str, str]:
    cells = _cells(plan)
    return {
        work_id: cells[work_id]["cost_reservation"]["successor_reservation_usd"]
        for work_id in batch["work_item_ids"]
    }


def _priced_reservation_payload(plan: dict, batch: dict) -> dict:
    allowances = _priced_allowances(plan, batch)
    return {
        "event_type": "endpoint_batch_reserved",
        "plan_sha256": plan["artifact_sha256"],
        "preflight_sha256": "1" * 64,
        "live_admission_sha256": "2" * 64,
        "batch_id": batch["batch_id"],
        "work_item_ids": batch["work_item_ids"],
        "reserved_usd": executor._decimal_text(
            sum((Decimal(value) for value in allowances.values()), Decimal(0))
        ),
        "cell_allowances_usd": allowances,
        "global_reservation_entry_sha256s": {
            work_id: executor.hashlib.sha256(work_id.encode()).hexdigest()
            for work_id in batch["work_item_ids"]
        },
        "locked_budget_rebase": {"admission_allowed": True},
        "later_source_snapshot": {"source_count": 0},
        "other_local_active_reservations": [],
        "other_canonical_global_active_reservations_usd": "0",
        "reservation_unit": "one_complete_endpoint_isolated_batch",
        "replay_permitted": False,
    }


def _reorder_plan(plan: dict, leading_batch_ids: list[str]) -> dict:
    value = copy.deepcopy(plan)
    remaining = [
        batch_id for batch_id in value["batch_execution_order"] if batch_id not in leading_batch_ids
    ]
    value["batch_execution_order"] = [*leading_batch_ids, *remaining]
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = successor._sha256(body)
    return value


def _operator_attestation(plan: dict) -> dict:
    envelope = plan["cohere_prospective_resource_envelope"]
    work_ids = successor._ordered_cohere_work_item_ids(plan)
    public = {
        "provider": "cohere_direct",
        "credential_program": "Cohere Scholars",
        "environment_variable_name": "COHERE_API_KEY",
        "credential_handle": "scholars-primary-public-handle",
        "scope": {
            "plan_sha256": plan["artifact_sha256"],
            "resource_envelope_sha256": envelope["envelope_sha256"],
            "work_item_ids_sha256": executor._sha256(work_ids),
            "authorized_use": "frontier_coverage_successor_cohere_direct_only",
        },
    }
    now = datetime.now(UTC)
    payload = {
        "schema_version": executor.COHERE_OPERATOR_ATTESTATION_SCHEMA,
        "status": "operator_authorized_exact_resource_scope",
        "plan_sha256": plan["artifact_sha256"],
        "decision": "authorize_exact_bounded_cohere_scholars_use",
        "credential_program": "Cohere Scholars",
        "provider": "cohere_direct",
        "operator": {"full_name": "Test Operator", "role": "quota owner"},
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "work_item_ids": work_ids,
        "resource_envelope_sha256": envelope["envelope_sha256"],
        "credential_binding_method": "sha256_canonical_public_binding_object",
        "credential_binding_public_object": public,
        "credential_binding_sha256": executor._sha256(public),
        "credential_binding_is_derived_from_secret": False,
        "contains_secret": False,
        "usd_cost_or_reservation_claimed": False,
        "provider_or_epicure_calls_made_by_attestation": False,
    }
    return {**payload, "artifact_sha256": executor._sha256(payload)}


def test_plan_repairs_every_identity_and_keeps_observed_support_honest(
    frozen_plan: dict,
) -> None:
    predecessor = successor._addressed(REPO_ROOT / successor.PREDECESSOR)
    retired = _identifiers(predecessor)
    new_ids: set[str] = set()
    for cell in frozen_plan["cells"]:
        arm_id = f"{cell['run_id']}:epicure_on"
        assert cell["arm_ids"] == {"epicure_on": arm_id}
        assert {slot["arm_id"] for slot in cell["attempt_slots"]} == {arm_id}
        assert len(cell["attempt_slots"]) == 29
        new_ids.update(
            {
                cell["cell_id"],
                cell["work_item_id"],
                cell["run_id"],
                arm_id,
                *(slot["attempt_id"] for slot in cell["attempt_slots"]),
            }
        )
    new_ids.update(batch["batch_id"] for batch in frozen_plan["endpoint_batches"])
    assert new_ids.isdisjoint(retired)
    assert frozen_plan["support"]["observed_supported_model_pair_family_cells"] == 407
    assert frozen_plan["support"]["observed_empty_model_pair_family_cells"] == 73
    assert frozen_plan["support"]["projection_is_observed"] is False
    assert frozen_plan["counts"]["synthetic_arms"] == 0


def test_offline_preflight_and_template_are_zero_call_and_secret_free(
    frozen_plan: dict,
) -> None:
    preflight = successor.build_preflight(plan=frozen_plan, repo_root=REPO_ROOT)
    dry_run = successor.build_dry_run(plan=frozen_plan, preflight=preflight, repo_root=REPO_ROOT)
    template = successor.build_cohere_operator_attestation_template(plan=frozen_plan)
    assert preflight["calls_made"] == {
        "provider_completions": 0,
        "catalog_gets": 0,
        "epicure": 0,
    }
    assert dry_run["counts"]["provider_completions"] == 0
    assert dry_run["counts"]["epicure_calls"] == 0
    assert template["status"] == "template_not_authorization"
    assert template["contains_secret"] is False
    assert template["resource_envelope_totals"]["max_reasoning_tokens"] == 327_680
    assert template["credential_binding_rule"]["must_not_hash_or_fingerprint_credential_material"]
    assert successor._contains_secret_material(template) is False


def test_exact_priced_and_non_usd_admission_envelopes(frozen_plan: dict) -> None:
    cells = _cells(frozen_plan)
    priced = _batch(frozen_plan, "openrouter")
    allowances = {
        work_id: cells[work_id]["cost_reservation"]["successor_reservation_usd"]
        for work_id in priced["work_item_ids"]
    }
    admitted = {
        "work_item_ids": priced["work_item_ids"],
        "reserved_usd": executor._decimal_text(
            sum((Decimal(value) for value in allowances.values()), Decimal(0))
        ),
        "cell_allowances_usd": allowances,
    }
    reserve, parsed, mode = executor._validate_admitted_reserve(
        plan=frozen_plan, batch=priced, admitted=admitted, blocker_evidence={}
    )
    assert mode == "priced_usd_reservation"
    assert reserve == sum(parsed.values(), Decimal(0))
    tiny = copy.deepcopy(admitted)
    tiny["reserved_usd"] = "0.000001"
    with pytest.raises(AdmissionDenied):
        executor._validate_admitted_reserve(
            plan=frozen_plan, batch=priced, admitted=tiny, blocker_evidence={}
        )

    cohere = _batch(frozen_plan, "cohere_direct")
    operator = _operator_attestation(frozen_plan)
    quota = {
        "work_item_ids": cohere["work_item_ids"],
        "reservation_kind": "cohere_scholars_operator_quota",
        "usd_cost_or_reservation_claimed": False,
        "resource_envelope_sha256": frozen_plan["cohere_prospective_resource_envelope"][
            "envelope_sha256"
        ],
        "operator_attestation_sha256": operator["artifact_sha256"],
        "cell_resource_limits": executor._cohere_batch_resource_limits(frozen_plan, cohere),
    }
    reserve, parsed, mode = executor._validate_admitted_reserve(
        plan=frozen_plan,
        batch=cohere,
        admitted=quota,
        blocker_evidence={"_cohere_operator_attestation": operator},
    )
    assert (reserve, mode) == (Decimal(0), "cohere_scholars_operator_quota")
    assert set(parsed.values()) == {Decimal(0)}
    forged = {**quota, "reserved_usd": "0"}
    with pytest.raises(AdmissionDenied):
        executor._validate_admitted_reserve(
            plan=frozen_plan,
            batch=cohere,
            admitted=forged,
            blocker_evidence={"_cohere_operator_attestation": operator},
        )


def test_v4_preserves_non_usd_unknown_and_priced_reservations_exactly(
    frozen_plan: dict,
) -> None:
    cohere_cells = [
        cell for cell in frozen_plan["cells"] if cell["execution_backend"] == "cohere_direct"
    ]
    priced_cells = [
        cell for cell in frozen_plan["cells"] if cell["execution_backend"] != "cohere_direct"
    ]
    assert len(cohere_cells) == 8
    assert len(priced_cells) == 42
    assert {cell["cost_reservation"]["status"] for cell in cohere_cells} == {
        successor.NON_USD_UNKNOWN_STATUS
    }
    assert all(
        cell["cost_reservation"]["successor_reservation_usd"] is None
        and cell["cost_reservation"]["currency"] is None
        and cell["cost_reservation"]["current_usd_price_or_reservation_available"] is False
        and cell["route"]["pricing_status"] == successor.NON_USD_UNKNOWN_STATUS
        and cell["route"]["pricing_currency"] is None
        and set(cell["route"]["pricing"].values()) == {None}
        and "reserved_worst_case_usd" not in cell
        for cell in cohere_cells
    )
    assert all(
        cell["cost_reservation"]["status"] == "priced_frozen_route"
        and Decimal(cell["cost_reservation"]["successor_reservation_usd"]) > 0
        for cell in priced_cells
    )

    cells = _cells(frozen_plan)
    cohere_batches = [
        batch
        for batch in frozen_plan["endpoint_batches"]
        if batch["execution_backend"] == "cohere_direct"
    ]
    priced_batches = [
        batch
        for batch in frozen_plan["endpoint_batches"]
        if batch["execution_backend"] != "cohere_direct"
    ]
    assert len(cohere_batches) == 2
    assert all(
        batch["successor_priced_reserve_usd"] is None
        and batch["unpriced_cell_count"] == batch["cell_count"]
        and batch["complete_reservation_bound"] is False
        for batch in cohere_batches
    )
    assert len(priced_batches) == 14
    for batch in priced_batches:
        exact = sum(
            (
                Decimal(cells[work_id]["cost_reservation"]["successor_reservation_usd"])
                for work_id in batch["work_item_ids"]
            ),
            Decimal(0),
        )
        assert batch["unpriced_cell_count"] == 0
        assert batch["complete_reservation_bound"] is True
        assert Decimal(batch["successor_priced_reserve_usd"]) == exact

    preflight = successor.build_preflight(plan=frozen_plan, repo_root=REPO_ROOT)
    dry_run = successor.build_dry_run(plan=frozen_plan, preflight=preflight, repo_root=REPO_ROOT)
    cohere_decisions = [
        decision
        for decision in dry_run["decisions"]
        if decision["execution_backend"] == "cohere_direct"
    ]
    assert len(cohere_decisions) == 2
    assert {decision["decision"] for decision in cohere_decisions} == {
        "blocked_cohere_resource_envelope_and_operator_attestation"
    }
    blocker = next(
        item
        for item in preflight["blockers"]
        if item["code"] == "cohere_complete_reservation_envelope_missing"
    )
    assert "economic authorization" in blocker["reason"]
    assert "no USD reservation or zero-price claim" in blocker["reason"]


def test_v4_recursive_current_cohere_economics_have_no_zero_or_free_ambiguity(
    frozen_plan: dict,
) -> None:
    assert successor._current_cohere_economic_ambiguities(frozen_plan) == []
    for cell in frozen_plan["cells"]:
        if cell["execution_backend"] != "cohere_direct":
            continue
        assert cell["route"]["historical_source_provenance"][
            "not_current_pricing_budget_or_free_tier_claim"
        ] is True
        assert cell["cost_reservation"]["historical_source_provenance"][
            "not_current_pricing_budget_or_free_tier_claim"
        ] is True

    forged = copy.deepcopy(frozen_plan)
    cohere = next(
        cell for cell in forged["cells"] if cell["execution_backend"] == "cohere_direct"
    )
    cohere["route"] = copy.deepcopy(cohere["route"])
    cohere["route"]["pricing"]["prompt"] = "0"
    findings = successor._current_cohere_economic_ambiguities(forged)
    assert findings == [
        {
            "path": (
                f"cells[{forged['cells'].index(cohere)}].route.pricing.prompt"
            ),
            "reason": "generic_numeric_zero",
        }
    ]
    forged_body = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = successor._sha256(forged_body)
    with pytest.raises(successor.CoverageSuccessorError, match="cost semantics"):
        successor.validate_plan(forged, repo_root=REPO_ROOT)


def test_operator_binding_is_public_only_and_not_a_credential_fingerprint(
    frozen_plan: dict,
) -> None:
    operator = _operator_attestation(frozen_plan)
    executor._validate_cohere_operator_attestation(plan=frozen_plan, operator=operator)
    forged = copy.deepcopy(operator)
    forged["credential_binding_public_object"]["credential_handle"] = (
        "sk-this-is-credential-material-and-not-a-public-handle"
    )
    forged["credential_binding_sha256"] = executor._sha256(
        forged["credential_binding_public_object"]
    )
    body = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = executor._sha256(body)
    with pytest.raises(AdmissionDenied, match="public binding object|credential material"):
        executor._validate_cohere_operator_attestation(plan=frozen_plan, operator=forged)


def test_operator_attestation_schema_is_identical_for_scanner_and_executor(
    frozen_plan: dict, tmp_path: Path
) -> None:
    accepted_schema = "flavourbench-cohere-scholars-operator-attestation-v1"
    rejected_schema = "flavourbench-cohere-scholars-operator-attestation-v2"
    assert successor.COHERE_OPERATOR_ATTESTATION_SCHEMA == accepted_schema
    assert executor.COHERE_OPERATOR_ATTESTATION_SCHEMA == accepted_schema
    assert successor.COHERE_OPERATOR_TEMPLATE_SCHEMA.endswith("template-v4-r1")

    operator = _operator_attestation(frozen_plan)
    executor._validate_cohere_operator_attestation(plan=frozen_plan, operator=operator)
    root = tmp_path / "artifacts"
    output = root / "successor"
    successor._write_artifact(output / "plan", "plan", frozen_plan)
    successor._write_artifact(output / "evidence", "operator", operator)
    successor._prior_identifiers(root, verified_successor_output=output)

    other = copy.deepcopy(operator)
    other["schema_version"] = rejected_schema
    body = {key: value for key, value in other.items() if key != "artifact_sha256"}
    other["artifact_sha256"] = successor._sha256(body)
    with pytest.raises(AdmissionDenied, match="attestation does not verify"):
        executor._validate_cohere_operator_attestation(plan=frozen_plan, operator=other)

    shutil.rmtree(output)
    successor._write_artifact(output / "plan", "plan", frozen_plan)
    successor._write_artifact(output / "evidence", "operator", other)
    with pytest.raises(successor.CoverageSuccessorError, match="unverified or foreign"):
        successor._prior_identifiers(root, verified_successor_output=output)


def _cohere_source(plan: dict, cell: dict) -> dict:
    arm_id = cell["arm_ids"]["epicure_on"]
    planning = next(slot for slot in cell["attempt_slots"] if slot["phase"] == "planning")
    session = next(slot for slot in cell["attempt_slots"] if slot["phase"] == "mcp_session")
    tool = next(slot for slot in cell["attempt_slots"] if slot["phase"] == "mcp_tool_0_0")
    generation_id = "generation-1"
    request_key = "1" * 64
    events = [
        {
            "arm_id": arm_id,
            "attempt_id": planning["attempt_id"],
            "attempt_index": planning["attempt_index"],
            "phase": "cohere_direct_planning",
            "event_type": "request_started",
            "request_key_sha256": request_key,
        },
        {
            "arm_id": arm_id,
            "attempt_id": session["attempt_id"],
            "attempt_index": session["attempt_index"],
            "phase": "mcp_session",
            "event_type": "mcp_session_started",
            "request_key_sha256": "2" * 64,
        },
        {
            "arm_id": arm_id,
            "attempt_id": session["attempt_id"],
            "attempt_index": session["attempt_index"],
            "phase": "mcp_attestation",
            "event_type": "mcp_session_attested",
            "request_key_sha256": "2" * 64,
        },
        {
            "arm_id": arm_id,
            "attempt_id": tool["attempt_id"],
            "attempt_index": tool["attempt_index"],
            "phase": "mcp_tool_0_0",
            "event_type": "mcp_call_started",
            "request_key_sha256": "3" * 64,
        },
        {
            "arm_id": arm_id,
            "attempt_id": tool["attempt_id"],
            "attempt_index": tool["attempt_index"],
            "phase": "mcp_tool_0_0",
            "event_type": "mcp_call_completed",
            "request_key_sha256": "3" * 64,
        },
        {
            "arm_id": arm_id,
            "attempt_id": planning["attempt_id"],
            "attempt_index": planning["attempt_index"],
            "phase": "cohere_direct_planning",
            "event_type": "response_received",
            "request_key_sha256": request_key,
            "generation_id": generation_id,
        },
    ]
    trace = {
        "round_index": 0,
        "name": "find_pairings",
        "arguments": {"ingredient": "tomato"},
        "result": "pairing evidence",
        "result_sha256": executor.hashlib.sha256(b"pairing evidence").hexdigest(),
        "latency_ms": 1,
        "is_error": False,
    }
    result = {
        "actual_model_id": cell["route"]["canonical_model_slug"],
        "actual_provider": cell["route"]["expected_actual_provider"],
        "finish_reason": "stop",
        "answer_markdown": "Use the evidence as one bounded signal.",
        "final_response_mode": "plain_text",
        "tool_trace": [trace],
        "generation_metadata": [
            {
                "accounting_basis": "frozen_rate_card_times_cohere_returned_usage",
                "billing_reconciliation_status": "provider_charge_unavailable",
                "cost_micros": 0,
                "generation_id": generation_id,
                "model": cell["route"]["canonical_model_slug"],
                "provider": "cohere-direct",
                "reasoning_tokens": 10,
                "reconciled": False,
                "tokens_completion": 100,
                "tokens_prompt": 200,
            }
        ],
    }
    journal = {
        "schema_version": "flavourbench-live-run-journal-v1",
        "filename": "flavourbench-live-smoke-journal-" + "4" * 64 + ".jsonl",
        "sha256": "4" * 64,
        "head_entry_sha256": "5" * 64,
        "entry_count": 8,
        "run_id": cell["run_id"],
        "finalized": True,
    }
    return {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": cell["run_id"],
        "status": "complete_rate_card_estimated",
        "execution_backend": "cohere_direct",
        "dataset_work_item_id": cell["work_item_id"],
        "dataset_task_id": cell["task_id"],
        "prompt_sha256": cell["prompt_sha256"],
        "category": cell["task_family"],
        "requested_model_id": cell["model_id"],
        "requested_provider": cell["provider_tag"],
        "requested_conditions": ["epicure_on"],
        "candidate_manifest_sha256": cell["route_manifest_sha256"],
        "endpoint_execution_contract_sha256": cell["endpoint_execution_sha256"],
        "execution_policy_sha256": cell["execution_policy_sha256"],
        "epicure_tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
        "epicure": dict(plan["epicure"]),
        "provider_attempt_events": events,
        "mcp_trace_events": [{"arm_id": arm_id, **trace}],
        "budget": {
            "cap_usd": "0",
            "forecast_worst_case_usd": "0",
            "actual_cost_micros": 0,
            "all_generation_costs_reconciled": False,
            "all_generation_usage_rate_card_accounted": True,
            "accounting_basis": "provider_usage_times_frozen_rate_card",
            "provider_charge_available": False,
            "provider_account_snapshot_before": "endpoint_not_available",
            "provider_account_snapshot_after": "endpoint_not_available",
        },
        "results": {"epicure_on": result},
        "incomplete_generation_metadata": [],
        "run_journal": journal,
    }


def _source_evidence(path: Path, digest: str, artifact: dict) -> executor.RecoveryEvidence:
    journal = artifact["run_journal"]
    return executor.RecoveryEvidence(
        path,
        digest,
        1,
        (
            {
                "filename": journal["filename"],
                "sha256": journal["sha256"],
                "head_entry_sha256": journal["head_entry_sha256"],
                "entry_count": journal["entry_count"],
                "finalized": journal["finalized"],
                "uncertain_attempt_ids": [],
            },
        ),
    )


def _write_real_cohere_source(plan: dict, cell: dict, root: Path) -> Path:
    artifact = _cohere_source(plan, cell)
    metadata = {
        "dataset_work_item_id": cell["work_item_id"],
        "dataset_task_id": cell["task_id"],
        "candidate_manifest_sha256": cell["route_manifest_sha256"],
        "prompt_sha256": cell["prompt_sha256"],
        "epicure_conditions": ["epicure_on"],
    }
    journal = RunJournal.create(root, run_id=cell["run_id"], metadata=metadata)
    for event in artifact["provider_attempt_events"]:
        journal.append("provider_attempt", event)
    for event in artifact["mcp_trace_events"]:
        journal.append("mcp_trace", event)
    artifact["run_journal"] = journal.finalize({"status": "complete"}).payload()
    artifact["official"] = False
    artifact["rank_eligible"] = False
    digest = _live_smoke_sha256(artifact)
    document = {**artifact, "artifact_sha256": digest}
    destination = root / f"flavourbench-live-smoke-{digest[:12]}.json"
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def test_cohere_source_audit_retains_resource_envelope_without_usd_claim(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cell = next(row for row in frozen_plan["cells"] if row["execution_backend"] == "cohere_direct")
    artifact = _cohere_source(frozen_plan, cell)
    digest = "a" * 64
    monkeypatch.setattr(executor, "_verify_live_artifact", lambda path: (artifact, digest))
    evidence = _source_evidence(tmp_path / "source.json", digest, artifact)
    terminal = executor._audit_source(
        plan=frozen_plan, cell=cell, evidence=evidence, cap_usd=Decimal(0)
    )
    assert terminal["disposition"] == "source_usable"
    assert terminal["provider_reported_cost_usd"] == "0"
    assert terminal["actual_cost_usd"] is None
    assert terminal["usd_cost_or_reservation_claimed"] is False
    assert terminal["resource_usage"]["tokens_prompt"] == 200
    assert (
        terminal["retained_resource_envelope"]
        == frozen_plan["cohere_prospective_resource_envelope"]["cell_limits"][cell["work_item_id"]]
    )

    artifact["mcp_trace_events"][0]["arm_id"] = "forged:epicure_on"
    with pytest.raises(executor.CoverageExecutionError, match="unrelated arm"):
        executor._audit_source(plan=frozen_plan, cell=cell, evidence=evidence, cap_usd=Decimal(0))


def test_missing_cohere_usage_terminalizes_as_reliability_failure_with_full_envelope(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cell = next(row for row in frozen_plan["cells"] if row["execution_backend"] == "cohere_direct")
    artifact = _cohere_source(frozen_plan, cell)
    artifact["results"]["epicure_on"]["generation_metadata"] = []
    digest = "b" * 64
    monkeypatch.setattr(executor, "_verify_live_artifact", lambda path: (artifact, digest))
    terminal = executor._audit_source(
        plan=frozen_plan,
        cell=cell,
        evidence=_source_evidence(tmp_path / "source.json", digest, artifact),
        cap_usd=Decimal(0),
    )
    assert terminal["disposition"] == "source_reliability_failure"
    usage = terminal["resource_usage"]
    assert usage["full_envelope_retained"] is True
    assert usage["accounting_status"].startswith("usage_missing_invalid")
    assert terminal["actual_cost_usd"] is None

    artifact = _cohere_source(frozen_plan, cell)
    artifact["results"]["epicure_on"]["generation_metadata"][0]["reasoning_tokens"] = 50_000
    terminal = executor._audit_source(
        plan=frozen_plan,
        cell=cell,
        evidence=_source_evidence(tmp_path / "source.json", digest, artifact),
        cap_usd=Decimal(0),
    )
    assert terminal["disposition"] == "source_reliability_failure"
    assert terminal["resource_usage"]["reasoning_tokens"] == 40_960
    assert terminal["resource_usage"]["full_envelope_retained"] is True


def test_crash_recovery_is_idempotent_and_never_replays_uncertain_delivery(
    frozen_plan: dict, tmp_path: Path
) -> None:
    original_batch = _batch(frozen_plan, "openrouter")
    plan = _reorder_plan(frozen_plan, [original_batch["batch_id"]])
    batch = next(
        row for row in plan["endpoint_batches"] if row["batch_id"] == original_batch["batch_id"]
    )
    cell = _cells(plan)[batch["work_item_ids"][0]]
    ledger = tmp_path / "ledger.jsonl"
    allowances = _priced_allowances(plan, batch)
    reserve = executor._append_ledger(
        ledger,
        _priced_reservation_payload(plan, batch),
    )
    executor._append_ledger(
        ledger,
        {
            "event_type": "item_execution_started",
            "plan_sha256": plan["artifact_sha256"],
            "batch_id": batch["batch_id"],
            "work_item_id": cell["work_item_id"],
            "run_id": cell["run_id"],
            "arm_id": cell["arm_ids"]["epicure_on"],
            "attempt_slots_sha256": cell["attempt_slots_sha256"],
            "batch_reservation_entry_sha256": reserve["entry_sha256"],
            "replay_permitted": False,
        },
    )

    def probe(cell: dict, root: Path) -> executor.RecoveryEvidence:
        del root
        return executor.RecoveryEvidence(
            None,
            None,
            1,
            (
                {
                    "filename": "flavourbench-live-smoke-journal-evidence.jsonl",
                    "sha256": "4" * 64,
                    "head_entry_sha256": "5" * 64,
                    "entry_count": 2,
                    "finalized": False,
                    "uncertain_attempt_ids": [cell["attempt_slots"][0]["attempt_id"]],
                },
            ),
        )

    first = executor._recover_started(
        ledger_path=ledger,
        plan=plan,
        batch=batch,
        cell=cell,
        reservation=reserve,
        endpoint_root=tmp_path,
        cap_usd=Decimal(allowances[cell["work_item_id"]]),
        probe=probe,
    )
    second = executor._recover_started(
        ledger_path=ledger,
        plan=plan,
        batch=batch,
        cell=cell,
        reservation=reserve,
        endpoint_root=tmp_path,
        cap_usd=Decimal(allowances[cell["work_item_id"]]),
        probe=probe,
    )
    state = executor._ledger_state(plan, executor._load_ledger(ledger))
    assert first == second == "uncertain_delivery_no_replay"
    assert len(state["incidents"]) == 1
    assert not state["terminals"]

    incident_index = next(
        index
        for index, entry in enumerate(executor._load_ledger(ledger))
        if entry["event_type"] == "execution_incident"
    )
    for name, field, value in (
        ("incident-enum", "incident", "retry_allowed"),
        ("incident-count", "request_started_count", 0),
        ("incident-descriptors", "journal_descriptors", []),
        ("incident-retention", "reservation_retained", False),
    ):
        forged = executor._load_ledger(ledger)
        forged[incident_index][field] = value
        forged_path = tmp_path / f"{name}.jsonl"
        _write_rehashed_successor_chain(forged_path, forged)
        with pytest.raises(executor.CoverageExecutionError, match="incident"):
            executor._ledger_state(plan, executor._load_ledger(forged_path))


def test_cross_ledger_accounting_separates_usd_reserves_from_unpriced_cohere(
    tmp_path: Path,
) -> None:
    bedrock = tmp_path / "bedrock.jsonl"
    _write_chain(
        bedrock,
        [
            {
                "schema_version": "flavourbench-bedrock-contract-smoke-ledger-v1",
                "event_type": "reservation_created",
                "reservation_id": "held",
                "reservation_micros": 500_000,
            },
            {
                "schema_version": "flavourbench-bedrock-contract-smoke-ledger-v1",
                "event_type": "reservation_held_uncertain",
                "reservation_id": "held",
            },
            {
                "schema_version": "flavourbench-bedrock-contract-smoke-ledger-v1",
                "event_type": "reservation_created",
                "reservation_id": "settled",
                "reservation_micros": 250_000,
            },
            {
                "schema_version": "flavourbench-bedrock-contract-smoke-ledger-v1",
                "event_type": "reservation_settled_rate_card_estimate",
                "reservation_id": "settled",
            },
        ],
    )
    cohere = tmp_path / "cohere.jsonl"
    _write_chain(
        cohere,
        [
            {
                "schema_version": executor.LEDGER_SCHEMA,
                "event_type": "endpoint_batch_reserved",
                "batch_id": "cohere-batch",
                "reservation_kind": "cohere_scholars_operator_quota",
                "usd_cost_or_reservation_claimed": False,
            }
        ],
    )
    active, rows = executor._generic_other_active_reservations([tmp_path], excluded_ledgers=[])
    assert active == Decimal("0.5")
    assert any(row.get("status") == "active_cohere_resource_quota_unpriced_unknown" for row in rows)
    assert all(row.get("reserved_usd") != "0" for row in rows)


def test_rfc3339_comparison_uses_instants_and_rejects_naive_values() -> None:
    assert executor._parse_rfc3339("2026-08-04T12:00:00Z", field="a") == (
        executor._parse_rfc3339("2026-08-04T14:00:00+02:00", field="b")
    )
    with pytest.raises(executor.CoverageExecutionError, match="timezone-aware"):
        executor._parse_rfc3339("2026-08-04T12:00:00", field="naive")


def test_owned_output_schema_fails_on_colliding_extra_key(
    frozen_plan: dict, tmp_path: Path
) -> None:
    output = tmp_path / "successor"
    successor._write_artifact(output / "plan", "plan", frozen_plan)
    preflight = successor.build_preflight(plan=frozen_plan, repo_root=REPO_ROOT, output_root=output)
    forged = {**preflight, "colliding_extra": frozen_plan["cells"][0]["work_item_id"]}
    body = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = successor._sha256(body)
    successor._write_artifact(output / "preflight", "forged", forged)
    with pytest.raises(successor.CoverageSuccessorError, match="unverified or foreign"):
        successor._prior_identifiers(
            tmp_path,
            verified_successor_output=output,
            successor_plan=frozen_plan,
        )


def test_historical_cohere_records_are_closed_unpriced_disclosures(
    frozen_plan: dict,
) -> None:
    disclosure = frozen_plan["budget"]["historical_source_provenance"]
    assert disclosure["status"] == "closed_no_replay_unpriced_unknown_cost_disclosed"
    assert disclosure["usd_exposure_claimed"] is False
    assert disclosure["blocks_priced_openrouter_or_kimi_batches"] is False
    assert {row["work_item_id"] for row in disclosure["records"]} == set(
        successor.HISTORICAL_UNPRICED_COHERE_WORK_IDS
    )
    assert all(row["successor_replay_permitted"] is False for row in disclosure["records"])


def test_prior_identifier_inventory_rejects_symlink_json_and_jsonl(tmp_path: Path) -> None:
    target_json = tmp_path / "target.json"
    target_json.write_text("{}\n", encoding="utf-8")
    (tmp_path / "linked.json").symlink_to(target_json)
    with pytest.raises(successor.CoverageSuccessorError, match="non-regular JSON"):
        successor._prior_identifiers(
            tmp_path,
            verified_successor_output=tmp_path / "successor",
        )
    (tmp_path / "linked.json").unlink()
    target_jsonl = tmp_path / "target.jsonl"
    target_jsonl.write_text("{}\n", encoding="utf-8")
    (tmp_path / "linked.jsonl").symlink_to(target_jsonl)
    with pytest.raises(successor.CoverageSuccessorError, match="non-regular JSONL"):
        successor._prior_identifiers(
            tmp_path,
            verified_successor_output=tmp_path / "successor",
        )


def test_evidence_and_output_roots_reject_symlink_aliases(
    frozen_plan: dict, tmp_path: Path
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    payload = {"schema_version": "test-evidence-v1", "decision": "pass"}
    evidence = {**payload, "artifact_sha256": successor._sha256(payload)}
    evidence_path = real / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    reference = {
        "path": "alias/evidence.json",
        "bytes": evidence_path.stat().st_size,
        "file_sha256": successor._file_sha256(evidence_path),
        "semantic_sha256": evidence["artifact_sha256"],
    }
    with pytest.raises(AdmissionDenied, match="symlink"):
        executor._load_evidence_reference(repo_root=tmp_path, reference=reference, label="test")

    plan = copy.deepcopy(frozen_plan)
    plan["execution_roots"]["coordinator"] = "study/run/coordinator"
    study = tmp_path / "study"
    study.mkdir()
    (tmp_path / "study-alias").symlink_to(study, target_is_directory=True)
    with pytest.raises(AdmissionDenied, match="symlink"):
        executor.execute_one_batch(
            plan=plan,
            preflight={},
            admission={},
            repo_root=tmp_path,
            output_root=tmp_path / "study-alias",
        )


def test_successor_owned_non_ascii_source_requires_and_verifies_journal(
    frozen_plan: dict, tmp_path: Path
) -> None:
    cell = frozen_plan["cells"][0]
    metadata = {
        "dataset_work_item_id": cell["work_item_id"],
        "dataset_task_id": cell["task_id"],
        "candidate_manifest_sha256": cell["route_manifest_sha256"],
        "prompt_sha256": cell["prompt_sha256"],
        "epicure_conditions": ["epicure_on"],
    }
    journal = RunJournal.create(tmp_path, run_id=cell["run_id"], metadata=metadata)
    descriptor = journal.finalize({"status": "complete"}).payload()
    body = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": cell["run_id"],
        "dataset_work_item_id": cell["work_item_id"],
        "dataset_task_id": cell["task_id"],
        "requested_conditions": ["epicure_on"],
        "candidate_manifest_sha256": cell["route_manifest_sha256"],
        "provider_attempt_events": [],
        "mcp_trace_events": [],
        "run_journal": descriptor,
        "official": False,
        "rank_eligible": False,
        "note": "crème brûlée",
    }
    digest = _live_smoke_sha256(body)
    source = {**body, "artifact_sha256": digest}
    source_path = tmp_path / f"flavourbench-live-smoke-{digest[:12]}.json"
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    found = successor._prior_identifiers(
        tmp_path,
        verified_successor_output=tmp_path / "successor",
        successor_plan=frozen_plan,
    )
    assert cell["work_item_id"] not in found

    source.pop("run_journal")
    body = {key: value for key, value in source.items() if key != "artifact_sha256"}
    source["artifact_sha256"] = _live_smoke_sha256(body)
    source_path.unlink()
    missing_path = tmp_path / f"flavourbench-live-smoke-{source['artifact_sha256'][:12]}.json"
    missing_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(successor.CoverageSuccessorError, match="run-journal"):
        successor._prior_identifiers(
            tmp_path,
            verified_successor_output=tmp_path / "successor",
            successor_plan=frozen_plan,
        )


@pytest.mark.parametrize("mutation", ["orphan", "unknown", "count_mismatch"])
def test_source_audit_rejects_forged_provider_lifecycles(
    frozen_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    cell = next(row for row in frozen_plan["cells"] if row["execution_backend"] == "cohere_direct")
    artifact = _cohere_source(frozen_plan, cell)
    evidence = _source_evidence(tmp_path / "source.json", "a" * 64, artifact)
    if mutation == "orphan":
        artifact["provider_attempt_events"] = [
            row
            for row in artifact["provider_attempt_events"]
            if row["event_type"] != "request_started"
        ]
        evidence = executor.RecoveryEvidence(
            evidence.source_path,
            evidence.source_artifact_sha256,
            0,
            evidence.journal_descriptors,
        )
    elif mutation == "unknown":
        artifact["provider_attempt_events"][0]["event_type"] = "provider_magic"
    else:
        evidence = executor.RecoveryEvidence(
            evidence.source_path,
            evidence.source_artifact_sha256,
            2,
            evidence.journal_descriptors,
        )
    monkeypatch.setattr(executor, "_verify_live_artifact", lambda path: (artifact, "a" * 64))
    with pytest.raises(executor.CoverageExecutionError):
        executor._audit_source(plan=frozen_plan, cell=cell, evidence=evidence, cap_usd=Decimal(0))


def test_coordinator_ledger_rejects_extra_keys_and_out_of_order_reservations(
    frozen_plan: dict, tmp_path: Path
) -> None:
    priced_ids = [
        row["batch_id"]
        for row in frozen_plan["endpoint_batches"]
        if row["execution_backend"] != "cohere_direct"
    ][:2]
    plan = _reorder_plan(frozen_plan, priced_ids)
    first_id, second_id = plan["batch_execution_order"][:2]
    batches = {row["batch_id"]: row for row in plan["endpoint_batches"]}
    first = batches[first_id]
    second = batches[second_id]

    ledger = tmp_path / "extra.jsonl"
    executor._append_ledger(ledger, _priced_reservation_payload(plan, first))
    row = json.loads(ledger.read_text(encoding="utf-8"))
    row["forged_extra"] = "accepted-by-hash-only-parser"
    row["entry_sha256"] = executor._ledger_digest(row)
    ledger.write_text(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(executor.CoverageExecutionError, match="exact schema variant"):
        executor._load_ledger(ledger)

    out_of_order = tmp_path / "out-of-order.jsonl"
    executor._append_ledger(out_of_order, _priced_reservation_payload(plan, second))
    with pytest.raises(executor.CoverageExecutionError, match="sequential execution order"):
        executor._ledger_state(plan, executor._load_ledger(out_of_order))


def test_cross_ledger_parser_rejects_unknown_event_with_reserve_field(tmp_path: Path) -> None:
    ledger = tmp_path / "forged.jsonl"
    _write_chain(
        ledger,
        [
            {
                "schema_version": "flavourbench-matched-protocol-preflight-ledger-v1",
                "event_type": "reservation_created",
                "work_item_id": "work",
                "reserved_usd": "1",
            },
            {
                "schema_version": "flavourbench-matched-protocol-preflight-ledger-v1",
                "event_type": "budget_held",
                "work_item_id": "work",
                "reserved_usd": "1",
            },
        ],
    )
    with pytest.raises(executor.CoverageExecutionError, match="unknown event"):
        executor._generic_other_active_reservations([tmp_path], excluded_ledgers=[])


def test_real_repository_ledgers_reconcile_exact_active_exposure(tmp_path: Path) -> None:
    supported = set(executor._HISTORICAL_EVENT_VOCABULARIES)
    copied = 0
    for source_root in (REPO_ROOT / "flavourbench/artifacts", REPO_ROOT / "artifacts"):
        for source in source_root.rglob("*.jsonl"):
            lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
            if not lines:
                continue
            first = json.loads(lines[0])
            if first.get("schema_version") not in supported:
                continue
            relative = source.relative_to(REPO_ROOT)
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    assert copied > 7
    active, rows = executor._generic_other_active_reservations([tmp_path], excluded_ledgers=[])
    assert active == Decimal("23.63161574666666666666666668")
    assert len(rows) == 38
    disclosures = [row for row in rows if row.get("reserved_usd") is None]
    assert len(disclosures) == 3
    assert {row["work_item_id"] for row in disclosures} == set(
        successor.HISTORICAL_UNPRICED_COHERE_WORK_IDS
    )

    canonical_entries, canonical_active = executor._canonical_global_state(
        repo_root=REPO_ROOT,
        ledger_path=REPO_ROOT / "flavourbench/artifacts/frontier-contract/ledger.jsonl",
        source_root=REPO_ROOT / "flavourbench/artifacts/live-smoke",
    )
    assert len(canonical_entries) == 29
    assert canonical_active == {}


def _execution_plan(frozen_plan: dict, batch: dict) -> dict:
    plan = _reorder_plan(frozen_plan, [batch["batch_id"]])
    plan["execution_roots"] = {
        **plan["execution_roots"],
        "coordinator": "study/run/coordinator",
        "canonical_global_reservation_ledger": "global/ledger.jsonl",
        "canonical_global_source": "global/source",
    }
    body = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    plan["artifact_sha256"] = successor._sha256(body)
    return plan


def _minimal_preflight(plan: dict) -> dict:
    payload = {
        "schema_version": successor.PREFLIGHT_SCHEMA,
        "plan_sha256": plan["artifact_sha256"],
        "decision": "execution_not_admitted",
    }
    return {**payload, "artifact_sha256": successor._sha256(payload)}


def _live_admission(plan: dict, preflight: dict, batch: dict) -> tuple[dict, dict | None]:
    if batch["execution_backend"] == "cohere_direct":
        operator = _operator_attestation(plan)
        admitted = {
            "work_item_ids": batch["work_item_ids"],
            "reservation_kind": "cohere_scholars_operator_quota",
            "usd_cost_or_reservation_claimed": False,
            "resource_envelope_sha256": plan["cohere_prospective_resource_envelope"][
                "envelope_sha256"
            ],
            "operator_attestation_sha256": operator["artifact_sha256"],
            "cell_resource_limits": executor._cohere_batch_resource_limits(plan, batch),
        }
    else:
        operator = None
        allowances = _priced_allowances(plan, batch)
        admitted = {
            "work_item_ids": batch["work_item_ids"],
            "reserved_usd": executor._decimal_text(
                sum((Decimal(value) for value in allowances.values()), Decimal(0))
            ),
            "cell_allowances_usd": allowances,
        }
    payload = {
        "schema_version": executor.LIVE_ADMISSION_SCHEMA,
        "plan_sha256": plan["artifact_sha256"],
        "preflight_sha256": preflight["artifact_sha256"],
        "authorized_batch_id": batch["batch_id"],
        "development_only": True,
        "official": False,
        "rank_eligible": False,
        "blocker_closures": {},
        "batch_reservations": {batch["batch_id"]: admitted},
        "provider_or_epicure_calls_made_by_admission": False,
    }
    return {**payload, "artifact_sha256": successor._sha256(payload)}, operator


@pytest.mark.parametrize("backend", ["openrouter", "cohere_direct"])
def test_execute_one_batch_mocked_end_to_end_is_idempotent_and_receipted(
    frozen_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
) -> None:
    original = _batch(frozen_plan, backend)
    plan = _execution_plan(frozen_plan, original)
    batch = next(row for row in plan["endpoint_batches"] if row["batch_id"] == original["batch_id"])
    preflight = _minimal_preflight(plan)
    admission, operator = _live_admission(plan, preflight, batch)
    monkeypatch.setattr(executor, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "verify_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        executor,
        "_validate_blocker_closures",
        lambda **kwargs: {"_cohere_operator_attestation": operator} if operator else {},
    )
    runner_calls: list[str] = []

    def runner(cell: dict, root: Path, cap: Decimal) -> None:
        del root, cap
        runner_calls.append(cell["work_item_id"])

    def zero_call_probe(cell: dict, root: Path) -> executor.RecoveryEvidence:
        del cell, root
        return executor.RecoveryEvidence(None, None, 0, ())

    output = tmp_path / "study"
    first = executor.execute_one_batch(
        plan=plan,
        preflight=preflight,
        admission=admission,
        repo_root=tmp_path,
        output_root=output,
        runner=runner,
        probe=zero_call_probe,
    )
    assert first["decision"] == "one_endpoint_batch_terminal"
    assert runner_calls == batch["work_item_ids"]
    receipt_path = Path(first["receipt_path"])
    original_receipt_bytes = receipt_path.read_bytes()
    original_ledger_bytes = (tmp_path / "study/run/coordinator/ledger.jsonl").read_bytes()

    second = executor.execute_one_batch(
        plan=plan,
        preflight=preflight,
        admission=admission,
        repo_root=tmp_path,
        output_root=output,
        runner=runner,
        probe=zero_call_probe,
    )
    assert second["decision"] == "authorized_batch_already_terminal"
    assert runner_calls == batch["work_item_ids"]
    assert receipt_path.read_bytes() == original_receipt_bytes
    assert (tmp_path / "study/run/coordinator/ledger.jsonl").read_bytes() == original_ledger_bytes

    other_preflight_body = {
        "schema_version": successor.PREFLIGHT_SCHEMA,
        "plan_sha256": plan["artifact_sha256"],
        "decision": "execution_not_admitted",
        "independent_revision": "different-content-addressed-preflight",
    }
    other_preflight = {
        **other_preflight_body,
        "artifact_sha256": successor._sha256(other_preflight_body),
    }
    other_admission = copy.deepcopy(admission)
    other_admission["preflight_sha256"] = other_preflight["artifact_sha256"]
    other_admission_body = {
        key: value for key, value in other_admission.items() if key != "artifact_sha256"
    }
    other_admission["artifact_sha256"] = successor._sha256(other_admission_body)
    with pytest.raises(executor.CoverageExecutionError, match="not bound"):
        executor.execute_one_batch(
            plan=plan,
            preflight=other_preflight,
            admission=other_admission,
            repo_root=tmp_path,
            output_root=output,
            runner=runner,
            probe=zero_call_probe,
        )
    assert runner_calls == batch["work_item_ids"]

    receipt_path.unlink()
    with pytest.raises(executor.CoverageExecutionError, match="not bound"):
        executor.execute_one_batch(
            plan=plan,
            preflight=other_preflight,
            admission=other_admission,
            repo_root=tmp_path,
            output_root=output,
            runner=runner,
            probe=zero_call_probe,
        )
    assert not receipt_path.exists()
    assert runner_calls == batch["work_item_ids"]
    recovered = executor.execute_one_batch(
        plan=plan,
        preflight=preflight,
        admission=admission,
        repo_root=tmp_path,
        output_root=output,
        runner=runner,
        probe=zero_call_probe,
    )
    assert recovered["decision"] == "recovered_terminal_receipt_before_new_batch"
    recovered_path = Path(recovered["receipt_path"])
    assert recovered_path.read_bytes() == original_receipt_bytes
    assert runner_calls == batch["work_item_ids"]

    canonical = tmp_path / "global/ledger.jsonl"
    if backend == "cohere_direct":
        assert not canonical.exists()
    else:
        assert canonical.is_file()

    forged = json.loads(recovered_path.read_text(encoding="utf-8"))
    forged["outcomes"][0]["disposition"] = "forged"
    forged_body = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = successor._sha256(forged_body)
    forged_path = recovered_path.parent / (
        f"frontier-coverage-primary-successor-receipt-{forged['artifact_sha256']}.json"
    )
    recovered_path.unlink()
    forged_path.write_text(json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(executor.CoverageExecutionError, match="outcomes are invalid"):
        executor._verified_receipts(
            forged_path.parent,
            plan=plan,
            ledger_path=tmp_path / "study/run/coordinator/ledger.jsonl",
            repo_root=tmp_path,
        )


def test_restart_rejects_globally_finalized_reservation_before_runner(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = _batch(frozen_plan, "openrouter")
    plan = _execution_plan(frozen_plan, original)
    batch = next(row for row in plan["endpoint_batches"] if row["batch_id"] == original["batch_id"])
    preflight = _minimal_preflight(plan)
    admission, _ = _live_admission(plan, preflight, batch)
    coordinator = tmp_path / "study/run/coordinator/ledger.jsonl"
    local_payload = _priced_reservation_payload(plan, batch)
    local_payload["preflight_sha256"] = preflight["artifact_sha256"]
    local_payload["live_admission_sha256"] = admission["artifact_sha256"]
    local = executor._append_ledger(coordinator, local_payload)
    cells = _cells(plan)
    global_entries = []
    global_active: dict[str, Decimal] = {}
    for work_id, digest in local["global_reservation_entry_sha256s"].items():
        cell = cells[work_id]
        entry = {
            "event_type": "reservation_created",
            "entry_sha256": digest,
            "coverage_successor_work_item_id": work_id,
            "coverage_successor_batch_id": batch["batch_id"],
            "coverage_successor_plan_sha256": plan["artifact_sha256"],
            "manifest_sha256": cell["route_manifest_sha256"],
            "model_id": cell["model_id"],
            "provider_tag": cell["provider_tag"],
            "endpoint_sha256": cell["route"]["endpoint_document_sha256"],
            "reserved_usd": local["cell_allowances_usd"][work_id],
        }
        global_entries.append(entry)
        global_active[digest] = Decimal(entry["reserved_usd"])
    global_active.pop(next(iter(local["global_reservation_entry_sha256s"].values())))
    monkeypatch.setattr(executor, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "verify_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "_validate_blocker_closures", lambda **kwargs: {})
    monkeypatch.setattr(
        executor,
        "_canonical_global_state",
        lambda **kwargs: (global_entries, global_active),
    )
    calls: list[str] = []
    with pytest.raises(executor.CoverageExecutionError, match="finalized before"):
        executor.execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=tmp_path,
            output_root=tmp_path / "study",
            runner=lambda cell, root, cap: calls.append(cell["work_item_id"]),
        )
    assert calls == []


def test_completed_batches_rebind_canonical_state_and_cohere_attestation(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(executor, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "verify_preflight", lambda *args, **kwargs: None)

    def complete_zero_call(
        backend: str, root: Path
    ) -> tuple[dict, dict, dict, dict | None, list[str]]:
        original = _batch(frozen_plan, backend)
        plan = _execution_plan(frozen_plan, original)
        batch = next(
            row for row in plan["endpoint_batches"] if row["batch_id"] == original["batch_id"]
        )
        preflight = _minimal_preflight(plan)
        admission, operator = _live_admission(plan, preflight, batch)
        monkeypatch.setattr(
            executor,
            "_validate_blocker_closures",
            lambda **kwargs: {"_cohere_operator_attestation": operator} if operator else {},
        )
        calls: list[str] = []
        executor.execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=root,
            output_root=root / "study",
            runner=lambda cell, source_root, cap: calls.append(cell["work_item_id"]),
            probe=lambda cell, source_root: executor.RecoveryEvidence(None, None, 0, ()),
        )
        return plan, preflight, admission, operator, calls

    priced_root = tmp_path / "priced-completed"
    priced_plan, priced_preflight, priced_admission, _, priced_calls = complete_zero_call(
        "openrouter", priced_root
    )
    priced_ledger = priced_root / "study/run/coordinator/ledger.jsonl"
    priced_entries = executor._load_ledger(priced_ledger)
    item_index = next(
        index
        for index, entry in enumerate(priced_entries)
        if entry["event_type"] == "item_terminalized"
    )
    item = priced_entries[item_index]
    work_id = str(item["work_item_id"])
    for key in ("journal_descriptors", "reliability_eligible"):
        item.pop(key)
    item.update(
        {
            "disposition": "source_usable",
            "source_artifact_sha256": "a" * 64,
            "source_filename": "fabricated-source.json",
            "tool_calls": 1,
            "successful_tool_calls": 1,
            "route_policy_epicure_hashes_verified": True,
            "request_started_count": 1,
        }
    )
    reserve = next(
        entry for entry in priced_entries if entry["event_type"] == "endpoint_batch_reserved"
    )
    batch_terminal = next(
        entry for entry in priced_entries if entry["event_type"] == "endpoint_batch_terminalized"
    )
    target_global = reserve["global_reservation_entry_sha256s"][work_id]
    batch_terminal["canonical_global_reservations_retained"] = [
        digest
        for digest in batch_terminal["canonical_global_reservations_retained"]
        if digest != target_global
    ]
    retained = sum(
        (
            Decimal(reserve["cell_allowances_usd"][candidate])
            for candidate in reserve["work_item_ids"]
            if candidate != work_id
        ),
        Decimal(0),
    )
    batch_terminal["canonical_global_retained_usd"] = executor._decimal_text(retained)
    batch_terminal["conservative_exposure_usd"] = executor._decimal_text(retained)
    batch_terminal["whole_batch_reservation_released"] = retained == 0
    _write_rehashed_successor_chain(priced_ledger, priced_entries)
    executor._ledger_state(priced_plan, executor._load_ledger(priced_ledger))
    with pytest.raises(executor.CoverageExecutionError, match="canonical finalization"):
        executor.execute_one_batch(
            plan=priced_plan,
            preflight=priced_preflight,
            admission=priced_admission,
            repo_root=priced_root,
            output_root=priced_root / "study",
            runner=lambda cell, source_root, cap: priced_calls.append(cell["work_item_id"]),
        )
    assert len(priced_calls) == len(_batch(priced_plan, "openrouter")["work_item_ids"])

    canonical_ledger = priced_root / "global/ledger.jsonl"
    target_cell = _cells(priced_plan)[work_id]
    executor.append_frontier_ledger_event(
        canonical_ledger,
        {
            "event_type": "artifact_recorded",
            "runner_run_id": priced_plan["artifact_sha256"],
            "reservation_entry_sha256": target_global,
            "manifest_sha256": target_cell["route_manifest_sha256"],
            "model_id": target_cell["model_id"],
            "provider_tag": target_cell["provider_tag"],
            "artifact_filename": item["source_filename"],
            "artifact_sha256": item["source_artifact_sha256"],
            "artifact_status": "complete",
            "artifact_exposure_usd": "1",
            "postflight_issues": [],
            "subprocess_returncode": None,
            "coverage_successor_plan_sha256": priced_plan["artifact_sha256"],
            "coverage_successor_batch_id": reserve["batch_id"],
            "coverage_successor_work_item_id": work_id,
        },
    )
    canonical_entries = executor.load_frontier_ledger(canonical_ledger)
    canonical_active = executor.active_ledger_reservations(canonical_entries)
    priced_state = executor._ledger_state(priced_plan, executor._load_ledger(priced_ledger))
    with pytest.raises(executor.CoverageExecutionError, match="canonical finalization"):
        executor._validate_completed_priced_global_state(
            plan=priced_plan,
            batch=_batch(priced_plan, "openrouter"),
            reservation=priced_state["reservations"][reserve["batch_id"]],
            state=priced_state,
            global_entries=canonical_entries,
            global_active=canonical_active,
        )

    cohere_root = tmp_path / "cohere-completed"
    cohere_plan, cohere_preflight, cohere_admission, _, cohere_calls = complete_zero_call(
        "cohere_direct", cohere_root
    )
    cohere_receipt = next((cohere_root / "study/run/receipts").glob("*.json"))
    cohere_receipt.unlink()
    cohere_ledger = cohere_root / "study/run/coordinator/ledger.jsonl"
    cohere_entries = executor._load_ledger(cohere_ledger)
    cohere_reserve = next(
        entry for entry in cohere_entries if entry["event_type"] == "endpoint_batch_reserved"
    )
    cohere_reserve["operator_attestation_sha256"] = "9" * 64
    _write_rehashed_successor_chain(cohere_ledger, cohere_entries)
    executor._ledger_state(cohere_plan, executor._load_ledger(cohere_ledger))
    with pytest.raises(
        executor.CoverageExecutionError, match="completed Cohere reservation differs"
    ):
        executor.execute_one_batch(
            plan=cohere_plan,
            preflight=cohere_preflight,
            admission=cohere_admission,
            repo_root=cohere_root,
            output_root=cohere_root / "study",
            runner=lambda cell, source_root, cap: cohere_calls.append(cell["work_item_id"]),
        )
    assert len(cohere_calls) == len(_batch(cohere_plan, "cohere_direct")["work_item_ids"])


def test_ledger_semantics_reject_rehashed_cost_disposition_and_total_forgeries(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(executor, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "verify_preflight", lambda *args, **kwargs: None)

    def execute_fixture(backend: str, root: Path) -> tuple[dict, dict, list[dict]]:
        original = _batch(frozen_plan, backend)
        plan = _execution_plan(frozen_plan, original)
        batch = next(
            row for row in plan["endpoint_batches"] if row["batch_id"] == original["batch_id"]
        )
        preflight = _minimal_preflight(plan)
        admission, operator = _live_admission(plan, preflight, batch)
        monkeypatch.setattr(
            executor,
            "_validate_blocker_closures",
            lambda **kwargs: {"_cohere_operator_attestation": operator} if operator else {},
        )
        executor.execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=root,
            output_root=root / "study",
            runner=lambda cell, source_root, cap: None,
            probe=lambda cell, source_root: executor.RecoveryEvidence(None, None, 0, ()),
        )
        ledger = root / "study/run/coordinator/ledger.jsonl"
        entries = executor._load_ledger(ledger)
        executor._ledger_state(plan, entries)
        return plan, batch, entries

    priced_plan, priced_batch, priced_entries = execute_fixture("openrouter", tmp_path / "priced")
    priced_terminal_index = next(
        index
        for index, entry in enumerate(priced_entries)
        if entry["event_type"] == "item_terminalized"
    )
    priced_work_id = str(priced_entries[priced_terminal_index]["work_item_id"])
    allowance = Decimal(_priced_allowances(priced_plan, priced_batch)[priced_work_id])
    priced_mutations: list[tuple[str, list[dict]]] = []

    forged_disposition = copy.deepcopy(priced_entries)
    forged_disposition[priced_terminal_index]["disposition"] = "fabricated_success"
    priced_mutations.append(("forged-disposition", forged_disposition))

    negative = copy.deepcopy(priced_entries)
    negative[priced_terminal_index]["actual_cost_usd"] = "-1"
    negative[priced_terminal_index]["provider_reported_cost_usd"] = "-1"
    priced_mutations.append(("negative-cost", negative))

    over_cap = copy.deepcopy(priced_entries)
    over_value = executor._decimal_text(allowance + Decimal(1))
    over_cap[priced_terminal_index]["actual_cost_usd"] = over_value
    over_cap[priced_terminal_index]["provider_reported_cost_usd"] = over_value
    priced_mutations.append(("over-cap", over_cap))

    mismatched_total = copy.deepcopy(priced_entries)
    priced_batch_terminal_index = next(
        index
        for index, entry in enumerate(mismatched_total)
        if entry["event_type"] == "endpoint_batch_terminalized"
    )
    mismatched_total[priced_batch_terminal_index]["actual_cost_usd"] = "1"
    priced_mutations.append(("batch-total", mismatched_total))

    for name, rows in priced_mutations:
        path = tmp_path / f"{name}.jsonl"
        _write_rehashed_successor_chain(path, rows)
        with pytest.raises(executor.CoverageExecutionError):
            executor._ledger_state(priced_plan, executor._load_ledger(path))

    cohere_plan, _, cohere_entries = execute_fixture("cohere_direct", tmp_path / "cohere")
    cohere_terminal_index = next(
        index
        for index, entry in enumerate(cohere_entries)
        if entry["event_type"] == "item_terminalized"
    )
    cohere_over = copy.deepcopy(cohere_entries)
    work_id = str(cohere_over[cohere_terminal_index]["work_item_id"])
    limit = cohere_plan["cohere_prospective_resource_envelope"]["cell_limits"][work_id]
    cohere_over[cohere_terminal_index]["resource_usage"]["tool_calls"] = (
        int(limit["max_actual_tool_calls"]) + 1
    )
    cohere_over_path = tmp_path / "cohere-over-bound.jsonl"
    _write_rehashed_successor_chain(cohere_over_path, cohere_over)
    with pytest.raises(executor.CoverageExecutionError, match="resource|zero-call"):
        executor._ledger_state(cohere_plan, executor._load_ledger(cohere_over_path))

    cohere_total = copy.deepcopy(cohere_entries)
    cohere_batch_terminal_index = next(
        index
        for index, entry in enumerate(cohere_total)
        if entry["event_type"] == "endpoint_batch_terminalized"
    )
    cohere_total[cohere_batch_terminal_index]["resource_usage_totals"]["tokens_completion"] += 1
    cohere_total_path = tmp_path / "cohere-total-mismatch.jsonl"
    _write_rehashed_successor_chain(cohere_total_path, cohere_total)
    with pytest.raises(executor.CoverageExecutionError, match="batch terminal totals"):
        executor._ledger_state(cohere_plan, executor._load_ledger(cohere_total_path))


def test_completed_cohere_receipt_rejects_changed_or_deleted_source(
    frozen_plan: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = _batch(frozen_plan, "cohere_direct")
    plan = _execution_plan(frozen_plan, original)
    batch = next(row for row in plan["endpoint_batches"] if row["batch_id"] == original["batch_id"])
    preflight = _minimal_preflight(plan)
    admission, operator = _live_admission(plan, preflight, batch)
    monkeypatch.setattr(executor, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "verify_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        executor,
        "_validate_blocker_closures",
        lambda **kwargs: {"_cohere_operator_attestation": operator},
    )
    source_paths: list[Path] = []

    def runner(cell: dict, root: Path, cap: Decimal) -> None:
        assert cap == 0
        source_paths.append(_write_real_cohere_source(plan, cell, root))

    completed = executor.execute_one_batch(
        plan=plan,
        preflight=preflight,
        admission=admission,
        repo_root=tmp_path,
        output_root=tmp_path / "study",
        runner=runner,
    )
    assert completed["decision"] == "one_endpoint_batch_terminal"
    assert len(source_paths) == len(batch["work_item_ids"])
    target = source_paths[0]
    original_bytes = target.read_bytes()
    changed = json.loads(original_bytes)
    changed["results"]["epicure_on"]["answer_markdown"] += " changed"
    changed_body = {key: value for key, value in changed.items() if key != "artifact_sha256"}
    changed["artifact_sha256"] = _live_smoke_sha256(changed_body)
    target.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="filename|content address"):
        executor.execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=tmp_path,
            output_root=tmp_path / "study",
            runner=runner,
        )
    assert len(source_paths) == len(batch["work_item_ids"])

    target.write_bytes(original_bytes)
    target.unlink()
    with pytest.raises(executor.CoverageExecutionError, match="requires one durable source"):
        executor.execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=tmp_path,
            output_root=tmp_path / "study",
            runner=runner,
        )
    assert len(source_paths) == len(batch["work_item_ids"])


def test_offline_freeze_writes_identical_artifacts_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "successor"
    snapshots: list[dict[str, bytes]] = []
    reports: list[dict] = []
    for _ in range(2):
        successor.run(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output),
                "freeze",
            ]
        )
        reports.append(json.loads(capsys.readouterr().out))
        snapshots.append(
            {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        )
    assert snapshots[0] == snapshots[1]
    assert reports[0] == reports[1]
    assert reports[0]["provider_completions"] == 0
    assert reports[0]["catalog_gets"] == 0
    assert reports[0]["epicure_calls"] == 0
    assert not (output / "run/coordinator/ledger.jsonl").exists()
    assert not list(output.rglob("*admission*.json"))


def test_two_distinct_output_roots_freeze_byte_identically_without_identity_collision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    canonical = repo / successor.DEFAULT_OUTPUT_ROOT
    alternate = (
        repo
        / "flavourbench/artifacts/season1/current-quality-run/"
        "frontier-coverage-primary-on-v5-successor-v4-r1-determinism-check"
    )
    historical = {
        relative: (repo / relative).read_bytes()
        for relative in (
            *(value for _, value, _, _, _ in successor.FAILED_V1_ARTIFACTS),
            *(value for _, value, _, _, _ in successor.FAILED_V2_ARTIFACTS),
            *(value for _, value, _, _, _ in successor.FAILED_V3_ARTIFACTS),
            successor.FAILED_V3_AUDIT,
            successor.RETIRED_V4_RECEIPT,
            *(
                f"{root}/{suffix}"
                for root in (
                    successor.RETIRED_V4_ROOT,
                    successor.RETIRED_V4_ALTERNATE_ROOT,
                )
                for _, suffix, _, _, _ in successor.RETIRED_V4_ARTIFACTS
            ),
        )
    }

    successor.run(["--repo-root", str(repo), "--output-root", str(canonical), "freeze"])
    canonical_report = json.loads(capsys.readouterr().out)
    successor.run(["--repo-root", str(repo), "--output-root", str(alternate), "freeze"])
    alternate_report = json.loads(capsys.readouterr().out)

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert snapshot(canonical) == snapshot(alternate)
    assert canonical_report["plan_sha256"] == alternate_report["plan_sha256"]
    assert canonical_report["preflight_sha256"] == alternate_report["preflight_sha256"]
    assert canonical_report["dry_run_sha256"] == alternate_report["dry_run_sha256"]
    assert canonical_report["provider_completions"] == 0
    assert canonical_report["catalog_gets"] == 0
    assert canonical_report["epicure_calls"] == 0
    assert all((repo / relative).read_bytes() == value for relative, value in historical.items())
    assert not list(canonical.rglob("*admission*.json"))
    assert not list(alternate.rglob("*admission*.json"))
    assert not list(canonical.rglob("*.jsonl"))
    assert not list(alternate.rglob("*.jsonl"))


def test_v4_real_scanned_root_refreeze_is_identical_and_rejects_foreign_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    historical_bytes = {
        relative: (repo / relative).read_bytes()
        for _, relative, _, _, _ in (
            *successor.FAILED_V1_ARTIFACTS,
            *successor.FAILED_V2_ARTIFACTS,
            *successor.FAILED_V3_ARTIFACTS,
        )
    }
    historical_bytes[successor.FAILED_V3_AUDIT] = (
        repo / successor.FAILED_V3_AUDIT
    ).read_bytes()
    historical_bytes[successor.RETIRED_V4_RECEIPT] = (
        repo / successor.RETIRED_V4_RECEIPT
    ).read_bytes()
    for root in (successor.RETIRED_V4_ROOT, successor.RETIRED_V4_ALTERNATE_ROOT):
        for _, suffix, _, _, _ in successor.RETIRED_V4_ARTIFACTS:
            relative = f"{root}/{suffix}"
            historical_bytes[relative] = (repo / relative).read_bytes()
    output = (
        repo / "flavourbench/artifacts/season1/current-quality-run/"
        "frontier-coverage-primary-on-v5-successor-v4-r1"
    )
    snapshots: list[dict[str, bytes]] = []
    reports: list[dict] = []
    for _ in range(2):
        successor.run(["--repo-root", str(repo), "--output-root", str(output), "freeze"])
        reports.append(json.loads(capsys.readouterr().out))
        snapshots.append(
            {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        )
    assert snapshots[0] == snapshots[1]
    assert reports[0] == reports[1]
    assert all(
        (repo / relative).read_bytes() == value for relative, value in historical_bytes.items()
    )
    assert set(snapshots[0]) == {
        next(path.relative_to(output).as_posix() for path in (output / "plan").glob("*.json")),
        next(path.relative_to(output).as_posix() for path in (output / "preflight").glob("*.json")),
        next(path.relative_to(output).as_posix() for path in (output / "dry-run").glob("*.json")),
        next(path.relative_to(output).as_posix() for path in (output / "templates").glob("*.json")),
    }
    stored_plan = successor._addressed(next((output / "plan").glob("*.json")))
    stored_template = successor._addressed(next((output / "templates").glob("*.json")))
    ordered_cohere = successor._ordered_cohere_work_item_ids(stored_plan)
    assert ordered_cohere != sorted(ordered_cohere)
    assert stored_template["work_item_ids"] == ordered_cohere

    failed_v2 = successor._addressed(repo / successor.FAILED_V2_ARTIFACTS[0][1])
    failed_v2_ids = successor._plan_execution_identifiers(failed_v2)
    prior = successor._prior_identifiers(
        repo / "flavourbench/artifacts",
        verified_successor_output=output,
        successor_plan=stored_plan,
    )
    assert len(failed_v2_ids) == 1666
    assert failed_v2_ids <= prior
    assert not (failed_v2_ids & successor._plan_execution_identifiers(stored_plan))

    failed_v3 = successor._addressed(repo / successor.FAILED_V3_ARTIFACTS[0][1])
    failed_v3_ids = successor._plan_execution_identifiers(failed_v3)
    assert len(failed_v3_ids) == 1666
    assert failed_v3_ids <= prior
    assert not (failed_v3_ids & successor._plan_execution_identifiers(stored_plan))

    retired_v4 = successor._addressed(
        repo / successor.RETIRED_V4_ROOT / successor.RETIRED_V4_ARTIFACTS[0][1]
    )
    retired_v4_ids = successor._plan_execution_identifiers(retired_v4)
    assert len(retired_v4_ids) == 1666
    assert retired_v4_ids <= prior
    assert not (retired_v4_ids & successor._plan_execution_identifiers(stored_plan))

    foreign = output / "foreign.txt"
    foreign.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unverified or foreign"):
        successor.run(["--repo-root", str(repo), "--output-root", str(output), "freeze"])
    foreign.unlink()

    symlink = output / "foreign-link.json"
    symlink.symlink_to(next((output / "plan").glob("*.json")))
    with pytest.raises(SystemExit, match="unverified or foreign"):
        successor.run(["--repo-root", str(repo), "--output-root", str(output), "freeze"])
    symlink.unlink()

    ledger = output / "run/coordinator/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unverified or foreign"):
        successor.run(["--repo-root", str(repo), "--output-root", str(output), "freeze"])


def test_failed_v1_four_artifact_evidence_fails_on_mutation_or_absence(
    tmp_path: Path,
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    evidence = successor._failed_v1_freeze_evidence(repo)
    assert evidence["failure_reason"] == "second_exact_real_root_freeze_rejected_owned_output"
    assert evidence["retired_v1_identifier_count"] == 1666
    assert evidence["provider_or_epicure_calls_made"] is False
    assert len(evidence["artifacts"]) == 4

    plan_path = repo / successor.FAILED_V1_ARTIFACTS[0][1]
    original = plan_path.read_bytes()
    changed = json.loads(original)
    changed["status"] = "forged"
    plan_path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(successor.CoverageSuccessorError, match="content address"):
        successor._failed_v1_freeze_evidence(repo)
    plan_path.write_bytes(original)

    preflight_path = repo / successor.FAILED_V1_ARTIFACTS[1][1]
    preflight_path.unlink()
    with pytest.raises(successor.CoverageSuccessorError, match="regular non-symlink"):
        successor._failed_v1_freeze_evidence(repo)


def test_failed_v2_four_artifact_evidence_fails_on_mutation_or_absence(
    tmp_path: Path,
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    evidence = successor._failed_v2_freeze_evidence(repo)
    assert (
        evidence["failure_reason"] == "cohere_non_usd_cells_misrepresented_as_zero_usd_reservations"
    )
    assert evidence["retired_v2_identifier_count"] == 1666
    assert evidence["provider_or_epicure_calls_made"] is False
    assert evidence["cohere_cells_with_zero_usd_misrepresentation"] == 8
    assert evidence["cohere_batches_with_false_complete_reservation_claim"] == 2
    assert len(evidence["artifacts"]) == 4

    plan_path = repo / successor.FAILED_V2_ARTIFACTS[0][1]
    original = plan_path.read_bytes()
    changed = json.loads(original)
    changed["status"] = "forged"
    plan_path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(successor.CoverageSuccessorError, match="content address"):
        successor._failed_v2_freeze_evidence(repo)
    plan_path.write_bytes(original)

    preflight_path = repo / successor.FAILED_V2_ARTIFACTS[1][1]
    preflight_path.unlink()
    with pytest.raises(successor.CoverageSuccessorError, match="regular non-symlink"):
        successor._failed_v2_freeze_evidence(repo)


def test_failed_v3_independent_no_go_is_bound_and_fails_on_mutation_or_absence(
    tmp_path: Path,
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    evidence = successor._failed_v3_freeze_evidence(repo)
    assert evidence["status"] == "independent_no_go_retired_zero_calls"
    assert evidence["independent_audit"]["decision"] == "NO_GO"
    assert evidence["retired_v3_identifier_count"] == 1666
    assert evidence["generic_current_cohere_zero_or_free_findings"] == 48
    assert evidence["alternate_output_refreeze_verified"] is False
    assert evidence["provider_or_epicure_calls_made"] is False
    assert len(evidence["artifacts"]) == 4

    audit_path = repo / successor.FAILED_V3_AUDIT
    original = audit_path.read_bytes()
    changed = json.loads(original)
    changed["decision"] = "GO"
    audit_path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(successor.CoverageSuccessorError, match="content address"):
        successor._failed_v3_freeze_evidence(repo)
    audit_path.write_bytes(original)

    plan_path = repo / successor.FAILED_V3_ARTIFACTS[0][1]
    plan_path.unlink()
    with pytest.raises(successor.CoverageSuccessorError, match="regular non-symlink"):
        successor._failed_v3_freeze_evidence(repo)


def test_retired_v4_format_only_freeze_is_append_only_and_exactly_bound(
    tmp_path: Path,
) -> None:
    repo = _minimal_governed_repo(tmp_path)
    evidence = successor._retired_v4_format_freeze_evidence(repo)
    assert evidence["status"] == "retired_zero_call_format_only_refreeze_required"
    assert evidence["retirement_reason"] == "ruff_e501_formatting_only_source_closure_change"
    assert evidence["retired_v4_identifier_count"] == 1666
    assert evidence["provider_or_epicure_calls_made"] is False
    assert evidence["execution_admission_granted"] is False
    assert evidence["retired_v4_identifiers_replay_permitted"] is False

    alternate_plan = (
        repo
        / successor.RETIRED_V4_ALTERNATE_ROOT
        / successor.RETIRED_V4_ARTIFACTS[0][1]
    )
    original = alternate_plan.read_bytes()
    changed = json.loads(original)
    changed["status"] = "forged"
    alternate_plan.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(successor.CoverageSuccessorError, match="content address"):
        successor._retired_v4_format_freeze_evidence(repo)
    alternate_plan.write_bytes(original)

    receipt = repo / successor.RETIRED_V4_RECEIPT
    receipt.unlink()
    with pytest.raises(successor.CoverageSuccessorError, match="regular non-symlink"):
        successor._retired_v4_format_freeze_evidence(repo)
