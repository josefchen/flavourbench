from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.frontier_contract_runner import IntegrityError
from flavourbench.real_dataset_runner import (
    append_dataset_ledger_event,
    load_dataset_ledger,
)
from flavourbench.real_dataset_verifier import (
    EXPLORATORY_EPICURE_RELEASE_ID,
    ExpectedPair,
    _sha256,
    audit_record_graph,
    verify_summary_content_address,
)

WORK_ITEM_ID = "a" * 64
SOURCE_SHA = "b" * 64
OFF_RESPONSE_SHA = "c" * 64
ON_RESPONSE_SHA = "d" * 64
RESERVATION_SHA = "e" * 64


def _expected() -> ExpectedPair:
    prompt = "Replace butter while preserving browning and a crisp edge."
    return ExpectedPair(
        ordinal=1,
        work_item_id=WORK_ITEM_ID,
        manifest_sha256="1" * 64,
        task_registry_sha256="2" * 64,
        task_id="sub-test",
        task_family="substitution",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        task_split="pilot",
        task_review_status="candidate",
        slot_id="closed-test",
        model_id="vendor/model",
        canonical_model_slug="vendor/model-20260715",
        provider_tag="provider/fixed",
        provider_name="Provider",
        endpoint_manifest_sha256="3" * 64,
        endpoint_execution_sha256="4" * 64,
        execution_policy_sha256="5" * 64,
    )


def _tool_trace() -> dict:
    return {
        "round_index": 0,
        "name": "find_pairings",
        "arguments": {"ingredients": ["butter", "oil"]},
        "result": "bounded evidence",
        "result_sha256": "6" * 64,
        "latency_ms": 4,
        "is_error": False,
    }


def _result(condition: str, expected: ExpectedPair) -> dict:
    generation_id = f"gen-{condition}"
    return {
        "answer_markdown": "Use a measured oil blend and monitor browning.",
        "output_json": {"answer_markdown": "Use a measured oil blend."},
        "actual_model_id": expected.canonical_model_slug,
        "actual_provider": expected.provider_name,
        "generation_id": generation_id,
        "generation_ids": [generation_id],
        "generation_metadata": [
            {
                "generation_id": generation_id,
                "cost_micros": 100,
                "reconciled": True,
            }
        ],
        "cost_micros": 100,
        "cost_reconciled": True,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "reasoning_tokens": 0,
        "latency_ms": 10,
        "retries": 0,
        "finish_reason": "stop",
        "tool_trace": [_tool_trace()] if condition == "epicure_on" else [],
    }


def _source(expected: ExpectedPair) -> dict:
    on_event = {"arm_id": "run:epicure_on", **_tool_trace()}
    return {
        "artifact_sha256": SOURCE_SHA,
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": "run-test",
        "dataset_work_item_id": expected.work_item_id,
        "dataset_task_id": expected.task_id,
        "candidate_manifest_sha256": expected.manifest_sha256,
        "prompt": expected.prompt,
        "prompt_sha256": expected.prompt_sha256,
        "category": expected.task_family,
        "requested_model_id": expected.model_id,
        "requested_provider": expected.provider_tag,
        "run_purpose": "epicure_on_off_pair",
        "endpoint_execution_contract_sha256": expected.endpoint_execution_sha256,
        "execution_policy_sha256": expected.execution_policy_sha256,
        "model_contract": {
            "id": expected.model_id,
            "canonical_slug": expected.canonical_model_slug,
        },
        "frozen_generation_contract": {
            "expected_actual_model_id": expected.canonical_model_slug,
            "expected_actual_provider_slug": expected.provider_name,
        },
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "status": "complete",
        "epicure": {
            "release_id": EXPLORATORY_EPICURE_RELEASE_ID,
            "application_sha256": "7" * 64,
            "bundle_sha256": "8" * 64,
        },
        "epicure_tool_schema_sha256": "9" * 64,
        "results": {
            "epicure_off": _result("epicure_off", expected),
            "epicure_on": _result("epicure_on", expected),
        },
        "errors": {},
        "incomplete_generation_metadata": [],
        "budget": {
            "actual_cost_micros": 200,
            "all_generation_costs_reconciled": True,
        },
        "provider_attempt_events": [
            {"attempt_id": "attempt-off", "event_type": "response_received"},
            {"attempt_id": "attempt-on", "event_type": "response_received"},
        ],
        "mcp_trace_events": [on_event],
    }


def _response(
    condition: str,
    expected: ExpectedPair,
    source: dict,
    artifact_sha: str,
) -> dict:
    result = copy.deepcopy(source["results"][condition])
    source_events = [
        copy.deepcopy(event)
        for event in source["mcp_trace_events"]
        if event["arm_id"].endswith(f":{condition}")
    ]
    return {
        "artifact_sha256": artifact_sha,
        "schema_version": "flavourbench-real-exploratory-response-v1",
        "work_item_id": expected.work_item_id,
        "manifest_sha256": expected.manifest_sha256,
        "task_registry_sha256": expected.task_registry_sha256,
        "execution_policy_sha256": expected.execution_policy_sha256,
        "condition": condition,
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "research_release_eligible": False,
        "task": {
            "public_id": expected.task_id,
            "family": expected.task_family,
            "prompt": expected.prompt,
            "prompt_sha256": expected.prompt_sha256,
            "split": expected.task_split,
            "review_status": expected.task_review_status,
        },
        "model": {
            "slot_id": expected.slot_id,
            "requested_model_id": expected.model_id,
            "canonical_model_slug": expected.canonical_model_slug,
            "actual_model_id": expected.canonical_model_slug,
            "provider_tag": expected.provider_tag,
            "actual_provider": expected.provider_name,
            "endpoint_manifest_sha256": expected.endpoint_manifest_sha256,
            "endpoint_execution_sha256": expected.endpoint_execution_sha256,
            "execution_policy_sha256": expected.execution_policy_sha256,
        },
        "source": {
            "artifact_sha256": source["artifact_sha256"],
            "run_id": source["run_id"],
        },
        "provenance": {
            "epicure_access": condition == "epicure_on",
            "epicure": copy.deepcopy(source["epicure"]),
            "epicure_tool_schema_sha256": source["epicure_tool_schema_sha256"],
            "mcp_trace_events": source_events,
        },
        "cost": {
            "actual_cost_micros": result["cost_micros"],
            "all_generation_costs_reconciled": True,
            "generation_ids": copy.deepcopy(result["generation_ids"]),
            "generation_metadata": copy.deepcopy(result["generation_metadata"]),
        },
        "response": result,
    }


def _ledger(response_sha256s: list[str] | None = None) -> list[dict]:
    return [
        {
            "event_type": "reservation_created",
            "entry_sha256": RESERVATION_SHA,
            "work_item_id": WORK_ITEM_ID,
            "reserved_usd": "1",
        },
        {
            "event_type": "source_artifact_recorded",
            "entry_sha256": "f" * 64,
            "reservation_entry_sha256": RESERVATION_SHA,
            "work_item_id": WORK_ITEM_ID,
            "source_artifact_sha256": SOURCE_SHA,
            "source_actual_cost_usd": "0.0002",
            "response_artifact_sha256s": response_sha256s
            if response_sha256s is not None
            else [OFF_RESPONSE_SHA, ON_RESPONSE_SHA],
        },
    ]


def _complete_graph() -> tuple[
    dict[str, ExpectedPair],
    list[dict],
    dict[str, dict],
    dict[tuple[str, str], dict],
]:
    expected = _expected()
    source = _source(expected)
    responses = {
        (WORK_ITEM_ID, "epicure_off"): _response("epicure_off", expected, source, OFF_RESPONSE_SHA),
        (WORK_ITEM_ID, "epicure_on"): _response("epicure_on", expected, source, ON_RESPONSE_SHA),
    }
    return (
        {WORK_ITEM_ID: expected},
        _ledger(),
        {WORK_ITEM_ID: source},
        responses,
    )


def _audit(
    expected: dict[str, ExpectedPair],
    ledger: list[dict],
    sources: dict[str, dict],
    responses: dict[tuple[str, str], dict],
    *,
    strict_final: bool = True,
):
    return audit_record_graph(
        expected_pairs=expected,
        ledger=ledger,
        sources=sources,
        responses=responses,
        strict_final=strict_final,
        max_tool_rounds=4,
        max_tool_calls_total=12,
        source_actual_costs_usd={WORK_ITEM_ID: Decimal("0.0002")} if sources else {},
    )


def _finding(findings, check_id: str):
    return next(item for item in findings if item.check_id == check_id)


def test_complete_record_graph_passes_strict_integrity_checks() -> None:
    findings, metrics = _audit(*_complete_graph())

    assert not [item for item in findings if item.status == "fail"]
    assert metrics["complete_pairs"] == 1
    assert metrics["generation_ids"] == 2
    assert metrics["unique_generation_ids"] == 2
    assert metrics["source_recorded_cost_usd"] == "0.0002"


def test_summary_content_address_rejects_tampering() -> None:
    payload = {
        "schema_version": "flavourbench-real-exploratory-summary-v1",
        "runner_run_id": "run-test",
        "mode": "execute",
    }
    digest = _sha256(payload)
    summary = {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
    path = Path(f"real-exploratory-summary-{digest}.json")

    assert verify_summary_content_address(summary, path)
    summary["mode"] = "dry_run_no_provider_calls"
    assert not verify_summary_content_address(summary, path)


def test_dataset_ledger_hash_chain_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append_dataset_ledger_event(
        path,
        {
            "event_type": "reservation_created",
            "runner_run_id": "run-test",
            "work_item_id": WORK_ITEM_ID,
            "reserved_usd": "1",
        },
        recorded_at="2026-07-15T00:00:00Z",
    )
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["reserved_usd"] = "2"
    path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="digest mismatch"):
        load_dataset_ledger(path)


def test_duplicate_generation_id_fails_uniqueness_gate() -> None:
    expected, ledger, sources, responses = _complete_graph()
    source = sources[WORK_ITEM_ID]
    on = source["results"]["epicure_on"]
    on["generation_id"] = "gen-epicure_off"
    on["generation_ids"] = ["gen-epicure_off"]
    on["generation_metadata"][0]["generation_id"] = "gen-epicure_off"
    responses[(WORK_ITEM_ID, "epicure_on")] = _response(
        "epicure_on", expected[WORK_ITEM_ID], source, ON_RESPONSE_SHA
    )

    findings, _ = _audit(expected, ledger, sources, responses)

    assert _finding(findings, "generation_and_attempt_uniqueness").status == "fail"


def test_returned_model_identity_drift_fails_exact_identity_gate() -> None:
    expected, ledger, sources, responses = _complete_graph()
    sources[WORK_ITEM_ID]["results"]["epicure_on"]["actual_model_id"] = "vendor/substituted-model"
    responses[(WORK_ITEM_ID, "epicure_on")]["response"]["actual_model_id"] = (
        "vendor/substituted-model"
    )

    findings, _ = _audit(expected, ledger, sources, responses)

    assert _finding(findings, "model_provider_identity").status == "fail"


def test_broken_pair_normalization_link_fails_record_graph() -> None:
    expected, ledger, sources, responses = _complete_graph()
    responses[(WORK_ITEM_ID, "epicure_on")]["source"]["artifact_sha256"] = "0" * 64

    findings, _ = _audit(expected, ledger, sources, responses)

    assert _finding(findings, "model_provider_identity").status == "fail"


def test_missing_eligible_arm_fails_but_provider_failure_is_reliability_only() -> None:
    expected, ledger, sources, responses = _complete_graph()
    responses.pop((WORK_ITEM_ID, "epicure_on"))
    ledger[1]["response_artifact_sha256s"] = [OFF_RESPONSE_SHA]

    broken_findings, _ = _audit(expected, ledger, sources, responses)
    assert _finding(broken_findings, "generation_accounting").status == "fail"

    source = sources[WORK_ITEM_ID]
    source["results"].pop("epicure_on")
    source["errors"] = {"epicure_on": "ProviderError: invalid final JSON"}
    source["budget"]["actual_cost_micros"] = 100
    ledger[1]["source_actual_cost_usd"] = "0.0001"
    valid_partial_findings, metrics = audit_record_graph(
        expected_pairs=expected,
        ledger=ledger,
        sources=sources,
        responses=responses,
        strict_final=True,
        max_tool_rounds=4,
        max_tool_calls_total=12,
        source_actual_costs_usd={WORK_ITEM_ID: Decimal("0.0001")},
    )

    assert _finding(valid_partial_findings, "generation_accounting").status == "pass"
    assert _finding(valid_partial_findings, "paired_response_reliability").status == "warn"
    assert metrics["partial_pairs"] == 1


def test_conservative_no_id_resolution_is_supported_warned_and_never_ranked() -> None:
    expected, _ledger_entries, sources, responses = _complete_graph()
    source = sources[WORK_ITEM_ID]
    source["results"].pop("epicure_off")
    source["errors"] = {"epicure_off": "ProviderError: OpenRouter returned no final choice"}
    source["budget"]["actual_cost_micros"] = 100
    responses.pop((WORK_ITEM_ID, "epicure_off"))
    incident_sha = "1" * 64
    resolution_sha = "2" * 64
    resolution_event_sha = "3" * 64
    ledger = [
        {
            "event_type": "reservation_created",
            "entry_sha256": RESERVATION_SHA,
            "work_item_id": WORK_ITEM_ID,
            "reserved_usd": "1",
        },
        {
            "event_type": "execution_incident",
            "entry_sha256": incident_sha,
            "work_item_id": WORK_ITEM_ID,
            "reservation_entry_sha256": RESERVATION_SHA,
            "incident": "generation_cost_unreconciled_reservation_retained",
            "source_artifact_sha256": SOURCE_SHA,
        },
        {
            "event_type": "source_incident_resolution_recorded",
            "entry_sha256": resolution_event_sha,
            "work_item_id": WORK_ITEM_ID,
            "reservation_entry_sha256": RESERVATION_SHA,
            "incident_entry_sha256": incident_sha,
            "source_artifact_sha256": SOURCE_SHA,
            "resolution_artifact_sha256": resolution_sha,
            "provider_reconciled_actual_cost_usd": "0.0001",
            "conservative_budget_exposure_usd": "1",
            "provider_cost_exact_for_unidentified_response": False,
            "safe_to_replay": False,
            "normalizable_conditions": ["epicure_on"],
        },
        {
            "event_type": "source_artifact_recorded",
            "entry_sha256": "4" * 64,
            "reservation_entry_sha256": RESERVATION_SHA,
            "work_item_id": WORK_ITEM_ID,
            "source_artifact_sha256": SOURCE_SHA,
            "source_actual_cost_usd": "0.0001",
            "source_budget_exposure_usd": "1",
            "source_incident_resolution_sha256": resolution_sha,
            "source_incident_resolution_ledger_entry_sha256": resolution_event_sha,
            "all_generation_costs_reconciled": False,
            "provider_cost_exact": False,
            "response_artifact_sha256s": [ON_RESPONSE_SHA],
        },
    ]

    findings, metrics = audit_record_graph(
        expected_pairs=expected,
        ledger=ledger,
        sources=sources,
        responses=responses,
        strict_final=True,
        max_tool_rounds=4,
        max_tool_calls_total=12,
        source_actual_costs_usd={WORK_ITEM_ID: Decimal("0.0001")},
    )

    assert not [item for item in findings if item.status == "fail"]
    resolution_finding = _finding(findings, "conservative_no_id_incident_resolutions")
    assert resolution_finding.status == "warn"
    assert resolution_finding.severity == "high"
    assert resolution_finding.observed["rank_eligible"] is False
    assert resolution_finding.observed["research_release_eligible"] is False
    assert metrics["conservative_incident_resolutions"] == 1
    assert metrics["conservative_incident_provider_actual_usd"] == "0.0001"
    assert metrics["conservative_incident_budget_exposure_usd"] == "1"

    ledger[2]["safe_to_replay"] = True
    tampered_findings, _ = audit_record_graph(
        expected_pairs=expected,
        ledger=ledger,
        sources=sources,
        responses=responses,
        strict_final=True,
        max_tool_rounds=4,
        max_tool_calls_total=12,
        source_actual_costs_usd={WORK_ITEM_ID: Decimal("0.0001")},
    )
    assert _finding(tampered_findings, "record_graph_links").status == "fail"


def test_unresolved_reservation_warns_in_progress_but_fails_strict_final() -> None:
    expected = {WORK_ITEM_ID: _expected()}
    ledger = [_ledger()[0]]

    incomplete_findings, _ = _audit(
        expected,
        ledger,
        {},
        {},
        strict_final=False,
    )
    strict_findings, _ = _audit(expected, ledger, {}, {}, strict_final=True)

    assert _finding(incomplete_findings, "workload_completion").status == "warn"
    assert not [item for item in incomplete_findings if item.status == "fail"]
    assert _finding(strict_findings, "workload_completion").status == "fail"
    assert _finding(strict_findings, "real_provider_generation_evidence").status == "fail"


def test_generation_cost_mismatch_fails_accounting_gate() -> None:
    expected, ledger, sources, responses = _complete_graph()
    sources[WORK_ITEM_ID]["results"]["epicure_on"]["generation_metadata"][0]["cost_micros"] = 99
    responses[(WORK_ITEM_ID, "epicure_on")]["response"]["generation_metadata"][0]["cost_micros"] = (
        99
    )
    responses[(WORK_ITEM_ID, "epicure_on")]["cost"]["generation_metadata"][0]["cost_micros"] = 99

    findings, _ = _audit(expected, ledger, sources, responses)

    assert _finding(findings, "generation_accounting").status == "fail"
