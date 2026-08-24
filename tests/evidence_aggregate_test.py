from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flavourbench.evidence_aggregate import (
    ConditionObservation,
    PairObservation,
    _outcome_class,
    _sha256,
    _structured_and_constraint_labels,
    _verify_content_addressed_summary,
    classify_collection_state,
    publish_copies,
    render_tex_macros,
    render_tex_section,
    render_tex_table,
    summarize_slice,
    validate_payload_invariants,
    write_aggregate,
)
from flavourbench.frontier_contract_runner import IntegrityError


def _condition(
    condition: str,
    *,
    normalized: bool,
    cost_micros: int,
    tool_calls: int = 0,
    tool_errors: int = 0,
) -> ConditionObservation:
    return ConditionObservation(
        condition=condition,
        provider_attempted=True,
        request_count=2 if tool_calls else 1,
        provider_generation_count=2 if tool_calls else 1,
        reconciled_generation_count=2 if tool_calls else 1,
        actual_cost_micros=cost_micros,
        normalized_response=normalized,
        outcome_class="normalized_response" if normalized else "invalid_final_json",
        normalized_response_cost_micros=cost_micros if normalized else 0,
        structured_valid=normalized,
        constraint_status="pass" if normalized else "not_evaluated_no_normalized_response",
        constraint_applicable=normalized,
        latency_ms=1_000 if normalized else None,
        prompt_tokens=100 if normalized else None,
        completion_tokens=50 if normalized else None,
        reasoning_tokens=10 if normalized else None,
        accounting_tokens_prompt=110,
        accounting_tokens_completion=60,
        accounting_native_tokens_prompt=100,
        accounting_native_tokens_completion=50,
        provider_generation_times_ms=(900, 950) if tool_calls else (900,),
        provider_upstream_latencies_ms=(700, 750) if tool_calls else (700,),
        route_http_statuses=(),
        tool_calls=tool_calls,
        tool_successes=tool_calls - tool_errors,
        tool_errors=tool_errors,
        actual_model_id="vendor/model-20260715" if normalized else None,
        actual_provider="Provider" if normalized else None,
        accounting_identities=(("vendor/model-20260715", "Provider"),),
        response_artifact_sha256="b" * 64 if normalized else None,
    )


def _pair(status: str, *, normalized: bool) -> PairObservation:
    work_item = SimpleNamespace(
        ordinal=1,
        work_item_id="a" * 64,
        task=SimpleNamespace(
            public_id="task-1",
            family="substitution",
            prompt_sha256="c" * 64,
        ),
        candidate=SimpleNamespace(model_id="vendor/model", provider_tag="provider/fp8"),
    )
    return PairObservation(
        work_item=work_item,
        pair_status=status,
        admitted=status != "pending",
        attempted=status in {"complete", "partial", "failed"},
        finalized=status in {"complete", "partial", "failed"},
        source_artifact_sha256="d" * 64 if status != "pending" else None,
        source_artifact_filename="source.json" if status != "pending" else None,
        conditions={
            "epicure_off": _condition(
                "epicure_off",
                normalized=normalized,
                cost_micros=200,
            ),
            "epicure_on": _condition(
                "epicure_on",
                normalized=normalized,
                cost_micros=300,
                tool_calls=2 if normalized else 0,
                tool_errors=1 if normalized else 0,
            ),
        },
    )


def test_slice_keeps_operational_metrics_and_denominators_separate() -> None:
    pairs = [_pair("complete", normalized=True), _pair("failed", normalized=False)]

    summary = summarize_slice(pairs, "epicure_on")

    assert summary["pairs"] == {
        "expected": 2,
        "admitted": 2,
        "attempted": 2,
        "finalized": 2,
        "complete": 1,
        "partial": 0,
        "failed": 1,
        "in_progress": 0,
        "pending": 0,
    }
    assert summary["arms"]["normalized_responses"] == 1
    assert summary["arms"]["structured_valid"] == 1
    assert summary["tools"] == {
        "attempted_arms_with_tool_use": 1,
        "normalized_arms_with_tool_use": 1,
        "calls": 2,
        "successful_calls": 1,
        "error_calls": 1,
        "normalized_trace_calls": 2,
        "normalized_trace_successful_calls": 1,
        "normalized_trace_error_calls": 1,
    }
    assert summary["latency_ms"]["n"] == 1
    assert summary["latency_ms"]["provider_generation"]["n"] == 3
    assert summary["tokens"]["accounting_generation_n"] == 3
    assert summary["tokens"]["accounting_native_prompt_total"] == 200
    assert summary["cost"]["provider_generations"] == 3
    assert summary["cost"]["actual_cost_micros"] == 600
    assert "preference" not in summary
    assert "uplift" not in summary


def test_constraint_validator_is_explicitly_only_an_acknowledgement_label() -> None:
    response = {
        "answer_markdown": "Use oat cream.",
        "output_json": {
            "answer_markdown": "Use oat cream.",
            "ingredient_mentions": ["oat cream"],
            "constraints_addressed": ["dairy-free"],
            "uncertainties": ["brand thickness varies"],
        },
        "tool_trace": [],
    }

    result = _structured_and_constraint_labels(
        response,
        prompt="Replace cream with a dairy-free option.",
        model_id="vendor/model",
    )

    assert result["structured_valid"] is True
    assert result["constraint_status"] == "pass"
    assert "not substantive" not in result["scope"]
    assert "expert review" in result["scope"]


def test_content_addressed_write_is_idempotent_and_publish_requires_completion(
    tmp_path: Path,
) -> None:
    payload = {"schema_version": "fixture", "collection_state": "active_checkpoint"}
    first, first_path = write_aggregate(payload, aggregate_directory=tmp_path / "aggregates")
    second, second_path = write_aggregate(payload, aggregate_directory=tmp_path / "aggregates")

    assert first == second
    assert first_path == second_path
    assert first_path.read_bytes() == second_path.read_bytes()
    with pytest.raises(IntegrityError, match="non-final"):
        publish_copies(first_path, first, destinations=[tmp_path / "public.json"])

    unverified, unverified_path = write_aggregate(
        {**payload, "collection_state": "complete", "verification": {}},
        aggregate_directory=tmp_path / "aggregates",
    )
    with pytest.raises(IntegrityError, match="terminal runner summary"):
        publish_copies(
            unverified_path,
            unverified,
            destinations=[tmp_path / "public.json"],
        )

    complete, complete_path = write_aggregate(
        {
            **payload,
            "collection_state": "complete",
            "verification": {"terminal_runner_summary_verified": True},
        },
        aggregate_directory=tmp_path / "aggregates",
    )
    publish_copies(complete_path, complete, destinations=[tmp_path / "public.json"])
    assert (tmp_path / "public.json").read_bytes() == complete_path.read_bytes()


def test_failed_pairs_are_terminal_collection_outcomes_not_unfinished_work() -> None:
    pairs = [_pair("failed", normalized=False)]
    summary = summarize_slice(pairs, "epicure_off")
    assert summary["pairs"]["finalized"] == 1
    assert summary["pairs"]["failed"] == 1
    assert summary["pairs"]["in_progress"] == 0
    assert summary["pairs"]["pending"] == 0
    assert classify_collection_state(
        {"complete": 0, "partial": 0, "failed": 1, "in_progress": 0},
        expected_pairs=1,
        active_journal_count=0,
    ) == "complete"


def test_no_id_http_200_incident_has_a_distinct_non_exact_outcome_class() -> None:
    assert _outcome_class(
        normalized=False,
        source_present=True,
        finalized=True,
        errors=["ProviderError: OpenRouter returned no final choice"],
    ) == "openrouter_http_200_no_choice_without_generation_id"


def test_summary_verifier_rejects_content_tampering(tmp_path: Path) -> None:
    payload = {"schema_version": "fixture", "completed_at": "2026-07-15T00:00:00Z"}
    digest = _sha256(payload)
    value = {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
    path = tmp_path / f"summary-{digest}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _verify_content_addressed_summary(path)["completed_at"] == payload["completed_at"]

    value["completed_at"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(IntegrityError, match="content address"):
        _verify_content_addressed_summary(path)


def test_tex_outputs_label_operational_evidence_as_unranked() -> None:
    slice_value = summarize_slice([_pair("complete", normalized=True)], "epicure_on")
    payload = {
        "content_address": {"digest": "f" * 64},
        "workload": {"expected_pairs": 1},
        "progress": {
            "finalized_pairs": 1,
            "normalized_responses": 2,
            "pair_status_counts": {
                "complete": 1,
                "partial": 0,
                "failed": 0,
            },
        },
        "cost": {
            "provider_generation_count": 3,
            "dataset_actual_cost_usd": "0.0005",
            "dataset_source_budget_exposure_usd": "0.001",
            "resolved_no_id_incident_count": 1,
            "resolved_no_id_full_allowance_usd": "0.001",
            "resolved_no_id_exposure_increment_usd": "0.0005",
            "actual_verified_exposure_usd": "1.322028",
            "conservative_total_exposure_usd": "1.322528",
        },
        "overall_by_condition": {
            "epicure_off": summarize_slice(
                [_pair("complete", normalized=True)], "epicure_off"
            ),
            "epicure_on": slice_value,
        },
        "models": [
            {
                "display_name": "Fixture & Model",
                "actual_identities": [
                    {"model_id": "vendor/model-20260715", "provider": "Provider"}
                ],
                "conditions": {
                    "epicure_off": summarize_slice(
                        [_pair("complete", normalized=True)], "epicure_off"
                    ),
                    "epicure_on": slice_value,
                },
            }
        ],
    }

    table = render_tex_table(payload)
    macros = render_tex_macros(payload)

    assert "zero human judgments and are unranked" in table
    assert "Fixture \\& Model" in table
    assert "\\realDatasetAggregateDigest" in macros
    section = render_tex_section()
    assert "real OpenRouter generations" in section
    assert "survivor conditioned" in section
    assert "exact provider cost is unknowable" in section
    assert "conservative total budget exposure" in section
    assert "preference and Epicure uplift remain undefined" in section


def test_payload_validation_rejects_denominator_drift() -> None:
    pair = _pair("complete", normalized=True)
    off = summarize_slice([pair], "epicure_off")
    on = summarize_slice([pair], "epicure_on")
    payload = {
        "collection_state": "complete",
        "workload": {
            "expected_pairs": 1,
            "model_count": 1,
            "task_family_count": 4,
            "pair_assignments_by_task_family": {
                "substitution": 1,
                "composition": 0,
                "cookability": 0,
                "evidence": 0,
            },
        },
        "progress": {
            "pair_status_counts": {
                "complete": 1,
                "partial": 0,
                "failed": 0,
                "in_progress": 0,
                "pending": 0,
            },
            "finalized_pairs": 1,
            "normalized_responses": 2,
            "active_journals": 0,
        },
        "pair_records": [{"work_item_id": "a" * 64}],
        "overall_by_condition": {"epicure_off": off, "epicure_on": on},
        "models": [{"conditions": {"epicure_off": off, "epicure_on": on}}],
        "by_task_family": {
            family: {
                "epicure_off": summarize_slice(
                    [pair] if family == "substitution" else [], "epicure_off"
                ),
                "epicure_on": summarize_slice(
                    [pair] if family == "substitution" else [], "epicure_on"
                ),
            }
            for family in ("substitution", "composition", "cookability", "evidence")
        },
        "cost": {
            "dataset_actual_cost_micros": 500,
            "provider_generation_count": 3,
        },
        "model_task_family_condition_cube": [
            {
                "model_id": "vendor/model",
                "task_family": family,
                "condition": condition,
            }
            for family in ("substitution", "composition", "cookability", "evidence")
            for condition in ("epicure_off", "epicure_on")
        ],
    }
    validate_payload_invariants(payload)

    payload["overall_by_condition"]["epicure_on"]["arms"][
        "normalized_responses"
    ] = 2
    with pytest.raises(IntegrityError, match="denominators"):
        validate_payload_invariants(payload)
