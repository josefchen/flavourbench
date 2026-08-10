from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.evidence_aggregate import SCHEMA_VERSION as AGGREGATE_SCHEMA_VERSION
from flavourbench.execution_policy import ExecutionPolicy
from flavourbench.frontier_contract_runner import IntegrityError
from flavourbench.operational_figure import (
    FIGURE_SCHEMA_VERSION,
    _sha256,
    build_figure_payload,
    content_address_figure,
    load_terminal_aggregate,
    publish_outputs,
    render_csv,
    render_tex,
    write_staged_outputs,
)


def _slice(
    condition: str,
    *,
    complete: int,
    partial: int,
    failed: int,
    normalized: int,
    cost_micros: int,
    successful_tools: int = 0,
    tool_errors: int = 0,
) -> dict[str, object]:
    expected = complete + partial + failed
    return {
        "condition": condition,
        "pairs": {
            "expected": expected,
            "admitted": expected,
            "attempted": expected,
            "finalized": expected,
            "complete": complete,
            "partial": partial,
            "failed": failed,
            "in_progress": 0,
            "pending": 0,
        },
        "arms": {
            "expected": expected,
            "provider_attempted": expected,
            "request_count": expected + successful_tools + tool_errors,
            "normalized_responses": normalized,
            "structured_valid": normalized,
            "structured_invalid_within_normalized": 0,
            "attempted_without_normalized_response": expected - normalized,
            "not_yet_provider_attempted": 0,
            "outcome_class_counts": {
                "normalized_response": normalized,
                "invalid_final_json": expected - normalized,
            },
        },
        "tools": {
            "attempted_arms_with_tool_use": int(successful_tools + tool_errors > 0),
            "normalized_arms_with_tool_use": int(successful_tools > 0),
            "calls": successful_tools + tool_errors,
            "successful_calls": successful_tools,
            "error_calls": tool_errors,
            "normalized_trace_calls": successful_tools,
            "normalized_trace_successful_calls": successful_tools,
            "normalized_trace_error_calls": 0,
        },
        "latency_ms": {
            "n": normalized,
            "minimum": 1_000 if normalized else None,
            "p50": 2_000 + normalized if normalized else None,
            "p95": 3_000 if normalized else None,
            "maximum": 3_000 if normalized else None,
            "mean": 2_100 if normalized else None,
            "basis": "normalized_response_end_to_end",
        },
        "cost": {
            "provider_generations": expected,
            "reconciled_provider_generations": expected,
            "actual_cost_micros": cost_micros,
            "actual_cost_usd": f"{cost_micros / 1_000_000:.6f}",
            "normalized_response_cost_micros": cost_micros,
            "failed_or_non_normalized_cost_micros": 0,
        },
    }


def _unhashed_aggregate() -> dict[str, object]:
    models: list[dict[str, object]] = []
    dataset_cost = 0
    complete_total = partial_total = failed_total = normalized_total = 0
    tool_success_total = tool_error_total = 0
    for index in range(12):
        complete = 10 - (index % 4)
        partial = index % 2
        failed = 10 - complete - partial
        off_normalized = max(0, 10 - (index % 5))
        on_normalized = max(0, 10 - ((index + 2) % 5))
        off_cost = 1_000 + index * 100
        on_cost = 2_000 + index * 100
        dataset_cost += off_cost + on_cost
        complete_total += complete
        partial_total += partial
        failed_total += failed
        normalized_total += off_normalized + on_normalized
        tool_success_total += index + 1
        tool_error_total += index % 3
        models.append(
            {
                "model_id": f"vendor/model-{index + 1:02d}",
                "display_name": f"Frontier Model {index + 1:02d}",
                "conditions": {
                    "epicure_off": _slice(
                        "epicure_off",
                        complete=complete,
                        partial=partial,
                        failed=failed,
                        normalized=off_normalized,
                        cost_micros=off_cost,
                    ),
                    "epicure_on": _slice(
                        "epicure_on",
                        complete=complete,
                        partial=partial,
                        failed=failed,
                        normalized=on_normalized,
                        cost_micros=on_cost,
                        successful_tools=index + 1,
                        tool_errors=index % 3,
                    ),
                },
            }
        )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "official": False,
        "rank_eligible": False,
        "collection_state": "complete",
        "workload": {
            "manifest_sha256": "a" * 64,
            "execution_policy_sha256": ExecutionPolicy().sha256,
            "model_count": 12,
            "expected_pairs": 120,
            "expected_arms": 240,
        },
        "progress": {
            "finalized_pairs": 120,
            "active_journals": 0,
            "normalized_responses": normalized_total,
            "pair_status_counts": {
                "complete": complete_total,
                "partial": partial_total,
                "failed": failed_total,
                "in_progress": 0,
                "pending": 0,
            },
        },
        "human_judgments": {
            "public": 0,
            "expert": 0,
            "preference_estimate": None,
            "bradley_terry_rating": None,
            "epicure_uplift_estimate": None,
        },
        "cost": {
            "dataset_actual_cost_micros": dataset_cost,
            "dataset_actual_cost_usd": f"{dataset_cost / 1_000_000:.6f}",
            "dataset_source_budget_exposure_usd": f"{(dataset_cost + 20_311) / 1_000_000:.6f}",
            "resolved_no_id_incident_count": 1,
            "resolved_no_id_exposure_increment_usd": "0.020311",
            "provider_cost_exact_for_all_attempts": False,
            "all_provider_attempts_have_generation_ids": False,
        },
        "overall_by_condition": {
            "epicure_on": {
                "tools": {
                    "calls": tool_success_total + tool_error_total,
                    "successful_calls": tool_success_total,
                    "error_calls": tool_error_total,
                    "normalized_trace_calls": tool_success_total,
                    "normalized_trace_successful_calls": tool_success_total,
                    "normalized_trace_error_calls": 0,
                }
            }
        },
        "models": models,
        "verification": {
            "all_checks_passed": True,
            "checkpoint_stable_during_aggregation": True,
            "terminal_runner_summary_verified": True,
            "terminal_runner_summary_sha256": "b" * 64,
        },
    }


def _address(value: dict[str, object]) -> tuple[dict[str, object], str]:
    digest = _sha256(value)
    return (
        {
            **value,
            "content_address": {
                "algorithm": "sha256",
                "digest": digest,
                "uri": f"sha256:{digest}",
            },
        },
        digest,
    )


def _write_aggregate(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    value, digest = _address(_unhashed_aggregate())
    path = tmp_path / f"real-exploratory-evidence-{digest}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, digest, value


def test_terminal_digest_and_collection_gate_fail_closed(tmp_path: Path) -> None:
    path, digest, value = _write_aggregate(tmp_path)
    assert load_terminal_aggregate(path, expected_digest=digest)["collection_state"] == "complete"

    with pytest.raises(IntegrityError, match="pinned terminal digest"):
        load_terminal_aggregate(path, expected_digest="f" * 64)

    value["collection_state"] = "active_checkpoint"
    unhashed = dict(value)
    unhashed.pop("content_address")
    partial, partial_digest = _address(unhashed)
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(IntegrityError, match="non-terminal"):
        load_terminal_aggregate(partial_path, expected_digest=partial_digest)


def test_human_judgment_or_estimate_cannot_enter_operational_figure(tmp_path: Path) -> None:
    value = _unhashed_aggregate()
    value["human_judgments"]["public"] = 1  # type: ignore[index]
    addressed, digest = _address(value)
    path = tmp_path / "judged.json"
    path.write_text(json.dumps(addressed), encoding="utf-8")
    with pytest.raises(IntegrityError, match="n=0 judgments"):
        load_terminal_aggregate(path, expected_digest=digest)


def test_view_model_preserves_manifest_order_and_separates_cost_exposure(
    tmp_path: Path,
) -> None:
    path, digest, _ = _write_aggregate(tmp_path)
    aggregate = load_terminal_aggregate(path, expected_digest=digest)
    figure = content_address_figure(build_figure_payload(aggregate))

    assert figure["schema_version"] == FIGURE_SCHEMA_VERSION
    assert [row["display_name"] for row in figure["rows"]][:3] == [
        "Frontier Model 01",
        "Frontier Model 02",
        "Frontier Model 03",
    ]
    assert all(row["order_basis"] == "frozen_manifest_not_performance" for row in figure["rows"])
    assert figure["judgments"] == {
        "public": 0,
        "expert": 0,
        "quality_estimate": None,
        "preference_estimate": None,
        "epicure_uplift_estimate": None,
    }
    assert figure["totals"]["no_id_increment_is_provider_spend"] is False
    assert figure["totals"]["no_id_exposure_increment_usd"] == "0.020311"
    assert figure["totals"]["epicure_journal_calls"] == 90
    assert figure["totals"]["epicure_journal_successful_calls"] == 78
    assert figure["totals"]["normalized_survivor_trace_calls"] == 78
    assert "no_id" not in " ".join(figure["rows"][0])
    assert (
        figure["rows"][0]["epicure_off"]["latency_n"]
        == figure["rows"][0]["epicure_off"]["normalized"]
    )
    assert figure["content_address"]["digest"] == _sha256(
        {key: value for key, value in figure.items() if key != "content_address"}
    )


def test_renderers_are_deterministic_and_explicitly_unranked(tmp_path: Path) -> None:
    path, digest, _ = _write_aggregate(tmp_path)
    aggregate = load_terminal_aggregate(path, expected_digest=digest)
    figure = content_address_figure(build_figure_payload(aggregate))

    tex_first = render_tex(figure)
    tex_second = render_tex(figure)
    csv_text = render_csv(figure)
    assert tex_first == tex_second
    assert "NOT A QUALITY LEADERBOARD" in tex_first
    assert "1,000-token response ceiling" in tex_first
    assert "zero public and zero expert judgments" in tex_first
    assert "Survivor p50 (n)" in tex_first
    assert "no-ID; not provider spend" in tex_first
    assert csv_text.splitlines()[0].startswith("display_order,model_id,display_name")
    assert len(csv_text.splitlines()) == 13

    staged_first = write_staged_outputs(figure, output_directory=tmp_path / "staged")
    staged_second = write_staged_outputs(figure, output_directory=tmp_path / "staged")
    assert staged_first == staged_second
    assert all(
        path.read_bytes() == staged_second[key].read_bytes() for key, path in staged_first.items()
    )


def test_publication_copies_are_byte_identical(tmp_path: Path) -> None:
    path, digest, _ = _write_aggregate(tmp_path)
    figure = content_address_figure(
        build_figure_payload(load_terminal_aggregate(path, expected_digest=digest))
    )
    staged = write_staged_outputs(figure, output_directory=tmp_path / "staged")
    destinations = {
        "json": (tmp_path / "paper" / "figure.json", tmp_path / "web" / "figure.json"),
        "csv": (tmp_path / "paper" / "figure.csv",),
        "tex": (tmp_path / "paper" / "figure.tex",),
    }
    publish_outputs(staged, destinations=destinations)

    assert (tmp_path / "paper" / "figure.json").read_bytes() == staged["json"].read_bytes()
    assert (tmp_path / "web" / "figure.json").read_bytes() == staged["json"].read_bytes()
    assert (tmp_path / "paper" / "figure.csv").read_bytes() == staged["csv"].read_bytes()
    assert (tmp_path / "paper" / "figure.tex").read_bytes() == staged["tex"].read_bytes()


def test_latency_denominator_drift_is_rejected(tmp_path: Path) -> None:
    path, digest, _ = _write_aggregate(tmp_path)
    aggregate = load_terminal_aggregate(path, expected_digest=digest)
    aggregate["models"][0]["conditions"]["epicure_off"]["latency_ms"]["n"] -= 1
    with pytest.raises(IntegrityError, match="survivor latency n"):
        build_figure_payload(aggregate)
