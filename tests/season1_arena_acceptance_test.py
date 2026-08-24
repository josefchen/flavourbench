from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from flavourbench.main import _require_season1_statistical_acceptance
from flavourbench.models import Season
from flavourbench.season1_arena_acceptance import (
    ARENA_INFERENCE_POLICY_SHA256,
    ArenaInferenceAcceptanceError,
    canonical_sha256,
    load_arena_inference_policy,
    publication_acceptance_deficits,
)
from flavourbench.season1_arena_monte_carlo import (
    RESULT_SCHEMA_VERSION,
    SCENARIOS,
    aggregate_production_results,
    bind_distributed_receipts,
    build_production_layout,
    engine_source_bundle,
    run_dataset,
    verify_production_result,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "contracts/season1/season1-arena-inference-acceptance-v1.json"
MONTE_CARLO_CONTRACT = ROOT / (
    "contracts/season1/method-validation/"
    "season1-arena-production-monte-carlo-v1.json"
)
MONTE_CARLO_ENGINE = ROOT / "src/flavourbench/season1_arena_monte_carlo.py"


def _passing_method_validation() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass",
        "policy_sha256": ARENA_INFERENCE_POLICY_SHA256,
        "engine_source_bundle_sha256": engine_source_bundle()["artifact_sha256"],
        "layout_sha256": build_production_layout()["artifact_sha256"],
        "checkpoint_record_set_sha256": "a" * 64,
        "scenario_dataset_counts": {scenario: 2_000 for scenario in SCENARIOS},
        "completed_records": 16_000,
        "required_records": 16_000,
        "bootstrap_replicates_per_record": 5_000,
        "metrics": {
            "pairwise_difference_coverage": 0.95,
            "pairwise_intervals_evaluated": 1_680_000,
            "type_i_error": 0.05,
            "null_pairwise_intervals_evaluated": 1_650_000,
            "power_at_50_elo": 0.8,
            "shift_50_pairwise_intervals_evaluated": 30_000,
            "probability_scale_absolute_bias": 0.01,
            "maximum_duplicate_interval_width_delta": 0.0,
            "duplication_checks_completed": 7,
            "sparse_withheld_records": 2_000,
            "sparse_invalid_records": 0,
        },
        "acceptance": {
            "pairwise_difference_coverage": True,
            "type_i_error": True,
            "power_at_50_elo": True,
            "probability_scale_absolute_bias": True,
            "no_interval_narrowing_under_exact_row_duplication": True,
            "deterministic_sparse_anchor_and_disconnected_withholding": True,
        },
        "claim_boundary": {
            "production_gate_complete": True,
            "pass_claimed": True,
            "model_quality_evidence": False,
        },
    }
    unsealed = {**body, "artifact_sha256": canonical_sha256(body)}
    distributed_contract = json.loads(
        (
            ROOT
            / "contracts/season1/method-validation/season1-arena-distributed-execution-v2.json"
        ).read_text(encoding="utf-8")
    )
    return bind_distributed_receipts(
        unsealed,
        execution_contract_sha256=str(distributed_contract["artifact_sha256"]),
        execution_manifest_sha256="b" * 64,
        shard_result_set_sha256="c" * 64,
        shard_count=16_000,
    )


def _publication_payload() -> dict[str, object]:
    models = ["model-a", "model-b"]
    support = {
        first: {
            second: {
                "shared_task_clusters": 10,
                "minimum_for_interval": 10,
                "interval_reportable": True,
            }
            for second in models
            if second != first
        }
        for first in models
    }
    intervals = {
        first: {
            second: {"lower": 0.4, "upper": 0.6}
            for second in models
            if second != first
        }
        for first in models
    }
    return {
        "ranking_status": "estimated",
        "bootstrap_replicates": 5_000,
        "production_layout_method_validation": _passing_method_validation(),
        "pairwise_reporting_support": support,
        "pairwise_win_probability_interval": intervals,
        "rows": [
            {
                "competitor_id": model_id,
                "rating": 1000.0,
                "rating_lower": 990.0,
                "rating_upper": 1010.0,
                "provisional": False,
            }
            for model_id in models
        ],
        "statistical_acceptance": {
            "status": "pass",
            "policy_sha256": ARENA_INFERENCE_POLICY_SHA256,
            "view": "all",
            "deficits": [],
            "metrics": {
                "admitted_scored_tasks": 160,
                "admitted_scored_tasks_by_family": {
                    "composition": 40,
                    "cookability": 40,
                    "evidence": 40,
                    "substitution": 40,
                },
                "unique_comparisons_by_model": {model_id: 100 for model_id in models},
                "unique_task_clusters_by_model_family": {
                    model_id: {
                        "composition": 20,
                        "cookability": 20,
                        "evidence": 20,
                        "substitution": 20,
                    }
                    for model_id in models
                },
                "minimum_distinct_raters_per_comparison": 2,
                "postcollection_item_audit_verified": True,
                "unresolved_material_task_defects": 0,
                "bootstrap_connected_rate": 0.99,
                "family_bootstrap_connected_rates": {
                    "composition": 0.99,
                    "cookability": 0.99,
                    "evidence": 0.99,
                    "substitution": 0.99,
                },
            },
        },
    }


def test_policy_content_address_is_exact_and_tampering_fails_closed(tmp_path: Path) -> None:
    policy = load_arena_inference_policy()
    assert policy["artifact_sha256"] == ARENA_INFERENCE_POLICY_SHA256

    tampered = json.loads(POLICY.read_text(encoding="utf-8"))
    tampered["global_fit"]["required_admitted_scored_tasks"] = 1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArenaInferenceAcceptanceError, match="content address"):
        load_arena_inference_policy(path)


def test_publication_rechecks_thresholds_and_pairwise_suppression() -> None:
    payload = _publication_payload()
    assert publication_acceptance_deficits(payload, view="all") == []

    payload["pairwise_reporting_support"]["model-a"]["model-b"] = {
        "shared_task_clusters": 9,
        "minimum_for_interval": 10,
        "interval_reportable": False,
    }
    assert "unsupported_pairwise_interval_not_suppressed" in publication_acceptance_deficits(
        payload,
        view="all",
    )


def test_season1_model_arena_publication_uses_frozen_gate() -> None:
    season = Season(slug="season-1", name="Season 1", epicure_release_id="release")
    snapshot = SimpleNamespace(category="all", cohort="public", track="model_arena")
    payload = _publication_payload()
    _require_season1_statistical_acceptance(season, snapshot, payload)  # type: ignore[arg-type]

    payload["statistical_acceptance"]["metrics"][
        "minimum_distinct_raters_per_comparison"
    ] = 1
    with pytest.raises(HTTPException, match="independent_rater_coverage_below_minimum"):
        _require_season1_statistical_acceptance(season, snapshot, payload)  # type: ignore[arg-type]


def test_production_layout_is_deterministic_and_balanced() -> None:
    first = build_production_layout()
    second = build_production_layout()
    assert first == second
    assert first["counts"] == {
        "models": 16,
        "families": 4,
        "admitted_scored_tasks": 160,
        "tasks_per_family": 40,
        "battles": 3200,
        "endpoint_appearances": 6400,
        "endpoint_appearances_per_model": 400,
        "endpoint_appearances_per_model_family": 100,
        "raters_per_comparison": 2,
    }


def test_sparse_production_layout_runs_resamples_and_withholds() -> None:
    result = run_dataset(
        scenario="observed_endpoint_family_missingness_and_near_disconnection",
        dataset_index=0,
        bootstrap_replicates=2,
        production_mode=False,
    )
    assert result["analysis"]["bootstrap_replicates_executed"] == 2
    assert result["analysis"]["ranking_status"] == "withheld_insufficient_task_clusters"
    assert result["analysis"]["ratings"] is None
    assert result["claim_boundary"]["counts_toward_production_gate"] is False


def test_production_mode_rejects_short_bootstrap() -> None:
    with pytest.raises(ValueError, match="exactly 5,000"):
        run_dataset(
            scenario=SCENARIOS[0],
            dataset_index=0,
            bootstrap_replicates=10,
            production_mode=True,
        )


def test_method_result_verifier_recomputes_acceptance() -> None:
    result = _passing_method_validation()
    assert verify_production_result(result)
    result["metrics"]["pairwise_difference_coverage"] = 0.5
    result["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    assert not verify_production_result(result)


def test_method_result_verifier_rejects_another_engine_source_bundle() -> None:
    result = _passing_method_validation()
    result["engine_source_bundle_sha256"] = "0" * 64
    result["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    assert not verify_production_result(result)


def test_method_result_verifier_rejects_altered_distributed_receipt() -> None:
    result = _passing_method_validation()
    result["distributed_receipts"]["execution_manifest_sha256"] = "d" * 64
    result["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    assert not verify_production_result(result)


def test_method_result_verifier_fails_closed_on_malformed_receipt_boundary() -> None:
    result = _passing_method_validation()
    result["distributed_receipts"]["claim_boundary"] = []
    receipt = result["distributed_receipts"]
    receipt["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "artifact_sha256"}
    )
    result["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    assert not verify_production_result(result)


def test_incomplete_aggregate_cannot_claim_execution(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = aggregate_production_results([empty])
    assert result["status"] == "required_not_yet_executed"
    assert result["completed_records"] == 0
    assert result["acceptance"] is None
    assert result["claim_boundary"]["pass_claimed"] is False


def test_monte_carlo_contract_binds_engine_and_claims_no_results() -> None:
    document = json.loads(MONTE_CARLO_CONTRACT.read_text(encoding="utf-8"))
    digest = document.pop("artifact_sha256")
    assert digest == canonical_sha256(document)
    assert document["engine"]["implementation_file_sha256"] == hashlib.sha256(
        MONTE_CARLO_ENGINE.read_bytes()
    ).hexdigest()
    assert document["engine"]["source_bundle_artifact_sha256"] == (
        engine_source_bundle()["artifact_sha256"]
    )
    assert document["engine"]["source_file_sha256s"] == engine_source_bundle()[
        "source_file_sha256s"
    ]
    assert document["status"] == "required_not_yet_executed"
    assert document["required_execution"]["total_bootstrap_resamples"] == 80_000_000
    assert document["current_execution"]["completed_production_dataset_records"] == 0
    assert document["claim_boundary"]["pass_claimed"] is False
