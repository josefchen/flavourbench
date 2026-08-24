from __future__ import annotations

from pathlib import Path

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_analysis import _implementation_manifest
from flavourbench.season0_paper_assets import generate_assets


def _sealed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "artifact_sha256": sha256_json(value)}


def test_paper_assets_are_generated_from_bound_artifacts(tmp_path: Path) -> None:
    comparison = _sealed(
        {"counts": {"comparisons": 3, "judgable": 2}, "schema_version": "comparison-v1"}
    )
    judge = _sealed(
        {
            "schema_version": "judge-v1",
            "judges": [{"judge_id": "judge-1", "display_name": "Judge One"}],
        }
    )
    model_row = {
        "season_model_id": "model-1",
        "display_name": "Model A&B",
        "rating": 1012.0,
        "rating_lower": 990.0,
        "rating_upper": 1034.0,
        "comparisons": 2,
        "end_to_end_failure_rate": 0.1,
    }
    uplift_row = {
        "season_model_id": "model-1",
        "display_name": "Model A&B",
        "epicure_win_share": 0.6,
        "interval_lower": 0.5,
        "interval_upper": 0.7,
        "epicure_wins": 1,
        "ties": 1,
        "unaided_wins": 0,
        "comparisons": 2,
    }
    families = ("substitution", "composition", "cookability", "evidence")
    target_cost = _sealed(
        {
            "schema_version": "cost-v1",
            "cost_usd": {
                "combined_attributed": "1.25",
                "combined_conservative_exposure": "1.50",
            },
        }
    )
    analysis = _sealed(
        {
            "schema_version": "analysis-v1",
            "status": "automated_cohort_analysis_complete",
            "synthetic_arms": 0,
            "synthetic_judgments": 0,
            "comparison_manifest_artifact_sha256": comparison["artifact_sha256"],
            "judge_manifest_artifact_sha256": judge["artifact_sha256"],
            "target_cost_audit_artifact_sha256": target_cost["artifact_sha256"],
            "implementation": _implementation_manifest(),
            "counts": {
                "scored_arms": 2,
                "judgment_records": 4,
                "consensus_available": 2,
                "consensus_rows": 2,
            },
            "comparison_consensus": [
                {"track": "model_arena", "primary_consensus_available": True},
                {"track": "epicure_uplift", "primary_consensus_available": True},
            ],
            "model_leaderboard": [model_row],
            "uplift_leaderboard": [uplift_row],
            "model_leaderboard_by_family": {family: [model_row] for family in families},
            "uplift_leaderboard_by_family": {family: [uplift_row] for family in families},
            "operational_metrics": {
                "model-1": {"success": 1, "failed": 1, "tool_calls": 2}
            },
            "judge_diagnostics": {
                "judges": {
                    "judge-1": {
                        "orientation_consistency_rate": 1.0,
                        "incomplete_comparisons": 0,
                        "self_judgments": 0,
                    }
                }
            },
            "judge_family_balanced_sensitivity": {
                "diagnostics": {"coverage": 0.75},
                "arena_graph": {"connected": True},
                "model_leaderboard": [model_row],
                "panel_uplift": {
                    "task_cluster_win_share": 0.55,
                    "task_cluster_interval_lower": 0.45,
                    "task_cluster_interval_upper": 0.65,
                    "valid_comparisons": 2,
                    "task_clusters": 2,
                    "epicure_wins": 1,
                    "ties": 1,
                    "unaided_wins": 0,
                },
            },
            "reference_overlap_audit": {
                "overall": {"novel_reference_12gram_match_rate": 0.0}
            },
            "verbosity_diagnostics": {"preferred_longer_rate_among_unequal": 0.5},
            "arena_graph_diagnostics": {"global": {"connected": True}},
            "arena_task_cluster_bootstrap": {
                "task_clusters": 2,
                "successful_replicates": 1000,
                "disconnected_replicates": 0,
                "models": {
                    "model-1": {
                        "rank_one_probability": 0.8,
                        "rating_interval_lower": 980.0,
                        "rating_interval_upper": 1040.0,
                    }
                }
            },
            "panel_uplift": {
                "task_cluster_win_share": 0.6,
                "task_cluster_interval_lower": 0.5,
                "task_cluster_interval_upper": 0.7,
                "epicure_wins": 1,
                "ties": 1,
                "unaided_wins": 0,
                "valid_comparisons": 2,
                "task_clusters": 2,
            },
        }
    )
    judgment = _sealed(
        {
            "schema_version": "judgment-v1",
            "status": "collection_complete",
            "synthetic_judgments": 0,
            "original_collection_summary_artifact_sha256": "e" * 64,
            "recovery_plan_artifact_sha256": "f" * 64,
            "comparison_manifest_artifact_sha256": comparison["artifact_sha256"],
            "judge_manifest_artifact_sha256": judge["artifact_sha256"],
            "counts": {
                "planned_judgments": 4,
                "terminal_judgments": 4,
                "provider_attempt_records": 5,
                "success": 4,
                "failed": 0,
                "first_pass_documented_throttle_rejections": 1,
                "planned_recovery_attempts": 1,
                "recovery_attempts": 1,
                "recovered_to_success": 1,
                "recovery_failures": 0,
                "documented_zero_delivery_throttle_rejections": 1,
            },
            "failure_reasons": {
                "OrphanedRequestEvent": 2,
                "ReadTimeoutError": 3,
                "ThrottlingException": 5,
            },
            "estimated_cost_usd": "0.75",
            "judgment_artifact_sha256s": ["a", "b", "c", "d"],
        }
    )
    paths = generate_assets(
        analysis=analysis,
        comparison_manifest=comparison,
        judge_manifest=judge,
        target_cost_audit=target_cost,
        judgment_summary=judgment,
        output_dir=tmp_path,
    )
    macros = Path(paths["macros"]).read_text(encoding="utf-8")
    pilot_macros = Path(paths["pilot_macros"]).read_text(encoding="utf-8")
    table = Path(paths["model_table"]).read_text(encoding="utf-8")
    assert r"\newcommand{\SeasonZeroCombinedCostUSD}{2.25}" in macros
    assert r"\newcommand{\SeasonZeroReadTimeoutJudgmentCount}{3}" in macros
    assert r"\newcommand{\SeasonZeroThrottledJudgmentCount}{5}" in macros
    assert r"\newcommand{\SeasonZeroJudgeProviderAttemptCount}{5}" in macros
    assert r"\newcommand{\SeasonZeroRecoveredThrottleJudgmentCount}{1}" in macros
    assert r"\newcommand{\SeasonZeroFamilyBalancedCoverage}{75.0\%}" in macros
    assert "Model A\\&B had the highest finite diagnostic rating" in macros
    assert (
        r"\newcommand{\SeasonZeroFamilyBalancedPanelUpliftEstimate}{0.550 [0.450, 0.650]}"
        in macros
    )
    assert r"Model A\&B" in table
    assert r"\newcommand{\SeasonZeroCombinedCostUSD}{2.25}" in pilot_macros
    assert "SeasonZeroTopModel" not in pilot_macros


def test_paper_assets_refuse_incomplete_judgment_registry(tmp_path: Path) -> None:
    comparison = _sealed(
        {"counts": {"comparisons": 1, "judgable": 1}, "schema_version": "comparison-v1"}
    )
    judge = _sealed(
        {
            "schema_version": "judge-v1",
            "judges": [{"judge_id": "judge-1", "display_name": "Judge One"}],
        }
    )
    target_cost = _sealed(
        {
            "schema_version": "cost-v1",
            "cost_usd": {
                "combined_attributed": "0",
                "combined_conservative_exposure": "0",
            },
        }
    )
    analysis = _sealed(
        {
            "schema_version": "analysis-v1",
            "status": "automated_cohort_analysis_complete",
            "synthetic_arms": 0,
            "synthetic_judgments": 0,
            "comparison_manifest_artifact_sha256": comparison["artifact_sha256"],
            "judge_manifest_artifact_sha256": judge["artifact_sha256"],
            "target_cost_audit_artifact_sha256": target_cost["artifact_sha256"],
            "implementation": _implementation_manifest(),
            "counts": {"judgment_records": 1},
        }
    )
    judgment = _sealed(
        {
            "schema_version": "judgment-v1",
            "status": "collection_complete",
            "synthetic_judgments": 0,
            "comparison_manifest_artifact_sha256": comparison["artifact_sha256"],
            "judge_manifest_artifact_sha256": judge["artifact_sha256"],
            "counts": {
                "planned_judgments": 2,
                "terminal_judgments": 1,
                "success": 1,
                "failed": 0,
            },
            "judgment_artifact_sha256s": ["a"],
        }
    )

    import pytest

    from flavourbench.season0_paper_assets import PaperAssetError

    with pytest.raises(PaperAssetError, match="incomplete"):
        generate_assets(
            analysis=analysis,
            comparison_manifest=comparison,
            judge_manifest=judge,
            target_cost_audit=target_cost,
            judgment_summary=judgment,
            output_dir=tmp_path,
        )
