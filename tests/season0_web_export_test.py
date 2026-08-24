from __future__ import annotations

import json
from pathlib import Path

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_web_export import build_export


def test_web_export_contains_aggregate_results_and_no_raw_text(tmp_path: Path) -> None:
    arena = {
        "season_model_id": "model-1",
        "display_name": "Model One",
        "rating": 1010.0,
        "rating_lower": 990.0,
        "rating_upper": 1030.0,
        "comparisons": 100,
        "judgments": 105,
        "both_bad": 5,
    }
    uplift = {
        "season_model_id": "model-1",
        "display_name": "Model One",
        "epicure_win_share": 0.6,
        "interval_lower": 0.5,
        "interval_upper": 0.7,
        "epicure_wins": 30,
        "unaided_wins": 20,
        "ties": 10,
        "comparisons": 60,
    }
    operational = {
        "model-1": {
            "provider": "bedrock",
            "mean_arm_cost_usd": 0.01,
            "latency_median_ms": 2000,
            "latency_p95_ms": 4000,
            "invalid_response_rate": 0.02,
            "end_to_end_failure_rate": 0.03,
            "provider_route_failure_rate": 0.01,
            "identity_leak_rate": 0.0,
            "tool_success_rate": 0.99,
            "epicure_on_tool_use_rate": 0.8,
            "arms": 240,
            "answer_words_median": 300,
        }
    }
    families = ("substitution", "composition", "cookability", "evidence")
    payload = {
        "schema_version": "analysis-v1",
        "synthetic_arms": 0,
        "synthetic_judgments": 0,
        "task_bank_artifact_sha256": "a" * 64,
        "model_manifest_artifact_sha256": "b" * 64,
        "comparison_manifest_artifact_sha256": "c" * 64,
        "judge_manifest_artifact_sha256": "d" * 64,
        "target_cost_audit_artifact_sha256": "f" * 64,
        "counts": {"scored_arms": 240},
        "model_leaderboard": [arena],
        "uplift_leaderboard": [uplift],
        "model_leaderboard_by_family": {family: [arena] for family in families},
        "uplift_leaderboard_by_family": {family: [uplift] for family in families},
        "operational_metrics": operational,
        "panel_uplift": {"task_cluster_win_share": 0.6},
        "judge_diagnostics": {"primary_consensus_coverage": 0.9},
        "judge_family_balanced_sensitivity": {
            "diagnostics": {
                "coverage": 0.75,
                "consensus_available": 3,
                "rows": 4,
            },
            "arena_graph": {"connected": True},
            "model_leaderboard": [arena],
            "uplift_leaderboard": [uplift],
            "panel_uplift": {
                "task_cluster_win_share": 0.55,
                "task_cluster_interval_lower": 0.45,
                "task_cluster_interval_upper": 0.65,
                "valid_comparisons": 3,
            },
        },
        "arena_graph_diagnostics": {"global": {"connected": True}},
        "arena_task_cluster_bootstrap": {"models": {}},
        "reference_overlap_audit": {"overall": {}},
        "verbosity_diagnostics": {"preferred_longer_rate_among_unequal": 0.5},
        "methods": {"model_arena": "arena-rank"},
    }
    model_manifest_payload = {
        "models": [
            {
                "season_model_id": "model-1",
                "canonical_model_id": "openai/model-one",
                "canonical_slug": "openai/model-one",
                "display_name": "Model One",
                "slot_role": "closed_family",
                "provider": "bedrock",
                "provider_name": "Provider",
                "requested_endpoint_id": "provider.model-one",
                "compatibility_artifact_sha256": "e" * 64,
                "endpoint": {},
            }
        ]
    }
    model_manifest = {
        **model_manifest_payload,
        "artifact_sha256": sha256_json(model_manifest_payload),
    }
    payload["model_manifest_artifact_sha256"] = model_manifest["artifact_sha256"]
    payload["counts"]["judgment_records"] = 1
    analysis = {**payload, "artifact_sha256": sha256_json(payload)}
    judgment_summary_payload = {
        "status": "collection_complete",
        "synthetic_judgments": 0,
        "comparison_manifest_artifact_sha256": payload[
            "comparison_manifest_artifact_sha256"
        ],
        "judge_manifest_artifact_sha256": payload["judge_manifest_artifact_sha256"],
        "counts": {
            "terminal_judgments": 1,
            "provider_attempt_records": 2,
            "success": 1,
            "failed": 0,
            "first_pass_documented_throttle_rejections": 1,
            "recovery_attempts": 1,
            "recovered_to_success": 1,
            "recovery_failures": 0,
        },
        "estimated_cost_usd": "0.12",
    }
    judgment_summary = {
        **judgment_summary_payload,
        "artifact_sha256": sha256_json(judgment_summary_payload),
    }
    result = build_export(analysis, model_manifest, judgment_summary, tmp_path)
    saved = json.loads(Path(result["output_path"]).read_bytes())
    assert saved["rows"][0]["competitor_id"] == "Model One"
    assert saved["rows"][0]["model_provisional"] is False
    assert saved["privacy"] == {
        "contains_prompts": False,
        "contains_answers": False,
        "contains_human_references": False,
        "contains_judge_rationales": False,
        "contains_personal_data": False,
    }
    assert "comparison_consensus" not in saved
    assert saved["models"][0]["name"] == "Model One"
    assert saved["models"][0]["provider"] == "bedrock"
    assert saved["judging"]["provider_attempt_records"] == 2
    assert saved["judging"]["recovered_to_success"] == 1
    assert saved["judge_family_balanced_sensitivity"]["diagnostics"]["coverage"] == 0.75
