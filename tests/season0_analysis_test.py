from __future__ import annotations

import json

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_analysis import (
    _arena_rows,
    _arena_task_cluster_bootstrap,
    _expected_judgment_ids,
    _failure_class,
    _implementation_manifest,
    _latest,
    _panel_uplift_dimension_rows,
    _panel_uplift_summary,
    _reference_overlap_audit,
    _uplift_rows,
    _validate_target_cost_audit,
    aggregate_consensus,
    family_balanced_consensus,
)
from flavourbench.season0_judge_protocol import DIMENSIONS


def _record(
    comparison_id: str, judge_id: str, orientation: str, normalized_choice: str
) -> dict[str, object]:
    side = {
        "scores": {dimension: 4 for dimension in DIMENSIONS},
        "fatal_failure": False,
        "summary": "Valid response",
    }
    return {
        "comparison_id": comparison_id,
        "judge": {"judge_id": judge_id},
        "orientation": orientation,
        "status": "success",
        "result": {
            "normalized_choice": normalized_choice,
            "judgment": {"left": side, "right": side},
        },
    }


def test_record_discovery_uses_payload_identity_not_release_filename(tmp_path) -> None:
    record = {
        "arm_id": "arm-content-addressed-id",
        "completed_at": "2026-07-21T00:00:00Z",
    }
    (tmp_path / "000000.json").write_text(json.dumps(record))

    loaded = _latest(tmp_path, "arm", "arm_id")

    assert loaded == {"arm-content-addressed-id": record}


def test_consensus_excludes_self_judge_and_requires_swap_consistency() -> None:
    comparison = {
        "comparison_id": "pair-1",
        "judgable": True,
        "track": "model_arena",
        "task_id": "task-1",
        "task_family": "composition",
        "left": {"season_model_id": "model-1"},
        "right": {"season_model_id": "model-2"},
    }
    judges = [
        {"judge_id": "judge-a", "self_season_model_id": "model-1"},
        {"judge_id": "judge-b", "self_season_model_id": "model-3"},
        {"judge_id": "judge-c", "self_season_model_id": "model-4"},
    ]
    records = {}
    choices = {"judge-a": "right", "judge-b": "left", "judge-c": "left"}
    for judge_id, choice in choices.items():
        for orientation in ("original", "swapped"):
            record = _record("pair-1", judge_id, orientation, choice)
            records[f"{judge_id}-{orientation}"] = record
    rows, diagnostics = aggregate_consensus(
        comparisons=[comparison], judges=judges, judgments=records
    )
    assert rows[0]["primary_consensus_choice"] == "left"
    assert rows[0]["primary_nonself_vote_count"] == 2
    assert diagnostics["judges"]["judge-a"]["self_judgments"] == 1


def test_family_balanced_consensus_caps_each_model_lineage_at_one_vote() -> None:
    judges = [
        {"judge_id": "claude-a", "canonical_model_id": "anthropic/claude-a"},
        {"judge_id": "claude-b", "canonical_model_id": "anthropic/claude-b"},
        {"judge_id": "qwen", "canonical_model_id": "qwen/qwen"},
        {"judge_id": "mistral", "canonical_model_id": "mistral/mistral"},
    ]
    row = {
        "consistent_judge_votes": [
            {"judge_id": "claude-a", "choice": "left", "self_judgment": False},
            {"judge_id": "claude-b", "choice": "left", "self_judgment": False},
            {"judge_id": "qwen", "choice": "right", "self_judgment": False},
            {"judge_id": "mistral", "choice": "right", "self_judgment": False},
        ]
    }

    rows, diagnostics = family_balanced_consensus([row], judges)

    assert rows[0]["primary_consensus_choice"] == "right"
    assert diagnostics["coverage"] == 1.0
    assert diagnostics["family_vote_counts"] == {
        "anthropic": 1,
        "mistral": 1,
        "qwen": 1,
    }


def test_uplift_sample_count_is_one_per_valid_paired_comparison() -> None:
    rows = [
        {
            "track": "epicure_uplift",
            "task_family": "composition",
            "season_model_id": "model-1",
            "primary_consensus_choice": choice,
            "left": {"condition": "epicure_on"},
            "right": {"condition": "epicure_off"},
        }
        for choice in ("left", "right", "tie")
    ]
    uplift = _uplift_rows(rows, {"model-1": "Model 1"}, None)
    assert uplift[0]["comparisons"] == 3
    assert uplift[0]["epicure_wins"] == 1
    assert uplift[0]["unaided_wins"] == 1
    assert uplift[0]["ties"] == 1


def test_disconnected_arena_graph_withholds_global_ratings() -> None:
    rows = [
        {
            "track": "model_arena",
            "task_family": "composition",
            "primary_consensus_choice": "left",
            "left": {"season_model_id": "model-1"},
            "right": {"season_model_id": "model-2"},
        }
    ]
    leaderboard = _arena_rows(
        rows,
        {"model-1": "Model 1", "model-2": "Model 2", "model-3": "Model 3"},
        None,
    )
    assert all(row["rating"] is None for row in leaderboard)


def test_failure_class_separates_model_and_provider_failures() -> None:
    assert _failure_class({"status": "failed", "delivery_state": "reconciled"}) == (
        "model_behavior_failure"
    )
    assert _failure_class(
        {"status": "failed", "delivery_state": "safe_pre_inference"}
    ) == "provider_pre_inference_failure"
    assert _failure_class({"status": "failed", "delivery_state": "uncertain"}) == (
        "uncertain_delivery"
    )
    assert _failure_class(
        {
            "status": "failed",
            "delivery_state": "safe_pre_inference",
            "error_type": "ReadTimeoutError",
        }
    ) == "uncertain_delivery"


def test_analysis_implementation_manifest_pins_source_and_arena_rank() -> None:
    manifest = _implementation_manifest()
    assert manifest["dependencies"]["arena-rank"] == "0.1.1"
    assert set(manifest["source_sha256"]) == {
        "season0_analysis.py",
        "season0_arm_corrections.py",
        "season0_completion_corrections.py",
        "ranking.py",
        "season0_judge_protocol.py",
        "season0_pairs.py",
            "season0_costs.py",
            "season0_judging.py",
            "season0_judgment_recovery.py",
            "season0_collection.py",
    }
    assert all(len(value) == 64 for value in manifest["source_sha256"].values())


def test_reference_overlap_excludes_prompt_ngrams_and_flags_novel_reference_text() -> None:
    task = {
        "task_id": "task-1",
        "prompt": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        "human_reference": {
            "text": (
                "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                "novel one two three four five six seven eight nine ten eleven twelve"
            )
        },
    }
    arms = {
        "prompt-only": {
            "status": "success",
            "task": {"task_id": "task-1"},
            "model": {"season_model_id": "model-1"},
            "result": {
                "answer_markdown": (
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
                )
            },
        },
        "novel-match": {
            "status": "success",
            "task": {"task_id": "task-1"},
            "model": {"season_model_id": "model-1"},
            "result": {
                "answer_markdown": (
                    "novel one two three four five six seven eight nine ten eleven twelve"
                )
            },
        },
        "failed-novel-match": {
            "status": "failed",
            "task": {"task_id": "task-1"},
            "model": {"season_model_id": "model-1"},
            "result": {
                "answer_markdown": (
                    "novel one two three four five six seven eight nine ten eleven twelve"
                )
            },
        },
    }
    audit, flagged = _reference_overlap_audit(
        arms=arms, tasks=[task], model_names={"model-1": "Model 1"}
    )
    assert flagged == {"novel-match"}
    assert audit["overall"]["answers_with_novel_reference_12gram_match"] == 1
    assert audit["overall"]["answers"] == 2


def test_panel_uplift_cluster_interval_recovers_directional_effect() -> None:
    rows = []
    for task_index in range(20):
        for model_index in range(3):
            rows.append(
                {
                    "track": "epicure_uplift",
                    "task_family": "composition",
                    "task_id": f"task-{task_index}",
                    "primary_consensus_choice": "left" if task_index < 15 else "right",
                    "left": {"condition": "epicure_on"},
                    "right": {"condition": "epicure_off"},
                    "season_model_id": f"model-{model_index}",
                }
            )
    summary = _panel_uplift_summary(rows, None)
    assert summary["valid_comparisons"] == 60
    assert summary["task_clusters"] == 20
    assert summary["task_cluster_win_share"] == 0.75
    assert summary["task_cluster_interval_lower"] > 0.5


def test_panel_dimension_uplift_resamples_tasks_and_recovers_effect() -> None:
    rows = []
    for task_index in range(20):
        for model_index in range(3):
            off_score = 2
            on_score = 4 if task_index < 18 else 1
            rows.append(
                {
                    "track": "epicure_uplift",
                    "task_id": f"task-{task_index}",
                    "primary_scores_available": True,
                    "left": {"condition": "epicure_on"},
                    "right": {"condition": "epicure_off"},
                    "primary_side_scores": {
                        "left": {
                            "scores": {
                                dimension: on_score for dimension in DIMENSIONS
                            }
                        },
                        "right": {
                            "scores": {
                                dimension: off_score for dimension in DIMENSIONS
                            }
                        },
                    },
                    "season_model_id": f"model-{model_index}",
                }
            )
    result = _panel_uplift_dimension_rows(rows, replicates=200)
    evidence = next(row for row in result if row["dimension"] == "evidence_use")
    assert evidence["comparisons"] == 60
    assert evidence["task_clusters"] == 20
    assert evidence["mean_delta"] == 1.7
    assert evidence["lower"] > 0


def test_arena_task_cluster_bootstrap_recovers_stable_leader() -> None:
    rows = []
    models = {"strong": "Strong", "middle": "Middle", "weak": "Weak"}
    for task_index in range(12):
        for left, right in (("strong", "middle"), ("strong", "weak"), ("middle", "weak")):
            rows.append(
                {
                    "track": "model_arena",
                    "task_id": f"task-{task_index}",
                    "primary_consensus_choice": "left",
                    "left": {"season_model_id": left},
                    "right": {"season_model_id": right},
                }
            )
    result = _arena_task_cluster_bootstrap(rows, models, replicates=100)
    assert result["successful_replicates"] == 100
    assert result["models"]["strong"]["rank_one_probability"] == 1.0


def test_expected_judgments_cover_only_judgable_rows_and_both_orientations() -> None:
    identifiers = _expected_judgment_ids(
        comparisons=[
            {"comparison_id": "comparison-a", "judgable": True},
            {"comparison_id": "comparison-b", "judgable": False},
        ],
        judges=[{"judge_id": "judge-a"}],
        comparison_sha="c" * 64,
        judge_sha="j" * 64,
    )
    assert len(identifiers) == 2


def test_cost_audit_mean_includes_unattributed_conservative_reservation() -> None:
    arms = {
        f"arm-{index}": {
            "model": {"season_model_id": "model-1", "provider": "bedrock"},
            "reservation_usd": "0.10",
        }
        for index in range(240)
    }
    rate_card = {"schema_version": "test-rate-card"}
    body = {
        "schema_version": "flavourbench-season0-cost-audit-v1",
        "synthetic_arms": 0,
        "complete_exposure_accounting": True,
        "complete_openrouter_request_level_attribution": True,
        "rate_card_sha256": sha256_json(rate_card),
        "counts": {"arms": 240, "attributed_arms": 239, "unattributed_arms": 1},
        "models": {
            "model-1": {
                "arms": 240,
                "attributed_arms": 239,
                "cost_usd": "2.39",
                "provider": "bedrock",
                "display_name": "Model 1",
            }
        },
        "unattributed": [
            {
                "arm_id": "arm-239",
                "provider": "bedrock",
                "conservative_reservation_usd": "0.10",
            }
        ],
        "cost_usd": {
            "combined_attributed": "2.39",
            "combined_conservative_exposure": "2.49",
            "unattributed_conservative_reservations": "0.10",
        },
    }
    audit = {**body, "artifact_sha256": sha256_json(body)}
    _sha, costs = _validate_target_cost_audit(
        target_cost_audit=audit,
        rate_card=rate_card,
        arms=arms,
        models=[
            {
                "season_model_id": "model-1",
                "provider": "bedrock",
                "display_name": "Model 1",
            }
        ],
    )
    assert costs["model-1"]["conservative_cost_usd"] == 2.49
    assert costs["model-1"]["mean_arm_cost_usd"] == 2.49 / 240
