from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from flavourbench.epicure_selection_powered_analysis import PanelData
from flavourbench.epicure_selection_powered_analysis_v2 import (
    _leaderboard_csv,
    analyze_panels,
    analyze_repeatability,
    anchor_cluster_available_bootstrap,
    anchor_cluster_weighted_sign_flip_pvalues,
)
from flavourbench.epicure_selection_powered_joint_analysis_v1 import (
    combine_panel_data,
    replication_stability,
)
from flavourbench.epicure_selection_powered_plan_v48 import verify_plan

ROOT = Path(__file__).resolve().parents[1]
JOINT_PLAN = ROOT / (
    "benchmark/powered-v48/plan/"
    "epicure-selection-joint-analysis-plan-"
    "099e7690c0406ad13307735ee0e2c98846d45a46cacd4be06a2bc5b196d3be97.json"
)


def _panel(*, panel: str, task_prefix: str, offset: float = 0.0) -> PanelData:
    families = (
        "substitution",
        "substitution",
        "pairing",
        "pairing",
        "constraint",
        "constraint",
        "cultural_composition",
        "cultural_composition",
    )
    scores = (
        np.asarray(
            [
                [80, 70, 80, 70, 80, 70, 80, 70],
                [60, 50, 60, 50, 60, 50, 60, 50],
                [40, 30, 40, 30, 40, 30, 40, 30],
            ],
            dtype=float,
        )
        + offset
    )
    valid = np.ones_like(scores, dtype=bool)
    return PanelData(
        panel=panel,
        model_ids=("high", "middle", "low"),
        model_names=("High", "Middle", "Low"),
        slot_ids=("a", "b", "c"),
        task_ids=tuple(f"{task_prefix}-{index}" for index in range(8)),
        families=families,
        scores=scores,
        completed=valid,
        parseable=valid,
        selections=tuple(tuple("ABC" for _ in range(8)) for _ in range(3)),
        response_artifact_sha256s=tuple(
            hashlib.sha256(f"{panel}-{index}".encode()).hexdigest() for index in range(24)
        ),
        spend_micros=100,
    )


def test_joint_plan_freezes_anchor_clustered_two_panel_inference() -> None:
    document = json.loads(JOINT_PLAN.read_text(encoding="utf-8"))
    assert verify_plan(document)
    assert document["design"]["primary_tasks_per_panel"] == 640
    assert document["design"]["scheduled_primary_tasks_per_model"] == 1280
    assert document["design"]["unique_anchor_clusters"] == 1178
    assert document["design"]["shared_anchor_clusters"] == 102
    assert document["design"]["same_family_shared_anchors"] == 35
    assert document["design"]["cross_family_shared_anchors"] == 67
    assert document["roster"]["pairwise_hypotheses"] == 325
    assert document["power"]["three_point_difference_power"] > 0.9
    assert document["source_rules"]["score_or_selection_inspected_before_freeze"] is False
    assert hashlib.sha256(JOINT_PLAN.read_bytes()).hexdigest() == (
        "29f8f12ff99183b1142d2500ba8a29f060b982ab1f6862f21bde69befcb191e9"
    )


def test_combine_panel_data_preserves_every_cell_and_artifact() -> None:
    left = _panel(panel="primary-1", task_prefix="p1")
    right = _panel(panel="primary-2", task_prefix="p2", offset=1.0)
    combined = combine_panel_data(left, right, panel="joint")
    assert combined.scores.shape == (3, 16)
    assert combined.task_ids == left.task_ids + right.task_ids
    assert combined.selections[0] == left.selections[0] + right.selections[0]
    assert len(combined.response_artifact_sha256s) == 48
    assert combined.spend_micros == 200


def test_anchor_cluster_resampling_moves_shared_anchor_tasks_together() -> None:
    panel = _panel(panel="primary", task_prefix="task")
    clusters = ("shared", "s2", "shared", "p2", "c1", "c2", "r1", "r2")
    bootstrap_1 = anchor_cluster_available_bootstrap(
        panel.scores,
        panel.completed,
        panel.families,
        clusters,
        resamples=50,
        seed=19,
    )
    bootstrap_2 = anchor_cluster_available_bootstrap(
        panel.scores,
        panel.completed,
        panel.families,
        clusters,
        resamples=50,
        seed=19,
    )
    observed, pvalues = anchor_cluster_weighted_sign_flip_pvalues(
        panel.scores - 50,
        panel.completed,
        panel.families,
        clusters,
        resamples=99,
        seed=23,
    )
    assert np.array_equal(bootstrap_1, bootstrap_2)
    assert bootstrap_1.shape == (50, 3)
    assert observed.shape == pvalues.shape == (3,)
    assert np.all((pvalues > 0) & (pvalues <= 1))


def test_joint_analysis_reports_anchor_count_and_panel_stability() -> None:
    left = _panel(panel="primary-1", task_prefix="p1")
    right = _panel(panel="primary-2", task_prefix="p2", offset=1.0)
    combined = combine_panel_data(left, right, panel="joint")
    clusters = (
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "a",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
    )
    tasks = [{"task_id": task_id, "chance_score_bps": 3000} for task_id in combined.task_ids]
    plan = {
        "artifact_sha256": "0" * 64,
        "outcomes": {"primary_definition": "joint synthetic score"},
        "inference": {
            "bootstrap_resamples": 100,
            "permutation_resamples": 100,
            "seed": 29,
        },
        "repeatability": {"acceptance_floor": 0.8},
    }
    analysis = analyze_panels(
        primary=combined,
        taskset={"tasks": tasks},
        plan=plan,
        cluster_ids=clusters,
    )
    stability = replication_stability(left, right)
    assert analysis["inference"]["independence_unit"] == "anchor_ingredient"
    assert analysis["inference"]["independent_cluster_count"] == 15
    assert analysis["inference"]["shared_anchor_tasks_move_together"] is True
    assert stability["model_score_pearson"] == 1.0
    assert stability["model_rank_spearman"] == 1.0
    assert len(stability["models"]) == 3


def test_repeatability_reports_unestimable_model_without_imputing_zero() -> None:
    families = (
        "substitution",
        "pairing",
        "constraint",
        "cultural_composition",
    )
    model_ids = ("complete", "filtered")
    primary_ids = tuple(f"primary-{family}" for family in families)
    repeat_ids = tuple(f"repeat-{family}" for family in families)
    primary_valid = np.ones((2, 4), dtype=bool)
    repeat_valid = np.asarray([[True, True, True, True], [True, True, True, False]], dtype=bool)
    primary = PanelData(
        panel="primary",
        model_ids=model_ids,
        model_names=("Complete", "Filtered"),
        slot_ids=("a", "b"),
        task_ids=primary_ids,
        families=families,
        scores=np.asarray([[80.0] * 4, [70.0] * 4]),
        completed=primary_valid,
        parseable=primary_valid,
        selections=tuple(tuple("ABC" for _ in families) for _ in model_ids),
        response_artifact_sha256s=tuple(),
        spend_micros=0,
    )
    repeat = PanelData(
        panel="repeat",
        model_ids=model_ids,
        model_names=("Complete", "Filtered"),
        slot_ids=("a", "b"),
        task_ids=repeat_ids,
        families=families,
        scores=np.asarray([[80.0] * 4, [70.0] * 4]),
        completed=repeat_valid,
        parseable=repeat_valid,
        selections=tuple(tuple("ABC" for _ in families) for _ in model_ids),
        response_artifact_sha256s=tuple(),
        spend_micros=0,
    )
    choices = {label: f"ingredient-{label}" for label in "ABCDEFGH"}
    taskset = {
        "tasks": [
            {"task_id": task_id, "family": family, "choices": choices}
            for task_id, family in zip(primary_ids, families, strict=True)
        ]
    }
    repeat_panel = {
        "tasks": [
            {
                "task_id": repeat_id,
                "original_task_id": primary_id,
                "family": family,
                "choices": choices,
            }
            for repeat_id, primary_id, family in zip(repeat_ids, primary_ids, families, strict=True)
        ]
    }

    rows = analyze_repeatability(
        primary=primary,
        repeat=repeat,
        taskset=taskset,
        repeat_panel=repeat_panel,
        bootstrap_resamples=50,
        seed=31,
    )

    assert rows[0]["repeatability_status"] == "estimated_equal_family_macro"
    assert rows[0]["mean_ingredient_set_jaccard"] == 1.0
    assert rows[1]["repeatability_status"] == "not_estimable_missing_family_pairs"
    assert rows[1]["valid_pairs_per_family"]["cultural_composition"] == 0
    assert rows[1]["mean_ingredient_set_jaccard"] is None
    assert rows[1]["jaccard_pointwise_95_ci"] is None
    assert rows[1]["exact_ingredient_set_match_rate"] is None
    assert rows[1]["mean_absolute_score_difference"] is None

    leaderboard = _leaderboard_csv(
        {
            "models": [
                {
                    "point_estimate_rank": 1,
                    "statistical_rank_group": 1,
                    "model_name": "Filtered",
                    "model_id": "filtered",
                    "flavourbench_score": 70.0,
                    "score_simultaneous_95_ci": [65.0, 75.0],
                    "coverage": {
                        "valid_scored": 4,
                        "scheduled": 4,
                        "valid_scored_rate": 1.0,
                        "completed": 4,
                        "parseable": 4,
                    },
                    "failure_exclusion_sensitivity": {
                        "scheduled_panel_worst_best_bounds": [70.0, 70.0]
                    },
                    "repeatability": rows[1],
                }
            ]
        }
    ).decode("utf-8")
    assert leaderboard.splitlines()[1].endswith(",")
