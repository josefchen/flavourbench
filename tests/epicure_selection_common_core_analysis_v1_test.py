from __future__ import annotations

import numpy as np
import pytest

from flavourbench.epicure_selection_common_core_analysis_v1 import (
    analyze_complete_common_core,
    anchor_cluster_bootstrap,
    anchor_cluster_sign_flip,
    equal_family_mean,
)
from flavourbench.epicure_selection_powered_analysis import PanelData

FAMILIES = ("substitution", "pairing", "constraint")
TASK_FAMILIES = (
    "substitution",
    "pairing",
    "constraint",
    "substitution",
    "pairing",
    "constraint",
)
CLUSTERS = ("a", "b", "c", "d", "e", "f")


def _panel() -> PanelData:
    scores = np.asarray(
        [
            [90.0, 88.0, 86.0, 92.0, 90.0, 88.0],
            [70.0, 68.0, 66.0, 72.0, 70.0, 68.0],
            [50.0, 48.0, 46.0, 52.0, 50.0, 48.0],
        ]
    )
    complete = np.ones_like(scores, dtype=bool)
    return PanelData(
        panel="primary",
        model_ids=("model/a", "model/b", "model/c"),
        model_names=("A", "B", "C"),
        slot_ids=("a", "b", "c"),
        task_ids=tuple(f"task-{index}" for index in range(6)),
        families=TASK_FAMILIES,
        scores=scores,
        completed=complete.copy(),
        parseable=complete.copy(),
        selections=tuple(tuple("ABC" for _ in range(6)) for _ in range(3)),
        response_artifact_sha256s=tuple(f"{index:064x}" for index in range(18)),
        spend_micros=123,
    )


def test_complete_common_core_family_mean_and_cluster_resampling() -> None:
    panel = _panel()
    point = equal_family_mean(panel.scores, panel.families, FAMILIES)
    assert point.tolist() == [89.0, 69.0, 49.0]
    bootstrap = anchor_cluster_bootstrap(
        panel.scores,
        panel.families,
        FAMILIES,
        CLUSTERS,
        resamples=199,
        seed=7,
    )
    assert bootstrap.shape == (199, 3)
    observed, pvalues = anchor_cluster_sign_flip(
        panel.scores[0:1] - panel.scores[1:2],
        panel.families,
        FAMILIES,
        CLUSTERS,
        resamples=499,
        seed=8,
    )
    assert observed.tolist() == pytest.approx([20.0])
    assert pvalues.shape == (1,)


def test_complete_common_core_analysis_reports_every_model_and_pair() -> None:
    panel = _panel()
    taskset = {
        "tasks": [{"task_id": task_id, "chance_score_bps": 4000} for task_id in panel.task_ids]
    }
    plan = {
        "artifact_sha256": "f" * 64,
        "common_core": {"estimand_label": "test common core"},
        "inference": {
            "bootstrap_resamples": 199,
            "permutation_resamples": 499,
            "seed": 11,
        },
    }
    analysis = analyze_complete_common_core(
        primary=panel,
        taskset=taskset,
        plan=plan,
        family_order=FAMILIES,
        cluster_ids=CLUSTERS,
        panel_ids=("panel_1", "panel_1", "panel_1", "panel_2", "panel_2", "panel_2"),
    )
    assert analysis["status"] == "final_complete_common_core"
    assert analysis["dnf_rows_emitted"] is False
    assert len(analysis["models"]) == 3
    assert len(analysis["pairwise_comparisons"]) == 3
    assert analysis["response_artifact_count"] == 18
    assert all(row["coverage"]["valid_scored_rate"] == 1.0 for row in analysis["models"])
    assert [row["point_estimate_rank"] for row in analysis["models"]] == [1, 2, 3]
    assert analysis["models"][0]["flavourbench_score"] == 89.0
    assert analysis["panel_replication"]["score_pearson"] > 0.99
