from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
ANALYSIS = REPOSITORY / "paper/generated/complete-core/complete-core-stability-analysis.json"
DATASET_ANALYSIS = REPOSITORY / "hf/dataset/data-analysis"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def test_stability_analysis_is_semantically_bound() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    recorded = analysis.pop("artifact_sha256")
    assert hashlib.sha256(_canonical(analysis)).hexdigest() == recorded
    assert recorded == "4b359ac51db465c7a3f49fb5567a624b1ce3ad6280d309f31545e17ff2797026"
    assert analysis["status"] == "retrospective_precision_and_stability_analysis"
    assert analysis["design"] == {
        "family_panel_strata": 6,
        "models": 27,
        "reference": "complete_534_task_point_order",
        "sampling": "without_replacement_within_each_family_panel_stratum",
        "tasks": 534,
        "tasks_per_stratum": 89,
    }


def test_task_count_curve_and_generalizability_are_consistent() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    rows = {row["tasks"]: row for row in analysis["task_count_stability"]}
    assert list(rows) == [30, 60, 90, 150, 270, 534]
    assert rows[270]["metrics"]["rank_spearman"]["median"] == pytest.approx(0.9523809523809523)
    assert rows[270]["metrics"]["top_1_preserved"]["mean"] == pytest.approx(0.4606)
    assert rows[534]["metrics"]["rank_spearman"]["median"] == 1.0
    assert rows[534]["metrics"]["mean_absolute_score_error"]["median"] == pytest.approx(
        0.0, abs=1e-12
    )
    variance = analysis["variance_partition"]
    assert variance["relative_decision_generalizability_at_534_tasks"] == pytest.approx(
        0.935995360066749
    )
    assert variance["estimated_balanced_tasks_for_relative_g_0_90"] == 329
    assert sum(
        row["fraction_of_total"] for row in variance["sum_squares_partition"]
    ) == pytest.approx(1.0)


def test_space_interface_requires_the_bound_stability_analysis() -> None:
    app = (REPOSITORY / "hf/space/app.py").read_text()
    builder = (REPOSITORY / "hf/space/build_complete_core_space_bundle.py").read_text()
    assert 'STABILITY = BUNDLE["stability_analysis"]' in app
    assert '"stability_analysis": stability' in builder
    assert '"flavourbench-task-count-stability-v1"' in builder


def test_dataset_view_tables_are_flat_and_bound_to_the_analysis() -> None:
    expected_hash = "4b359ac51db465c7a3f49fb5567a624b1ce3ad6280d309f31545e17ff2797026"
    stability = [
        json.loads(line)
        for line in (DATASET_ANALYSIS / "task_count_stability.jsonl").read_text().splitlines()
    ]
    variance = [
        json.loads(line)
        for line in (DATASET_ANALYSIS / "variance_partition.jsonl").read_text().splitlines()
    ]
    assert [row["tasks"] for row in stability] == [30, 60, 90, 150, 270, 534]
    assert {row["analysis_artifact_sha256"] for row in stability} == {expected_hash}
    assert {row["component"] for row in variance} == {
        "model",
        "family",
        "panel",
        "family_by_panel",
        "task_within_family_panel",
        "model_by_task",
    }
    assert {row["analysis_artifact_sha256"] for row in variance} == {expected_hash}
