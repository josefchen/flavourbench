from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "paper/generated/complete-core"
ANALYSIS = DIRECTORY / "complete-core-selection-robustness.json"
BUILDER = ROOT / "paper/build_selection_robustness_assets.py"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def test_selection_robustness_artifact_is_bound_to_its_builder() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = dict(analysis)
    recorded = payload.pop("artifact_sha256")
    assert recorded == hashlib.sha256(_canonical(payload)).hexdigest()
    assert analysis["schema_version"] == "flavourbench-selection-robustness-v1"
    assert analysis["status"] == "post_collection_robustness_analysis"
    assert (
        analysis["inputs"]["builder_physical_sha256"]
        == hashlib.sha256(BUILDER.read_bytes()).hexdigest()
    )
    assert analysis["design"] == {
        "models": 27,
        "candidate_tasks_per_panel": 640,
        "included_families": ["substitution", "pairing", "constraint"],
        "selected_tasks_per_family_panel": 89,
        "primary_selected_tasks": 534,
        "selection_rule": "same fixed SHA-256 order as the primary complete-core plan",
    }


def test_leave_one_model_out_rows_and_companion_are_complete() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = analysis["leave_one_model_out"]
    assert len(rows) == 27
    assert len({row["omitted_model_id"] for row in rows}) == 27
    assert all(0 < int(row["selected_task_overlap"]) <= 534 for row in rows)
    assert all(0.0 <= float(row["pair_order_agreement"]) <= 1.0 for row in rows)
    most_influential = min(
        rows,
        key=lambda row: (
            float(row["selected_task_overlap_fraction"]),
            row["omitted_model_id"],
        ),
    )
    assert analysis["most_influential_omission"] == most_influential

    companion = list(
        csv.DictReader(
            io.StringIO(
                (DIRECTORY / "complete-core-leave-one-model-out.csv").read_text(encoding="utf-8")
            )
        )
    )
    assert len(companion) == 27
    assert [row["omitted_model_id"] for row in companion] == [
        row["omitted_model_id"] for row in rows
    ]


def test_score_and_selection_sensitivity_claims_match_records() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    metrics = analysis["score_definition_sensitivity"]
    assert [row["metric"] for row in metrics] == [
        "flavourbench_score",
        "chance_adjusted_gain",
        "action_percentile",
        "exact_optimum_rate",
    ]
    alternatives = metrics[1:]
    assert min(float(row["rank_spearman_with_flavourbench"]) for row in alternatives) > 0.93
    assert min(float(row["pair_order_agreement_with_flavourbench"]) for row in alternatives) > 0.89
    profile = analysis["selection_profile"]
    assert profile["random_subset_hypotheses"] == 42
    assert profile["random_subset_draws_per_stratum"] == 20_000
    assert profile["random_subset_holm_resolved_characteristics"] == 0
    weights = analysis["family_weight_sensitivity"]
    assert weights["grid_points"] == 696
    assert float(weights["rank_spearman"]["minimum"]) > 0.98
    assert "do not establish external culinary validity" in analysis["claim_boundary"]
