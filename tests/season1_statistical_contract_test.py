from __future__ import annotations

import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from flavourbench.main import _require_season1_statistical_acceptance
from flavourbench.models import Season
from flavourbench.season1_statistics import (
    FAMILIES,
    ArenaObservation,
    StatisticalContractError,
    UpliftJudgment,
    UpliftScheduledPair,
    analyze_controlled_arena,
    analyze_controlled_uplift,
)


def _arena_panel(
    *, tasks_per_family: int = 8, battles_per_task: int = 2
) -> tuple[list[ArenaObservation], list[str]]:
    roster = ["model-a", "model-b", "model-c", "model-d"]
    matchups = list(itertools.combinations(roster, 2))
    rows: list[ArenaObservation] = []
    ordinal = 0
    for family_index, family in enumerate(FAMILIES):
        for task_index in range(tasks_per_family):
            task_id = f"{family}-task-{task_index}"
            for repetition in range(battles_per_task):
                first, second = matchups[(task_index * 2 + repetition) % len(matchups)]
                outcome = (1.0, 0.5, 0.0)[
                    (task_index + repetition + family_index) % 3
                ]
                battle_id = f"{task_id}-battle-{repetition}"
                for rater in range(2):
                    rows.append(
                        ArenaObservation(
                            observation_id=f"arena-{ordinal}",
                            task_id=task_id,
                            family=family,
                            battle_id=battle_id,
                            rater_id=f"rater-{rater}",
                            model_a=first,
                            model_b=second,
                            response_a_id=f"{task_id}-response-{first}",
                            response_b_id=f"{task_id}-response-{second}",
                            outcome=outcome,
                        )
                    )
                    ordinal += 1
    return rows, roster


def _postcollection_audit() -> dict[str, object]:
    records = [
        {
            "task_id": f"audit-task-{index}",
            "task_content_sha256": hashlib.sha256(
                f"audit-task-{index}".encode()
            ).hexdigest(),
            "selection_reasons": ["random"],
            "auditor_commitments_sha256": [
                hashlib.sha256(f"auditor-a-{index}".encode()).hexdigest(),
                hashlib.sha256(f"auditor-b-{index}".encode()).hexdigest(),
            ],
            "material_defect": False,
            "resolution_status": "no_material_defect",
        }
        for index in range(60)
    ]
    payload: dict[str, object] = {
        "schema_version": "flavourbench-season1-post-collection-item-audit-v1",
        "status": "complete",
        "study_design_artifact_sha256": (
            "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"
        ),
        "synthetic_observations": 0,
        "sampling_seed_committed_before_model_results": True,
        "original_task_roles_excluded": True,
        "affected_snapshots_recomputed": True,
        "counts": {
            "population_tasks": 240,
            "random_tasks_audited": 60,
            "anomaly_flagged_tasks": 0,
            "anomaly_flagged_tasks_audited": 0,
            "unique_tasks_audited": 60,
            "minimum_independent_auditors_per_task": 2,
            "confirmed_material_defects": 0,
            "retired_material_defects": 0,
            "unresolved_material_defects": 0,
        },
        "task_records": records,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _uplift_panel() -> tuple[list[UpliftJudgment], list[UpliftScheduledPair]]:
    judgments: list[UpliftJudgment] = []
    schedule: list[UpliftScheduledPair] = []
    choices = {
        "composition": "epicure_win",
        "cookability": "tie",
        "evidence": "unaided_win",
        "substitution": "epicure_win",
    }
    for family in FAMILIES:
        for task_index in range(50):
            task_id = f"{family}-task-{task_index}"
            pair_id = f"model-a-{task_id}-pair"
            schedule.append(
                UpliftScheduledPair(
                    pair_id=pair_id,
                    task_id=task_id,
                    family=family,
                    model_id="model-a",
                    repetition_index=1,
                    epicure_valid=True,
                    unaided_valid=True,
                )
            )
            judgments.append(
                UpliftJudgment(
                    judgment_id=f"judgment-{pair_id}",
                    task_id=task_id,
                    family=family,
                    battle_id=pair_id,
                    rater_id="rater-primary",
                    model_id="model-a",
                    choice=choices[family],  # type: ignore[arg-type]
                )
            )
    return judgments, schedule


def test_isolated_frozen_model_withholds_every_cross_model_rating() -> None:
    observations, roster = _arena_panel()
    result = analyze_controlled_arena(
        observations,
        [*roster, "isolated-model"],
        bootstrap_replicates=25,
        postcollection_item_audit=_postcollection_audit(),
    )

    assert result["ranking_status"] == "withheld_insufficient_task_clusters"
    assert result["statistical_acceptance"]["status"] == "fail"  # type: ignore[index]
    assert all(row["rating"] is None for row in result["rows"])  # type: ignore[union-attr]
    assert ["isolated-model"] in result["comparison_components"]
    deficit_codes = {
        item["code"] for item in result["statistical_acceptance"]["deficits"]
    }
    assert "admitted_scored_tasks_per_family_below_minimum" in deficit_codes
    assert "unique_task_clusters_per_model_family_below_minimum" in deficit_codes
    assert "family_comparison_graph_disconnected" in deficit_codes


def test_arena_is_invariant_to_side_swap_and_within_battle_duplication() -> None:
    observations, roster = _arena_panel(tasks_per_family=40)
    swapped = [
        ArenaObservation(
            observation_id=row.observation_id,
            task_id=row.task_id,
            family=row.family,
            battle_id=row.battle_id,
            rater_id=row.rater_id,
            model_a=row.model_b,
            model_b=row.model_a,
            response_a_id=row.response_b_id,
            response_b_id=row.response_a_id,
            outcome=1.0 - row.outcome,
        )
        for row in observations
    ]
    duplicated = [
        *observations,
        *[
            ArenaObservation(
                observation_id=f"duplicate-{row.observation_id}",
                task_id=row.task_id,
                family=row.family,
                battle_id=row.battle_id,
                rater_id=row.rater_id,
                model_a=row.model_a,
                model_b=row.model_b,
                response_a_id=row.response_a_id,
                response_b_id=row.response_b_id,
                outcome=row.outcome,
            )
            for row in observations
        ],
    ]

    original = analyze_controlled_arena(
        observations,
        roster,
        bootstrap_replicates=120,
        seed=17,
        postcollection_item_audit=_postcollection_audit(),
    )
    side_swapped = analyze_controlled_arena(
        swapped,
        roster,
        bootstrap_replicates=120,
        seed=17,
        postcollection_item_audit=_postcollection_audit(),
    )
    repeated = analyze_controlled_arena(
        duplicated,
        roster,
        bootstrap_replicates=120,
        seed=17,
        postcollection_item_audit=_postcollection_audit(),
    )
    def by_model(result):
        return {row["competitor_id"]: row for row in result["rows"]}

    for comparison in (side_swapped, repeated):
        for model_id, row in by_model(original).items():
            observed = by_model(comparison)[model_id]
            assert observed["rating"] == pytest.approx(row["rating"], abs=1e-5)
            assert observed["rating_lower"] == pytest.approx(row["rating_lower"], abs=1e-5)
            assert observed["rating_upper"] == pytest.approx(row["rating_upper"], abs=1e-5)

    evidence = original["evidence_units"]
    assert evidence["raw_preference_rows"] > evidence["unique_response_arms"]
    assert evidence["comparison_rows_treated_as_independent"] is False
    assert evidence["primary_resampling_unit"] == "task_cluster"


def test_arena_rejects_response_reuse_across_task_clusters() -> None:
    observations, roster = _arena_panel()
    first = observations[0]
    conflicting = ArenaObservation(
        observation_id="cross-task-response-reuse",
        task_id="composition-other-task",
        family="composition",
        battle_id="other-battle",
        rater_id="rater-other",
        model_a=first.model_a,
        model_b=first.model_b,
        response_a_id=first.response_a_id,
        response_b_id="other-response-b",
        outcome=1.0,
    )

    with pytest.raises(StatisticalContractError, match="reused across task"):
        analyze_controlled_arena(
            [*observations, conflicting],
            roster,
            bootstrap_replicates=5,
        )


def test_arena_withholds_when_comparisons_lack_two_independent_raters() -> None:
    observations, roster = _arena_panel(tasks_per_family=40)
    one_rater = [row for row in observations if row.rater_id == "rater-0"]
    result = analyze_controlled_arena(
        one_rater,
        roster,
        bootstrap_replicates=5,
        postcollection_item_audit=_postcollection_audit(),
    )
    assert result["ranking_status"] == "withheld_insufficient_task_clusters"
    assert all(row["rating"] is None for row in result["rows"])
    deficits = result["statistical_acceptance"]["deficits"]
    rater_deficit = next(
        item
        for item in deficits
        if item["code"]
        == "distinct_independent_raters_per_comparison_below_minimum"
    )
    assert rater_deficit["minimum_observed"] == 1
    assert rater_deficit["required"] == 2


def test_arena_rejects_rater_ids_not_present_in_preference_rows() -> None:
    observations, roster = _arena_panel(tasks_per_family=40)
    one_rater = [row for row in observations if row.rater_id == "rater-0"]
    supplied: dict[str, set[str]] = {}
    for row in one_rater:
        supplied.setdefault(row.battle_id, set()).add(row.rater_id)
    for raters in supplied.values():
        raters.add("injected-rater")

    with pytest.raises(StatisticalContractError, match="does not match"):
        analyze_controlled_arena(
            one_rater,
            roster,
            bootstrap_replicates=5,
            comparison_raters={
                battle_id: sorted(raters)
                for battle_id, raters in supplied.items()
            },
            postcollection_item_audit=_postcollection_audit(),
        )


def test_pairwise_interval_is_suppressed_below_shared_task_floor() -> None:
    observations, roster = _arena_panel(tasks_per_family=40)
    filtered = [
        row
        for row in observations
        if {row.model_a, row.model_b} != {"model-a", "model-b"}
        or row.task_id.endswith("task-0")
    ]
    result = analyze_controlled_arena(
        filtered,
        roster,
        bootstrap_replicates=20,
        seed=9,
        postcollection_item_audit=_postcollection_audit(),
    )
    assert result["ranking_status"] == "estimated"
    support = result["pairwise_reporting_support"]["model-a"]["model-b"]
    assert support["shared_task_clusters"] == 4
    assert support["minimum_for_interval"] == 10
    assert support["interval_reportable"] is False
    assert result["pairwise_win_probability_interval"]["model-a"]["model-b"] is None


def test_family_view_uses_analogous_task_cluster_gate() -> None:
    observations, roster = _arena_panel(
        tasks_per_family=40,
        battles_per_task=6,
    )
    composition = [row for row in observations if row.family == "composition"]
    result = analyze_controlled_arena(
        composition,
        roster,
        view="composition",
        bootstrap_replicates=20,
        seed=11,
        postcollection_item_audit=_postcollection_audit(),
    )
    assert result["ranking_status"] == "estimated"
    assert result["statistical_acceptance"]["view"] == "composition"
    assert result["statistical_acceptance"]["deficits"] == []
    assert result["family_bootstrap_connected_rates"] == {"composition": 1.0}


def test_uplift_is_family_standardized_and_immune_to_one_family_row_duplication() -> None:
    judgments, schedule = _uplift_panel()
    duplicated = [
        *judgments,
        *[
            UpliftJudgment(
                judgment_id=f"duplicate-{row.judgment_id}",
                task_id=row.task_id,
                family=row.family,
                battle_id=row.battle_id,
                rater_id=row.rater_id,
                model_id=row.model_id,
                choice=row.choice,
            )
            for row in judgments
            if row.family == "composition"
        ],
    ]
    original = analyze_controlled_uplift(
        judgments,
        schedule,
        ["model-a"],
        bootstrap_replicates=120,
        seed=23,
    )
    repeated = analyze_controlled_uplift(
        duplicated,
        schedule,
        ["model-a"],
        bootstrap_replicates=120,
        seed=23,
    )
    first = original["rows"][0]  # type: ignore[index]
    second = repeated["rows"][0]  # type: ignore[index]

    assert first["epicure_win_share"] == pytest.approx(0.625)
    assert second["epicure_win_share"] == first["epicure_win_share"]
    assert second["interval_lower"] == first["interval_lower"]
    assert second["interval_upper"] == first["interval_upper"]


def test_uplift_schedule_retains_failures_both_bad_and_missingness_bounds() -> None:
    judgments, schedule = _uplift_panel()
    first_pair = schedule[0]
    second_pair = schedule[1]
    third_pair = schedule[2]
    schedule = [
        *[
            UpliftScheduledPair(
                pair_id=row.pair_id,
                task_id=row.task_id,
                family=row.family,
                model_id=row.model_id,
                repetition_index=row.repetition_index,
                epicure_valid=(False if row.pair_id == first_pair.pair_id else row.epicure_valid),
                unaided_valid=(False if row.pair_id == second_pair.pair_id else row.unaided_valid),
            )
            for row in schedule
        ]
    ]
    judgments = [
        row
        for row in judgments
        if row.battle_id not in {first_pair.pair_id, second_pair.pair_id, third_pair.pair_id}
    ]
    judgments.append(
        UpliftJudgment(
            judgment_id="both-bad-observation",
            task_id=third_pair.task_id,
            family=third_pair.family,
            battle_id=third_pair.pair_id,
            rater_id="rater-primary",
            model_id="model-a",
            choice="both_bad",
        )
    )
    result = analyze_controlled_uplift(
        judgments,
        schedule,
        ["model-a"],
        bootstrap_replicates=50,
        seed=29,
    )
    row = result["rows"][0]  # type: ignore[index]

    assert row["epicure_only_valid_pairs"] == 1
    assert row["unaided_only_valid_pairs"] == 1
    assert row["judged_both_bad_pairs"] == 1
    assert row["missing_preference_pairs"] == 2
    assert row["missingness_lower"] < row["epicure_win_share"] <= row["missingness_upper"]


def test_statistical_artifacts_are_byte_deterministic_for_canonical_inputs() -> None:
    observations, roster = _arena_panel(tasks_per_family=40)
    first = analyze_controlled_arena(
        observations,
        roster,
        bootstrap_replicates=40,
        seed=31,
        postcollection_item_audit=_postcollection_audit(),
    )
    second = analyze_controlled_arena(
        list(reversed(observations)),
        list(reversed(roster)),
        bootstrap_replicates=40,
        seed=31,
        postcollection_item_audit=_postcollection_audit(),
    )

    assert first == second


def test_season1_publication_fails_closed_without_accepted_nonprovisional_statistics() -> None:
    season = Season(slug="season-1", name="Season 1", epicure_release_id="release")
    snapshot = SimpleNamespace(category="all", cohort="public", track="epicure_uplift")
    incomplete = {
        "ranking_status": "estimated",
        "bootstrap_replicates": 5_000,
        "statistical_acceptance": {"status": "pass"},
        "rows": [{"competitor_id": "model-a", "provisional": True}],
    }

    with pytest.raises(HTTPException, match="no provisional model rows"):
        _require_season1_statistical_acceptance(season, snapshot, incomplete)  # type: ignore[arg-type]

    accepted = {
        **incomplete,
        "rows": [{"competitor_id": "model-a", "provisional": False}],
    }
    _require_season1_statistical_acceptance(season, snapshot, accepted)  # type: ignore[arg-type]
