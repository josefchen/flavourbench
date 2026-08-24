from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from flavourbench import task_campaign_sampling_power_validation_v1 as validation
from flavourbench.sampling_power_engine_v1 import (
    DEFAULT_SCENARIOS,
    FAMILIES,
    _fit_arena,
    analytic_power_curves,
    build_frame_spec,
    evaluate_scenario,
    frame_diagnostics,
)
from flavourbench.season1_statistics import (
    ArenaObservation,
    _hierarchical_weights,
    _numpy_bt_refit,
)
from flavourbench.task_campaign_human_sampling_successor_v2 import (
    materialize_sampling_frame_v2,
)


@pytest.fixture(scope="module")
def frame():
    source = json.loads(validation.DEFAULT_SAMPLING_V2.read_text(encoding="utf-8"))
    return build_frame_spec(materialize_sampling_frame_v2(source))


@pytest.fixture(scope="module")
def small_artifact():
    return validation.build_validation_artifact(datasets=20, seed=77)


def test_exact_bound_frame_and_frozen_policy_deficits(frame) -> None:
    diagnostics = frame_diagnostics(frame)

    assert diagnostics["roster_size"] == 14
    assert diagnostics["tasks_per_family"] == {
        "substitution": 20,
        "composition": 20,
        "cookability": 20,
        "evidence": 20,
    }
    assert diagnostics["arena_comparisons"] == 800
    assert diagnostics["uplift_comparisons"] == 800
    assert diagnostics["primary_rating_slots"] == 3200
    assert diagnostics["repeat_presentations"] == 400
    assert diagnostics["repeat_rate_of_primary_slots"] == 0.125
    assert diagnostics["arena_support"] == {
        "minimum_shared_task_clusters_per_pair": 8,
        "maximum_shared_task_clusters_per_pair": 9,
        "minimum_global_comparisons_per_model": 114,
        "maximum_global_comparisons_per_model": 115,
        "minimum_family_comparisons_per_model": 28,
        "maximum_family_comparisons_per_model": 29,
        "minimum_unique_task_clusters_per_model_family": 20,
        "maximum_unique_task_clusters_per_model_family": 20,
    }


def test_small_build_is_deterministic_strict_and_never_a_power_pass(small_artifact) -> None:
    repeated = validation.build_validation_artifact(datasets=20, seed=77)

    assert repeated == small_artifact
    body = {key: value for key, value in small_artifact.items() if key != "artifact_sha256"}
    assert small_artifact["artifact_sha256"] == validation.sha256_json(body)
    assert small_artifact["status"].startswith("blocked_no_go")
    assert small_artifact["conclusions"]["power_validated"] is False
    assert small_artifact["prespecified_candidate_acceptance"]["status"] == "fail"
    assert small_artifact["old_frozen_arena_v1_reconciliation"]["status"] == (
        "non_transferable_and_current_frame_fails_frozen_v1"
    )
    assert not any(
        value for value in small_artifact["claim_boundary"].values() if isinstance(value, bool)
    )


def test_null_and_calibrated_paths_report_mc_error_and_use_task_clusters(frame) -> None:
    null = evaluate_scenario(frame, DEFAULT_SCENARIOS[0], datasets=40, seed=101)
    calibrated = evaluate_scenario(frame, DEFAULT_SCENARIOS[2], datasets=40, seed=102)

    assert null["overall_uplift"]["two_sided_type_i_error"] is not None
    assert null["overall_uplift"]["type_i_error_mcse"] is not None
    assert calibrated["overall_uplift"]["one_sided_power"] is not None
    assert calibrated["overall_uplift"]["power_mcse"] is not None
    assert calibrated["overall_uplift"]["one_sided_power"] > 0.5
    assert calibrated["data_generating_parameters"]["tie_probability"] == 0.2
    assert frame.task_count == 80
    assert frame.primary_slots == 3200


def test_per_model_planning_curve_uses_one_comparison_per_model_task(frame) -> None:
    planning = analytic_power_curves(frame)
    assumptions = planning["assumptions"]
    curve = planning["curves"]["one_model_uplift_minimum_task_clusters_bonferroni_roster"]
    power_at_008 = next(row["power"] for row in curve if row["effect"] == 0.08)

    assert assumptions["minimum_comparisons_per_model_task"] == 1
    assert assumptions["minimum_ratings_per_model_task"] == 2
    assert assumptions["model_task_cluster_sd"] == pytest.approx(0.29664794)
    assert power_at_008 == pytest.approx(0.165358)
    assert power_at_008 < 0.20


def test_arena_proxy_point_fit_matches_production_hierarchical_weights_with_missing_battles(
    frame,
) -> None:
    rng = np.random.default_rng(2026080901)
    values = rng.choice([0.0, 0.5, 1.0], size=len(frame.arena_task), p=[0.4, 0.2, 0.4])
    observed = np.ones(len(values), dtype=bool)
    first_task = np.flatnonzero(frame.arena_task == 0)
    observed[first_task[1:]] = False
    comparison_values = np.where(observed, values, np.nan)

    fitted = _fit_arena(frame, comparison_values)
    assert fitted is not None
    proxy_merits = fitted[0]
    production_rows = [
        ArenaObservation(
            observation_id=f"observation-{index}",
            task_id=str(int(frame.arena_task[index])),
            family=FAMILIES[int(frame.arena_family[index])],
            battle_id=frame.arena_comparison_ids[index],
            rater_id="comparison-aggregated",
            model_a=frame.model_ids[int(frame.arena_first[index])],
            model_b=frame.model_ids[int(frame.arena_second[index])],
            response_a_id=f"response-a-{index}",
            response_b_id=f"response-b-{index}",
            outcome=float(values[index]),
        )
        for index in np.flatnonzero(observed)
    ]
    production_ratings = _numpy_bt_refit(
        production_rows,
        frame.model_ids,
        _hierarchical_weights(production_rows),
    )
    production_merits = np.asarray(
        [
            (production_ratings[model_id] - 1000.0) * math.log(10.0) / 400.0
            for model_id in frame.model_ids
        ]
    )

    assert np.max(np.abs(proxy_merits - production_merits)) < 1e-7


def test_exact_binomial_acceptance_bounds_do_not_degenerate_at_boundaries() -> None:
    one_lower, one_upper = validation._exact_binomial_rate_bounds(1.0, 500)
    zero_lower, zero_upper = validation._exact_binomial_rate_bounds(0.0, 500)

    assert 0.99 < one_lower < 1.0
    assert one_upper == 1.0
    assert zero_lower == 0.0
    assert 0.0 < zero_upper < 0.01


def test_contract_separates_production_bt_from_diagnostic_proxy(small_artifact) -> None:
    simulation = small_artifact["simulation_contract"]
    estimand = small_artifact["prespecified_estimands_and_inference"]["primary"][
        "arena_model_ranking"
    ]

    assert simulation["nested_production_bootstrap_run"] is False
    assert simulation["diagnostic_arena_proxy_only"] is True
    assert simulation["production_method_validation_claimed"] is False
    assert "Bradley-Terry" in estimand["production_estimand"]
    assert "crossed rater-cluster bootstrap" in estimand["production_interval"]
    assert estimand["current_status"].startswith("withheld")


def test_full_v5_layout_is_recommended_but_not_declared_validated(small_artifact) -> None:
    comparison = small_artifact["reduced_vs_full_benchmark_comparison"]
    full = comparison["design_options"]["full_v5_k16_primary_layout"]
    workload = comparison["full_layout_human_rating_workload"]
    gate = comparison["production_bt_bootstrap_validation"]

    assert full["scored_task_clusters"] == 160
    assert full["arena_comparisons"] == 3200
    assert full["uplift_comparisons"] == 3200
    assert full["production_structure_matches_frozen_numeric_gates"] is True
    assert full["production_power_validated"] is False
    assert workload == {
        "unique_arena_comparisons": 3200,
        "unique_uplift_comparisons": 3200,
        "unique_comparisons": 6400,
        "distinct_raters_per_comparison": 2,
        "primary_rating_presentations": 12800,
        "concealed_repeat_rate_of_primary_ratings": 0.125,
        "concealed_repeat_presentations": 1600,
        "total_rating_presentations": 14400,
        "rating_hours": {
            "at_3_minutes_each": 720,
            "at_4_minutes_each": 960,
            "at_5_minutes_each": 1200,
            "at_8_minutes_each": 1920,
        },
        "authorization_or_cost_claim": False,
    }
    assert gate["nominal_total_bootstrap_refits"] == 80_000_000
    assert gate["can_be_validated"] is True
    assert gate["validated_now"] is False


def test_truncated_requested_digest_is_not_silently_accepted(small_artifact) -> None:
    reconciliation = small_artifact["physical_digest_reconciliation"]

    assert len(validation.REQUESTED_TRUNCATED_PHYSICAL_LITERAL) == 63
    assert reconciliation["is_valid_sha256"] is False
    assert reconciliation["normalization_or_silent_acceptance"] is False
    assert reconciliation["verified_complete_physical_sha256"].endswith("f")
    assert (
        hashlib.sha256(validation.DEFAULT_SAMPLING_V2.read_bytes()).hexdigest()
        == (reconciliation["verified_complete_physical_sha256"])
    )


def test_source_byte_change_fails_before_simulation(tmp_path: Path) -> None:
    changed = tmp_path / "sampling.json"
    changed.write_bytes(validation.DEFAULT_SAMPLING_V2.read_bytes() + b"\n")

    with pytest.raises(validation.SamplingPowerValidationError, match="physical digest"):
        validation.build_validation_artifact(
            datasets=20,
            seed=1,
            sampling_v2_path=changed,
        )


def test_verifier_rejects_relabelled_no_go_without_rebuilding(small_artifact) -> None:
    tampered = deepcopy(small_artifact)
    tampered["conclusions"]["power_validated"] = True

    with pytest.raises(validation.SamplingPowerValidationError, match="semantic digest"):
        validation.verify_validation_artifact(tampered)


def test_writer_uses_no_replace_hard_link_and_is_idempotent(
    small_artifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "verify_validation_artifact", lambda _document: None)
    first = validation.write_validation_artifact(small_artifact, tmp_path)
    first_inode = first.stat().st_ino
    second = validation.write_validation_artifact(small_artifact, tmp_path)

    assert second == first
    assert second.stat().st_ino == first_inode
    assert json.loads(first.read_text(encoding="utf-8")) == small_artifact
    assert not list(tmp_path.glob(".sampling-power-v1-*"))


def test_writer_rejects_existing_conflict_and_racing_final(
    small_artifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "verify_validation_artifact", lambda _document: None)
    digest = small_artifact["artifact_sha256"]
    destination = tmp_path / f"sampling-power-validation-v1-candidate-{digest}.json"
    destination.write_text("conflict", encoding="utf-8")
    with pytest.raises(validation.SamplingPowerValidationError, match="conflicts"):
        validation.write_validation_artifact(small_artifact, tmp_path)

    destination.unlink()
    real_link = os.link

    def racing_link(source, final, *, follow_symlinks=True):
        Path(final).write_text("racing final", encoding="utf-8")
        raise FileExistsError(final)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(validation.SamplingPowerValidationError, match="appeared"):
        validation.write_validation_artifact(small_artifact, tmp_path)
    monkeypatch.setattr(os, "link", real_link)

    assert destination.read_text(encoding="utf-8") == "racing final"
    assert not list(tmp_path.glob(".sampling-power-v1-*"))
