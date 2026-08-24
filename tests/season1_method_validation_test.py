from __future__ import annotations

from copy import deepcopy

import pytest

from flavourbench.season1_method_validation import (
    MethodValidationError,
    OrdinalScenario,
    build_artifact,
    canonical_sha256,
    evaluate_scenario,
    verify_artifact,
    write_artifact,
)


def test_small_method_validation_run_is_deterministic_and_non_scored() -> None:
    first = build_artifact(datasets=40, bootstrap_replicates=49, seed=17)
    second = build_artifact(datasets=40, bootstrap_replicates=49, seed=17)

    assert first == second
    assert first["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "artifact_sha256"}
    )
    assert first["claim_boundary"]["scored_benchmark_observations"] == 0
    assert first["claim_boundary"]["leaderboard_use"] is False
    assert first["acceptance"]["status"] == "fail"
    assert verify_artifact(first) is False


def test_null_and_alternative_flags_must_match_the_estimand() -> None:
    invalid = OrdinalScenario(
        scenario_id="invalid-null",
        loss_probability=0.3,
        tie_probability=0.2,
        win_probability=0.5,
        alternative=False,
    )

    with pytest.raises(MethodValidationError, match="alternative flag"):
        evaluate_scenario(invalid, datasets=2, bootstrap_replicates=2, seed=1)


def test_verifier_rejects_tampering_even_when_acceptance_is_relabelled() -> None:
    artifact = build_artifact(datasets=40, bootstrap_replicates=49, seed=19)
    tampered = deepcopy(artifact)
    tampered["acceptance"]["status"] = "pass"
    tampered["acceptance"]["all_scenarios_pass"] = True
    tampered["parameters"]["monte_carlo_datasets_per_scenario"] = 2_000

    assert verify_artifact(tampered) is False


def test_writer_refuses_underpowered_validation(tmp_path) -> None:
    artifact = build_artifact(datasets=40, bootstrap_replicates=49, seed=23)

    with pytest.raises(MethodValidationError, match="failed method-validation"):
        write_artifact(artifact, tmp_path)
