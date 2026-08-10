from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from flavourbench import full_k16_arena_resolution_audit_v1 as audit
from flavourbench.season1_arena_monte_carlo import build_production_layout


def test_baseline_reproduces_frozen_layout_and_requested_workloads() -> None:
    baseline = audit.build_layout(audit.BASELINE)
    frozen = build_production_layout()

    assert baseline["model_ids"] == frozen["model_ids"]
    assert baseline["tasks"] == frozen["tasks"]
    assert baseline["battles"] == frozen["battles"]
    assert baseline["counts"]["unique_comparisons"] == 3200
    assert baseline["counts"]["primary_human_presentations"] == 6400
    assert baseline["counts"]["total_human_presentations_with_repeats"] == 7200

    expected = {
        "p40_r2": (6400, 12800, 14400),
        "p80_r2": (12800, 25600, 28800),
        "p20_r4": (3200, 12800, 14400),
        "p120_r2": (19200, 38400, 43200),
    }
    for spec in (*audit.REQUESTED_EXPANSIONS, audit.MAXIMAL_PAIR_EXPANSION):
        counts = audit.build_layout(spec)["counts"]
        assert (
            counts["unique_comparisons"],
            counts["primary_human_presentations"],
            counts["total_human_presentations_with_repeats"],
        ) == expected[spec.config_id]
        assert counts["unique_response_identities_reused"] == 2560


def test_every_layout_has_distinct_within_task_pairs() -> None:
    for spec in (
        audit.BASELINE,
        *audit.REQUESTED_EXPANSIONS,
        audit.MAXIMAL_PAIR_EXPANSION,
    ):
        layout = audit.build_layout(spec)
        by_task: dict[str, set[tuple[str, str]]] = {}
        for battle in layout["battles"]:
            pair = tuple(sorted((battle["model_a"], battle["model_b"])))
            assert pair not in by_task.setdefault(battle["task_id"], set())
            by_task[battle["task_id"]].add(pair)
        assert {len(pairs) for pairs in by_task.values()} == {spec.distinct_pairs_per_task}


def test_exact_clopper_pearson_acceptance_uses_lower_not_point() -> None:
    eighty_eight = audit._exact_rate(88, 100)
    eighty_nine = audit._exact_rate(89, 100)

    assert eighty_eight["point_estimate"] >= audit.FROZEN_POWER_TARGET
    assert eighty_eight["clopper_pearson_95_lower"] < audit.FROZEN_POWER_TARGET
    assert eighty_nine["clopper_pearson_95_lower"] >= audit.FROZEN_POWER_TARGET
    assert audit._exact_rate(0, 100)["clopper_pearson_95_upper"] > 0.0
    assert audit._exact_rate(100, 100)["clopper_pearson_95_lower"] < 1.0


@pytest.fixture(scope="module")
def tiny_record():
    return audit.run_dataset(
        spec=audit.BASELINE,
        shift_elo=50.0,
        dataset_index=0,
        bootstrap_replicates=5,
    )


def test_exact_engine_record_is_deterministic_and_separates_estimands(tiny_record) -> None:
    repeated = audit.run_dataset(
        spec=audit.BASELINE,
        shift_elo=50.0,
        dataset_index=0,
        bootstrap_replicates=5,
    )

    assert repeated == tiny_record
    assert tiny_record["engine"] == ("exact_production_bt_plus_crossed_task_rater_bootstrap")
    assert len(tiny_record["peer_intervals"]) == 15
    assert tiny_record["uniform_peer_model_id"].startswith("model-")
    assert tiny_record["claim_boundary"] == {
        "counts_toward_production_gate": False,
        "production_method_validation_complete": False,
        "model_quality_evidence": False,
        "human_judgments_created": False,
    }


def test_aggregate_does_not_use_correlated_peer_intervals_as_binomial_trials(
    tiny_record,
) -> None:
    aggregate = audit.aggregate_condition([tiny_record])

    assert aggregate["frozen_average_marginal_pair_power_uniform_peer_estimator"]["trials"] == 1
    assert aggregate["fixed_focal_vs_model_01_detection"]["trials"] == 1
    assert aggregate["all_15_pointwise_intervals_positive_same_dataset"]["trials"] == 1
    assert "descriptive_all_peer_detection_fraction_correlated_not_binomial" in aggregate

    without_duplication = deepcopy(tiny_record)
    without_duplication["duplicate_interval_width_delta"] = None
    defensive = audit.aggregate_condition([without_duplication])
    assert defensive["maximum_absolute_duplicate_interval_width_delta"] is None


def _patch_fast_condition_runner(
    monkeypatch: pytest.MonkeyPatch, tiny_record: dict[str, Any]
) -> None:
    def fake_run_condition(
        *,
        spec: audit.LayoutSpec,
        shift_elo: float,
        datasets: int,
        bootstrap_replicates: int,
        workers: int,
        dataset_start: int = 0,
        record_lineage: str = "generated_current_run",
    ):
        del datasets, workers
        record = deepcopy(tiny_record)
        record["config_id"] = spec.config_id
        record["layout_sha256"] = audit.build_layout(spec)["artifact_sha256"]
        record["shift_elo"] = shift_elo
        record["bootstrap_replicates"] = bootstrap_replicates
        record["dataset_index"] = dataset_start
        record["record_lineage"] = record_lineage
        record_body = {key: value for key, value in record.items() if key != "record_sha256"}
        record["record_sha256"] = audit._semantic_sha256(record_body)
        return [record], audit.aggregate_condition([record])

    monkeypatch.setattr(audit, "run_condition", fake_run_condition)


def test_small_artifact_stays_no_go_and_all_authority_flags_false(
    tiny_record, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fast_condition_runner(monkeypatch, tiny_record)
    document = audit.build_audit_artifact(
        workers=1,
        baseline_datasets=1,
        baseline_bootstraps=2,
        screen_datasets=1,
        screen_bootstraps=2,
        confirm_datasets=1,
        confirm_bootstraps=3,
        enforce_minimums=False,
        reuse_verified_predecessor_stages=False,
    )

    audit.verify_audit_artifact(document)
    assert document["decision"]["overall_verdict"] == "NO-GO"
    assert document["claim_boundary"]["development_only"] is True
    assert not any(
        value for key, value in document["claim_boundary"].items() if key != "development_only"
    )
    assert (
        document["simulation_contract"]["production_gate_required_but_not_run"][
            "nominal_bootstrap_refits"
        ]
        == 80_000_000
    )
    two_stage = document["simulation_contract"]["two_stage_plan"]
    assert two_stage["screen_dataset_indices"] == [0, 0]
    assert two_stage["confirmation_dataset_indices"] == [1, 1]
    assert two_stage["screen_confirmation_dataset_overlap"] == 0
    assert document["supersession"]["all_predecessors_retained_append_only"] is True
    assert {row["semantic_sha256"] for row in document["supersession"]["superseded_artifacts"]} == {
        audit.PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
        audit.INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
    }
    assert {row["status"] for row in document["supersession"]["superseded_artifacts"]} == {
        "superseded_not_for_use"
    }

    tampered = deepcopy(document)
    tampered["claim_boundary"]["official_benchmark_or_rank_authority"] = True
    with pytest.raises(audit.ArenaResolutionAuditError, match="semantic digest"):
        audit.verify_audit_artifact(tampered)


def test_writer_is_content_addressed_idempotent_and_no_replace(
    tiny_record, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fast_condition_runner(monkeypatch, tiny_record)
    document = audit.build_audit_artifact(
        workers=1,
        baseline_datasets=1,
        baseline_bootstraps=2,
        screen_datasets=1,
        screen_bootstraps=2,
        confirm_datasets=1,
        confirm_bootstraps=3,
        enforce_minimums=False,
        reuse_verified_predecessor_stages=False,
    )
    first = audit.write_audit_artifact(document, tmp_path)
    inode = first.stat().st_ino
    second = audit.write_audit_artifact(document, tmp_path)

    assert first == second
    assert second.stat().st_ino == inode
    assert first.name.endswith(f"{document['artifact_sha256']}.json")
    assert not list(tmp_path.glob(".full-k16-resolution-v1-*"))


def test_simulation_identity_excludes_lineage_hash_and_bootstrap_but_not_dgp_fields() -> None:
    predecessor = audit._load_provisional_predecessor()
    invalidated = audit._load_invalidated_successor()
    left = next(
        row
        for row in audit._artifact_confirmation_records(predecessor)
        if row["dataset_index"] == 40
    )
    right = next(
        row
        for row in audit._artifact_confirmation_records(invalidated)
        if row["dataset_index"] == 40
    )

    assert left["record_sha256"] != right["record_sha256"]
    assert left.get("record_lineage") != right.get("record_lineage")
    assert audit._simulation_identity_key(left) == audit._simulation_identity_key(right)

    bootstrap_changed = deepcopy(left)
    bootstrap_changed["bootstrap_replicates"] += 1
    assert audit._simulation_identity_key(bootstrap_changed) == (
        audit._simulation_identity_key(left)
    )
    for field, value in (
        ("layout_sha256", "0" * 64),
        ("config_id", "different"),
        ("shift_elo", 75.0),
        ("dataset_index", 41),
        ("dataset_seed", int(left["dataset_seed"]) + 1),
    ):
        changed = deepcopy(left)
        changed[field] = value
        assert audit._simulation_identity_key(changed) != (audit._simulation_identity_key(left))


def test_bound_predecessor_identity_profile_exposes_hash_blind_overlap() -> None:
    predecessor = audit._load_provisional_predecessor()
    invalidated = audit._load_invalidated_successor()
    predecessor_identities = {
        audit._simulation_identity_key(row) for row in predecessor["dataset_records"]
    }
    invalidated_identities = {
        audit._simulation_identity_key(row) for row in invalidated["dataset_records"]
    }
    predecessor_confirmation = {
        audit._simulation_identity_key(row)
        for row in audit._artifact_confirmation_records(predecessor)
    }
    invalidated_confirmation = {
        audit._simulation_identity_key(row)
        for row in audit._artifact_confirmation_records(invalidated)
    }
    shared_hashes = {row["record_sha256"] for row in predecessor["dataset_records"]}.intersection(
        row["record_sha256"] for row in invalidated["dataset_records"]
    )

    assert len(predecessor["dataset_records"]) == 520
    assert len(invalidated["dataset_records"]) == 520
    assert len(predecessor_identities) == 480
    assert len(invalidated_identities) == 520
    assert len(shared_hashes) == 420
    assert len(predecessor_confirmation.intersection(invalidated_confirmation)) == 60
    assert {
        identity[3] for identity in predecessor_confirmation.intersection(invalidated_confirmation)
    } == set(range(40, 100))
    assert len(predecessor_identities.union(invalidated_identities)) == 520


def test_overlapping_confirmation_plan_fails_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "FRESH_CONFIRMATION_DATASET_START", 40)
    monkeypatch.setattr(audit, "FRESH_CONFIRMATION_DATASET_END", 139)
    runner_called = False

    def forbidden_runner(**kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError(f"runner must not be reached: {kwargs}")

    monkeypatch.setattr(audit, "run_condition", forbidden_runner)
    with pytest.raises(
        audit.ArenaResolutionAuditError,
        match="planned confirmation overlaps predecessor simulation identity",
    ):
        audit.build_audit_artifact(workers=1)
    assert runner_called is False


def test_fresh_520_record_successor_serializes_and_verifies_before_expensive_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalidated = audit._load_invalidated_successor()
    old_confirmation = sorted(
        (deepcopy(row) for row in audit._artifact_confirmation_records(invalidated)),
        key=lambda row: row["dataset_index"],
    )
    assert len(old_confirmation) == 100
    synthetic_confirmation = []
    for offset, dataset_index in enumerate(range(140, 240)):
        record = deepcopy(old_confirmation[offset])
        record["dataset_index"] = dataset_index
        record["dataset_seed"] = audit._dataset_seed("single_model_50_elo_shift", dataset_index)
        record["record_lineage"] = audit.FRESH_CONFIRMATION_LINEAGE
        record["duplicate_interval_width_delta"] = 0.0 if offset == 0 else None
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        record["record_sha256"] = audit._semantic_sha256(body)
        synthetic_confirmation.append(record)

    def fake_final_condition(**kwargs):
        assert kwargs["spec"].config_id == "p80_r2"
        assert kwargs["dataset_start"] == 140
        assert kwargs["datasets"] == 100
        assert kwargs["bootstrap_replicates"] == 200
        assert kwargs["record_lineage"] == audit.FRESH_CONFIRMATION_LINEAGE
        return synthetic_confirmation, audit.aggregate_condition(synthetic_confirmation)

    monkeypatch.setattr(audit, "run_condition", fake_final_condition)
    document = audit.build_audit_artifact(workers=1)
    path = audit.write_audit_artifact(document, tmp_path)
    audit.verify_audit_artifact(document)

    assert path.exists()
    assert len(document["dataset_records"]) == 520
    assert document["stage_lineage"]["reused_record_count"] == 420
    assert document["stage_lineage"]["excluded_overlapping_confirmation_record_count"] == 100
    assert document["stage_lineage"]["fresh_confirmation_dataset_indices"] == [
        140,
        239,
    ]
    assert document["stage_lineage"]["fresh_confirmation_overlap_with_prior_observed_union"] == 0
    assert document["simulation_contract"]["two_stage_plan"]["confirmation_dataset_indices"] == [
        140,
        239,
    ]
    predecessor = audit._load_provisional_predecessor()
    old_hashes = {
        row["record_sha256"]
        for source in (predecessor, invalidated)
        for row in audit._artifact_confirmation_records(source)
    }
    current_hashes = {row["record_sha256"] for row in document["dataset_records"]}
    assert old_hashes.isdisjoint(current_hashes)
    fresh_records = [
        row
        for row in document["dataset_records"]
        if row.get("record_lineage") == audit.FRESH_CONFIRMATION_LINEAGE
    ]
    prior_identities = {
        audit._simulation_identity_key(row)
        for source in (predecessor, invalidated)
        for row in source["dataset_records"]
    }
    assert len(fresh_records) == 100
    assert {audit._simulation_identity_key(row) for row in fresh_records}.isdisjoint(
        prior_identities
    )

    tampered = deepcopy(document)
    tampered["stage_lineage"]["fresh_confirmation_overlap_with_prior_observed_union"] = 1
    tampered_body = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = audit._semantic_sha256(tampered_body)
    with pytest.raises(
        audit.ArenaResolutionAuditError,
        match="identity-lineage metadata",
    ):
        audit.verify_audit_artifact(tampered)


def test_materialized_fresh_successor_recomputes_and_remains_no_go() -> None:
    path = (
        audit.DEFAULT_OUTPUT_DIR / "full-k16-arena-resolution-audit-v1-candidate-"
        "596ef6cd38132a351605ce0f734489262e372e1c148337a88238dbb466a12ddc.json"
    )
    raw = path.read_bytes()
    document = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "b07fa02b5633c13d331ad5490c150a514f92aa4bb7abdce985b156aa9e8af17c"
    )
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    assert (
        document["artifact_sha256"]
        == audit._semantic_sha256(body)
        == ("596ef6cd38132a351605ce0f734489262e372e1c148337a88238dbb466a12ddc")
    )
    audit.verify_audit_artifact(document)

    for record in document["dataset_records"]:
        record_body = {key: value for key, value in record.items() if key != "record_sha256"}
        assert record["record_sha256"] == audit._semantic_sha256(record_body)

    fresh = [
        record
        for record in document["dataset_records"]
        if record.get("record_lineage") == audit.FRESH_CONFIRMATION_LINEAGE
    ]
    predecessor_records = [
        record
        for predecessor in (
            audit._load_provisional_predecessor(),
            audit._load_invalidated_successor(),
        )
        for record in predecessor["dataset_records"]
    ]
    assert len(fresh) == 100
    assert {record["dataset_index"] for record in fresh} == set(range(140, 240))
    assert {audit._simulation_identity_key(record) for record in fresh}.isdisjoint(
        audit._simulation_identity_key(record) for record in predecessor_records
    )
    assert {record["record_sha256"] for record in fresh}.isdisjoint(
        record["record_sha256"] for record in predecessor_records
    )

    aggregate = audit.aggregate_condition(fresh)
    stored = next(
        row for row in document["condition_results"] if row["stage"] == "fresh_final_confirmation"
    )
    assert {key: value for key, value in stored.items() if key != "stage"} == aggregate
    assert aggregate["frozen_average_marginal_pair_power_uniform_peer_estimator"] == {
        "successes": 97,
        "trials": 100,
        "point_estimate": 0.97,
        "mc_standard_error": 0.017058722,
        "clopper_pearson_95_lower": 0.914823947,
        "clopper_pearson_95_upper": 0.993770028,
    }
    assert aggregate["all_15_pointwise_intervals_positive_same_dataset"] == {
        "successes": 67,
        "trials": 100,
        "point_estimate": 0.67,
        "mc_standard_error": 0.047021272,
        "clopper_pearson_95_lower": 0.568827249,
        "clopper_pearson_95_upper": 0.760801465,
    }
    assert document["decision"]["overall_verdict"] == "NO-GO"
    assert document["claim_boundary"]["development_only"] is True
    assert not any(
        value for key, value in document["claim_boundary"].items() if key != "development_only"
    )
