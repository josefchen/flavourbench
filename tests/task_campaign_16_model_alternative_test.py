from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.task_campaign_16_model_alternative import (
    CURRENT_14_MODELS,
    EXPECTED_16_MODELS,
    FAMILIES,
    SCHEMA_VERSION,
    STATUS,
    AlternativeDesignError,
    build_alternative_candidate,
    materialize_human_sampling_frame,
    verify_alternative_candidate,
    write_alternative_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / (
    "artifacts/season1/study-design-16-model-alternative-v1-candidate/"
    "study-design-16-model-alternative-v1-candidate-"
    "675cdb81bcbd54cf3532025ae70069723d7e9843b0eeeb92f1ea38bee7c58278.json"
)


@pytest.fixture(scope="module")
def document() -> dict:
    return build_alternative_candidate()


def test_exact_sources_produce_only_a_blocked_nonactivating_alternative(
    document: dict,
) -> None:
    verify_alternative_candidate(document)
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["status"] == STATUS
    assert document["artifact_sha256"] == sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    assert len(document["source_commitments"]) == 8
    assert document["non_activation"] == {
        "current_14_roster_or_schedule_modified": False,
        "current_14_design_semantic_sha256": (
            "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e"
        ),
        "supersedes_current_design": False,
        "activates_candidate": False,
        "runtime_or_importer_change": False,
        "paper_or_result_change": False,
    }
    assert document["claim_boundary"] == {
        "activation_effect": "none",
        "official": False,
        "rank_eligible": False,
        "calls_authorized": False,
        "model_calls_authorized": False,
        "epicure_calls_authorized": False,
        "human_contact_authorized": False,
        "human_judgment_collection_authorized": False,
        "compensation_or_spend_authorized": False,
        "quality_eligible": False,
        "quality_observations_used": 0,
        "generation_calls_made_by_this_artifact": 0,
        "epicure_calls_made_by_this_artifact": 0,
        "human_judgments_made_by_this_artifact": 0,
        "tasks_admitted_by_this_artifact": 0,
        "research_result": False,
        "paper_or_public_claim_authorized": False,
    }


def test_panel_is_exact_current_14_plus_qwen_and_command_a_plus(document: dict) -> None:
    panel = document["candidate_model_panel"]
    observed = tuple(
        (row["model_id"], row["canonical_model_slug"]) for row in panel["models"]
    )
    assert observed == EXPECTED_16_MODELS
    assert observed[:14] == CURRENT_14_MODELS
    assert observed[14:] == (
        ("qwen3.8-max", "qwen3.8-max"),
        ("cohere/command-a-plus-05-2026", "command-a-plus-05-2026"),
    )
    assert panel["model_count"] == 16
    assert panel["selection_uses_quality_observations"] is False
    assert panel["quality_observations_used"] == 0
    assert panel["official"] is False
    assert panel["rank_eligible"] is False
    assert panel["calls_authorized"] is False
    assert panel["quality_eligible"] is False


def test_k16_arena_is_five_regular_with_exact_400_appearances(document: dict) -> None:
    arena = document["study_design"]["model_arena"]
    models = [row[0] for row in EXPECTED_16_MODELS]
    matchings = [
        [tuple(pair) for pair in row["model_pairs"]]
        for row in arena["factorization"]["matchings"]
    ]
    factor_pairs = [tuple(sorted(pair)) for matching in matchings for pair in matching]
    assert len(matchings) == 15
    assert len(factor_pairs) == 120
    assert len(set(factor_pairs)) == 120

    appearances: Counter[str] = Counter()
    pair_repetitions: Counter[tuple[str, str]] = Counter()
    family_tasks: Counter[str] = Counter()
    for slot in arena["abstract_task_schedule"]:
        family_tasks[slot["family"]] += 1
        pairs = [
            pair
            for matching_index in slot["matching_indices_zero_based"]
            for pair in matchings[matching_index]
        ]
        assert len(pairs) == 40
        assert len(set(pairs)) == 40
        degree = Counter(model for pair in pairs for model in pair)
        assert degree == Counter({model: 5 for model in models})
        appearances.update(degree)
        pair_repetitions.update(tuple(sorted(pair)) for pair in pairs)

    assert family_tasks == Counter({family: 20 for family in FAMILIES})
    assert appearances == Counter({model: 400 for model in models})
    assert Counter(pair_repetitions.values()) == Counter({27: 80, 26: 40})
    assert sum(pair_repetitions.values()) == 3_200
    assert arena["total_response_arms"] == 6_400


def test_uplift_reliability_sensitivity_and_total_arithmetic(document: dict) -> None:
    design = document["study_design"]
    uplift = design["epicure_uplift"]
    models = [row[0] for row in EXPECTED_16_MODELS]
    totals: Counter[str] = Counter()
    family_totals = {family: Counter() for family in FAMILIES}
    for slot in uplift["abstract_task_schedule"]:
        third = set(slot["models_with_third_repetition"])
        repetitions = Counter({model: 2 + int(model in third) for model in models})
        assert len(third) == 8
        assert sum(repetitions.values()) == 40
        totals.update(repetitions)
        family_totals[slot["family"]].update(repetitions)
    assert totals == Counter({model: 200 for model in models})
    assert all(
        counts == Counter({model: 50 for model in models})
        for counts in family_totals.values()
    )
    assert uplift["total_pairs"] == 3_200
    assert uplift["total_response_arms"] == 6_400
    assert design["generation_reliability_panel"]["total_unique_response_arms"] == 1_920
    assert design["generation_reliability_panel"]["disjoint_from_scored_primary"] is True
    assert design["prompt_sensitivity_audit"]["endpoint_count"] == 8
    assert design["prompt_sensitivity_audit"]["endpoint_ids_assigned_in_this_artifact"] == 0
    assert design["prompt_sensitivity_audit"]["total_new_unique_response_arms"] == 480
    assert document["arithmetic"] == {
        "arena_battles": 3_200,
        "arena_response_arms": 6_400,
        "uplift_pairs": 3_200,
        "uplift_response_arms": 6_400,
        "primary_response_arms": 12_800,
        "disjoint_development_reliability_response_arms": 1_920,
        "prompt_sensitivity_new_response_arms": 480,
        "total_planned_unique_real_response_arms": 15_200,
        "power_or_precision_conclusion": "none_arithmetic_feasibility_only",
    }


def test_human_frame_is_outcome_blind_and_exactly_balanced(document: dict) -> None:
    frame = materialize_human_sampling_frame(document)
    arena = frame["arena"]
    uplift = frame["uplift"]
    repeats = frame["concealed_repeats"]
    models = [row[0] for row in EXPECTED_16_MODELS]
    assert len(arena) == 800
    assert len(uplift) == 800
    assert Counter(model for row in arena for model in row["model_ids"]) == Counter(
        {model: 100 for model in models}
    )
    assert Counter(row["model_id"] for row in uplift) == Counter(
        {model: 50 for model in models}
    )
    assert Counter(Counter(tuple(row["model_ids"]) for row in arena).values()) == Counter(
        {7: 80, 6: 40}
    )
    assert len(repeats) == 400
    summary = document["human_sampling_frame"]
    assert summary["selection_timing_and_inputs"] == {
        "outcome_blind": True,
        "must_be_frozen_before_any_model_output_or_quality_outcome": True,
        "response_texts_used": 0,
        "model_scores_used": 0,
        "preference_labels_used": 0,
        "quality_observations_used": 0,
        "reviewer_identities_used": 0,
        "outcome_dependent_reselection_allowed": False,
    }
    certificate = summary["balance_certificate"]
    assert certificate["arena"]["exact_appearances_per_model"] == 100
    assert certificate["arena"]["arithmetic_average_appearances_per_model_family"] == 25
    assert certificate["arena"]["model_appearance_range_across_family_cells"] == [19, 39]
    assert certificate["arena"]["unique_task_cluster_range_across_model_family_cells"] == [
        19,
        20,
    ]
    assert certificate["uplift"]["exact_appearances_per_model"] == 50
    assert certificate["human_presentations"]["primary_judgment_slots"] == 3_200
    assert certificate["human_presentations"]["concealed_repeat_presentations"] == 400
    assert certificate["human_presentations"]["total_rating_presentations"] == 3_600


def test_identity_route_cost_and_eligibility_matrix_fails_closed(document: dict) -> None:
    rows = {row["model_id"]: row for row in document["validation_matrix"]["rows"]}
    kimi = rows["moonshotai/kimi-k3"]
    assert kimi["identity"]["requested_model_id"] == "k3"
    assert kimi["identity"]["immutable_served_revision_proven"] is False
    assert kimi["identity"]["continuity_or_version_stability_inferred"] is False
    assert kimi["route"]["status"] == "passed_unranked"
    assert kimi["cost"]["candidate_15200_arm_cost_validated"] is False

    qwen = rows["qwen3.8-max"]
    assert qwen["identity"]["status"] == "time_bounded_mutable_alias_only"
    assert qwen["identity"]["identity_kind"] == "mutable_alias"
    assert qwen["identity"]["catalog_observed_at"] == "2026-08-08T16:29:04Z"
    assert qwen["identity"]["continuity_or_version_stability_inferred"] is False
    assert qwen["cost"]["recorded_zero_cost_means"] == "unknown_not_free"

    cohere = rows["cohere/command-a-plus-05-2026"]
    assert cohere["identity"]["canonical_model_slug"] == "command-a-plus-05-2026"
    assert cohere["route"]["status"] == "passed_unranked"
    assert cohere["cost"]["separate_cost_governor_required"] is True
    assert cohere["cost"]["zero_public_rate_may_not_be_treated_as_reconciled_cost"] is True

    for row in rows.values():
        assert row["eligibility"] == {
            "alternative_candidate_member": True,
            "official": False,
            "rank_eligible": False,
            "call_authorized": False,
            "quality_eligible": False,
        }
    requirements = document["remaining_eligibility_requirements"]
    assert all(
        item["status"] == "missing" and item["satisfied_by_this_artifact"] is False
        for group in requirements.values()
        for item in group
    )


def test_two_lanes_and_missing_slot_consequences_are_explicit(document: dict) -> None:
    lanes = document["two_lane_assessment"]
    immutable = lanes["immutable_confirmatory_roster"]
    observational = lanes["timestamped_current_frontier_observational_extension"]
    missing = lanes["missing_official_slot_consequence"]
    assert immutable["qwen_3_8_max_eligible_now"] is False
    assert immutable["kimi_k3_eligible_as_immutable_now"] is False
    assert immutable["this_k16_content_address_reusable"] is False
    assert immutable["official"] is False
    assert immutable["rank_eligible"] is False
    assert observational["qwen_3_8_max_may_appear"] is True
    assert observational["qwen_alias_commitment"]["identity_kind"] == "mutable_alias"
    assert (
        observational["qwen_alias_commitment"][
            "continuity_or_version_stability_inferred"
        ]
        is False
    )
    assert observational["k16_balanced_schedule_reusable_if_all_16_routes_are_retained"] is True
    assert observational["official"] is False
    assert observational["rank_eligible"] is False
    assert missing["resulting_model_count"] == 15
    assert missing["existing_k16_schedule_reusable"] is False
    assert missing["factorization_breaks"] is True
    assert missing["estimands_unchanged"] is False


def test_frozen_v1_inference_contract_is_bound_and_explicitly_failed(
    document: dict,
) -> None:
    commitments = {row["role"]: row for row in document["source_commitments"]}
    source = commitments[
        "frozen_official_arena_inference_acceptance_v1_non_applicable_audit"
    ]
    assert source["semantic_sha256"] == (
        "bdc0fa93c6365cdcd45694d1d5500d82ccbd622f3be897be9217e252855ffff5"
    )
    assert source["physical_sha256"] == (
        "02adfc4a32e2690c1f8f5ddce6edba3f1974159956027b88003a484f5a0655bc"
    )

    mismatch = document["inference_acceptance_mismatch"]
    assert mismatch["status"] == "no_go_successor_inference_and_power_contract_required"
    assert mismatch["official_v1_inherited"] is False
    matrix = mismatch["validation_matrix"]
    assert matrix["global_required_admitted_scored_tasks"] == {
        "required": 160,
        "alternative_design_capacity": 80,
        "admitted_now": 0,
        "design_capacity_meets_requirement": False,
        "acceptance_satisfied_now": False,
    }
    assert matrix["global_required_admitted_scored_tasks_per_family"]["required"] == 40
    assert matrix["global_required_admitted_scored_tasks_per_family"][
        "alternative_design_capacity"
    ] == 20
    assert matrix["global_minimum_unique_comparisons_per_model"][
        "outcome_blind_frame_capacity"
    ] == 100
    assert matrix["global_minimum_unique_comparisons_per_model"][
        "design_capacity_meets_numeric_requirement"
    ] is True
    assert matrix["family_fit_minimum_unique_comparisons_per_model"][
        "required_per_family"
    ] == 100
    assert matrix["family_fit_minimum_unique_comparisons_per_model"][
        "arithmetic_average_per_model_family"
    ] == 25
    assert matrix["family_fit_minimum_unique_comparisons_per_model"][
        "outcome_blind_frame_range"
    ] == [19, 39]
    assert mismatch["pairwise_direct_support"] == {
        "required_shared_task_clusters_per_pair": 10,
        "outcome_blind_frame_pair_repetition_range": [6, 7],
        "distinct_pairs_below_floor": 120,
        "distinct_pairs_at_or_above_floor": 0,
        "v1_below_floor_action": "suppress_pair_specific_interval",
        "passes": False,
    }
    simulation = mismatch["simulation_contract"]
    assert simulation["required_model_count"] == 16
    assert simulation["minimum_datasets_per_scenario"] == 2_000
    assert simulation["required_bootstrap_replicates"] == 5_000
    assert simulation["passes"] is False
    remediation = mismatch[
        "remediation_options_requiring_new_reviewed_content_addresses"
    ]
    assert remediation["retain_direct_pairwise_interval_support_floor"][
        "minimum_arena_comparisons"
    ] == 1_200
    assert remediation["retain_direct_pairwise_interval_support_floor"][
        "cures_160_task_and_40_per_family_fit_gates"
    ] is False
    assert remediation["version_or_narrow_estimands"][
        "pair_intervals_below_10_under_v1_remain_suppressed"
    ] is True


def test_comparison_is_arithmetic_only_and_14_sampling_is_non_applicable(
    document: dict,
) -> None:
    comparison = document["comparison_with_current_14_model_design"]
    assert comparison["comparison_basis"] == "arithmetic_and_membership_only_not_quality"
    assert comparison["quality_based_choice_made"] is False
    assert comparison["selected_design"] is None
    assert comparison["current_14"]["total_unique_real_response_arms"] == 13_300
    assert comparison["alternative_16"]["total_unique_real_response_arms"] == 15_200
    assert comparison["delta_16_minus_14"]["total_unique_real_response_arms"] == 1_900
    disposition = document["human_sampling_frame"]["current_14_sampling_disposition"]
    assert disposition["applicable_to_this_16_model_alternative"] is False
    assert disposition["semantic_sha256"] == (
        "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8"
    )


def test_source_and_authority_tampering_fail_closed(tmp_path: Path, document: dict) -> None:
    source = ROOT / (
        "artifacts/season1/current-quality-run/"
        "manifest-v44-floor-replenishment-cohere-direct/"
        "flavourbench-cohere-unranked-"
        "a93791bb929bfc45d483ff031016760c6f042e6a7539fa9ef6f23f94b47ebabf.json"
    )
    tampered_source = tmp_path / source.name
    shutil.copyfile(source, tampered_source)
    value = json.loads(tampered_source.read_text(encoding="utf-8"))
    value["official_results_authorised"] = True
    tampered_source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AlternativeDesignError, match="physical digest"):
        build_alternative_candidate(cohere_manifest_path=tampered_source)

    tampered_candidate = copy.deepcopy(document)
    tampered_candidate["claim_boundary"]["rank_eligible"] = True
    body = {
        key: value
        for key, value in tampered_candidate.items()
        if key != "artifact_sha256"
    }
    tampered_candidate["artifact_sha256"] = sha256_json(body)
    with pytest.raises(AlternativeDesignError, match="differs from exact"):
        verify_alternative_candidate(tampered_candidate)


def test_writer_and_checked_in_candidate_are_deterministic(
    tmp_path: Path,
    document: dict,
) -> None:
    first = write_alternative_candidate(document, tmp_path)
    second = write_alternative_candidate(document, tmp_path)
    assert first == second
    assert first.name == (
        f"study-design-16-model-alternative-v1-candidate-{document['artifact_sha256']}.json"
    )
    assert json.loads(first.read_text(encoding="utf-8")) == document

    stored = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert stored == document
    verify_alternative_candidate(stored)
