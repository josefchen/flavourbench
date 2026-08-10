from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.task_campaign_study_design_successor import (
    DEFAULT_LEGACY_DESIGN,
    EXPECTED_MODELS,
    FAMILIES,
    SCHEMA_VERSION,
    STATUS,
    SuccessorDesignError,
    build_successor_design,
    verify_successor_design,
    write_successor_design,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / (
    "artifacts/season1/study-design-v6-candidate/"
    "study-design-v6-candidate-"
    "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e.json"
)


def _document() -> dict:
    return build_successor_design()


def test_exact_evidence_builds_only_a_blocked_offline_candidate() -> None:
    document = _document()
    verify_successor_design(document)

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["status"] == STATUS
    assert document["artifact_sha256"] == sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    assert len(document["source_commitments"]) == 6
    assert document["reconciliation"]["legacy_design"] == {
        "task_source": "newly_human_authored_hidden",
        "task_total": 240,
        "tasks_per_family": 60,
        "splits": {"scored": 160, "development": 40, "private_reserve": 40},
        "model_count": 16,
        "primary_response_arms": 12800,
        "total_planned_real_response_arms": 14560,
        "paper_claim_class": "prospective_design_not_result",
    }
    campaign = document["reconciliation"]["public_source_campaign"]
    assert campaign["candidate_slate"] == 180
    assert campaign["campaign_attrition_reserve_candidates"] == 60
    assert campaign["target_admitted_tasks"] == 120
    assert campaign["currently_admitted_tasks"] == 0
    assert campaign["human_ballots"] == 0
    assert campaign["contamination_claim"] == (
        "contamination_limited_not_contamination_free"
    )

    boundary = document["claim_boundary"]
    assert boundary == {
        "activation_effect": "none",
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "paper_or_public_claim_authorized": False,
        "task_bank_import_authorized": False,
        "human_contact_authorized": False,
        "compensation_or_spend_authorized": False,
        "model_or_epicure_calls_authorized": False,
        "human_judgments": 0,
        "quality_observations": 0,
        "tasks_created": 0,
        "tasks_admitted": 0,
    }


def test_proposed_bank_split_is_80_20_20_without_fabricated_tasks() -> None:
    bank = _document()["prospective_task_bank_if_120_tasks_are_admitted"]
    assert bank["total"] == 120
    assert bank["families"] == {family: 30 for family in FAMILIES}
    assert bank["splits"] == {"scored": 80, "development": 20, "rotation_reserve": 20}
    assert bank["split_per_family"] == {
        "scored": 20,
        "development": 5,
        "rotation_reserve": 5,
    }
    assert bank["campaign_attrition_reserve_is_not_bank_split"] is True
    assert bank["rotation_reserve_is_not_a_private_holdout"] is True
    assert bank["task_ids_assigned_in_this_artifact"] == 0
    assert bank["task_content_created_in_this_artifact"] == 0


def test_arena_schedule_is_five_regular_and_pair_balanced() -> None:
    document = _document()
    arena = document["primary_schedule"]["model_arena"]
    models = [row[0] for row in EXPECTED_MODELS]
    matchings = [
        [tuple(pair) for pair in row["model_pairs"]]
        for row in arena["factorization"]["matchings"]
    ]
    all_factor_pairs = [tuple(sorted(pair)) for matching in matchings for pair in matching]
    assert len(matchings) == 13
    assert len(all_factor_pairs) == 91
    assert len(set(all_factor_pairs)) == 91

    appearances: Counter[str] = Counter()
    pair_repetitions: Counter[tuple[str, str]] = Counter()
    family_slots: Counter[str] = Counter()
    for slot in arena["abstract_task_schedule"]:
        family_slots[slot["family"]] += 1
        pairs = [
            pair
            for matching_index in slot["matching_indices_zero_based"]
            for pair in matchings[matching_index]
        ]
        assert len(pairs) == 35
        assert len(set(pairs)) == 35
        degree = Counter(model for pair in pairs for model in pair)
        assert degree == Counter({model: 5 for model in models})
        appearances.update(degree)
        pair_repetitions.update(tuple(sorted(pair)) for pair in pairs)

    assert family_slots == Counter({family: 20 for family in FAMILIES})
    assert appearances == Counter({model: 400 for model in models})
    assert len(pair_repetitions) == 91
    assert set(pair_repetitions.values()) == {30, 31}
    assert sum(pair_repetitions.values()) == 2800
    assert arena["total_response_arms"] == 5600


def test_uplift_schedule_hits_exact_model_and_family_floors() -> None:
    uplift = _document()["primary_schedule"]["epicure_uplift"]
    models = [row[0] for row in EXPECTED_MODELS]
    totals: Counter[str] = Counter()
    family_totals = {family: Counter() for family in FAMILIES}

    for slot in uplift["abstract_task_schedule"]:
        third = set(slot["models_with_third_repetition"])
        repetitions = Counter({model: 2 + int(model in third) for model in models})
        assert len(third) == 7
        assert set(repetitions.values()) == {2, 3}
        assert sum(repetitions.values()) == 35
        totals.update(repetitions)
        family_totals[slot["family"]].update(repetitions)

    assert totals == Counter({model: 200 for model in models})
    assert all(
        counts == Counter({model: 50 for model in models})
        for counts in family_totals.values()
    )
    assert uplift["total_pairs"] == 2800
    assert uplift["total_response_arms"] == 5600


def test_unique_arm_and_human_presentation_arithmetic_is_explicit() -> None:
    document = _document()
    assert document["arithmetic"] == {
        "arena_battles": 2800,
        "arena_response_arms": 5600,
        "uplift_pairs": 2800,
        "uplift_response_arms": 5600,
        "primary_response_arms": 11200,
        "reliability_response_arms": 1680,
        "prompt_sensitivity_response_arms": 420,
        "total_planned_unique_real_response_arms": 13300,
        "power_or_precision_conclusion": "none_arithmetic_feasibility_only",
    }
    assert document["prospective_human_review_floor"] == {
        "unique_comparisons": {"model_arena": 800, "epicure_uplift": 800},
        "independent_raters_per_comparison": 2,
        "primary_judgments": 3200,
        "concealed_repeat_rate": "0.125",
        "additional_repeat_presentations": 400,
        "total_rating_presentations": 3600,
        "authorized": False,
        "reviewers_contacted_by_this_artifact": 0,
    }


def test_roster_is_exactly_14_unranked_candidates_and_qwen_is_excluded() -> None:
    document = _document()
    panel = document["candidate_model_panel"]
    observed = [
        (row["model_id"], row["canonical_model_slug"]) for row in panel["models"]
    ]
    assert observed == list(EXPECTED_MODELS)
    assert panel["model_count"] == 14
    assert panel["official_roster"] is False
    assert panel["quality_observations_used"] == 0
    assert "qwen3.8-max" not in {row["model_id"] for row in panel["models"]}

    qwen = document["qwen_3_8_max_eligibility"]
    assert qwen["identity_kind"] == "mutable_alias"
    assert qwen["frozen_release"] is False
    assert qwen["candidate_roster_member"] is False
    assert qwen["confirmatory_eligible"] is False
    assert qwen["silent_replacement_allowed"] is False


def test_source_tampering_and_candidate_eligibility_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    tampered_legacy = tmp_path / "season1-study-design-v5.json"
    shutil.copyfile(DEFAULT_LEGACY_DESIGN, tampered_legacy)
    legacy = json.loads(tampered_legacy.read_text(encoding="utf-8"))
    legacy["task_bank"]["total"] = 239
    tampered_legacy.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(SuccessorDesignError, match="physical digest mismatch"):
        build_successor_design(legacy_design_path=tampered_legacy)

    document = _document()
    document["qwen_3_8_max_eligibility"]["confirmatory_eligible"] = True
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    document["artifact_sha256"] = sha256_json(body)
    with pytest.raises(SuccessorDesignError, match="Qwen eligibility"):
        verify_successor_design(document)


def test_writer_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    document = _document()
    first = write_successor_design(document, tmp_path)
    second = write_successor_design(document, tmp_path)
    assert first == second
    assert first.name == f"study-design-v6-candidate-{document['artifact_sha256']}.json"
    written = json.loads(first.read_text(encoding="utf-8"))
    assert written == document
    verify_successor_design(written)


def test_checked_in_candidate_exactly_matches_current_verified_sources() -> None:
    stored = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert stored == _document()
    verify_successor_design(stored)
