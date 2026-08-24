from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.task_campaign_human_sampling_successor import (
    DEFAULT_GATE_AUDIT,
    DEFAULT_SOURCE_DESIGN,
    DEFAULT_STALE_REVIEW,
    FAMILIES,
    GATE_AUDIT_PHYSICAL_SHA256,
    GATE_AUDIT_SEMANTIC_SHA256,
    SOURCE_DESIGN_PHYSICAL_SHA256,
    SOURCE_DESIGN_SEMANTIC_SHA256,
    STALE_REVIEW_PHYSICAL_SHA256,
    HumanSamplingError,
    build_sampling_artifact,
    materialize_sampling_frame,
    verify_sampling_artifact,
    write_sampling_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "artifacts/season1/human-judgment-sampling-v1-candidate"


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return build_sampling_artifact()


@pytest.fixture(scope="module")
def frame(artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return materialize_sampling_frame(artifact)


def _source_design() -> dict[str, Any]:
    return json.loads(DEFAULT_SOURCE_DESIGN.read_text(encoding="utf-8"))


def _connected(models: list[str], pairs: set[tuple[str, str]]) -> bool:
    adjacency = {model: set() for model in models}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    pending = [models[0]]
    while pending:
        model = pending.pop()
        if model not in seen:
            seen.add(model)
            pending.extend(adjacency[model] - seen)
    return seen == set(models)


def test_sources_are_bound_by_exact_semantic_and_physical_hashes(
    artifact: dict[str, Any],
) -> None:
    assert hashlib.sha256(DEFAULT_SOURCE_DESIGN.read_bytes()).hexdigest() == (
        SOURCE_DESIGN_PHYSICAL_SHA256
    )
    assert hashlib.sha256(DEFAULT_GATE_AUDIT.read_bytes()).hexdigest() == (
        GATE_AUDIT_PHYSICAL_SHA256
    )
    assert hashlib.sha256(DEFAULT_STALE_REVIEW.read_bytes()).hexdigest() == (
        STALE_REVIEW_PHYSICAL_SHA256
    )
    commitments = artifact["source_commitments"]
    assert commitments[0]["semantic_sha256"] == SOURCE_DESIGN_SEMANTIC_SHA256
    assert commitments[0]["physical_sha256"] == SOURCE_DESIGN_PHYSICAL_SHA256
    assert commitments[1]["semantic_sha256"] == GATE_AUDIT_SEMANTIC_SHA256
    assert commitments[1]["physical_sha256"] == GATE_AUDIT_PHYSICAL_SHA256
    assert commitments[2]["semantic_sha256"] is None
    assert commitments[2]["physical_sha256"] == STALE_REVIEW_PHYSICAL_SHA256


def test_arena_frame_is_exact_balanced_subset_and_connected(
    artifact: dict[str, Any],
    frame: dict[str, list[dict[str, Any]]],
) -> None:
    source = _source_design()
    models = [row["model_id"] for row in source["candidate_model_panel"]["models"]]
    arena_source = source["primary_schedule"]["model_arena"]
    matchings = {
        row["matching_index_zero_based"]: [tuple(sorted(pair)) for pair in row["model_pairs"]]
        for row in arena_source["factorization"]["matchings"]
    }
    candidate_pairs: dict[int, set[tuple[str, str]]] = {}
    for slot in arena_source["abstract_task_schedule"]:
        candidate_pairs[slot["design_slot_ordinal"]] = {
            pair
            for matching_index in slot["matching_indices_zero_based"]
            for pair in matchings[matching_index]
        }

    arena = frame["arena_comparisons"]
    assert len(arena) == 800
    by_task: Counter[int] = Counter()
    by_family: Counter[str] = Counter()
    global_pairs: Counter[tuple[str, str]] = Counter()
    family_pairs: dict[str, Counter[tuple[str, str]]] = {
        family: Counter() for family in FAMILIES
    }
    global_models: Counter[str] = Counter()
    family_models: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    task_models: dict[int, Counter[str]] = defaultdict(Counter)

    for row in arena:
        task = row["design_slot_ordinal"]
        family = row["family"]
        pair = tuple(row["model_ids"])
        assert pair in candidate_pairs[task]
        assert 0 <= row["source_generated_pair_index_zero_based"] < 35
        by_task[task] += 1
        by_family[family] += 1
        global_pairs[pair] += 1
        family_pairs[family][pair] += 1
        global_models.update(pair)
        family_models[family].update(pair)
        task_models[task].update(pair)

    assert by_task == Counter({task: 10 for task in range(1, 81)})
    assert by_family == Counter({family: 200 for family in FAMILIES})
    assert Counter(global_pairs.values()) == Counter({9: 72, 8: 19})
    assert Counter(global_models.values()) == Counter({114: 10, 115: 4})
    assert all(
        Counter(counts.values()) == Counter({2: 73, 3: 18})
        for counts in family_pairs.values()
    )
    assert all(
        Counter(counts.values()) == Counter({28: 6, 29: 8})
        for counts in family_models.values()
    )
    assert all(
        Counter(counts.values()) == Counter({1: 8, 2: 6})
        for counts in task_models.values()
    )
    assert _connected(models, set(global_pairs))
    assert all(_connected(models, set(family_pairs[family])) for family in FAMILIES)
    assert artifact["balance_certificate"]["arena"]["full_graph"] == {
        "nodes": 14,
        "distinct_edges": 91,
        "connected": True,
    }


def test_uplift_frame_selects_ten_distinct_models_and_existing_repetitions(
    frame: dict[str, list[dict[str, Any]]],
) -> None:
    uplift = frame["uplift_comparisons"]
    assert len(uplift) == 800
    by_task: Counter[int] = Counter()
    task_models: dict[int, set[str]] = defaultdict(set)
    global_models: Counter[str] = Counter()
    family_models: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    for row in uplift:
        task = row["design_slot_ordinal"]
        model = row["model_id"]
        available = row["source_generated_repetitions_available"]
        repetition = row["source_generated_repetition_index_one_based"]
        assert available in {2, 3}
        assert 1 <= repetition <= available
        assert row["conditions"] == ["epicure_off", "epicure_on"]
        by_task[task] += 1
        task_models[task].add(model)
        global_models[model] += 1
        family_models[row["family"]][model] += 1

    assert by_task == Counter({task: 10 for task in range(1, 81)})
    assert all(len(models) == 10 for models in task_models.values())
    assert Counter(global_models.values()) == Counter({57: 12, 58: 2})
    assert all(
        Counter(counts.values()) == Counter({14: 10, 15: 4})
        for counts in family_models.values()
    )


def test_two_rater_slots_and_concealed_repeats_are_exact(
    frame: dict[str, list[dict[str, Any]]],
) -> None:
    comparisons = [*frame["arena_comparisons"], *frame["uplift_comparisons"]]
    assert len({row["comparison_id"] for row in comparisons}) == 1600
    assert all(row["required_distinct_raters"] == 2 for row in comparisons)
    assert all(len(set(row["judgment_slot_ids"])) == 2 for row in comparisons)

    slots = frame["primary_judgment_slots"]
    assert len(slots) == 3200
    assert len({row["judgment_slot_id"] for row in slots}) == 3200
    slots_by_comparison: dict[str, set[int]] = defaultdict(set)
    for row in slots:
        assert row["distinct_from_other_rater_slot_for_comparison"] is True
        slots_by_comparison[row["comparison_id"]].add(row["rater_slot"])
    assert all(rater_slots == {1, 2} for rater_slots in slots_by_comparison.values())

    repeats = frame["concealed_repeat_presentations"]
    assert len(repeats) == 400
    assert len({row["repeat_presentation_id"] for row in repeats}) == 400
    assert Counter(row["design_slot_ordinal"] for row in repeats) == Counter(
        {task: 5 for task in range(1, 81)}
    )
    assert Counter(row["track"] for row in repeats) == Counter(
        {"model_arena": 200, "epicure_uplift": 200}
    )
    assert Counter(row["rater_slot"] for row in repeats) == Counter({1: 200, 2: 200})
    assert all(row["same_rater_as_source_required"] is True for row in repeats)


def test_authority_and_3072_wording_reconciliation_fail_closed(
    artifact: dict[str, Any],
) -> None:
    reconciliation = artifact["judgment_count_reconciliation"]
    assert reconciliation["authoritative_primary_judgments"] == 3200
    assert reconciliation["source_design_primary_judgments"] == 3200
    assert reconciliation["stale_workload_wording_judgments"] == 3072
    assert reconciliation["difference"] == 128
    assert reconciliation["stale_document_modified"] is False

    boundary = artifact["claim_boundary"]
    assert boundary["official"] is False
    assert boundary["rank_eligible"] is False
    assert boundary["model_calls_authorized"] is False
    assert boundary["epicure_calls_authorized"] is False
    assert boundary["human_contact_authorized"] is False
    assert boundary["human_judgment_collection_authorized"] is False
    assert boundary["quality_evidence_observed"] is False
    assert boundary["quality_observations"] == 0
    assert boundary["human_judgments"] == 0
    assert artifact["validation_status"]["power_validated"] is False
    assert artifact["validation_status"]["cost_validated"] is False


def test_tampering_with_sources_recipe_or_authority_fails_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    tampered_source = tmp_path / "source.json"
    shutil.copyfile(DEFAULT_SOURCE_DESIGN, tampered_source)
    source = json.loads(tampered_source.read_text(encoding="utf-8"))
    source["claim_boundary"]["quality_observations"] = 1
    tampered_source.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(HumanSamplingError, match="physical digest mismatch"):
        build_sampling_artifact(design_path=tampered_source)

    tampered = json.loads(json.dumps(artifact))
    tampered["claim_boundary"]["official"] = True
    body = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = sha256_json(body)
    with pytest.raises(HumanSamplingError, match="differs from exact recipe"):
        verify_sampling_artifact(tampered)

    tampered = json.loads(json.dumps(artifact))
    tampered["sampling_recipe"]["arena"]["positions_by_design_slot_ordinal"][0][0] = 0
    body = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = sha256_json(body)
    with pytest.raises(HumanSamplingError, match="differs from exact recipe"):
        verify_sampling_artifact(tampered)


def test_builder_writer_and_checked_in_candidate_are_deterministic(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    assert build_sampling_artifact() == artifact
    verify_sampling_artifact(artifact)
    first = write_sampling_artifact(artifact, tmp_path)
    second = write_sampling_artifact(artifact, tmp_path)
    assert first == second
    assert first.name == (
        f"human-judgment-sampling-v1-candidate-{artifact['artifact_sha256']}.json"
    )
    assert json.loads(first.read_text(encoding="utf-8")) == artifact

    checked_in = CANDIDATE_DIR / first.name
    assert json.loads(checked_in.read_text(encoding="utf-8")) == artifact
