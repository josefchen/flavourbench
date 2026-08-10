from __future__ import annotations

from collections import Counter

from flavourbench.epicure_native_taskset import (
    build_taskset,
    parse_final_choice,
    score_answer,
    verify_taskset,
    write_taskset,
)
from flavourbench.real_dataset_runner import load_epicure_native_task_inventory


def _neighbors(ingredient: str, *, top_k: int) -> dict[str, object]:
    assert top_k == 4
    return {
        "ingredient": ingredient,
        "neighbors": [
            {"name": f"{ingredient}_neighbor_{index}", "rank": index, "sim": 1 - index / 10}
            for index in range(1, 5)
        ],
    }


def _pairing_score(left: str, right: str) -> dict[str, object]:
    return {"ingredient_a": left, "ingredient_b": right, "pairing_score": 0.4}


def _compare(left: str, right: str, axis: str) -> dict[str, object]:
    return {
        "ingredient_a": left,
        "ingredient_b": right,
        "axis": axis,
        "projection_a": 0.5,
        "projection_b": -0.2,
    }


def _cultural(ingredient: str) -> dict[str, object]:
    return {
        "ingredient": ingredient,
        "cuisines": {
            "Italian": {"score": 0.8},
            "Japanese": {"score": 0.7},
            "Mexican": {"score": 0.6},
            "Thai": {"score": 0.5},
        },
    }


def _document() -> dict[str, object]:
    return build_taskset(
        neighbors=_neighbors,
        pairing_score=_pairing_score,
        compare_on_axis=_compare,
        cultural_profile=_cultural,
        epicure_provenance={
            "schema_version": "epicure-provenance-v1",
            "release_id": "unit-test-release",
            "bundle_sha256": "a" * 64,
            "application_sha256": "b" * 64,
            "ingredient_count": 1_790,
            "embedding_dimensions": 300,
        },
    )


def test_taskset_is_balanced_content_addressed_and_runner_loadable(tmp_path) -> None:
    document = _document()
    assert verify_taskset(document)
    assert len(document["tasks"]) == 32
    assert Counter(task["family"] for task in document["tasks"]) == {
        "substitution": 8,
        "composition": 8,
        "cookability": 8,
        "evidence": 8,
    }

    path = write_taskset(document, tmp_path)
    inventory, source = load_epicure_native_task_inventory(path)
    assert len(inventory) == 32
    assert source["artifact_sha256"] == document["artifact_sha256"]
    assert source["automated_ground_truth"] is True
    assert source["human_judgments_required"] == 0
    assert source["rank_eligible"] is True


def test_exact_final_choice_scoring_uses_last_valid_marker() -> None:
    task = _document()["tasks"][0]
    expected = task["expected_choice"]
    assert parse_final_choice(f"FINAL_CHOICE: A\nFINAL_CHOICE: {expected}") == expected
    assert score_answer(task, f"Short reason.\nFINAL_CHOICE: {expected}") == {
        "observed_choice": expected,
        "expected_choice": expected,
        "parseable": True,
        "correct": True,
        "score": 1,
    }
    assert score_answer(task, "No marker")["score"] == 0
