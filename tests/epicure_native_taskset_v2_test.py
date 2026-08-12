from __future__ import annotations

import hashlib
from collections import Counter

from flavourbench.epicure_native_taskset_v2 import (
    CHOICE_LABELS,
    FAMILIES,
    TASK_COUNT,
    build_taskset,
    parse_final_choice,
    score_answer,
    verify_taskset,
    write_taskset,
)


def _index(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _ingredients() -> list[dict[str, str]]:
    categories = [f"Category {index}" for index in range(12)]
    return [
        {"name": f"ingredient_{index:04d}", "primary_category": categories[index % 12]}
        for index in range(1, 801)
    ]


def _band_margin(name: str, family: str) -> float:
    bands = {
        "substitution": (0.004, 0.012, 0.028, 0.060),
        "composition": (0.010, 0.025, 0.045, 0.080),
        "cookability": (0.010, 0.030, 0.070, 0.140),
        "evidence": (0.020, 0.055, 0.110, 0.210),
    }
    return bands[family][_index(name) % 4]


def _neighbors(name: str, *, top_k: int) -> dict[str, object]:
    assert top_k in {8, 16}
    gap = _band_margin(name, "substitution")
    rows = [
        {"name": f"{name}_near_{rank:02d}", "rank": rank, "sim": 0.8 - gap - rank * 0.01}
        for rank in range(1, top_k + 1)
    ]
    rows[0]["sim"] = float(rows[1]["sim"]) + gap
    return {"ingredient": name, "neighbors": rows}


def _pairing(left: str, right: str) -> dict[str, object]:
    partner_rank = int(right.rsplit("_", 1)[1])
    margin = _band_margin(left, "composition")
    scores = {1: 0.5 + margin, 4: 0.5, 8: 0.4, 12: 0.3}
    return {"resolved_a": left, "resolved_b": right, "pairing_score": scores[partner_rank]}


def _axis(left: str, right: str, axis: str) -> dict[str, object]:
    margin = _band_margin(left, "cookability")
    # The first call establishes the anchor projection.  The remaining hashed
    # alternatives get deterministic values below it.
    suffix = int(hashlib.sha256(right.encode()).hexdigest(), 16) % 1000
    return {
        "resolved_a": left,
        "resolved_b": right,
        "axis": axis,
        "projection_a": 0.5 + margin,
        "projection_b": 0.5 - suffix / 100_000,
    }


def _culture(name: str) -> dict[str, object]:
    margin = _band_margin(name, "evidence")
    return {
        "resolved": name,
        "cuisines": {
            "North": {"score": 0.5 + margin},
            "South": {"score": 0.5},
            "East": {"score": 0.3},
            "West": {"score": 0.1},
        },
    }


def _document() -> dict[str, object]:
    return build_taskset(
        ingredient_records=_ingredients(),
        neighbors=_neighbors,
        pairing_score=_pairing,
        compare_on_axis=_axis,
        cultural_profile=_culture,
        ingredient_inventory_sha256="c" * 64,
        epicure_provenance={
            "schema_version": "epicure-provenance-v1",
            "release_id": "unit-test-release",
            "bundle_sha256": "a" * 64,
            "application_sha256": "b" * 64,
            "ingredient_count": 1_790,
            "embedding_dimensions": 300,
        },
    )


def test_powered_taskset_is_balanced_broad_and_content_addressed(tmp_path) -> None:
    document = _document()
    assert verify_taskset(document)
    assert len(document["tasks"]) == TASK_COUNT
    assert Counter(task["family"] for task in document["tasks"]) == {
        family: 160 for family in FAMILIES
    }
    assert len({task["anchor_ingredient"] for task in document["tasks"]}) == TASK_COUNT
    for family in FAMILIES:
        tasks = [task for task in document["tasks"] if task["family"] == family]
        assert Counter(task["difficulty_band"] for task in tasks) == {
            1: 40,
            2: 40,
            3: 40,
            4: 40,
        }
        assert Counter(task["expected_choice"] for task in tasks) == {
            label: 40 for label in CHOICE_LABELS
        }
        assert len({task["primary_category"] for task in tasks}) >= 10

    path = write_taskset(document, tmp_path)
    assert path.name == f"epicure-native-powered-taskset-{document['artifact_sha256']}.json"
    assert write_taskset(document, tmp_path) == path


def test_powered_taskset_is_deterministic() -> None:
    assert _document() == _document()


def test_powered_scoring_uses_the_last_exact_marker() -> None:
    task = _document()["tasks"][0]
    expected = task["expected_choice"]
    assert parse_final_choice(f"FINAL_CHOICE: A\nFINAL_CHOICE: {expected}") == expected
    assert score_answer(task, f"FINAL_CHOICE: {expected}")["score"] == 1
    assert score_answer(task, "The answer is A")["score"] == 0
