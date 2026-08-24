from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from flavourbench.epicure_native_taskset_v3 import (
    FAMILIES,
    TASK_COUNT,
    parse_final_choice,
    score_answer,
    verify_taskset,
)

ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> dict:
    path = next((ROOT / "benchmark/powered-v3/taskset").glob("*.json"))
    return json.loads(path.read_text())


def test_materialized_construct_validated_taskset() -> None:
    artifact = _artifact()
    assert verify_taskset(artifact)
    assert len(artifact["tasks"]) == TASK_COUNT
    assert Counter(task["family"] for task in artifact["tasks"]) == Counter(
        {family: 160 for family in FAMILIES}
    )
    assert len({task["anchor_ingredient"] for task in artifact["tasks"]}) == 640
    assert all(task["construct_validity"] for task in artifact["tasks"])


def test_oracle_validity_rules_are_machine_checkable() -> None:
    artifact = _artifact()
    for task in artifact["tasks"]:
        validity = task["construct_validity"]
        assert all(value is True for value in validity.values() if isinstance(value, bool))
        if task["family"] == "provenance":
            reference = task["oracle_reference"]
            assert reference["recorded_region"] == reference["ranked_directions"][0]["region"]
        if task["family"] == "constraint":
            choices = task["oracle_reference"]["choices"]
            if task["oracle_reference"]["property"] == "is_vegan":
                target = task["choices"][task["expected_choice"]]
                desired = task["oracle_reference"]["target_value"]
                assert next(row for row in choices if row["name"] == target)["is_vegan"] is desired


def test_exact_marker_scoring() -> None:
    task = _artifact()["tasks"][0]
    assert parse_final_choice("analysis\nFINAL_CHOICE: b") == "B"
    expected = task["expected_choice"]
    assert score_answer(task, f"FINAL_CHOICE: {expected}")["score"] == 1
    assert score_answer(task, "I refuse")["score"] == 0
