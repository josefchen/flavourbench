from __future__ import annotations

import pytest

from flavourbench.season0_judge_protocol import (
    DIMENSIONS,
    JudgmentProtocolError,
    build_judge_prompt,
    normalize_choice,
    validate_judgment,
)


def _valid() -> dict[str, object]:
    side = {
        "scores": {dimension: 4 for dimension in DIMENSIONS},
        "fatal_failure": False,
        "summary": "A practical and coherent answer.",
    }
    return {
        "choice": "left",
        "left": side,
        "right": side,
        "confidence": "high",
        "reason_tags": ["none"],
        "rationale": "The left answer addresses the task more directly.",
    }


def test_swap_prompt_and_choice_are_symmetric() -> None:
    task = {
        "family": "substitution",
        "prompt": "Question",
        "human_reference": {"text": "Reference"},
    }
    original = build_judge_prompt(
        task=task, left_answer="AAA", right_answer="BBB", orientation="original"
    )
    swapped = build_judge_prompt(
        task=task, left_answer="AAA", right_answer="BBB", orientation="swapped"
    )
    assert '"candidate_left": "AAA"' in original
    assert '"candidate_left": "BBB"' in swapped
    assert normalize_choice("left", "swapped") == "right"
    assert normalize_choice("tie", "swapped") == "tie"


def test_judgment_validation_rejects_out_of_range_scores() -> None:
    valid = _valid()
    assert validate_judgment(valid)["choice"] == "left"
    valid["left"]["scores"]["cookability"] = 6  # type: ignore[index]
    with pytest.raises(JudgmentProtocolError):
        validate_judgment(valid)
