from __future__ import annotations

import json

from flavourbench.task_curation import (
    _cohen_kappa,
    build_batch_prompt,
    parse_judgments,
)


def _candidate(question_id: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "title": "How can I balance acid and sweetness in this sauce?",
        "prompt": (
            "How can I balance acid and sweetness in this sauce?\n\n"
            "It contains tomatoes and honey."
        ),
        "source": {"tags": ["sauce", "flavor"], "created_utc": "2026-01-01T00:00:00Z"},
    }


def _judgment(question_id: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "include": True,
        "family": "composition",
        "family_fit": 4,
        "difficulty": 3,
        "specificity": 3,
        "epicure_relevance": 4,
        "specialist_risk": "none",
        "self_contained": True,
        "requires_multi_step": True,
        "rationale": "Requires balancing several interacting flavour components.",
    }


def test_build_batch_prompt_omits_hidden_reference() -> None:
    candidate = _candidate(7)
    candidate["human_reference"] = {"text": "secret accepted answer"}
    prompt = build_batch_prompt([candidate])
    assert "secret accepted answer" not in prompt
    assert '"question_id":7' in prompt


def test_parse_judgments_accepts_fenced_json_and_preserves_order() -> None:
    raw = "```json\n" + json.dumps([_judgment(7), _judgment(8)]) + "\n```"
    parsed = parse_judgments(raw, [7, 8])
    assert [item["question_id"] for item in parsed] == [7, 8]
    assert all(item["family"] == "composition" for item in parsed)


def test_parse_judgments_records_and_repairs_logical_include_contradiction() -> None:
    judgment = _judgment(7)
    judgment["specialist_risk"] = "food_safety"
    parsed = parse_judgments(json.dumps([judgment]), [7])[0]
    assert parsed["include_reported"] is True
    assert parsed["include"] is False
    assert parsed["include_consistent"] is False


def test_cohen_kappa_recovers_perfect_and_chance_adjusted_agreement() -> None:
    assert _cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert _cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
