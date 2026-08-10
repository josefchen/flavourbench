from __future__ import annotations

from flavourbench.real_task_bank import (
    SelectionPolicy,
    html_to_text,
    select_tasks,
    sha256_json,
)


def _question(question_id: int, family_tag: str, created: int) -> dict[str, object]:
    return {
        "question_id": question_id,
        "accepted_answer_id": 10_000 + question_id,
        "is_answered": True,
        "title": f"Question {question_id}",
        "body": "<p>This is a sufficiently detailed real culinary question body.</p>",
        "creation_date": created,
        "last_activity_date": created + 60,
        "link": f"https://cooking.stackexchange.com/questions/{question_id}/example",
        "content_license": "CC BY-SA 4.0",
        "owner": {"display_name": "Cook", "user_id": question_id},
        "score": question_id,
        "answer_count": 1,
        "tags": [family_tag],
    }


def _answer(answer_id: int, created: int) -> dict[str, object]:
    return {
        "answer_id": answer_id,
        "body": "<p>This accepted human answer is long enough to be a hidden reference.</p>",
        "creation_date": created,
        "last_activity_date": created + 60,
        "content_license": "CC BY-SA 4.0",
        "owner": {"display_name": "Answerer", "user_id": answer_id},
        "score": 3,
    }


def test_html_to_text_preserves_blocks_and_decodes_entities() -> None:
    assert html_to_text("<p>Salt &amp; acid</p><ul><li>first</li><li>second</li></ul>") == (
        "Salt & acid\n\nfirst\n\nsecond"
    )


def test_select_tasks_is_balanced_unique_and_deterministic() -> None:
    tags = {
        "substitution": "substitutions",
        "composition": "flavor",
        "cookability": "baking",
        "evidence": "food-science",
    }
    questions: list[dict[str, object]] = []
    answers: dict[int, dict[str, object]] = {}
    question_id = 1
    for tag in tags.values():
        for offset in range(3):
            created = 1_735_689_600 + offset
            question = _question(question_id, tag, created)
            questions.append(question)
            answer_id = int(question["accepted_answer_id"])
            answers[answer_id] = _answer(answer_id, created)
            question_id += 1

    policy = SelectionPolicy(per_family=2, min_recent_per_family=1)
    first = select_tasks(questions, answers, policy=policy)
    second = select_tasks(reversed(questions), answers, policy=policy)

    assert first == second
    assert len(first) == 8
    assert len({task["source"]["question_id"] for task in first}) == 8
    assert {family: sum(task["family"] == family for task in first) for family in tags} == {
        family: 2 for family in tags
    }
    assert sha256_json(first) == sha256_json(second)


def test_select_tasks_excludes_specialist_tags() -> None:
    created = 1_735_689_600
    question = _question(1, "substitutions", created)
    question["tags"] = ["substitutions", "food-safety"]
    answer_id = int(question["accepted_answer_id"])
    policy = SelectionPolicy(per_family=1, min_recent_per_family=0)

    try:
        select_tasks([question], {answer_id: _answer(answer_id, created)}, policy=policy)
    except ValueError as exc:
        assert "0 eligible unique tasks" in str(exc)
    else:
        raise AssertionError("specialist-tagged question should not be selected")
