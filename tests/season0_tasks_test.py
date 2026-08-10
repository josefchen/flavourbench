from __future__ import annotations

from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json, sha256_text
from flavourbench.season0_tasks import (
    FAMILIES,
    CuratedSource,
    TaskFreezeError,
    build_pi_review_queue,
    freeze_task_bank,
)


def _source(per_family: int = 2) -> CuratedSource:
    candidates = []
    strict: dict[str, list[dict[str, object]]] = {family: [] for family in FAMILIES}
    question_id = 100
    for family in FAMILIES:
        for offset in range(per_family):
            prompt = f"Real human question {question_id} about {family}?"
            candidate = {
                "question_id": question_id,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "source": {
                    "human_origin": True,
                    "url": f"https://example.test/q/{question_id}",
                    "score_at_import": offset,
                },
                "human_reference": {
                    "accepted": True,
                    "text": f"Human answer {question_id}",
                    "score_at_import": offset,
                },
                "heuristics": {"recent": offset == 1},
            }
            candidate["record_sha256"] = sha256_json(candidate)
            candidates.append(candidate)
            judgments = {
                curator: {
                    "question_id": question_id,
                    "include": True,
                    "family": family,
                    "family_fit": 4,
                    "difficulty": 3,
                    "specificity": 3,
                    "epicure_relevance": 4,
                    "specialist_risk": "none",
                    "self_contained": True,
                    "requires_multi_step": offset == 1,
                }
                for curator in ("curator-a", "curator-b")
            }
            strict[family].append(
                {
                    "question_id": question_id,
                    "title": prompt,
                    "mean_curation_score": 13 + offset,
                    "judgments": judgments,
                }
            )
            question_id += 1
    pool = {
        "synthetic_tasks": 0,
        "curation_answers_visible_to_models": False,
        "candidates": candidates,
    }
    pool["content_sha256"] = "a" * 64
    audit = {
        "synthetic_tasks": 0,
        "candidate_pool_sha256": pool["content_sha256"],
        "curator_ids": ["curator-a", "curator-b"],
        "strict_consensus": strict,
    }
    audit["artifact_sha256"] = sha256_json(audit)
    return CuratedSource(Path("pool.json"), Path("audit.json"), pool, audit)


def test_freeze_selects_real_strict_unique_tasks_and_builds_blinded_queue() -> None:
    task_bank = freeze_task_bank([_source()], per_family=1)
    assert task_bank["counts"] == {
        "total": 4,
        "per_family": {family: 1 for family in FAMILIES},
        "accepted_human_references": 4,
        "synthetic": 0,
    }
    assert all(task["source_question_id"] % 2 == 1 for task in task_bank["tasks"])
    assert len({task["prompt_sha256"] for task in task_bank["tasks"]}) == 4
    queue = build_pi_review_queue(task_bank)
    assert len(queue["items"]) == 4
    assert "human_reference" not in queue["items"][0]
    assert queue["reviewer"]["compensation"] == "unpaid_volunteer"


def test_freeze_refuses_insufficient_family_instead_of_padding() -> None:
    with pytest.raises(TaskFreezeError, match="strict tasks"):
        freeze_task_bank([_source(per_family=1)], per_family=2)
