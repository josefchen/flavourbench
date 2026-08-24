from __future__ import annotations

import json
from decimal import Decimal

from flavourbench.real_task_bank import sha256_json, sha256_text
from flavourbench.season0_collection import (
    CONDITIONS,
    FAMILIES,
    _existing_collection_state,
    _unhandled_delivery_state,
    build_work_items,
)


def _task_bank() -> dict:
    tasks = []
    for family in FAMILIES:
        for index in range(2):
            prompt = f"Real {family} prompt {index}"
            task = {
                "task_id": f"{family}-{index}",
                "family": family,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "task_sha256": sha256_json([family, index]),
                "source_question_id": len(tasks) + 1,
            }
            tasks.append(task)
    return {
        "synthetic_tasks": 0,
        "task_set_sha256": "a" * 64,
        "tasks": tasks,
    }


def _manifest() -> dict:
    models = [
        {
            "season_model_id": f"m{index}",
            "canonical_model_id": f"lab/model-{index}",
            "provider": "bedrock" if index < 7 else "openrouter",
        }
        for index in range(12)
    ]
    return {
        "task_set_sha256": "a" * 64,
        "model_set_sha256": "b" * 64,
        "models": models,
    }


def test_work_items_are_dense_paired_real_arms() -> None:
    items = build_work_items(_task_bank(), _manifest(), phase="calibration", per_family=1)
    assert len(items) == 4 * 12 * 2
    assert {item.condition for item in items} == set(CONDITIONS)
    assert len({item.arm_id for item in items}) == len(items)
    counts = {
        model_id: sum(item.model["season_model_id"] == model_id for item in items)
        for model_id in {item.model["season_model_id"] for item in items}
    }
    assert set(counts.values()) == {8}


def test_resume_quarantines_orphaned_started_request_and_seeds_budget(tmp_path) -> None:
    arms = tmp_path / "arms"
    events = tmp_path / "events"
    arms.mkdir()
    events.mkdir()
    completed = {
        "arm_id": "completed",
        "completed_at": "2026-07-16T00:00:00Z",
        "reservation_usd": "1",
        "model": {"provider": "openrouter"},
        "result": {"actual_cost_usd": "0.25"},
    }
    (arms / "arm-completed-a.json").write_text(json.dumps(completed))
    orphan = {
        "arm_id": "orphan",
        "provider": "bedrock",
        "reservation_usd": "1",
    }
    (events / "event-orphan-request-started-a.json").write_text(json.dumps(orphan))

    terminal, orphaned, spent = _existing_collection_state(arms, events)

    assert terminal == {"completed"}
    assert orphaned == {"orphan"}
    assert spent == {"bedrock": Decimal("1"), "openrouter": Decimal("0.25")}


def test_unhandled_read_timeout_is_uncertain_delivery() -> None:
    read_timeout_error = type("ReadTimeoutError", (Exception,), {})
    assert _unhandled_delivery_state(read_timeout_error()) == "uncertain"
    assert _unhandled_delivery_state(ConnectionError()) == "safe_pre_inference"
