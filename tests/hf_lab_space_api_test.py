from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from flavourbench.lab import read_json_records, score_submission

REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "hf/space/lab_api.py"
SPEC = importlib.util.spec_from_file_location("hf_space_lab_api", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SPACE_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPACE_API)


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks = read_json_records(REPOSITORY / "hf/dataset/data-lab/validation_tasks.jsonl")
    responses = [
        {
            "task_id": task["task_id"],
            "status": "completed",
            "response": "FINAL_SELECTION: " + ",".join(str(task["optimal_selection"])),
        }
        for task in tasks
    ]
    return tasks, responses


def test_space_batch_api_matches_the_local_score_contract() -> None:
    tasks, responses = _fixture()
    payload = "\n".join(json.dumps(row) for row in responses)
    space_report, rows = SPACE_API.score_payload(tasks, payload)
    local_report = score_submission(tasks, responses, include_inference=False)

    assert len(rows) == 84
    assert space_report["comparable"] is True
    assert space_report["flavourbench_score"] == 100.0
    assert space_report["coverage"] == local_report["coverage"]
    assert space_report["task_set_semantic_sha256"] == local_report["task_set_semantic_sha256"]
    assert (
        space_report["response_set_semantic_sha256"] == local_report["response_set_semantic_sha256"]
    )


def test_space_single_reward_and_partial_submission() -> None:
    tasks, responses = _fixture()
    task = tasks[0]
    completion = responses[0]["response"]
    scored = SPACE_API.score_completion({str(task["task_id"]): task}, task["task_id"], completion)
    assert scored["reward"] == 1.0
    report, _ = SPACE_API.score_payload(tasks, json.dumps(responses[:-1]))
    assert report["comparable"] is False
    assert report["flavourbench_score"] is None
    assert report["coverage"]["missing"] == 1


def test_space_batch_api_rejects_unknown_and_duplicate_tasks() -> None:
    tasks, responses = _fixture()
    with pytest.raises(SPACE_API.SpaceLabError, match="duplicate"):
        SPACE_API.score_payload(tasks, json.dumps([responses[0], responses[0]]))
    with pytest.raises(SPACE_API.SpaceLabError, match="unknown"):
        SPACE_API.score_payload(tasks, json.dumps([{"task_id": "not-a-task", "response": "x"}]))
