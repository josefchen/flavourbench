from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from flavourbench.lab import (
    LabValidationError,
    read_json_records,
    reward,
    score_submission,
    semantic_sha256,
    trl_reward,
    validate_tasks,
    verify_report,
)
from flavourbench.lab_cli import _smoke_tasks

REPOSITORY = Path(__file__).resolve().parents[1]
TASKS_PATH = REPOSITORY / "hf/dataset/data-lab/train_tasks.jsonl"


def _tasks() -> list[dict[str, object]]:
    return read_json_records(TASKS_PATH)


def _optimal_responses(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "flavourbench-lab-response-v1",
            "task_id": task["task_id"],
            "status": "completed",
            "response": "FINAL_SELECTION: " + ",".join(str(task["optimal_selection"])),
        }
        for task in tasks
    ]


def test_lab_scores_a_complete_content_addressed_submission() -> None:
    tasks = _tasks()
    validate_tasks(tasks)
    responses = _optimal_responses(tasks)
    report = score_submission(
        tasks,
        responses,
        bootstrap_resamples=200,
        sign_flip_resamples=400,
        seed=7,
    )

    assert len(tasks) == 270
    assert report["comparable"] is True
    assert report["flavourbench_score"] == 100.0
    assert report["coverage"] == {
        "tasks": 270,
        "submitted": 270,
        "valid": 270,
        "missing": 0,
        "invalid": 0,
        "fraction_valid": 1.0,
    }
    assert report["inference"]["confidence_interval_95"] == [100.0, 100.0]
    assert report["exact_chance_score"] < 100.0
    assert report["inference"]["mean_difference_from_exact_chance"] > 0
    artifact = report.pop("artifact_sha256")
    assert artifact == semantic_sha256(report)


def test_lab_withholds_score_for_missing_or_invalid_cells() -> None:
    tasks = _tasks()
    responses = _optimal_responses(tasks)
    responses[0]["response"] = "I cannot decide."
    responses.pop()
    report = score_submission(tasks, responses, include_inference=False)

    assert report["comparable"] is False
    assert report["flavourbench_score"] is None
    assert report["coverage"]["valid"] == 268
    assert report["coverage"]["invalid"] == 1
    assert report["coverage"]["missing"] == 1
    assert report["diagnostic_valid_score"] == 100.0
    assert report["inference"] is None


def test_lab_rejects_duplicate_cells_and_bad_inference_controls() -> None:
    tasks = _tasks()
    responses = _optimal_responses(tasks)
    with pytest.raises(LabValidationError, match="duplicated"):
        score_submission(tasks, [responses[0], responses[0]], include_inference=False)
    with pytest.raises(LabValidationError, match="positive"):
        score_submission(tasks, responses, bootstrap_resamples=0)


def test_smoke_sample_is_family_balanced_and_never_a_complete_score() -> None:
    tasks = _tasks()
    sample = _smoke_tasks(tasks, 8)
    assert [task["family"] for task in sample] == [
        "substitution",
        "pairing",
        "constraint",
        "substitution",
        "pairing",
        "constraint",
        "substitution",
        "pairing",
    ]
    responses = _optimal_responses(sample)
    report = score_submission(tasks, responses, include_inference=False)
    assert report["comparable"] is False
    assert report["flavourbench_score"] is None
    assert report["coverage"]["valid"] == 8
    assert report["diagnostic_valid_score"] == 100.0


def test_dense_reward_and_trl_adapter_use_the_same_map() -> None:
    task = _tasks()[0]
    optimum = str(task["optimal_selection"])
    completion = "FINAL_SELECTION: " + ",".join(optimum)
    assert reward(task, completion) == 1.0
    assert trl_reward(
        [completion, [{"role": "assistant", "content": completion}]],
        [task["selection_scores_bps"], task["selection_scores_bps"]],
        [task["choices"], task["choices"]],
    ) == [1.0, 1.0]


def test_lab_json_schemas_accept_the_published_contracts() -> None:
    tasks = _tasks()
    responses = _optimal_responses(tasks)
    response_schema = json.loads(
        (REPOSITORY / "schemas/lab-response-v1.schema.json").read_text(encoding="utf-8")
    )
    report_schema = json.loads(
        (REPOSITORY / "schemas/lab-report-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(response_schema).validate(responses[0])
    report = score_submission(tasks, responses, include_inference=False)
    Draft202012Validator(report_schema).validate(report)
    assert verify_report(report) == report["artifact_sha256"]
    payload = dict(report)
    recorded = payload.pop("artifact_sha256")
    observed = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert observed == recorded

    report["coverage"]["valid"] -= 1
    with pytest.raises(LabValidationError, match="digest"):
        verify_report(report)
