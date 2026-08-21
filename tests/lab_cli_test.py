from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flavourbench import lab_cli
from flavourbench.lab import verify_report

REPOSITORY = Path(__file__).resolve().parents[1]


def test_limit_run_checkpoints_a_partial_balanced_smoke_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    async def fake_runner(tasks, **kwargs: Any):
        calls.append([str(task["task_id"]) for task in tasks])
        rows = []
        for task in tasks:
            row = {
                "schema_version": "flavourbench-lab-response-v1",
                "task_id": task["task_id"],
                "status": "completed",
                "response": "FINAL_SELECTION: " + ",".join(task["optimal_selection"]),
                "model": kwargs["model"],
                "backend": "openai_compatible",
            }
            rows.append(row)
            kwargs["on_result"](row)
        return rows

    monkeypatch.setattr(lab_cli, "run_openai_compatible", fake_runner)
    responses = tmp_path / "responses.jsonl"
    report_path = tmp_path / "report.json"
    arguments = [
        "run",
        "--tasks",
        str(REPOSITORY / "hf/dataset/data-lab/train_tasks.jsonl"),
        "--backend",
        "openai-compatible",
        "--model",
        "lab/checkpoint",
        "--responses",
        str(responses),
        "--report",
        str(report_path),
        "--limit",
        "8",
        "--no-inference",
    ]
    args = lab_cli.build_parser().parse_args(arguments)
    assert args.handler(args) == 0

    response_rows = [json.loads(line) for line in responses.read_text().splitlines()]
    report = json.loads(report_path.read_text())
    assert len(response_rows) == 8
    assert report["comparable"] is False
    assert report["coverage"]["valid"] == 8
    assert report["diagnostic_valid_score"] == 100.0
    assert verify_report(report) == report["artifact_sha256"]

    resumed = lab_cli.build_parser().parse_args([*arguments, "--resume"])
    assert resumed.handler(resumed) == 0
    assert len(calls) == 1
