from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from flavourbench import lab_runner
from flavourbench.lab import LabValidationError


class _Response:
    status_code = 200
    request = httpx.Request("POST", "https://lab.invalid/v1/chat/completions")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "FINAL_SELECTION: A,B,C"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 8},
        }


class _Client:
    requests: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **_: Any) -> None:
        pass

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _Response:
        self.requests.append((url, json))
        return _Response()


def test_openai_compatible_runner_keeps_credentials_out_of_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("LAB_TEST_KEY", "not-a-real-secret")
    monkeypatch.setattr(lab_runner.httpx, "AsyncClient", _Client)
    _Client.requests.clear()
    tasks = [
        {"task_id": "task-1", "prompt": "Choose."},
        {"task_id": "task-2", "prompt": "Choose again."},
    ]
    checkpointed = []
    rows = asyncio.run(
        lab_runner.run_openai_compatible(
            tasks,
            model="lab/checkpoint",
            base_url="https://lab.invalid/v1",
            api_key_env="LAB_TEST_KEY",
            extra_body={"seed": 42},
            on_result=checkpointed.append,
        )
    )

    assert [row["task_id"] for row in rows] == ["task-1", "task-2"]
    assert {row["task_id"] for row in checkpointed} == {"task-1", "task-2"}
    assert rows[0]["status"] == "completed"
    assert rows[0]["response"] == "FINAL_SELECTION: A,B,C"
    assert rows[0]["usage"]["completion_tokens"] == 8
    assert "not-a-real-secret" not in json.dumps(rows)
    assert _Client.requests[-1][0] == "https://lab.invalid/v1/chat/completions"
    assert _Client.requests[-1][1]["seed"] == 42


def test_openai_compatible_runner_requires_named_environment_credential(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_LAB_KEY", raising=False)
    with pytest.raises(LabValidationError, match="unset"):
        asyncio.run(
            lab_runner.run_openai_compatible(
                [{"task_id": "task-1", "prompt": "Choose."}],
                model="lab/checkpoint",
                base_url="https://lab.invalid/v1",
                api_key_env="MISSING_LAB_KEY",
            )
        )


def test_openai_compatible_runner_rejects_remote_plain_http(monkeypatch) -> None:
    monkeypatch.setenv("LAB_TEST_KEY", "not-a-real-secret")
    with pytest.raises(LabValidationError, match="plain HTTP"):
        asyncio.run(
            lab_runner.run_openai_compatible(
                [{"task_id": "task-1", "prompt": "Choose."}],
                model="lab/checkpoint",
                base_url="http://lab.example/v1",
                api_key_env="LAB_TEST_KEY",
            )
        )
