from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import flavourbench.task_validation_runtime as task_validation_runtime
from flavourbench.main import app


def _wait_for(path: Path, *, directory_count: int | None = None) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if directory_count is None and path.exists():
            return
        if directory_count is not None and len(tuple(path.glob("ready-*"))) >= directory_count:
            return
        time.sleep(0.01)
    raise RuntimeError("cross-process task-validation barrier timed out")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--body-json", required=True)
    parser.add_argument("--barrier-directory", type=Path, required=True)
    parser.add_argument("--order", choices=("first", "second"), required=True)
    args = parser.parse_args()

    original = task_validation_runtime._append_event_locked

    def ordered_append(*append_args: Any, **append_kwargs: Any) -> Any:
        args.barrier_directory.mkdir(parents=True, exist_ok=True)
        (args.barrier_directory / f"ready-{args.order}").write_text("ready\n", encoding="utf-8")
        _wait_for(args.barrier_directory, directory_count=2)
        completed = args.barrier_directory / "first-completed"
        if args.order == "second":
            _wait_for(completed)
            return original(*append_args, **append_kwargs)
        try:
            return original(*append_args, **append_kwargs)
        finally:
            completed.write_text("complete\n", encoding="utf-8")

    task_validation_runtime._append_event_locked = ordered_append
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            f"/v1/expert/task-validation/candidates/{args.candidate}/ballots",
            headers={
                "X-FlavourBench-Service-Token": "test-service-token",
                "Authorization": f"Bearer {args.token}",
                "Idempotency-Key": args.key,
            },
            json=json.loads(args.body_json),
        )
    print(json.dumps({"status_code": response.status_code}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
