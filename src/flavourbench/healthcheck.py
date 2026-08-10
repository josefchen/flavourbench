from __future__ import annotations

import json
import sys

from .config import get_settings
from .database import SessionLocal, database_readiness


def run() -> None:
    settings = get_settings()
    expected_role = {
        "api": "flavourbench_api",
        "worker": "flavourbench_worker",
        "migration": "flavourbench_owner",
    }[settings.service_role]
    with SessionLocal() as session:
        result = database_readiness(session, expected_role=expected_role)
    print(json.dumps({"status": "ready", **result}, sort_keys=True))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"not ready: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
