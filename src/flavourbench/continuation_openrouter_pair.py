"""Narrow frozen-identifier wrapper around the qualified OpenRouter pair runner."""

from __future__ import annotations

import asyncio
import json
import sys

from .execution_policy import assert_legacy_paid_cli_allowed
from .live_smoke import _parser, live_smoke


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-live-smoke")
    parser = _parser()
    parser.add_argument("--frozen-run-id", required=True)
    parser.add_argument("--frozen-attempt-slots-json", required=True)
    try:
        args = parser.parse_args()
        slots = json.loads(args.frozen_attempt_slots_json)
        if not isinstance(slots, list) or not slots:
            raise RuntimeError("frozen attempt slots JSON must decode to a non-empty array")
        args.frozen_attempt_slots = slots
        summary = asyncio.run(live_smoke(args))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    run()
