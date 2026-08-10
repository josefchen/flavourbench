"""Run one immutable Epicure off/on pair through Cohere Chat V2."""

from __future__ import annotations

import asyncio
import json
import sys

from .direct_kimi_pair import _parser, _run_direct_pair
from .execution_policy import assert_legacy_paid_cli_allowed
from .service_cohere import (
    COHERE_ACCOUNTING_BASIS,
    CohereDirectProvider,
)


async def run_pair(args):  # type: ignore[no-untyped-def]
    return await _run_direct_pair(
        args,
        execution_backend="cohere_direct",
        provider_factory=CohereDirectProvider,
        credential_attribute="cohere_api_key",
        accounting_basis=COHERE_ACCOUNTING_BASIS,
        provider_label="Cohere",
        allow_zero_cap=True,
    )


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-direct-cohere-pair")
    try:
        args = _parser(__doc__, reasoning_required=False).parse_args()
        if args.frozen_attempt_slots_json:
            slots = json.loads(args.frozen_attempt_slots_json)
            if not isinstance(slots, list):
                raise RuntimeError("frozen attempt slots JSON must decode to an array")
            args.frozen_attempt_slots = slots
        summary = asyncio.run(run_pair(args))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] not in {
        "complete_rate_card_estimated",
        "preflight_passed_no_provider_calls",
    }:
        sys.exit(2)


if __name__ == "__main__":
    run()
