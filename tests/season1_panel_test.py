from __future__ import annotations

from copy import deepcopy

import pytest

from flavourbench.season1_panel import Season1PanelError, _verify_openrouter


def _entry() -> dict[str, str]:
    return {
        "model_id": "publisher/model",
        "canonical_slug": "publisher/model-20260731",
        "provider_slug": "provider/fp8",
    }


def _artifact() -> dict[str, object]:
    return {
        "requested_model_id": "publisher/model",
        "canonical_slug": "publisher/model-20260731",
        "requested_provider_slug": "provider/fp8",
        "returned_provider_name": "Provider",
        "real_provider_calls": 2,
        "real_epicure_calls": 1,
        "finish_reason": "stop",
        "output_json": {"answer_markdown": "A complete answer."},
        "generation_ids": ["gen-1", "gen-2"],
        "generation_accounting": [
            {
                "generation_id": "gen-1",
                "model": "publisher/model-20260731",
                "provider_name": "Provider",
                "reconciled": True,
                "total_cost_usd": "0.01",
            },
            {
                "generation_id": "gen-2",
                "model": "publisher/model-20260731",
                "provider_name": "Provider",
                "reconciled": True,
                "total_cost_usd": "0.02",
            },
        ],
        "cost_usd": "0.03",
        "provider_structured_output_required": False,
    }


def test_openrouter_panel_receipt_requires_normal_completion() -> None:
    artifact = deepcopy(_artifact())
    artifact["finish_reason"] = "length"

    with pytest.raises(Season1PanelError, match="call contract"):
        _verify_openrouter(_entry(), artifact)


def test_openrouter_panel_receipt_records_structured_output_parity() -> None:
    identity = _verify_openrouter(_entry(), _artifact())

    assert identity["provider_structured_output_required"] is False
