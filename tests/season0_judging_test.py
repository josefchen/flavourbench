from __future__ import annotations

from decimal import Decimal

import pytest

from flavourbench.season0_judging import (
    BEDROCK_CONNECT_TIMEOUT_SECONDS,
    BEDROCK_READ_TIMEOUT_SECONDS,
    BEDROCK_TOTAL_MAX_ATTEMPTS,
    JudgingError,
    JudgmentBudget,
    _bedrock_transport_config,
    _orphan_terminal_record,
    build_work_items,
)


def test_work_items_skip_unjudgable_and_cover_two_orientations() -> None:
    comparisons = {
        "artifact_sha256": "c" * 64,
        "comparisons": [
            {"comparison_id": "pair-1", "judgable": True},
            {"comparison_id": "pair-2", "judgable": False},
        ],
    }
    judges = {
        "artifact_sha256": "j" * 64,
        "judges": [
            {"judge_id": "judge-1"},
            {"judge_id": "judge-2"},
            {"judge_id": "judge-3"},
        ],
    }
    rows = build_work_items(comparisons, judges)
    assert len(rows) == 6
    assert {row.orientation for row in rows} == {"original", "swapped"}
    assert len({row.judgment_id for row in rows}) == 6


def test_bedrock_judging_transport_allows_slow_models_without_hidden_retries() -> None:
    config = _bedrock_transport_config(37)
    assert config.connect_timeout == BEDROCK_CONNECT_TIMEOUT_SECONDS
    assert config.read_timeout == BEDROCK_READ_TIMEOUT_SECONDS
    assert config.max_pool_connections == 37
    assert config.retries["total_max_attempts"] == BEDROCK_TOTAL_MAX_ATTEMPTS == 1


@pytest.mark.asyncio
async def test_judgment_budget_hard_admission_threshold() -> None:
    budget = JudgmentBudget(Decimal("100"), committed=Decimal("84"))
    with pytest.raises(JudgingError):
        await budget.reserve("too-much", Decimal("1"))
    await budget.reserve("allowed", Decimal("0.5"))
    await budget.finalize("allowed", Decimal("0.1"))
    assert budget.committed == Decimal("84.1")


def test_orphaned_request_becomes_terminal_uncertain_without_resend() -> None:
    comparisons = {
        "artifact_sha256": "c" * 64,
        "comparisons": [
            {
                "comparison_id": "pair-1",
                "judgable": True,
                "track": "model_arena",
                "task_id": "task-1",
                "task_family": "composition",
            }
        ],
    }
    judges = {
        "artifact_sha256": "j" * 64,
        "judges": [{"judge_id": "judge-1", "display_name": "Judge One"}],
    }
    item = build_work_items(comparisons, judges)[0]
    event = {
        "judgment_id": item.judgment_id,
        "comparison_id": "pair-1",
        "judge_id": "judge-1",
        "orientation": item.orientation,
        "request_payload_sha256": "r" * 64,
        "reservation_usd": "0.125000000",
        "started_at": "2026-07-16T12:00:00Z",
    }
    record = _orphan_terminal_record(
        item=item,
        event=event,
        prompt_sha256="p" * 64,
        task_bank_sha256="t" * 64,
        comparison_manifest_sha256="c" * 64,
        judge_manifest_sha256="j" * 64,
        judge_cost_envelope_sha256="e" * 64,
        completed_at="2026-07-16T12:01:00Z",
    )
    assert record["status"] == "failed"
    assert record["delivery_state"] == "uncertain"
    assert record["error_type"] == "OrphanedRequestEvent"
    assert record["reservation_usd"] == "0.125000000"
    assert record["result"]["estimated_cost_usd"] is None
    assert record["synthetic"] is False
