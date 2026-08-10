from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from flavourbench.engine import has_unresolved_paid_attempt, is_complete_finish_reason
from flavourbench.models import GenerationAttempt


def _event(
    attempt_id: str,
    event_type: str,
    *,
    arm_id: str = "arm-1",
) -> GenerationAttempt:
    return cast(
        GenerationAttempt,
        SimpleNamespace(arm_id=arm_id, attempt_id=attempt_id, event_type=event_type),
    )


def test_completed_mcp_attempt_does_not_hold_model_budget() -> None:
    events = [
        _event("model-1", "request_started"),
        _event("model-1", "accounting_reconciled"),
        _event("mcp-1", "mcp_call_started"),
        _event("mcp-1", "mcp_call_completed"),
    ]

    assert has_unresolved_paid_attempt(events) is False


def test_incomplete_mcp_attempt_does_not_masquerade_as_paid_generation() -> None:
    assert has_unresolved_paid_attempt([_event("mcp-1", "mcp_call_started")]) is False


def test_accepted_model_attempt_without_accounting_holds_budget() -> None:
    events = [
        _event("model-1", "request_started"),
        _event("model-1", "response_received"),
        _event("mcp-1", "mcp_call_completed"),
    ]

    assert has_unresolved_paid_attempt(events) is True


def test_attempt_identifiers_are_partitioned_by_arm() -> None:
    events = [
        _event("shared", "accounting_reconciled", arm_id="arm-1"),
        _event("shared", "request_started", arm_id="arm-2"),
    ]

    assert has_unresolved_paid_attempt(events) is True


def test_only_normal_provider_stops_are_complete() -> None:
    for reason in ("stop", "end_turn", "stop_sequence", "completed"):
        assert is_complete_finish_reason(reason) is True
    for reason in (None, "", "length", "max_tokens", "content_filter", "unknown"):
        assert is_complete_finish_reason(reason) is False
