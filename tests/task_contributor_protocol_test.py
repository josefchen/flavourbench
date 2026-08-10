from __future__ import annotations

import hashlib
from pathlib import Path

from flavourbench.season1_readiness import CONTROL_PLANE_SQL
from flavourbench.task_contributor_protocol import (
    PROTOCOL_SCOPE,
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    protocol_binding_active,
    protocol_text,
)

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_and_governance_protocol_bytes_are_identical_and_content_addressed() -> None:
    runtime_text = protocol_text()
    governance_text = (ROOT / "protocol/FLAVOURBENCH-TASK-CONTRIBUTOR-v2.md").read_text(
        encoding="utf-8"
    )
    assert runtime_text == governance_text
    assert hashlib.sha256(runtime_text.encode("utf-8")).hexdigest() == PROTOCOL_SHA256
    assert f"Protocol version: `{PROTOCOL_VERSION}`" in runtime_text


def test_protocol_binding_requires_exact_version_digest_scope_and_acceptance_event() -> None:
    complete = {
        "task_contributor_status": "active",
        "task_contributor_protocol_version": PROTOCOL_VERSION,
        "task_contributor_protocol_sha256": PROTOCOL_SHA256,
        "task_contributor_protocol_scope": PROTOCOL_SCOPE,
        "task_contributor_protocol_accepted": True,
        "task_contributor_protocol_acceptance_event_id": "event-id",
    }
    assert protocol_binding_active(complete) is True
    for field in complete:
        changed = dict(complete)
        changed.pop(field)
        assert protocol_binding_active(changed) is False


def test_readiness_query_is_bound_to_current_protocol_without_unresolved_tokens() -> None:
    assert "__TASK_CONTRIBUTOR_PROTOCOL_" not in CONTROL_PLANE_SQL
    assert PROTOCOL_VERSION in CONTROL_PLANE_SQL
    assert PROTOCOL_SHA256 in CONTROL_PLANE_SQL
    assert PROTOCOL_SCOPE in CONTROL_PLANE_SQL
