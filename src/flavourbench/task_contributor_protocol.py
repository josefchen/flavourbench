from __future__ import annotations

import hashlib
from pathlib import Path

PROTOCOL_VERSION = "flavourbench-human-task-contributor-v2"
PROTOCOL_SHA256 = "5e00d8ee71767d4c01ad9a07c733f169ae57085ffe809fd5f03e9c3c7a5f6fa5"
PROTOCOL_SCOPE = "task_authorship_and_redistribution_only"
PROTOCOL_RELATIVE_PATH = "contracts/season1/task-contributor-protocol-v2.md"


class TaskContributorProtocolError(RuntimeError):
    pass


def _protocol_path() -> Path:
    candidates = (
        Path.cwd().resolve() / PROTOCOL_RELATIVE_PATH,
        Path(__file__).resolve().parents[2] / PROTOCOL_RELATIVE_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TaskContributorProtocolError(
        f"task-contributor protocol is unavailable at {PROTOCOL_RELATIVE_PATH}"
    )


def protocol_text() -> str:
    try:
        value = _protocol_path().read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskContributorProtocolError(
            f"task-contributor protocol is unavailable at {PROTOCOL_RELATIVE_PATH}"
        ) from exc
    observed = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if observed != PROTOCOL_SHA256:
        raise TaskContributorProtocolError(
            "task-contributor protocol bytes do not match the frozen digest"
        )
    return value


def protocol_binding_active(profile: dict[str, object]) -> bool:
    return bool(
        profile.get("task_contributor_status") == "active"
        and profile.get("task_contributor_protocol_version") == PROTOCOL_VERSION
        and profile.get("task_contributor_protocol_sha256") == PROTOCOL_SHA256
        and profile.get("task_contributor_protocol_scope") == PROTOCOL_SCOPE
        and profile.get("task_contributor_protocol_accepted") is True
        and isinstance(profile.get("task_contributor_protocol_acceptance_event_id"), str)
    )
