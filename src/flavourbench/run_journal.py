"""Durable append-only journals for paid or potentially paid live-smoke runs.

The provider's synchronous attempt sink is a write-ahead boundary: a
``request_started`` event is appended and fsynced before HTTP bytes may be
sent.  A crash therefore leaves enough local evidence to hold uncertain
requests, reconcile returned generation IDs, and avoid blind replay.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JOURNAL_SCHEMA_VERSION = "flavourbench-live-run-journal-v1"
FINAL_JOURNAL_PREFIX = "flavourbench-live-smoke-journal-"
IN_PROGRESS_SUFFIX = ".inprogress.jsonl"


class JournalIntegrityError(RuntimeError):
    """A run journal is malformed, secret-bearing, or fails its hash chain."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _event_digest(event: Mapping[str, Any]) -> str:
    unhashed = dict(event)
    unhashed.pop("entry_sha256", None)
    return _sha256(unhashed)


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "api_key",
        "authorization",
        "cloudflare_ai_gateway_token",
        "cookie",
        "environment",
        "headers",
        "mcp_token",
        "openrouter_api_key",
        "password",
        "raw_request",
        "request_payload",
        "response_body",
        "secret",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _load_lines(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise JournalIntegrityError(f"journal must be a regular non-symlink file: {path}")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise JournalIntegrityError(f"blank journal line at sequence {number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise JournalIntegrityError(f"invalid journal JSON at sequence {number}") from error
        if not isinstance(entry, dict):
            raise JournalIntegrityError(f"journal sequence {number} is not an object")
        if entry.get("schema_version") != JOURNAL_SCHEMA_VERSION:
            raise JournalIntegrityError(f"unsupported journal schema at sequence {number}")
        if entry.get("sequence") != number:
            raise JournalIntegrityError(f"journal sequence mismatch at line {number}")
        if entry.get("previous_entry_sha256") != previous:
            raise JournalIntegrityError(f"journal hash chain mismatch at sequence {number}")
        digest = entry.get("entry_sha256")
        if not isinstance(digest, str) or digest != _event_digest(entry):
            raise JournalIntegrityError(f"journal digest mismatch at sequence {number}")
        if _contains_forbidden_key(entry):
            raise JournalIntegrityError(f"journal sequence {number} contains a forbidden field")
        entries.append(entry)
        previous = digest
    if not entries or entries[0].get("event_type") != "run_started":
        raise JournalIntegrityError("journal must begin with run_started")
    run_id = entries[0].get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise JournalIntegrityError("journal has no run ID")
    if any(entry.get("run_id") != run_id for entry in entries):
        raise JournalIntegrityError("journal entries do not share one run ID")
    finalized = [entry for entry in entries if entry.get("event_type") == "run_finalized"]
    if len(finalized) > 1 or (finalized and entries[-1] is not finalized[0]):
        raise JournalIntegrityError("run_finalized must occur once and be the final event")
    return entries


def load_run_journal(path: str | Path) -> list[dict[str, Any]]:
    """Verify and return one in-progress or finalized journal."""

    return _load_lines(Path(path))


@dataclass(frozen=True)
class JournalDescriptor:
    filename: str
    sha256: str
    head_entry_sha256: str
    entry_count: int
    run_id: str
    finalized: bool

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "filename": self.filename,
            "sha256": self.sha256,
            "head_entry_sha256": self.head_entry_sha256,
            "entry_count": self.entry_count,
            "run_id": self.run_id,
            "finalized": self.finalized,
        }


@dataclass(frozen=True)
class JournalRecoveryState:
    path: Path
    journal_sha256: str
    head_entry_sha256: str
    entry_count: int
    run_id: str
    finalized: bool
    generation_ids: tuple[str, ...]
    unreconciled_generation_ids: tuple[str, ...]
    uncertain_attempt_ids: tuple[str, ...]
    mcp_trace_count: int
    recovery_action: str
    safe_to_replay: bool


def recovery_state(path: str | Path) -> JournalRecoveryState:
    """Classify crash evidence without contacting or replaying a provider."""

    journal_path = Path(path)
    entries = load_run_journal(journal_path)
    attempts: dict[str, set[str]] = {}
    generations: set[str] = set()
    reconciled: set[str] = set()
    mcp_count = 0
    for entry in entries:
        event_type = entry.get("event_type")
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event_type == "provider_attempt":
            attempt_id = str(payload.get("attempt_id") or "")
            provider_event = str(payload.get("event_type") or "")
            if attempt_id and provider_event:
                attempts.setdefault(attempt_id, set()).add(provider_event)
            generation_id = str(payload.get("generation_id") or "")
            if provider_event == "response_received" and generation_id:
                generations.add(generation_id)
            metadata = payload.get("metadata")
            if (
                provider_event == "accounting_reconciled"
                and generation_id
                and isinstance(metadata, Mapping)
                and metadata.get("reconciled") is True
            ):
                reconciled.add(generation_id)
        elif event_type == "mcp_trace":
            mcp_count += 1
    safe_terminals = {"pre_send_failure", "request_rejected", "retry_scheduled"}
    uncertain = sorted(
        attempt_id
        for attempt_id, states in attempts.items()
        if "request_started" in states
        and not (
            states & safe_terminals
            or "response_received" in states
            or "uncertain_delivery" in states
        )
        or "uncertain_delivery" in states
    )
    unreconciled = sorted(generations - reconciled)
    finalized = entries[-1].get("event_type") == "run_finalized"
    pre_provider_failure = not attempts
    safe_to_replay = pre_provider_failure or (
        not generations
        and not uncertain
        and all(bool(states & safe_terminals) for states in attempts.values())
    )
    if unreconciled:
        action = "reconcile_generation_ids_before_any_retry"
    elif uncertain:
        action = "hold_uncertain_delivery_and_reconcile_provider_state"
    elif finalized:
        action = "verify_linked_artifact"
    elif pre_provider_failure:
        action = "pre_provider_failure_safe_under_parent_reservation_policy"
    elif safe_to_replay:
        action = "new_attempt_permitted_only_under_parent_reservation_policy"
    else:
        action = "retain_journal_and_require_manual_reconciliation"
    return JournalRecoveryState(
        path=journal_path,
        journal_sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        head_entry_sha256=str(entries[-1]["entry_sha256"]),
        entry_count=len(entries),
        run_id=str(entries[0]["run_id"]),
        finalized=finalized,
        generation_ids=tuple(sorted(generations)),
        unreconciled_generation_ids=tuple(unreconciled),
        uncertain_attempt_ids=tuple(uncertain),
        mcp_trace_count=mcp_count,
        recovery_action=action,
        safe_to_replay=safe_to_replay,
    )


def scan_recovery_journals(
    directory: str | Path,
    *,
    dataset_work_item_id: str | None = None,
) -> list[JournalRecoveryState]:
    """Find crash/final journals, optionally for one immutable dataset item."""

    root = Path(directory)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise JournalIntegrityError(f"journal root must be a directory: {root}")
    paths = sorted(
        [*root.glob(f".{FINAL_JOURNAL_PREFIX}*{IN_PROGRESS_SUFFIX}")]
        + [*root.glob(f"{FINAL_JOURNAL_PREFIX}*.jsonl")]
    )
    states: list[JournalRecoveryState] = []
    seen_runs: set[str] = set()
    for path in paths:
        entries = load_run_journal(path)
        if not path.name.endswith(IN_PROGRESS_SUFFIX):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name != f"{FINAL_JOURNAL_PREFIX}{digest}.jsonl":
                raise JournalIntegrityError(
                    f"finalized journal filename is not content-addressed: {path}"
                )
        metadata = entries[0].get("payload")
        if not isinstance(metadata, Mapping):
            raise JournalIntegrityError(f"run_started payload is malformed: {path}")
        if (
            dataset_work_item_id is not None
            and metadata.get("dataset_work_item_id") != dataset_work_item_id
        ):
            continue
        run_id = str(entries[0]["run_id"])
        if run_id in seen_runs:
            raise JournalIntegrityError(f"duplicate live journal run ID: {run_id}")
        seen_runs.add(run_id)
        states.append(recovery_state(path))
    return states


class RunJournal:
    """One synchronous fsynced hash-chain writer."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    @classmethod
    def create(
        cls,
        output_directory: str | Path,
        *,
        run_id: str,
        metadata: Mapping[str, Any],
    ) -> RunJournal:
        if _contains_forbidden_key(metadata):
            raise JournalIntegrityError("journal metadata contains a forbidden field")
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f".{FINAL_JOURNAL_PREFIX}{run_id}{IN_PROGRESS_SUFFIX}"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        journal = cls(path, run_id)
        journal.append("run_started", dict(metadata))
        return journal

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        if not event_type or event_type == "run_started" and self.path.stat().st_size:
            raise JournalIntegrityError("invalid or duplicate journal event type")
        if _contains_forbidden_key(payload):
            raise JournalIntegrityError("journal payload contains a forbidden field")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            content = handle.read()
            entries = _load_lines(self.path) if content else []
            if entries and entries[-1].get("event_type") == "run_finalized":
                raise JournalIntegrityError("cannot append after run_finalized")
            entry = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "sequence": len(entries) + 1,
                "recorded_at": recorded_at or _utc_now(),
                "previous_entry_sha256": (entries[-1]["entry_sha256"] if entries else None),
                "run_id": self.run_id,
                "event_type": event_type,
                "payload": dict(payload),
            }
            entry["entry_sha256"] = _event_digest(entry)
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical(entry).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return entry

    def finalize(self, payload: Mapping[str, Any]) -> JournalDescriptor:
        self.append("run_finalized", payload)
        entries = load_run_journal(self.path)
        rendered = self.path.read_bytes()
        digest = hashlib.sha256(rendered).hexdigest()
        destination = self.path.parent / f"{FINAL_JOURNAL_PREFIX}{digest}.jsonl"
        if destination.exists():
            if destination.read_bytes() != rendered:
                raise JournalIntegrityError(
                    f"refusing to overwrite conflicting finalized journal: {destination}"
                )
            self.path.unlink()
        else:
            os.replace(self.path, destination)
        destination.chmod(0o600)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        self.path = destination
        return JournalDescriptor(
            filename=destination.name,
            sha256=digest,
            head_entry_sha256=str(entries[-1]["entry_sha256"]),
            entry_count=len(entries),
            run_id=self.run_id,
            finalized=True,
        )


def verify_journal_descriptor(
    directory: str | Path, descriptor: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verify an artifact's journal link, exact bytes, hash chain, and final state."""

    filename = descriptor.get("filename")
    digest = descriptor.get("sha256")
    if (
        descriptor.get("schema_version") != JOURNAL_SCHEMA_VERSION
        or not isinstance(filename, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or filename != f"{FINAL_JOURNAL_PREFIX}{digest}.jsonl"
        or descriptor.get("finalized") is not True
    ):
        raise JournalIntegrityError("artifact has an invalid journal descriptor")
    path = Path(directory) / filename
    entries = load_run_journal(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise JournalIntegrityError("finalized journal byte hash does not match its descriptor")
    if (
        entries[-1].get("event_type") != "run_finalized"
        or descriptor.get("entry_count") != len(entries)
        or descriptor.get("head_entry_sha256") != entries[-1].get("entry_sha256")
        or descriptor.get("run_id") != entries[0].get("run_id")
    ):
        raise JournalIntegrityError("artifact journal descriptor does not match its journal")
    return entries


def write_fixture_journal(
    output_directory: str | Path,
    *,
    run_id: str,
    events: list[tuple[str, Mapping[str, Any]]],
) -> JournalDescriptor:
    """Test/helper writer that exercises the production fsync path."""

    journal = RunJournal.create(output_directory, run_id=run_id, metadata={"fixture": True})
    for event_type, payload in events:
        journal.append(event_type, payload)
    return journal.finalize({"status": "fixture_complete"})
