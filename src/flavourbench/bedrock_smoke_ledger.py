"""Locked, append-only spend evidence for the Bedrock contract-smoke lane.

The ledger is deliberately independent from the OpenRouter ledger and from
PostgreSQL.  It exists only for the two-arm B1 engineering smoke.  A file lock
serializes admission and every event is hash chained and fsynced before the
lock is released.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .bedrock_auth import BedrockLaneSettings
from .bedrock_budget import BedrockBudgetSnapshot, BedrockCostGovernor
from .bedrock_manifest import assert_public_catalog_safe

LEDGER_SCHEMA_VERSION = "flavourbench-bedrock-contract-smoke-ledger-v1"
TERMINAL_EVENTS = frozenset(
    {
        "reservation_released_pre_send",
        "reservation_released_service_rejection",
        "reservation_settled_rate_card_estimate",
        "reservation_held_uncertain",
    }
)
PROTECTED_ENTRY_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "recorded_at",
        "previous_entry_sha256",
        "entry_sha256",
        "event_type",
        "run_key",
        "arm_id",
        "reservation_id",
        "reservation_micros",
        "admission_status",
        "governed_exposure_before_usd",
        "governed_exposure_after_usd",
        "effective_stage_cap_usd",
        "hard_cap_usd",
    }
)


class BedrockSmokeLedgerError(RuntimeError):
    """The local smoke ledger is malformed or cannot safely admit work."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _event_digest(event: Mapping[str, Any]) -> str:
    unhashed = dict(event)
    unhashed.pop("entry_sha256", None)
    return sha256_json(unhashed)


def _read_locked(handle: Any) -> list[dict[str, Any]]:
    handle.seek(0)
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(handle.read().splitlines(), 1):
        if not line:
            raise BedrockSmokeLedgerError(f"blank Bedrock ledger line {line_number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise BedrockSmokeLedgerError(
                f"invalid Bedrock ledger JSON at line {line_number}"
            ) from error
        if not isinstance(entry, dict):
            raise BedrockSmokeLedgerError(f"Bedrock ledger line {line_number} is not an object")
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise BedrockSmokeLedgerError(
                f"unsupported Bedrock ledger schema at line {line_number}"
            )
        if entry.get("sequence") != line_number:
            raise BedrockSmokeLedgerError(f"Bedrock ledger sequence mismatch at line {line_number}")
        if entry.get("previous_entry_sha256") != previous:
            raise BedrockSmokeLedgerError(
                f"Bedrock ledger hash-chain mismatch at line {line_number}"
            )
        digest = entry.get("entry_sha256")
        if not isinstance(digest, str) or digest != _event_digest(entry):
            raise BedrockSmokeLedgerError(f"Bedrock ledger digest mismatch at line {line_number}")
        assert_public_catalog_safe(entry, path=f"$bedrock_ledger[{line_number - 1}]")
        entries.append(entry)
        previous = digest
    return entries


@dataclass(frozen=True)
class BedrockSmokeExposure:
    settled_rate_card_estimate_micros: int
    outstanding_reservations_micros: int
    uncertain_held_reservations_micros: int

    @property
    def governed_exposure_micros(self) -> int:
        return (
            self.settled_rate_card_estimate_micros
            + self.outstanding_reservations_micros
            + self.uncertain_held_reservations_micros
        )

    @property
    def governed_exposure_usd(self) -> Decimal:
        return Decimal(self.governed_exposure_micros) / Decimal(1_000_000)


def _reservation_events(entries: list[dict[str, Any]], reservation_id: str) -> list[dict[str, Any]]:
    return [entry for entry in entries if entry.get("reservation_id") == reservation_id]


def _payload_details(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    details = dict(payload or {})
    collisions = PROTECTED_ENTRY_FIELDS.intersection(details)
    if collisions:
        raise BedrockSmokeLedgerError(
            "Bedrock ledger payload overrides protected fields: " + ", ".join(sorted(collisions))
        )
    assert_public_catalog_safe(details, path="$bedrock_ledger_payload")
    return details


def _validate_event_identity(
    *,
    run_key: str,
    arm_id: str,
    reservation_id: str,
    reservation_micros: int,
) -> None:
    if not run_key or not arm_id or not reservation_id:
        raise BedrockSmokeLedgerError("Bedrock ledger identity fields are required")
    if (
        not isinstance(reservation_micros, int)
        or isinstance(reservation_micros, bool)
        or reservation_micros <= 0
    ):
        raise BedrockSmokeLedgerError("Bedrock reservation must be positive")


def exposure_from_entries(entries: list[dict[str, Any]]) -> BedrockSmokeExposure:
    reservations: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    for entry in entries:
        reservation_id = str(entry.get("reservation_id") or "")
        if not reservation_id:
            continue
        event_type = str(entry.get("event_type") or "")
        if event_type == "reservation_created":
            if reservation_id in reservations:
                raise BedrockSmokeLedgerError("duplicate Bedrock reservation ID")
            reservations[reservation_id] = entry
        elif event_type in TERMINAL_EVENTS:
            if reservation_id not in reservations:
                raise BedrockSmokeLedgerError("Bedrock terminal event has no reservation")
            if reservation_id in terminals:
                raise BedrockSmokeLedgerError("Bedrock reservation has multiple terminals")
            terminals[reservation_id] = entry

    settled = 0
    outstanding = 0
    held = 0
    for reservation_id, reservation in reservations.items():
        amount = reservation.get("reservation_micros")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise BedrockSmokeLedgerError("Bedrock reservation amount is invalid")
        terminal = terminals.get(reservation_id)
        if terminal is None:
            outstanding += amount
            continue
        event_type = terminal["event_type"]
        if event_type == "reservation_held_uncertain":
            held += amount
        elif event_type == "reservation_settled_rate_card_estimate":
            estimated = terminal.get("rate_card_estimated_cost_micros")
            if (
                not isinstance(estimated, int)
                or isinstance(estimated, bool)
                or estimated < 0
                or estimated > amount
            ):
                raise BedrockSmokeLedgerError(
                    "settled Bedrock estimate is invalid or exceeds its reservation"
                )
            settled += estimated
    return BedrockSmokeExposure(
        settled_rate_card_estimate_micros=settled,
        outstanding_reservations_micros=outstanding,
        uncertain_held_reservations_micros=held,
    )


class BedrockSmokeLedger:
    """Synchronous file-locked ledger suitable for SDK thread callbacks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise BedrockSmokeLedgerError("Bedrock ledger must be a regular non-symlink file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.path.chmod(0o600)

    def entries(self) -> list[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return _read_locked(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def exposure(self) -> BedrockSmokeExposure:
        return exposure_from_entries(self.entries())

    def append(
        self,
        event_type: str,
        *,
        run_key: str,
        arm_id: str,
        reservation_id: str,
        reservation_micros: int,
        payload: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        if not event_type:
            raise BedrockSmokeLedgerError("Bedrock ledger event type is required")
        _validate_event_identity(
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
        )
        details = _payload_details(payload)
        with self.path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                entries = _read_locked(handle)
                related = _reservation_events(entries, reservation_id)
                if event_type == "reservation_created" and related:
                    raise BedrockSmokeLedgerError("Bedrock reservation ID already exists")
                if event_type != "reservation_created" and not related:
                    raise BedrockSmokeLedgerError("Bedrock event has no reservation")
                if any(entry["event_type"] in TERMINAL_EVENTS for entry in related):
                    raise BedrockSmokeLedgerError("cannot append after a reservation terminal")
                entry: dict[str, Any] = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "sequence": len(entries) + 1,
                    "recorded_at": recorded_at or _utc_now(),
                    "previous_entry_sha256": (entries[-1]["entry_sha256"] if entries else None),
                    "event_type": event_type,
                    "run_key": run_key,
                    "arm_id": arm_id,
                    "reservation_id": reservation_id,
                    "reservation_micros": reservation_micros,
                    **details,
                }
                assert_public_catalog_safe(entry, path="$bedrock_ledger_entry")
                entry["entry_sha256"] = _event_digest(entry)
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(entry).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return entry
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def reserve(
        self,
        *,
        settings: BedrockLaneSettings,
        run_key: str,
        arm_id: str,
        reservation_id: str,
        reservation_micros: int,
        payload: Mapping[str, Any],
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        """Atomically evaluate the caps and append a reservation if admitted."""

        _validate_event_identity(
            run_key=run_key,
            arm_id=arm_id,
            reservation_id=reservation_id,
            reservation_micros=reservation_micros,
        )
        details = _payload_details(payload)
        with self.path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                entries = _read_locked(handle)
                if _reservation_events(entries, reservation_id):
                    raise BedrockSmokeLedgerError("Bedrock reservation ID already exists")
                exposure = exposure_from_entries(entries)
                decision = BedrockCostGovernor(settings).decide(
                    BedrockBudgetSnapshot(
                        actual_spend_usd=(
                            Decimal(exposure.settled_rate_card_estimate_micros) / Decimal(1_000_000)
                        ),
                        outstanding_reservations_usd=(
                            Decimal(
                                exposure.outstanding_reservations_micros
                                + exposure.uncertain_held_reservations_micros
                            )
                            / Decimal(1_000_000)
                        ),
                    ),
                    worst_case_reservation_usd=(Decimal(reservation_micros) / Decimal(1_000_000)),
                )
                if not decision.admitted:
                    raise BedrockSmokeLedgerError(
                        "Bedrock reservation was not admitted: "
                        f"{decision.status}: {decision.reason}"
                    )
                entry: dict[str, Any] = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "sequence": len(entries) + 1,
                    "recorded_at": recorded_at or _utc_now(),
                    "previous_entry_sha256": (entries[-1]["entry_sha256"] if entries else None),
                    "event_type": "reservation_created",
                    "run_key": run_key,
                    "arm_id": arm_id,
                    "reservation_id": reservation_id,
                    "reservation_micros": reservation_micros,
                    "admission_status": decision.status,
                    "governed_exposure_before_usd": str(decision.exposure_before_usd),
                    "governed_exposure_after_usd": str(decision.exposure_after_usd),
                    "effective_stage_cap_usd": str(decision.effective_stage_cap_usd),
                    "hard_cap_usd": str(decision.hard_cap_usd),
                    **details,
                }
                assert_public_catalog_safe(entry, path="$bedrock_reservation_entry")
                entry["entry_sha256"] = _event_digest(entry)
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(entry).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return entry
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def next_attempt_index(self, *, run_key: str, arm_id: str) -> int:
        return 1 + sum(
            entry.get("event_type") == "reservation_created"
            and entry.get("run_key") == run_key
            and entry.get("arm_id") == arm_id
            for entry in self.entries()
        )

    def arm_entries(self, *, run_key: str, arm_id: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self.entries()
            if entry.get("run_key") == run_key and entry.get("arm_id") == arm_id
        ]

    def descriptor(self) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                raw = handle.read()
                try:
                    rendered = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise BedrockSmokeLedgerError("Bedrock ledger is not valid UTF-8") from error
                entries = _read_locked(io.StringIO(rendered))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "filename": self.path.name,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "entry_count": len(entries),
            "head_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        }
