"""Bound executor for one successor coverage endpoint batch.

The offline successor preflight intentionally blocks this entry point.  A
later, independently content-addressed live-admission record must close every
preflight blocker and provide a complete reserve for the selected batch.  The
executor then admits at most one endpoint-isolated batch under the shared
frontier lock.  Its ledger is append-only and a started item is never replayed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .frontier_contract_runner import (
    AdmissionDenied,
    _exclusive_runner_lock,
    _verify_live_artifact,
    active_ledger_reservations,
    scan_live_smoke_artifacts,
    validate_ledger_artifact_links,
)
from .frontier_contract_runner import (
    append_ledger_event as append_frontier_ledger_event,
)
from .frontier_contract_runner import (
    load_ledger as load_frontier_ledger,
)
from .frontier_coverage_primary_successor_v1 import (
    ADMISSION_CEILING_USD,
    HARD_CAP_USD,
    HISTORICAL_UNPRICED_COHERE_WORK_IDS,
    PLAN_SCHEMA,
    PREFLIGHT_SCHEMA,
    CoverageSuccessorError,
    _addressed,
    _decimal_text,
    _file_sha256,
    _ordered_cohere_work_item_ids,
    _relative,
    _sha256,
    validate_plan,
    verify_preflight,
)

LEDGER_SCHEMA = "flavourbench-frontier-coverage-primary-successor-ledger-v1"
RECEIPT_SCHEMA = "flavourbench-frontier-coverage-primary-successor-receipt-v1"
LIVE_ADMISSION_SCHEMA = "flavourbench-frontier-coverage-primary-live-admission-v1"
EXECUTION_CONFIRMATION = "RUN_ONE_COVERAGE_SUCCESSOR_ENDPOINT_BATCH"
BLOCKER_EVIDENCE_SCHEMA = "flavourbench-frontier-coverage-primary-blocker-evidence-v1"
COHERE_ENVELOPE_SCHEMA = "flavourbench-cohere-direct-resource-envelope-v1"
COHERE_OPERATOR_ATTESTATION_SCHEMA = "flavourbench-cohere-scholars-operator-attestation-v1"
ALLOWED_EVENTS = {
    "endpoint_batch_reserved",
    "item_execution_started",
    "item_terminalized",
    "execution_incident",
    "endpoint_batch_terminalized",
}

_LEDGER_COMMON_KEYS = {
    "schema_version",
    "sequence",
    "previous_entry_sha256",
    "recorded_at",
    "entry_sha256",
    "event_type",
    "plan_sha256",
    "batch_id",
}
_LEDGER_EVENT_KEY_VARIANTS: dict[str, tuple[frozenset[str], ...]] = {
    "endpoint_batch_reserved": (
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "preflight_sha256",
                "live_admission_sha256",
                "work_item_ids",
                "reserved_usd",
                "cell_allowances_usd",
                "global_reservation_entry_sha256s",
                "locked_budget_rebase",
                "later_source_snapshot",
                "other_local_active_reservations",
                "other_canonical_global_active_reservations_usd",
                "reservation_unit",
                "replay_permitted",
            }
        ),
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "preflight_sha256",
                "live_admission_sha256",
                "work_item_ids",
                "reservation_kind",
                "usd_cost_or_reservation_claimed",
                "resource_envelope_sha256",
                "operator_attestation_sha256",
                "cell_resource_limits",
                "reservation_unit",
                "replay_permitted",
            }
        ),
    ),
    "item_execution_started": (
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "work_item_id",
                "run_id",
                "arm_id",
                "attempt_slots_sha256",
                "batch_reservation_entry_sha256",
                "replay_permitted",
            }
        ),
    ),
    "item_terminalized": tuple(
        frozenset(_LEDGER_COMMON_KEYS | fields)
        for fields in (
            {
                "work_item_id",
                "batch_reservation_entry_sha256",
                "disposition",
                "source_artifact_sha256",
                "source_filename",
                "tool_calls",
                "successful_tool_calls",
                "route_policy_epicure_hashes_verified",
                "request_started_count",
                "provider_reported_cost_usd",
                "cost_status",
                "actual_cost_usd",
                "usd_cost_or_reservation_claimed",
                "replay_permitted",
                "rank_eligible",
            },
            {
                "work_item_id",
                "batch_reservation_entry_sha256",
                "disposition",
                "source_artifact_sha256",
                "source_filename",
                "tool_calls",
                "successful_tool_calls",
                "route_policy_epicure_hashes_verified",
                "request_started_count",
                "provider_reported_cost_usd",
                "cost_status",
                "actual_cost_usd",
                "usd_cost_or_reservation_claimed",
                "resource_usage",
                "retained_resource_envelope",
                "replay_permitted",
                "rank_eligible",
            },
            {
                "work_item_id",
                "batch_reservation_entry_sha256",
                "disposition",
                "provider_reported_cost_usd",
                "cost_status",
                "actual_cost_usd",
                "usd_cost_or_reservation_claimed",
                "request_started_count",
                "journal_descriptors",
                "reliability_eligible",
                "replay_permitted",
                "rank_eligible",
            },
            {
                "work_item_id",
                "batch_reservation_entry_sha256",
                "disposition",
                "provider_reported_cost_usd",
                "cost_status",
                "actual_cost_usd",
                "usd_cost_or_reservation_claimed",
                "request_started_count",
                "journal_descriptors",
                "reliability_eligible",
                "resource_usage",
                "retained_resource_envelope",
                "replay_permitted",
                "rank_eligible",
            },
        )
    ),
    "execution_incident": (
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "work_item_id",
                "batch_reservation_entry_sha256",
                "incident",
                "request_started_count",
                "journal_descriptors",
                "reservation_retained",
                "replay_permitted",
            }
        ),
    ),
    "endpoint_batch_terminalized": (
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "batch_reservation_entry_sha256",
                "work_item_ids",
                "item_terminal_entry_sha256s",
                "actual_cost_usd",
                "conservative_exposure_usd",
                "canonical_global_reservations_retained",
                "canonical_global_retained_usd",
                "whole_batch_reservation_released",
                "replay_permitted",
            }
        ),
        frozenset(
            _LEDGER_COMMON_KEYS
            | {
                "batch_reservation_entry_sha256",
                "work_item_ids",
                "item_terminal_entry_sha256s",
                "provider_reported_cost_usd",
                "cost_status",
                "actual_cost_usd",
                "usd_cost_or_reservation_claimed",
                "canonical_usd_reservations_created",
                "resource_envelope_totals",
                "resource_usage_totals",
                "resource_quota_terminalized",
                "replay_permitted",
            }
        ),
    ),
}

_HISTORICAL_EVENT_VOCABULARIES = {
    "flavourbench-bedrock-contract-smoke-ledger-v1": {
        "reservation_created",
        "count_tokens_request_started",
        "count_tokens_response_received",
        "count_tokens_failed_pre_send",
        "converse_request_started",
        "converse_response_received",
        "converse_delivery_uncertain",
        "arm_artifact_recorded",
        "reservation_held_uncertain",
        "reservation_released_pre_send",
        "reservation_released_service_rejection",
        "reservation_settled_rate_card_estimate",
    },
    "flavourbench-matched-protocol-preflight-ledger-v1": {
        "reservation_created",
        "receipt_recorded",
        "execution_failed",
        "accounting_disposition_recorded",
    },
    "flavourbench-real-exploratory-ledger-v1": {
        "reservation_created",
        "source_artifact_recorded",
        "execution_incident",
        "source_incident_resolution_recorded",
    },
    "flavourbench-reasoning-effort-family-block-ledger-v2": {
        "family_block_reservation_created",
        "family_block_item_terminalized",
        "family_block_terminalized",
        "item_execution_started",
        "pre_generation_failure_terminalized",
    },
    LEDGER_SCHEMA: ALLOWED_EVENTS,
}


class CoverageExecutionError(RuntimeError):
    """Execution cannot continue without violating a frozen invariant."""


@dataclass(frozen=True)
class RecoveryEvidence:
    """Immutable evidence visible after an interrupted item invocation."""

    source_path: Path | None
    source_artifact_sha256: str | None
    request_started_count: int
    journal_descriptors: tuple[Mapping[str, Any], ...] = ()


class Runner(Protocol):
    def __call__(self, cell: Mapping[str, Any], source_root: Path, cap_usd: Decimal) -> None:
        """Execute one cell, returning only after a source is durably written."""


class Probe(Protocol):
    def __call__(self, cell: Mapping[str, Any], source_root: Path) -> RecoveryEvidence:
        """Inspect durable source and request-start evidence without provider I/O."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_rfc3339(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CoverageExecutionError(f"{field} is absent")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise CoverageExecutionError(f"{field} is not RFC3339") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CoverageExecutionError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise CoverageExecutionError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise CoverageExecutionError(f"{field} must be finite and non-negative")
    return parsed


def _ledger_digest(entry: Mapping[str, Any]) -> str:
    return _sha256({key: value for key, value in entry.items() if key != "entry_sha256"})


def _validate_ledger_event_shape(entry: Mapping[str, Any]) -> None:
    event = str(entry.get("event_type") or "")
    variants = _LEDGER_EVENT_KEY_VARIANTS.get(event, ())
    if not variants or frozenset(entry) not in variants:
        raise CoverageExecutionError(
            f"coverage ledger event {event!r} does not match one exact schema variant"
        )


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if (
                lowered
                in {
                    "contains_secret",
                    "credential_binding_is_derived_from_secret",
                }
                and child is False
            ):
                continue
            if (
                "api_key" in lowered
                or "secret_hash" in lowered
                or "secret_sha" in lowered
                or "credential_fingerprint" in lowered
                or lowered in {"credential_sha256", "key_fingerprint"}
            ):
                return True
            hash_or_bound = lowered.endswith("_sha256") or any(
                marker in lowered
                for marker in (
                    "max_tokens",
                    "token_bound",
                    "token_limit",
                    "token_count",
                    "tokens_across",
                )
            )
            if not hash_or_bound and (
                lowered
                in {
                    "api_key",
                    "authorization",
                    "credential",
                    "credentials",
                    "password",
                    "secret",
                    "token",
                }
                or lowered.endswith("_api_key")
                or lowered.endswith("_credential")
                or lowered.endswith("_secret")
            ):
                return True
            if _contains_secret_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(child) for child in value)
    elif isinstance(value, str):
        if value in {
            "cohere_scholars_operator_quota",
            "one_exact_cohere_resource_quota_batch",
        }:
            return False
        return bool(
            re.search(r"(?:sk-[A-Za-z0-9_-]{16,}|cohere_[A-Za-z0-9._-]{16,})", value)
            or re.search(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", value, re.IGNORECASE)
            or re.search(r"\bAKIA[0-9A-Z]{16}\b", value)
        )
    return False


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise CoverageExecutionError(f"ledger is not a regular file: {path}")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise CoverageExecutionError(f"blank ledger line {number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise CoverageExecutionError(f"invalid ledger line {number}") from error
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != LEDGER_SCHEMA
            or entry.get("sequence") != number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("event_type") not in ALLOWED_EVENTS
            or entry.get("entry_sha256") != _ledger_digest(entry)
            or _contains_secret_key(entry)
        ):
            raise CoverageExecutionError(f"ledger integrity failed at line {number}")
        _validate_ledger_event_shape(entry)
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    return entries


def _append_ledger(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    protected = {
        "schema_version",
        "sequence",
        "previous_entry_sha256",
        "recorded_at",
        "entry_sha256",
    }
    if protected.intersection(event) or _contains_secret_key(event):
        raise CoverageExecutionError("ledger event is protected or secret-bearing")
    entries = _load_ledger(path)
    entry = {
        "schema_version": LEDGER_SCHEMA,
        "sequence": len(entries) + 1,
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        "recorded_at": _utc_now(),
        **dict(event),
    }
    if entry.get("event_type") not in ALLOWED_EVENTS:
        raise CoverageExecutionError("unsupported coverage ledger event")
    entry["entry_sha256"] = _ledger_digest(entry)
    _validate_ledger_event_shape(entry)
    rendered = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        data = rendered.encode()
        if os.write(descriptor, data) != len(data):
            raise OSError("short coverage ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


@contextmanager
def _ledger_lock(path: Path) -> Iterable[None]:
    import fcntl

    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _plan_maps(
    plan: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cells = {str(cell["work_item_id"]): dict(cell) for cell in plan["cells"]}
    batches = {str(batch["batch_id"]): dict(batch) for batch in plan["endpoint_batches"]}
    if len(cells) != 50 or len(batches) != 16:
        raise CoverageExecutionError("plan does not contain the exact successor workload")
    return cells, batches


def _cohere_batch_resource_limits(
    plan: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    envelope = plan.get("cohere_prospective_resource_envelope") or {}
    limits = envelope.get("cell_limits") or {}
    if not isinstance(limits, Mapping):
        raise CoverageExecutionError("frozen Cohere resource limits are absent")
    try:
        selected = {str(work_id): limits[str(work_id)] for work_id in batch["work_item_ids"]}
    except KeyError as error:
        raise CoverageExecutionError(
            "Cohere batch is outside the frozen resource envelope"
        ) from error
    if any(not isinstance(value, Mapping) for value in selected.values()):
        raise CoverageExecutionError("frozen Cohere cell resource limit is malformed")
    return selected


def _cohere_resource_totals(limits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fresh_arms": len(limits),
        "provider_attempt_slots": sum(
            int(value["provider_attempt_slots"]) for value in limits.values()
        ),
        "semantic_successful_response_bound": sum(
            int(value["semantic_successful_response_bound"]) for value in limits.values()
        ),
        "mcp_session_slots": sum(int(value["mcp_session_slots"]) for value in limits.values()),
        "mcp_tool_call_slots": sum(int(value["mcp_tool_call_slots"]) for value in limits.values()),
        "max_actual_tool_calls": sum(
            int(value["max_actual_tool_calls"]) for value in limits.values()
        ),
        "max_output_tokens": sum(
            int(value["max_output_tokens_across_successful_responses"]) for value in limits.values()
        ),
        "max_reasoning_tokens": sum(
            int(value["max_reasoning_tokens_across_successful_responses"])
            for value in limits.values()
        ),
        "max_input_tokens": _decimal_text(
            sum(
                (
                    Decimal(str(value["max_input_tokens_across_successful_responses"]))
                    for value in limits.values()
                ),
                Decimal(0),
            )
        ),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_item_terminal_semantics(
    *,
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    reservation: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    disposition = str(entry.get("disposition") or "")
    source_dispositions = {"source_usable", "source_reliability_failure"}
    source_present = "source_artifact_sha256" in entry
    if entry.get("replay_permitted") is not False or entry.get("rank_eligible") is not False:
        raise CoverageExecutionError("item terminal permits replay or ranking")
    if disposition in source_dispositions:
        if (
            not source_present
            or not _is_sha256(entry.get("source_artifact_sha256"))
            or not isinstance(entry.get("source_filename"), str)
            or not entry.get("source_filename")
            or not _nonnegative_int(entry.get("tool_calls"))
            or not _nonnegative_int(entry.get("successful_tool_calls"))
            or int(entry["successful_tool_calls"]) > int(entry["tool_calls"])
            or entry.get("route_policy_epicure_hashes_verified") is not True
            or not _nonnegative_int(entry.get("request_started_count"))
            or int(entry["request_started_count"]) < 1
        ):
            raise CoverageExecutionError("source terminal identity or evidence is invalid")
    elif cell.get("execution_backend") == "cohere_direct":
        if disposition != "pre_generation_failure_no_provider_request" or source_present:
            raise CoverageExecutionError("Cohere item terminal has an unknown disposition")
        if (
            entry.get("request_started_count") != 0
            or entry.get("reliability_eligible") is not False
            or not isinstance(entry.get("journal_descriptors"), list)
        ):
            raise CoverageExecutionError("Cohere pre-generation terminal is malformed")
    else:
        if disposition != "pre_generation_failure_zero_cost" or source_present:
            raise CoverageExecutionError("priced item terminal has an unknown disposition")
        if (
            entry.get("request_started_count") != 0
            or entry.get("reliability_eligible") is not False
            or not isinstance(entry.get("journal_descriptors"), list)
        ):
            raise CoverageExecutionError("priced pre-generation terminal is malformed")

    if cell.get("execution_backend") != "cohere_direct":
        actual = _decimal(entry.get("actual_cost_usd"), field="priced item actual cost")
        reported = _decimal(
            entry.get("provider_reported_cost_usd"), field="priced provider-reported cost"
        )
        allowances = reservation.get("cell_allowances_usd")
        if (
            entry.get("cost_status") != "provider_accounted_usd"
            or entry.get("usd_cost_or_reservation_claimed") is not True
            or actual != reported
            or not isinstance(allowances, Mapping)
            or actual
            > _decimal(allowances.get(str(cell["work_item_id"])), field="priced item allowance")
        ):
            raise CoverageExecutionError("priced item cost semantics are invalid")
        return

    limit = plan["cohere_prospective_resource_envelope"]["cell_limits"][str(cell["work_item_id"])]
    usage = entry.get("resource_usage")
    usage_keys = {
        "accounting_status",
        "successful_responses",
        "provider_request_attempts",
        "tokens_prompt",
        "tokens_completion",
        "reasoning_tokens",
        "tool_calls",
        "generation_ids_sha256",
        "full_envelope_retained",
    }
    if (
        entry.get("actual_cost_usd") is not None
        or entry.get("provider_reported_cost_usd") != "0"
        or entry.get("cost_status") != "unpriced_unknown"
        or entry.get("usd_cost_or_reservation_claimed") is not False
        or entry.get("retained_resource_envelope") != limit
        or not isinstance(usage, Mapping)
        or set(usage) != usage_keys
        or usage.get("full_envelope_retained") is not True
        or any(
            not _nonnegative_int(usage.get(field))
            for field in (
                "successful_responses",
                "provider_request_attempts",
                "tokens_completion",
                "reasoning_tokens",
                "tool_calls",
            )
        )
    ):
        raise CoverageExecutionError("Cohere terminal resource or cost semantics are invalid")
    prompt_tokens = _decimal(usage.get("tokens_prompt"), field="Cohere prompt tokens")
    within = (
        int(usage["successful_responses"]) <= int(limit["semantic_successful_response_bound"])
        and int(usage["provider_request_attempts"]) <= int(limit["provider_attempt_slots"])
        and prompt_tokens <= Decimal(str(limit["max_input_tokens_across_successful_responses"]))
        and int(usage["tokens_completion"])
        <= int(limit["max_output_tokens_across_successful_responses"])
        and int(usage["reasoning_tokens"])
        <= int(limit["max_reasoning_tokens_across_successful_responses"])
        and int(usage["tool_calls"]) <= int(limit["max_actual_tool_calls"])
    )
    status = usage.get("accounting_status")
    if not within:
        raise CoverageExecutionError("Cohere terminal exceeds its frozen resource envelope")
    if status == "provider_usage_observed_within_frozen_resource_envelope":
        if (
            disposition not in source_dispositions
            or int(usage["successful_responses"]) < 1
            or int(usage["provider_request_attempts"]) < int(usage["successful_responses"])
            or usage.get("provider_request_attempts") != entry.get("request_started_count")
            or not _is_sha256(usage.get("generation_ids_sha256"))
        ):
            raise CoverageExecutionError("observed Cohere usage semantics are invalid")
    elif status == "usage_missing_invalid_or_over_bound_full_envelope_retained":
        expected = {
            "successful_responses": int(limit["semantic_successful_response_bound"]),
            "provider_request_attempts": int(limit["provider_attempt_slots"]),
            "tokens_prompt": Decimal(str(limit["max_input_tokens_across_successful_responses"])),
            "tokens_completion": int(limit["max_output_tokens_across_successful_responses"]),
            "reasoning_tokens": int(limit["max_reasoning_tokens_across_successful_responses"]),
            "tool_calls": int(limit["max_actual_tool_calls"]),
        }
        if (
            disposition != "source_reliability_failure"
            or usage.get("generation_ids_sha256") is not None
            or int(usage["successful_responses"]) != expected["successful_responses"]
            or int(usage["provider_request_attempts"]) != expected["provider_request_attempts"]
            or prompt_tokens != expected["tokens_prompt"]
            or int(usage["tokens_completion"]) != expected["tokens_completion"]
            or int(usage["reasoning_tokens"]) != expected["reasoning_tokens"]
            or int(usage["tool_calls"]) != expected["tool_calls"]
        ):
            raise CoverageExecutionError("retained Cohere envelope semantics are invalid")
    elif status == "no_provider_request_observed_full_envelope_retained":
        if (
            disposition != "pre_generation_failure_no_provider_request"
            or any(
                _decimal(usage.get(field), field=f"zero-call Cohere {field}") != 0
                for field in (
                    "successful_responses",
                    "provider_request_attempts",
                    "tokens_prompt",
                    "tokens_completion",
                    "reasoning_tokens",
                    "tool_calls",
                )
            )
            or usage.get("generation_ids_sha256") != _sha256([])
        ):
            raise CoverageExecutionError("zero-call Cohere usage semantics are invalid")
    else:
        raise CoverageExecutionError("unknown Cohere usage-accounting status")


def _validate_batch_terminal_semantics(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    reservation: Mapping[str, Any],
    terminals: Mapping[str, Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> None:
    work_ids = [str(value) for value in batch["work_item_ids"]]
    if entry.get("replay_permitted") is not False:
        raise CoverageExecutionError("batch terminal permits replay")
    if batch.get("execution_backend") == "cohere_direct":
        usages = [terminals[work_id]["resource_usage"] for work_id in work_ids]
        reasoning = [usage["reasoning_tokens"] for usage in usages]
        expected_usage = {
            "successful_responses": sum(int(value["successful_responses"]) for value in usages),
            "provider_request_attempts": sum(
                int(value["provider_request_attempts"]) for value in usages
            ),
            "tokens_prompt": _decimal_text(
                sum((Decimal(str(value["tokens_prompt"])) for value in usages), Decimal(0))
            ),
            "tokens_completion": sum(int(value["tokens_completion"]) for value in usages),
            "reasoning_tokens": sum(int(value) for value in reasoning),
            "tool_calls": sum(int(value["tool_calls"]) for value in usages),
            "full_envelope_retained": True,
        }
        if (
            entry.get("actual_cost_usd") is not None
            or entry.get("provider_reported_cost_usd") != "0"
            or entry.get("cost_status") != "unpriced_unknown"
            or entry.get("usd_cost_or_reservation_claimed") is not False
            or entry.get("canonical_usd_reservations_created") is not False
            or entry.get("resource_envelope_totals")
            != _cohere_resource_totals(_cohere_batch_resource_limits(plan, batch))
            or entry.get("resource_usage_totals") != expected_usage
            or entry.get("resource_quota_terminalized") is not True
        ):
            raise CoverageExecutionError("Cohere batch terminal totals are invalid")
        return

    actual = sum(
        (
            _decimal(terminals[work_id]["actual_cost_usd"], field="priced terminal cost")
            for work_id in work_ids
        ),
        Decimal(0),
    )
    batch_actual = _decimal(entry.get("actual_cost_usd"), field="priced batch actual")
    retained_ids = entry.get("canonical_global_reservations_retained")
    global_refs = reservation.get("global_reservation_entry_sha256s")
    allowances = reservation.get("cell_allowances_usd")
    if (
        not isinstance(retained_ids, list)
        or not isinstance(global_refs, Mapping)
        or not isinstance(allowances, Mapping)
    ):
        raise CoverageExecutionError("priced batch retention evidence is malformed")
    expected_retained = [
        global_refs[work_id]
        for work_id in work_ids
        if "source_artifact_sha256" not in terminals[work_id]
    ]
    retained = sum(
        (
            _decimal(allowances[work_id], field="retained priced allowance")
            for work_id in work_ids
            if global_refs[work_id] in expected_retained
        ),
        Decimal(0),
    )
    if (
        batch_actual != actual
        or actual > _decimal(reservation.get("reserved_usd"), field="priced batch reserve")
        or retained_ids != expected_retained
        or _decimal(entry.get("canonical_global_retained_usd"), field="retained USD") != retained
        or _decimal(entry.get("conservative_exposure_usd"), field="conservative exposure")
        != actual + retained
        or entry.get("whole_batch_reservation_released") is not (retained == 0)
    ):
        raise CoverageExecutionError("priced batch terminal totals are invalid")


def _validate_incident_semantics(entry: Mapping[str, Any]) -> None:
    descriptors = entry.get("journal_descriptors")
    if (
        entry.get("incident") != "request_started_without_source_uncertain_delivery_no_replay"
        or not isinstance(entry.get("request_started_count"), int)
        or isinstance(entry.get("request_started_count"), bool)
        or int(entry["request_started_count"]) <= 0
        or entry.get("reservation_retained") is not True
        or entry.get("replay_permitted") is not False
        or not isinstance(descriptors, list)
        or not descriptors
    ):
        raise CoverageExecutionError("execution incident semantics are invalid")
    expected_keys = {
        "filename",
        "sha256",
        "head_entry_sha256",
        "entry_count",
        "finalized",
        "uncertain_attempt_ids",
    }
    filenames: set[str] = set()
    for descriptor in descriptors:
        filename = descriptor.get("filename") if isinstance(descriptor, Mapping) else None
        uncertain = (
            descriptor.get("uncertain_attempt_ids") if isinstance(descriptor, Mapping) else None
        )
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != expected_keys
            or not isinstance(filename, str)
            or not filename.endswith((".jsonl", ".inprogress.jsonl"))
            or filename in filenames
            or not _is_sha256(descriptor.get("sha256"))
            or not _is_sha256(descriptor.get("head_entry_sha256"))
            or not isinstance(descriptor.get("entry_count"), int)
            or isinstance(descriptor.get("entry_count"), bool)
            or int(descriptor["entry_count"]) <= 0
            or not isinstance(descriptor.get("finalized"), bool)
            or not isinstance(uncertain, list)
            or any(not isinstance(value, str) or not value for value in uncertain)
            or len(set(uncertain)) != len(uncertain)
        ):
            raise CoverageExecutionError("execution incident journal descriptor is invalid")
        filenames.add(filename)


def _ledger_state(plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells, batches = _plan_maps(plan)
    order = [str(value) for value in plan.get("batch_execution_order") or []]
    if len(order) != len(batches) or set(order) != set(batches):
        raise CoverageExecutionError("frozen batch execution order is not an exact permutation")
    reservations: dict[str, Mapping[str, Any]] = {}
    starts: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    incidents: dict[str, Mapping[str, Any]] = {}
    completed: dict[str, Mapping[str, Any]] = {}
    next_batch_index = 0
    open_batch_id: str | None = None
    for entry in entries:
        _validate_ledger_event_shape(entry)
        if entry.get("plan_sha256") != plan["artifact_sha256"]:
            raise CoverageExecutionError("ledger event has a different plan binding")
        event = str(entry["event_type"])
        batch_id = str(entry.get("batch_id") or "")
        if batch_id not in batches:
            raise CoverageExecutionError("ledger event names an unknown batch")
        work_id = str(entry.get("work_item_id") or "")
        if work_id and work_id not in cells:
            raise CoverageExecutionError("ledger event names an unknown work item")
        if work_id and work_id not in batches[batch_id]["work_item_ids"]:
            raise CoverageExecutionError("ledger work item is not a member of its named batch")
        if event == "endpoint_batch_reserved":
            if (
                batch_id in reservations
                or open_batch_id is not None
                or next_batch_index >= len(order)
                or batch_id != order[next_batch_index]
            ):
                raise CoverageExecutionError(
                    "batch reservation violates the frozen sequential execution order"
                )
            open_batch_id = batch_id
            batch = batches[batch_id]
            if batch.get("execution_backend") == "cohere_direct":
                forbidden_usd = {
                    "reserved_usd",
                    "cell_allowances_usd",
                    "global_reservation_entry_sha256s",
                    "locked_budget_rebase",
                    "other_canonical_global_active_reservations_usd",
                }
                expected_limits = _cohere_batch_resource_limits(plan, batch)
                attestation_sha = str(entry.get("operator_attestation_sha256") or "")
                if (
                    entry.get("work_item_ids") != batch["work_item_ids"]
                    or entry.get("reservation_kind") != "cohere_scholars_operator_quota"
                    or entry.get("usd_cost_or_reservation_claimed") is not False
                    or forbidden_usd.intersection(entry)
                    or entry.get("resource_envelope_sha256")
                    != plan["cohere_prospective_resource_envelope"]["envelope_sha256"]
                    or len(attestation_sha) != 64
                    or any(character not in "0123456789abcdef" for character in attestation_sha)
                    or entry.get("cell_resource_limits") != expected_limits
                ):
                    raise CoverageExecutionError("Cohere quota reservation membership differs")
            else:
                allowances = entry.get("cell_allowances_usd")
                global_refs = entry.get("global_reservation_entry_sha256s")
                if (
                    entry.get("work_item_ids") != batch["work_item_ids"]
                    or not isinstance(allowances, Mapping)
                    or set(map(str, allowances)) != set(batch["work_item_ids"])
                    or not isinstance(global_refs, Mapping)
                    or set(map(str, global_refs)) != set(batch["work_item_ids"])
                    or any(
                        not isinstance(value, str) or len(value) != 64
                        for value in global_refs.values()
                    )
                    or sum(
                        (
                            _decimal(value, field="ledger cell allowance")
                            for value in allowances.values()
                        ),
                        Decimal(0),
                    )
                    != _decimal(entry.get("reserved_usd"), field="ledger batch reserve")
                ):
                    raise CoverageExecutionError("batch reservation membership differs")
            reservations[batch_id] = entry
        elif event == "item_execution_started":
            if open_batch_id != batch_id:
                raise CoverageExecutionError("item start is outside the active frozen batch")
            reservation = reservations.get(batch_id)
            cell = cells.get(work_id) or {}
            if (
                reservation is None
                or work_id in starts
                or entry.get("batch_reservation_entry_sha256") != reservation.get("entry_sha256")
                or entry.get("run_id") != cell.get("run_id")
                or entry.get("arm_id") != (cell.get("arm_ids") or {}).get("epicure_on")
                or entry.get("attempt_slots_sha256") != cell.get("attempt_slots_sha256")
            ):
                raise CoverageExecutionError("invalid or duplicate item start")
            starts[work_id] = entry
        elif event == "item_terminalized":
            if open_batch_id != batch_id:
                raise CoverageExecutionError("item terminal is outside the active frozen batch")
            reservation = reservations.get(batch_id)
            if (
                work_id not in starts
                or work_id in terminals
                or reservation is None
                or entry.get("batch_reservation_entry_sha256") != reservation.get("entry_sha256")
            ):
                raise CoverageExecutionError("item terminal lacks exactly one start")
            if cells[work_id].get("execution_backend") == "cohere_direct":
                if (
                    entry.get("actual_cost_usd") is not None
                    or entry.get("provider_reported_cost_usd") != "0"
                    or entry.get("cost_status") != "unpriced_unknown"
                    or entry.get("usd_cost_or_reservation_claimed") is not False
                    or not isinstance(entry.get("resource_usage"), Mapping)
                    or entry.get("retained_resource_envelope")
                    != plan["cohere_prospective_resource_envelope"]["cell_limits"][work_id]
                ):
                    raise CoverageExecutionError("Cohere terminal makes an invalid USD claim")
            _validate_item_terminal_semantics(
                plan=plan,
                cell=cells[work_id],
                reservation=reservation,
                entry=entry,
            )
            terminals[work_id] = entry
        elif event == "execution_incident":
            if open_batch_id != batch_id:
                raise CoverageExecutionError("incident is outside the active frozen batch")
            reservation = reservations.get(batch_id)
            if (
                work_id not in starts
                or work_id in terminals
                or work_id in incidents
                or reservation is None
                or entry.get("batch_reservation_entry_sha256") != reservation.get("entry_sha256")
            ):
                raise CoverageExecutionError("incident does not bind one unresolved start")
            _validate_incident_semantics(entry)
            incidents[work_id] = entry
        elif event == "endpoint_batch_terminalized":
            if open_batch_id != batch_id or batch_id not in reservations or batch_id in completed:
                raise CoverageExecutionError("invalid or duplicate batch terminal")
            expected = batches[batch_id]["work_item_ids"]
            reservation = reservations[batch_id]
            if (
                any(work_id not in terminals for work_id in expected)
                or entry.get("batch_reservation_entry_sha256") != reservation.get("entry_sha256")
                or entry.get("work_item_ids") != expected
                or entry.get("item_terminal_entry_sha256s")
                != [terminals[item]["entry_sha256"] for item in expected]
            ):
                raise CoverageExecutionError("batch terminal precedes item terminals")
            if batches[batch_id].get("execution_backend") == "cohere_direct" and (
                entry.get("actual_cost_usd") is not None
                or entry.get("provider_reported_cost_usd") != "0"
                or entry.get("cost_status") != "unpriced_unknown"
                or entry.get("usd_cost_or_reservation_claimed") is not False
                or entry.get("canonical_usd_reservations_created") is not False
                or entry.get("resource_envelope_totals")
                != _cohere_resource_totals(_cohere_batch_resource_limits(plan, batches[batch_id]))
            ):
                raise CoverageExecutionError("Cohere batch terminal makes an invalid USD claim")
            _validate_batch_terminal_semantics(
                plan=plan,
                batch=batches[batch_id],
                reservation=reservation,
                terminals=terminals,
                entry=entry,
            )
            completed[batch_id] = entry
            open_batch_id = None
            next_batch_index += 1
    active = [batch_id for batch_id in reservations if batch_id not in completed]
    if len(active) > 1:
        raise CoverageExecutionError("more than one coverage batch is active")
    return {
        "reservations": reservations,
        "starts": starts,
        "terminals": terminals,
        "incidents": incidents,
        "completed": completed,
        "active_batch_id": active[0] if active else None,
    }


def _load_hash_chained_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load any repository ledger only after verifying its common hash-chain contract."""

    if path.is_symlink() or not path.is_file():
        raise CoverageExecutionError(f"cross-program ledger is not a regular file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CoverageExecutionError(f"cannot read cross-program ledger: {path}") from error
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CoverageExecutionError(f"blank cross-program ledger line {path}:{number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise CoverageExecutionError(f"invalid cross-program ledger {path}:{number}") from error
        if (
            not isinstance(entry, dict)
            or entry.get("sequence") != number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("entry_sha256") != _ledger_digest(entry)
        ):
            raise CoverageExecutionError(
                f"cross-program ledger hash chain failed at {path}:{number}"
            )
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    return entries


def _generic_other_active_reservations(
    artifact_roots: Sequence[Path], *, excluded_ledgers: Sequence[Path]
) -> tuple[Decimal, list[dict[str, Any]]]:
    """Inventory active reserves with an explicit parser for every discovered schema."""

    excluded = {path.resolve() for path in excluded_ledgers}
    paths = {
        path.resolve(): path
        for root in artifact_roots
        if root.exists()
        for path in root.rglob("*.jsonl")
        if path.resolve() not in excluded
    }
    active: dict[str, tuple[Decimal, set[str]]] = {}
    unpriced_disclosures: list[dict[str, Any]] = []

    reasoning_reservations: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for _, candidate_path in sorted(paths.items(), key=lambda item: str(item[0])):
        candidate_entries = _load_hash_chained_jsonl(candidate_path)
        candidate_schemas = {str(entry.get("schema_version") or "") for entry in candidate_entries}
        if candidate_schemas != {"flavourbench-reasoning-effort-family-block-ledger-v2"}:
            continue
        for entry in candidate_entries:
            if entry.get("event_type") != "family_block_reservation_created":
                continue
            digest = str(entry.get("entry_sha256") or "")
            prior = reasoning_reservations.get(digest)
            if not digest or (prior is not None and prior[0] != entry):
                raise CoverageExecutionError(
                    "reasoning-family reservation digest is absent or conflicting"
                )
            reasoning_reservations[digest] = (entry, candidate_path)

    def add(digest: str, value: Decimal, path: Path) -> None:
        if not digest or value <= 0:
            raise CoverageExecutionError("cross-program reservation identity or amount is invalid")
        prior = active.get(digest)
        if prior is not None and prior[0] != value:
            raise CoverageExecutionError("copied reservation has conflicting active amounts")
        paths_for_reserve = prior[1] if prior is not None else set()
        paths_for_reserve.add(str(path))
        active[digest] = (value, paths_for_reserve)

    for resolved, path in sorted(paths.items(), key=lambda item: str(item[0])):
        del resolved
        entries = _load_hash_chained_jsonl(path)
        schemas = {str(entry.get("schema_version") or "") for entry in entries}
        if len(schemas) > 1:
            raise CoverageExecutionError(f"cross-program ledger mixes schemas: {path}")
        schema = next(iter(schemas), "")
        events = {str(entry.get("event_type") or "") for entry in entries}
        vocabulary = _HISTORICAL_EVENT_VOCABULARIES.get(schema)
        if vocabulary is not None and not events <= vocabulary:
            unknown = sorted(events - vocabulary)
            raise CoverageExecutionError(
                f"recognized ledger contains unknown event types {unknown}: {path}"
            )
        reservation_fields = {
            "reserved_usd",
            "reservation_micros",
            "reservation_id",
            "reservation_entry_sha256",
            "batch_reservation_entry_sha256",
            "block_reservation_entry_sha256",
            "reservation_kind",
            "cell_allowances_usd",
        }
        bedrock_events = set(
            _HISTORICAL_EVENT_VOCABULARIES["flavourbench-bedrock-contract-smoke-ledger-v1"]
        )
        field_events_by_schema: dict[str, dict[str, set[str]]] = {
            "flavourbench-bedrock-contract-smoke-ledger-v1": {
                "reservation_micros": bedrock_events,
                "reservation_id": bedrock_events,
            },
            "flavourbench-matched-protocol-preflight-ledger-v1": {
                "reserved_usd": {"reservation_created"},
            },
            "flavourbench-real-exploratory-ledger-v1": {
                "reserved_usd": {"reservation_created", "execution_incident"},
                "reservation_entry_sha256": {
                    "source_artifact_recorded",
                    "execution_incident",
                    "source_incident_resolution_recorded",
                },
            },
            "flavourbench-reasoning-effort-family-block-ledger-v2": {
                "reserved_usd": {"family_block_reservation_created"},
                "block_reservation_entry_sha256": {
                    "family_block_item_terminalized",
                    "family_block_terminalized",
                    "item_execution_started",
                    "pre_generation_failure_terminalized",
                },
            },
            LEDGER_SCHEMA: {
                "reserved_usd": {"endpoint_batch_reserved"},
                "batch_reservation_entry_sha256": {
                    "item_execution_started",
                    "item_terminalized",
                    "execution_incident",
                    "endpoint_batch_terminalized",
                },
                "reservation_kind": {"endpoint_batch_reserved"},
                "cell_allowances_usd": {"endpoint_batch_reserved"},
            },
        }
        field_events = field_events_by_schema.get(schema, {})
        for entry in entries:
            event = str(entry.get("event_type") or "")
            forbidden = {
                field
                for field in reservation_fields.intersection(entry)
                if event not in field_events.get(field, set())
            }
            if forbidden:
                raise CoverageExecutionError(
                    f"ledger event {event!r} carries forbidden reservation fields "
                    f"{sorted(forbidden)}: {path}"
                )
        if schema == "flavourbench-reasoning-effort-family-block-ledger-v2":
            for entry in entries:
                reference = entry.get("block_reservation_entry_sha256")
                if reference is not None and str(reference) not in reasoning_reservations:
                    raise CoverageExecutionError(
                        f"reasoning-family worker references an absent coordinator reserve: {path}"
                    )
            if "family_block_reservation_created" not in events:
                # Worker ledgers carry only references to coordinator reservations.  They
                # are verified above but do not independently reserve or double-count USD.
                continue
        has_reservations = (
            any("reservation" in event for event in events)
            or ("endpoint_batch_reserved" in events)
            or any(reservation_fields.intersection(entry) for entry in entries)
        )
        if not has_reservations:
            continue

        if schema == "flavourbench-bedrock-contract-smoke-ledger-v1":
            reservations: dict[str, Mapping[str, Any]] = {}
            terminals: dict[str, Mapping[str, Any]] = {}
            terminal_events = {
                "reservation_released_pre_send",
                "reservation_released_service_rejection",
                "reservation_settled_rate_card_estimate",
                "reservation_held_uncertain",
            }
            if {event for event in events if "reservation" in event} - (
                terminal_events | {"reservation_created"}
            ):
                raise CoverageExecutionError("unknown Bedrock reservation event")
            for entry in entries:
                reservation_id = str(entry.get("reservation_id") or "")
                event = str(entry.get("event_type") or "")
                if event == "reservation_created":
                    if not reservation_id or reservation_id in reservations:
                        raise CoverageExecutionError("invalid Bedrock reservation identity")
                    reservations[reservation_id] = entry
                elif event in terminal_events:
                    if reservation_id not in reservations or reservation_id in terminals:
                        raise CoverageExecutionError("invalid Bedrock reservation terminal")
                    terminals[reservation_id] = entry
            for reservation_id, reservation in reservations.items():
                terminal = terminals.get(reservation_id)
                if (
                    terminal is not None
                    and terminal.get("event_type") != "reservation_held_uncertain"
                ):
                    continue
                micros = reservation.get("reservation_micros")
                if not isinstance(micros, int) or isinstance(micros, bool) or micros <= 0:
                    raise CoverageExecutionError("invalid Bedrock reservation micros")
                add(reservation_id, Decimal(micros) / Decimal(1_000_000), path)
            continue

        if schema == "flavourbench-matched-protocol-preflight-ledger-v1":
            if {event for event in events if "reservation" in event} != {"reservation_created"}:
                raise CoverageExecutionError("unknown matched-preflight reservation event")
            reservations: dict[str, Mapping[str, Any]] = {}
            finalized: set[str] = set()
            for entry in entries:
                work_id = str(entry.get("work_item_id") or "")
                event = str(entry.get("event_type") or "")
                if event == "reservation_created":
                    if not work_id or work_id in reservations:
                        raise CoverageExecutionError("invalid matched-preflight reservation")
                    reservations[work_id] = entry
                elif event in {"receipt_recorded", "existing_receipt_adopted"}:
                    if work_id not in reservations or work_id in finalized:
                        raise CoverageExecutionError("invalid matched-preflight finalization")
                    finalized.add(work_id)
            for work_id, reservation in reservations.items():
                if work_id not in finalized:
                    add(
                        str(reservation["entry_sha256"]),
                        _decimal(reservation.get("reserved_usd"), field="preflight reserve"),
                        path,
                    )
            continue

        if schema == "flavourbench-real-exploratory-ledger-v1":
            if {event for event in events if "reservation" in event} != {"reservation_created"}:
                raise CoverageExecutionError("unknown real-run reservation event")
            reservations: dict[str, Mapping[str, Any]] = {}
            finalizations: set[str] = set()
            by_digest: dict[str, str] = {}
            for entry in entries:
                work_id = str(entry.get("work_item_id") or "")
                event = str(entry.get("event_type") or "")
                if event == "reservation_created":
                    if not work_id or work_id in reservations:
                        raise CoverageExecutionError("invalid real-run reservation")
                    reservations[work_id] = entry
                    by_digest[str(entry["entry_sha256"])] = work_id
                elif event == "source_artifact_recorded":
                    target = str(entry.get("reservation_entry_sha256") or "")
                    if by_digest.get(target) != work_id or work_id in finalizations:
                        raise CoverageExecutionError("invalid real-run source finalization")
                    finalizations.add(work_id)
            for work_id, reservation in reservations.items():
                if work_id not in finalizations:
                    value = _decimal(reservation.get("reserved_usd"), field="real-run reserve")
                    if value == 0:
                        if work_id not in HISTORICAL_UNPRICED_COHERE_WORK_IDS:
                            raise CoverageExecutionError(
                                f"unknown active zero-priced real-run reservation: {path}:{work_id}"
                            )
                        unpriced_disclosures.append(
                            {
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "reserved_usd": None,
                                "work_item_id": work_id,
                                "ledger": str(path),
                                "status": "closed_no_replay_unpriced_unknown_cost_disclosed",
                                "blocks_priced_batches": False,
                            }
                        )
                        continue
                    add(
                        str(reservation["entry_sha256"]),
                        value,
                        path,
                    )
            continue

        if schema == "flavourbench-reasoning-effort-family-block-ledger-v2":
            if {event for event in events if "reservation" in event} != {
                "family_block_reservation_created"
            }:
                raise CoverageExecutionError("unknown reasoning-family reservation event")
            reservations: dict[str, Mapping[str, Any]] = {}
            completed: set[str] = set()
            for entry in entries:
                block_id = str(entry.get("admission_block_id") or "")
                event = str(entry.get("event_type") or "")
                if event == "family_block_reservation_created":
                    if not block_id or block_id in reservations:
                        raise CoverageExecutionError("invalid reasoning-family reservation")
                    reservations[block_id] = entry
                elif event == "family_block_terminalized":
                    if block_id not in reservations or block_id in completed:
                        raise CoverageExecutionError("invalid reasoning-family terminal")
                    completed.add(block_id)
            for block_id, reservation in reservations.items():
                if block_id not in completed:
                    add(
                        str(reservation["entry_sha256"]),
                        _decimal(reservation.get("reserved_usd"), field="reasoning reserve"),
                        path,
                    )
            continue

        if schema == LEDGER_SCHEMA:
            if {event for event in events if "reservation" in event}:
                raise CoverageExecutionError("unknown successor reservation event")
            # Another successor ledger is parsed with the same strict state transition names.
            reservations: dict[str, Mapping[str, Any]] = {}
            completed: set[str] = set()
            for entry in entries:
                batch_id = str(entry.get("batch_id") or "")
                event = str(entry.get("event_type") or "")
                if event == "endpoint_batch_reserved":
                    if not batch_id or batch_id in reservations:
                        raise CoverageExecutionError("invalid successor batch reservation")
                    reservations[batch_id] = entry
                elif event == "endpoint_batch_terminalized":
                    if batch_id not in reservations or batch_id in completed:
                        raise CoverageExecutionError("invalid successor batch terminal")
                    completed.add(batch_id)
            for batch_id, reservation in reservations.items():
                if batch_id not in completed:
                    if reservation.get("reservation_kind") == "cohere_scholars_operator_quota":
                        if (
                            reservation.get("usd_cost_or_reservation_claimed") is not False
                            or "reserved_usd" in reservation
                            or "cell_allowances_usd" in reservation
                            or "global_reservation_entry_sha256s" in reservation
                        ):
                            raise CoverageExecutionError(
                                "invalid non-USD successor Cohere quota reservation"
                            )
                        unpriced_disclosures.append(
                            {
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "reserved_usd": None,
                                "batch_id": batch_id,
                                "ledger": str(path),
                                "status": "active_cohere_resource_quota_unpriced_unknown",
                                "blocks_priced_batches": False,
                            }
                        )
                        continue
                    add(
                        str(reservation["entry_sha256"]),
                        _decimal(reservation.get("reserved_usd"), field="successor reserve"),
                        path,
                    )
            continue

        raise CoverageExecutionError(
            f"unrecognized reservation-bearing ledger schema {schema!r}: {path}"
        )
    rows = [
        {
            "reservation_entry_sha256": digest,
            "reserved_usd": _decimal_text(value),
            "ledgers": sorted(paths_for_reserve),
        }
        for digest, (value, paths_for_reserve) in sorted(active.items())
    ] + sorted(
        unpriced_disclosures,
        key=lambda value: str(value.get("work_item_id") or value.get("batch_id") or ""),
    )
    return sum((value for value, _ in active.values()), Decimal(0)), rows


def _later_source_exposure(
    plan: Mapping[str, Any], artifact_roots: Sequence[Path]
) -> tuple[Decimal, dict[str, Any]]:
    snapshot = plan["budget"]["bound_predecessor_snapshot_derivation"]["live_source_snapshot"]
    cutoff = str((snapshot or {}).get("latest_completed_at") or "")
    if not cutoff:
        raise CoverageExecutionError("bound predecessor source cutoff is absent")
    cutoff_instant = _parse_rfc3339(cutoff, field="bound predecessor source cutoff")
    seen: set[str] = set()
    exposure = Decimal(0)
    records: list[dict[str, Any]] = []
    paths = {
        path.resolve(): path
        for root in artifact_roots
        if root.exists()
        for path in root.rglob("*.json")
    }
    for resolved, path in sorted(paths.items(), key=lambda item: str(item[0])):
        del resolved
        if path.is_symlink() or not path.is_file():
            raise CoverageExecutionError(f"later-source scan found a non-regular JSON: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CoverageExecutionError(f"later-source scan cannot parse JSON: {path}") from error
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != "flavourbench-live-smoke-v1"
        ):
            continue
        completed = str(raw.get("completed_at") or "")
        completed_instant = _parse_rfc3339(completed, field=f"live source completed_at {path}")
        if completed_instant <= cutoff_instant:
            continue
        artifact, digest = _verify_live_artifact(path)
        if digest in seen:
            continue
        seen.add(digest)
        if artifact.get("execution_backend") == "cohere_direct":
            budget = artifact.get("budget") or {}
            if (
                not isinstance(budget, Mapping)
                or budget.get("provider_charge_available") is not False
                or budget.get("actual_cost_micros") != 0
            ):
                raise CoverageExecutionError(
                    "later Cohere source does not preserve unpriced provider accounting"
                )
            records.append(
                {
                    "artifact_sha256": digest,
                    "completed_at": completed,
                    "exposure_usd": None,
                    "provider_reported_cost_usd": "0",
                    "cost_status": "unpriced_unknown",
                }
            )
            continue
        scan = scan_live_smoke_artifacts(path.parent)
        matches = [value for value in scan.artifacts if value.artifact_sha256 == digest]
        if len(matches) != 1:
            raise CoverageExecutionError("later live source exposure is ambiguous")
        value = matches[0].exposure_usd
        exposure += value
        records.append(
            {
                "artifact_sha256": digest,
                "completed_at": completed,
                "exposure_usd": _decimal_text(value),
            }
        )
        del artifact
    return exposure, {
        "cutoff_completed_at": cutoff,
        "source_count": len(records),
        "priced_source_count": sum(row["exposure_usd"] is not None for row in records),
        "unpriced_unknown_source_count": sum(row["exposure_usd"] is None for row in records),
        "records_sha256": _sha256(records),
    }


def rebase_budget(
    *,
    baseline_snapshot_usd: Decimal,
    later_source_exposure_usd: Decimal,
    other_active_reservations_usd: Decimal,
    own_active_reservation_usd: Decimal,
    next_batch_reserve_usd: Decimal,
) -> dict[str, Any]:
    current = (
        baseline_snapshot_usd
        + later_source_exposure_usd
        + other_active_reservations_usd
        + own_active_reservation_usd
    )
    projected = current + next_batch_reserve_usd
    return {
        "baseline_snapshot_usd": _decimal_text(baseline_snapshot_usd),
        "later_source_exposure_usd": _decimal_text(later_source_exposure_usd),
        "other_active_reservations_usd": _decimal_text(other_active_reservations_usd),
        "own_active_reservation_usd": _decimal_text(own_active_reservation_usd),
        "current_conservative_exposure_usd": _decimal_text(current),
        "next_batch_reserve_usd": _decimal_text(next_batch_reserve_usd),
        "projected_with_next_batch_usd": _decimal_text(projected),
        "admission_ceiling_usd": _decimal_text(ADMISSION_CEILING_USD),
        "hard_cap_usd": _decimal_text(HARD_CAP_USD),
        "admission_allowed": projected <= ADMISSION_CEILING_USD and projected <= HARD_CAP_USD,
    }


def _lexical_repo_path(*, repo_root: Path, relative: str, label: str) -> Path:
    anchor = Path(os.path.abspath(repo_root))
    candidate = Path(os.path.abspath(anchor / relative))
    try:
        candidate.relative_to(anchor)
    except ValueError as error:
        raise AdmissionDenied(f"{label} escapes the repository") from error
    current = anchor
    for component in candidate.relative_to(anchor).parts:
        current = current / component
        if current.is_symlink():
            raise AdmissionDenied(f"{label} contains a symlink component")
    return candidate


def _load_evidence_reference(
    *, repo_root: Path, reference: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if set(reference) != {"path", "bytes", "file_sha256", "semantic_sha256"} or (
        _contains_secret_key(reference)
    ):
        raise AdmissionDenied(f"{label} reference schema is not exact or is secret-bearing")
    relative = str(reference.get("path") or "")
    if not relative:
        raise AdmissionDenied(f"{label} has no evidence path")
    path = _lexical_repo_path(repo_root=repo_root, relative=relative, label=f"{label} evidence")
    expected = str(reference.get("semantic_sha256") or "")
    try:
        document = _addressed(path, expected_sha256=expected)
    except CoverageSuccessorError as error:
        raise AdmissionDenied(f"{label} evidence does not content-verify") from error
    if reference.get("bytes") != path.stat().st_size or reference.get(
        "file_sha256"
    ) != _file_sha256(path):
        raise AdmissionDenied(f"{label} evidence file identity differs")
    if _contains_secret_key(document):
        raise AdmissionDenied(f"{label} evidence is secret-bearing")
    return document


def _validate_cohere_operator_attestation(
    *, plan: Mapping[str, Any], operator: Mapping[str, Any]
) -> None:
    envelope = plan["cohere_prospective_resource_envelope"]
    work_item_ids = _ordered_cohere_work_item_ids(plan)
    public_binding = operator.get("credential_binding_public_object")
    person = operator.get("operator")
    expected_keys = {
        "schema_version",
        "status",
        "plan_sha256",
        "decision",
        "credential_program",
        "provider",
        "operator",
        "issued_at",
        "expires_at",
        "work_item_ids",
        "resource_envelope_sha256",
        "credential_binding_method",
        "credential_binding_public_object",
        "credential_binding_sha256",
        "credential_binding_is_derived_from_secret",
        "contains_secret",
        "usd_cost_or_reservation_claimed",
        "provider_or_epicure_calls_made_by_attestation",
        "artifact_sha256",
    }
    if (
        set(operator) != expected_keys
        or operator.get("schema_version") != COHERE_OPERATOR_ATTESTATION_SCHEMA
        or operator.get("status") != "operator_authorized_exact_resource_scope"
        or operator.get("plan_sha256") != plan["artifact_sha256"]
        or operator.get("decision") != "authorize_exact_bounded_cohere_scholars_use"
        or operator.get("credential_program") != "Cohere Scholars"
        or operator.get("provider") != "cohere_direct"
        or operator.get("work_item_ids") != work_item_ids
        or operator.get("resource_envelope_sha256") != envelope["envelope_sha256"]
        or operator.get("credential_binding_method") != "sha256_canonical_public_binding_object"
        or operator.get("credential_binding_is_derived_from_secret") is not False
        or operator.get("contains_secret") is not False
        or operator.get("usd_cost_or_reservation_claimed") is not False
        or operator.get("provider_or_epicure_calls_made_by_attestation") is not False
        or not isinstance(person, Mapping)
        or set(person) != {"full_name", "role"}
        or not all(isinstance(person.get(key), str) and person[key].strip() for key in person)
        or not isinstance(public_binding, Mapping)
        or set(public_binding)
        != {
            "provider",
            "credential_program",
            "environment_variable_name",
            "credential_handle",
            "scope",
        }
    ):
        raise AdmissionDenied("Cohere Scholars operator attestation does not verify")
    scope = public_binding.get("scope")
    handle = public_binding.get("credential_handle")
    if (
        public_binding.get("provider") != "cohere_direct"
        or public_binding.get("credential_program") != "Cohere Scholars"
        or public_binding.get("environment_variable_name") != "COHERE_API_KEY"
        or not isinstance(handle, str)
        or not 3 <= len(handle) <= 100
        or _contains_secret_key(handle)
        or not isinstance(scope, Mapping)
        or dict(scope)
        != {
            "plan_sha256": plan["artifact_sha256"],
            "resource_envelope_sha256": envelope["envelope_sha256"],
            "work_item_ids_sha256": _sha256(work_item_ids),
            "authorized_use": "frontier_coverage_successor_cohere_direct_only",
        }
        or operator.get("credential_binding_sha256") != _sha256(public_binding)
    ):
        raise AdmissionDenied(
            "Cohere credential binding must derive only from the exact public binding object"
        )
    issued = _parse_rfc3339(operator.get("issued_at"), field="Cohere attestation issued_at")
    expires = _parse_rfc3339(operator.get("expires_at"), field="Cohere attestation expires_at")
    now = datetime.now(UTC)
    if issued > now or expires <= now or expires <= issued:
        raise AdmissionDenied("Cohere operator attestation is not currently valid")
    if _contains_secret_key(operator):
        raise AdmissionDenied("Cohere operator attestation contains credential material")


def _validate_blocker_closures(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    admission: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    blockers = preflight.get("blockers")
    closures = admission.get("blocker_closures")
    if not isinstance(blockers, list) or not isinstance(closures, Mapping):
        raise AdmissionDenied("live admission has no exact preflight blocker closure set")
    codes = {str(blocker.get("code") or "") for blocker in blockers if isinstance(blocker, Mapping)}
    authorized_batch_id = str(admission.get("authorized_batch_id") or "")
    target_batches = [
        batch for batch in plan["endpoint_batches"] if batch.get("batch_id") == authorized_batch_id
    ]
    if len(target_batches) != 1:
        raise AdmissionDenied("live admission target batch is absent")
    target_is_cohere = target_batches[0].get("execution_backend") == "cohere_direct"
    if not target_is_cohere:
        codes.discard("cohere_complete_reservation_envelope_missing")
    if "" in codes or set(map(str, closures)) != codes:
        raise AdmissionDenied("live admission does not close exactly every preflight blocker")
    evidence: dict[str, dict[str, Any]] = {}
    for code in sorted(codes):
        closure = closures.get(code)
        if (
            not isinstance(closure, Mapping)
            or set(closure) != {"status", "evidence"}
            or closure.get("status") != "closed_for_one_batch_development_execution"
            or not isinstance(closure.get("evidence"), Mapping)
        ):
            raise AdmissionDenied(f"preflight blocker is not explicitly closed: {code}")
        document = _load_evidence_reference(
            repo_root=repo_root,
            reference=closure["evidence"],
            label=f"blocker {code}",
        )
        if (
            document.get("blocker_code") != code
            or document.get("plan_sha256") != plan["artifact_sha256"]
            or document.get("preflight_sha256") != preflight["artifact_sha256"]
            or document.get("decision") != "pass_for_one_batch_development_execution"
        ):
            raise AdmissionDenied(f"blocker evidence binding differs: {code}")
        if code == "cohere_complete_reservation_envelope_missing":
            if document.get("schema_version") != COHERE_ENVELOPE_SCHEMA or set(document) != {
                "schema_version",
                "blocker_code",
                "plan_sha256",
                "preflight_sha256",
                "decision",
                "resource_envelope",
                "operator_attestation",
                "artifact_sha256",
            }:
                raise AdmissionDenied("Cohere closure is not an exact resource envelope")
        else:
            specific_keys = {
                "live_route_availability_not_tested": {"route_records"},
                "reasoning_effort_sensitivity_precedes_coverage": {"ordering_released"},
                "cross_study_budget_contention_requires_locked_rebase": {
                    "locked_rebase_required_at_execution"
                },
                "epicure_lineage_not_independently_reconstructable": {
                    "epicure",
                    "execution_scope",
                    "official_release_blocker_remains",
                },
                "independent_governance_go_required": {"independent_governance_decision"},
                "successor_execution_root_not_empty": {"run_roots_reset"},
            }.get(code)
            common_keys = {
                "schema_version",
                "blocker_code",
                "plan_sha256",
                "preflight_sha256",
                "decision",
                "artifact_sha256",
            }
            if (
                document.get("schema_version") != BLOCKER_EVIDENCE_SCHEMA
                or specific_keys is None
                or set(document) != common_keys | specific_keys
            ):
                raise AdmissionDenied(f"blocker evidence schema differs: {code}")
        evidence[code] = document

    if target_is_cohere:
        envelope = evidence.get("cohere_complete_reservation_envelope_missing") or {}
        if envelope.get("resource_envelope") != plan[
            "cohere_prospective_resource_envelope"
        ] or not isinstance(envelope.get("operator_attestation"), Mapping):
            raise AdmissionDenied("Cohere resource envelope differs from the frozen derivation")
        operator = _load_evidence_reference(
            repo_root=repo_root,
            reference=envelope["operator_attestation"],
            label="Cohere Scholars operator attestation",
        )
        _validate_cohere_operator_attestation(plan=plan, operator=operator)
        evidence["_cohere_operator_attestation"] = operator

    route_evidence = evidence.get("live_route_availability_not_tested") or {}
    expected_routes = [
        {
            "model_id": cell["model_id"],
            "provider_tag": cell["provider_tag"],
            "candidate_manifest_sha256": cell["route_manifest_sha256"],
            "endpoint_execution_sha256": cell["endpoint_execution_sha256"],
            "canonical_model_slug": cell["route"]["canonical_model_slug"],
            "actual_provider": cell["route"]["expected_actual_provider"],
            "fallback_disabled": True,
            "tool_and_structured_contract_passed": True,
        }
        for cell in sorted(plan["cells"], key=lambda value: value["model_id"])
        if cell["ordinal"]
        == min(row["ordinal"] for row in plan["cells"] if row["model_id"] == cell["model_id"])
    ]
    if route_evidence.get("route_records") != expected_routes:
        raise AdmissionDenied("live route evidence does not cover the exact 16-route panel")
    if (
        (evidence.get("reasoning_effort_sensitivity_precedes_coverage") or {}).get(
            "ordering_released"
        )
        is not True
        or (evidence.get("cross_study_budget_contention_requires_locked_rebase") or {}).get(
            "locked_rebase_required_at_execution"
        )
        is not True
        or (evidence.get("independent_governance_go_required") or {}).get(
            "independent_governance_decision"
        )
        != "go"
    ):
        raise AdmissionDenied("ordering, budget, or independent-governance evidence is absent")
    lineage = evidence.get("epicure_lineage_not_independently_reconstructable") or {}
    if (
        lineage.get("epicure") != plan["epicure"]
        or lineage.get("execution_scope")
        != "development_only_nonofficial_non_rank_eligible_exception"
        or lineage.get("official_release_blocker_remains") is not True
    ):
        raise AdmissionDenied(
            "Epicure-lineage execution exception does not preserve the claim boundary"
        )
    return evidence


def _validate_admitted_reserve(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    admitted: Mapping[str, Any],
    blocker_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[Decimal, dict[str, Decimal], str]:
    cells = {str(cell["work_item_id"]): cell for cell in plan["cells"]}
    if batch.get("execution_backend") == "cohere_direct":
        expected_limits = {
            work_id: plan["cohere_prospective_resource_envelope"]["cell_limits"][work_id]
            for work_id in batch["work_item_ids"]
        }
        operator = blocker_evidence.get("_cohere_operator_attestation") or {}
        if (
            set(admitted)
            != {
                "work_item_ids",
                "reservation_kind",
                "usd_cost_or_reservation_claimed",
                "resource_envelope_sha256",
                "operator_attestation_sha256",
                "cell_resource_limits",
            }
            or admitted.get("work_item_ids") != batch["work_item_ids"]
            or admitted.get("reservation_kind") != "cohere_scholars_operator_quota"
            or admitted.get("usd_cost_or_reservation_claimed") is not False
            or "reserved_usd" in admitted
            or "cell_allowances_usd" in admitted
            or admitted.get("resource_envelope_sha256")
            != plan["cohere_prospective_resource_envelope"]["envelope_sha256"]
            or admitted.get("operator_attestation_sha256") != operator.get("artifact_sha256")
            or admitted.get("cell_resource_limits") != expected_limits
        ):
            raise AdmissionDenied("Cohere admission differs from the exact non-USD resource quota")
        return (
            Decimal(0),
            {str(work_id): Decimal(0) for work_id in batch["work_item_ids"]},
            "cohere_scholars_operator_quota",
        )

    raw_allowances = admitted.get("cell_allowances_usd")
    if (
        set(admitted) != {"work_item_ids", "reserved_usd", "cell_allowances_usd"}
        or admitted.get("work_item_ids") != batch["work_item_ids"]
        or not isinstance(raw_allowances, Mapping)
        or set(map(str, raw_allowances)) != set(batch["work_item_ids"])
    ):
        raise AdmissionDenied("admitted allowances are not the exact batch membership")
    expected: dict[str, Decimal] = {}
    for work_id in batch["work_item_ids"]:
        cell = cells[str(work_id)]
        frozen = cell["cost_reservation"].get("successor_reservation_usd")
        if frozen is None:
            raise AdmissionDenied("unpriced cell appeared in a priced endpoint batch")
        expected[str(work_id)] = _decimal(frozen, field="frozen cell reservation")
    actual = {
        str(work_id): _decimal(value, field="admitted cell allowance")
        for work_id, value in raw_allowances.items()
    }
    if actual != expected or any(value <= 0 for value in actual.values()):
        raise AdmissionDenied("admitted cell allowances differ from exact frozen/enveloped bounds")
    reserve = _decimal(admitted.get("reserved_usd"), field="admitted batch reserve")
    exact_sum = sum(actual.values(), Decimal(0))
    if reserve != exact_sum:
        raise AdmissionDenied("admitted batch reserve is not the exact allowance sum")
    if batch["complete_reservation_bound"]:
        frozen_batch = _decimal(
            batch["successor_priced_reserve_usd"], field="frozen priced batch reserve"
        )
        if reserve != frozen_batch:
            raise AdmissionDenied("priced batch reserve differs from its frozen exact sum")
    return reserve, actual, "priced_usd_reservation"


def _canonical_global_state(
    *, repo_root: Path, ledger_path: Path, source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Decimal]]:
    entries = load_frontier_ledger(ledger_path)
    scan = scan_live_smoke_artifacts(
        source_root,
        corrections_directory=repo_root / "flavourbench/artifacts/corrections",
    )
    validate_ledger_artifact_links(
        entries,
        scan,
        reconciliation_directory=repo_root
        / "flavourbench/artifacts/frontier-contract/reconciliations",
    )
    return entries, active_ledger_reservations(entries)


def _existing_global_batch_reservations(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    expected = set(batch["work_item_ids"])
    matches: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if (
            entry.get("event_type") != "reservation_created"
            or entry.get("coverage_successor_plan_sha256") != plan["artifact_sha256"]
            or entry.get("coverage_successor_batch_id") != batch["batch_id"]
        ):
            continue
        work_id = str(entry.get("coverage_successor_work_item_id") or "")
        if work_id not in expected or work_id in matches:
            raise CoverageExecutionError("global coverage reservation identity is duplicated")
        matches[work_id] = entry
    return matches


def _rebind_global_batch_reservations(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    local_reservation: Mapping[str, Any],
    local_state: Mapping[str, Any],
    global_entries: Sequence[Mapping[str, Any]],
    global_active: Mapping[str, Decimal],
) -> dict[str, Mapping[str, Any]]:
    cells, _ = _plan_maps(plan)
    by_digest = {str(entry.get("entry_sha256") or ""): entry for entry in global_entries}
    rebound: dict[str, Mapping[str, Any]] = {}
    for work_id, digest in local_reservation["global_reservation_entry_sha256s"].items():
        entry = by_digest.get(str(digest))
        cell = cells[str(work_id)]
        if (
            entry is None
            or entry.get("event_type") != "reservation_created"
            or entry.get("coverage_successor_work_item_id") != work_id
            or entry.get("coverage_successor_batch_id") != batch["batch_id"]
            or entry.get("coverage_successor_plan_sha256") != plan["artifact_sha256"]
            or entry.get("manifest_sha256") != cell["route_manifest_sha256"]
            or entry.get("model_id") != cell["model_id"]
            or entry.get("provider_tag") != cell["provider_tag"]
            or entry.get("endpoint_sha256") != cell["route"]["endpoint_document_sha256"]
            or _decimal(entry.get("reserved_usd"), field="canonical rebound reserve")
            != _decimal(
                local_reservation["cell_allowances_usd"][work_id],
                field="local rebound allowance",
            )
        ):
            raise CoverageExecutionError("local reserve lost its canonical global binding")
        reservation_sha = str(entry["entry_sha256"])
        if reservation_sha not in global_active:
            local_terminal = local_state["terminals"].get(str(work_id))
            finalizations = [
                row
                for row in global_entries
                if row.get("event_type") == "artifact_recorded"
                and row.get("reservation_entry_sha256") == reservation_sha
            ]
            if (
                local_terminal is None
                or len(finalizations) != 1
                or not local_terminal.get("source_artifact_sha256")
                or finalizations[0].get("artifact_sha256")
                != local_terminal.get("source_artifact_sha256")
            ):
                raise CoverageExecutionError(
                    "canonical reserve finalized before an exact local source terminal"
                )
        rebound[str(work_id)] = entry
    if set(rebound) != set(batch["work_item_ids"]):
        raise CoverageExecutionError("local canonical reservation set is incomplete")
    return rebound


def _validate_completed_priced_global_state(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    reservation: Mapping[str, Any],
    state: Mapping[str, Any],
    global_entries: Sequence[Mapping[str, Any]],
    global_active: Mapping[str, Decimal],
) -> None:
    rebound = _rebind_global_batch_reservations(
        plan=plan,
        batch=batch,
        local_reservation=reservation,
        local_state=state,
        global_entries=global_entries,
        global_active=global_active,
    )
    allowances = reservation["cell_allowances_usd"]
    for work_id in batch["work_item_ids"]:
        global_reservation = rebound[str(work_id)]
        digest = str(global_reservation["entry_sha256"])
        terminal = state["terminals"][str(work_id)]
        finalizations = [
            entry
            for entry in global_entries
            if entry.get("event_type") == "artifact_recorded"
            and entry.get("reservation_entry_sha256") == digest
        ]
        source_digest = terminal.get("source_artifact_sha256")
        if source_digest is not None:
            finalization = finalizations[0] if len(finalizations) == 1 else {}
            expected_issues = (
                []
                if terminal.get("disposition") == "source_usable"
                else ["source_reliability_failure"]
            )
            if (
                digest in global_active
                or len(finalizations) != 1
                or finalization.get("artifact_sha256") != source_digest
                or finalization.get("artifact_filename") != terminal.get("source_filename")
                or finalization.get("coverage_successor_plan_sha256") != plan["artifact_sha256"]
                or finalization.get("coverage_successor_batch_id") != batch["batch_id"]
                or finalization.get("coverage_successor_work_item_id") != work_id
                or _decimal(
                    finalization.get("artifact_exposure_usd"),
                    field="completed canonical artifact exposure",
                )
                != _decimal(
                    terminal.get("actual_cost_usd"),
                    field="completed local source cost",
                )
                or finalization.get("postflight_issues") != expected_issues
            ):
                raise CoverageExecutionError(
                    "completed priced source lacks one exact canonical finalization"
                )
        elif (
            len(finalizations) != 0
            or digest not in global_active
            or global_active[digest]
            != _decimal(allowances[work_id], field="completed retained allowance")
        ):
            raise CoverageExecutionError(
                "completed priced no-source item lost its exact active reservation"
            )


def _ensure_global_batch_reservations(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    allowances: Mapping[str, Decimal],
    repo_root: Path,
    ledger_path: Path,
    source_root: Path,
) -> dict[str, Mapping[str, Any]]:
    """Durably reserve every cell in the canonical shared ledger before any call."""

    cells = {str(cell["work_item_id"]): cell for cell in plan["cells"]}
    entries, _ = _canonical_global_state(
        repo_root=repo_root, ledger_path=ledger_path, source_root=source_root
    )
    existing = _existing_global_batch_reservations(plan=plan, batch=batch, entries=entries)
    for work_id in batch["work_item_ids"]:
        cell = cells[str(work_id)]
        value = allowances[str(work_id)]
        prior = existing.get(str(work_id))
        if prior is not None:
            if (
                _decimal(prior.get("reserved_usd"), field="global cell reserve") != value
                or prior.get("manifest_sha256") != cell["route_manifest_sha256"]
                or prior.get("model_id") != cell["model_id"]
                or prior.get("provider_tag") != cell["provider_tag"]
                or prior.get("endpoint_sha256") != cell["route"]["endpoint_document_sha256"]
            ):
                raise CoverageExecutionError("existing global coverage reserve differs")
            continue
        existing[str(work_id)] = append_frontier_ledger_event(
            ledger_path,
            {
                "event_type": "reservation_created",
                "runner_run_id": str(plan["artifact_sha256"]),
                "coverage_successor_plan_sha256": plan["artifact_sha256"],
                "coverage_successor_batch_id": batch["batch_id"],
                "coverage_successor_work_item_id": work_id,
                "manifest_sha256": cell["route_manifest_sha256"],
                "model_id": cell["model_id"],
                "canonical_model_slug": cell["route"]["canonical_model_slug"],
                "provider_tag": cell["provider_tag"],
                "endpoint_sha256": cell["route"]["endpoint_document_sha256"],
                "reserved_usd": _decimal_text(value),
                "reservation_unit": "one_successor_epicure_on_cell",
                "replay_permitted": False,
            },
        )
    if set(existing) != set(batch["work_item_ids"]):
        raise CoverageExecutionError("global batch reservation set is incomplete")
    return existing


def _record_global_source(
    *,
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    terminal: Mapping[str, Any],
    global_reservation: Mapping[str, Any],
    repo_root: Path,
    ledger_path: Path,
    source_root: Path,
) -> Mapping[str, Any] | None:
    digest = terminal.get("source_artifact_sha256")
    filename = terminal.get("source_filename")
    if not isinstance(digest, str) or not isinstance(filename, str):
        return None
    if cell.get("execution_backend") == "cohere_direct":
        raise CoverageExecutionError("Cohere sources must not enter the canonical USD ledger")
    entries, active = _canonical_global_state(
        repo_root=repo_root, ledger_path=ledger_path, source_root=source_root
    )
    reservation_sha = str(global_reservation["entry_sha256"])
    prior = [
        entry
        for entry in entries
        if entry.get("event_type") == "artifact_recorded"
        and entry.get("reservation_entry_sha256") == reservation_sha
    ]
    if len(prior) > 1:
        raise CoverageExecutionError("global source reservation was finalized twice")
    if prior:
        if prior[0].get("artifact_sha256") != digest:
            raise CoverageExecutionError("global source finalization digest differs")
        return prior[0]
    if reservation_sha not in active:
        raise CoverageExecutionError("global source reservation is neither active nor finalized")
    source_path = source_root / filename
    artifact, verified = _verify_live_artifact(source_path)
    if verified != digest:
        raise CoverageExecutionError("global source changed before reservation finalization")
    return append_frontier_ledger_event(
        ledger_path,
        {
            "event_type": "artifact_recorded",
            "runner_run_id": str(plan["artifact_sha256"]),
            "reservation_entry_sha256": reservation_sha,
            "manifest_sha256": cell["route_manifest_sha256"],
            "model_id": cell["model_id"],
            "provider_tag": cell["provider_tag"],
            "artifact_filename": filename,
            "artifact_sha256": digest,
            "artifact_status": artifact.get("status"),
            "artifact_exposure_usd": terminal["actual_cost_usd"],
            "postflight_issues": (
                []
                if terminal.get("disposition") == "source_usable"
                else ["source_reliability_failure"]
            ),
            "subprocess_returncode": None,
            "coverage_successor_plan_sha256": plan["artifact_sha256"],
            "coverage_successor_batch_id": global_reservation["coverage_successor_batch_id"],
            "coverage_successor_work_item_id": cell["work_item_id"],
        },
    )


def _terminal_outcomes(
    *, state: Mapping[str, Any], batch: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "work_item_id": work_id,
            "disposition": state["terminals"][work_id]["disposition"],
            "provider_reported_cost_usd": state["terminals"][work_id].get(
                "provider_reported_cost_usd"
            ),
            "cost_status": state["terminals"][work_id].get("cost_status"),
            "actual_cost_usd": state["terminals"][work_id].get("actual_cost_usd"),
            "usd_cost_or_reservation_claimed": state["terminals"][work_id].get(
                "usd_cost_or_reservation_claimed"
            ),
            "resource_usage": state["terminals"][work_id].get("resource_usage"),
            "retained_resource_envelope": state["terminals"][work_id].get(
                "retained_resource_envelope"
            ),
            "source_artifact_sha256": state["terminals"][work_id].get("source_artifact_sha256"),
        }
        for work_id in batch["work_item_ids"]
    ]


def _revalidate_completed_sources(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    reservation: Mapping[str, Any],
    state: Mapping[str, Any],
    repo_root: Path,
) -> None:
    cells, _ = _plan_maps(plan)
    source_root = _lexical_repo_path(
        repo_root=repo_root,
        relative=str(plan["execution_roots"]["canonical_global_source"]),
        label="completed source root",
    )
    allowances = reservation.get("cell_allowances_usd")
    for work_id in batch["work_item_ids"]:
        terminal = state["terminals"][work_id]
        if "source_artifact_sha256" not in terminal:
            continue
        cell = cells[str(work_id)]
        evidence = _default_probe(cell, source_root)
        cap = (
            Decimal(0)
            if cell.get("execution_backend") == "cohere_direct"
            else _decimal(
                allowances.get(work_id) if isinstance(allowances, Mapping) else None,
                field="completed source allowance",
            )
        )
        reconstructed = _audit_source(
            plan=plan,
            cell=cell,
            evidence=evidence,
            cap_usd=cap,
        )
        if any(terminal.get(key) != value for key, value in reconstructed.items()):
            raise CoverageExecutionError(
                "completed source no longer reconstructs its exact terminal payload"
            )


def _verified_receipts(
    receipt_dir: Path,
    *,
    plan: Mapping[str, Any],
    ledger_path: Path,
    repo_root: Path,
) -> dict[str, Path]:
    if not receipt_dir.exists():
        return {}
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise CoverageExecutionError("receipt root is not a regular directory")
    entries = _load_ledger(ledger_path)
    state = _ledger_state(plan, entries)
    _, batches = _plan_maps(plan)
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    found: dict[str, Path] = {}
    for path in sorted(receipt_dir.glob("*.json")):
        try:
            receipt = _addressed(path)
        except CoverageSuccessorError as error:
            raise CoverageExecutionError(f"receipt does not content-verify: {path}") from error
        batch_id = str(receipt.get("batch_id") or "")
        batch = batches.get(batch_id)
        completed = state["completed"].get(batch_id)
        reservation = state["reservations"].get(batch_id)
        if batch is not None and completed is not None and reservation is not None:
            _revalidate_completed_sources(
                plan=plan,
                batch=batch,
                reservation=reservation,
                state=state,
                repo_root=repo_root,
            )
        terminal_sequence = int((completed or {}).get("sequence") or 0)
        prefix_bytes = b"".join(lines[:terminal_sequence]) if terminal_sequence else b""
        outcomes = _terminal_outcomes(state=state, batch=batch) if batch is not None else []
        support = {
            "supported_cells": 407,
            "empty_cells": 73,
            "total_cells": 480,
            "source": plan["source_artifacts"]["corrected_arena"],
            "successor_source_usable_terminals_in_this_batch": sum(
                row["disposition"] == "source_usable" for row in outcomes
            ),
            "successor_terminals_change_observed_support_before_analysis_rebuild": False,
        }
        prefix = receipt.get("ledger_terminal_prefix")
        expected_filename = (
            f"frontier-coverage-primary-successor-receipt-{receipt.get('artifact_sha256')}.json"
        )
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("plan_sha256") != plan["artifact_sha256"]
            or set(receipt)
            != {
                "schema_version",
                "status",
                "plan_sha256",
                "preflight_sha256",
                "live_admission_sha256",
                "batch_id",
                "ledger_terminal_prefix",
                "outcomes",
                "observed_model_pair_family_support",
                "provider_or_epicure_calls_made_by_receipt",
                "claim_boundary",
                "artifact_sha256",
            }
            or receipt.get("status") != "one_endpoint_batch_terminal"
            or batch is None
            or completed is None
            or reservation is None
            or batch_id in found
            or receipt.get("preflight_sha256") != reservation.get("preflight_sha256")
            or receipt.get("live_admission_sha256") != reservation.get("live_admission_sha256")
            or receipt.get("outcomes") != outcomes
            or receipt.get("observed_model_pair_family_support") != support
            or receipt.get("provider_or_epicure_calls_made_by_receipt") is not False
            or receipt.get("claim_boundary") != plan["claim_boundary"]
            or path.name != expected_filename
            or not isinstance(prefix, Mapping)
            or dict(prefix)
            != {
                "path": _relative(repo_root, ledger_path),
                "entry_count": terminal_sequence,
                "prefix_file_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
                "head_entry_sha256": (completed or {}).get("entry_sha256"),
            }
        ):
            raise CoverageExecutionError(
                "receipt identity, terminal prefix, or outcomes are invalid"
            )
        found[batch_id] = path
    return found


def _materialize_receipt(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    batch: Mapping[str, Any],
    ledger_path: Path,
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    entries = _load_ledger(ledger_path)
    state = _ledger_state(plan, entries)
    batch_id = str(batch["batch_id"])
    completed = state["completed"].get(batch_id)
    reservation = state["reservations"].get(batch_id)
    if completed is None or reservation is None:
        raise CoverageExecutionError("receipt requires a terminalized batch")
    _revalidate_completed_sources(
        plan=plan,
        batch=batch,
        reservation=reservation,
        state=state,
        repo_root=repo_root,
    )
    terminal_sequence = int(completed["sequence"])
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    if terminal_sequence < 1 or terminal_sequence > len(lines):
        raise CoverageExecutionError("receipt ledger prefix boundary is invalid")
    prefix = b"".join(lines[:terminal_sequence])
    outcomes = _terminal_outcomes(state=state, batch=batch)
    source_usable = sum(row["disposition"] == "source_usable" for row in outcomes)
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "one_endpoint_batch_terminal",
        "plan_sha256": plan["artifact_sha256"],
        "preflight_sha256": preflight["artifact_sha256"],
        "live_admission_sha256": reservation["live_admission_sha256"],
        "batch_id": batch_id,
        "ledger_terminal_prefix": {
            "path": _relative(repo_root, ledger_path),
            "entry_count": terminal_sequence,
            "prefix_file_sha256": hashlib.sha256(prefix).hexdigest(),
            "head_entry_sha256": completed["entry_sha256"],
        },
        "outcomes": outcomes,
        "observed_model_pair_family_support": {
            "supported_cells": 407,
            "empty_cells": 73,
            "total_cells": 480,
            "source": plan["source_artifacts"]["corrected_arena"],
            "successor_source_usable_terminals_in_this_batch": source_usable,
            "successor_terminals_change_observed_support_before_analysis_rebuild": False,
        },
        "provider_or_epicure_calls_made_by_receipt": False,
        "claim_boundary": dict(plan["claim_boundary"]),
    }
    receipt = {**payload, "artifact_sha256": _sha256(payload)}
    receipt_dir = output_root / "run/receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        receipt_dir
        / f"frontier-coverage-primary-successor-receipt-{receipt['artifact_sha256']}.json"
    )
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
        raise CoverageExecutionError("content-addressed receipt conflict")
    if not destination.exists():
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=receipt_dir, delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(rendered)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, destination)
    return receipt, destination


def _default_probe(cell: Mapping[str, Any], source_root: Path) -> RecoveryEvidence:
    from .run_journal import load_run_journal, scan_recovery_journals

    work_id = str(cell["work_item_id"])
    source_matches: list[tuple[Path, str]] = []
    for path in sorted(source_root.glob("*.json")) if source_root.exists() else []:
        artifact, digest = _verify_live_artifact(path)
        if artifact.get("dataset_work_item_id") == work_id:
            source_matches.append((path, digest))
    if len(source_matches) > 1:
        raise CoverageExecutionError("more than one source exists for a successor work item")
    states = scan_recovery_journals(source_root, dataset_work_item_id=work_id)
    request_started = 0
    descriptors: list[dict[str, Any]] = []
    for state in states:
        entries = load_run_journal(state.path)
        request_started += sum(
            entry.get("event_type") == "provider_attempt"
            and isinstance(entry.get("payload"), Mapping)
            and entry["payload"].get("event_type") == "request_started"
            for entry in entries
        )
        descriptors.append(
            {
                "filename": state.path.name,
                "sha256": state.journal_sha256,
                "head_entry_sha256": state.head_entry_sha256,
                "entry_count": state.entry_count,
                "finalized": state.finalized,
                "uncertain_attempt_ids": list(state.uncertain_attempt_ids),
            }
        )
    source = source_matches[0] if source_matches else None
    return RecoveryEvidence(
        source_path=source[0] if source else None,
        source_artifact_sha256=source[1] if source else None,
        request_started_count=request_started,
        journal_descriptors=tuple(descriptors),
    )


def _phase_matches_frozen_slot(*, cell: Mapping[str, Any], observed: str, expected: str) -> bool:
    if observed == expected or (expected == "mcp_session" and observed == "mcp_attestation"):
        return True
    backend = str(cell.get("execution_backend") or "")
    return backend in {"kimi_direct", "cohere_direct"} and observed == f"{backend}_{expected}"


def _cohere_usage_accounting(
    *,
    artifact: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
    cell_limit: Mapping[str, Any],
    tool_calls: int,
) -> tuple[bool, dict[str, Any]]:
    result_rows = result.get("generation_metadata") if isinstance(result, Mapping) else []
    incomplete_rows = artifact.get("incomplete_generation_metadata")
    valid = isinstance(result_rows, list) and isinstance(incomplete_rows, list)
    rows = [
        *(result_rows if isinstance(result_rows, list) else []),
        *(incomplete_rows if isinstance(incomplete_rows, list) else []),
    ]
    generation_ids: set[str] = set()
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    if valid:
        for row in rows:
            generation_id = str(row.get("generation_id") or "") if isinstance(row, Mapping) else ""
            counters = (
                row.get("tokens_prompt") if isinstance(row, Mapping) else None,
                row.get("tokens_completion") if isinstance(row, Mapping) else None,
                row.get("reasoning_tokens") if isinstance(row, Mapping) else None,
            )
            if (
                not isinstance(row, Mapping)
                or not generation_id
                or generation_id in generation_ids
                or row.get("provider") != "cohere-direct"
                or row.get("model") != cell_limit["model_id"].removeprefix("cohere/")
                or row.get("cost_micros") != 0
                or row.get("reconciled") is not False
                or row.get("billing_reconciliation_status") != "provider_charge_unavailable"
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in counters
                )
            ):
                valid = False
                break
            generation_ids.add(generation_id)
            prompt_tokens += int(counters[0])
            completion_tokens += int(counters[1])
            reasoning_tokens += int(counters[2])
    response_ids = {
        str(event.get("generation_id") or "")
        for event in events
        if event.get("event_type") == "response_received"
    }
    request_attempts = sum(event.get("event_type") == "request_started" for event in events)
    if "" in response_ids or generation_ids != response_ids:
        valid = False
    within = bool(
        valid
        and len(rows) <= int(cell_limit["semantic_successful_response_bound"])
        and request_attempts <= int(cell_limit["provider_attempt_slots"])
        and prompt_tokens
        <= Decimal(str(cell_limit["max_input_tokens_across_successful_responses"]))
        and completion_tokens <= int(cell_limit["max_output_tokens_across_successful_responses"])
        and reasoning_tokens <= int(cell_limit["max_reasoning_tokens_across_successful_responses"])
        and tool_calls <= int(cell_limit["max_actual_tool_calls"])
    )
    if within:
        usage = {
            "accounting_status": "provider_usage_observed_within_frozen_resource_envelope",
            "successful_responses": len(rows),
            "provider_request_attempts": request_attempts,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_calls": tool_calls,
            "generation_ids_sha256": _sha256(sorted(generation_ids)),
            "full_envelope_retained": True,
        }
    else:
        usage = {
            "accounting_status": "usage_missing_invalid_or_over_bound_full_envelope_retained",
            "successful_responses": int(cell_limit["semantic_successful_response_bound"]),
            "provider_request_attempts": int(cell_limit["provider_attempt_slots"]),
            "tokens_prompt": cell_limit["max_input_tokens_across_successful_responses"],
            "tokens_completion": int(cell_limit["max_output_tokens_across_successful_responses"]),
            "reasoning_tokens": int(cell_limit["max_reasoning_tokens_across_successful_responses"]),
            "tool_calls": int(cell_limit["max_actual_tool_calls"]),
            "generation_ids_sha256": None,
            "full_envelope_retained": True,
        }
    return within, usage


def _audit_source(
    *,
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    evidence: RecoveryEvidence,
    cap_usd: Decimal,
) -> dict[str, Any]:
    if evidence.source_path is None or evidence.source_artifact_sha256 is None:
        raise CoverageExecutionError("source audit requires one durable source")
    artifact, digest = _verify_live_artifact(evidence.source_path)
    if digest != evidence.source_artifact_sha256:
        raise CoverageExecutionError("source digest changed during audit")
    journal = artifact.get("run_journal")
    if not isinstance(journal, Mapping):
        raise CoverageExecutionError("source lacks a required run-journal descriptor")
    matching_journals = [
        descriptor
        for descriptor in evidence.journal_descriptors
        if descriptor.get("filename") == journal.get("filename")
        and descriptor.get("sha256") == journal.get("sha256")
        and descriptor.get("head_entry_sha256") == journal.get("head_entry_sha256")
        and descriptor.get("entry_count") == journal.get("entry_count")
        and descriptor.get("finalized") == journal.get("finalized")
    ]
    if (
        journal.get("run_id") != cell["run_id"]
        or len(evidence.journal_descriptors) != 1
        or len(matching_journals) != 1
    ):
        raise CoverageExecutionError("source recovery evidence does not bind its exact journal")
    arm_id = str(cell["arm_ids"]["epicure_on"])
    allowed_attempts = {str(slot["attempt_id"]) for slot in cell["attempt_slots"]}
    slots_by_attempt = {
        str(slot["attempt_id"]): (str(slot["phase"]), int(slot["attempt_index"]))
        for slot in cell["attempt_slots"]
    }
    events = artifact.get("provider_attempt_events") or []
    if not isinstance(events, list) or any(not isinstance(event, Mapping) for event in events):
        raise CoverageExecutionError("source provider-attempt evidence is malformed")
    observed_attempts = {str(event.get("attempt_id") or "") for event in events}
    epicure = artifact.get("epicure") or {}
    if (
        artifact.get("run_id") != cell["run_id"]
        or artifact.get("dataset_work_item_id") != cell["work_item_id"]
        or artifact.get("dataset_task_id") != cell["task_id"]
        or artifact.get("prompt_sha256") != cell["prompt_sha256"]
        or artifact.get("category") != cell["task_family"]
        or artifact.get("requested_model_id") != cell["model_id"]
        or artifact.get("requested_provider") != cell["provider_tag"]
        or artifact.get("requested_conditions") != ["epicure_on"]
        or artifact.get("candidate_manifest_sha256") != cell["route_manifest_sha256"]
        or artifact.get("endpoint_execution_contract_sha256") != cell["endpoint_execution_sha256"]
        or artifact.get("execution_policy_sha256") != cell["execution_policy_sha256"]
        or artifact.get("epicure_tool_schema_sha256") != plan["epicure"]["tool_schema_sha256"]
        or not isinstance(epicure, Mapping)
        or epicure.get("release_id") != plan["epicure"]["release_id"]
        or epicure.get("bundle_sha256") != plan["epicure"]["bundle_sha256"]
        or epicure.get("application_sha256") != plan["epicure"]["application_sha256"]
        or not observed_attempts
        or "" in observed_attempts
        or not observed_attempts <= allowed_attempts
        or any(str(event.get("arm_id") or "") != arm_id for event in events)
    ):
        raise CoverageExecutionError(
            "source differs from its frozen identity, route, policy, Epicure, or attempt slots"
        )
    events_by_attempt: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, event in enumerate(events):
        attempt_id = str(event["attempt_id"])
        expected_phase, expected_index = slots_by_attempt[attempt_id]
        observed_phase = str(event.get("phase") or "")
        phase_matches = _phase_matches_frozen_slot(
            cell=cell,
            observed=observed_phase,
            expected=expected_phase,
        )
        if not phase_matches or event.get("attempt_index") != expected_index:
            raise CoverageExecutionError("source provider event differs from its frozen slot")
        events_by_attempt.setdefault(attempt_id, []).append((index, event))
    for attempt_id, attempt_events in events_by_attempt.items():
        expected_phase, _ = slots_by_attempt[attempt_id]
        event_types = [str(event.get("event_type") or "") for _, event in attempt_events]
        starts = [
            (index, event)
            for index, event in attempt_events
            if event.get("event_type") == "request_started"
        ]
        if expected_phase in {"planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"}:
            allowed = {
                "request_started",
                "response_received",
                "request_rejected",
                "pre_send_failure",
                "uncertain_delivery",
                "invalid_response",
                "retry_scheduled",
                "accounting_reconciled",
            }
            if set(event_types) - allowed or len(starts) != 1:
                raise CoverageExecutionError(
                    "provider attempt has an unknown event or lacks exactly one request_started"
                )
            terminal = [
                (index, event)
                for index, event in attempt_events
                if event.get("event_type")
                in {
                    "response_received",
                    "request_rejected",
                    "pre_send_failure",
                    "uncertain_delivery",
                    "invalid_response",
                }
            ]
            accounting = [
                (index, event)
                for index, event in attempt_events
                if event.get("event_type") == "accounting_reconciled"
            ]
            retries = [
                (index, event)
                for index, event in attempt_events
                if event.get("event_type") == "retry_scheduled"
            ]
            request_keys = {
                str(event.get("request_key_sha256") or "") for _, event in attempt_events
            }
            if (
                len(terminal) != 1
                or terminal[0][0] <= starts[0][0]
                or len(accounting) > 1
                or any(index <= terminal[0][0] for index, _ in accounting)
                or len(retries) > 1
                or any(index <= terminal[0][0] for index, _ in retries)
                or (accounting and terminal[0][1].get("event_type") != "response_received")
                or (
                    retries
                    and terminal[0][1].get("event_type")
                    not in {"request_rejected", "pre_send_failure"}
                )
                or "" in request_keys
                or len(request_keys) != 1
                or (
                    terminal[0][1].get("event_type") == "response_received"
                    and not str(terminal[0][1].get("generation_id") or "")
                )
            ):
                raise CoverageExecutionError(
                    f"provider attempt lifecycle is incomplete or ambiguous: {attempt_id}"
                )
        elif expected_phase == "mcp_session":
            if event_types != ["mcp_session_started", "mcp_session_attested"]:
                raise CoverageExecutionError("MCP session lifecycle is incomplete or ambiguous")
        elif expected_phase.startswith("mcp_tool_"):
            if (
                len(event_types) != 2
                or event_types[0] != "mcp_call_started"
                or event_types[1] not in {"mcp_call_completed", "mcp_call_failed"}
            ):
                raise CoverageExecutionError("MCP tool lifecycle is incomplete or ambiguous")
        else:
            raise CoverageExecutionError("source uses an unknown frozen attempt phase")
    observed_request_started = sum(event.get("event_type") == "request_started" for event in events)
    if observed_request_started != evidence.request_started_count:
        raise CoverageExecutionError(
            "source request_started count differs from its recovery-journal evidence"
        )
    budget = artifact.get("budget") or {}
    actual_micros = budget.get("actual_cost_micros")
    if not isinstance(actual_micros, int) or isinstance(actual_micros, bool) or actual_micros < 0:
        raise CoverageExecutionError("source actual cost is absent or malformed")
    results = artifact.get("results")
    result = results.get("epicure_on") if isinstance(results, Mapping) else None
    usable = isinstance(result, Mapping) and artifact.get("status") in {
        "complete",
        "complete_rate_card_estimated",
    }
    trace = result.get("tool_trace") if isinstance(result, Mapping) else None
    trace_valid = isinstance(trace, list) and 1 <= len(trace) <= int(
        plan["execution_policy"]["max_tool_calls_total"]
    )
    round_counts: dict[int, int] = {}
    successful_calls = 0
    if trace_valid:
        for call in trace:
            if not isinstance(call, Mapping):
                trace_valid = False
                break
            round_index = call.get("round_index")
            if (
                not isinstance(round_index, int)
                or isinstance(round_index, bool)
                or not 0 <= round_index < int(plan["execution_policy"]["max_tool_rounds"])
            ):
                trace_valid = False
                break
            round_counts[round_index] = round_counts.get(round_index, 0) + 1
            successful_calls += call.get("is_error") is False
        if any(
            count > int(plan["execution_policy"]["max_tool_calls_per_round"])
            for count in round_counts.values()
        ):
            trace_valid = False
    mcp_events = artifact.get("mcp_trace_events")
    if not isinstance(mcp_events, list) or any(
        not isinstance(event, Mapping) for event in mcp_events
    ):
        raise CoverageExecutionError("source MCP trace is malformed")
    if any(str(event.get("arm_id") or "") != arm_id for event in mcp_events):
        raise CoverageExecutionError("source MCP trace includes an unrelated arm")
    normalized_mcp = [
        {key: value for key, value in event.items() if key != "arm_id"} for event in mcp_events
    ]
    normalized_trace = [dict(call) for call in trace] if isinstance(trace, list) else []
    if normalized_mcp != normalized_trace:
        raise CoverageExecutionError("source result tool trace differs from the exact MCP trace")
    session_lifecycles = sum(event.get("event_type") == "mcp_session_attested" for event in events)
    tool_lifecycles = sum(
        event.get("event_type") in {"mcp_call_completed", "mcp_call_failed"} for event in events
    )
    if session_lifecycles != 1 or tool_lifecycles != len(normalized_trace):
        raise CoverageExecutionError(
            "source MCP lifecycle does not bind the exact result tool trace"
        )
    if trace_valid and any(
        hashlib.sha256(str(call.get("result") or "").encode()).hexdigest()
        != call.get("result_sha256")
        for call in trace
    ):
        trace_valid = False
    if usable and (
        result.get("actual_model_id") != cell["route"]["canonical_model_slug"]
        or result.get("actual_provider") != cell["route"]["expected_actual_provider"]
        or result.get("finish_reason") not in {"stop", "end_turn"}
        or not isinstance(result.get("answer_markdown"), str)
        or not str(result.get("answer_markdown")).strip()
        or result.get("final_response_mode") != "plain_text"
        or not trace_valid
        or successful_calls < 1
    ):
        usable = False
    terminal: dict[str, Any] = {
        "disposition": "source_usable" if usable else "source_reliability_failure",
        "source_artifact_sha256": digest,
        "source_filename": evidence.source_path.name,
        "tool_calls": len(trace) if isinstance(trace, list) else 0,
        "successful_tool_calls": successful_calls,
        "route_policy_epicure_hashes_verified": True,
        "request_started_count": evidence.request_started_count,
        "replay_permitted": False,
        "rank_eligible": False,
    }
    if cell.get("execution_backend") == "cohere_direct":
        budget = artifact.get("budget") or {}
        if (
            actual_micros != 0
            or not isinstance(budget, Mapping)
            or budget.get("provider_charge_available") is not False
        ):
            raise CoverageExecutionError("Cohere source makes a priced cost claim")
        cell_limit = plan["cohere_prospective_resource_envelope"]["cell_limits"][
            cell["work_item_id"]
        ]
        usage_valid, resource_usage = _cohere_usage_accounting(
            artifact=artifact,
            result=result if isinstance(result, Mapping) else None,
            events=events,
            cell_limit=cell_limit,
            tool_calls=len(trace) if isinstance(trace, list) else 0,
        )
        if not usage_valid:
            terminal["disposition"] = "source_reliability_failure"
        terminal.update(
            {
                "provider_reported_cost_usd": "0",
                "cost_status": "unpriced_unknown",
                "actual_cost_usd": None,
                "usd_cost_or_reservation_claimed": False,
                "resource_usage": resource_usage,
                "retained_resource_envelope": dict(cell_limit),
            }
        )
        return terminal
    actual = Decimal(actual_micros) / Decimal(1_000_000)
    if actual > cap_usd:
        raise CoverageExecutionError("source actual cost exceeds its admitted cell allowance")
    terminal.update(
        {
            "provider_reported_cost_usd": _decimal_text(actual),
            "cost_status": "provider_accounted_usd",
            "actual_cost_usd": _decimal_text(actual),
            "usd_cost_or_reservation_claimed": True,
        }
    )
    return terminal


@contextmanager
def _policy_environment(plan: Mapping[str, Any], cell: Mapping[str, Any]) -> Iterable[None]:
    from unittest.mock import patch

    from .frontier_coverage_repair_executor import _policy_from_document

    policy = _policy_from_document(plan["execution_policy"]["execution_policy"])
    pricing = cell["route"]["pricing"]
    additions = dict(policy.settings_environment())
    if pricing.get("prompt") is not None and pricing.get("completion") is not None:
        additions.update(
            {
                "FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK": _decimal_text(
                    Decimal(pricing["prompt"]) * Decimal(1_000_000)
                ),
                "FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK": _decimal_text(
                    Decimal(pricing["completion"]) * Decimal(1_000_000)
                ),
            }
        )
    elif cell.get("execution_backend") != "cohere_direct":
        raise CoverageExecutionError("priced route lacks its frozen USD pricing fields")
    with patch.dict(os.environ, additions, clear=False):
        yield


def _live_namespace(
    plan: Mapping[str, Any], cell: Mapping[str, Any], source_root: Path, cap_usd: Decimal
) -> argparse.Namespace:
    policy = plan["execution_policy"]["execution_policy"]
    attempt_slots = [dict(slot) for slot in cell["attempt_slots"]]
    backend = str(cell.get("execution_backend") or "")
    if backend in {"kimi_direct", "cohere_direct"}:
        provider_phases = {"planning", "tool_round_0", "tool_round_1", "tool_round_2", "final"}
        for slot in attempt_slots:
            if slot["phase"] in provider_phases:
                slot["phase"] = f"{backend}_{slot['phase']}"
    return argparse.Namespace(
        confirm="RUN_REAL_FRONTIER_SMOKE",
        cap_usd=cap_usd,
        route_manifest=Path(cell["route"]["manifest"]["path"]),
        candidate_manifest_sha256=cell["route_manifest_sha256"],
        model_id=cell["model_id"],
        provider_slug=cell["provider_tag"],
        prompt=cell["task"]["prompt"],
        category=cell["task_family"],
        output_dir=str(source_root),
        dataset_work_item_id=cell["work_item_id"],
        dataset_task_id=cell["task_id"],
        expected_canonical_model_slug=cell["route"]["canonical_model_slug"],
        expected_endpoint_execution_sha256=cell["endpoint_execution_sha256"],
        expected_execution_policy_sha256=cell["execution_policy_sha256"],
        condition=["epicure_on"],
        expected_epicure_release_id=plan["epicure"]["release_id"],
        expected_epicure_bundle_sha256=plan["epicure"]["bundle_sha256"],
        expected_epicure_application_sha256=plan["epicure"]["application_sha256"],
        expected_epicure_tool_schema_sha256=plan["epicure"]["tool_schema_sha256"],
        plain_text_final=True,
        tool_catalog_bytes_bound=policy["cost_forecast"]["tool_catalog_bytes_bound"],
        require_epicure_call=True,
        evidence_protocol=policy["evidence_protocol"],
        intermediate_reasoning_effort=policy["reasoning"]["intermediate_effort"],
        final_reasoning_effort=policy["reasoning"]["final_effort"],
        sequential_arms=False,
        frozen_run_id=cell["run_id"],
        frozen_attempt_slots=attempt_slots,
        preflight_only=False,
        skip_tool_contract=True,
        contract_only=False,
    )


def _run_live_cell(
    cell: Mapping[str, Any],
    source_root: Path,
    cap_usd: Decimal,
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
) -> None:
    from .direct_cohere_pair import run_pair as run_cohere
    from .direct_kimi_pair import run_pair as run_kimi
    from .live_smoke import live_smoke

    args = _live_namespace(plan, cell, source_root, cap_usd)
    args.route_manifest = repo_root / str(cell["route"]["manifest"]["path"])
    with _policy_environment(plan, cell):
        if cell["execution_backend"] == "openrouter":
            asyncio.run(live_smoke(args))
        elif cell["execution_backend"] == "kimi_direct":
            asyncio.run(run_kimi(args))
        elif cell["execution_backend"] == "cohere_direct":
            asyncio.run(run_cohere(args))
        else:
            raise CoverageExecutionError("unsupported frozen execution backend")


def _append_terminal(
    *,
    ledger_path: Path,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    cell: Mapping[str, Any],
    reservation: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _append_ledger(
        ledger_path,
        {
            "event_type": "item_terminalized",
            "plan_sha256": plan["artifact_sha256"],
            "batch_id": batch["batch_id"],
            "work_item_id": cell["work_item_id"],
            "batch_reservation_entry_sha256": reservation["entry_sha256"],
            **dict(payload),
            "replay_permitted": False,
            "rank_eligible": False,
        },
    )


def _recover_started(
    *,
    ledger_path: Path,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    cell: Mapping[str, Any],
    reservation: Mapping[str, Any],
    endpoint_root: Path,
    cap_usd: Decimal,
    probe: Probe,
) -> str:
    state = _ledger_state(plan, _load_ledger(ledger_path))
    existing_terminal = state["terminals"].get(cell["work_item_id"])
    if existing_terminal is not None:
        return str(existing_terminal["disposition"])
    if cell["work_item_id"] in state["incidents"]:
        return "uncertain_delivery_no_replay"
    evidence = probe(cell, endpoint_root)
    if evidence.source_path is not None:
        payload = _audit_source(plan=plan, cell=cell, evidence=evidence, cap_usd=cap_usd)
        _append_terminal(
            ledger_path=ledger_path,
            plan=plan,
            batch=batch,
            cell=cell,
            reservation=reservation,
            payload=payload,
        )
        return str(payload["disposition"])
    if evidence.request_started_count:
        _append_ledger(
            ledger_path,
            {
                "event_type": "execution_incident",
                "plan_sha256": plan["artifact_sha256"],
                "batch_id": batch["batch_id"],
                "work_item_id": cell["work_item_id"],
                "batch_reservation_entry_sha256": reservation["entry_sha256"],
                "incident": "request_started_without_source_uncertain_delivery_no_replay",
                "request_started_count": evidence.request_started_count,
                "journal_descriptors": list(evidence.journal_descriptors),
                "reservation_retained": True,
                "replay_permitted": False,
            },
        )
        return "uncertain_delivery_no_replay"
    if cell.get("execution_backend") == "cohere_direct":
        cell_limit = plan["cohere_prospective_resource_envelope"]["cell_limits"][
            cell["work_item_id"]
        ]
        zero_call_payload: dict[str, Any] = {
            "disposition": "pre_generation_failure_no_provider_request",
            "provider_reported_cost_usd": "0",
            "cost_status": "unpriced_unknown",
            "actual_cost_usd": None,
            "usd_cost_or_reservation_claimed": False,
            "request_started_count": 0,
            "journal_descriptors": list(evidence.journal_descriptors),
            "reliability_eligible": False,
            "resource_usage": {
                "accounting_status": "no_provider_request_observed_full_envelope_retained",
                "successful_responses": 0,
                "provider_request_attempts": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "reasoning_tokens": 0,
                "tool_calls": 0,
                "generation_ids_sha256": _sha256([]),
                "full_envelope_retained": True,
            },
            "retained_resource_envelope": dict(cell_limit),
        }
    else:
        zero_call_payload = {
            "disposition": "pre_generation_failure_zero_cost",
            "provider_reported_cost_usd": "0",
            "cost_status": "provider_accounted_usd",
            "actual_cost_usd": "0",
            "usd_cost_or_reservation_claimed": True,
            "request_started_count": 0,
            "journal_descriptors": list(evidence.journal_descriptors),
            "reliability_eligible": False,
        }
    _append_terminal(
        ledger_path=ledger_path,
        plan=plan,
        batch=batch,
        cell=cell,
        reservation=reservation,
        payload=zero_call_payload,
    )
    return str(zero_call_payload["disposition"])


def _validate_completed_admission_binding(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    admission: Mapping[str, Any],
    batch: Mapping[str, Any],
    reservation: Mapping[str, Any],
    blocker_evidence: Mapping[str, Mapping[str, Any]],
) -> None:
    batch_id = str(batch["batch_id"])
    documents = admission.get("batch_reservations")
    admitted = documents.get(batch_id) if isinstance(documents, Mapping) else None
    if (
        reservation.get("preflight_sha256") != preflight.get("artifact_sha256")
        or reservation.get("live_admission_sha256") != admission.get("artifact_sha256")
        or admission.get("authorized_batch_id") != batch_id
        or not isinstance(documents, Mapping)
        or set(map(str, documents)) != {batch_id}
        or not isinstance(admitted, Mapping)
    ):
        raise CoverageExecutionError(
            "completed batch is not bound to the supplied preflight and live admission"
        )
    reserve, allowances, mode = _validate_admitted_reserve(
        plan=plan,
        batch=batch,
        admitted=admitted,
        blocker_evidence=blocker_evidence,
    )
    if mode == "cohere_scholars_operator_quota":
        operator = blocker_evidence.get("_cohere_operator_attestation")
        if (
            reservation.get("reservation_kind") != mode
            or reservation.get("resource_envelope_sha256")
            != admitted.get("resource_envelope_sha256")
            or reservation.get("operator_attestation_sha256")
            != admitted.get("operator_attestation_sha256")
            or not isinstance(operator, Mapping)
            or reservation.get("operator_attestation_sha256") != operator.get("artifact_sha256")
            or reservation.get("cell_resource_limits") != _cohere_batch_resource_limits(plan, batch)
        ):
            raise CoverageExecutionError("completed Cohere reservation differs from admission")
    elif (
        _decimal(reservation.get("reserved_usd"), field="completed batch reserve") != reserve
        or {
            str(work_id): _decimal(value, field="completed cell allowance")
            for work_id, value in reservation.get("cell_allowances_usd", {}).items()
        }
        != allowances
    ):
        raise CoverageExecutionError("completed priced reservation differs from admission")


def execute_one_batch(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    admission: Mapping[str, Any],
    repo_root: Path,
    output_root: Path,
    runner: Runner | None = None,
    probe: Probe = _default_probe,
) -> dict[str, Any]:
    frozen_coordinator = _lexical_repo_path(
        repo_root=repo_root,
        relative=str(plan["execution_roots"]["coordinator"]),
        label="frozen coordinator root",
    )
    expected_output_root = frozen_coordinator.parent.parent
    lexical_output_root = _lexical_repo_path(
        repo_root=repo_root,
        relative=str(output_root),
        label="execution output root",
    )
    if lexical_output_root != expected_output_root:
        raise CoverageExecutionError(
            "output root differs from the plan-frozen coordinator/receipt root"
        )
    validate_plan(plan, repo_root=repo_root, output_root=output_root)
    verify_preflight(
        plan=plan,
        preflight=preflight,
        repo_root=repo_root,
        output_root=output_root,
    )
    admission_body = {key: value for key, value in admission.items() if key != "artifact_sha256"}
    if (
        set(admission)
        != {
            "schema_version",
            "plan_sha256",
            "preflight_sha256",
            "authorized_batch_id",
            "development_only",
            "official",
            "rank_eligible",
            "blocker_closures",
            "batch_reservations",
            "provider_or_epicure_calls_made_by_admission",
            "artifact_sha256",
        }
        or _contains_secret_key(admission)
        or admission.get("schema_version") != LIVE_ADMISSION_SCHEMA
        or admission.get("artifact_sha256") != _sha256(admission_body)
        or admission.get("plan_sha256") != plan["artifact_sha256"]
        or admission.get("preflight_sha256") != preflight["artifact_sha256"]
        or admission.get("development_only") is not True
        or admission.get("official") is not False
        or admission.get("rank_eligible") is not False
        or admission.get("provider_or_epicure_calls_made_by_admission") is not False
    ):
        raise AdmissionDenied("a passing exact live-admission artifact is required")
    blocker_evidence = _validate_blocker_closures(
        plan=plan,
        preflight=preflight,
        admission=admission,
        repo_root=repo_root,
    )
    cells, batches = _plan_maps(plan)
    coordinator = repo_root / str(plan["execution_roots"]["coordinator"])
    ledger_path = coordinator / "ledger.jsonl"
    global_lock = repo_root / str(plan["execution_roots"]["canonical_global_reservation_ledger"])
    source_root = repo_root / str(plan["execution_roots"]["canonical_global_source"])
    artifact_roots = (
        repo_root / "flavourbench/artifacts",
        repo_root / "artifacts",
    )
    outcomes: list[dict[str, Any]] = []
    live_runner = runner or (
        lambda cell, root, cap: _run_live_cell(cell, root, cap, plan=plan, repo_root=repo_root)
    )
    with _exclusive_runner_lock(global_lock):
        with _ledger_lock(ledger_path):
            entries = _load_ledger(ledger_path)
            state = _ledger_state(plan, entries)
            order = list(plan["batch_execution_order"])
            completed_priced = [
                batch_id
                for batch_id in order
                if batch_id in state["completed"]
                and batches[batch_id].get("execution_backend") != "cohere_direct"
            ]
            if completed_priced:
                completed_global_entries, completed_global_active = _canonical_global_state(
                    repo_root=repo_root,
                    ledger_path=global_lock,
                    source_root=source_root,
                )
                for completed_batch_id in completed_priced:
                    _validate_completed_priced_global_state(
                        plan=plan,
                        batch=batches[completed_batch_id],
                        reservation=state["reservations"][completed_batch_id],
                        state=state,
                        global_entries=completed_global_entries,
                        global_active=completed_global_active,
                    )
            receipt_dir = output_root / "run/receipts"
            receipts = _verified_receipts(
                receipt_dir,
                plan=plan,
                ledger_path=ledger_path,
                repo_root=repo_root,
            )
            unreceipted = [
                value for value in order if value in state["completed"] and value not in receipts
            ]
            if unreceipted:
                recovery_batch = batches[str(unreceipted[0])]
                recovery_reservation = state["reservations"][str(unreceipted[0])]
                _validate_completed_admission_binding(
                    plan=plan,
                    preflight=preflight,
                    admission=admission,
                    batch=recovery_batch,
                    reservation=recovery_reservation,
                    blocker_evidence=blocker_evidence,
                )
                receipt, destination = _materialize_receipt(
                    plan=plan,
                    preflight=preflight,
                    batch=recovery_batch,
                    ledger_path=ledger_path,
                    repo_root=repo_root,
                    output_root=output_root,
                )
                return {
                    "decision": "recovered_terminal_receipt_before_new_batch",
                    "receipt": receipt,
                    "receipt_path": str(destination),
                    "outcomes": receipt["outcomes"],
                }
            authorized_batch_id = str(admission.get("authorized_batch_id") or "")
            if authorized_batch_id in state["completed"]:
                completed_batch = batches[authorized_batch_id]
                _validate_completed_admission_binding(
                    plan=plan,
                    preflight=preflight,
                    admission=admission,
                    batch=completed_batch,
                    reservation=state["reservations"][authorized_batch_id],
                    blocker_evidence=blocker_evidence,
                )
                receipt_path = receipts.get(authorized_batch_id)
                if receipt_path is None:
                    raise CoverageExecutionError("terminal admission batch has no receipt")
                return {
                    "decision": "authorized_batch_already_terminal",
                    "receipt": _addressed(receipt_path),
                    "receipt_path": str(receipt_path),
                    "outcomes": [],
                }
            if state["active_batch_id"]:
                batch_id = str(state["active_batch_id"])
            else:
                remaining = [value for value in order if value not in state["completed"]]
                if not remaining:
                    return {"decision": "all_batches_already_terminal", "outcomes": []}
                batch_id = str(remaining[0])
            batch = batches[batch_id]
            reservation_documents = admission.get("batch_reservations")
            if (
                authorized_batch_id != batch_id
                or not isinstance(reservation_documents, Mapping)
                or set(map(str, reservation_documents)) != {batch_id}
            ):
                raise AdmissionDenied("live admission does not authorize exactly the next batch")
            admitted = reservation_documents.get(batch_id)
            if (
                not isinstance(admitted, Mapping)
                or admitted.get("work_item_ids") != batch["work_item_ids"]
            ):
                raise AdmissionDenied("live admission has no complete reserve for next batch")
            reserve, allowances, reservation_mode = _validate_admitted_reserve(
                plan=plan,
                batch=batch,
                admitted=admitted,
                blocker_evidence=blocker_evidence,
            )
            is_cohere = reservation_mode == "cohere_scholars_operator_quota"
            reservation = state["reservations"].get(batch_id)
            global_reservations: dict[str, Mapping[str, Any]] = {}
            if is_cohere:
                global_entries, _ = _canonical_global_state(
                    repo_root=repo_root,
                    ledger_path=global_lock,
                    source_root=source_root,
                )
                cohere_work_ids = {
                    str(cell["work_item_id"])
                    for cell in plan["cells"]
                    if cell.get("execution_backend") == "cohere_direct"
                }
                if any(
                    entry.get("coverage_successor_plan_sha256") == plan["artifact_sha256"]
                    and entry.get("coverage_successor_work_item_id") in cohere_work_ids
                    for entry in global_entries
                ):
                    raise CoverageExecutionError(
                        "Cohere work appeared in the canonical frontier USD ledger"
                    )
                expected_limits = _cohere_batch_resource_limits(plan, batch)
                operator = blocker_evidence["_cohere_operator_attestation"]
                if reservation is None:
                    reservation = _append_ledger(
                        ledger_path,
                        {
                            "event_type": "endpoint_batch_reserved",
                            "plan_sha256": plan["artifact_sha256"],
                            "preflight_sha256": preflight["artifact_sha256"],
                            "live_admission_sha256": admission["artifact_sha256"],
                            "batch_id": batch_id,
                            "work_item_ids": batch["work_item_ids"],
                            "reservation_kind": "cohere_scholars_operator_quota",
                            "usd_cost_or_reservation_claimed": False,
                            "resource_envelope_sha256": plan[
                                "cohere_prospective_resource_envelope"
                            ]["envelope_sha256"],
                            "operator_attestation_sha256": operator["artifact_sha256"],
                            "cell_resource_limits": expected_limits,
                            "reservation_unit": "one_exact_cohere_resource_quota_batch",
                            "replay_permitted": False,
                        },
                    )
                elif (
                    reservation.get("preflight_sha256") != preflight["artifact_sha256"]
                    or reservation.get("live_admission_sha256") != admission["artifact_sha256"]
                    or reservation.get("reservation_kind") != "cohere_scholars_operator_quota"
                    or reservation.get("operator_attestation_sha256") != operator["artifact_sha256"]
                    or reservation.get("cell_resource_limits") != expected_limits
                ):
                    raise CoverageExecutionError("active Cohere quota differs from live admission")
            elif reservation is None:
                global_entries, global_active = _canonical_global_state(
                    repo_root=repo_root,
                    ledger_path=global_lock,
                    source_root=source_root,
                )
                own_existing = _existing_global_batch_reservations(
                    plan=plan, batch=batch, entries=global_entries
                )
                own_active_refs = {
                    work_id: entry
                    for work_id, entry in own_existing.items()
                    if str(entry["entry_sha256"]) in global_active
                }
                if len(own_active_refs) != len(own_existing):
                    raise CoverageExecutionError(
                        "unbound global batch reservations were already finalized"
                    )
                own_active = sum(
                    (
                        global_active[str(entry["entry_sha256"])]
                        for entry in own_active_refs.values()
                    ),
                    Decimal(0),
                )
                other_global = sum(global_active.values(), Decimal(0)) - own_active
                later, later_snapshot = _later_source_exposure(plan, artifact_roots)
                other_local, active_rows = _generic_other_active_reservations(
                    artifact_roots,
                    excluded_ledgers=(ledger_path, global_lock),
                )
                missing_reserve = reserve - own_active
                if missing_reserve < 0:
                    raise CoverageExecutionError("partial global reserve exceeds exact batch bound")
                budget = rebase_budget(
                    baseline_snapshot_usd=Decimal(
                        plan["budget"]["bound_predecessor_snapshot_exposure_usd"]
                    ),
                    later_source_exposure_usd=later,
                    other_active_reservations_usd=other_global + other_local,
                    own_active_reservation_usd=own_active,
                    next_batch_reserve_usd=missing_reserve,
                )
                if not budget["admission_allowed"]:
                    raise AdmissionDenied("locked current budget rebase rejects the endpoint batch")
                global_reservations = _ensure_global_batch_reservations(
                    plan=plan,
                    batch=batch,
                    allowances=allowances,
                    repo_root=repo_root,
                    ledger_path=global_lock,
                    source_root=source_root,
                )
                reservation = _append_ledger(
                    ledger_path,
                    {
                        "event_type": "endpoint_batch_reserved",
                        "plan_sha256": plan["artifact_sha256"],
                        "preflight_sha256": preflight["artifact_sha256"],
                        "live_admission_sha256": admission["artifact_sha256"],
                        "batch_id": batch_id,
                        "work_item_ids": batch["work_item_ids"],
                        "reserved_usd": _decimal_text(reserve),
                        "cell_allowances_usd": {
                            work_id: _decimal_text(allowances[work_id])
                            for work_id in batch["work_item_ids"]
                        },
                        "global_reservation_entry_sha256s": {
                            work_id: global_reservations[work_id]["entry_sha256"]
                            for work_id in batch["work_item_ids"]
                        },
                        "locked_budget_rebase": budget,
                        "later_source_snapshot": later_snapshot,
                        "other_local_active_reservations": active_rows,
                        "other_canonical_global_active_reservations_usd": _decimal_text(
                            other_global
                        ),
                        "reservation_unit": "one_complete_endpoint_isolated_batch",
                        "replay_permitted": False,
                    },
                )
            else:
                if (
                    _decimal(reservation["reserved_usd"], field="active reserve") != reserve
                    or reservation.get("preflight_sha256") != preflight["artifact_sha256"]
                    or reservation.get("live_admission_sha256") != admission["artifact_sha256"]
                    or {
                        str(work_id): _decimal(value, field="active cell allowance")
                        for work_id, value in reservation["cell_allowances_usd"].items()
                    }
                    != allowances
                ):
                    raise CoverageExecutionError("active reservation differs from live admission")
                global_entries, global_active = _canonical_global_state(
                    repo_root=repo_root,
                    ledger_path=global_lock,
                    source_root=source_root,
                )
                global_reservations = _rebind_global_batch_reservations(
                    plan=plan,
                    batch=batch,
                    local_reservation=reservation,
                    local_state=state,
                    global_entries=global_entries,
                    global_active=global_active,
                )

            for work_id in batch["work_item_ids"]:
                state = _ledger_state(plan, _load_ledger(ledger_path))
                if state["incidents"]:
                    break
                if work_id in state["terminals"]:
                    if not is_cohere:
                        _record_global_source(
                            plan=plan,
                            cell=cells[work_id],
                            terminal=state["terminals"][work_id],
                            global_reservation=global_reservations[work_id],
                            repo_root=repo_root,
                            ledger_path=global_lock,
                            source_root=source_root,
                        )
                    continue
                cell = cells[work_id]
                cell_cap = allowances[work_id]
                if work_id in state["starts"]:
                    decision = _recover_started(
                        ledger_path=ledger_path,
                        plan=plan,
                        batch=batch,
                        cell=cell,
                        reservation=reservation,
                        endpoint_root=source_root,
                        cap_usd=cell_cap,
                        probe=probe,
                    )
                    refreshed = _ledger_state(plan, _load_ledger(ledger_path))
                    if work_id in refreshed["terminals"] and not is_cohere:
                        _record_global_source(
                            plan=plan,
                            cell=cell,
                            terminal=refreshed["terminals"][work_id],
                            global_reservation=global_reservations[work_id],
                            repo_root=repo_root,
                            ledger_path=global_lock,
                            source_root=source_root,
                        )
                    outcomes.append({"work_item_id": work_id, "decision": decision})
                    if decision == "uncertain_delivery_no_replay":
                        break
                    continue
                _append_ledger(
                    ledger_path,
                    {
                        "event_type": "item_execution_started",
                        "plan_sha256": plan["artifact_sha256"],
                        "batch_id": batch_id,
                        "work_item_id": work_id,
                        "run_id": cell["run_id"],
                        "arm_id": cell["arm_ids"]["epicure_on"],
                        "attempt_slots_sha256": cell["attempt_slots_sha256"],
                        "batch_reservation_entry_sha256": reservation["entry_sha256"],
                        "replay_permitted": False,
                    },
                )
                error: BaseException | None = None
                try:
                    live_runner(cell, source_root, cell_cap)
                except BaseException as caught:  # evidence, not exception type, decides replay
                    error = caught
                decision = _recover_started(
                    ledger_path=ledger_path,
                    plan=plan,
                    batch=batch,
                    cell=cell,
                    reservation=reservation,
                    endpoint_root=source_root,
                    cap_usd=cell_cap,
                    probe=probe,
                )
                refreshed = _ledger_state(plan, _load_ledger(ledger_path))
                if work_id in refreshed["terminals"] and not is_cohere:
                    _record_global_source(
                        plan=plan,
                        cell=cell,
                        terminal=refreshed["terminals"][work_id],
                        global_reservation=global_reservations[work_id],
                        repo_root=repo_root,
                        ledger_path=global_lock,
                        source_root=source_root,
                    )
                outcomes.append(
                    {
                        "work_item_id": work_id,
                        "decision": decision,
                        "runner_error_type": type(error).__name__ if error else None,
                    }
                )
                if decision == "uncertain_delivery_no_replay":
                    break

            state = _ledger_state(plan, _load_ledger(ledger_path))
            if not state["incidents"] and all(
                work_id in state["terminals"] for work_id in batch["work_item_ids"]
            ):
                if is_cohere and batch_id not in state["completed"]:
                    usages = [
                        state["terminals"][work_id]["resource_usage"]
                        for work_id in batch["work_item_ids"]
                    ]
                    reasoning_values = [usage.get("reasoning_tokens") for usage in usages]
                    _append_ledger(
                        ledger_path,
                        {
                            "event_type": "endpoint_batch_terminalized",
                            "plan_sha256": plan["artifact_sha256"],
                            "batch_id": batch_id,
                            "batch_reservation_entry_sha256": reservation["entry_sha256"],
                            "work_item_ids": batch["work_item_ids"],
                            "item_terminal_entry_sha256s": [
                                state["terminals"][work_id]["entry_sha256"]
                                for work_id in batch["work_item_ids"]
                            ],
                            "provider_reported_cost_usd": "0",
                            "cost_status": "unpriced_unknown",
                            "actual_cost_usd": None,
                            "usd_cost_or_reservation_claimed": False,
                            "canonical_usd_reservations_created": False,
                            "resource_envelope_totals": _cohere_resource_totals(
                                _cohere_batch_resource_limits(plan, batch)
                            ),
                            "resource_usage_totals": {
                                "successful_responses": sum(
                                    int(usage["successful_responses"]) for usage in usages
                                ),
                                "provider_request_attempts": sum(
                                    int(usage["provider_request_attempts"]) for usage in usages
                                ),
                                "tokens_prompt": _decimal_text(
                                    sum(
                                        (Decimal(str(usage["tokens_prompt"])) for usage in usages),
                                        Decimal(0),
                                    )
                                ),
                                "tokens_completion": sum(
                                    int(usage["tokens_completion"]) for usage in usages
                                ),
                                "reasoning_tokens": (
                                    None
                                    if any(value is None for value in reasoning_values)
                                    else sum(int(value) for value in reasoning_values)
                                ),
                                "tool_calls": sum(int(usage["tool_calls"]) for usage in usages),
                                "full_envelope_retained": True,
                            },
                            "resource_quota_terminalized": True,
                            "replay_permitted": False,
                        },
                    )
                    global_entries, _ = _canonical_global_state(
                        repo_root=repo_root,
                        ledger_path=global_lock,
                        source_root=source_root,
                    )
                    if any(
                        entry.get("coverage_successor_plan_sha256") == plan["artifact_sha256"]
                        and entry.get("coverage_successor_work_item_id")
                        in set(batch["work_item_ids"])
                        for entry in global_entries
                    ):
                        raise CoverageExecutionError(
                            "Cohere terminal contaminated the canonical frontier USD ledger"
                        )
                elif not is_cohere:
                    actual = sum(
                        (
                            Decimal(state["terminals"][work_id]["actual_cost_usd"])
                            for work_id in batch["work_item_ids"]
                        ),
                        Decimal(0),
                    )
                    if actual > reserve:
                        raise CoverageExecutionError("batch actual cost exceeds reservation")
                if not is_cohere and batch_id not in state["completed"]:
                    _, global_active = _canonical_global_state(
                        repo_root=repo_root,
                        ledger_path=global_lock,
                        source_root=source_root,
                    )
                    retained_ids = [
                        str(global_reservations[work_id]["entry_sha256"])
                        for work_id in batch["work_item_ids"]
                        if str(global_reservations[work_id]["entry_sha256"]) in global_active
                    ]
                    retained = sum((global_active[digest] for digest in retained_ids), Decimal(0))
                    _append_ledger(
                        ledger_path,
                        {
                            "event_type": "endpoint_batch_terminalized",
                            "plan_sha256": plan["artifact_sha256"],
                            "batch_id": batch_id,
                            "batch_reservation_entry_sha256": reservation["entry_sha256"],
                            "work_item_ids": batch["work_item_ids"],
                            "item_terminal_entry_sha256s": [
                                state["terminals"][work_id]["entry_sha256"]
                                for work_id in batch["work_item_ids"]
                            ],
                            "actual_cost_usd": _decimal_text(actual),
                            "conservative_exposure_usd": _decimal_text(actual + retained),
                            "canonical_global_reservations_retained": retained_ids,
                            "canonical_global_retained_usd": _decimal_text(retained),
                            "whole_batch_reservation_released": retained == 0,
                            "replay_permitted": False,
                        },
                    )

            final_entries = _load_ledger(ledger_path)
            final_state = _ledger_state(plan, final_entries)
            if batch_id in final_state["completed"]:
                receipt, destination = _materialize_receipt(
                    plan=plan,
                    preflight=preflight,
                    batch=batch,
                    ledger_path=ledger_path,
                    repo_root=repo_root,
                    output_root=output_root,
                )
                return {
                    "decision": "one_endpoint_batch_terminal",
                    "receipt": receipt,
                    "receipt_path": str(destination),
                    "outcomes": outcomes,
                }
            return {
                "decision": "safely_stopped_reservation_retained",
                "batch_id": batch_id,
                "outcomes": outcomes,
                "observed_model_pair_family_support": {
                    "supported_cells": 407,
                    "empty_cells": 73,
                    "total_cells": 480,
                    "successor_changes_observed_support": False,
                },
            }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--live-admission", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--max-new-batches", type=int, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.confirm != EXECUTION_CONFIRMATION or args.max_new_batches != 1:
        raise SystemExit(
            f"execution requires --confirm {EXECUTION_CONFIRMATION} --max-new-batches 1"
        )
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    try:
        plan = _addressed(args.plan)
        preflight = _addressed(args.preflight)
        admission = _addressed(args.live_admission)
        if (
            plan.get("schema_version") != PLAN_SCHEMA
            or preflight.get("schema_version") != PREFLIGHT_SCHEMA
        ):
            raise CoverageExecutionError("executor received an incompatible plan or preflight")
        result = execute_one_batch(
            plan=plan,
            preflight=preflight,
            admission=admission,
            repo_root=repo_root,
            output_root=output_root,
        )
    except (CoverageExecutionError, CoverageSuccessorError, AdmissionDenied) as error:
        raise SystemExit(f"coverage successor execution refused: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
