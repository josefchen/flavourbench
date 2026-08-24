"""Crash-safe admission for one direct Qwen 3.8 exploratory Epicure pair.

The provider does not publish a price or charged amount for the mutable
``qwen3.8-max`` alias.  This ledger therefore treats the complete admitted
ceiling as permanent exposure.  It never releases the reserve from a token
usage record and never interprets a recorded zero provider cost as free.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from .budget_policy import provider_account_hard_cap_micros, provider_account_scope_sha256
from .execution_policy import ExecutionPolicy, verify_policy_document
from .frontier_contract_runner import (
    ContractCandidate,
    load_candidate_manifest,
    select_candidates,
)
from .qwencloud_catalog import QWEN38_TOOL_AUTO_INSTRUCTION
from .real_task_bank import sha256_json, sha256_text

LEDGER_SCHEMA_VERSION = "flavourbench-qwencloud-exploratory-ledger-v1"
GO_TEMPLATE_SCHEMA_VERSION = "flavourbench-qwencloud-one-pair-pi-go-template-v3"
GO_AUTHORIZATION_SCHEMA_VERSION = "flavourbench-qwencloud-one-pair-human-pi-go-v3"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-direct-provider-preflight-v1"
RESERVATION_CONFIRMATION = "RESERVE_QWEN38_ALIAS_FULL_USD_2_CEILING_V1"
HUMAN_PI_CONFIRMATION = "AUTHORIZE_ONE_QWEN38_ALIAS_EPICURE_OFF_ON_PAIR_V1"
LIVE_CONFIRMATION = "EXECUTE_ONE_QWEN38_ALIAS_EPICURE_OFF_ON_PAIR_V1"
MODEL_ID = "qwen3.8-max"
PROVIDER_SLUG = "qwencloud-direct"
EXECUTION_BACKEND = "qwencloud_direct"
MODEL_IDENTITY_LABEL = "catalog_pinned_at_observation_not_a_frozen_model"
TASK_ID = "fb-s0-composition-024"
TASK_FAMILY = "composition"
MAX_PAIR_CEILING_USD = Decimal("2")
ACCOUNT_ADMISSION_FRACTION = Decimal("0.85")
CONDITIONS = ("epicure_off", "epicure_on")
EPICURE_MCP_URL = "http://127.0.0.1:8081/mcp"
EPICURE_PROVENANCE_URL = "http://127.0.0.1:8081/provenance"
PREDECESSOR_FAILURE_ARTIFACT_SHA256 = (
    "a9e863df14ef690fd194cb5689da3f3c947e615ac017d2264aff51b3b0d51a96"
)
TOOL_CHOICE_TRANSPORT_MODE = "auto_with_required_success_postcondition"
MESSAGE_CANONICALIZATION = "official_qwen_chat_tool_continuation_shape_v1"


class QwenCloudSmokeAdmissionError(RuntimeError):
    """The exploratory pair did not satisfy its exact admission contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _live_source_sha256(value: object) -> str:
    """Match the established live-smoke producer's ASCII JSON canonicalization."""

    rendered = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise QwenCloudSmokeAdmissionError(f"{field} is not a decimal") from error
    if not result.is_finite() or result < 0:
        raise QwenCloudSmokeAdmissionError(f"{field} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _regular_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise QwenCloudSmokeAdmissionError(f"{label} must be a regular file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise QwenCloudSmokeAdmissionError(f"could not read {label}") from error
    if not isinstance(value, dict):
        raise QwenCloudSmokeAdmissionError(f"{label} must contain an object")
    return value


def _verified_artifact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    value = _regular_json(path, label=label)
    digest = str(value.get("artifact_sha256") or "")
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if digest != expected_sha256 or len(digest) != 64 or _sha256(body) != digest:
        raise QwenCloudSmokeAdmissionError(f"{label} content address does not verify")
    return value


def qwen38_smoke_execution_policy() -> ExecutionPolicy:
    """Return the exact no-reasoning-control policy for the direct alias smoke."""

    policy = ExecutionPolicy(
        max_output_tokens=4_096,
        max_tool_rounds=8,
        max_tool_result_bytes=32_768,
        max_cumulative_tool_result_bytes=98_304,
        max_tool_calls_per_round=4,
        max_tool_calls_total=16,
        max_provider_attempts=1,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=20_260_715,
        tool_argument_repair_turns=1,
        approximate_non_user_prompt_bytes=2_000,
        conservative_bytes_per_token=3,
        pair_arm_scheduling="concurrent",
        final_response_mode="plain_text",
        max_intermediate_tokens=4_096,
        required_tool_contract_max_intermediate_tokens=2_048,
        matched_planning=True,
        evidence_protocol="matched_evidence_v2",
        intermediate_reasoning_effort=None,
        final_reasoning_effort=None,
        required_tool_contract_protocol="direct_tool_first_v1",
        tool_catalog_bytes_bound=24_000,
        epicure_on_tool_required=True,
    )
    policy.validate()
    return policy


@dataclass(frozen=True)
class SmokeBinding:
    route_manifest_sha256: str
    candidate: ContractCandidate
    task_validity_sha256: str
    task: Mapping[str, Any]
    execution_policy: ExecutionPolicy
    cap_usd: Decimal
    work_item_id: str
    frozen_run_id: str
    frozen_attempt_slots: tuple[Mapping[str, Any], ...]


def _task_record(
    *,
    task_validity_path: Path,
    expected_task_validity_sha256: str,
) -> tuple[str, Mapping[str, Any]]:
    dossier = _verified_artifact(
        task_validity_path,
        expected_sha256=expected_task_validity_sha256,
        label="task-validity dossier",
    )
    tasks = dossier.get("tasks")
    if not isinstance(tasks, list):
        raise QwenCloudSmokeAdmissionError("task-validity dossier has no tasks")
    matches = [
        task
        for task in tasks
        if isinstance(task, Mapping) and task.get("task_id") == TASK_ID
    ]
    if len(matches) != 1:
        raise QwenCloudSmokeAdmissionError("composition-024 is absent or duplicated")
    task = matches[0]
    prompt = str(task.get("prompt") or "")
    surface = task.get("surface_dependency_screen")
    if (
        task.get("family") != TASK_FAMILY
        or sha256_text(prompt) != task.get("prompt_sha256")
        or not str(task.get("source_license") or "").startswith("CC BY-SA ")
        or not isinstance(surface, Mapping)
        or surface.get("status") != "pass"
        or surface.get("failure_reasons") != []
        or task.get("confirmatory_eligible") is not False
        or task.get("rank_eligible") is not False
    ):
        raise QwenCloudSmokeAdmissionError(
            "composition-024 is not the licensed, surface-clean development task"
        )
    return str(dossier["artifact_sha256"]), task


def _candidate(
    *,
    route_manifest_path: Path,
    expected_route_manifest_sha256: str,
) -> ContractCandidate:
    manifest = load_candidate_manifest(
        route_manifest_path,
        expected_digest=expected_route_manifest_sha256,
    )
    matches = select_candidates(manifest, (MODEL_ID,))
    if len(matches) != 1:
        raise QwenCloudSmokeAdmissionError("Qwen 3.8 route is absent or duplicated")
    candidate = matches[0]
    contract = candidate.backend_contract
    pricing = candidate.endpoint.get("pricing")
    if (
        candidate.execution_backend != EXECUTION_BACKEND
        or candidate.provider_tag != PROVIDER_SLUG
        or candidate.canonical_model_slug != MODEL_ID
        or contract.get("identity_kind") != "mutable_alias"
        or contract.get("model_identity_label") != MODEL_IDENTITY_LABEL
        or contract.get("catalog_pinned_at_observation") is not True
        or contract.get("mutable_alias_execution_requires_explicit_opt_in") is not True
        or contract.get("official") is not False
        or contract.get("season_eligible") is not False
        or contract.get("rank_eligible") is not False
        or contract.get("allow_fallbacks") is not False
        or contract.get("openrouter_alternate_route")
        != "separate_stratum_only_no_identity_pooling"
        or contract.get("tool_choice_transport_mode")
        != TOOL_CHOICE_TRANSPORT_MODE
        or contract.get("tool_choice_required_supported") is not False
        or contract.get("required_success_postcondition")
        != "at_least_one_successful_real_epicure_tool_trace"
        or contract.get("tool_selection_system_instruction")
        != QWEN38_TOOL_AUTO_INSTRUCTION
        or contract.get("tool_selection_system_instruction_sha256")
        != sha256_json(QWEN38_TOOL_AUTO_INSTRUCTION)
        or contract.get("message_canonicalization") != MESSAGE_CANONICALIZATION
        or contract.get("predecessor_failure_artifact_sha256")
        != PREDECESSOR_FAILURE_ARTIFACT_SHA256
        or not isinstance(pricing, Mapping)
        or pricing.get("provider_rate_known") is not False
        or pricing.get("zero_values_mean") != "unknown_cost_not_free"
    ):
        raise QwenCloudSmokeAdmissionError(
            "route is not the exact fail-closed mutable-alias contract"
        )
    for field in ("catalog_sha256", "catalog_entry_sha256"):
        if len(str(contract.get(field) or "")) != 64:
            raise QwenCloudSmokeAdmissionError(f"route lacks exact {field}")
    observed_at = str(contract.get("catalog_observed_at") or "")
    try:
        parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise QwenCloudSmokeAdmissionError("route catalog observation is invalid") from error
    if parsed.tzinfo is None:
        raise QwenCloudSmokeAdmissionError("route catalog observation lacks a timezone")
    return candidate


def _attempt_slots(
    *,
    work_item_id: str,
    frozen_run_id: str,
    policy: ExecutionPolicy,
) -> tuple[Mapping[str, Any], ...]:
    phases: dict[str, list[str]] = {
        "epicure_off": ["planning", "evidence_decision", "final"],
        "epicure_on": [
            "planning",
            "mcp_session",
            *[f"tool_round_{index}" for index in range(policy.max_tool_rounds)],
            *[
                f"mcp_tool_{round_index}_{call_index}"
                for round_index in range(policy.max_tool_rounds)
                for call_index in range(policy.max_tool_calls_per_round)
            ],
            "final",
        ],
    }
    namespace = uuid.UUID(frozen_run_id)
    slots: list[Mapping[str, Any]] = []
    for condition in CONDITIONS:
        arm_id = f"{frozen_run_id}:{condition}"
        for phase in phases[condition]:
            slots.append(
                {
                    "arm_id": arm_id,
                    "phase": phase,
                    "attempt_index": 0,
                    "attempt_id": str(
                        uuid.uuid5(
                            namespace,
                            f"{work_item_id}:{condition}:{phase}:0",
                        )
                    ),
                }
            )
    if len({str(slot["attempt_id"]) for slot in slots}) != len(slots):
        raise QwenCloudSmokeAdmissionError("attempt-slot identifiers are not unique")
    return tuple(slots)


def build_smoke_binding(
    *,
    route_manifest_path: Path,
    expected_route_manifest_sha256: str,
    task_validity_path: Path,
    expected_task_validity_sha256: str,
    cap_usd: Decimal,
) -> SmokeBinding:
    candidate = _candidate(
        route_manifest_path=route_manifest_path,
        expected_route_manifest_sha256=expected_route_manifest_sha256,
    )
    task_validity_sha256, task = _task_record(
        task_validity_path=task_validity_path,
        expected_task_validity_sha256=expected_task_validity_sha256,
    )
    policy = qwen38_smoke_execution_policy()
    pricing = candidate.endpoint["pricing"]
    route_ceiling = _decimal(
        pricing.get("operational_reservation_ceiling_usd"),
        field="route operational ceiling",
    )
    if cap_usd != route_ceiling or cap_usd <= 0 or cap_usd > MAX_PAIR_CEILING_USD:
        raise QwenCloudSmokeAdmissionError(
            "pair cap must equal the route's positive full ceiling and may not exceed $2"
        )
    work_payload = {
        "schema_version": "flavourbench-qwencloud-one-pair-work-item-v1",
        "route_manifest_sha256": expected_route_manifest_sha256,
        "catalog_sha256": candidate.backend_contract["catalog_sha256"],
        "catalog_entry_sha256": candidate.backend_contract["catalog_entry_sha256"],
        "catalog_observed_at": candidate.backend_contract["catalog_observed_at"],
        "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
        "backend_contract_sha256": candidate.backend_contract_sha256,
        "task_validity_sha256": task_validity_sha256,
        "task_id": task["task_id"],
        "task_sha256": task["task_sha256"],
        "prompt_sha256": task["prompt_sha256"],
        "execution_policy_sha256": policy.sha256,
        "conditions": list(CONDITIONS),
        "cap_usd": _decimal_text(cap_usd),
        "official": False,
        "rank_eligible": False,
    }
    work_item_id = _sha256(work_payload)
    frozen_run_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"flavourbench:qwencloud:{work_item_id}")
    )
    return SmokeBinding(
        route_manifest_sha256=expected_route_manifest_sha256,
        candidate=candidate,
        task_validity_sha256=task_validity_sha256,
        task=task,
        execution_policy=policy,
        cap_usd=cap_usd,
        work_item_id=work_item_id,
        frozen_run_id=frozen_run_id,
        frozen_attempt_slots=_attempt_slots(
            work_item_id=work_item_id,
            frozen_run_id=frozen_run_id,
            policy=policy,
        ),
    )


def _entry_digest(entry: Mapping[str, Any]) -> str:
    body = dict(entry)
    body.pop("entry_sha256", None)
    return _sha256(body)


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "api_key",
        "authorization_header",
        "cloudflare_ai_gateway_token",
        "environment",
        "mcp_token",
        "stderr",
        "stdout",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise QwenCloudSmokeAdmissionError("QwenCloud ledger must be a regular file")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise QwenCloudSmokeAdmissionError(
                f"blank QwenCloud ledger line {line_number}"
            )
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise QwenCloudSmokeAdmissionError(
                f"invalid QwenCloud ledger JSON line {line_number}"
            ) from error
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != LEDGER_SCHEMA_VERSION
            or entry.get("sequence") != line_number
            or entry.get("previous_entry_sha256") != previous
            or entry.get("entry_sha256") != _entry_digest(entry)
        ):
            raise QwenCloudSmokeAdmissionError(
                f"QwenCloud ledger chain fails at line {line_number}"
            )
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    validate_ledger_state(entries)
    return entries


def append_ledger_event(
    path: Path,
    event: Mapping[str, Any],
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    if _contains_forbidden_key(event):
        raise QwenCloudSmokeAdmissionError("ledger event contains a secret-bearing field")
    protected = {
        "entry_sha256",
        "previous_entry_sha256",
        "recorded_at",
        "schema_version",
        "sequence",
    }
    if protected.intersection(event):
        raise QwenCloudSmokeAdmissionError("ledger event overrides chain fields")
    if event.get("event_type") not in {
        "reservation_created",
        "reservation_scope_bound",
        "reservation_scope_adjusted",
        "execution_started",
        "source_terminalized",
        "execution_incident",
    }:
        raise QwenCloudSmokeAdmissionError("unsupported QwenCloud ledger event")
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    entries = load_ledger(path)
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(entries) + 1,
        "recorded_at": recorded_at or _utc_now(),
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        **dict(event),
    }
    entry["entry_sha256"] = _entry_digest(entry)
    line = _canonical(entry) + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, line) != len(line):
            raise OSError("short QwenCloud ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    validate_ledger_state(load_ledger(path))
    return entry


@dataclass(frozen=True)
class LedgerState:
    reservations: Mapping[str, Mapping[str, Any]]
    reservations_by_work_item: Mapping[str, Mapping[str, Any]]
    scope_bindings: Mapping[str, Mapping[str, Any]]
    scope_adjustments: Mapping[str, Mapping[str, Any]]
    starts: Mapping[str, Mapping[str, Any]]
    terminalizations: Mapping[str, Mapping[str, Any]]
    incidents: Mapping[str, Mapping[str, Any]]
    total_retained_exposure_usd: Decimal


def validate_ledger_state(entries: Sequence[Mapping[str, Any]]) -> LedgerState:
    reservations: dict[str, Mapping[str, Any]] = {}
    by_work: dict[str, Mapping[str, Any]] = {}
    scope_bindings: dict[str, Mapping[str, Any]] = {}
    scope_adjustments: dict[str, Mapping[str, Any]] = {}
    starts: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    incidents: dict[str, Mapping[str, Any]] = {}
    exposure = Decimal(0)
    for entry in entries:
        event_type = entry.get("event_type")
        reservation_sha = str(entry.get("reservation_entry_sha256") or "")
        work_item_id = str(entry.get("work_item_id") or "")
        if event_type == "reservation_created":
            reservation_sha = str(entry.get("entry_sha256") or "")
            amount = _decimal(entry.get("reserved_usd"), field="reserved_usd")
            if (
                not reservation_sha
                or not work_item_id
                or reservation_sha in reservations
                or work_item_id in by_work
                or amount <= 0
                or entry.get("provider_cost_known") is not False
                or entry.get("full_ceiling_permanently_retained") is not True
            ):
                raise QwenCloudSmokeAdmissionError("invalid or duplicate reservation")
            reservations[reservation_sha] = entry
            by_work[work_item_id] = entry
            exposure += amount
        elif event_type == "reservation_scope_bound":
            reservation = reservations.get(reservation_sha)
            if (
                reservation is None
                or reservation.get("work_item_id") != work_item_id
                or reservation_sha in scope_bindings
                or entry.get("season_scope") != "season1_unranked_development"
                or _decimal(entry.get("season_budget_cap_usd"), field="season cap")
                != _decimal(reservation.get("reserved_usd"), field="reserved exposure")
                or entry.get("provider_account_scope_sha256")
                != reservation.get("provider_account_scope_sha256")
                or entry.get("full_ceiling_permanently_retained") is not True
            ):
                raise QwenCloudSmokeAdmissionError("invalid reservation scope binding")
            scope_bindings[reservation_sha] = entry
        elif event_type == "reservation_scope_adjusted":
            reservation = reservations.get(reservation_sha)
            scope_binding = scope_bindings.get(reservation_sha)
            season_retained = _decimal(
                entry.get("season_retained_exposure_usd"),
                field="adjusted season retained exposure",
            )
            season_cap = _decimal(
                entry.get("season_budget_cap_usd"),
                field="adjusted season cap",
            )
            if (
                reservation is None
                or scope_binding is None
                or reservation.get("work_item_id") != work_item_id
                or reservation_sha in scope_adjustments
                or entry.get("supersedes_scope_binding_entry_sha256")
                != scope_binding.get("entry_sha256")
                or entry.get("season_scope") != "season1_unranked_development"
                or season_retained != exposure
                or season_cap < season_retained
                or entry.get("provider_account_scope_sha256")
                != reservation.get("provider_account_scope_sha256")
                or entry.get("full_ceiling_permanently_retained") is not True
            ):
                raise QwenCloudSmokeAdmissionError("invalid reservation scope adjustment")
            scope_adjustments[reservation_sha] = entry
        elif event_type == "execution_started":
            reservation = reservations.get(reservation_sha)
            if (
                reservation is None
                or reservation_sha not in scope_bindings
                or reservation.get("work_item_id") != work_item_id
                or reservation_sha in starts
                or reservation_sha in terminals
                or len(str(entry.get("human_pi_authorization_sha256") or "")) != 64
                or len(str(entry.get("preflight_artifact_sha256") or "")) != 64
            ):
                raise QwenCloudSmokeAdmissionError("invalid or duplicate execution start")
            starts[reservation_sha] = entry
        elif event_type == "source_terminalized":
            reservation = reservations.get(reservation_sha)
            if (
                reservation is None
                or reservation_sha not in starts
                or reservation_sha in terminals
                or reservation.get("work_item_id") != work_item_id
                or _decimal(entry.get("retained_exposure_usd"), field="retained exposure")
                != _decimal(reservation.get("reserved_usd"), field="reserved exposure")
                or entry.get("provider_cost_known") is not False
                or entry.get("zero_recorded_cost_means") != "unknown_not_free"
            ):
                raise QwenCloudSmokeAdmissionError("invalid source terminalization")
            terminals[reservation_sha] = entry
        elif event_type == "execution_incident":
            reservation = reservations.get(reservation_sha)
            if (
                reservation is None
                or reservation_sha not in starts
                or reservation_sha in incidents
                or reservation.get("work_item_id") != work_item_id
                or entry.get("safe_to_replay") is not False
                or _decimal(entry.get("retained_exposure_usd"), field="incident exposure")
                != _decimal(reservation.get("reserved_usd"), field="reserved exposure")
            ):
                raise QwenCloudSmokeAdmissionError("invalid execution incident")
            incidents[reservation_sha] = entry
    return LedgerState(
        reservations=reservations,
        reservations_by_work_item=by_work,
        scope_bindings=scope_bindings,
        scope_adjustments=scope_adjustments,
        starts=starts,
        terminalizations=terminals,
        incidents=incidents,
        total_retained_exposure_usd=exposure,
    )


@contextmanager
def ledger_lock(path: Path) -> Iterable[TextIO]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise QwenCloudSmokeAdmissionError("QwenCloud ledger lock may not be a symlink")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reservation_event(binding: SmokeBinding, *, prior_exposure: Decimal) -> dict[str, Any]:
    candidate = binding.candidate
    return {
        "event_type": "reservation_created",
        "work_item_id": binding.work_item_id,
        "frozen_run_id": binding.frozen_run_id,
        "route_manifest_sha256": binding.route_manifest_sha256,
        "catalog_sha256": candidate.backend_contract["catalog_sha256"],
        "catalog_entry_sha256": candidate.backend_contract["catalog_entry_sha256"],
        "catalog_observed_at": candidate.backend_contract["catalog_observed_at"],
        "model_identity_label": MODEL_IDENTITY_LABEL,
        "model_id": MODEL_ID,
        "canonical_model_slug": MODEL_ID,
        "provider_slug": PROVIDER_SLUG,
        "execution_backend": EXECUTION_BACKEND,
        "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
        "backend_contract_sha256": candidate.backend_contract_sha256,
        "task_validity_sha256": binding.task_validity_sha256,
        "task_id": binding.task["task_id"],
        "task_sha256": binding.task["task_sha256"],
        "task_family": binding.task["family"],
        "prompt_sha256": binding.task["prompt_sha256"],
        "execution_policy_sha256": binding.execution_policy.sha256,
        "attempt_slot_plan_sha256": _sha256(list(binding.frozen_attempt_slots)),
        "conditions": list(CONDITIONS),
        "maximum_pairs": 1,
        "maximum_response_arms": 2,
        "reserved_usd": _decimal_text(binding.cap_usd),
        "total_retained_exposure_before_usd": _decimal_text(prior_exposure),
        "provider_account_cap_usd": _decimal_text(
            Decimal(provider_account_hard_cap_micros(EXECUTION_BACKEND)) / Decimal(1_000_000)
        ),
        "provider_account_scope_sha256": provider_account_scope_sha256(EXECUTION_BACKEND),
        "cost_accounting_basis": "full_unpriced_budget_ceiling_permanently_retained",
        "provider_cost_known": False,
        "full_ceiling_permanently_retained": True,
        "zero_recorded_provider_cost_means": "unknown_not_free",
        "official": False,
        "season_eligible": False,
        "rank_eligible": False,
        "automatic_fallback": False,
        "openrouter_route_policy": "separate_explicit_stratum_only",
    }


def _write_content_addressed(
    directory: Path,
    prefix: str,
    payload: Mapping[str, Any],
) -> Path:
    body = dict(payload)
    if "artifact_sha256" in body:
        raise QwenCloudSmokeAdmissionError("artifact payload already has a digest")
    digest = _sha256(body)
    artifact = {**body, "artifact_sha256": digest}
    rendered = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise QwenCloudSmokeAdmissionError("content-addressed artifact conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=directory,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def build_go_template(
    *,
    binding: SmokeBinding,
    reservation: Mapping[str, Any],
    scope_binding: Mapping[str, Any],
    effective_scope: Mapping[str, Any],
    ledger_path: Path,
    human_pi_identity_record: Mapping[str, Any],
) -> dict[str, Any]:
    identity = human_pi_identity_record.get("reviewer_identity")
    if (
        human_pi_identity_record.get("authorization_is_transparent_human_pi_record")
        is not True
        or not isinstance(identity, Mapping)
        or identity.get("full_name") != "Josef Chen"
        or identity.get("role") != "human_principal_investigator"
    ):
        raise QwenCloudSmokeAdmissionError(
            "standing Human-PI identity record is not the transparent Josef record"
        )
    candidate = binding.candidate
    return {
        "schema_version": GO_TEMPLATE_SCHEMA_VERSION,
        "record_role": "one_pair_human_pi_go_template_not_authorization",
        "authorization_status": "fresh_exact_human_pi_confirmation_required",
        "standing_human_pi_identity_record_sha256": human_pi_identity_record[
            "artifact_sha256"
        ],
        "required_human_pi": {
            "full_name": "Josef Chen",
            "role": "human_principal_investigator",
            "prior_record_is_identity_and_overall_envelope_evidence_only": True,
            "prior_record_does_not_substitute_for_this_exact_go": True,
            "required_confirmation": HUMAN_PI_CONFIRMATION,
        },
        "reservation": {
            "ledger_path": str(ledger_path.resolve()),
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "entry_sha256": reservation["entry_sha256"],
            "scope_binding_entry_sha256": scope_binding["entry_sha256"],
            "scope_adjustment_entry_sha256": (
                effective_scope["entry_sha256"]
                if effective_scope.get("event_type") == "reservation_scope_adjusted"
                else None
            ),
            "sequence": reservation["sequence"],
            "work_item_id": binding.work_item_id,
            "full_ceiling_usd": _decimal_text(binding.cap_usd),
            "season_scope": effective_scope["season_scope"],
            "season_budget_cap_usd": effective_scope["season_budget_cap_usd"],
            "season_retained_exposure_usd": effective_scope[
                "season_retained_exposure_usd"
            ],
            "provider_account_scope_sha256": effective_scope[
                "provider_account_scope_sha256"
            ],
            "provider_cost_known": False,
            "full_ceiling_permanently_retained": True,
            "zero_recorded_provider_cost_means": "unknown_not_free",
        },
        "model_identity": {
            "requested_model_id": MODEL_ID,
            "canonical_model_slug": MODEL_ID,
            "provider_slug": PROVIDER_SLUG,
            "execution_backend": EXECUTION_BACKEND,
            "identity_kind": "mutable_alias",
            "identity_label": MODEL_IDENTITY_LABEL,
            "catalog_sha256": candidate.backend_contract["catalog_sha256"],
            "catalog_entry_sha256": candidate.backend_contract[
                "catalog_entry_sha256"
            ],
            "catalog_observed_at": candidate.backend_contract["catalog_observed_at"],
            "route_manifest_sha256": binding.route_manifest_sha256,
            "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            "backend_contract_sha256": candidate.backend_contract_sha256,
            "catalog_pinned_at_observation": True,
            "frozen_model_claimed": False,
            "tool_choice_transport_mode": candidate.backend_contract[
                "tool_choice_transport_mode"
            ],
            "tool_selection_system_instruction_sha256": candidate.backend_contract[
                "tool_selection_system_instruction_sha256"
            ],
            "message_canonicalization": candidate.backend_contract[
                "message_canonicalization"
            ],
            "predecessor_failure_artifact_sha256": candidate.backend_contract[
                "predecessor_failure_artifact_sha256"
            ],
        },
        "task": {
            "task_validity_sha256": binding.task_validity_sha256,
            "task_id": binding.task["task_id"],
            "task_sha256": binding.task["task_sha256"],
            "family": binding.task["family"],
            "prompt_sha256": binding.task["prompt_sha256"],
            "prompt": binding.task["prompt"],
            "source_url": binding.task["source_url"],
            "source_license": binding.task["source_license"],
            "surface_dependency_screen": binding.task["surface_dependency_screen"],
            "confirmatory_eligible": False,
            "rank_eligible": False,
        },
        "execution": {
            "frozen_run_id": binding.frozen_run_id,
            "conditions": list(CONDITIONS),
            "maximum_pairs": 1,
            "maximum_response_arms": 2,
            "arm_scheduling": "concurrent",
            "maximum_provider_attempts_per_request": 1,
            "maximum_tool_rounds": binding.execution_policy.max_tool_rounds,
            "maximum_tool_calls_total": binding.execution_policy.max_tool_calls_total,
            "epicure_on_tool_required": True,
            "allow_mutable_alias_exploratory": True,
            "attempt_slots": list(binding.frozen_attempt_slots),
            "attempt_slot_plan_sha256": _sha256(list(binding.frozen_attempt_slots)),
            "execution_policy": binding.execution_policy.document(),
            "execution_policy_sha256": binding.execution_policy.sha256,
            "replay_policy": "permanently_block_after_execution_started",
            "provider_fallback": False,
            "openrouter_alternate_route": "separate_explicit_stratum_only",
            "epicure_transport": {
                "mcp_url": EPICURE_MCP_URL,
                "provenance_url": EPICURE_PROVENANCE_URL,
                "private_host_binding_required": True,
            },
        },
        "required_preflight": {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "provider_calls_made": False,
            "must_bind_route_task_policy_reservation_and_current_epicure_attestation": True,
            "must_be_content_addressed": True,
            "fresh_authorization_must_bind_preflight_sha256": True,
        },
        "claim_boundary": {
            "official": False,
            "season_eligible": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_observations_authorized": 0,
            "leaderboard_comparisons_authorized": 0,
            "cost_reconciled": False,
            "result_label": "catalog-pinned-at-observation exploratory development pair",
        },
    }


def reserve_and_write_template(
    *,
    binding: SmokeBinding,
    ledger_path: Path,
    output_directory: Path,
    human_pi_identity_record_path: Path,
    expected_human_pi_identity_record_sha256: str,
    confirmation: str,
) -> tuple[Mapping[str, Any], Path]:
    if confirmation != RESERVATION_CONFIRMATION:
        raise QwenCloudSmokeAdmissionError("exact $2 reservation confirmation is required")
    pi_record = _verified_artifact(
        human_pi_identity_record_path,
        expected_sha256=expected_human_pi_identity_record_sha256,
        label="standing Human-PI identity record",
    )
    with ledger_lock(ledger_path):
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        existing = state.reservations_by_work_item.get(binding.work_item_id)
        if existing is None:
            account_cap = (
                Decimal(provider_account_hard_cap_micros(EXECUTION_BACKEND))
                / Decimal(1_000_000)
            )
            projected = state.total_retained_exposure_usd + binding.cap_usd
            if projected > account_cap or projected > account_cap * ACCOUNT_ADMISSION_FRACTION:
                raise QwenCloudSmokeAdmissionError(
                    "QwenCloud retained exposure exceeds the governed admission ceiling"
                )
            reservation = append_ledger_event(
                ledger_path,
                _reservation_event(
                    binding,
                    prior_exposure=state.total_retained_exposure_usd,
                ),
            )
        else:
            reservation = existing
            expected = _reservation_event(
                binding,
                prior_exposure=_decimal(
                    reservation.get("total_retained_exposure_before_usd"),
                    field="reservation prior exposure",
                ),
            )
            for key, value in expected.items():
                if reservation.get(key) != value:
                    raise QwenCloudSmokeAdmissionError(
                        "existing reservation differs from the exact work item"
                    )
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        scope_binding = state.scope_bindings.get(str(reservation["entry_sha256"]))
        if scope_binding is None:
            scope_binding = append_ledger_event(
                ledger_path,
                {
                    "event_type": "reservation_scope_bound",
                    "reservation_entry_sha256": reservation["entry_sha256"],
                    "work_item_id": binding.work_item_id,
                    "season_scope": "season1_unranked_development",
                    "season_budget_cap_usd": reservation["reserved_usd"],
                    "season_retained_exposure_usd": reservation["reserved_usd"],
                    "provider_account_scope_sha256": reservation[
                        "provider_account_scope_sha256"
                    ],
                    "provider_account_cap_usd": reservation[
                        "provider_account_cap_usd"
                    ],
                    "full_ceiling_permanently_retained": True,
                    "official": False,
                    "rank_eligible": False,
                },
            )
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        reservation_sha = str(reservation["entry_sha256"])
        scope_adjustment = state.scope_adjustments.get(reservation_sha)
        if (
            scope_adjustment is None
            and _decimal(
                scope_binding["season_retained_exposure_usd"],
                field="bound season retained exposure",
            )
            != state.total_retained_exposure_usd
        ):
            scope_adjustment = append_ledger_event(
                ledger_path,
                {
                    "event_type": "reservation_scope_adjusted",
                    "reservation_entry_sha256": reservation_sha,
                    "work_item_id": binding.work_item_id,
                    "supersedes_scope_binding_entry_sha256": scope_binding[
                        "entry_sha256"
                    ],
                    "season_scope": "season1_unranked_development",
                    "season_budget_cap_usd": _decimal_text(
                        state.total_retained_exposure_usd
                    ),
                    "season_retained_exposure_usd": _decimal_text(
                        state.total_retained_exposure_usd
                    ),
                    "provider_account_scope_sha256": reservation[
                        "provider_account_scope_sha256"
                    ],
                    "provider_account_cap_usd": reservation[
                        "provider_account_cap_usd"
                    ],
                    "full_ceiling_permanently_retained": True,
                    "correction_is_append_only": True,
                    "official": False,
                    "rank_eligible": False,
                },
            )
        effective_scope = scope_adjustment or scope_binding
        template = build_go_template(
            binding=binding,
            reservation=reservation,
            scope_binding=scope_binding,
            effective_scope=effective_scope,
            ledger_path=ledger_path,
            human_pi_identity_record=pi_record,
        )
        template_path = _write_content_addressed(
            output_directory,
            "qwencloud-one-pair-pi-go-template",
            template,
        )
    return reservation, template_path


def verify_go_template(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    template = _verified_artifact(
        path,
        expected_sha256=expected_sha256,
        label="QwenCloud PI-GO template",
    )
    reservation = template.get("reservation")
    model = template.get("model_identity")
    task = template.get("task")
    execution = template.get("execution")
    claims = template.get("claim_boundary")
    if (
        template.get("schema_version") != GO_TEMPLATE_SCHEMA_VERSION
        or template.get("authorization_status")
        != "fresh_exact_human_pi_confirmation_required"
        or not all(
            isinstance(value, Mapping)
            for value in (reservation, model, task, execution, claims)
        )
        or model.get("requested_model_id") != MODEL_ID
        or model.get("identity_label") != MODEL_IDENTITY_LABEL
        or model.get("tool_choice_transport_mode") != TOOL_CHOICE_TRANSPORT_MODE
        or model.get("tool_selection_system_instruction_sha256")
        != sha256_json(QWEN38_TOOL_AUTO_INSTRUCTION)
        or model.get("message_canonicalization") != MESSAGE_CANONICALIZATION
        or model.get("predecessor_failure_artifact_sha256")
        != PREDECESSOR_FAILURE_ARTIFACT_SHA256
        or task.get("task_id") != TASK_ID
        or sha256_text(str(task.get("prompt") or "")) != task.get("prompt_sha256")
        or _decimal(reservation.get("full_ceiling_usd"), field="GO ceiling")
        > MAX_PAIR_CEILING_USD
        or len(str(reservation.get("scope_binding_entry_sha256") or "")) != 64
        or (
            reservation.get("scope_adjustment_entry_sha256") is not None
            and len(str(reservation.get("scope_adjustment_entry_sha256"))) != 64
        )
        or reservation.get("season_scope") != "season1_unranked_development"
        or _decimal(reservation.get("season_budget_cap_usd"), field="season cap")
        < _decimal(
            reservation.get("season_retained_exposure_usd"),
            field="season retained exposure",
        )
        or _decimal(
            reservation.get("season_retained_exposure_usd"),
            field="season retained exposure",
        )
        < _decimal(reservation.get("full_ceiling_usd"), field="GO ceiling")
        or len(str(reservation.get("provider_account_scope_sha256") or "")) != 64
        or reservation.get("full_ceiling_permanently_retained") is not True
        or execution.get("maximum_pairs") != 1
        or execution.get("maximum_response_arms") != 2
        or execution.get("maximum_tool_rounds") > 8
        or execution.get("maximum_provider_attempts_per_request") != 1
        or execution.get("conditions") != list(CONDITIONS)
        or execution.get("allow_mutable_alias_exploratory") is not True
        or execution.get("epicure_transport")
        != {
            "mcp_url": EPICURE_MCP_URL,
            "provenance_url": EPICURE_PROVENANCE_URL,
            "private_host_binding_required": True,
        }
        or not verify_policy_document(execution.get("execution_policy"))
        or execution.get("execution_policy_sha256")
        != execution["execution_policy"]["content_address"]["digest"]
        or claims.get("official") is not False
        or claims.get("rank_eligible") is not False
        or claims.get("leaderboard_comparisons_authorized") != 0
    ):
        raise QwenCloudSmokeAdmissionError("QwenCloud PI-GO template is invalid")
    return template


def verify_preflight_artifact(
    path: Path,
    *,
    expected_sha256: str,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = _verified_artifact(
        path,
        expected_sha256=expected_sha256,
        label="QwenCloud preflight artifact",
    )
    model = template["model_identity"]
    task = template["task"]
    execution = template["execution"]
    reservation = template["reservation"]
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "preflight_passed_no_provider_calls"
        or preflight.get("provider_calls_made") is not False
        or preflight.get("epicure_attestation_performed") is not True
        or preflight.get("official") is not False
        or preflight.get("rank_eligible") is not False
        or preflight.get("model_id") != MODEL_ID
        or preflight.get("canonical_model_slug") != MODEL_ID
        or preflight.get("provider_slug") != PROVIDER_SLUG
        or preflight.get("execution_backend") != EXECUTION_BACKEND
        or preflight.get("candidate_manifest_sha256")
        != model["route_manifest_sha256"]
        or preflight.get("endpoint_execution_sha256")
        != model["endpoint_execution_sha256"]
        or preflight.get("backend_contract_sha256")
        != model["backend_contract_sha256"]
        or preflight.get("execution_policy_sha256")
        != execution["execution_policy_sha256"]
        or preflight.get("dataset_work_item_id") != reservation["work_item_id"]
        or preflight.get("dataset_task_id") != task["task_id"]
        or preflight.get("category") != task["family"]
        or preflight.get("prompt_sha256") != task["prompt_sha256"]
        or preflight.get("conditions") != list(CONDITIONS)
        or _decimal(preflight.get("cap_usd"), field="preflight cap")
        != _decimal(reservation["full_ceiling_usd"], field="reserved ceiling")
        or _decimal(preflight.get("forecast_worst_case_usd"), field="preflight forecast")
        != _decimal(reservation["full_ceiling_usd"], field="reserved ceiling")
        or preflight.get("full_unpriced_budget_ceiling_retained") is not True
        or preflight.get("provider_cost_known") is not False
        or preflight.get("reservation_entry_sha256") != reservation["entry_sha256"]
        or preflight.get("go_template_sha256") != template["artifact_sha256"]
        or preflight.get("model_identity_label") != MODEL_IDENTITY_LABEL
        or preflight.get("epicure_mcp_url") != EPICURE_MCP_URL
        or preflight.get("epicure_provenance_url") != EPICURE_PROVENANCE_URL
    ):
        raise QwenCloudSmokeAdmissionError("QwenCloud preflight differs from the GO template")
    for field in (
        "epicure_release_id",
        "epicure_bundle_sha256",
        "epicure_application_sha256",
        "epicure_tool_schema_sha256",
        "protocol_bundle_sha256",
    ):
        value = str(preflight.get(field) or "")
        if not value or (field != "epicure_release_id" and len(value) != 64):
            raise QwenCloudSmokeAdmissionError(f"preflight lacks exact {field}")
    return preflight


def build_human_pi_authorization(
    *,
    template: Mapping[str, Any],
    preflight: Mapping[str, Any],
    standing_human_pi_record: Mapping[str, Any],
    confirmation: str,
    recorded_at: str,
) -> dict[str, Any]:
    if confirmation != HUMAN_PI_CONFIRMATION:
        raise QwenCloudSmokeAdmissionError("exact Human-PI confirmation is required")
    identity = standing_human_pi_record.get("reviewer_identity")
    try:
        parsed_at = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise QwenCloudSmokeAdmissionError("Human-PI authorization time is invalid") from error
    if (
        parsed_at.tzinfo is None
        or standing_human_pi_record.get("artifact_sha256")
        != template.get("standing_human_pi_identity_record_sha256")
        or not isinstance(identity, Mapping)
        or identity.get("full_name") != "Josef Chen"
        or identity.get("role") != "human_principal_investigator"
    ):
        raise QwenCloudSmokeAdmissionError("Human-PI authorization identity is invalid")
    return {
        "schema_version": GO_AUTHORIZATION_SCHEMA_VERSION,
        "record_role": "fresh_exact_one_pair_human_pi_authorization",
        "decision": "go_for_exactly_one_qwen38_alias_epicure_off_on_pair",
        "authorization_is_transparent_human_pi_record": True,
        "recorded_at": recorded_at,
        "source": "direct_written_authorization_by_named_human_pi",
        "confirmation": confirmation,
        "human_pi": {
            "full_name": identity["full_name"],
            "role": identity["role"],
            "affiliation": identity.get("affiliation", "independent_research"),
        },
        "standing_human_pi_identity_record_sha256": standing_human_pi_record[
            "artifact_sha256"
        ],
        "go_template_sha256": template["artifact_sha256"],
        "reservation_entry_sha256": template["reservation"]["entry_sha256"],
        "work_item_id": template["reservation"]["work_item_id"],
        "preflight_artifact_sha256": preflight["artifact_sha256"],
        "preflight_epicure": {
            "mcp_url": preflight["epicure_mcp_url"],
            "provenance_url": preflight["epicure_provenance_url"],
            "release_id": preflight["epicure_release_id"],
            "bundle_sha256": preflight["epicure_bundle_sha256"],
            "application_sha256": preflight["epicure_application_sha256"],
            "tool_schema_sha256": preflight["epicure_tool_schema_sha256"],
        },
        "scope": {
            "maximum_pairs": 1,
            "maximum_response_arms": 2,
            "conditions": list(CONDITIONS),
            "maximum_tool_rounds": template["execution"]["maximum_tool_rounds"],
            "full_ceiling_usd": template["reservation"]["full_ceiling_usd"],
            "season_scope": template["reservation"]["season_scope"],
            "season_budget_cap_usd": template["reservation"][
                "season_budget_cap_usd"
            ],
            "season_retained_exposure_usd": template["reservation"][
                "season_retained_exposure_usd"
            ],
            "provider_account_scope_sha256": template["reservation"][
                "provider_account_scope_sha256"
            ],
            "full_ceiling_permanently_retained": True,
            "provider_cost_known": False,
            "official": False,
            "season_eligible": False,
            "rank_eligible": False,
            "leaderboard_comparisons_authorized": 0,
        },
        "acknowledgements": {
            "mutable_alias_not_frozen_model": True,
            "catalog_pinned_at_observation_only": True,
            "zero_provider_cost_means_unknown_not_free": True,
            "no_replay_after_execution_started": True,
            "openrouter_is_separate_explicit_stratum_only": True,
            "external_independent_governance_claimed": False,
            "cryptographic_human_signature_present": False,
            "authorization_is_direct_written_pi_record": True,
        },
    }


def verify_human_pi_authorization(
    path: Path,
    *,
    expected_sha256: str,
    template: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _verified_artifact(
        path,
        expected_sha256=expected_sha256,
        label="QwenCloud Human-PI authorization",
    )
    scope = authorization.get("scope")
    acknowledgements = authorization.get("acknowledgements")
    human_pi = authorization.get("human_pi")
    if (
        authorization.get("schema_version") != GO_AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("decision")
        != "go_for_exactly_one_qwen38_alias_epicure_off_on_pair"
        or authorization.get("authorization_is_transparent_human_pi_record") is not True
        or authorization.get("source")
        != "direct_written_authorization_by_named_human_pi"
        or authorization.get("confirmation") != HUMAN_PI_CONFIRMATION
        or authorization.get("go_template_sha256") != template["artifact_sha256"]
        or authorization.get("reservation_entry_sha256")
        != template["reservation"]["entry_sha256"]
        or authorization.get("work_item_id") != template["reservation"]["work_item_id"]
        or authorization.get("preflight_artifact_sha256")
        != preflight["artifact_sha256"]
        or authorization.get("preflight_epicure")
        != {
            "mcp_url": EPICURE_MCP_URL,
            "provenance_url": EPICURE_PROVENANCE_URL,
            "release_id": preflight["epicure_release_id"],
            "bundle_sha256": preflight["epicure_bundle_sha256"],
            "application_sha256": preflight["epicure_application_sha256"],
            "tool_schema_sha256": preflight["epicure_tool_schema_sha256"],
        }
        or human_pi
        != {
            "full_name": "Josef Chen",
            "role": "human_principal_investigator",
            "affiliation": "independent_research",
        }
        or not isinstance(scope, Mapping)
        or scope.get("maximum_pairs") != 1
        or scope.get("maximum_response_arms") != 2
        or scope.get("conditions") != list(CONDITIONS)
        or scope.get("full_ceiling_permanently_retained") is not True
        or scope.get("season_scope") != "season1_unranked_development"
        or scope.get("season_budget_cap_usd")
        != template["reservation"]["season_budget_cap_usd"]
        or scope.get("season_retained_exposure_usd")
        != template["reservation"]["season_retained_exposure_usd"]
        or _decimal(scope.get("season_budget_cap_usd"), field="authorized season cap")
        < _decimal(
            scope.get("season_retained_exposure_usd"),
            field="authorized season exposure",
        )
        or scope.get("provider_cost_known") is not False
        or scope.get("official") is not False
        or scope.get("rank_eligible") is not False
        or scope.get("leaderboard_comparisons_authorized") != 0
        or not isinstance(acknowledgements, Mapping)
        or acknowledgements.get("mutable_alias_not_frozen_model") is not True
        or acknowledgements.get("zero_provider_cost_means_unknown_not_free") is not True
        or acknowledgements.get("no_replay_after_execution_started") is not True
        or acknowledgements.get("external_independent_governance_claimed") is not False
        or acknowledgements.get("cryptographic_human_signature_present") is not False
        or acknowledgements.get("authorization_is_direct_written_pi_record") is not True
    ):
        raise QwenCloudSmokeAdmissionError("QwenCloud Human-PI authorization is invalid")
    return authorization


def begin_execution(
    *,
    ledger_path: Path,
    template: Mapping[str, Any],
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
    confirmation: str,
) -> Mapping[str, Any]:
    if confirmation != LIVE_CONFIRMATION:
        raise QwenCloudSmokeAdmissionError("exact one-pair live confirmation is required")
    reservation_sha = str(template["reservation"]["entry_sha256"])
    work_item_id = str(template["reservation"]["work_item_id"])
    with ledger_lock(ledger_path):
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        reservation = state.reservations.get(reservation_sha)
        scope_binding = state.scope_bindings.get(reservation_sha)
        scope_adjustment = state.scope_adjustments.get(reservation_sha)
        expected_adjustment_sha = template["reservation"].get(
            "scope_adjustment_entry_sha256"
        )
        effective_scope = scope_adjustment or scope_binding
        if (
            reservation is None
            or scope_binding is None
            or effective_scope is None
            or reservation.get("work_item_id") != work_item_id
            or scope_binding.get("entry_sha256")
            != template["reservation"]["scope_binding_entry_sha256"]
            or (
                str(scope_adjustment.get("entry_sha256"))
                if scope_adjustment is not None
                else None
            )
            != expected_adjustment_sha
            or effective_scope.get("season_retained_exposure_usd")
            != template["reservation"]["season_retained_exposure_usd"]
            or effective_scope.get("season_budget_cap_usd")
            != template["reservation"]["season_budget_cap_usd"]
            or reservation.get("route_manifest_sha256")
            != template["model_identity"]["route_manifest_sha256"]
            or reservation.get("prompt_sha256") != template["task"]["prompt_sha256"]
            or reservation.get("execution_policy_sha256")
            != template["execution"]["execution_policy_sha256"]
        ):
            raise QwenCloudSmokeAdmissionError("GO template has no exact ledger reservation")
        if reservation_sha in state.starts or reservation_sha in state.terminalizations:
            raise QwenCloudSmokeAdmissionError(
                "QwenCloud work item already started; replay is permanently prohibited"
            )
        return append_ledger_event(
            ledger_path,
            {
                "event_type": "execution_started",
                "reservation_entry_sha256": reservation_sha,
                "work_item_id": work_item_id,
                "frozen_run_id": template["execution"]["frozen_run_id"],
                "go_template_sha256": template["artifact_sha256"],
                "human_pi_authorization_sha256": authorization["artifact_sha256"],
                "preflight_artifact_sha256": preflight["artifact_sha256"],
                "attempt_slot_plan_sha256": template["execution"][
                    "attempt_slot_plan_sha256"
                ],
                "maximum_provider_attempts_per_request": 1,
                "provider_delivery_may_begin_after_this_fsync": True,
                "safe_to_replay_after_this_event": False,
                "live_confirmation_sha256": hashlib.sha256(
                    confirmation.encode("utf-8")
                ).hexdigest(),
                "retained_exposure_usd": reservation["reserved_usd"],
            },
        )


def terminalize_source(
    *,
    ledger_path: Path,
    template: Mapping[str, Any],
    authorization: Mapping[str, Any],
    artifact_path: Path,
    recovery_artifact_path: Path | None = None,
    expected_recovery_artifact_sha256: str = "",
) -> Mapping[str, Any]:
    artifact = _regular_json(artifact_path, label="QwenCloud live source artifact")
    digest = str(artifact.get("artifact_sha256") or "")
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    budget = artifact.get("budget")
    if (
        len(digest) != 64
        or _live_source_sha256(body) != digest
        or artifact.get("schema_version") != "flavourbench-live-smoke-v1"
        or artifact.get("status") != "complete_unpriced_budget_ceiling"
        or artifact.get("requested_model_id") != MODEL_ID
        or artifact.get("requested_provider") != PROVIDER_SLUG
        or artifact.get("execution_backend") != EXECUTION_BACKEND
        or artifact.get("candidate_manifest_sha256")
        != template["model_identity"]["route_manifest_sha256"]
        or artifact.get("dataset_work_item_id")
        != template["reservation"]["work_item_id"]
        or artifact.get("dataset_task_id") != template["task"]["task_id"]
        or artifact.get("prompt_sha256") != template["task"]["prompt_sha256"]
        or artifact.get("execution_policy_sha256")
        != template["execution"]["execution_policy_sha256"]
        or artifact.get("mutable_alias_exploratory_opt_in") is not True
        or artifact.get("official") is not False
        or artifact.get("rank_eligible") is not False
        or artifact.get("requested_conditions") != list(CONDITIONS)
        or not isinstance(budget, Mapping)
        or _decimal(budget.get("cap_usd"), field="source cap")
        != _decimal(template["reservation"]["full_ceiling_usd"], field="GO cap")
        or budget.get("provider_cost_known") is not False
        or budget.get("full_unpriced_budget_ceiling_retained") is not True
        or _decimal(budget.get("retained_exposure_usd"), field="source exposure")
        != _decimal(template["reservation"]["full_ceiling_usd"], field="GO cap")
        or budget.get("zero_recorded_cost_means") != "unknown_not_free"
    ):
        raise QwenCloudSmokeAdmissionError(
            "QwenCloud source does not satisfy the full-ceiling terminal contract"
        )
    reservation_sha = str(template["reservation"]["entry_sha256"])
    with ledger_lock(ledger_path):
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        started = state.starts.get(reservation_sha)
        existing = state.terminalizations.get(reservation_sha)
        if existing is not None:
            if existing.get("source_artifact_sha256") != digest:
                raise QwenCloudSmokeAdmissionError(
                    "reservation is terminalized by a different source"
                )
            return existing
        if (
            started is None
            or started.get("human_pi_authorization_sha256")
            != authorization["artifact_sha256"]
        ):
            raise QwenCloudSmokeAdmissionError("source has no exact execution-start event")
        reservation = state.reservations[reservation_sha]
        incident = state.incidents.get(reservation_sha)
        recovery: Mapping[str, Any] | None = None
        if incident is not None:
            if recovery_artifact_path is None or not expected_recovery_artifact_sha256:
                raise QwenCloudSmokeAdmissionError(
                    "post-incident source terminalization requires an exact recovery artifact"
                )
            recovery = _verified_artifact(
                recovery_artifact_path,
                expected_sha256=expected_recovery_artifact_sha256,
                label="QwenCloud zero-call source recovery",
            )
            if (
                recovery.get("schema_version")
                != "flavourbench-qwencloud-zero-call-source-recovery-v1"
                or recovery.get("source_artifact_sha256") != digest
                or recovery.get("incident_entry_sha256")
                != incident.get("entry_sha256")
                or recovery.get("go_template_sha256")
                != template.get("artifact_sha256")
                or recovery.get("human_pi_authorization_sha256")
                != authorization.get("artifact_sha256")
                or recovery.get("provider_calls_made") is not False
                or recovery.get("epicure_calls_made") is not False
                or recovery.get("source_mutated") is not False
                or recovery.get("recovery_decision")
                != "terminalize_complete_source_after_local_digest_validator_fix"
            ):
                raise QwenCloudSmokeAdmissionError(
                    "QwenCloud zero-call recovery does not bind the incident source"
                )
        return append_ledger_event(
            ledger_path,
            {
                "event_type": "source_terminalized",
                "reservation_entry_sha256": reservation_sha,
                "execution_start_entry_sha256": started["entry_sha256"],
                "work_item_id": reservation["work_item_id"],
                "source_artifact_sha256": digest,
                "source_artifact_filename": artifact_path.name,
                "run_id": artifact["run_id"],
                "retained_exposure_usd": reservation["reserved_usd"],
                "provider_reported_cost_micros": artifact["budget"][
                    "actual_cost_micros"
                ],
                "provider_cost_known": False,
                "zero_recorded_cost_means": "unknown_not_free",
                "cost_accounting_basis": (
                    "full_unpriced_budget_ceiling_permanently_retained"
                ),
                "official": False,
                "rank_eligible": False,
                "safe_to_replay": False,
                "post_incident_local_reconciliation": incident is not None,
                "superseded_incident_entry_sha256": (
                    incident.get("entry_sha256") if incident is not None else None
                ),
                "zero_call_recovery_artifact_sha256": (
                    recovery.get("artifact_sha256") if recovery is not None else None
                ),
            },
        )


def record_execution_incident(
    *,
    ledger_path: Path,
    template: Mapping[str, Any],
    error: Exception,
) -> Mapping[str, Any]:
    reservation_sha = str(template["reservation"]["entry_sha256"])
    with ledger_lock(ledger_path):
        entries = load_ledger(ledger_path)
        state = validate_ledger_state(entries)
        if reservation_sha in state.incidents:
            return state.incidents[reservation_sha]
        reservation = state.reservations.get(reservation_sha)
        started = state.starts.get(reservation_sha)
        if reservation is None or started is None:
            raise QwenCloudSmokeAdmissionError(
                "cannot record an incident before execution is durably started"
            )
        return append_ledger_event(
            ledger_path,
            {
                "event_type": "execution_incident",
                "reservation_entry_sha256": reservation_sha,
                "execution_start_entry_sha256": started["entry_sha256"],
                "work_item_id": reservation["work_item_id"],
                "incident": "live_execution_failed_or_delivery_uncertain",
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                "retained_exposure_usd": reservation["reserved_usd"],
                "provider_cost_known": False,
                "safe_to_replay": False,
            },
        )


def _freeze_cli(args: argparse.Namespace) -> dict[str, Any]:
    cap = _decimal(args.cap_usd, field="cap_usd")
    binding = build_smoke_binding(
        route_manifest_path=args.route_manifest,
        expected_route_manifest_sha256=args.expected_route_manifest_sha256,
        task_validity_path=args.task_validity,
        expected_task_validity_sha256=args.expected_task_validity_sha256,
        cap_usd=cap,
    )
    reservation, template_path = reserve_and_write_template(
        binding=binding,
        ledger_path=args.ledger,
        output_directory=args.output_dir,
        human_pi_identity_record_path=args.human_pi_identity_record,
        expected_human_pi_identity_record_sha256=(
            args.expected_human_pi_identity_record_sha256
        ),
        confirmation=args.confirm,
    )
    template = _regular_json(template_path, label="QwenCloud PI-GO template")
    return {
        "status": "full_ceiling_reserved_go_template_frozen_no_external_calls",
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "ledger": str(args.ledger.resolve()),
        "reservation_entry_sha256": reservation["entry_sha256"],
        "work_item_id": binding.work_item_id,
        "frozen_run_id": binding.frozen_run_id,
        "execution_policy_sha256": binding.execution_policy.sha256,
        "endpoint_execution_sha256": binding.candidate.endpoint_execution_sha256,
        "template": str(template_path.resolve()),
        "template_sha256": template["artifact_sha256"],
        "retained_exposure_usd": _decimal_text(binding.cap_usd),
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--expected-route-manifest-sha256", required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--expected-task-validity-sha256", required=True)
    parser.add_argument("--human-pi-identity-record", type=Path, required=True)
    parser.add_argument("--expected-human-pi-identity-record-sha256", required=True)
    parser.add_argument("--cap-usd", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    try:
        result = _freeze_cli(args)
    except (OSError, QwenCloudSmokeAdmissionError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
