"""Governed real-data collection for the unranked FlavourBench exploration.

This module is intentionally separate from both the public arena worker and the
frontier contract-smoke runner.  It deterministically chooses candidate tasks,
executes one exact-endpoint model/task pair at a time, and normalises each
reconciled Epicure-off/on response into an immutable artifact.  The default is
a plan that makes no provider calls.

Nothing produced here is official, rank eligible, or an approved research
release.  Human judgments and a frozen governed season remain separate gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from .development_task_validity import SCHEMA_VERSION as DEVELOPMENT_TASK_SCHEMA_VERSION
from .development_task_validity_v2 import (
    SCHEMA_VERSION as DEVELOPMENT_TASK_V2_SCHEMA_VERSION,
)
from .epicure_native_taskset import (
    SCHEMA_VERSION as EPICURE_NATIVE_TASK_SCHEMA_VERSION,
)
from .epicure_native_taskset import verify_taskset as verify_epicure_native_taskset
from .execution_policy import (
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOLS,
    PORTABLE_TEXT_TOOL_PROTOCOL_V1,
    ExecutionPolicy,
    assert_legacy_paid_cli_allowed,
    verify_policy_document,
)
from .frontier_contract_runner import (
    AUTHORIZED_TOTAL_CAP_USD,
    DEFAULT_ADMISSION_FRACTION,
    QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS,
    QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS,
    RATE_CARD_ACCOUNTING_BASIS_BY_BACKEND,
    AdmissionDenied,
    ArtifactExposure,
    ContractCandidate,
    IntegrityError,
    PriceEnvelope,
    _decimal,
    _decimal_text,
    _effective_rate,
    _exclusive_runner_lock,
    _extract_artifact_path,
    _safe_process_hash,
    _sha256,
    _verify_live_artifact,
    active_ledger_reservations,
    load_candidate_manifest,
    scan_live_smoke_artifacts,
    select_candidates,
    validate_ledger_artifact_links,
)
from .frontier_contract_runner import (
    load_ledger as load_frontier_ledger,
)
from .live_smoke import (
    CONFIRMATION as LIVE_SMOKE_CONFIRMATION,
)
from .live_smoke import (
    endpoint_execution_contract,
    endpoint_execution_contract_sha256,
)
from .matched_protocol_preflight import (
    ProtocolPreflightError,
    verify_registry_for_manifest,
)
from .run_journal import JournalIntegrityError, scan_recovery_journals
from .tasks import CandidateTask, candidate_tasks
from .tool_contract import required_tool_contract

CURRENT_DATASET_MANIFEST_SHA256 = "aaf43f1bd770df5f120d79b66058cfad5092d5fb950e80bd24fac6d1d2e9acb5"
KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD = Decimal("1.321528")
EXECUTION_CONFIRMATION = "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET"
FINALIZE_EXISTING_CONFIRMATION = "FINALIZE_RECONCILED_DATASET_SOURCES"
FINALIZE_UNRESOLVED_CONFIRMATION = "FINALIZE_UNRESOLVED_SOURCE_WITH_FULL_ALLOWANCE"
SELECTION_SCHEMA_VERSION = "flavourbench-balanced-candidate-selection-v1"
WORK_ITEM_SCHEMA_VERSION = "flavourbench-real-exploratory-work-item-v1"
RESPONSE_SCHEMA_VERSION = "flavourbench-real-exploratory-response-v1"
LEDGER_SCHEMA_VERSION = "flavourbench-real-exploratory-ledger-v1"
SUMMARY_SCHEMA_VERSION = "flavourbench-real-exploratory-summary-v1"
V2_ROUTE_PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-validation-plan-v1"
V3_ROUTE_PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-v3-route-validation-plan-v1"
RUNNER_SCHEMA_VERSION = "flavourbench-real-exploratory-runner-v1"
SOURCE_INCIDENT_RESOLUTION_SCHEMA_VERSION = (
    "flavourbench-real-exploratory-source-incident-resolution-v1"
)
SOURCE_INCIDENT_RESOLUTION_RECORD_TYPE = "conservative_http_200_no_choice_without_generation_id"
SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE = "source_incident_resolution_recorded"
SOURCE_INCIDENT_RESOLUTION_CONFIRMATION = "RESOLVE_NO_ID_HTTP_200_NO_CHOICE_WITH_FULL_ALLOWANCE"
RESOLVED_CONSERVATIVE_EXPOSURE_BASIS = (
    "resolved_no_generation_id_conservative_full_admitted_allowance"
)
DEFAULT_SELECTION_SEED = "flavourbench-real-exploratory-v1"
TASK_FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONDITIONS = ("epicure_off", "epicure_on")
OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER = Decimal("1.20")


class DatasetRunnerError(RuntimeError):
    """Base error for the exploratory real-data runner."""


@dataclass(frozen=True)
class PairForecast:
    forecast_usd: Decimal
    price_envelope: PriceEnvelope
    prompt_tokens_per_request_bound: Decimal
    completion_tokens_per_request_bound: int
    intermediate_completion_tokens_bound: int
    total_completion_tokens_bound: int
    request_bound: int
    conditions: tuple[str, ...] = CONDITIONS

    def public_payload(self) -> dict[str, Any]:
        return {
            "forecast_usd": _decimal_text(self.forecast_usd),
            "price_envelope": self.price_envelope.public_payload(),
            "prompt_tokens_per_request_bound": _decimal_text(self.prompt_tokens_per_request_bound),
            "completion_tokens_per_request_bound": self.completion_tokens_per_request_bound,
            "intermediate_completion_tokens_bound": (self.intermediate_completion_tokens_bound),
            "total_completion_tokens_bound": self.total_completion_tokens_bound,
            "request_bound": self.request_bound,
            "conditions": list(self.conditions),
            "forecast_basis": (
                "matched planning/tool-selection turns plus one final answer per selected "
                "condition, using frozen "
                "endpoint pricing and bounded prompt, catalog, evidence, and completion caps"
            ),
        }


@dataclass(frozen=True)
class WorkItem:
    ordinal: int
    work_item_id: str
    manifest_sha256: str
    task_registry_sha256: str
    task: CandidateTask
    candidate: ContractCandidate
    endpoint_execution_sha256: str
    execution_policy_sha256: str
    execution_policy: ExecutionPolicy

    def public_payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "work_item_id": self.work_item_id,
            "task_id": self.task.public_id,
            "task_family": self.task.family,
            "prompt_sha256": self.task.prompt_sha256,
            "slot_id": self.candidate.slot_id,
            "model_id": self.candidate.model_id,
            "canonical_model_slug": self.candidate.canonical_model_slug,
            "provider_tag": self.candidate.provider_tag,
            "execution_backend": self.candidate.execution_backend,
            "backend_contract_sha256": self.candidate.backend_contract_sha256,
            "route_selection_reason": self.candidate.route_selection.get("selection_reason"),
            "endpoint_execution_sha256": self.endpoint_execution_sha256,
            "execution_policy_sha256": self.execution_policy_sha256,
        }


@dataclass(frozen=True)
class DatasetSource:
    path: Path
    artifact_sha256: str
    work_item_id: str
    artifact: Mapping[str, Any]
    exposure: ArtifactExposure


@dataclass(frozen=True)
class ResponseArtifact:
    path: Path
    artifact_sha256: str
    work_item_id: str
    condition: str
    task_id: str
    task_family: str
    model_id: str
    provider_tag: str
    source_artifact_sha256: str
    actual_cost_usd: Decimal
    tool_used: bool


@dataclass(frozen=True)
class DatasetState:
    prior_verified_exposure_usd: Decimal
    prior_effective_exposure_usd: Decimal
    prior_active_reservation_usd: Decimal
    dataset_actual_cost_usd: Decimal
    dataset_source_exposure_usd: Decimal
    unresolved_dataset_source_reserve_usd: Decimal
    sources: Mapping[str, DatasetSource]
    responses: Mapping[tuple[str, str], ResponseArtifact]
    ledger: tuple[Mapping[str, Any], ...]
    reservations: Mapping[str, Mapping[str, Any]]
    finalizations: Mapping[str, Mapping[str, Any]]
    orphan_reservation_usd: Decimal
    incident_resolutions: Mapping[str, SourceIncidentResolution] = dataclasses.field(
        default_factory=dict
    )

    @property
    def total_exposure_usd(self) -> Decimal:
        return (
            self.prior_effective_exposure_usd
            + self.prior_active_reservation_usd
            + self.dataset_source_exposure_usd
            + self.orphan_reservation_usd
        )


@dataclass(frozen=True)
class SourceIncidentResolution:
    path: Path
    artifact_sha256: str
    work_item_id: str
    source_artifact_sha256: str
    reservation_entry_sha256: str
    incident_entry_sha256: str
    affected_condition: str
    normalizable_conditions: tuple[str, ...]
    provider_reconciled_actual_cost_usd: Decimal
    conservative_budget_exposure_usd: Decimal
    ledger_event_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} must be a regular, non-symlink file: {path}")


def task_registry_snapshot(tasks: Sequence[CandidateTask] | None = None) -> dict[str, Any]:
    inventory = list(tasks if tasks is not None else candidate_tasks())
    rows = [
        {
            "public_id": task.public_id,
            "family": task.family,
            "prompt": task.prompt,
            "prompt_sha256": task.prompt_sha256,
            "split": task.split,
            "review_status": task.review_status,
        }
        for task in sorted(inventory, key=lambda item: item.public_id)
    ]
    if len(rows) != len({row["public_id"] for row in rows}):
        raise IntegrityError("candidate task registry contains duplicate public IDs")
    if len(rows) != len({row["prompt_sha256"] for row in rows}):
        raise IntegrityError("candidate task registry contains duplicate prompts")
    return {
        "schema_version": "flavourbench-candidate-task-registry-v1",
        "task_count": len(rows),
        "tasks": rows,
    }


def task_registry_sha256(tasks: Sequence[CandidateTask] | None = None) -> str:
    return _sha256(task_registry_snapshot(tasks))


def load_development_task_inventory(
    path: str | Path,
) -> tuple[list[CandidateTask], dict[str, Any]]:
    """Load only the source-verified, explicitly development-only task dossier."""

    source_path = Path(path)
    _require_regular_file(source_path, label="development task-validity artifact")
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError("development task-validity artifact is invalid JSON") from error
    if not isinstance(document, Mapping):
        raise IntegrityError("development task-validity artifact is not an object")
    recorded = str(document.get("artifact_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if recorded != _sha256(unhashed) or recorded not in source_path.name:
        raise IntegrityError("development task-validity content address does not verify")
    boundary = document.get("claim_boundary")
    counts = document.get("counts")
    records = document.get("tasks")
    if (
        document.get("schema_version")
        not in {DEVELOPMENT_TASK_SCHEMA_VERSION, DEVELOPMENT_TASK_V2_SCHEMA_VERSION}
        or document.get("status")
        not in {
            "source_verified_development_candidate_not_confirmatory",
            "surface_clean_source_verified_development_candidate_not_confirmatory",
        }
        or not isinstance(boundary, Mapping)
        or boundary.get("supports_real_current_model_development_runs") is not True
        or not (
            boundary.get("official") is False
            or (
                document.get("schema_version") == DEVELOPMENT_TASK_V2_SCHEMA_VERSION
                and boundary.get("supports_official_leaderboard") is False
            )
        )
        or boundary.get("rank_eligible") is not False
        or not isinstance(counts, Mapping)
        or counts.get("synthetic_tasks") != 0
        or not isinstance(records, list)
        or len(records) != counts.get("selected_development_tasks")
    ):
        raise IntegrityError("development task-validity artifact violates its claim boundary")
    inventory: list[CandidateTask] = []
    task_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise IntegrityError("development task inventory contains a non-object record")
        task_id = str(record.get("task_id") or "")
        family = str(record.get("family") or "")
        prompt = str(record.get("prompt") or "")
        prompt_sha256 = str(record.get("prompt_sha256") or "")
        if (
            family not in TASK_FAMILIES
            or not task_id
            or record.get("confirmatory_eligible") is not False
            or record.get("rank_eligible") is not False
            or record.get("task_specific_criterion_status") != "pending_independent_human_authoring"
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256
            or task_id in task_ids
            or prompt_sha256 in prompt_hashes
            or (
                document.get("schema_version") == DEVELOPMENT_TASK_V2_SCHEMA_VERSION
                and (
                    not isinstance(record.get("surface_dependency_screen"), Mapping)
                    or record["surface_dependency_screen"].get("status") != "pass"
                    or record["surface_dependency_screen"].get("failure_reasons") != []
                )
            )
        ):
            raise IntegrityError("development task record is malformed or duplicated")
        task_ids.add(task_id)
        prompt_hashes.add(prompt_sha256)
        inventory.append(
            CandidateTask(
                public_id=task_id,
                family=family,
                prompt=prompt,
                split="pilot",
                review_status="candidate",
            )
        )
    return inventory, {
        "artifact_path": str(source_path),
        "artifact_sha256": recorded,
        "source_task_bank_sha256": document.get("source_task_bank_sha256"),
        "candidate_coordinate_sha256": document.get("candidate_coordinate_sha256"),
        "source_class": "licensed_real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "confirmatory_eligible": False,
        "rank_eligible": False,
    }


def load_epicure_native_task_inventory(
    path: str | Path,
) -> tuple[list[CandidateTask], dict[str, Any]]:
    """Load the content-addressed deterministic Epicure-native task set."""

    source_path = Path(path)
    _require_regular_file(source_path, label="Epicure-native task artifact")
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError("Epicure-native task artifact is invalid JSON") from error
    if not isinstance(document, Mapping):
        raise IntegrityError("Epicure-native task artifact is not an object")
    recorded = str(document.get("artifact_sha256") or "")
    counts = document.get("counts")
    records = document.get("tasks")
    if (
        document.get("schema_version") != EPICURE_NATIVE_TASK_SCHEMA_VERSION
        or not verify_epicure_native_taskset(document)
        or recorded not in source_path.name
        or document.get("track") != "epicure_native_exact"
        or not isinstance(counts, Mapping)
        or counts.get("tasks") != 32
        or counts.get("human_judgments_required") != 0
        or not isinstance(records, list)
        or len(records) != 32
    ):
        raise IntegrityError("Epicure-native task artifact does not verify")

    inventory: list[CandidateTask] = []
    task_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise IntegrityError("Epicure-native task inventory contains a non-object")
        task_id = str(record.get("task_id") or "")
        family = str(record.get("family") or "")
        prompt = str(record.get("prompt") or "")
        prompt_sha256 = str(record.get("prompt_sha256") or "")
        choices = record.get("choices")
        scoring = record.get("scoring")
        if (
            family not in TASK_FAMILIES
            or not task_id
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256
            or task_id in task_ids
            or prompt_sha256 in prompt_hashes
            or not isinstance(choices, Mapping)
            or set(choices) != {"A", "B", "C", "D"}
            or record.get("expected_choice") not in choices
            or not isinstance(scoring, Mapping)
            or scoring.get("method") != "exact_final_choice_marker_v1"
        ):
            raise IntegrityError("Epicure-native task record is malformed or duplicated")
        task_ids.add(task_id)
        prompt_hashes.add(prompt_sha256)
        inventory.append(
            CandidateTask(
                public_id=task_id,
                family=family,
                prompt=prompt,
                split="pilot",
                review_status="candidate",
            )
        )
    family_counts = Counter(task.family for task in inventory)
    if family_counts != Counter({family: 8 for family in TASK_FAMILIES}):
        raise IntegrityError("Epicure-native task inventory is not balanced")
    return inventory, {
        "artifact_path": str(source_path),
        "artifact_sha256": recorded,
        "task_set_sha256": document.get("task_set_sha256"),
        "epicure_provenance": document.get("epicure_provenance"),
        "source_class": document.get("source_class"),
        "programmatically_generated_tasks": 32,
        "human_judgments_required": 0,
        "automated_ground_truth": True,
        "confirmatory_eligible": True,
        "rank_eligible": True,
    }


def select_balanced_tasks(
    *,
    tasks_per_family: int = 1,
    seed: str = DEFAULT_SELECTION_SEED,
    tasks: Sequence[CandidateTask] | None = None,
) -> tuple[list[CandidateTask], str]:
    """Choose a nested, deterministic equal-size subset of all four families."""

    if not isinstance(tasks_per_family, int) or isinstance(tasks_per_family, bool):
        raise ValueError("tasks_per_family must be an integer")
    if tasks_per_family < 1 or tasks_per_family > 30:
        raise ValueError("tasks_per_family must be between 1 and 30")
    if not seed:
        raise ValueError("selection seed cannot be empty")
    inventory = list(tasks if tasks is not None else candidate_tasks())
    registry_sha = task_registry_sha256(inventory)
    by_family: dict[str, list[CandidateTask]] = defaultdict(list)
    for task in inventory:
        if task.family not in TASK_FAMILIES:
            raise IntegrityError(f"unsupported candidate task family: {task.family}")
        if task.review_status != "candidate" or task.split != "pilot":
            raise IntegrityError(
                f"exploratory selection requires candidate/pilot tasks: {task.public_id}"
            )
        by_family[task.family].append(task)
    selected: list[CandidateTask] = []
    for family in TASK_FAMILIES:
        family_tasks = by_family.get(family, [])
        if len(family_tasks) < tasks_per_family:
            raise IntegrityError(
                f"family {family} contains only {len(family_tasks)} candidate tasks"
            )

        def selection_key(task: CandidateTask, *, selected_family: str = family) -> tuple[str, str]:
            payload = (
                f"{SELECTION_SCHEMA_VERSION}\0{seed}\0{selected_family}\0"
                f"{task.public_id}\0{task.prompt_sha256}"
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest(), task.public_id

        selected.extend(sorted(family_tasks, key=selection_key)[:tasks_per_family])
    counts = Counter(task.family for task in selected)
    if set(counts) != set(TASK_FAMILIES) or set(counts.values()) != {tasks_per_family}:
        raise IntegrityError("balanced task selection invariant failed")
    return selected, registry_sha


def _execution_endpoint_sha256(candidate: ContractCandidate) -> str:
    return endpoint_execution_contract_sha256(dict(candidate.endpoint))


def _work_item_digest(
    *,
    manifest_sha256: str,
    task_registry_digest: str,
    task: CandidateTask,
    candidate: ContractCandidate,
    execution_policy_sha256: str,
) -> str:
    return _sha256(
        {
            "schema_version": WORK_ITEM_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "task_registry_sha256": task_registry_digest,
            "task": {
                "public_id": task.public_id,
                "family": task.family,
                "prompt_sha256": task.prompt_sha256,
                "split": task.split,
                "review_status": task.review_status,
            },
            "model": {
                "model_id": candidate.model_id,
                "canonical_model_slug": candidate.canonical_model_slug,
                "provider_tag": candidate.provider_tag,
                "execution_backend": candidate.execution_backend,
                "backend_contract_sha256": candidate.backend_contract_sha256,
                "route_selection": dict(candidate.route_selection),
                "endpoint_execution_sha256": _execution_endpoint_sha256(candidate),
            },
            "conditions": list(CONDITIONS),
            "execution_policy_sha256": execution_policy_sha256,
        }
    )


def build_balanced_work_items(
    *,
    manifest_sha256: str,
    task_registry_digest: str,
    selected_tasks: Sequence[CandidateTask],
    candidates: Sequence[ContractCandidate],
    execution_policy: ExecutionPolicy,
    assignments_per_model: int = 10,
) -> list[WorkItem]:
    """Assign and interleave a near-equal family mix for every model.

    Ten assignments use a 2/2/3/3 mix.  The two extra families rotate by model,
    which gives exactly 30 assignments per family for the complete 12-model
    panel while still placing every model in each ten-item scheduling tranche.
    """

    if not candidates:
        raise IntegrityError("real-data workload contains no selected models")
    tasks_by_family: dict[str, list[CandidateTask]] = defaultdict(list)
    for task in selected_tasks:
        tasks_by_family[task.family].append(task)
    family_sizes = {family: len(tasks_by_family.get(family, [])) for family in TASK_FAMILIES}
    if not family_sizes or len(set(family_sizes.values())) != 1 or 0 in family_sizes.values():
        raise IntegrityError("real-data workload tasks are not balanced across families")
    task_pool_per_family = next(iter(family_sizes.values()))
    if (
        not isinstance(assignments_per_model, int)
        or isinstance(assignments_per_model, bool)
        or assignments_per_model < len(TASK_FAMILIES)
        or assignments_per_model > task_pool_per_family * len(TASK_FAMILIES)
    ):
        raise ValueError(
            "assignments_per_model must be between four and the selected task-pool size"
        )
    base_count, extra_count = divmod(assignments_per_model, len(TASK_FAMILIES))
    if base_count + int(extra_count > 0) > task_pool_per_family:
        raise IntegrityError("selected task pool is too small for the per-model assignment target")
    assignments_by_model: list[list[CandidateTask]] = []
    for model_index, _candidate in enumerate(candidates):
        counts = {family: base_count for family in TASK_FAMILIES}
        for offset in range(extra_count):
            family = TASK_FAMILIES[(model_index + offset) % len(TASK_FAMILIES)]
            counts[family] += 1
        model_tasks: list[CandidateTask] = []
        used_by_family = Counter()
        # Rotate the family traversal per model.  This keeps every prefix broad
        # while preserving the deterministic family-count allocation above.
        for round_index in range(task_pool_per_family):
            for shift in range(len(TASK_FAMILIES)):
                family = TASK_FAMILIES[(model_index + shift) % len(TASK_FAMILIES)]
                if counts[family] <= round_index:
                    continue
                model_tasks.append(tasks_by_family[family][round_index])
                used_by_family[family] += 1
        if len(model_tasks) != assignments_per_model or used_by_family != Counter(counts):
            raise IntegrityError("per-model balanced task assignment invariant failed")
        assignments_by_model.append(model_tasks)
    ordered: list[tuple[CandidateTask, ContractCandidate]] = []
    for position in range(assignments_per_model):
        for model_index, candidate in enumerate(candidates):
            ordered.append((assignments_by_model[model_index][position], candidate))
    work_items = [
        WorkItem(
            ordinal=index,
            work_item_id=_work_item_digest(
                manifest_sha256=manifest_sha256,
                task_registry_digest=task_registry_digest,
                task=task,
                candidate=candidate,
                execution_policy_sha256=execution_policy.sha256,
            ),
            manifest_sha256=manifest_sha256,
            task_registry_sha256=task_registry_digest,
            task=task,
            candidate=candidate,
            endpoint_execution_sha256=_execution_endpoint_sha256(candidate),
            execution_policy_sha256=execution_policy.sha256,
            execution_policy=execution_policy,
        )
        for index, (task, candidate) in enumerate(ordered, start=1)
    ]
    expected = assignments_per_model * len(candidates)
    if len(work_items) != expected or len({item.work_item_id for item in work_items}) != expected:
        raise IntegrityError("balanced work-item schedule is incomplete or duplicated")
    return work_items


def validate_current_development_run_binding(
    *,
    manifest: Mapping[str, Any],
    task_source: Mapping[str, Any],
    task_inventory: Sequence[CandidateTask],
    selected_tasks: Sequence[CandidateTask],
    candidates: Sequence[ContractCandidate],
    work_items: Sequence[WorkItem],
    task_pool_per_family: int,
    assignments_per_model: int,
    selection_seed: str,
    execution_policy: ExecutionPolicy,
) -> None:
    """Fail if a current-quality manifest is invoked with a different workload."""

    accepted_roles = {
        "current_frontier_real_development_quality_run",
        "current_frontier_routed_development_quality_run",
    }
    if manifest.get("manifest_role") not in accepted_roles:
        return
    source = manifest.get("source")
    design = manifest.get("run_design")
    governance = manifest.get("governance")
    if not all(isinstance(value, Mapping) for value in (source, design, governance)):
        raise IntegrityError("current development manifest lacks frozen run metadata")
    assert isinstance(source, Mapping)
    assert isinstance(design, Mapping)
    assert isinstance(governance, Mapping)
    expected = {
        "tasks_per_family_in_pool": task_pool_per_family,
        "selection_seed": selection_seed,
        "selected_task_count": len(selected_tasks),
        "assignments_per_model": assignments_per_model,
        "expected_pairs": len(work_items),
        "expected_arms": len(work_items) * len(CONDITIONS),
        "execution_policy_sha256": execution_policy.sha256,
    }
    if any(design.get(field) != value for field, value in expected.items()):
        raise IntegrityError("runtime workload differs from the current development manifest")
    if (
        design.get("conditions") != list(CONDITIONS)
        or (design.get("generation_protocol") or {}).get("schema_version")
        not in {
            "flavourbench-live-development-protocol-v1",
            "flavourbench-live-development-protocol-v2",
            "flavourbench-live-development-protocol-v3",
            "flavourbench-live-development-protocol-v4",
            "flavourbench-live-development-protocol-v5",
            "flavourbench-live-development-protocol-v6",
            "flavourbench-live-development-protocol-v7",
            "flavourbench-live-development-protocol-v8",
            "flavourbench-live-development-protocol-v9",
            "flavourbench-live-development-protocol-v10",
        }
        or (design.get("generation_protocol") or {}).get("full_epicure_catalog_required")
        is not True
        or (design.get("generation_protocol") or {}).get("final_response_mode", "structured_json")
        != execution_policy.final_response_mode
        or (design.get("generation_protocol") or {}).get("matched_planning", False)
        != execution_policy.matched_planning
        or (design.get("generation_protocol") or {}).get("max_intermediate_tokens", 700)
        != execution_policy.max_intermediate_tokens
        or (design.get("generation_protocol") or {}).get(
            "required_tool_contract_max_intermediate_tokens"
        )
        != execution_policy.required_tool_contract_max_intermediate_tokens
        or (design.get("generation_protocol") or {}).get("tool_catalog_bytes_bound", 0)
        != execution_policy.tool_catalog_bytes_bound
        or (design.get("generation_protocol") or {}).get("evidence_protocol", "legacy_v6")
        != execution_policy.evidence_protocol
        or (design.get("generation_protocol") or {}).get("epicure_on_tool_required", False)
        != execution_policy.epicure_on_tool_required
        or (design.get("generation_protocol") or {}).get("required_tool_contract_protocol")
        != execution_policy.required_tool_contract_protocol
        or (design.get("generation_protocol") or {}).get("required_tool_contract")
        != required_tool_contract(execution_policy)
        or (design.get("generation_protocol") or {}).get("required_tool_contract_sha256")
        != required_tool_contract(execution_policy)["content_address"]["digest"]
        or (design.get("generation_protocol") or {}).get("intermediate_reasoning_effort")
        != execution_policy.intermediate_reasoning_effort
        or (design.get("generation_protocol") or {}).get("final_reasoning_effort")
        != execution_policy.final_reasoning_effort
        or source.get("task_validity_artifact_sha256") != task_source.get("artifact_sha256")
        or source.get("task_candidate_coordinate_sha256")
        != task_source.get("candidate_coordinate_sha256")
        or source.get("task_registry_sha256") != task_registry_sha256(task_inventory)
        or governance.get("official") is not False
        or governance.get("rank_eligible") is not False
        or task_source.get("synthetic_tasks") != 0
        or len(candidates) != int(manifest.get("selection", {}).get("model_count") or 0)
    ):
        raise IntegrityError("current development task/model binding does not verify")
    if manifest.get("manifest_role") == "current_frontier_routed_development_quality_run":
        routing_policy = manifest.get("routing_policy")
        route_counts = Counter(candidate.execution_backend for candidate in candidates)
        recorded_route_counts = (manifest.get("selection") or {}).get("route_counts")
        expected_route_counts = {
            "kimi_direct": route_counts.get("kimi_direct", 0),
            "bedrock": route_counts.get("bedrock", 0),
            "openrouter_fallback": route_counts.get("openrouter", 0),
        }
        if route_counts.get("cohere_direct", 0):
            expected_route_counts["cohere_direct"] = route_counts["cohere_direct"]
        if (
            not isinstance(routing_policy, Mapping)
            or routing_policy.get("resolved_before_generation") is not True
            or routing_policy.get("generation_time_automatic_fallback") is not False
            or routing_policy.get("provider_substitution") != "prohibited"
            or recorded_route_counts != expected_route_counts
            or any(
                candidate.route_selection.get("selection_frozen_before_generation") is not True
                or candidate.route_selection.get("generation_time_automatic_fallback") is not False
                for candidate in candidates
            )
        ):
            raise IntegrityError("routed development provider precedence does not verify")


def derive_conditions_forecast(
    work_item: WorkItem,
    *,
    policy: ExecutionPolicy,
    conditions: Sequence[str],
) -> PairForecast:
    """Reserve an exact non-empty condition subset from frozen endpoint prices."""

    policy.validate()
    selected = tuple(conditions)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(CONDITIONS):
        raise IntegrityError("forecast requires a unique non-empty Epicure condition subset")
    pricing = work_item.candidate.endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise IntegrityError(f"{work_item.candidate.model_id} endpoint has no pricing contract")
    possible_tool_calls = min(
        policy.max_tool_calls_total,
        policy.max_tool_rounds * policy.max_tool_calls_per_round,
    )
    tool_context_bytes = min(
        policy.max_cumulative_tool_result_bytes,
        policy.max_tool_result_bytes * possible_tool_calls,
    )
    prompt_tokens_per_request = Decimal(
        len(work_item.task.prompt.encode("utf-8"))
        + policy.approximate_non_user_prompt_bytes
        + policy.tool_catalog_bytes_bound
        + tool_context_bytes
    ) / Decimal(policy.conservative_bytes_per_token)
    prompt_tokens_ceil = math.ceil(prompt_tokens_per_request)
    prompt_rate = _effective_rate(
        pricing,
        "prompt",
        prompt_tokens_per_request=prompt_tokens_ceil,
        required=True,
    )
    completion_rate = _effective_rate(
        pricing,
        "completion",
        prompt_tokens_per_request=prompt_tokens_ceil,
        required=True,
    )
    reasoning_rate = _effective_rate(
        pricing,
        "internal_reasoning",
        prompt_tokens_per_request=prompt_tokens_ceil,
    )
    request_rate = _effective_rate(
        pricing,
        "request",
        prompt_tokens_per_request=prompt_tokens_ceil,
    )
    if work_item.candidate.execution_backend == "openrouter":
        prompt_rate *= OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER
        completion_rate *= OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER
        reasoning_rate *= OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER
        request_rate *= OPENROUTER_PRICE_DRIFT_RESERVE_MULTIPLIER
    request_bound = 0
    total_completion_token_count = 0
    if "epicure_off" in selected:
        off_intermediate = 0
        if policy.matched_planning:
            off_intermediate += 1
            if policy.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS:
                off_intermediate += 1
        off_requests = 1 + off_intermediate
        request_bound += off_requests
        total_completion_token_count += (
            policy.max_output_tokens + off_intermediate * policy.max_intermediate_tokens
            if policy.matched_planning or policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else off_requests * policy.max_output_tokens
        )
    if "epicure_on" in selected:
        on_intermediate = (
            1
            if policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else policy.max_tool_rounds
        )
        if policy.matched_planning and policy.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS:
            on_intermediate += 1
        on_requests = 1 + on_intermediate
        request_bound += on_requests
        total_completion_token_count += (
            policy.max_output_tokens + on_intermediate * policy.max_intermediate_tokens
            if policy.matched_planning or policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else on_requests * policy.max_output_tokens
        )
    total_prompt_tokens = Decimal(request_bound) * prompt_tokens_per_request
    total_completion_tokens = Decimal(total_completion_token_count)
    forecast = (
        prompt_rate * total_prompt_tokens
        + completion_rate * total_completion_tokens
        + reasoning_rate * total_completion_tokens
        + request_rate * request_bound
    )
    if work_item.candidate.cost_accounting_policy == "provider_usage_with_unpriced_budget_ceiling":
        ceiling = _decimal(
            pricing.get("operational_reservation_ceiling_usd"),
            field=f"{work_item.candidate.model_id} unpriced budget ceiling",
        )
        if (
            ceiling <= 0
            or pricing.get("provider_rate_known") is not False
            or pricing.get("zero_values_mean") != "unknown_cost_not_free"
        ):
            raise IntegrityError(
                f"{work_item.candidate.model_id} unpriced ceiling contract is invalid"
            )
        # No direct provider rate is published for the mutable alias.  Retain
        # the complete frozen ceiling even for a condition subset instead of
        # manufacturing a token-price estimate.
        forecast = ceiling
    envelope = PriceEnvelope(
        prompt_usd_per_token=prompt_rate,
        completion_usd_per_token=completion_rate,
        reasoning_usd_per_token=reasoning_rate,
        request_usd=request_rate,
        prompt_usd_per_mtok=prompt_rate * Decimal(1_000_000),
        completion_usd_per_mtok=completion_rate * Decimal(1_000_000),
    )
    return PairForecast(
        forecast_usd=forecast,
        price_envelope=envelope,
        prompt_tokens_per_request_bound=prompt_tokens_per_request,
        completion_tokens_per_request_bound=policy.max_output_tokens,
        intermediate_completion_tokens_bound=(
            policy.max_intermediate_tokens
            if policy.matched_planning or policy.evidence_protocol == PORTABLE_TEXT_TOOL_PROTOCOL_V1
            else policy.max_output_tokens
        ),
        total_completion_tokens_bound=total_completion_token_count,
        request_bound=request_bound,
        conditions=selected,
    )


def derive_pair_forecast(
    work_item: WorkItem,
    *,
    policy: ExecutionPolicy,
) -> PairForecast:
    """Reserve one complete Epicure off/on pair."""

    return derive_conditions_forecast(work_item, policy=policy, conditions=CONDITIONS)


def _ledger_digest(entry: Mapping[str, Any]) -> str:
    value = dict(entry)
    value.pop("entry_sha256", None)
    return _sha256(value)


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {
        "api_key",
        "authorization",
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


def load_dataset_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    _require_regular_file(ledger_path, label="exploratory dataset ledger")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise IntegrityError(f"blank dataset ledger line {line_number}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise IntegrityError(f"invalid dataset ledger JSON line {line_number}") from error
        if not isinstance(entry, dict):
            raise IntegrityError(f"dataset ledger line {line_number} is not an object")
        if entry.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise IntegrityError(f"unsupported dataset ledger schema at line {line_number}")
        if entry.get("sequence") != line_number:
            raise IntegrityError(f"dataset ledger sequence mismatch at line {line_number}")
        if entry.get("previous_entry_sha256") != previous:
            raise IntegrityError(f"dataset ledger hash-chain mismatch at line {line_number}")
        digest = entry.get("entry_sha256")
        if not isinstance(digest, str) or digest != _ledger_digest(entry):
            raise IntegrityError(f"dataset ledger digest mismatch at line {line_number}")
        entries.append(entry)
        previous = digest
    return entries


def append_dataset_ledger_event(
    path: str | Path,
    event: Mapping[str, Any],
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Append and fsync one secret-free, hash-chained budget event."""

    if _contains_forbidden_key(event):
        raise IntegrityError("dataset ledger event contains a forbidden secret-bearing field")
    protected = {
        "entry_sha256",
        "previous_entry_sha256",
        "recorded_at",
        "schema_version",
        "sequence",
    }
    if protected.intersection(event):
        raise IntegrityError("dataset ledger event overrides protected hash-chain fields")
    if event.get("event_type") not in {
        "reservation_created",
        "source_artifact_recorded",
        "execution_incident",
        SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE,
    }:
        raise IntegrityError("unsupported dataset ledger event type")
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_dataset_ledger(ledger_path)
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": len(entries) + 1,
        "recorded_at": recorded_at or _utc_now(),
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        **dict(event),
    }
    entry["entry_sha256"] = _ledger_digest(entry)
    line = _canonical(entry) + b"\n"
    descriptor = os.open(ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError("short append while writing exploratory dataset ledger")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


def dataset_ledger_state(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    reservations: dict[str, Mapping[str, Any]] = {}
    finalizations: dict[str, Mapping[str, Any]] = {}
    reservation_digests: dict[str, str] = {}
    for entry in entries:
        event_type = entry.get("event_type")
        work_item_id = str(entry.get("work_item_id") or "")
        if event_type == "reservation_created":
            if not work_item_id or work_item_id in reservations:
                raise IntegrityError(f"duplicate or absent work-item reservation: {work_item_id}")
            reservations[work_item_id] = entry
            reservation_digests[str(entry["entry_sha256"])] = work_item_id
        elif event_type == "source_artifact_recorded":
            reservation_digest = str(entry.get("reservation_entry_sha256") or "")
            expected_work_item = reservation_digests.get(reservation_digest)
            if expected_work_item is None or expected_work_item != work_item_id:
                raise IntegrityError(
                    f"dataset finalization refers to an invalid reservation: {work_item_id}"
                )
            if work_item_id in finalizations:
                raise IntegrityError(f"dataset work item finalized twice: {work_item_id}")
            finalizations[work_item_id] = entry
    return reservations, finalizations


def dataset_incident_resolution_state(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Return one append-only conservative incident resolution per work item."""

    reservations: dict[str, Mapping[str, Any]] = {}
    incidents_by_digest: dict[str, Mapping[str, Any]] = {}
    resolutions: dict[str, Mapping[str, Any]] = {}
    finalized: set[str] = set()
    for entry in entries:
        event_type = entry.get("event_type")
        work_item_id = str(entry.get("work_item_id") or "")
        if event_type == "reservation_created":
            reservations[str(entry.get("entry_sha256") or "")] = entry
        elif event_type == "execution_incident":
            incidents_by_digest[str(entry.get("entry_sha256") or "")] = entry
        elif event_type == SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE:
            if not work_item_id or work_item_id in resolutions or work_item_id in finalized:
                raise IntegrityError(
                    f"duplicate, absent, or post-finalization incident resolution: {work_item_id}"
                )
            reservation_digest = str(entry.get("reservation_entry_sha256") or "")
            incident_digest = str(entry.get("incident_entry_sha256") or "")
            reservation = reservations.get(reservation_digest)
            incident = incidents_by_digest.get(incident_digest)
            if (
                reservation is None
                or incident is None
                or reservation.get("work_item_id") != work_item_id
                or incident.get("work_item_id") != work_item_id
                or incident.get("reservation_entry_sha256") != reservation_digest
            ):
                raise IntegrityError(
                    f"incident resolution has invalid reservation/incident links: {work_item_id}"
                )
            if incident.get("incident") != "generation_cost_unreconciled_reservation_retained":
                raise IntegrityError(
                    f"incident resolution targets an unsupported incident: {work_item_id}"
                )
            resolutions[work_item_id] = entry
        elif event_type == "source_artifact_recorded":
            finalized.add(work_item_id)
    return resolutions


@contextmanager
def _dataset_ledger_lock(path: Path) -> Iterable[TextIO]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _affected_no_choice_attempt_events(
    source_artifact: Mapping[str, Any],
    *,
    condition: str,
) -> list[Mapping[str, Any]]:
    events = source_artifact.get("provider_attempt_events")
    if not isinstance(events, list):
        raise IntegrityError("no-choice source has no provider-attempt evidence")
    suffix = f":{condition}"
    affected = [
        event
        for event in events
        if isinstance(event, Mapping) and str(event.get("arm_id") or "").endswith(suffix)
    ]
    if len(affected) != 2:
        raise IntegrityError("no-choice incident must contain exactly two affected events")
    started = [event for event in affected if event.get("event_type") == "request_started"]
    received = [event for event in affected if event.get("event_type") == "response_received"]
    if len(started) != 1 or len(received) != 1:
        raise IntegrityError("no-choice incident lacks one request/response pair")
    request, response = started[0], received[0]
    exact_link_fields = ("attempt_id", "payload_sha256", "request_key_sha256", "phase")
    if any(request.get(field) != response.get(field) for field in exact_link_fields):
        raise IntegrityError("no-choice request/response evidence does not share one attempt")
    if (
        request.get("phase") != "final"
        or response.get("http_status") != 200
        or str(response.get("generation_id") or "")
        or str(request.get("generation_id") or "")
    ):
        raise IntegrityError("incident is not an HTTP-200 final response without a generation ID")
    return affected


def build_source_incident_resolution_payload(
    *,
    source: DatasetSource,
    reservation: Mapping[str, Any],
    incident: Mapping[str, Any],
    affected_condition: str,
) -> dict[str, Any]:
    """Build deterministic evidence for conservative closure without inventing an ID."""

    if affected_condition not in CONDITIONS:
        raise IntegrityError("incident resolution has an unsupported condition")
    artifact = source.artifact
    work_item_id = source.work_item_id
    reservation_digest = str(reservation.get("entry_sha256") or "")
    incident_digest = str(incident.get("entry_sha256") or "")
    if (
        reservation.get("event_type") != "reservation_created"
        or reservation.get("work_item_id") != work_item_id
        or incident.get("event_type") != "execution_incident"
        or incident.get("work_item_id") != work_item_id
        or incident.get("reservation_entry_sha256") != reservation_digest
        or incident.get("source_artifact_sha256") != source.artifact_sha256
        or incident.get("incident") != "generation_cost_unreconciled_reservation_retained"
    ):
        raise IntegrityError("incident resolution reservation/source/incident links are invalid")
    errors = artifact.get("errors")
    results = artifact.get("results")
    budget = artifact.get("budget")
    if not all(isinstance(value, Mapping) for value in (errors, results, budget)):
        raise IntegrityError("incident source lacks errors/results/budget evidence")
    if errors.get(affected_condition) != "ProviderError: OpenRouter returned no final choice":
        raise IntegrityError("incident source does not record the approved no-choice error")
    if affected_condition in results:
        raise IntegrityError("the no-choice condition unexpectedly has a normalized result")
    normalizable_conditions = tuple(
        condition
        for condition in CONDITIONS
        if isinstance(results.get(condition), Mapping)
        and _result_is_reconciled(results[condition])
        and not _condition_has_identity_error(errors, condition)
    )
    if normalizable_conditions != ("epicure_on",):
        raise IntegrityError(
            "approved no-choice resolution requires exactly one valid Epicure-on result"
        )
    affected_events = _affected_no_choice_attempt_events(
        artifact,
        condition=affected_condition,
    )
    actual_micros = budget.get("actual_cost_micros")
    if (
        not isinstance(actual_micros, int)
        or isinstance(actual_micros, bool)
        or actual_micros < 0
        or budget.get("all_generation_costs_reconciled") is not True
    ):
        raise IntegrityError("source provider-reconciled actual cost is invalid")
    reserved = _decimal(
        reservation.get("reserved_usd"),
        field=f"incident resolution reservation {reservation_digest}",
    )
    source_cap = _decimal(budget.get("cap_usd"), field="incident source budget cap")
    forecast = _decimal(
        budget.get("forecast_worst_case_usd"),
        field="incident source forecast",
    )
    if reserved != source_cap or reserved != forecast:
        raise IntegrityError("incident reservation, source cap, and forecast are not exact")
    if source.exposure.exposure_basis != "failed_or_unreconciled_full_admitted_allowance":
        raise IntegrityError("source does not require conservative full-allowance closure")
    journal = artifact.get("run_journal")
    if not isinstance(journal, Mapping) or journal.get("finalized") is not True:
        raise IntegrityError("incident source has no finalized journal descriptor")
    affected_attempt_ids = sorted({str(event.get("attempt_id") or "") for event in affected_events})
    if len(affected_attempt_ids) != 1 or not affected_attempt_ids[0]:
        raise IntegrityError("no-choice evidence does not identify exactly one provider attempt")
    known_generation_ids = sorted(
        {
            str(generation_id)
            for condition in normalizable_conditions
            for generation_id in results[condition].get("generation_ids") or []
        }
    )
    actual = Decimal(actual_micros) / Decimal(1_000_000)
    return {
        "schema_version": SOURCE_INCIDENT_RESOLUTION_SCHEMA_VERSION,
        "record_type": SOURCE_INCIDENT_RESOLUTION_RECORD_TYPE,
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "research_release_eligible": False,
        "work_item_id": work_item_id,
        "source": {
            "artifact_filename": source.path.name,
            "artifact_sha256": source.artifact_sha256,
            "run_id": artifact.get("run_id"),
            "requested_model_id": artifact.get("requested_model_id"),
            "requested_provider": artifact.get("requested_provider"),
            "status": artifact.get("status"),
        },
        "reservation": {
            "entry_sha256": reservation_digest,
            "reserved_usd": _decimal_text(reserved),
        },
        "incident": {
            "entry_sha256": incident_digest,
            "incident": incident.get("incident"),
            "affected_condition": affected_condition,
            "error_sha256": hashlib.sha256(
                str(errors[affected_condition]).encode("utf-8")
            ).hexdigest(),
        },
        "attempt_evidence": {
            "attempt_ids": affected_attempt_ids,
            "event_count": len(affected_events),
            "events_sha256": _sha256(affected_events),
            "event_sha256s": [_sha256(event) for event in affected_events],
            "request_payload_sha256": affected_events[0].get("payload_sha256"),
            "request_key_sha256": affected_events[0].get("request_key_sha256"),
            "http_status": 200,
            "run_journal_filename": journal.get("filename"),
            "run_journal_sha256": journal.get("sha256"),
            "run_journal_head_entry_sha256": journal.get("head_entry_sha256"),
        },
        "unidentified_response": {
            "generation_id_known": False,
            "generation_ids": [],
            "generation_id_was_inferred": False,
            "affected_condition": affected_condition,
        },
        "preserved_valid_result": {
            "conditions": list(normalizable_conditions),
            "generation_ids": known_generation_ids,
            "generation_ids_sha256": _sha256(known_generation_ids),
        },
        "cost": {
            "currency": "USD",
            "provider_reconciled_actual_cost_micros": actual_micros,
            "provider_reconciled_actual_cost_usd": _decimal_text(actual),
            "provider_cost_exact_for_unidentified_response": False,
            "admitted_allowance_usd": _decimal_text(reserved),
            "conservative_budget_exposure_usd": _decimal_text(reserved),
            "budget_exposure_is_conservative_not_actual": True,
        },
        "resolution": {
            "status": "closed_with_full_allowance_held",
            "safe_to_replay": False,
            "provider_retry_authorized": False,
            "reservation_may_be_finalized": True,
            "normalizable_conditions": list(normalizable_conditions),
            "failed_conditions": [affected_condition],
            "budget_exposure_basis": RESOLVED_CONSERVATIVE_EXPOSURE_BASIS,
        },
        "limitations": [
            "No generation ID exists for the affected HTTP-200/no-choice response.",
            "No generation ID or exact provider cost is inferred by this resolution.",
            "The complete admitted allowance remains charged as budget exposure.",
            "The record remains exploratory, unranked, and ineligible for research release.",
        ],
    }


def _verify_source_incident_resolution(
    path: Path,
    *,
    source: DatasetSource,
    reservation: Mapping[str, Any],
    incident: Mapping[str, Any],
    ledger_event: Mapping[str, Any],
) -> SourceIncidentResolution:
    _require_regular_file(path, label="source incident resolution")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read source incident resolution {path}") from error
    if not isinstance(document, dict):
        raise IntegrityError(f"source incident resolution is not an object: {path}")
    digest = document.get("artifact_sha256")
    unhashed = dict(document)
    unhashed.pop("artifact_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or _sha256(unhashed) != digest
        or path.name != f"source-incident-resolution-{digest}.json"
    ):
        raise IntegrityError(f"source incident resolution is not content addressed: {path}")
    expected = build_source_incident_resolution_payload(
        source=source,
        reservation=reservation,
        incident=incident,
        affected_condition=str((document.get("incident") or {}).get("affected_condition") or ""),
    )
    if unhashed != expected:
        raise IntegrityError(f"source incident resolution differs from source evidence: {path}")
    protected_flags = (
        "official",
        "rank_eligible",
        "research_result",
        "research_release_eligible",
    )
    if any(document.get(field) is not False for field in protected_flags):
        raise IntegrityError(f"source incident resolution claims governed status: {path}")
    event_exact = {
        "work_item_id": source.work_item_id,
        "reservation_entry_sha256": reservation.get("entry_sha256"),
        "incident_entry_sha256": incident.get("entry_sha256"),
        "source_artifact_sha256": source.artifact_sha256,
        "resolution_filename": path.name,
        "resolution_artifact_sha256": digest,
        "affected_condition": expected["incident"]["affected_condition"],
        "provider_reconciled_actual_cost_usd": expected["cost"][
            "provider_reconciled_actual_cost_usd"
        ],
        "conservative_budget_exposure_usd": expected["cost"]["conservative_budget_exposure_usd"],
        "provider_cost_exact_for_unidentified_response": False,
        "safe_to_replay": False,
        "normalizable_conditions": expected["resolution"]["normalizable_conditions"],
    }
    for field, value in event_exact.items():
        if ledger_event.get(field) != value:
            raise IntegrityError(
                f"source incident resolution ledger event {field} mismatch: {path}"
            )
    return SourceIncidentResolution(
        path=path,
        artifact_sha256=digest,
        work_item_id=source.work_item_id,
        source_artifact_sha256=source.artifact_sha256,
        reservation_entry_sha256=str(reservation["entry_sha256"]),
        incident_entry_sha256=str(incident["entry_sha256"]),
        affected_condition=str(expected["incident"]["affected_condition"]),
        normalizable_conditions=tuple(expected["resolution"]["normalizable_conditions"]),
        provider_reconciled_actual_cost_usd=_decimal(
            expected["cost"]["provider_reconciled_actual_cost_usd"],
            field="resolution provider actual",
        ),
        conservative_budget_exposure_usd=_decimal(
            expected["cost"]["conservative_budget_exposure_usd"],
            field="resolution conservative exposure",
        ),
        ledger_event_sha256=str(ledger_event["entry_sha256"]),
    )


def _load_source_incident_resolutions(
    directory: str | Path | None,
    *,
    sources: Mapping[str, DatasetSource],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, SourceIncidentResolution]:
    ledger_events = dataset_incident_resolution_state(ledger)
    if directory is None:
        if ledger_events:
            raise IntegrityError("ledger has source incident resolutions but no resolution root")
        return {}
    root = Path(directory)
    if not root.exists():
        if ledger_events:
            raise IntegrityError("ledger refers to absent source incident resolution root")
        return {}
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"source incident resolution root must be a directory: {root}")
    reservations = {
        str(entry.get("entry_sha256") or ""): entry
        for entry in ledger
        if entry.get("event_type") == "reservation_created"
    }
    incidents = {
        str(entry.get("entry_sha256") or ""): entry
        for entry in ledger
        if entry.get("event_type") == "execution_incident"
    }
    events_by_digest = {
        str(entry.get("resolution_artifact_sha256") or ""): entry
        for entry in ledger_events.values()
    }
    resolutions: dict[str, SourceIncidentResolution] = {}
    seen_digests: set[str] = set()
    for path in sorted(root.glob("*.json")):
        raw = _read_json_regular(path, label="source incident resolution")
        digest = str(raw.get("artifact_sha256") or "")
        event = events_by_digest.get(digest)
        if event is None or digest in seen_digests:
            raise IntegrityError(f"orphan or duplicate source incident resolution: {path}")
        work_item_id = str(event.get("work_item_id") or "")
        source = sources.get(work_item_id)
        reservation = reservations.get(str(event.get("reservation_entry_sha256") or ""))
        incident = incidents.get(str(event.get("incident_entry_sha256") or ""))
        if source is None or reservation is None or incident is None:
            raise IntegrityError(f"source incident resolution has missing immutable links: {path}")
        resolution = _verify_source_incident_resolution(
            path,
            source=source,
            reservation=reservation,
            incident=incident,
            ledger_event=event,
        )
        if work_item_id in resolutions:
            raise IntegrityError(f"more than one resolution exists for {work_item_id}")
        resolutions[work_item_id] = resolution
        seen_digests.add(digest)
    if seen_digests != set(events_by_digest):
        raise IntegrityError("ledger source incident resolution artifact is absent")
    return resolutions


def _read_json_regular(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not an object: {path}")
    return value


def _load_dataset_sources(
    directory: str | Path,
    *,
    corrections_directory: str | Path | None,
    resolution_directory: str | Path | None,
    ledger: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, DatasetSource],
    Decimal,
    Decimal,
    Decimal,
    dict[str, SourceIncidentResolution],
]:
    root = Path(directory)
    scan = scan_live_smoke_artifacts(
        root,
        corrections_directory=corrections_directory,
    )
    exposure_by_digest = {item.artifact_sha256: item for item in scan.artifacts}
    sources: dict[str, DatasetSource] = {}
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        artifact, digest = _verify_live_artifact(path)
        work_item_id = str(artifact.get("dataset_work_item_id") or "")
        if len(work_item_id) != 64:
            raise IntegrityError(f"dataset source has no valid work-item ID: {path}")
        if work_item_id in sources:
            raise IntegrityError(f"more than one source exists for work item {work_item_id}")
        if artifact.get("run_purpose") != "epicure_on_off_pair":
            raise IntegrityError(f"dataset source is not an Epicure on/off pair: {path}")
        if artifact.get("official") is not False:
            raise IntegrityError(f"dataset source unexpectedly claims official status: {path}")
        if artifact.get("rank_eligible") is not False:
            raise IntegrityError(f"dataset source unexpectedly claims rank eligibility: {path}")
        if artifact.get("research_result") is not False:
            raise IntegrityError(f"dataset source unexpectedly claims research status: {path}")
        sources[work_item_id] = DatasetSource(
            path=path,
            artifact_sha256=digest,
            work_item_id=work_item_id,
            artifact=artifact,
            exposure=exposure_by_digest[digest],
        )
    resolutions = _load_source_incident_resolutions(
        resolution_directory,
        sources=sources,
        ledger=ledger,
    )
    for work_item_id, resolution in resolutions.items():
        source = sources[work_item_id]
        exposure = source.exposure
        if exposure.actual_cost_usd != resolution.provider_reconciled_actual_cost_usd:
            raise IntegrityError(
                f"incident resolution provider actual differs from source: {work_item_id}"
            )
        if exposure.exposure_usd != resolution.conservative_budget_exposure_usd:
            raise IntegrityError(
                f"incident resolution must retain the original full exposure: {work_item_id}"
            )
        sources[work_item_id] = DatasetSource(
            path=source.path,
            artifact_sha256=source.artifact_sha256,
            work_item_id=source.work_item_id,
            artifact=source.artifact,
            exposure=ArtifactExposure(
                path=exposure.path,
                artifact_sha256=exposure.artifact_sha256,
                status=exposure.status,
                requested_model_id=exposure.requested_model_id,
                requested_provider=exposure.requested_provider,
                candidate_manifest_sha256=exposure.candidate_manifest_sha256,
                actual_cost_usd=exposure.actual_cost_usd,
                forecast_usd=exposure.forecast_usd,
                admitted_cap_usd=exposure.admitted_cap_usd,
                exposure_usd=resolution.conservative_budget_exposure_usd,
                exposure_basis=RESOLVED_CONSERVATIVE_EXPOSURE_BASIS,
                contract_passed=exposure.contract_passed,
                cost_correction_sha256=exposure.cost_correction_sha256,
            ),
        )
    effective_exposure = sum(
        (source.exposure.exposure_usd for source in sources.values()),
        Decimal(0),
    )
    unresolved = sum(
        (
            source.exposure.exposure_usd
            for source in sources.values()
            if source.exposure.exposure_basis == "failed_or_unreconciled_full_admitted_allowance"
        ),
        Decimal(0),
    )
    return (
        sources,
        scan.actual_cost_usd,
        effective_exposure,
        unresolved,
        resolutions,
    )


def _verify_response_artifact(path: Path) -> ResponseArtifact:
    _require_regular_file(path, label="exploratory response artifact")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise IntegrityError(f"could not read response artifact {path}") from error
    if not isinstance(artifact, dict):
        raise IntegrityError(f"response artifact is not an object: {path}")
    digest = artifact.get("artifact_sha256")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if not isinstance(digest, str) or len(digest) != 64 or _sha256(unhashed) != digest:
        raise IntegrityError(f"response artifact content address is invalid: {path}")
    if not path.stem.endswith(digest):
        raise IntegrityError(f"response artifact filename does not contain its digest: {path}")
    if artifact.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise IntegrityError(f"unsupported response artifact schema: {path}")
    for field in ("official", "rank_eligible", "research_result"):
        if artifact.get(field) is not False:
            raise IntegrityError(f"response artifact unexpectedly sets {field}: {path}")
    policy_document = artifact.get("execution_policy")
    if (
        not verify_policy_document(policy_document)
        or artifact.get("execution_policy_sha256") != policy_document["content_address"]["digest"]
    ):
        raise IntegrityError(f"response artifact has an invalid execution policy: {path}")
    work_item_id = str(artifact.get("work_item_id") or "")
    condition = str(artifact.get("condition") or "")
    task = artifact.get("task")
    model = artifact.get("model")
    source = artifact.get("source")
    cost = artifact.get("cost")
    response = artifact.get("response")
    if len(work_item_id) != 64 or condition not in CONDITIONS:
        raise IntegrityError(f"response artifact has invalid work-item/condition: {path}")
    if not all(isinstance(value, Mapping) for value in (task, model, source, cost, response)):
        raise IntegrityError(f"response artifact lacks task/model/source/cost/response: {path}")
    legacy_cost_contract = "recorded_cost_micros" not in cost
    cost_micros = (
        cost.get("actual_cost_micros") if legacy_cost_contract else cost.get("recorded_cost_micros")
    )
    exact_cost = (legacy_cost_contract and cost.get("all_generation_costs_reconciled") is True) or (
        not legacy_cost_contract
        and cost.get("all_generation_costs_reconciled") is True
        and cost.get("provider_cost_exact") is True
        and cost.get("cost_status") == "provider_reconciled_actual"
        and cost.get("actual_cost_micros") == cost_micros
        and cost.get("estimated_cost_micros") is None
    )
    estimated_cost = (
        cost.get("all_generation_costs_reconciled") is False
        and cost.get("all_generation_usage_accounted") is True
        and cost.get("provider_cost_exact") is False
        and cost.get("cost_status") == "provider_usage_times_frozen_rate_card_estimate"
        and cost.get("actual_cost_micros") is None
        and cost.get("estimated_cost_micros") == cost_micros
        and cost.get("exact_cost_ranking_eligible") is False
    )
    unpriced_cost = (
        cost.get("all_generation_costs_reconciled") is False
        and cost.get("all_generation_usage_accounted") is True
        and cost.get("provider_cost_exact") is False
        and cost.get("provider_cost_known") is False
        and cost.get("cost_status") == "provider_cost_unavailable_full_budget_ceiling_retained"
        and cost.get("actual_cost_micros") is None
        and cost.get("estimated_cost_micros") is None
        and cost.get("recorded_cost_micros") == 0
        and cost.get("exact_cost_ranking_eligible") is False
        and Decimal(str(cost.get("full_budget_ceiling_retained_usd") or "0")) > 0
    )
    if (
        not isinstance(cost_micros, int)
        or isinstance(cost_micros, bool)
        or cost_micros < 0
        or not (exact_cost or estimated_cost or unpriced_cost)
    ):
        raise IntegrityError(f"response artifact has invalid cost provenance: {path}")
    return ResponseArtifact(
        path=path,
        artifact_sha256=digest,
        work_item_id=work_item_id,
        condition=condition,
        task_id=str(task.get("public_id") or ""),
        task_family=str(task.get("family") or ""),
        model_id=str(model.get("requested_model_id") or ""),
        provider_tag=str(model.get("provider_tag") or ""),
        source_artifact_sha256=str(source.get("artifact_sha256") or ""),
        actual_cost_usd=Decimal(cost_micros) / Decimal(1_000_000),
        tool_used=bool(response.get("tool_trace")),
    )


def scan_response_artifacts(
    directory: str | Path,
) -> dict[tuple[str, str], ResponseArtifact]:
    root = Path(directory)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise IntegrityError(f"response artifact root must be a directory: {root}")
    responses: dict[tuple[str, str], ResponseArtifact] = {}
    seen_digests: set[str] = set()
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        artifact = _verify_response_artifact(path)
        key = (artifact.work_item_id, artifact.condition)
        if key in responses or artifact.artifact_sha256 in seen_digests:
            raise IntegrityError(f"duplicate response artifact: {path}")
        responses[key] = artifact
        seen_digests.add(artifact.artifact_sha256)
    return responses


def _load_state(
    *,
    prior_artifact_directory: str | Path,
    prior_corrections_directory: str | Path | None,
    prior_reservation_ledger_path: str | Path,
    source_directory: str | Path,
    source_corrections_directory: str | Path | None,
    response_directory: str | Path,
    ledger_path: str | Path,
    source_resolution_directory: str | Path | None = None,
) -> DatasetState:
    prior = scan_live_smoke_artifacts(
        prior_artifact_directory,
        corrections_directory=prior_corrections_directory,
    )
    prior_effective = max(prior.exposure_usd, KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD)
    frontier_ledger = load_frontier_ledger(prior_reservation_ledger_path)
    validate_ledger_artifact_links(
        frontier_ledger,
        prior,
        reconciliation_directory=(Path(prior_reservation_ledger_path).parent / "reconciliations"),
    )
    prior_active_reservations = active_ledger_reservations(frontier_ledger)
    prior_active_total = sum(prior_active_reservations.values(), Decimal(0))
    ledger = load_dataset_ledger(ledger_path)
    resolution_root = (
        Path(source_resolution_directory)
        if source_resolution_directory is not None
        else Path(source_directory).parent / "resolutions"
    )
    (
        sources,
        dataset_actual,
        dataset_exposure,
        dataset_unresolved,
        incident_resolutions,
    ) = _load_dataset_sources(
        source_directory,
        corrections_directory=source_corrections_directory,
        resolution_directory=resolution_root,
        ledger=ledger,
    )
    responses = scan_response_artifacts(response_directory)
    reservations, finalizations = dataset_ledger_state(ledger)
    for work_item_id, finalization in finalizations.items():
        source = sources.get(work_item_id)
        if source is None:
            raise IntegrityError(f"finalized work item has no source artifact: {work_item_id}")
        if finalization.get("source_artifact_sha256") != source.artifact_sha256:
            raise IntegrityError(f"finalized work item source digest mismatch: {work_item_id}")
        resolution = incident_resolutions.get(work_item_id)
        if resolution is not None:
            if (
                finalization.get("source_incident_resolution_sha256") != resolution.artifact_sha256
                or finalization.get("source_incident_resolution_ledger_entry_sha256")
                != resolution.ledger_event_sha256
                or finalization.get("all_generation_costs_reconciled") is not False
                or finalization.get("provider_cost_exact") is not False
                or finalization.get("source_budget_exposure_usd")
                != _decimal_text(resolution.conservative_budget_exposure_usd)
            ):
                raise IntegrityError(
                    f"finalized conservative incident resolution mismatch: {work_item_id}"
                )
        recorded_digests = set(finalization.get("response_artifact_sha256s") or [])
        current_digests = {
            response.artifact_sha256
            for (response_work_item, _), response in responses.items()
            if response_work_item == work_item_id
        }
        if recorded_digests != current_digests:
            raise IntegrityError(f"finalized work item response digest mismatch: {work_item_id}")
    for (work_item_id, _), response in responses.items():
        source = sources.get(work_item_id)
        if source is None or response.source_artifact_sha256 != source.artifact_sha256:
            raise IntegrityError(
                f"response artifact has no matching immutable source: {response.path}"
            )
    orphan_reservations = [
        reservation
        for work_item_id, reservation in reservations.items()
        if work_item_id not in finalizations and work_item_id not in sources
    ]
    orphan_total = sum(
        (
            _decimal(
                reservation.get("reserved_usd"),
                field=f"dataset reservation {reservation.get('entry_sha256')}",
            )
            for reservation in orphan_reservations
        ),
        Decimal(0),
    )
    return DatasetState(
        prior_verified_exposure_usd=prior.exposure_usd,
        prior_effective_exposure_usd=prior_effective,
        prior_active_reservation_usd=prior_active_total,
        dataset_actual_cost_usd=dataset_actual,
        dataset_source_exposure_usd=dataset_exposure,
        unresolved_dataset_source_reserve_usd=dataset_unresolved,
        sources=sources,
        responses=responses,
        ledger=tuple(ledger),
        reservations=reservations,
        finalizations=finalizations,
        orphan_reservation_usd=orphan_total,
        incident_resolutions=incident_resolutions,
    )


def _source_postflight_issues(
    source: DatasetSource,
    work_item: WorkItem,
    *,
    expected_conditions: Sequence[str] = CONDITIONS,
    expected_epicure: Mapping[str, str] | None = None,
) -> list[str]:
    artifact = source.artifact
    issues: list[str] = []
    selected_conditions = tuple(expected_conditions)
    expected_run_purpose = (
        "epicure_on_off_pair" if selected_conditions == CONDITIONS else "epicure_condition_subset"
    )
    exact_fields = {
        "dataset_work_item_id": work_item.work_item_id,
        "dataset_task_id": work_item.task.public_id,
        "candidate_manifest_sha256": work_item.manifest_sha256,
        "prompt": work_item.task.prompt,
        "prompt_sha256": work_item.task.prompt_sha256,
        "category": work_item.task.family,
        "requested_model_id": work_item.candidate.model_id,
        "requested_provider": work_item.candidate.provider_tag,
        "run_purpose": expected_run_purpose,
    }
    for field, expected in exact_fields.items():
        if artifact.get(field) != expected:
            issues.append(f"{field}_mismatch")
    if work_item.candidate.execution_backend != "openrouter":
        if artifact.get("execution_backend") != work_item.candidate.execution_backend:
            issues.append("execution_backend_mismatch")
        if artifact.get("backend_contract") != work_item.candidate.backend_contract:
            issues.append("backend_contract_mismatch")
        if artifact.get("backend_contract_sha256") != work_item.candidate.backend_contract_sha256:
            issues.append("backend_contract_sha256_mismatch")
        if artifact.get("execution_route") != work_item.candidate.route_selection:
            issues.append("execution_route_mismatch")
    model = artifact.get("model_contract")
    if not isinstance(model, Mapping):
        issues.append("missing_model_contract")
    else:
        if model.get("id") != work_item.candidate.model_id:
            issues.append("model_contract_id_mismatch")
        if model.get("canonical_slug") != work_item.candidate.canonical_model_slug:
            issues.append("canonical_model_slug_mismatch")
    endpoint = artifact.get("endpoint_contract")
    if not isinstance(endpoint, Mapping):
        issues.append("missing_endpoint_contract")
    elif endpoint_execution_contract(dict(endpoint)) != endpoint_execution_contract(
        dict(work_item.candidate.endpoint)
    ):
        issues.append("endpoint_execution_contract_mismatch")
    if artifact.get("endpoint_execution_contract_sha256") != (work_item.endpoint_execution_sha256):
        issues.append("endpoint_execution_contract_sha256_mismatch")
    policy_document = artifact.get("execution_policy")
    if not verify_policy_document(policy_document):
        issues.append("invalid_execution_policy_document")
    elif policy_document != work_item.execution_policy.document():
        issues.append("execution_policy_document_mismatch")
    if artifact.get("execution_policy_sha256") != work_item.execution_policy_sha256:
        issues.append("execution_policy_sha256_mismatch")
    recorded_conditions = artifact.get("requested_conditions")
    if recorded_conditions is not None and recorded_conditions != list(selected_conditions):
        issues.append("requested_conditions_mismatch")
    if expected_run_purpose == "epicure_condition_subset" and artifact.get(
        "requested_conditions"
    ) != list(selected_conditions):
        issues.append("requested_conditions_missing")
    if expected_epicure is not None:
        observed_epicure = artifact.get("epicure")
        if not isinstance(observed_epicure, Mapping):
            issues.append("missing_epicure_provenance")
        else:
            for field in ("release_id", "bundle_sha256", "application_sha256"):
                if observed_epicure.get(field) != expected_epicure.get(field):
                    issues.append(f"epicure_{field}_mismatch")
        if artifact.get("epicure_tool_schema_sha256") != expected_epicure.get("tool_schema_sha256"):
            issues.append("epicure_tool_schema_sha256_mismatch")
    expected_decoding = {
        "temperature": work_item.execution_policy.decoding_temperature,
        "top_p": work_item.execution_policy.decoding_top_p,
        "seed": work_item.execution_policy.decoding_seed,
        "max_output_tokens": work_item.execution_policy.max_output_tokens,
        "max_tool_rounds": work_item.execution_policy.max_tool_rounds,
        "max_tool_calls_per_round": (work_item.execution_policy.max_tool_calls_per_round),
        "max_tool_calls_total": work_item.execution_policy.max_tool_calls_total,
        "max_tool_result_bytes": work_item.execution_policy.max_tool_result_bytes,
        "max_cumulative_tool_result_bytes": (
            work_item.execution_policy.max_cumulative_tool_result_bytes
        ),
        "max_provider_attempts": work_item.execution_policy.max_provider_attempts,
        "parallel_tool_calls_enforcement": "bounded_sequential_execution",
    }
    if artifact.get("decoding") != expected_decoding:
        issues.append("source_decoding_or_tool_limits_mismatch")
    generation_contract = artifact.get("frozen_generation_contract")
    supported = set(work_item.candidate.endpoint.get("supported_parameters") or [])
    requested_decoding = {
        "max_tokens": work_item.execution_policy.max_output_tokens,
        "temperature": work_item.execution_policy.decoding_temperature,
        "top_p": work_item.execution_policy.decoding_top_p,
        "seed": work_item.execution_policy.decoding_seed,
    }
    expected_effective_decoding = {
        name: value for name, value in requested_decoding.items() if name in supported
    }
    if not isinstance(generation_contract, Mapping):
        issues.append("missing_frozen_generation_contract")
    else:
        if generation_contract.get("supported_parameters") != sorted(supported):
            issues.append("generation_supported_parameters_mismatch")
        if generation_contract.get("decoding_parameters") != expected_effective_decoding:
            issues.append("generation_decoding_parameters_mismatch")
        if generation_contract.get("expected_actual_model_id") != (
            work_item.candidate.canonical_model_slug
        ):
            issues.append("generation_expected_model_mismatch")
        if generation_contract.get("expected_actual_provider_slug") != (
            work_item.candidate.provider_name
        ):
            issues.append("generation_expected_provider_mismatch")
        if work_item.candidate.execution_backend != "openrouter" and (
            generation_contract.get("execution_backend") != work_item.candidate.execution_backend
            or generation_contract.get("backend_contract_sha256")
            != work_item.candidate.backend_contract_sha256
        ):
            issues.append("generation_backend_contract_mismatch")
        if generation_contract.get("final_response_mode", "structured_json") != (
            work_item.execution_policy.final_response_mode
        ):
            issues.append("generation_final_response_mode_mismatch")
        if generation_contract.get("matched_planning", False) != (
            work_item.execution_policy.matched_planning
        ):
            issues.append("generation_matched_planning_mismatch")
        if generation_contract.get("intermediate_max_tokens", 700) != (
            work_item.execution_policy.max_intermediate_tokens
        ):
            issues.append("generation_intermediate_token_limit_mismatch")
        if generation_contract.get("required_tool_contract_max_intermediate_tokens") != (
            work_item.execution_policy.required_tool_contract_max_intermediate_tokens
        ):
            issues.append("generation_required_tool_intermediate_token_limit_mismatch")
        if generation_contract.get("evidence_protocol", "legacy_v6") != (
            work_item.execution_policy.evidence_protocol
        ):
            issues.append("generation_evidence_protocol_mismatch")
        if generation_contract.get("epicure_on_tool_required", False) != (
            work_item.execution_policy.epicure_on_tool_required
        ):
            issues.append("generation_epicure_tool_requirement_mismatch")
        if generation_contract.get("required_tool_contract_protocol") != (
            work_item.execution_policy.required_tool_contract_protocol
        ):
            issues.append("generation_required_tool_contract_protocol_mismatch")
        if (
            generation_contract.get("required_tool_contract_sha256")
            != (required_tool_contract(work_item.execution_policy)["content_address"]["digest"])
        ):
            issues.append("generation_required_tool_contract_hash_mismatch")
        if generation_contract.get("intermediate_reasoning_effort") != (
            work_item.execution_policy.intermediate_reasoning_effort
        ):
            issues.append("generation_intermediate_reasoning_mismatch")
        if generation_contract.get("final_reasoning_effort") != (
            work_item.execution_policy.final_reasoning_effort
        ):
            issues.append("generation_final_reasoning_mismatch")
        expected_mutable_alias_opt_in = (
            work_item.candidate.execution_backend == "qwencloud_direct"
            and work_item.candidate.backend_contract.get("identity_kind") == "mutable_alias"
        )
        if generation_contract.get("allow_mutable_alias_exploratory", False) is not (
            expected_mutable_alias_opt_in
        ):
            issues.append("generation_mutable_alias_opt_in_mismatch")
    protocol_bundle = artifact.get("protocol_bundle")
    protocol_bundle_sha256 = artifact.get("protocol_bundle_sha256")
    if (
        not isinstance(protocol_bundle, Mapping)
        or not isinstance(protocol_bundle_sha256, str)
        or len(protocol_bundle_sha256) != 64
        or _sha256(protocol_bundle) != protocol_bundle_sha256
    ):
        issues.append("invalid_protocol_bundle")
    else:
        run_binding = protocol_bundle.get("run_binding")
        core_bundle = protocol_bundle.get("core_protocol_bundle")
        if not isinstance(run_binding, Mapping) or not isinstance(core_bundle, Mapping):
            issues.append("invalid_protocol_bundle_structure")
        else:
            expected_run_binding = {
                "candidate_manifest_sha256": work_item.manifest_sha256,
                "dataset_work_item_id": work_item.work_item_id,
                "dataset_task_id": work_item.task.public_id,
                "prompt_sha256": work_item.task.prompt_sha256,
                "category": work_item.task.family,
                "canonical_model_slug": work_item.candidate.canonical_model_slug,
                "provider_tag": work_item.candidate.provider_tag,
                "endpoint_contract_sha256": (
                    generation_contract.get("endpoint_contract_sha256")
                    if isinstance(generation_contract, Mapping)
                    else None
                ),
                "execution_policy_sha256": work_item.execution_policy_sha256,
                "final_response_mode": work_item.execution_policy.final_response_mode,
                "matched_planning": work_item.execution_policy.matched_planning,
                "max_intermediate_tokens": (work_item.execution_policy.max_intermediate_tokens),
                "required_tool_contract_max_intermediate_tokens": (
                    work_item.execution_policy.required_tool_contract_max_intermediate_tokens
                ),
                "evidence_protocol": work_item.execution_policy.evidence_protocol,
                "required_tool_contract_protocol": (
                    work_item.execution_policy.required_tool_contract_protocol
                ),
                "required_tool_contract_sha256": required_tool_contract(work_item.execution_policy)[
                    "content_address"
                ]["digest"],
                "epicure_on_tool_required": (work_item.execution_policy.epicure_on_tool_required),
                "intermediate_reasoning_effort": (
                    work_item.execution_policy.intermediate_reasoning_effort
                ),
                "final_reasoning_effort": work_item.execution_policy.final_reasoning_effort,
            }
            if (
                work_item.candidate.execution_backend == "qwencloud_direct"
                and work_item.candidate.backend_contract.get("identity_kind") == "mutable_alias"
            ):
                expected_run_binding["allow_mutable_alias_exploratory"] = True
            if expected_run_purpose == "epicure_condition_subset":
                expected_run_binding["selected_conditions"] = list(selected_conditions)
            if any(
                run_binding.get(
                    field,
                    (
                        "structured_json"
                        if field == "final_response_mode"
                        else False
                        if field == "matched_planning"
                        else 700
                        if field == "max_intermediate_tokens"
                        else None
                        if field == "required_tool_contract_max_intermediate_tokens"
                        else "legacy_v6"
                        if field == "evidence_protocol"
                        else False
                        if field == "epicure_on_tool_required"
                        else None
                    ),
                )
                != expected
                for field, expected in expected_run_binding.items()
            ):
                issues.append("protocol_run_binding_mismatch")
            if core_bundle.get("tool_registry_sha256") != artifact.get(
                "epicure_tool_schema_sha256"
            ):
                issues.append("protocol_epicure_tool_catalog_mismatch")
            if core_bundle.get("model_smoke_registry_sha256") != work_item.manifest_sha256:
                issues.append("protocol_manifest_binding_mismatch")
        if (
            isinstance(generation_contract, Mapping)
            and generation_contract.get("protocol_bundle_sha256") != protocol_bundle_sha256
        ):
            issues.append("generation_protocol_bundle_mismatch")
    results = artifact.get("results")
    if not isinstance(results, Mapping) or set(results) - set(selected_conditions):
        issues.append("unexpected_results_contract")
    elif work_item.execution_policy.epicure_on_tool_required and isinstance(
        results.get("epicure_on"), Mapping
    ):
        epicure_on = results.get("epicure_on")
        traces = epicure_on.get("tool_trace") if isinstance(epicure_on, Mapping) else None
        if not isinstance(traces, list) or not traces:
            issues.append("required_epicure_trace_missing")
        elif not any(
            isinstance(trace, Mapping) and trace.get("is_error") is False for trace in traces
        ):
            issues.append("required_epicure_success_missing")
    return issues


def _result_is_reconciled(result: Mapping[str, Any]) -> bool:
    metadata = result.get("generation_metadata")
    generation_ids = result.get("generation_ids")
    cost_micros = result.get("cost_micros")
    if (
        result.get("cost_reconciled") is not True
        or not isinstance(metadata, list)
        or not isinstance(generation_ids, list)
        or not isinstance(cost_micros, int)
        or isinstance(cost_micros, bool)
        or cost_micros < 0
    ):
        return False
    seen_ids: set[str] = set()
    summed_cost = 0
    for generation in metadata:
        if not isinstance(generation, Mapping) or generation.get("reconciled") is not True:
            return False
        generation_id = str(generation.get("generation_id") or "")
        generation_cost = generation.get("cost_micros")
        if (
            not generation_id
            or generation_id in seen_ids
            or not isinstance(generation_cost, int)
            or isinstance(generation_cost, bool)
            or generation_cost < 0
        ):
            return False
        seen_ids.add(generation_id)
        summed_cost += generation_cost
    return set(map(str, generation_ids)) == seen_ids and summed_cost == cost_micros


def _result_has_rate_card_accounting(
    result: Mapping[str, Any],
    execution_backend: str = "kimi_direct",
) -> bool:
    accounting_basis = RATE_CARD_ACCOUNTING_BASIS_BY_BACKEND.get(execution_backend)
    metadata = result.get("generation_metadata")
    generation_ids = result.get("generation_ids")
    cost_micros = result.get("cost_micros")
    if (
        accounting_basis is None
        or result.get("cost_reconciled") is not False
        or result.get("cost_accounting_basis") != accounting_basis
        or result.get("billing_reconciliation_status") != "provider_charge_unavailable"
        or not isinstance(metadata, list)
        or not isinstance(generation_ids, list)
        or not isinstance(cost_micros, int)
        or isinstance(cost_micros, bool)
        or cost_micros < 0
    ):
        return False
    seen_ids: set[str] = set()
    summed_cost = 0
    for generation in metadata:
        if (
            not isinstance(generation, Mapping)
            or generation.get("reconciled") is not False
            or generation.get("accounting_basis") != accounting_basis
            or generation.get("billing_reconciliation_status") != "provider_charge_unavailable"
        ):
            return False
        generation_id = str(generation.get("generation_id") or "")
        generation_cost = generation.get("cost_micros")
        if (
            not generation_id
            or generation_id in seen_ids
            or not isinstance(generation_cost, int)
            or isinstance(generation_cost, bool)
            or generation_cost < 0
        ):
            return False
        seen_ids.add(generation_id)
        summed_cost += generation_cost
    return set(map(str, generation_ids)) == seen_ids and summed_cost == cost_micros


def _result_has_unpriced_qwencloud_usage(result: Mapping[str, Any]) -> bool:
    metadata = result.get("generation_metadata")
    generation_ids = result.get("generation_ids")
    if (
        result.get("cost_reconciled") is not False
        or result.get("cost_micros") != 0
        or result.get("cost_accounting_basis") != QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
        or result.get("billing_reconciliation_status") != QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
        or not isinstance(metadata, list)
        or not isinstance(generation_ids, list)
        or not generation_ids
    ):
        return False
    seen_ids: set[str] = set()
    for generation in metadata:
        if (
            not isinstance(generation, Mapping)
            or generation.get("reconciled") is not False
            or generation.get("cost_micros") != 0
            or generation.get("provider_cost_known") is not False
            or generation.get("accounting_basis") != QWENCLOUD_MUTABLE_ALIAS_ACCOUNTING_BASIS
            or generation.get("billing_reconciliation_status")
            != QWENCLOUD_MUTABLE_ALIAS_BILLING_STATUS
        ):
            return False
        generation_id = str(generation.get("generation_id") or "")
        if not generation_id or generation_id in seen_ids:
            return False
        seen_ids.add(generation_id)
    return set(map(str, generation_ids)) == seen_ids


def _result_accounting_is_accepted(
    result: Mapping[str, Any],
    candidate: ContractCandidate,
) -> bool:
    if candidate.cost_accounting_policy == "provider_generation_metadata":
        return _result_is_reconciled(result)
    if candidate.cost_accounting_policy == "provider_usage_times_frozen_rate_card":
        return _result_has_rate_card_accounting(result, candidate.execution_backend)
    if candidate.cost_accounting_policy == "provider_usage_with_unpriced_budget_ceiling":
        return (
            candidate.execution_backend == "qwencloud_direct"
            and candidate.backend_contract.get("identity_kind") == "mutable_alias"
            and _result_has_unpriced_qwencloud_usage(result)
        )
    return False


def _condition_has_identity_error(errors: Mapping[str, Any], condition: str) -> bool:
    return any(str(key) == condition or str(key).startswith(f"{condition}_") for key in errors)


def _response_payload(
    *,
    source: DatasetSource,
    work_item: WorkItem,
    condition: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = source.artifact
    result_payload = json.loads(json.dumps(result, ensure_ascii=False))
    cost_micros = int(result_payload["cost_micros"])
    provider_cost_exact = _result_is_reconciled(result_payload)
    rate_card_accounted = _result_has_rate_card_accounting(
        result_payload, work_item.candidate.execution_backend
    )
    unpriced_usage_accounted = _result_has_unpriced_qwencloud_usage(result_payload)
    source_mcp_events = artifact.get("mcp_trace_events") or []
    arm_suffix = f":{condition}"
    condition_mcp_events = [
        event
        for event in source_mcp_events
        if isinstance(event, Mapping) and str(event.get("arm_id") or "").endswith(arm_suffix)
    ]
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "artifact_type": "model_response",
        "collection_status": "exploratory_unranked_candidate_tasks",
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "research_release_eligible": False,
        "work_item_id": work_item.work_item_id,
        "manifest_sha256": work_item.manifest_sha256,
        "task_registry_sha256": work_item.task_registry_sha256,
        "execution_policy_sha256": work_item.execution_policy_sha256,
        "execution_policy": work_item.execution_policy.document(),
        "condition": condition,
        "task": {
            "public_id": work_item.task.public_id,
            "family": work_item.task.family,
            "prompt": work_item.task.prompt,
            "prompt_sha256": work_item.task.prompt_sha256,
            "split": work_item.task.split,
            "review_status": work_item.task.review_status,
        },
        "model": {
            "slot_id": work_item.candidate.slot_id,
            "requested_model_id": work_item.candidate.model_id,
            "canonical_model_slug": work_item.candidate.canonical_model_slug,
            "actual_model_id": result_payload.get("actual_model_id"),
            "provider_tag": work_item.candidate.provider_tag,
            "execution_backend": work_item.candidate.execution_backend,
            "backend_contract_sha256": work_item.candidate.backend_contract_sha256,
            "execution_policy_sha256": work_item.execution_policy_sha256,
            "actual_provider": result_payload.get("actual_provider"),
            "endpoint_manifest_sha256": work_item.candidate.endpoint_sha256,
            "endpoint_execution_sha256": work_item.endpoint_execution_sha256,
        },
        "source": {
            "artifact_sha256": source.artifact_sha256,
            "artifact_filename": source.path.name,
            "run_id": artifact.get("run_id"),
        },
        "provenance": {
            "model_contract": artifact.get("model_contract"),
            "endpoint_execution_contract": endpoint_execution_contract(
                dict(artifact.get("endpoint_contract") or {})
            ),
            "decoding": artifact.get("decoding"),
            "system_prompt_sha256": (artifact.get("system_prompt_sha256") or {}).get(condition),
            "response_schema_sha256": artifact.get("response_schema_sha256"),
            "epicure_access": condition == "epicure_on",
            "epicure": artifact.get("epicure"),
            "epicure_tool_schema_sha256": artifact.get("epicure_tool_schema_sha256"),
            "protocol_bundle": artifact.get("protocol_bundle"),
            "protocol_bundle_sha256": artifact.get("protocol_bundle_sha256"),
            "execution_route": dict(work_item.candidate.route_selection),
            "backend_contract": dict(work_item.candidate.backend_contract),
            "backend_contract_sha256": work_item.candidate.backend_contract_sha256,
            "mcp_trace_events": condition_mcp_events,
        },
        "cost": {
            "recorded_cost_micros": cost_micros,
            "actual_cost_micros": cost_micros if provider_cost_exact else None,
            "estimated_cost_micros": cost_micros if rate_card_accounted else None,
            "all_generation_costs_reconciled": provider_cost_exact,
            "all_generation_usage_accounted": (
                provider_cost_exact or rate_card_accounted or unpriced_usage_accounted
            ),
            "provider_cost_exact": provider_cost_exact,
            "cost_status": (
                "provider_reconciled_actual"
                if provider_cost_exact
                else "provider_usage_times_frozen_rate_card_estimate"
                if rate_card_accounted
                else "provider_cost_unavailable_full_budget_ceiling_retained"
            ),
            "exact_cost_ranking_eligible": provider_cost_exact,
            "provider_cost_known": provider_cost_exact,
            "full_budget_ceiling_retained_usd": (
                str(source.exposure.exposure_usd) if unpriced_usage_accounted else None
            ),
            "generation_ids": result_payload.get("generation_ids"),
            "generation_metadata": result_payload.get("generation_metadata"),
        },
        "response": result_payload,
        "limitations": [
            "This response is an unranked exploratory engineering record.",
            "The candidate task has not passed confirmatory human review.",
            "No public or expert preference judgment is attached to this response.",
            "This record is not an approved research release or an official benchmark result.",
            (
                "Epicure-on requires at least one successful real tool call."
                if work_item.execution_policy.epicure_on_tool_required
                else "Epicure-on denotes tool access; a model may choose not to call a tool."
            ),
            (
                "Cost is a provider-usage rate-card estimate, not a provider-reconciled charge."
                if rate_card_accounted and not provider_cost_exact
                else "Provider rate and charge are unavailable; recorded zero means unknown, "
                "not free, and the full admitted ceiling remains reserved."
                if unpriced_usage_accounted
                else "Cost was reconciled through provider generation metadata."
            ),
            *(
                [
                    "This model identity is a mutable alias pinned only at one authenticated "
                    "catalog observation, not a frozen model release."
                ]
                if unpriced_usage_accounted
                else []
            ),
        ],
    }


def _atomic_content_addressed_write(
    payload: Mapping[str, Any],
    *,
    destination: Path,
) -> Path:
    value = dict(payload)
    value["artifact_sha256"] = _sha256(value)
    digest = value["artifact_sha256"]
    expected = destination.with_name(f"{destination.name}-{digest}.json")
    expected.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if expected.exists():
        existing = json.loads(expected.read_text(encoding="utf-8"))
        if existing != value:
            raise IntegrityError(f"refusing to overwrite conflicting artifact: {expected}")
        return expected
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=expected.parent,
        prefix=f".{expected.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, expected)
    expected.chmod(0o644)
    directory_descriptor = os.open(expected.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return expected


def record_no_id_source_incident_resolution(
    *,
    ledger_path: str | Path,
    source_directory: str | Path,
    source_corrections_directory: str | Path | None,
    resolution_directory: str | Path,
    reservation_entry_sha256: str,
    incident_entry_sha256: str,
    affected_condition: str,
    confirmation: str,
) -> tuple[SourceIncidentResolution, Mapping[str, Any]]:
    """Append a no-replay conservative closure; this function has no network path."""

    if confirmation != SOURCE_INCIDENT_RESOLUTION_CONFIRMATION:
        raise AdmissionDenied(
            "source incident resolution requires --confirm "
            f"{SOURCE_INCIDENT_RESOLUTION_CONFIRMATION}"
        )
    ledger_file = Path(ledger_path)
    source_root = Path(source_directory)
    resolution_root = Path(resolution_directory)
    with _dataset_ledger_lock(ledger_file):
        ledger = load_dataset_ledger(ledger_file)
        reservations = {
            str(entry.get("entry_sha256") or ""): entry
            for entry in ledger
            if entry.get("event_type") == "reservation_created"
        }
        incidents = {
            str(entry.get("entry_sha256") or ""): entry
            for entry in ledger
            if entry.get("event_type") == "execution_incident"
        }
        reservation = reservations.get(reservation_entry_sha256)
        incident = incidents.get(incident_entry_sha256)
        if reservation is None or incident is None:
            raise IntegrityError("resolution reservation or incident entry is absent")
        work_item_id = str(reservation.get("work_item_id") or "")
        if incident.get("work_item_id") != work_item_id:
            raise IntegrityError("resolution reservation and incident work items differ")
        _, finalizations = dataset_ledger_state(ledger)
        if work_item_id in finalizations:
            raise IntegrityError("cannot resolve an already finalized dataset work item")
        source_digest = str(incident.get("source_artifact_sha256") or "")
        source_scan = scan_live_smoke_artifacts(
            source_root,
            corrections_directory=source_corrections_directory,
        )
        exposure = next(
            (item for item in source_scan.artifacts if item.artifact_sha256 == source_digest),
            None,
        )
        source_path = next(
            (
                path
                for path in sorted(source_root.glob("*.json"))
                if path.stem.endswith(source_digest[:12])
            ),
            None,
        )
        if exposure is None or source_path is None:
            raise IntegrityError("incident source artifact is absent")
        artifact, verified_digest = _verify_live_artifact(source_path)
        if verified_digest != source_digest or artifact.get("dataset_work_item_id") != work_item_id:
            raise IntegrityError("incident source digest/work-item identity mismatch")
        source = DatasetSource(
            path=source_path,
            artifact_sha256=source_digest,
            work_item_id=work_item_id,
            artifact=artifact,
            exposure=exposure,
        )
        existing_events = dataset_incident_resolution_state(ledger)
        existing_event = existing_events.get(work_item_id)
        if existing_event is not None:
            if (
                existing_event.get("reservation_entry_sha256") != reservation_entry_sha256
                or existing_event.get("incident_entry_sha256") != incident_entry_sha256
            ):
                raise IntegrityError("work item already has a different incident resolution")
            existing_path = resolution_root / str(existing_event.get("resolution_filename"))
            return (
                _verify_source_incident_resolution(
                    existing_path,
                    source=source,
                    reservation=reservation,
                    incident=incident,
                    ledger_event=existing_event,
                ),
                existing_event,
            )
        payload = build_source_incident_resolution_payload(
            source=source,
            reservation=reservation,
            incident=incident,
            affected_condition=affected_condition,
        )
        resolution_path = _atomic_content_addressed_write(
            payload,
            destination=resolution_root / "source-incident-resolution",
        )
        resolution_document = _read_json_regular(
            resolution_path,
            label="source incident resolution",
        )
        resolution_digest = str(resolution_document["artifact_sha256"])
        event = append_dataset_ledger_event(
            ledger_file,
            {
                "event_type": SOURCE_INCIDENT_RESOLUTION_EVENT_TYPE,
                "resolution_operation_id": _sha256(
                    {
                        "schema_version": SOURCE_INCIDENT_RESOLUTION_SCHEMA_VERSION,
                        "reservation_entry_sha256": reservation_entry_sha256,
                        "incident_entry_sha256": incident_entry_sha256,
                        "source_artifact_sha256": source_digest,
                    }
                ),
                "work_item_id": work_item_id,
                "reservation_entry_sha256": reservation_entry_sha256,
                "incident_entry_sha256": incident_entry_sha256,
                "source_artifact_sha256": source_digest,
                "resolution_filename": resolution_path.name,
                "resolution_artifact_sha256": resolution_digest,
                "affected_condition": affected_condition,
                "provider_reconciled_actual_cost_usd": payload["cost"][
                    "provider_reconciled_actual_cost_usd"
                ],
                "conservative_budget_exposure_usd": payload["cost"][
                    "conservative_budget_exposure_usd"
                ],
                "provider_cost_exact_for_unidentified_response": False,
                "safe_to_replay": False,
                "normalizable_conditions": payload["resolution"]["normalizable_conditions"],
            },
        )
        return (
            _verify_source_incident_resolution(
                resolution_path,
                source=source,
                reservation=reservation,
                incident=incident,
                ledger_event=event,
            ),
            event,
        )


def normalise_source_responses(
    source: DatasetSource,
    work_item: WorkItem,
    *,
    response_directory: str | Path,
    expected_conditions: Sequence[str] = CONDITIONS,
    expected_epicure: Mapping[str, str] | None = None,
) -> tuple[list[ResponseArtifact], list[str]]:
    """Write identity-clean responses with accepted, explicitly typed accounting."""

    selected_conditions = tuple(expected_conditions)
    postflight_issues = _source_postflight_issues(
        source,
        work_item,
        expected_conditions=selected_conditions,
        expected_epicure=expected_epicure,
    )
    if postflight_issues:
        return [], postflight_issues
    artifact = source.artifact
    results = artifact.get("results") or {}
    errors = artifact.get("errors") or {}
    if not isinstance(results, Mapping) or not isinstance(errors, Mapping):
        return [], ["invalid_source_results_or_errors"]
    written: list[ResponseArtifact] = []
    issues: list[str] = []
    for condition in selected_conditions:
        result = results.get(condition)
        if not isinstance(result, Mapping):
            issues.append(f"{condition}_missing_or_invalid")
            continue
        if _condition_has_identity_error(errors, condition):
            issues.append(f"{condition}_identity_or_generation_error")
            continue
        if not _result_accounting_is_accepted(result, work_item.candidate):
            suffix = (
                "cost_not_fully_reconciled"
                if work_item.candidate.cost_accounting_policy == "provider_generation_metadata"
                else "cost_accounting_not_accepted"
            )
            issues.append(f"{condition}_{suffix}")
            continue
        tool_trace = result.get("tool_trace") or []
        if condition == "epicure_off" and tool_trace:
            issues.append("epicure_off_unexpected_tool_trace")
            continue
        base = Path(response_directory) / (f"response-{work_item.work_item_id}-{condition}")
        path = _atomic_content_addressed_write(
            _response_payload(
                source=source,
                work_item=work_item,
                condition=condition,
                result=result,
            ),
            destination=base,
        )
        written.append(_verify_response_artifact(path))
    return written, issues


def _subprocess_command(
    work_item: WorkItem,
    *,
    forecast: PairForecast,
    source_directory: Path,
    manifest_path: Path,
    conditions: Sequence[str] = CONDITIONS,
    expected_epicure: Mapping[str, str] | None = None,
) -> list[str]:
    selected_conditions = tuple(conditions)
    if (
        not selected_conditions
        or len(set(selected_conditions)) != len(selected_conditions)
        or not set(selected_conditions) <= set(CONDITIONS)
    ):
        raise IntegrityError("subprocess requires a unique non-empty condition subset")
    if work_item.candidate.execution_backend == "openrouter":
        module = "flavourbench.live_smoke"
    elif work_item.candidate.execution_backend == "kimi_direct":
        module = "flavourbench.direct_kimi_pair"
    elif work_item.candidate.execution_backend == "cohere_direct":
        module = "flavourbench.direct_cohere_pair"
    elif work_item.candidate.execution_backend == "qwencloud_direct":
        module = "flavourbench.direct_qwencloud_pair"
    else:
        raise IntegrityError(
            "the real dataset runner has no admitted subprocess for "
            f"{work_item.candidate.execution_backend}"
        )
    command = [
        sys.executable,
        "-m",
        module,
        "--confirm",
        LIVE_SMOKE_CONFIRMATION,
        "--cap-usd",
        _decimal_text(forecast.forecast_usd),
        "--model-id",
        work_item.candidate.model_id,
        "--provider-slug",
        work_item.candidate.provider_tag,
        "--prompt",
        work_item.task.prompt,
        "--category",
        work_item.task.family,
        "--output-dir",
        str(source_directory.resolve()),
        "--candidate-manifest-sha256",
        work_item.manifest_sha256,
        "--dataset-work-item-id",
        work_item.work_item_id,
        "--dataset-task-id",
        work_item.task.public_id,
        "--expected-canonical-model-slug",
        work_item.candidate.canonical_model_slug,
        "--expected-endpoint-execution-sha256",
        work_item.endpoint_execution_sha256,
        "--expected-execution-policy-sha256",
        work_item.execution_policy_sha256,
    ]
    if work_item.candidate.execution_backend == "openrouter":
        command.append("--skip-tool-contract")
    else:
        command.extend(["--route-manifest", str(manifest_path.resolve())])
    if (
        work_item.candidate.execution_backend == "qwencloud_direct"
        and work_item.candidate.backend_contract.get("identity_kind") == "mutable_alias"
    ):
        command.append("--allow-mutable-alias-exploratory")
    if work_item.execution_policy.final_response_mode == "plain_text":
        command.append("--plain-text-final")
        command.extend(
            [
                "--tool-catalog-bytes-bound",
                str(work_item.execution_policy.tool_catalog_bytes_bound),
            ]
        )
    command.extend(["--evidence-protocol", work_item.execution_policy.evidence_protocol])
    if work_item.execution_policy.epicure_on_tool_required:
        command.append("--require-epicure-call")
    if work_item.execution_policy.intermediate_reasoning_effort is not None:
        command.extend(
            [
                "--intermediate-reasoning-effort",
                work_item.execution_policy.intermediate_reasoning_effort,
            ]
        )
    if work_item.execution_policy.final_reasoning_effort is not None:
        command.extend(
            ["--final-reasoning-effort", work_item.execution_policy.final_reasoning_effort]
        )
    if selected_conditions != CONDITIONS:
        for condition in selected_conditions:
            command.extend(["--condition", condition])
    if expected_epicure is not None:
        fields = {
            "release_id": "--expected-epicure-release-id",
            "bundle_sha256": "--expected-epicure-bundle-sha256",
            "application_sha256": "--expected-epicure-application-sha256",
            "tool_schema_sha256": "--expected-epicure-tool-schema-sha256",
        }
        for field, option in fields.items():
            value = str(expected_epicure.get(field) or "")
            if not value:
                raise IntegrityError(f"coverage subprocess has no expected Epicure {field}")
            command.extend([option, value])
    return command


def _work_item_map(work_items: Sequence[WorkItem]) -> dict[str, WorkItem]:
    mapped = {item.work_item_id: item for item in work_items}
    if len(mapped) != len(work_items):
        raise IntegrityError("workload contains duplicate work-item identities")
    return mapped


def _apply_v2_route_validation_override(
    *,
    work_items: Sequence[WorkItem],
    plan_path: str | Path,
    variant_id: str,
) -> tuple[list[WorkItem], dict[str, Any]]:
    """Bind one runner cell to a fresh v2 or v3 diagnostic work-item ID."""

    path = Path(plan_path)
    plan = _read_json_regular(path, label="route-validation plan")
    digest = str(plan.get("artifact_sha256") or "")
    unhashed = {key: value for key, value in plan.items() if key != "artifact_sha256"}
    schema_version = plan.get("schema_version")
    if (
        schema_version not in {V2_ROUTE_PLAN_SCHEMA_VERSION, V3_ROUTE_PLAN_SCHEMA_VERSION}
        or len(digest) != 64
        or _sha256(unhashed) != digest
    ):
        raise IntegrityError("route-validation plan content address does not verify")
    route_revision = "v3" if schema_version == V3_ROUTE_PLAN_SCHEMA_VERSION else "v2"
    preflight = plan.get("preflight")
    route = plan.get("route_validation")
    envelope = plan.get("safe_response_envelope_contract")
    source = plan.get("source")
    if not all(isinstance(value, Mapping) for value in (preflight, route, envelope, source)):
        raise IntegrityError("route-validation plan is incomplete")
    assert isinstance(preflight, Mapping)
    assert isinstance(route, Mapping)
    assert isinstance(envelope, Mapping)
    assert isinstance(source, Mapping)
    if (
        preflight.get("decision") != f"ready_to_materialize_{route_revision}_route_validation_only"
        or preflight.get("collection_blockers") != []
        or preflight.get("provider_calls_made") is not False
        or preflight.get("epicure_calls_made") is not False
        or route.get("matched_pairs") != 3
        or route.get("response_arms") != 6
        or route.get("diagnostic_outputs_enter_quality_fit") is not False
    ):
        raise IntegrityError("route-validation plan is not admitted for diagnostic execution")
    provider_source = Path(__file__).with_name("provider.py")
    expected_action = (
        "retry_allowlisted_error_envelopes_without_generation_or_cost_reconciliation"
        if route_revision == "v3"
        else "fail_closed_no_automatic_retry"
    )
    envelope_valid = (
        source.get("provider_source_sha256") == _sha256_file(provider_source)
        and envelope.get("accepted_classification") == "chat_completions"
        and envelope.get("http_200_non_chat_action") == expected_action
    )
    if route_revision == "v3":
        envelope_valid = envelope_valid and envelope.get("retryable_error_codes") == [
            408,
            429,
            502,
            503,
        ]
    if not envelope_valid:
        raise IntegrityError("route plan differs from the installed provider classifier")
    planned_items = route.get("work_items")
    if not isinstance(planned_items, list):
        raise IntegrityError("route plan has no work items")
    planned = [
        item
        for item in planned_items
        if isinstance(item, Mapping) and item.get("variant_id") == variant_id
    ]
    if len(planned) != 1:
        raise IntegrityError("route variant does not identify exactly one planned pair")
    item = planned[0]
    planned_id = str(item.get("work_item_id") or "")
    if (
        len(planned_id) != 64
        or planned_id
        in set(
            plan.get("closed_work_item_ids_never_replayed")
            or plan.get("v1_work_item_ids_never_replayed")
            or []
        )
        or item.get("diagnostic_only") is not True
        or item.get("official") is not False
        or item.get("rank_eligible") is not False
    ):
        raise IntegrityError("route work-item identity is not fresh and diagnostic")
    matching = [
        candidate
        for candidate in work_items
        if candidate.candidate.model_id == item.get("model_id")
        and candidate.task.public_id == item.get("task_id")
    ]
    if len(matching) != 1:
        raise IntegrityError("route work item does not map to one runner cell")
    original = matching[0]
    exact = {
        "canonical_model_slug": original.candidate.canonical_model_slug,
        "provider_endpoint": original.candidate.provider_tag,
        "endpoint_execution_sha256": original.endpoint_execution_sha256,
        "task_family": original.task.family,
        "prompt_sha256": original.task.prompt_sha256,
        "execution_policy_sha256": original.execution_policy_sha256,
    }
    if any(item.get(field) != value for field, value in exact.items()):
        raise IntegrityError("route work item differs from the derived runner contract")
    selected = dataclasses.replace(original, ordinal=1, work_item_id=planned_id)
    return [selected], {
        "plan_filename": path.name,
        "plan_sha256": digest,
        "route_revision": route_revision,
        "route_cell_id": route.get("route_cell_id"),
        "variant_id": variant_id,
        "derived_runner_work_item_id": original.work_item_id,
        "effective_fresh_work_item_id": planned_id,
        "model_id": original.candidate.model_id,
        "task_id": original.task.public_id,
        "diagnostic_only": True,
        "quality_fit_eligible": False,
    }


def _validate_state_against_workload(
    state: DatasetState,
    work_items: Sequence[WorkItem],
) -> None:
    known = _work_item_map(work_items)
    ledger_work_items = set(state.reservations)
    source_work_items = set(state.sources)
    response_work_items = {key[0] for key in state.responses}
    unknown = (ledger_work_items | source_work_items | response_work_items) - set(known)
    if unknown:
        raise IntegrityError(
            "existing dataset state is outside this exact manifest/task workload: "
            f"{sorted(unknown)[:3]}"
        )
    for work_item_id, source in state.sources.items():
        issues = _source_postflight_issues(source, known[work_item_id])
        if issues:
            raise IntegrityError(
                f"source artifact {source.path} fails frozen workload checks: {issues}"
            )
    for (work_item_id, condition), response in state.responses.items():
        work_item = known[work_item_id]
        if (
            response.task_id != work_item.task.public_id
            or response.task_family != work_item.task.family
            or response.model_id != work_item.candidate.model_id
            or response.provider_tag != work_item.candidate.provider_tag
            or response.condition != condition
        ):
            raise IntegrityError(f"response artifact disagrees with workload: {response.path}")


def build_dataset_plan(
    work_items: Sequence[WorkItem],
    *,
    state: DatasetState,
    policy: ExecutionPolicy,
    cap_usd: Decimal,
    admission_fraction: Decimal,
) -> list[dict[str, Any]]:
    exposure = state.total_exposure_usd
    ceiling = cap_usd * admission_fraction
    forecasts = {
        work_item.work_item_id: derive_pair_forecast(work_item, policy=policy)
        for work_item in work_items
    }
    blockers: dict[str, str] = {}
    pending: list[WorkItem] = []
    for work_item in work_items:
        work_item_id = work_item.work_item_id
        if work_item_id in state.finalizations:
            continue
        if work_item_id in state.reservations:
            source = state.sources.get(work_item_id)
            if source is None:
                blockers[work_item_id] = "block_active_reservation_without_source"
            elif source.exposure.exposure_basis == "failed_or_unreconciled_full_admitted_allowance":
                blockers[work_item_id] = "block_existing_source_needs_cost_reconciliation"
            continue
        if work_item_id in state.sources:
            raise IntegrityError(f"dataset source has no governed reservation: {work_item_id}")
        pending.append(work_item)
    required_reserve = sum(
        (forecasts[item.work_item_id].forecast_usd for item in pending),
        Decimal(0),
    )
    complete_projected = exposure + required_reserve
    budget_block: str | None = None
    if complete_projected > cap_usd:
        budget_block = "block_complete_workload_hard_cap"
    elif complete_projected > ceiling:
        budget_block = "block_complete_workload_85_percent_admission_ceiling"
    plan: list[dict[str, Any]] = []
    for work_item in work_items:
        forecast = forecasts[work_item.work_item_id]
        base = {**work_item.public_payload(), "forecast": forecast.public_payload()}
        if work_item.work_item_id in state.finalizations:
            plan.append(
                {
                    **base,
                    "decision": "skip_finalized_work_item",
                    "source_artifact_sha256": state.sources[work_item.work_item_id].artifact_sha256,
                }
            )
            continue
        if work_item.work_item_id in state.reservations:
            decision = blockers.get(
                work_item.work_item_id,
                "recover_existing_reconciled_source",
            )
            plan.append({**base, "decision": decision})
            continue
        if blockers or budget_block:
            plan.append(
                {
                    **base,
                    "decision": (budget_block or "block_complete_workload_due_active_incident"),
                    "complete_workload_exposure_before_usd": _decimal_text(exposure),
                    "complete_workload_required_reserve_usd": _decimal_text(required_reserve),
                    "complete_workload_projected_exposure_usd": _decimal_text(complete_projected),
                }
            )
            continue
        projected = exposure + forecast.forecast_usd
        plan.append(
            {
                **base,
                "decision": "admit_sequentially",
                "exposure_before_usd": _decimal_text(exposure),
                "projected_exposure_usd": _decimal_text(projected),
            }
        )
        exposure = projected
    return plan


def _finalize_source(
    *,
    ledger_path: Path,
    runner_run_id: str,
    reservation: Mapping[str, Any],
    source: DatasetSource,
    work_item: WorkItem,
    response_directory: Path,
    incident_resolution: SourceIncidentResolution | None = None,
    unresolved_incident: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[ResponseArtifact], list[str]]:
    unresolved_basis = (
        source.exposure.exposure_basis == "failed_or_unreconciled_full_admitted_allowance"
    )
    if unresolved_basis:
        if (
            unresolved_incident is None
            or unresolved_incident.get("event_type") != "execution_incident"
            or unresolved_incident.get("incident")
            != "generation_cost_unreconciled_reservation_retained"
            or unresolved_incident.get("work_item_id") != work_item.work_item_id
            or unresolved_incident.get("reservation_entry_sha256")
            != reservation.get("entry_sha256")
            or unresolved_incident.get("source_artifact_sha256") != source.artifact_sha256
        ):
            raise IntegrityError(
                "unresolved source finalization requires its exact immutable incident"
            )
    elif unresolved_incident is not None:
        raise IntegrityError("unresolved incident supplied for an accounted source")
    if source.exposure.exposure_basis == RESOLVED_CONSERVATIVE_EXPOSURE_BASIS:
        if incident_resolution is None:
            raise IntegrityError(
                "cannot finalize conservative exposure without its incident resolution"
            )
        if (
            incident_resolution.work_item_id != work_item.work_item_id
            or incident_resolution.source_artifact_sha256 != source.artifact_sha256
            or incident_resolution.reservation_entry_sha256 != reservation.get("entry_sha256")
            or incident_resolution.provider_reconciled_actual_cost_usd
            != source.exposure.actual_cost_usd
            or incident_resolution.conservative_budget_exposure_usd != source.exposure.exposure_usd
        ):
            raise IntegrityError("incident resolution does not match finalization inputs")
    elif incident_resolution is not None:
        raise IntegrityError("incident resolution supplied for a non-resolved source")
    responses, issues = normalise_source_responses(
        source,
        work_item,
        response_directory=response_directory,
    )
    if any(
        issue.endswith(("cost_not_fully_reconciled", "cost_accounting_not_accepted"))
        for issue in issues
    ):
        raise IntegrityError("cannot finalize a response with rejected generation accounting")
    if incident_resolution is not None and tuple(
        sorted(response.condition for response in responses)
    ) != tuple(sorted(incident_resolution.normalizable_conditions)):
        raise IntegrityError("normalized response conditions differ from the incident resolution")
    provider_cost_exact = (
        incident_resolution is None
        and unresolved_incident is None
        and work_item.candidate.cost_accounting_policy == "provider_generation_metadata"
    )
    recorded = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "source_artifact_recorded",
            "runner_run_id": runner_run_id,
            "reservation_entry_sha256": reservation["entry_sha256"],
            "work_item_id": work_item.work_item_id,
            "manifest_sha256": work_item.manifest_sha256,
            "task_id": work_item.task.public_id,
            "task_family": work_item.task.family,
            "model_id": work_item.candidate.model_id,
            "provider_tag": work_item.candidate.provider_tag,
            "execution_backend": work_item.candidate.execution_backend,
            "backend_contract_sha256": work_item.candidate.backend_contract_sha256,
            "execution_policy_sha256": work_item.execution_policy_sha256,
            "source_artifact_filename": source.path.name,
            "source_artifact_sha256": source.artifact_sha256,
            "source_status": source.exposure.status,
            "source_actual_cost_usd": _decimal_text(source.exposure.actual_cost_usd),
            "provider_reconciled_actual_cost_usd": (
                _decimal_text(source.exposure.actual_cost_usd) if provider_cost_exact else None
            ),
            "rate_card_estimated_cost_usd": (
                _decimal_text(source.exposure.actual_cost_usd)
                if not provider_cost_exact
                and unresolved_incident is None
                and work_item.candidate.cost_accounting_policy != "provider_generation_metadata"
                else None
            ),
            "known_provider_reconciled_cost_usd": (
                _decimal_text(source.exposure.actual_cost_usd)
                if unresolved_incident is not None
                else None
            ),
            "source_budget_exposure_usd": _decimal_text(source.exposure.exposure_usd),
            "source_exposure_basis": source.exposure.exposure_basis,
            "all_generation_costs_reconciled": provider_cost_exact,
            "provider_cost_exact": provider_cost_exact,
            "response_artifact_sha256s": sorted(response.artifact_sha256 for response in responses),
            "response_conditions": sorted(response.condition for response in responses),
            "normalization_issues": issues,
            **(
                {
                    "source_incident_resolution_sha256": (incident_resolution.artifact_sha256),
                    "source_incident_resolution_ledger_entry_sha256": (
                        incident_resolution.ledger_event_sha256
                    ),
                    "unidentified_response_generation_id_known": False,
                    "unidentified_response_provider_cost_exact": False,
                    "conservative_full_admitted_allowance_usd": _decimal_text(
                        incident_resolution.conservative_budget_exposure_usd
                    ),
                }
                if incident_resolution is not None
                else {}
            ),
            **(
                {
                    "unresolved_delivery_incident_entry_sha256": unresolved_incident[
                        "entry_sha256"
                    ],
                    "unresolved_full_admitted_allowance_retained": True,
                    "safe_to_replay": False,
                }
                if unresolved_incident is not None
                else {}
            ),
        },
    )
    return recorded, responses, issues


def _summary_coverage(work_items: Sequence[WorkItem], state: DatasetState) -> dict[str, Any]:
    overall = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for work_item in work_items:
        source = state.sources.get(work_item.work_item_id)
        off = state.responses.get((work_item.work_item_id, "epicure_off"))
        on = state.responses.get((work_item.work_item_id, "epicure_on"))
        finalized = work_item.work_item_id in state.finalizations
        metrics = {
            "expected_pairs": 1,
            "expected_arms": 2,
            "source_attempts": int(source is not None),
            "finalized_pairs": int(finalized),
            "epicure_off_responses": int(off is not None),
            "epicure_on_responses": int(on is not None),
            "complete_pairs": int(off is not None and on is not None),
            "failed_or_partial_pairs": int(finalized and (off is None or on is None)),
            "epicure_on_tool_used": int(on is not None and on.tool_used),
        }
        overall.update(metrics)
        by_family[work_item.task.family].update(metrics)
        by_model[work_item.candidate.model_id].update(metrics)
    return {
        "overall": dict(overall),
        "by_task_family": {family: dict(by_family[family]) for family in TASK_FAMILIES},
        "by_model": {model_id: dict(metrics) for model_id, metrics in sorted(by_model.items())},
    }


def _write_summary(summary: Mapping[str, Any], directory: str | Path) -> Path:
    payload = dict(summary)
    digest = _sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"real-exploratory-summary-{digest}.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise IntegrityError(f"refusing to overwrite conflicting summary: {destination}")
        return destination
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=root,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run_real_exploratory_dataset(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str = CURRENT_DATASET_MANIFEST_SHA256,
    selectors: Iterable[str] = (),
    task_pool_per_family: int = 3,
    assignments_per_model: int = 10,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    task_validity_artifact_path: str | Path | None = None,
    epicure_native_task_artifact_path: str | Path | None = None,
    protocol_preflight_registry_path: str | Path | None = None,
    prior_artifact_directory: str | Path = "artifacts/live-smoke",
    prior_corrections_directory: str | Path | None = "artifacts/corrections",
    source_directory: str | Path = "artifacts/real-exploratory/source-runs",
    source_corrections_directory: str | Path | None = ("artifacts/real-exploratory/corrections"),
    source_resolution_directory: str | Path | None = None,
    response_directory: str | Path = "artifacts/real-exploratory/responses",
    ledger_path: str | Path = "artifacts/real-exploratory/ledger.jsonl",
    global_budget_lock_path: str | Path = "artifacts/frontier-contract/ledger.jsonl",
    summary_directory: str | Path = "artifacts/real-exploratory/summaries",
    cap_usd: Decimal = AUTHORIZED_TOTAL_CAP_USD,
    admission_fraction: Decimal = DEFAULT_ADMISSION_FRACTION,
    execution_policy: ExecutionPolicy | None = None,
    execute: bool = False,
    finalize_existing_only: bool = False,
    finalize_unresolved_existing_only: bool = False,
    confirmation: str = "",
    process_timeout_seconds: int = 3_600,
    max_new_pairs: int | None = None,
    route_validation_plan_path: str | Path | None = None,
    route_validation_variant_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Plan or execute a balanced exact-endpoint exploratory real-data block."""

    if cap_usd <= 0 or cap_usd > AUTHORIZED_TOTAL_CAP_USD:
        raise AdmissionDenied(
            f"cap must be positive and at most the authorised ${AUTHORIZED_TOTAL_CAP_USD}"
        )
    if admission_fraction <= 0 or admission_fraction > DEFAULT_ADMISSION_FRACTION:
        raise AdmissionDenied(f"admission fraction must be in (0, {DEFAULT_ADMISSION_FRACTION}]")
    if execute and confirmation != EXECUTION_CONFIRMATION:
        raise AdmissionDenied(
            f"execution requires --confirm {EXECUTION_CONFIRMATION}; dry-run needs no confirmation"
        )
    if execute and (finalize_existing_only or finalize_unresolved_existing_only):
        raise AdmissionDenied("paid execution and finalize-existing-only are mutually exclusive")
    if finalize_existing_only and finalize_unresolved_existing_only:
        raise AdmissionDenied("choose one existing-source finalization mode")
    if finalize_existing_only and confirmation != FINALIZE_EXISTING_CONFIRMATION:
        raise AdmissionDenied(
            f"finalizing existing sources requires --confirm {FINALIZE_EXISTING_CONFIRMATION}"
        )
    if finalize_unresolved_existing_only and confirmation != FINALIZE_UNRESOLVED_CONFIRMATION:
        raise AdmissionDenied(
            f"finalizing an unresolved source requires --confirm {FINALIZE_UNRESOLVED_CONFIRMATION}"
        )
    if process_timeout_seconds <= 0:
        raise ValueError("process timeout must be positive")
    if max_new_pairs is not None and max_new_pairs <= 0:
        raise ValueError("max_new_pairs must be positive when provided")
    manifest = load_candidate_manifest(
        manifest_path,
        expected_digest=expected_manifest_sha256,
    )
    protocol_preflight_registry: dict[str, Any] | None = None
    if manifest.get("manifest_role") == "current_frontier_real_development_quality_run":
        if protocol_preflight_registry_path is None:
            raise IntegrityError(
                "current development collection requires a promoted manifest and its "
                "exact matched-protocol preflight registry"
            )
        try:
            protocol_preflight_registry = verify_registry_for_manifest(
                registry_path=Path(protocol_preflight_registry_path),
                manifest=manifest,
            )
        except ProtocolPreflightError as error:
            raise IntegrityError(str(error)) from error
    candidates = select_candidates(manifest, selectors)
    if task_validity_artifact_path is not None and epicure_native_task_artifact_path is not None:
        raise IntegrityError(
            "choose either the human-authored development tasks or the Epicure-native tasks"
        )
    if epicure_native_task_artifact_path is not None:
        task_inventory, task_source = load_epicure_native_task_inventory(
            epicure_native_task_artifact_path
        )
    elif task_validity_artifact_path is None:
        task_inventory = candidate_tasks()
        task_source: dict[str, Any] = {
            "source_class": "legacy_hand_authored_candidate_registry",
            "synthetic_tasks": None,
            "confirmatory_eligible": False,
            "rank_eligible": False,
        }
    else:
        task_inventory, task_source = load_development_task_inventory(task_validity_artifact_path)
    selected_tasks, registry_sha = select_balanced_tasks(
        tasks_per_family=task_pool_per_family,
        seed=selection_seed,
        tasks=task_inventory,
    )
    manifest_sha = str(manifest["content_address"]["digest"])
    policy = execution_policy or ExecutionPolicy()
    policy.validate()
    work_items = build_balanced_work_items(
        manifest_sha256=manifest_sha,
        task_registry_digest=registry_sha,
        selected_tasks=selected_tasks,
        candidates=candidates,
        execution_policy=policy,
        assignments_per_model=assignments_per_model,
    )
    validate_current_development_run_binding(
        manifest=manifest,
        task_source=task_source,
        task_inventory=task_inventory,
        selected_tasks=selected_tasks,
        candidates=candidates,
        work_items=work_items,
        task_pool_per_family=task_pool_per_family,
        assignments_per_model=assignments_per_model,
        selection_seed=selection_seed,
        execution_policy=policy,
    )
    route_validation_override: dict[str, Any] | None = None
    if (route_validation_plan_path is None) != (route_validation_variant_id is None):
        raise IntegrityError("route validation requires both a plan and an effort variant")
    if route_validation_plan_path is not None and route_validation_variant_id is not None:
        work_items, route_validation_override = _apply_v2_route_validation_override(
            work_items=work_items,
            plan_path=route_validation_plan_path,
            variant_id=route_validation_variant_id,
        )
        selected_tasks = [work_items[0].task]
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    ledger = Path(ledger_path)
    source_root = Path(source_directory)
    resolution_root = (
        Path(source_resolution_directory)
        if source_resolution_directory is not None
        else source_root.parent / "resolutions"
    )
    response_root = Path(response_directory)
    outcomes: list[dict[str, Any]] = []
    subprocesses_started = 0

    # Reuse the frontier contract runner's lock path as a global OpenRouter
    # budget mutex, then take this runner's own ledger lock.  This prevents the
    # two governed runners from admitting against the same USD 100 concurrently.
    with _exclusive_runner_lock(Path(global_budget_lock_path)):
        with _dataset_ledger_lock(ledger):
            initial_state = _load_state(
                prior_artifact_directory=prior_artifact_directory,
                prior_corrections_directory=prior_corrections_directory,
                prior_reservation_ledger_path=global_budget_lock_path,
                source_directory=source_root,
                source_corrections_directory=source_corrections_directory,
                source_resolution_directory=resolution_root,
                response_directory=response_root,
                ledger_path=ledger,
            )
            _validate_state_against_workload(initial_state, work_items)
            plan = build_dataset_plan(
                work_items,
                state=initial_state,
                policy=policy,
                cap_usd=cap_usd,
                admission_fraction=admission_fraction,
            )
            plan_blocked = any(
                str(item.get("decision") or "").startswith("block_") for item in plan
            )
            target_forecast_usd = sum(
                (derive_pair_forecast(item, policy=policy).forecast_usd for item in work_items),
                Decimal(0),
            )
            outstanding_forecast_usd = sum(
                (
                    derive_pair_forecast(item, policy=policy).forecast_usd
                    for item in work_items
                    if item.work_item_id not in initial_state.finalizations
                    and item.work_item_id not in initial_state.reservations
                ),
                Decimal(0),
            )
            initial_complete_block_projected_usd = (
                initial_state.total_exposure_usd + outstanding_forecast_usd
            )
            finalize_only = finalize_existing_only or finalize_unresolved_existing_only
            if (not execute and not finalize_only) or (execute and plan_blocked):
                outcomes = plan
            else:
                source_root.mkdir(parents=True, exist_ok=True)
                response_root.mkdir(parents=True, exist_ok=True)
                for work_item in work_items:
                    state = _load_state(
                        prior_artifact_directory=prior_artifact_directory,
                        prior_corrections_directory=prior_corrections_directory,
                        prior_reservation_ledger_path=global_budget_lock_path,
                        source_directory=source_root,
                        source_corrections_directory=source_corrections_directory,
                        source_resolution_directory=resolution_root,
                        response_directory=response_root,
                        ledger_path=ledger,
                    )
                    _validate_state_against_workload(state, work_items)
                    forecast = derive_pair_forecast(work_item, policy=policy)
                    base = {**work_item.public_payload(), "forecast": forecast.public_payload()}
                    if work_item.work_item_id in state.finalizations:
                        outcomes.append({**base, "decision": "skip_finalized_work_item"})
                        continue
                    reservation = state.reservations.get(work_item.work_item_id)
                    source = state.sources.get(work_item.work_item_id)
                    if reservation is not None:
                        if source is None:
                            outcomes.append(
                                {
                                    **base,
                                    "decision": "stop_active_reservation_without_source",
                                    "reservation_entry_sha256": reservation["entry_sha256"],
                                }
                            )
                            break
                        if (
                            source.exposure.exposure_basis
                            == "failed_or_unreconciled_full_admitted_allowance"
                        ):
                            if not finalize_unresolved_existing_only:
                                outcomes.append(
                                    {
                                        **base,
                                        "decision": (
                                            "stop_existing_source_needs_cost_reconciliation"
                                        ),
                                        "source_artifact_sha256": source.artifact_sha256,
                                    }
                                )
                                break
                            incidents = [
                                entry
                                for entry in state.ledger
                                if entry.get("event_type") == "execution_incident"
                                and entry.get("incident")
                                == "generation_cost_unreconciled_reservation_retained"
                                and entry.get("work_item_id") == work_item.work_item_id
                                and entry.get("reservation_entry_sha256")
                                == reservation.get("entry_sha256")
                                and entry.get("source_artifact_sha256") == source.artifact_sha256
                            ]
                            if len(incidents) != 1:
                                raise IntegrityError(
                                    "unresolved source lacks one exact immutable incident"
                                )
                            recorded, responses, issues = _finalize_source(
                                ledger_path=ledger,
                                runner_run_id=run_id,
                                reservation=reservation,
                                source=source,
                                work_item=work_item,
                                response_directory=response_root,
                                unresolved_incident=incidents[0],
                            )
                            outcomes.append(
                                {
                                    **base,
                                    "decision": (
                                        "quarantined_unresolved_source_with_full_allowance"
                                    ),
                                    "source_artifact_sha256": source.artifact_sha256,
                                    "ledger_entry_sha256": recorded["entry_sha256"],
                                    "response_artifact_sha256s": sorted(
                                        response.artifact_sha256 for response in responses
                                    ),
                                    "normalization_issues": issues,
                                    "safe_to_replay": False,
                                }
                            )
                            continue
                        recorded, responses, issues = _finalize_source(
                            ledger_path=ledger,
                            runner_run_id=run_id,
                            reservation=reservation,
                            source=source,
                            work_item=work_item,
                            response_directory=response_root,
                            incident_resolution=state.incident_resolutions.get(
                                work_item.work_item_id
                            ),
                        )
                        outcomes.append(
                            {
                                **base,
                                "decision": "recovered_existing_source_without_provider_call",
                                "source_artifact_sha256": source.artifact_sha256,
                                "ledger_entry_sha256": recorded["entry_sha256"],
                                "response_artifact_sha256s": sorted(
                                    response.artifact_sha256 for response in responses
                                ),
                                "normalization_issues": issues,
                            }
                        )
                        continue
                    if source is not None:
                        raise IntegrityError(
                            f"source exists without governed reservation: {work_item.work_item_id}"
                        )
                    if finalize_only:
                        outcomes.append(
                            {
                                **base,
                                "decision": "skip_no_existing_reserved_source",
                            }
                        )
                        continue
                    if max_new_pairs is not None and subprocesses_started >= max_new_pairs:
                        outcomes.append(
                            {
                                **base,
                                "decision": "stop_execution_batch_limit",
                                "max_new_pairs": max_new_pairs,
                            }
                        )
                        break
                    remaining_block_reserve = sum(
                        (
                            derive_pair_forecast(item, policy=policy).forecast_usd
                            for item in work_items
                            if item.work_item_id not in state.finalizations
                            and item.work_item_id not in state.reservations
                        ),
                        Decimal(0),
                    )
                    remaining_block_projected = state.total_exposure_usd + remaining_block_reserve
                    if remaining_block_projected > cap_usd * admission_fraction:
                        outcomes.append(
                            {
                                **base,
                                "decision": (
                                    "block_remaining_complete_workload_85_percent_admission_ceiling"
                                ),
                                "remaining_block_reserve_usd": _decimal_text(
                                    remaining_block_reserve
                                ),
                                "remaining_block_projected_exposure_usd": _decimal_text(
                                    remaining_block_projected
                                ),
                            }
                        )
                        break
                    projected = state.total_exposure_usd + forecast.forecast_usd
                    if projected > cap_usd:
                        outcomes.append(
                            {
                                **base,
                                "decision": "stop_hard_cap",
                                "exposure_before_usd": _decimal_text(state.total_exposure_usd),
                                "projected_exposure_usd": _decimal_text(projected),
                            }
                        )
                        break
                    if projected > cap_usd * admission_fraction:
                        outcomes.append(
                            {
                                **base,
                                "decision": "stop_85_percent_admission_ceiling",
                                "exposure_before_usd": _decimal_text(state.total_exposure_usd),
                                "projected_exposure_usd": _decimal_text(projected),
                            }
                        )
                        break
                    reservation = append_dataset_ledger_event(
                        ledger,
                        {
                            "event_type": "reservation_created",
                            "runner_run_id": run_id,
                            "work_item_id": work_item.work_item_id,
                            "manifest_sha256": manifest_sha,
                            "task_registry_sha256": registry_sha,
                            "task_id": work_item.task.public_id,
                            "task_family": work_item.task.family,
                            "prompt_sha256": work_item.task.prompt_sha256,
                            "model_id": work_item.candidate.model_id,
                            "canonical_model_slug": work_item.candidate.canonical_model_slug,
                            "provider_tag": work_item.candidate.provider_tag,
                            "execution_backend": work_item.candidate.execution_backend,
                            "backend_contract_sha256": (
                                work_item.candidate.backend_contract_sha256
                            ),
                            "route_selection_reason": work_item.candidate.route_selection.get(
                                "selection_reason"
                            ),
                            "endpoint_execution_sha256": (work_item.endpoint_execution_sha256),
                            "execution_policy_sha256": (work_item.execution_policy_sha256),
                            "reserved_usd": _decimal_text(forecast.forecast_usd),
                            "total_exposure_before_usd": _decimal_text(state.total_exposure_usd),
                            "derived_max_price": {
                                "prompt_usd_per_mtok": _decimal_text(
                                    forecast.price_envelope.prompt_usd_per_mtok
                                ),
                                "completion_usd_per_mtok": _decimal_text(
                                    forecast.price_envelope.completion_usd_per_mtok
                                ),
                            },
                        },
                    )
                    command = _subprocess_command(
                        work_item,
                        forecast=forecast,
                        source_directory=source_root,
                        manifest_path=Path(manifest_path),
                    )
                    environment = os.environ.copy()
                    environment.update(work_item.execution_policy.settings_environment())
                    environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] = (
                        _decimal_text(forecast.price_envelope.prompt_usd_per_mtok)
                    )
                    environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] = (
                        _decimal_text(forecast.price_envelope.completion_usd_per_mtok)
                    )
                    subprocesses_started += 1
                    try:
                        completed = subprocess.run(
                            command,
                            env=environment,
                            capture_output=True,
                            text=True,
                            timeout=process_timeout_seconds,
                            check=False,
                        )
                    except subprocess.TimeoutExpired as error:
                        try:
                            timeout_journals = scan_recovery_journals(
                                source_root,
                                dataset_work_item_id=work_item.work_item_id,
                            )
                        except JournalIntegrityError as journal_error:
                            raise IntegrityError(
                                "timed-out live-smoke journal failed integrity validation"
                            ) from journal_error
                        timeout_evidence = [
                            {
                                "filename": item.path.name,
                                "journal_sha256": item.journal_sha256,
                                "head_entry_sha256": item.head_entry_sha256,
                                "entry_count": item.entry_count,
                                "run_id": item.run_id,
                                "generation_ids": list(item.generation_ids),
                                "unreconciled_generation_ids": list(
                                    item.unreconciled_generation_ids
                                ),
                                "uncertain_attempt_ids": list(item.uncertain_attempt_ids),
                                "recovery_action": item.recovery_action,
                                "safe_to_replay": item.safe_to_replay,
                            }
                            for item in timeout_journals
                        ]
                        append_dataset_ledger_event(
                            ledger,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item.work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "subprocess_timeout_uncertain_delivery",
                                "timeout_seconds": process_timeout_seconds,
                                "output_sha256": _safe_process_hash(str(error.output or "")),
                                "recovery_journals": timeout_evidence,
                            },
                        )
                        outcomes.append(
                            {
                                **base,
                                "decision": "execution_timeout_reservation_retained",
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "recovery_journals": timeout_evidence,
                            }
                        )
                        break
                    artifact_path = _extract_artifact_path(completed.stdout, source_root)
                    if artifact_path is None or not artifact_path.exists():
                        try:
                            recovery_journals = scan_recovery_journals(
                                source_root,
                                dataset_work_item_id=work_item.work_item_id,
                            )
                        except JournalIntegrityError as error:
                            raise IntegrityError(
                                "live-smoke crash journal failed integrity validation"
                            ) from error
                        recovery_evidence = [
                            {
                                "filename": item.path.name,
                                "journal_sha256": item.journal_sha256,
                                "head_entry_sha256": item.head_entry_sha256,
                                "entry_count": item.entry_count,
                                "run_id": item.run_id,
                                "finalized": item.finalized,
                                "generation_ids": list(item.generation_ids),
                                "unreconciled_generation_ids": list(
                                    item.unreconciled_generation_ids
                                ),
                                "uncertain_attempt_ids": list(item.uncertain_attempt_ids),
                                "recovery_action": item.recovery_action,
                                "safe_to_replay": item.safe_to_replay,
                            }
                            for item in recovery_journals
                        ]
                        append_dataset_ledger_event(
                            ledger,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item.work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "no_verifiable_artifact_reservation_retained",
                                "subprocess_returncode": completed.returncode,
                                "stdout_sha256": _safe_process_hash(completed.stdout),
                                "stderr_sha256": _safe_process_hash(completed.stderr),
                                "recovery_journals": recovery_evidence,
                            },
                        )
                        outcomes.append(
                            {
                                **base,
                                "decision": "no_artifact_reservation_retained",
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "subprocess_returncode": completed.returncode,
                                "recovery_journals": recovery_evidence,
                            }
                        )
                        break
                    state = _load_state(
                        prior_artifact_directory=prior_artifact_directory,
                        prior_corrections_directory=prior_corrections_directory,
                        prior_reservation_ledger_path=global_budget_lock_path,
                        source_directory=source_root,
                        source_corrections_directory=source_corrections_directory,
                        source_resolution_directory=resolution_root,
                        response_directory=response_root,
                        ledger_path=ledger,
                    )
                    source = state.sources.get(work_item.work_item_id)
                    if source is None or source.path.resolve() != artifact_path.resolve():
                        raise IntegrityError(
                            "delegated dataset artifact cannot be mapped to work item"
                        )
                    issues = _source_postflight_issues(source, work_item)
                    if issues:
                        append_dataset_ledger_event(
                            ledger,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item.work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "source_postflight_mismatch_reservation_retained",
                                "source_artifact_sha256": source.artifact_sha256,
                                "postflight_issues": issues,
                            },
                        )
                        outcomes.append(
                            {
                                **base,
                                "decision": "source_postflight_mismatch_reservation_retained",
                                "source_artifact_sha256": source.artifact_sha256,
                                "postflight_issues": issues,
                            }
                        )
                        break
                    if (
                        source.exposure.exposure_basis
                        == "failed_or_unreconciled_full_admitted_allowance"
                    ):
                        append_dataset_ledger_event(
                            ledger,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item.work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "generation_cost_unreconciled_reservation_retained",
                                "source_artifact_sha256": source.artifact_sha256,
                            },
                        )
                        outcomes.append(
                            {
                                **base,
                                "decision": "generation_cost_unreconciled_reservation_retained",
                                "source_artifact_sha256": source.artifact_sha256,
                            }
                        )
                        break
                    recorded, responses, normalization_issues = _finalize_source(
                        ledger_path=ledger,
                        runner_run_id=run_id,
                        reservation=reservation,
                        source=source,
                        work_item=work_item,
                        response_directory=response_root,
                        incident_resolution=state.incident_resolutions.get(work_item.work_item_id),
                    )
                    outcomes.append(
                        {
                            **base,
                            "decision": (
                                "pair_recorded"
                                if len(responses) == len(CONDITIONS)
                                else "partial_or_failed_pair_recorded"
                            ),
                            "reservation_entry_sha256": reservation["entry_sha256"],
                            "ledger_entry_sha256": recorded["entry_sha256"],
                            "source_artifact_sha256": source.artifact_sha256,
                            "source_actual_cost_usd": _decimal_text(
                                source.exposure.actual_cost_usd
                            ),
                            "response_artifact_sha256s": sorted(
                                response.artifact_sha256 for response in responses
                            ),
                            "normalization_issues": normalization_issues,
                            "subprocess_returncode": completed.returncode,
                            "stdout_sha256": _safe_process_hash(completed.stdout),
                            "stderr_sha256": _safe_process_hash(completed.stderr),
                        }
                    )

            final_state = _load_state(
                prior_artifact_directory=prior_artifact_directory,
                prior_corrections_directory=prior_corrections_directory,
                prior_reservation_ledger_path=global_budget_lock_path,
                source_directory=source_root,
                source_corrections_directory=source_corrections_directory,
                source_resolution_directory=resolution_root,
                response_directory=response_root,
                ledger_path=ledger,
            )
            _validate_state_against_workload(final_state, work_items)
            if final_state.total_exposure_usd > cap_usd:
                raise AdmissionDenied(
                    f"verified total exposure ${final_state.total_exposure_usd} exceeds "
                    f"the authorised ${cap_usd}; no further calls are permitted"
                )
            selected_by_family = {
                family: [
                    {
                        "public_id": task.public_id,
                        "prompt_sha256": task.prompt_sha256,
                        "review_status": task.review_status,
                    }
                    for task in selected_tasks
                    if task.family == family
                ]
                for family in TASK_FAMILIES
            }
            summary: dict[str, Any] = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "runner_run_id": run_id,
                "mode": (
                    "execute"
                    if execute
                    else (
                        "finalize_existing_no_provider_calls"
                        if finalize_existing_only
                        else "dry_run_no_provider_calls"
                    )
                ),
                "provider_calls_made": execute and subprocesses_started > 0,
                "paid_subprocesses_started": subprocesses_started,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "official": False,
                "rank_eligible": False,
                "research_result": False,
                "manifest": {
                    "filename": Path(manifest_path).name,
                    "sha256": manifest_sha,
                    "selected_model_count": len(candidates),
                    "models": [
                        {
                            "slot_id": candidate.slot_id,
                            "model_id": candidate.model_id,
                            "canonical_model_slug": candidate.canonical_model_slug,
                            "provider_tag": candidate.provider_tag,
                            "execution_backend": candidate.execution_backend,
                            "backend_contract_sha256": candidate.backend_contract_sha256,
                            "route_selection": dict(candidate.route_selection),
                            "endpoint_execution_sha256": _execution_endpoint_sha256(candidate),
                        }
                        for candidate in candidates
                    ],
                },
                "protocol_preflight": (
                    {
                        "registry_filename": Path(str(protocol_preflight_registry_path)).name,
                        "registry_sha256": protocol_preflight_registry["artifact_sha256"],
                        "plan_sha256": protocol_preflight_registry["preflight_plan_sha256"],
                        "model_count": protocol_preflight_registry["model_count"],
                        "epicure": protocol_preflight_registry["epicure"],
                        "implementation": protocol_preflight_registry["implementation"],
                    }
                    if protocol_preflight_registry is not None
                    else None
                ),
                "task_selection": {
                    "schema_version": SELECTION_SCHEMA_VERSION,
                    "seed": selection_seed,
                    "task_pool_per_family": task_pool_per_family,
                    "assignments_per_model": assignments_per_model,
                    "selected_task_count": len(selected_tasks),
                    "candidate_registry_sha256": registry_sha,
                    "candidate_registry_task_count": len(task_inventory),
                    "source": task_source,
                    "selected_by_family": selected_by_family,
                    "human_review_status": "candidate_not_confirmatory",
                },
                "workload": {
                    "expected_pair_count": len(work_items),
                    "expected_response_count": len(work_items) * len(CONDITIONS),
                    "schedule": "model-family diagonal round robin",
                    "work_items": [item.public_payload() for item in work_items],
                    "route_validation_override": route_validation_override,
                },
                "budget": {
                    "currency": "USD",
                    "authorised_hard_cap_usd": _decimal_text(cap_usd),
                    "admission_fraction": _decimal_text(admission_fraction),
                    "admission_ceiling_usd": _decimal_text(cap_usd * admission_fraction),
                    "known_prior_exposure_floor_usd": _decimal_text(
                        KNOWN_PRIOR_OPENROUTER_EXPOSURE_USD
                    ),
                    "verified_prior_artifact_exposure_usd": _decimal_text(
                        final_state.prior_verified_exposure_usd
                    ),
                    "effective_prior_exposure_usd": _decimal_text(
                        final_state.prior_effective_exposure_usd
                    ),
                    "active_frontier_ledger_reservations_usd": _decimal_text(
                        final_state.prior_active_reservation_usd
                    ),
                    "dataset_actual_cost_usd": _decimal_text(final_state.dataset_actual_cost_usd),
                    "dataset_provider_reconciled_actual_cost_usd": _decimal_text(
                        sum(
                            (
                                source.exposure.actual_cost_usd
                                for source in final_state.sources.values()
                                if source.exposure.exposure_basis
                                in {
                                    "fully_reconciled_actual",
                                    "failed_but_all_attempts_cost_reconciled_actual",
                                }
                            ),
                            Decimal(0),
                        )
                    ),
                    "dataset_rate_card_estimated_cost_usd": _decimal_text(
                        sum(
                            (
                                source.exposure.actual_cost_usd
                                for source in final_state.sources.values()
                                if source.exposure.exposure_basis
                                in {
                                    "complete_rate_card_estimated_full_forecast_reserve",
                                    "failed_rate_card_estimated_full_forecast_reserve",
                                }
                            ),
                            Decimal(0),
                        )
                    ),
                    "rate_card_estimated_source_count": sum(
                        source.exposure.exposure_basis
                        in {
                            "complete_rate_card_estimated_full_forecast_reserve",
                            "failed_rate_card_estimated_full_forecast_reserve",
                        }
                        for source in final_state.sources.values()
                    ),
                    "failed_rate_card_estimated_source_count": sum(
                        source.exposure.exposure_basis
                        == "failed_rate_card_estimated_full_forecast_reserve"
                        for source in final_state.sources.values()
                    ),
                    "dataset_source_exposure_usd": _decimal_text(
                        final_state.dataset_source_exposure_usd
                    ),
                    "unresolved_dataset_source_reserve_usd": _decimal_text(
                        final_state.unresolved_dataset_source_reserve_usd
                    ),
                    "conservative_incident_resolution_count": len(final_state.incident_resolutions),
                    "conservative_incident_provider_reconciled_actual_usd": (
                        _decimal_text(
                            sum(
                                (
                                    resolution.provider_reconciled_actual_cost_usd
                                    for resolution in final_state.incident_resolutions.values()
                                ),
                                Decimal(0),
                            )
                        )
                    ),
                    "conservative_incident_full_allowance_exposure_usd": (
                        _decimal_text(
                            sum(
                                (
                                    resolution.conservative_budget_exposure_usd
                                    for resolution in final_state.incident_resolutions.values()
                                ),
                                Decimal(0),
                            )
                        )
                    ),
                    "active_reservations_without_source_usd": _decimal_text(
                        final_state.orphan_reservation_usd
                    ),
                    "final_total_exposure_usd": _decimal_text(final_state.total_exposure_usd),
                    "remaining_hard_cap_usd": _decimal_text(
                        cap_usd - final_state.total_exposure_usd
                    ),
                    "clean_workload_worst_case_forecast_usd": _decimal_text(target_forecast_usd),
                    "initial_outstanding_worst_case_reserve_usd": _decimal_text(
                        outstanding_forecast_usd
                    ),
                    "initial_complete_block_projected_exposure_usd": _decimal_text(
                        initial_complete_block_projected_usd
                    ),
                    "initial_complete_block_admissible": not plan_blocked,
                    "execution_blocked_by_complete_workload_policy": (execute and plan_blocked),
                },
                "coverage_and_reliability": _summary_coverage(work_items, final_state),
                "execution_policy": policy.document(),
                "execution_policy_sha256": policy.sha256,
                "runner_policy": {
                    "conditions": list(CONDITIONS),
                    "paid_execution": (
                        "work items run strictly sequentially; the blinded on/off pair runs "
                        "concurrently under one reservation to avoid condition-order confounding"
                    ),
                    "resume": (
                        "never replay an active reservation; recover only a matching, "
                        "fully accounted immutable source or an append-only conservative "
                        "no-generation-ID incident resolution"
                    ),
                },
                "outcomes": outcomes,
                "ledger": {
                    "filename": ledger.name,
                    "entry_count": len(final_state.ledger),
                    "head_entry_sha256": (
                        final_state.ledger[-1]["entry_sha256"] if final_state.ledger else None
                    ),
                },
                "limitations": [
                    "These records are exploratory and permanently excluded from rankings.",
                    "Candidate tasks are not a frozen, human-reviewed official task set.",
                    "No public or expert preferences are collected by this runner.",
                    "This is not an approved research export or official Season 0 result.",
                    (
                        "Epicure-on requires at least one successful real tool call; failed "
                        "treatments remain in reliability metrics and are not preference pairs."
                        if policy.epicure_on_tool_required
                        else "Epicure-on means tool access; tool use and tool success are "
                        "reported separately."
                    ),
                    "The known prior $1.321528 exposure is charged even if prior artifacts move.",
                    (
                        "Direct live_smoke invocations outside the shared budget lock remain "
                        "prohibited."
                    ),
                    (
                        "Direct Kimi cost uses returned token usage and a frozen public rate "
                        "card because the endpoint exposes no per-generation charged amount; "
                        "those records are excluded from exact cost rankings."
                    ),
                    (
                        "A resolved HTTP-200/no-choice response has no generation ID or exact "
                        "provider cost; its full admitted allowance remains budget exposure, "
                        "and only the separately reconciled valid arm may be normalized."
                    ),
                ],
            }
            summary_path = _write_summary(summary, summary_directory)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    return written, summary_path


def _parser() -> argparse.ArgumentParser:
    def policy_default(name: str, default: str, cast: type[int] | type[float]) -> int | float:
        value = os.environ.get(f"FLAVOURBENCH_DATASET_{name}")
        if value is None:
            value = os.environ.get(f"FLAVOURBENCH_{name}", default)
        return cast(value)

    parser = argparse.ArgumentParser(
        description=(
            "Plan or sequentially collect an unranked real OpenRouter + Epicure dataset. "
            "The default dry-run makes no provider calls."
        )
    )
    parser.add_argument("--manifest")
    parser.add_argument(
        "--expected-manifest-sha256",
        default=CURRENT_DATASET_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Manifest slot ID, model ID, or canonical slug; repeat to select several",
    )
    parser.add_argument("--task-pool-per-family", type=int, default=3)
    parser.add_argument("--assignments-per-model", type=int, default=10)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--task-validity-artifact",
        help=(
            "Content-addressed real-human development task dossier. When omitted, the "
            "legacy candidate registry is used."
        ),
    )
    parser.add_argument(
        "--epicure-native-task-artifact",
        help=(
            "Content-addressed deterministic 32-task Epicure answer-key artifact. "
            "Mutually exclusive with --task-validity-artifact."
        ),
    )
    parser.add_argument(
        "--protocol-preflight-registry",
        help=(
            "Passing content-addressed exact-protocol registry required by current "
            "development manifests."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=policy_default("MAX_OUTPUT_TOKENS", "1000", int),
    )
    parser.add_argument(
        "--max-intermediate-tokens",
        type=int,
        default=policy_default("MAX_INTERMEDIATE_TOKENS", "700", int),
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=policy_default("MAX_TOOL_ROUNDS", "4", int),
    )
    parser.add_argument(
        "--max-tool-result-bytes",
        type=int,
        default=policy_default("MAX_TOOL_RESULT_BYTES", "16384", int),
    )
    parser.add_argument(
        "--max-cumulative-tool-result-bytes",
        type=int,
        default=policy_default("MAX_CUMULATIVE_TOOL_RESULT_BYTES", "49152", int),
    )
    parser.add_argument(
        "--max-tool-calls-per-round",
        type=int,
        default=policy_default("MAX_TOOL_CALLS_PER_ROUND", "4", int),
    )
    parser.add_argument(
        "--max-tool-calls-total",
        type=int,
        default=policy_default("MAX_TOOL_CALLS_TOTAL", "12", int),
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        choices=[1, 2],
        default=policy_default("MAX_PROVIDER_ATTEMPTS", "1", int),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=policy_default("DECODING_TEMPERATURE", "0.2", float),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=policy_default("DECODING_TOP_P", "0.95", float),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=policy_default("DECODING_SEED", "20260715", int),
    )
    parser.add_argument(
        "--final-response-mode",
        choices=["structured_json", "plain_text"],
        default="structured_json",
    )
    parser.add_argument("--tool-catalog-bytes-bound", type=int, default=0)
    parser.add_argument(
        "--evidence-protocol",
        choices=["legacy_v6", *sorted(GOVERNED_EPICURE_PROTOCOLS)],
        default="legacy_v6",
    )
    parser.add_argument(
        "--require-epicure-call",
        action="store_true",
        help="Require one successful real Epicure call in every Epicure-on arm.",
    )
    parser.add_argument(
        "--intermediate-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument(
        "--final-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    parser.add_argument("--prior-artifact-directory", default="artifacts/live-smoke")
    parser.add_argument("--prior-corrections-directory", default="artifacts/corrections")
    parser.add_argument("--source-directory", default="artifacts/real-exploratory/source-runs")
    parser.add_argument(
        "--source-corrections-directory",
        default="artifacts/real-exploratory/corrections",
    )
    parser.add_argument(
        "--source-resolution-directory",
        default="artifacts/real-exploratory/resolutions",
    )
    parser.add_argument("--response-directory", default="artifacts/real-exploratory/responses")
    parser.add_argument("--ledger", default="artifacts/real-exploratory/ledger.jsonl")
    parser.add_argument(
        "--global-budget-lock-path", default="artifacts/frontier-contract/ledger.jsonl"
    )
    parser.add_argument("--summary-directory", default="artifacts/real-exploratory/summaries")
    parser.add_argument("--cap-usd", type=Decimal, default=AUTHORIZED_TOTAL_CAP_USD)
    parser.add_argument(
        "--admission-fraction",
        type=Decimal,
        default=DEFAULT_ADMISSION_FRACTION,
        help="Fail closed when projected total exposure exceeds this fraction of the cap.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--finalize-existing-only",
        action="store_true",
        help="Finalize already reconciled source artifacts without starting provider calls.",
    )
    parser.add_argument(
        "--finalize-unresolved-existing-only",
        action="store_true",
        help=(
            "Finalize one existing unresolved source without replay, while retaining "
            "its complete admitted allowance as unresolved exposure."
        ),
    )
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--record-no-id-resolution",
        action="store_true",
        help=(
            "Append only the conservative HTTP-200/no-choice resolution; this mode "
            "has no provider or MCP call path"
        ),
    )
    parser.add_argument("--resolution-reservation-entry-sha256", default="")
    parser.add_argument("--resolution-incident-entry-sha256", default="")
    parser.add_argument(
        "--resolution-affected-condition",
        choices=CONDITIONS,
        default="epicure_off",
    )
    parser.add_argument("--process-timeout-seconds", type=int, default=3_600)
    parser.add_argument(
        "--max-new-pairs",
        type=int,
        default=None,
        help="Bound newly started pair subprocesses for a resumable execution batch.",
    )
    parser.add_argument(
        "--route-validation-plan",
        help="Content-addressed one-cell route plan that supplies a fresh work-item ID.",
    )
    parser.add_argument(
        "--route-validation-variant",
        choices=["explicit_low", "provider_default", "explicit_high"],
        help="Exact effort variant to execute from the route-validation plan.",
    )
    return parser


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-run-real-dataset")
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.record_no_id_resolution:
        if arguments.execute:
            parser.error("--record-no-id-resolution must not be combined with --execute")
        if not (
            arguments.resolution_reservation_entry_sha256
            and arguments.resolution_incident_entry_sha256
        ):
            parser.error(
                "incident resolution requires both reservation and incident entry SHA-256s"
            )
        try:
            resolution, event = record_no_id_source_incident_resolution(
                ledger_path=arguments.ledger,
                source_directory=arguments.source_directory,
                source_corrections_directory=(arguments.source_corrections_directory or None),
                resolution_directory=arguments.source_resolution_directory,
                reservation_entry_sha256=(arguments.resolution_reservation_entry_sha256),
                incident_entry_sha256=arguments.resolution_incident_entry_sha256,
                affected_condition=arguments.resolution_affected_condition,
                confirmation=arguments.confirm,
            )
        except Exception as error:
            print(
                json.dumps(
                    {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                    sort_keys=True,
                )
            )
            raise SystemExit(1) from error
        print(
            json.dumps(
                {
                    "status": "incident_resolution_recorded",
                    "provider_calls_made": False,
                    "resolution": str(resolution.path.resolve()),
                    "resolution_artifact_sha256": resolution.artifact_sha256,
                    "ledger_entry_sha256": event["entry_sha256"],
                    "work_item_id": resolution.work_item_id,
                    "provider_reconciled_actual_cost_usd": _decimal_text(
                        resolution.provider_reconciled_actual_cost_usd
                    ),
                    "conservative_budget_exposure_usd": _decimal_text(
                        resolution.conservative_budget_exposure_usd
                    ),
                    "normalizable_conditions": list(resolution.normalizable_conditions),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not arguments.manifest:
        parser.error("--manifest is required unless --record-no-id-resolution is used")
    execution_policy = ExecutionPolicy(
        max_output_tokens=arguments.max_output_tokens,
        max_intermediate_tokens=arguments.max_intermediate_tokens,
        max_tool_rounds=arguments.max_tool_rounds,
        max_tool_result_bytes=arguments.max_tool_result_bytes,
        max_cumulative_tool_result_bytes=(arguments.max_cumulative_tool_result_bytes),
        max_tool_calls_per_round=arguments.max_tool_calls_per_round,
        max_tool_calls_total=arguments.max_tool_calls_total,
        max_provider_attempts=arguments.max_provider_attempts,
        decoding_temperature=arguments.temperature,
        decoding_top_p=arguments.top_p,
        decoding_seed=arguments.seed,
        final_response_mode=arguments.final_response_mode,
        matched_planning=arguments.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS,
        evidence_protocol=arguments.evidence_protocol,
        intermediate_reasoning_effort=arguments.intermediate_reasoning_effort,
        final_reasoning_effort=arguments.final_reasoning_effort,
        tool_catalog_bytes_bound=arguments.tool_catalog_bytes_bound,
        epicure_on_tool_required=arguments.require_epicure_call,
    )
    try:
        summary, path = run_real_exploratory_dataset(
            manifest_path=arguments.manifest,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
            selectors=arguments.model,
            task_pool_per_family=arguments.task_pool_per_family,
            assignments_per_model=arguments.assignments_per_model,
            selection_seed=arguments.selection_seed,
            task_validity_artifact_path=arguments.task_validity_artifact,
            epicure_native_task_artifact_path=(arguments.epicure_native_task_artifact),
            protocol_preflight_registry_path=arguments.protocol_preflight_registry,
            prior_artifact_directory=arguments.prior_artifact_directory,
            prior_corrections_directory=arguments.prior_corrections_directory,
            source_directory=arguments.source_directory,
            source_corrections_directory=arguments.source_corrections_directory,
            source_resolution_directory=arguments.source_resolution_directory,
            response_directory=arguments.response_directory,
            ledger_path=arguments.ledger,
            global_budget_lock_path=arguments.global_budget_lock_path,
            summary_directory=arguments.summary_directory,
            cap_usd=arguments.cap_usd,
            admission_fraction=arguments.admission_fraction,
            execution_policy=execution_policy,
            execute=arguments.execute,
            finalize_existing_only=arguments.finalize_existing_only,
            finalize_unresolved_existing_only=(arguments.finalize_unresolved_existing_only),
            confirmation=arguments.confirm,
            process_timeout_seconds=arguments.process_timeout_seconds,
            max_new_pairs=arguments.max_new_pairs,
            route_validation_plan_path=arguments.route_validation_plan,
            route_validation_variant_id=arguments.route_validation_variant,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "status": "finished" if arguments.execute else "planned",
                "provider_calls_made": summary["provider_calls_made"],
                "summary": str(path.resolve()),
                "summary_sha256": summary["content_address"]["digest"],
                "budget": summary["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
