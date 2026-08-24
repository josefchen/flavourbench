"""Materialize and execute the frozen frontier family-coverage repair.

The executor is deliberately narrower than the general development runner.  It
accepts only the content-addressed 13-cell coverage schedule, reuses the exact
high-resource endpoint contracts and Epicure attestation already present in the
corrected arena, and requests only the 25 missing real conditions.  Planning is
the default and performs no provider or MCP calls.

Execution uses an append-only ledger and takes the global frontier ledger lock
as a shared budget mutex.  A reserved work item is never replayed: an existing
source is finalized, while a reservation without a source blocks collection
until an explicit accounting correction is added by a separate governed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .execution_policy import ExecutionPolicy, verify_policy_document
from .frontier_budget_audit import SCHEMA_VERSION as BUDGET_AUDIT_SCHEMA_VERSION
from .frontier_contract_runner import (
    AUTHORIZED_TOTAL_CAP_USD,
    DEFAULT_ADMISSION_FRACTION,
    AdmissionDenied,
    ContractCandidate,
    IntegrityError,
    _exclusive_runner_lock,
    _extract_artifact_path,
    _safe_process_hash,
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
from .frontier_coverage_repair import SCHEMA_VERSION as COVERAGE_SCHEDULE_SCHEMA_VERSION
from .provider import OpenRouterProvider
from .real_dataset_runner import (
    CONDITIONS,
    DatasetSource,
    PairForecast,
    ResponseArtifact,
    WorkItem,
    _dataset_ledger_lock,
    _source_postflight_issues,
    _subprocess_command,
    append_dataset_ledger_event,
    dataset_ledger_state,
    derive_conditions_forecast,
    load_dataset_ledger,
    load_development_task_inventory,
    normalise_source_responses,
    scan_response_artifacts,
    task_registry_sha256,
)
from .real_task_bank import sha256_json
from .reasoning_effort_route_recovery import (
    V3_AUDIT_SCHEMA_VERSION,
    verify_v3_route_plan,
    verify_v3_route_validation_pass_audit,
)
from .response_envelope_route_v4 import (
    AUDIT_SCHEMA_VERSION as V4_AUDIT_SCHEMA_VERSION,
)
from .response_envelope_route_v4 import (
    CLOSURE_SCHEMA_VERSION as V4_CLOSURE_SCHEMA_VERSION,
)
from .response_envelope_route_v4 import (
    PLAN_SCHEMA_VERSION as V4_PLAN_SCHEMA_VERSION,
)
from .response_envelope_route_v4 import (
    verify_v4_closure,
    verify_v4_plan,
    verify_v4_route_acceptance_paths,
)

MATERIALIZATION_SCHEMA_VERSION = "flavourbench-frontier-coverage-materialization-v1"
PLAN_SCHEMA_VERSION = "flavourbench-frontier-coverage-execution-plan-v2"
WORK_ITEM_SCHEMA_VERSION = "flavourbench-frontier-coverage-work-item-v1"
EXECUTION_CONFIRMATION = "RUN_EXACT_COVERAGE_REPAIR_25_REAL_ARMS"
EXPECTED_CELL_COUNT = 13
EXPECTED_ARM_COUNT = 25
EXPECTED_PAIR_CELLS = 12
EXPECTED_PARTIAL_CELLS = 1
HIGH_RESOURCE_STRATUM = "high-resource"
HARD_POSTFLIGHT_EXEMPTIONS = {
    "required_epicure_trace_missing",
    "required_epicure_success_missing",
    "unexpected_results_contract",
}
def _response_envelope_classifier_contract(
    provider_source_sha256: str,
) -> dict[str, Any]:
    """Return the exact v3 route-gate classifier contract.

    The provider implementation digest is part of the contract itself.  This prevents a
    coverage plan from claiming compatibility with a route receipt produced by different
    classifier code, even when the behavioural examples still happen to pass.
    """

    contract = {
        "schema_version": "flavourbench-safe-provider-envelope-classifier-v2",
        "provider_source_sha256": provider_source_sha256,
        "accepted_classification": "chat_completions",
        "rejected_classifications": [
            "openrouter_error_envelope",
            "gateway_api_envelope",
            "responses_api_schema_mismatch",
            "unknown_non_chat_completion_envelope",
        ],
        "retryable_error_codes": [408, 429, 502, 503],
        "http_200_non_chat_action": (
            "retry_allowlisted_error_envelopes_without_generation_or_cost_reconciliation"
        ),
        "attempt_semantics": {
            "rejection_event": "request_rejected",
            "retry_event": "retry_scheduled",
            "retry_uses_fresh_attempt_id": True,
            "maximum_provider_attempts_per_phase": 2,
        },
        "accounting_semantics": {
            "error_envelope_is_generation": False,
            "generation_id_recorded": False,
            "generation_cost_reconciliation_attempted": False,
            "full_pair_reservation_retained_on_terminal_failure": True,
        },
        "persisted_error_metadata": [
            "classification",
            "code",
            "type",
            "provider",
            "retryable",
        ],
        "prohibited_persistence": [
            "raw response body",
            "provider error message",
            "provider metadata raw field",
            "request prompt",
            "authorization material",
        ],
    }
    return contract


_PROVIDER_SOURCE_PATH = Path(__file__).with_name("provider.py")
_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_PROVIDER_SOURCE_PATH.read_bytes()).hexdigest()
RESPONSE_ENVELOPE_CLASSIFIER_CONTRACT = _response_envelope_classifier_contract(
    _PROVIDER_SOURCE_SHA256
)
RESPONSE_ENVELOPE_CLASSIFIER_SHA256 = sha256_json(RESPONSE_ENVELOPE_CLASSIFIER_CONTRACT)


@dataclass(frozen=True)
class CoverageCell:
    """One exact endpoint/task invocation in the repair schedule."""

    schedule_cell_id: str
    work_item: WorkItem
    route_manifest_path: Path
    route_manifest_sha256: str
    existing_conditions: tuple[str, ...]
    conditions: tuple[str, ...]
    arm_ids: tuple[str, ...]
    forecast: PairForecast

    def public_payload(self) -> dict[str, Any]:
        return {
            "ordinal": self.work_item.ordinal,
            "schedule_cell_id": self.schedule_cell_id,
            "work_item": self.work_item.public_payload(),
            "route_manifest": {
                "filename": self.route_manifest_path.name,
                "artifact_sha256": self.route_manifest_sha256,
                "usage_boundary": "exact_endpoint_contract_only",
            },
            "existing_real_conditions_reused": list(self.existing_conditions),
            "required_new_conditions": list(self.conditions),
            "required_new_real_arms": len(self.conditions),
            "arm_ids": list(self.arm_ids),
            "forecast": self.forecast.public_payload(),
        }


@dataclass(frozen=True)
class CoverageMaterialization:
    document: Mapping[str, Any]
    cells: tuple[CoverageCell, ...]
    policy: ExecutionPolicy
    epicure: Mapping[str, str]
    schedule_sha256: str
    task_validity_sha256: str
    task_registry_sha256: str


@dataclass(frozen=True)
class SupplementalRun:
    source_directory: Path
    ledger_path: Path
    corrections_directory: Path | None = None


@dataclass(frozen=True)
class RunAccounting:
    source_count: int
    actual_cost_usd: Decimal
    exposure_usd: Decimal
    orphan_reservation_usd: Decimal
    artifact_sha256s: frozenset[str]
    sources: Mapping[str, DatasetSource]
    reservations: Mapping[str, Mapping[str, Any]]
    finalizations: Mapping[str, Mapping[str, Any]]
    blockers: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CoverageState:
    accounting: RunAccounting
    responses: Mapping[tuple[str, str], ResponseArtifact]
    ledger: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class BudgetState:
    baseline_exposure_usd: Decimal
    global_active_reservation_usd: Decimal
    supplemental_actual_cost_usd: Decimal
    supplemental_exposure_usd: Decimal
    supplemental_orphan_reservation_usd: Decimal
    coverage_actual_cost_usd: Decimal
    coverage_exposure_usd: Decimal
    coverage_orphan_reservation_usd: Decimal
    current_total_exposure_usd: Decimal
    outstanding_repair_forecast_usd: Decimal
    projected_total_exposure_usd: Decimal
    admission_ceiling_usd: Decimal
    hard_cap_usd: Decimal
    blockers: tuple[Mapping[str, Any], ...]

    @property
    def budget_within_limits(self) -> bool:
        return (
            self.projected_total_exposure_usd <= self.admission_ceiling_usd
            and self.projected_total_exposure_usd <= self.hard_cap_usd
        )

    @property
    def admission_allowed(self) -> bool:
        return not self.blockers and self.budget_within_limits


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise IntegrityError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise IntegrityError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise IntegrityError(f"{field} must be a lowercase SHA-256")
    return digest


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"{label} must be a regular, non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{label} is not a JSON object: {path}")
    return value


def _load_content_addressed_artifact(
    path: Path,
    *,
    label: str,
    schema_version: str,
) -> tuple[dict[str, Any], str]:
    document = _load_json(path, label=label)
    digest = _require_sha256(document.get("artifact_sha256"), field=f"{label} digest")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema_version") != schema_version
        or sha256_json(payload) != digest
        or digest not in path.name
    ):
        raise IntegrityError(f"{label} content address or schema does not verify")
    return document, digest


def _policy_from_document(document: object) -> ExecutionPolicy:
    if not verify_policy_document(document) or not isinstance(document, Mapping):
        raise IntegrityError("route manifest has no valid content-addressed execution policy")
    limits = document.get("limits")
    decoding = document.get("decoding")
    forecast = document.get("cost_forecast")
    reasoning = document.get("reasoning") or {}
    if not all(isinstance(value, Mapping) for value in (limits, decoding, forecast, reasoning)):
        raise IntegrityError("route execution policy lacks required objects")
    assert isinstance(limits, Mapping)
    assert isinstance(decoding, Mapping)
    assert isinstance(forecast, Mapping)
    assert isinstance(reasoning, Mapping)
    try:
        policy = ExecutionPolicy(
            max_output_tokens=int(limits["max_output_tokens"]),
            max_tool_rounds=int(limits["max_tool_rounds"]),
            max_tool_result_bytes=int(limits["max_tool_result_bytes"]),
            max_cumulative_tool_result_bytes=int(limits["max_cumulative_tool_result_bytes"]),
            max_tool_calls_per_round=int(limits["max_tool_calls_per_round"]),
            max_tool_calls_total=int(limits["max_tool_calls_total"]),
            max_provider_attempts=int(limits["max_provider_attempts"]),
            tool_argument_repair_turns=int(limits["tool_argument_repair_turns"]),
            decoding_temperature=float(decoding["temperature"]),
            decoding_top_p=float(decoding["top_p"]),
            decoding_seed=int(decoding["seed"]),
            approximate_non_user_prompt_bytes=int(forecast["approximate_non_user_prompt_bytes"]),
            conservative_bytes_per_token=int(forecast["conservative_bytes_per_token"]),
            pair_arm_scheduling=str(document["pair_arm_scheduling"]),
            final_response_mode=str(document.get("final_response_mode", "structured_json")),
            max_intermediate_tokens=int(limits.get("max_intermediate_tokens", 700)),
            required_tool_contract_max_intermediate_tokens=int(
                limits.get("required_tool_contract_max_intermediate_tokens", 2_048)
            ),
            matched_planning=bool(document.get("matched_planning", False)),
            evidence_protocol=str(document.get("evidence_protocol", "legacy_v6")),
            intermediate_reasoning_effort=reasoning.get("intermediate_effort"),
            final_reasoning_effort=reasoning.get("final_effort"),
            required_tool_contract_protocol=str(
                document.get("required_tool_contract_protocol", "direct_tool_first_v1")
            ),
            tool_catalog_bytes_bound=int(forecast.get("tool_catalog_bytes_bound", 0)),
            epicure_on_tool_required=bool(document.get("epicure_on_tool_required", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("route execution policy contains invalid fields") from error
    policy.validate()
    if policy.document() != document:
        raise IntegrityError("route execution policy cannot be reconstructed exactly")
    return policy


def _manifest_policy(manifest: Mapping[str, Any]) -> ExecutionPolicy:
    design = manifest.get("run_design")
    if not isinstance(design, Mapping):
        raise IntegrityError("route manifest has no run design")
    policy = _policy_from_document(design.get("execution_policy"))
    if design.get("execution_policy_sha256") != policy.sha256:
        raise IntegrityError("route manifest execution-policy digest does not verify")
    return policy


def _coverage_work_item_id(
    *,
    schedule_sha256: str,
    schedule_cell_id: str,
    task_validity_sha256: str,
    task_registry_digest: str,
    route_manifest_sha256: str,
    task_id: str,
    prompt_sha256: str,
    family: str,
    candidate: ContractCandidate,
    conditions: Sequence[str],
    existing_conditions: Sequence[str],
    execution_policy_sha256: str,
    epicure: Mapping[str, str],
) -> str:
    return sha256_json(
        {
            "schema_version": WORK_ITEM_SCHEMA_VERSION,
            "coverage_schedule_sha256": schedule_sha256,
            "schedule_cell_id": schedule_cell_id,
            "task_validity_sha256": task_validity_sha256,
            "task_registry_sha256": task_registry_digest,
            "route_manifest_sha256": route_manifest_sha256,
            "task": {
                "task_id": task_id,
                "family": family,
                "prompt_sha256": prompt_sha256,
            },
            "model": {
                "model_id": candidate.model_id,
                "canonical_model_slug": candidate.canonical_model_slug,
                "provider_tag": candidate.provider_tag,
                "execution_backend": candidate.execution_backend,
                "backend_contract_sha256": candidate.backend_contract_sha256,
                "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            },
            "existing_real_conditions_reused": list(existing_conditions),
            "required_new_conditions": list(conditions),
            "execution_policy_sha256": execution_policy_sha256,
            "epicure": dict(epicure),
        }
    )


def _arm_id(work_item_id: str, condition: str) -> str:
    return sha256_json(
        {
            "schema_version": "flavourbench-frontier-coverage-arm-v1",
            "work_item_id": work_item_id,
            "condition": condition,
        }
    )


def _verify_schedule_shape(schedule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    counts = schedule.get("counts")
    cells = schedule.get("missing_endpoint_task_cells")
    boundary = schedule.get("claim_boundary")
    policy = schedule.get("collection_policy")
    if not all(isinstance(value, Mapping) for value in (counts, boundary, policy)):
        raise IntegrityError("coverage schedule lacks counts, boundary, or collection policy")
    if not isinstance(cells, list) or not all(isinstance(value, Mapping) for value in cells):
        raise IntegrityError("coverage schedule cells are malformed")
    assert isinstance(counts, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(policy, Mapping)
    lengths = Counter(int(cell.get("required_new_real_arms") or 0) for cell in cells)
    if (
        schedule.get("status") != "frozen_development_schedule_no_calls_executed"
        or len(cells) != EXPECTED_CELL_COUNT
        or counts.get("missing_endpoint_task_cells") != EXPECTED_CELL_COUNT
        or counts.get("required_new_real_arms") != EXPECTED_ARM_COUNT
        or counts.get("synthetic_tasks") != 0
        or counts.get("synthetic_arms") != 0
        or lengths != Counter({2: EXPECTED_PAIR_CELLS, 1: EXPECTED_PARTIAL_CELLS})
        or boundary.get("official") is not False
        or boundary.get("rank_eligible") is not False
        or boundary.get("zero_synthetic_tasks") is not True
        or boundary.get("zero_synthetic_arms") is not True
        or policy.get("provider_fallbacks") is not False
        or policy.get("provider_substitution") is not False
        or policy.get("paid_calls_executed_by_this_artifact") != 0
        or policy.get("reuse_existing_content_addressed_real_arms") is not True
    ):
        raise IntegrityError("coverage schedule is not the frozen 13-cell/25-arm contract")
    return list(cells)


def _verify_arena_bindings(
    *,
    arena: Mapping[str, Any],
    anchor_task_ids: set[str],
) -> tuple[str, dict[str, str], set[tuple[str, str]], dict[str, int]]:
    source = arena.get("source")
    epicure = arena.get("epicure")
    if arena.get("track") != "model_arena" or not isinstance(source, Mapping):
        raise IntegrityError("corrected arena input is not a model-arena artifact")
    if not isinstance(epicure, Mapping):
        raise IntegrityError("corrected arena has no Epicure provenance")
    strata = source.get("strata")
    high = strata.get(HIGH_RESOURCE_STRATUM) if isinstance(strata, Mapping) else None
    if not isinstance(high, Mapping):
        raise IntegrityError("corrected arena has no high-resource stratum")
    policy_sha256 = _require_sha256(
        high.get("execution_policy_sha256"),
        field="high-resource execution policy",
    )
    bound_epicure = {
        "release_id": str(epicure.get("release_id") or ""),
        "bundle_sha256": _require_sha256(epicure.get("bundle_sha256"), field="Epicure bundle"),
        "application_sha256": _require_sha256(
            epicure.get("application_sha256"), field="Epicure application"
        ),
        "tool_schema_sha256": _require_sha256(
            epicure.get("tool_schema_sha256"), field="Epicure tool schema"
        ),
    }
    if not bound_epicure["release_id"]:
        raise IntegrityError("Epicure release ID is absent")
    observed_on: set[tuple[str, str]] = set()
    pair_family_support: set[tuple[str, str, str]] = set()
    observed_anchor_items = 0
    for raw in arena.get("items") or []:
        if not isinstance(raw, Mapping):
            continue
        left = raw.get("left")
        right = raw.get("right")
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            first, second = sorted(
                (
                    str(left.get("requested_model_id") or ""),
                    str(right.get("requested_model_id") or ""),
                )
            )
            pair_family_support.add((first, second, str(raw.get("task_family") or "")))
        if str(raw.get("task_id") or "") not in anchor_task_ids:
            continue
        observed_anchor_items += 1
        if (
            raw.get("execution_stratum") != HIGH_RESOURCE_STRATUM
            or raw.get("execution_policy_sha256") != policy_sha256
        ):
            raise IntegrityError("an existing anchor arm is outside the high-resource policy")
        task_id = str(raw["task_id"])
        for side in ("left", "right"):
            record = raw.get(side)
            if not isinstance(record, Mapping) or record.get("condition") != "epicure_on":
                raise IntegrityError("corrected arena anchor side is malformed")
            observed_on.add((task_id, str(record.get("requested_model_id") or "")))
    if observed_anchor_items == 0:
        raise IntegrityError("corrected arena contains no selected anchor items")
    roster = tuple(str(value) for value in arena.get("model_order") or [])
    if len(roster) != 16 or len(set(roster)) != len(roster):
        raise IntegrityError("corrected arena does not contain the frozen 16-model roster")
    pair_count = len(roster) * (len(roster) - 1) // 2
    missing_by_family = {
        family: pair_count
        - sum(1 for _, _, observed_family in pair_family_support if observed_family == family)
        for family in ("composition", "cookability", "evidence", "substitution")
    }
    return policy_sha256, bound_epicure, observed_on, missing_by_family


def _classifier_binding() -> dict[str, Any]:
    """Verify and bind the fail-closed HTTP-200 response-envelope classifier."""

    cases = (
        ({"choices": [{"message": {"content": "ok"}}]}, "chat_completions", True),
        ({"error": {"code": 400, "message": "redacted"}}, "openrouter_error_envelope", False),
        ({"success": False, "errors": []}, "gateway_api_envelope", False),
        ({"object": "response", "output": []}, "responses_api_schema_mismatch", False),
        ({"unexpected": "shape"}, "unknown_non_chat_completion_envelope", False),
    )
    for payload, classification, accepted in cases:
        observed = OpenRouterProvider.classify_response_envelope(payload)
        if (
            observed.get("classification") != classification
            or observed.get("accepted_chat_completion") is not accepted
        ):
            raise IntegrityError("installed provider response-envelope classifier drifted")
    source_path = _PROVIDER_SOURCE_PATH
    if source_path.is_symlink() or not source_path.is_file():
        raise IntegrityError("provider classifier source is not a regular file")
    provider_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    contract = _response_envelope_classifier_contract(provider_source_sha256)
    return {
        "contract": contract,
        "contract_sha256": sha256_json(contract),
        "provider_source_filename": source_path.name,
        "provider_source_sha256": provider_source_sha256,
        "behavioral_self_test_passed": True,
        "fresh_identifier_gate": (
            "a separate content-addressed live route acceptance must prove non-empty, "
            "unique attempt and generation IDs with no overlap to closed prior runs"
        ),
    }


def _response_envelope_route_gate(
    *,
    route_plan_path: Path | None,
    route_audit_path: Path | None,
    route_closure_path: Path | None,
    classifier_binding: Mapping[str, Any],
    project_root: Path,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Verify a source-reconstructed route PASS before any coverage call."""

    if route_plan_path is None:
        missing = ["route_plan"]
        if route_audit_path is None:
            missing.append("route_pass_audit")
        return (
            {
                "status": "blocked_pending_v3_route_validation",
                "classifier_contract_sha256": classifier_binding["contract_sha256"],
                "provider_source_sha256": classifier_binding["provider_source_sha256"],
                "missing": missing,
            },
            (
                {
                    "gate": "response_envelope_route_acceptance",
                    "reason": (
                        "coverage requires a verified PASS audit using the fail-closed "
                        "classifier and fresh non-replayed work-item IDs"
                    ),
                    "missing": missing,
                },
            ),
        )
    plan = _load_json(route_plan_path, label="response-envelope route-validation plan")
    if plan.get("schema_version") == V4_PLAN_SCHEMA_VERSION:
        if not verify_v4_plan(
            plan,
            repo_root=project_root,
            require_current_sources=True,
        ):
            raise IntegrityError("v4 response-envelope route plan does not verify")
        plan_binding = {
            "route_protocol": "v4_source_reconstructed_one_pair",
            "route_plan_sha256": plan["artifact_sha256"],
            "classifier_contract_sha256": classifier_binding["contract_sha256"],
            "provider_source_sha256": classifier_binding["provider_source_sha256"],
            "source_bundle_sha256": plan["source_code"]["bundle_sha256"],
            "fresh_work_item_ids_sha256": sha256_json(
                [plan["work"]["work_item_id"]]
            ),
            "prior_closed_identifiers_sha256": plan["prior_closed_identifiers"][
                "inventory_sha256"
            ],
            "route_validation_outputs_reused_for_coverage": False,
        }
        missing = []
        if route_audit_path is None:
            missing.append("v4_pass_audit")
        if route_closure_path is None:
            missing.append("v4_identifier_closure")
        if missing:
            return (
                {
                    "status": "blocked_pending_v4_source_reconstructed_evidence",
                    **plan_binding,
                    "missing": missing,
                },
                (
                    {
                        "gate": "response_envelope_route_acceptance",
                        "reason": (
                            "the v4 plan verifies, but its source-reconstructed PASS audit "
                            "and permanent identifier closure are both required"
                        ),
                        "missing": missing,
                    },
                ),
            )
        assert route_audit_path is not None and route_closure_path is not None
        audit = _load_json(route_audit_path, label="v4 route-validation audit")
        closure = _load_json(route_closure_path, label="v4 route-validation closure")
        if (
            audit.get("schema_version") != V4_AUDIT_SCHEMA_VERSION
            or closure.get("schema_version") != V4_CLOSURE_SCHEMA_VERSION
            or not verify_v4_closure(closure, plan=plan, audit=audit)
        ):
            raise IntegrityError("v4 route audit or closure is malformed")
        if verify_v4_route_acceptance_paths(
            plan_path=route_plan_path,
            audit_path=route_audit_path,
            closure_path=route_closure_path,
            repo_root=project_root,
        ):
            return (
                {
                    "status": "passed_all_predicates_v4_source_reconstructed",
                    **plan_binding,
                    "route_audit_sha256": audit["artifact_sha256"],
                    "route_closure_sha256": closure["artifact_sha256"],
                    "source_reconstruction_passed": True,
                    "route_smoke_quality_observations": 0,
                },
                (),
            )
        if audit.get("decision") != "failed_one_or_more_predicates":
            raise IntegrityError(
                "v4 claimed PASS but failed independent source reconstruction"
            )
        return (
            {
                "status": "blocked_failed_v4_route_validation",
                **plan_binding,
                "route_audit_sha256": audit["artifact_sha256"],
                "route_closure_sha256": closure["artifact_sha256"],
                "source_reconstruction_passed": False,
            },
            (
                {
                    "gate": "response_envelope_route_acceptance",
                    "reason": "the source-reconstructed v4 route audit did not pass",
                    "missing": ["v4_pass_audit"],
                },
            ),
        )
    if route_closure_path is not None:
        raise IntegrityError("a v3 route plan cannot be combined with a v4 closure")
    if not verify_v3_route_plan(plan):
        raise IntegrityError("v3 route-validation plan does not verify")
    envelope = plan.get("safe_response_envelope_contract")
    source = plan.get("source")
    route = plan.get("route_validation")
    if not all(isinstance(value, Mapping) for value in (envelope, source, route)):
        raise IntegrityError("v3 route-validation plan lacks classifier or route bindings")
    assert isinstance(envelope, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(route, Mapping)
    if (
        envelope.get("contract_sha256") != classifier_binding["contract_sha256"]
        or source.get("provider_source_sha256") != classifier_binding["provider_source_sha256"]
    ):
        stale_audit_decision = None
        if route_audit_path is not None:
            stale_audit_decision = _load_json(
                route_audit_path, label="stale v3 route-validation audit"
            ).get("decision")
        return (
            {
                "status": "blocked_failed_v3_route_validation",
                "stale_classifier_binding": True,
                "route_audit_decision": stale_audit_decision,
                "route_plan_sha256": plan["artifact_sha256"],
                "classifier_contract_sha256": classifier_binding["contract_sha256"],
                "provider_source_sha256": classifier_binding["provider_source_sha256"],
            },
            (
                {
                    "gate": "response_envelope_route_acceptance",
                    "reason": (
                        "the v3 receipt is bound to superseded classifier source; a fresh "
                        "source-reconstructed v4 qualification is required"
                    ),
                    "missing": ["v4_pass_audit", "v4_identifier_closure"],
                },
            ),
        )
    work_items = route.get("work_items")
    closed_ids = {
        str(value) for value in plan.get("closed_work_item_ids_never_replayed") or []
    }
    fresh_ids = {
        str(item.get("work_item_id") or "")
        for item in work_items or []
        if isinstance(item, Mapping)
    }
    if (
        not isinstance(work_items, list)
        or len(work_items) != 3
        or len(fresh_ids) != 3
        or any(len(value) != 64 for value in fresh_ids | closed_ids)
        or fresh_ids & closed_ids
    ):
        raise IntegrityError("v3 route-validation work-item IDs are not fresh and non-replayed")
    plan_binding = {
        "route_plan_sha256": plan["artifact_sha256"],
        "classifier_contract_sha256": classifier_binding["contract_sha256"],
        "provider_source_sha256": classifier_binding["provider_source_sha256"],
        "fresh_work_item_ids_sha256": sha256_json(sorted(fresh_ids)),
        "closed_v1_v2_work_item_ids_sha256": sha256_json(sorted(closed_ids)),
        "fresh_work_item_ids_do_not_overlap_closed_v1_v2_ids": True,
        "route_validation_outputs_reused_for_coverage": False,
    }
    if route_audit_path is None:
        return (
            {
                "status": "blocked_pending_v3_route_validation_pass_audit",
                **plan_binding,
                "missing": ["v3_pass_audit"],
            },
            (
                {
                    "gate": "response_envelope_route_acceptance",
                    "reason": (
                        "the fresh-ID v3 route plan verifies, but no content-addressed "
                        "PASS audit exists"
                    ),
                    "missing": ["v3_pass_audit"],
                },
            ),
        )
    audit = _load_json(route_audit_path, label="v3 route-validation audit")
    if verify_v3_route_validation_pass_audit(audit, plan):
        return (
            {
                "status": "passed_all_predicates",
                **plan_binding,
                "route_audit_sha256": audit["artifact_sha256"],
            },
            (),
        )
    digest = audit.get("artifact_sha256")
    unhashed = {key: value for key, value in audit.items() if key != "artifact_sha256"}
    if (
        audit.get("schema_version") != V3_AUDIT_SCHEMA_VERSION
        or digest != sha256_json(unhashed)
        or audit.get("v3_route_plan_sha256") != plan["artifact_sha256"]
        or audit.get("decision") not in {"not_executed", "failed_one_or_more_predicates"}
    ):
        raise IntegrityError("v3 route-validation audit is malformed or does not bind the plan")
    decision = str(audit["decision"])
    status = (
        "blocked_failed_v3_route_validation"
        if decision == "failed_one_or_more_predicates"
        else "blocked_pending_v3_route_validation_execution"
    )
    return (
        {
            "status": status,
            **plan_binding,
            "route_audit_sha256": digest,
            "route_audit_decision": decision,
        },
        (
            {
                "gate": "response_envelope_route_acceptance",
                "reason": (
                    "the content-addressed v3 route audit did not pass; coverage execution "
                    "remains blocked and no provider call is admitted"
                ),
                "missing": ["v3_pass_audit"],
            },
        ),
    )


def build_materialization(
    *,
    schedule_path: Path,
    arena_path: Path,
    task_validity_path: Path,
    route_manifest_paths: Sequence[Path],
) -> CoverageMaterialization:
    """Build the immutable 13-cell, 25-real-arm workload without provider calls."""

    schedule, schedule_sha256 = _load_content_addressed_artifact(
        schedule_path,
        label="coverage schedule",
        schema_version=COVERAGE_SCHEDULE_SCHEMA_VERSION,
    )
    cells = _verify_schedule_shape(schedule)
    arena, arena_sha256 = _load_content_addressed_artifact(
        arena_path,
        label="corrected arena",
        schema_version="flavourbench-frontier-model-arena-review-pool-v1",
    )
    task_inventory, task_source = load_development_task_inventory(task_validity_path)
    task_validity_sha256 = str(task_source["artifact_sha256"])
    source = schedule.get("source")
    if not isinstance(source, Mapping):
        raise IntegrityError("coverage schedule has no immutable source bindings")
    if (
        source.get("arena_pool_sha256") != arena_sha256
        or source.get("task_validity_sha256") != task_validity_sha256
    ):
        raise IntegrityError("coverage schedule source bindings do not match supplied artifacts")

    tasks_by_id = {task.public_id: task for task in task_inventory}
    anchors = schedule.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 4:
        raise IntegrityError("coverage schedule must freeze four task-family anchors")
    anchor_task_ids = {
        str(anchor.get("task_id") or "") for anchor in anchors if isinstance(anchor, Mapping)
    }
    if len(anchor_task_ids) != 4:
        raise IntegrityError("coverage anchor task IDs are malformed or duplicated")
    high_policy_sha256, epicure, observed_on, missing_by_family = _verify_arena_bindings(
        arena=arena,
        anchor_task_ids=anchor_task_ids,
    )

    if not route_manifest_paths:
        raise IntegrityError("at least one exact route manifest is required")
    candidates: dict[str, tuple[ContractCandidate, Path, str]] = {}
    route_manifest_digests: list[str] = []
    frozen_policy: ExecutionPolicy | None = None
    for path in route_manifest_paths:
        manifest = load_candidate_manifest(path, expected_digest="")
        digest = str(manifest["content_address"]["digest"])
        policy = _manifest_policy(manifest)
        if policy.sha256 != high_policy_sha256:
            raise IntegrityError("route manifest is outside the arena high-resource policy")
        if frozen_policy is None:
            frozen_policy = policy
        elif frozen_policy.document() != policy.document():
            raise IntegrityError("route manifests do not share one exact execution policy")
        route_manifest_digests.append(digest)
        for candidate in select_candidates(manifest):
            if candidate.model_id in candidates:
                raise IntegrityError(
                    f"model appears in more than one route manifest: {candidate.model_id}"
                )
            if candidate.execution_backend not in {"openrouter", "kimi_direct", "cohere_direct"}:
                raise IntegrityError(
                    f"coverage executor has no exact subprocess for {candidate.execution_backend}"
                )
            if (
                candidate.route_selection.get("selection_frozen_before_generation") is not True
                or candidate.route_selection.get("generation_time_automatic_fallback") is not False
            ):
                raise IntegrityError(f"route is not frozen before generation: {candidate.model_id}")
            candidates[candidate.model_id] = (candidate, path, digest)
    if frozen_policy is None:
        raise IntegrityError("no execution policy was loaded")

    registry_digest = task_registry_sha256(task_inventory)
    runtime_cells: list[CoverageCell] = []
    seen_schedule_cells: set[str] = set()
    seen_work_items: set[str] = set()
    seen_target_coordinates: set[tuple[str, str]] = set()
    for ordinal, raw in enumerate(cells, start=1):
        schedule_cell_id = _require_sha256(raw.get("schedule_cell_id"), field="schedule cell ID")
        family = str(raw.get("family") or "")
        task_id = str(raw.get("task_id") or "")
        model_id = str(raw.get("model_id") or "")
        prompt_sha256 = _require_sha256(
            raw.get("prompt_sha256"), field=f"{schedule_cell_id} prompt"
        )
        conditions = tuple(str(value) for value in raw.get("required_new_conditions") or [])
        existing = tuple(str(value) for value in raw.get("existing_real_conditions") or [])
        expected_cell_id = sha256_json(
            {
                "schema_version": COVERAGE_SCHEDULE_SCHEMA_VERSION,
                "family": family,
                "task_id": task_id,
                "model_id": model_id,
                "required_conditions": list(conditions),
            }
        )
        if (
            schedule_cell_id != expected_cell_id
            or schedule_cell_id in seen_schedule_cells
            or (task_id, model_id) in seen_target_coordinates
            or not conditions
            or len(set(conditions)) != len(conditions)
            or not set(conditions) <= set(CONDITIONS)
            or len(set(existing)) != len(existing)
            or not set(existing) <= set(CONDITIONS)
            or set(existing) & set(conditions)
            or int(raw.get("required_new_real_arms") or 0) != len(conditions)
        ):
            raise IntegrityError(f"coverage schedule cell is malformed: {schedule_cell_id}")
        if len(conditions) == 1 and not (
            conditions == ("epicure_off",) and existing == ("epicure_on",)
        ):
            raise IntegrityError("the sole partial cell must request only the missing off arm")
        task = tasks_by_id.get(task_id)
        if (
            task is None
            or task.family != family
            or task.prompt_sha256 != prompt_sha256
            or task_id not in anchor_task_ids
        ):
            raise IntegrityError(
                f"coverage cell is not bound to the corrected task dossier: {task_id}"
            )
        existing_on = (task_id, model_id) in observed_on
        if existing_on != ("epicure_on" in existing):
            raise IntegrityError(
                f"coverage schedule existing on-arm state drifted: {schedule_cell_id}"
            )
        route = candidates.get(model_id)
        if route is None:
            raise IntegrityError(f"coverage model has no supplied exact route: {model_id}")
        candidate, manifest_path, manifest_sha256 = route
        expected_contract = {
            "canonical_model_slug": candidate.canonical_model_slug,
            "execution_backend": candidate.execution_backend,
            "provider_tag": candidate.provider_tag,
        }
        if raw.get("model_contract") != expected_contract:
            raise IntegrityError(f"coverage model contract differs from its route: {model_id}")
        work_item_id = _coverage_work_item_id(
            schedule_sha256=schedule_sha256,
            schedule_cell_id=schedule_cell_id,
            task_validity_sha256=task_validity_sha256,
            task_registry_digest=registry_digest,
            route_manifest_sha256=manifest_sha256,
            task_id=task_id,
            prompt_sha256=prompt_sha256,
            family=family,
            candidate=candidate,
            conditions=conditions,
            existing_conditions=existing,
            execution_policy_sha256=frozen_policy.sha256,
            epicure=epicure,
        )
        if work_item_id in seen_work_items:
            raise IntegrityError("coverage work-item identity is duplicated")
        work_item = WorkItem(
            ordinal=ordinal,
            work_item_id=work_item_id,
            manifest_sha256=manifest_sha256,
            task_registry_sha256=registry_digest,
            task=task,
            candidate=candidate,
            endpoint_execution_sha256=candidate.endpoint_execution_sha256,
            execution_policy_sha256=frozen_policy.sha256,
            execution_policy=frozen_policy,
        )
        forecast = derive_conditions_forecast(
            work_item,
            policy=frozen_policy,
            conditions=conditions,
        )
        runtime_cells.append(
            CoverageCell(
                schedule_cell_id=schedule_cell_id,
                work_item=work_item,
                route_manifest_path=manifest_path,
                route_manifest_sha256=manifest_sha256,
                existing_conditions=existing,
                conditions=conditions,
                arm_ids=tuple(_arm_id(work_item_id, condition) for condition in conditions),
                forecast=forecast,
            )
        )
        seen_schedule_cells.add(schedule_cell_id)
        seen_work_items.add(work_item_id)
        seen_target_coordinates.add((task_id, model_id))

    if (
        len(runtime_cells) != EXPECTED_CELL_COUNT
        or sum(len(cell.conditions) for cell in runtime_cells) != EXPECTED_ARM_COUNT
    ):
        raise IntegrityError("materialized coverage workload is not exactly 13 cells and 25 arms")
    total_forecast = sum((cell.forecast.forecast_usd for cell in runtime_cells), Decimal(0))
    if sum(missing_by_family.values()) != int(
        (schedule.get("counts") or {}).get("current_missing_model_pair_family_cells") or -1
    ):
        raise IntegrityError("corrected arena family support differs from the frozen schedule")
    by_model: dict[str, Decimal] = defaultdict(Decimal)
    by_provider: dict[str, Decimal] = defaultdict(Decimal)
    by_family: dict[str, Decimal] = defaultdict(Decimal)
    for cell in runtime_cells:
        by_model[cell.work_item.candidate.model_id] += cell.forecast.forecast_usd
        by_provider[cell.work_item.candidate.provider_tag] += cell.forecast.forecast_usd
        by_family[cell.work_item.task.family] += cell.forecast.forecast_usd
    payload: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "artifact_role": "exact_real_arm_coverage_repair_materialization",
        "status": "frozen_dry_run_no_calls_executed",
        "source": {
            "coverage_schedule_sha256": schedule_sha256,
            "corrected_arena_sha256": arena_sha256,
            "task_validity_sha256": task_validity_sha256,
            "task_registry_sha256": registry_digest,
            "route_manifest_sha256s": sorted(route_manifest_digests),
        },
        "execution_policy": frozen_policy.document(),
        "execution_policy_sha256": frozen_policy.sha256,
        "execution_stratum": HIGH_RESOURCE_STRATUM,
        "reasoning_effort_disclosure": {
            "intermediate": frozen_policy.intermediate_reasoning_effort,
            "final": frozen_policy.final_reasoning_effort,
        },
        "epicure": dict(epicure),
        "response_envelope_classifier": _classifier_binding(),
        "counts": {
            "endpoint_task_cells": len(runtime_cells),
            "full_pair_cells": sum(cell.conditions == CONDITIONS for cell in runtime_cells),
            "partial_condition_cells": sum(cell.conditions != CONDITIONS for cell in runtime_cells),
            "new_real_arms": sum(len(cell.conditions) for cell in runtime_cells),
            "reused_existing_real_arms": sum(
                len(cell.existing_conditions) for cell in runtime_cells
            ),
            "planned_provider_work_items": len(runtime_cells),
            "provider_calls_executed_by_materialization": 0,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
            "paid_calls_executed_by_materialization": 0,
            "current_missing_model_pair_family_cells": sum(missing_by_family.values()),
            "current_missing_model_pair_family_cells_by_family": missing_by_family,
            "projected_missing_model_pair_family_cells_after_repair": 0,
        },
        "worst_case_budget": {
            "currency": "USD",
            "total_usd": _decimal_text(total_forecast),
            "by_model": {key: _decimal_text(value) for key, value in sorted(by_model.items())},
            "by_provider": {
                key: _decimal_text(value) for key, value in sorted(by_provider.items())
            },
            "by_task_family": {
                key: _decimal_text(value) for key, value in sorted(by_family.items())
            },
            "reservation_granularity": "one exact schedule cell before subprocess start",
        },
        "work_items": [cell.public_payload() for cell in runtime_cells],
        "execution_invariants": {
            "partial_condition_completion_without_existing_arm_replay": True,
            "one_exact_provider_endpoint_per_model": True,
            "provider_fallbacks": False,
            "provider_substitution": False,
            "shared_global_budget_mutex": True,
            "append_only_coverage_ledger": True,
            "reservation_without_source_is_replay_blocking": True,
            "source_without_finalization_is_recovered_without_provider_call": True,
        },
        "claim_boundary": {
            "development_only": True,
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
            "schedule_completion_is_not_sufficient_for_publication": True,
        },
    }
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    return CoverageMaterialization(
        document=document,
        cells=tuple(runtime_cells),
        policy=frozen_policy,
        epicure=epicure,
        schedule_sha256=schedule_sha256,
        task_validity_sha256=task_validity_sha256,
        task_registry_sha256=registry_digest,
    )


def _verify_budget_audit(
    path: Path,
    *,
    project_root: Path,
    cap_usd: Decimal,
    admission_fraction: Decimal,
) -> tuple[Mapping[str, Any], frozenset[str]]:
    audit, _ = _load_content_addressed_artifact(
        path,
        label="frontier budget audit",
        schema_version=BUDGET_AUDIT_SCHEMA_VERSION,
    )
    inputs = audit.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise IntegrityError("frontier budget audit has no source inputs")
    if (
        audit.get("currency") != "USD"
        or audit.get("synthetic_sources") != 0
        or _decimal(audit.get("hard_cap_usd"), field="budget hard cap") != cap_usd
        or _decimal(audit.get("admission_fraction"), field="budget admission fraction")
        != admission_fraction
        or _decimal(audit.get("next_reservation_usd"), field="audit next reservation") != 0
    ):
        raise IntegrityError("frontier budget audit policy differs from the executor cap")
    seen: set[str] = set()
    actual = Decimal(0)
    exposure = Decimal(0)
    by_provider: dict[str, dict[str, Decimal | int]] = {}
    by_basis: Counter[str] = Counter()
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise IntegrityError("frontier budget audit input is malformed")
        raw_path = Path(str(record.get("source_directory") or ""))
        source_path = raw_path if raw_path.is_absolute() else project_root / raw_path
        scan = scan_live_smoke_artifacts(source_path)
        digests = sorted(item.artifact_sha256 for item in scan.artifacts)
        if seen.intersection(digests):
            raise IntegrityError("frontier budget audit repeats a source artifact")
        seen.update(digests)
        if (
            int(record.get("source_count") or -1) != len(scan.artifacts)
            or _decimal(record.get("actual_cost_usd"), field=f"budget input {index} actual")
            != scan.actual_cost_usd
            or _decimal(
                record.get("conservative_exposure_usd"),
                field=f"budget input {index} exposure",
            )
            != scan.exposure_usd
            or record.get("artifact_set_sha256") != sha256_json(digests)
        ):
            raise IntegrityError("frontier budget audit input has drifted from immutable sources")
        actual += scan.actual_cost_usd
        exposure += scan.exposure_usd
        for item in scan.artifacts:
            row = by_provider.setdefault(
                item.requested_provider,
                {"source_count": 0, "actual": Decimal(0), "exposure": Decimal(0)},
            )
            row["source_count"] = int(row["source_count"]) + 1
            row["actual"] = Decimal(row["actual"]) + item.actual_cost_usd
            row["exposure"] = Decimal(row["exposure"]) + item.exposure_usd
            by_basis[item.exposure_basis] += 1
    prior = _decimal(audit.get("prior_verified_exposure_usd"), field="prior exposure")
    current = prior + exposure
    ceiling = cap_usd * admission_fraction
    recorded_providers = audit.get("by_provider")
    if not isinstance(recorded_providers, Mapping) or set(recorded_providers) != set(by_provider):
        raise IntegrityError("frontier budget provider totals are malformed")
    for provider, row in by_provider.items():
        recorded = recorded_providers.get(provider)
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("source_count") != row["source_count"]
            or _decimal(recorded.get("actual_cost_usd"), field=f"{provider} actual")
            != row["actual"]
            or _decimal(recorded.get("conservative_exposure_usd"), field=f"{provider} exposure")
            != row["exposure"]
        ):
            raise IntegrityError("frontier budget provider total has drifted")
    exact_checks = {
        "real_source_count": len(seen),
        "real_source_actual_or_rate_card_estimate_usd": actual,
        "real_source_conservative_exposure_usd": exposure,
        "current_total_exposure_usd": current,
        "projected_total_exposure_usd": current,
        "admission_ceiling_usd": ceiling,
        "remaining_to_admission_ceiling_usd": ceiling - current,
        "remaining_to_hard_cap_usd": cap_usd - current,
    }
    for field, expected in exact_checks.items():
        observed = audit.get(field)
        if isinstance(expected, int):
            if observed != expected:
                raise IntegrityError(f"frontier budget audit {field} has drifted")
        elif _decimal(observed, field=f"budget {field}") != expected:
            raise IntegrityError(f"frontier budget audit {field} has drifted")
    if (
        audit.get("exposure_basis_counts") != dict(sorted(by_basis.items()))
        or audit.get("admission_allowed") is not (current <= ceiling)
        or audit.get("hard_cap_respected") is not (current <= cap_usd)
    ):
        raise IntegrityError("frontier budget audit aggregate state has drifted")
    return audit, frozenset(seen)


def _source_map_from_scan(
    source_directory: Path,
    *,
    corrections_directory: Path | None,
) -> tuple[Mapping[str, DatasetSource], Decimal, Decimal, frozenset[str]]:
    scan = scan_live_smoke_artifacts(
        source_directory,
        corrections_directory=corrections_directory,
    )
    sources: dict[str, DatasetSource] = {}
    for exposure in scan.artifacts:
        artifact, digest = _verify_live_artifact(exposure.path)
        work_item_id = _require_sha256(
            artifact.get("dataset_work_item_id"), field="source work-item ID"
        )
        if work_item_id in sources:
            raise IntegrityError(f"more than one source exists for work item {work_item_id}")
        sources[work_item_id] = DatasetSource(
            path=exposure.path,
            artifact_sha256=digest,
            work_item_id=work_item_id,
            artifact=artifact,
            exposure=exposure,
        )
    return (
        sources,
        scan.actual_cost_usd,
        scan.exposure_usd,
        frozenset(item.artifact_sha256 for item in scan.artifacts),
    )


def _run_accounting(
    run: SupplementalRun,
    *,
    label: str,
) -> RunAccounting:
    sources, actual, source_exposure, digests = _source_map_from_scan(
        run.source_directory,
        corrections_directory=run.corrections_directory,
    )
    ledger = load_dataset_ledger(run.ledger_path)
    reservations, finalizations = dataset_ledger_state(ledger)
    blockers: list[Mapping[str, Any]] = []
    orphan = Decimal(0)
    effective_exposure = source_exposure
    for work_item_id, reservation in reservations.items():
        reserved = _decimal(
            reservation.get("reserved_usd"),
            field=f"{label} reservation {work_item_id}",
        )
        source = sources.get(work_item_id)
        finalized = finalizations.get(work_item_id)
        if source is None and finalized is not None:
            raise IntegrityError(f"{label} finalized reservation has no source: {work_item_id}")
        if source is None:
            orphan += reserved
            blockers.append(
                {
                    "gate": "active_reservation_without_source",
                    "run": label,
                    "work_item_id": work_item_id,
                    "reservation_entry_sha256": reservation.get("entry_sha256"),
                    "reserved_usd": _decimal_text(reserved),
                }
            )
            continue
        if finalized is None and source.exposure.exposure_usd < reserved:
            # The active reservation remains authoritative until the source is
            # finalized.  Count only the difference, never the full amount twice.
            effective_exposure += reserved - source.exposure.exposure_usd
        if finalized is not None and finalized.get("source_artifact_sha256") != (
            source.artifact_sha256
        ):
            raise IntegrityError(f"{label} finalization source digest mismatch: {work_item_id}")
    unknown_sources = set(sources) - set(reservations)
    if unknown_sources:
        raise IntegrityError(f"{label} has sources without reservations: {sorted(unknown_sources)}")
    return RunAccounting(
        source_count=len(sources),
        actual_cost_usd=actual,
        exposure_usd=effective_exposure,
        orphan_reservation_usd=orphan,
        artifact_sha256s=digests,
        sources=sources,
        reservations=reservations,
        finalizations=finalizations,
        blockers=tuple(blockers),
    )


def _coverage_state(
    materialization: CoverageMaterialization,
    *,
    source_directory: Path,
    corrections_directory: Path | None,
    response_directory: Path,
    ledger_path: Path,
) -> CoverageState:
    accounting = _run_accounting(
        SupplementalRun(
            source_directory=source_directory,
            ledger_path=ledger_path,
            corrections_directory=corrections_directory,
        ),
        label="coverage_repair",
    )
    ledger = tuple(load_dataset_ledger(ledger_path))
    responses = scan_response_artifacts(response_directory)
    cells = {cell.work_item.work_item_id: cell for cell in materialization.cells}
    unknown_ledger = (set(accounting.reservations) | set(accounting.finalizations)) - set(cells)
    if unknown_ledger:
        raise IntegrityError(
            f"coverage ledger contains unknown work items: {sorted(unknown_ledger)}"
        )
    for work_item_id, reservation in accounting.reservations.items():
        cell = cells[work_item_id]
        exact = {
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "schedule_cell_id": cell.schedule_cell_id,
            "manifest_sha256": cell.route_manifest_sha256,
            "task_registry_sha256": materialization.task_registry_sha256,
            "task_id": cell.work_item.task.public_id,
            "task_family": cell.work_item.task.family,
            "prompt_sha256": cell.work_item.task.prompt_sha256,
            "model_id": cell.work_item.candidate.model_id,
            "canonical_model_slug": cell.work_item.candidate.canonical_model_slug,
            "provider_tag": cell.work_item.candidate.provider_tag,
            "execution_backend": cell.work_item.candidate.execution_backend,
            "endpoint_execution_sha256": cell.work_item.endpoint_execution_sha256,
            "execution_policy_sha256": cell.work_item.execution_policy_sha256,
            "conditions": list(cell.conditions),
            "epicure": dict(materialization.epicure),
            "reserved_usd": _decimal_text(cell.forecast.forecast_usd),
        }
        if any(reservation.get(field) != value for field, value in exact.items()):
            raise IntegrityError(
                f"coverage reservation differs from frozen work item: {work_item_id}"
            )
    for work_item_id, source in accounting.sources.items():
        cell = cells[work_item_id]
        issues = _source_postflight_issues(
            source,
            cell.work_item,
            expected_conditions=cell.conditions,
            expected_epicure=materialization.epicure,
        )
        hard_issues = sorted(set(issues) - HARD_POSTFLIGHT_EXEMPTIONS)
        if hard_issues:
            raise IntegrityError(
                f"coverage source violates frozen route/protocol {work_item_id}: {hard_issues}"
            )
    for key, response in responses.items():
        work_item_id, condition = key
        cell = cells.get(work_item_id)
        source = accounting.sources.get(work_item_id)
        if (
            cell is None
            or condition not in cell.conditions
            or source is None
            or response.source_artifact_sha256 != source.artifact_sha256
        ):
            raise IntegrityError(f"coverage response is not bound to a scheduled source: {key}")
    for work_item_id, finalization in accounting.finalizations.items():
        current_digests = sorted(
            response.artifact_sha256
            for (response_work_item, _), response in responses.items()
            if response_work_item == work_item_id
        )
        if sorted(finalization.get("response_artifact_sha256s") or []) != current_digests:
            raise IntegrityError(f"coverage finalization response digests drifted: {work_item_id}")
    return CoverageState(accounting=accounting, responses=responses, ledger=ledger)


def _global_ledger_state(
    *,
    ledger_path: Path,
    artifact_directory: Path,
    corrections_directory: Path | None,
    reconciliation_directory: Path | None,
) -> tuple[Decimal, tuple[Mapping[str, Any], ...]]:
    entries = load_frontier_ledger(ledger_path)
    scan = scan_live_smoke_artifacts(
        artifact_directory,
        corrections_directory=corrections_directory,
    )
    validate_ledger_artifact_links(
        entries,
        scan,
        reconciliation_directory=reconciliation_directory,
    )
    active = active_ledger_reservations(entries)
    total = sum(active.values(), Decimal(0))
    blockers = tuple(
        {
            "gate": "global_frontier_active_reservation",
            "reservation_entry_sha256": digest,
            "reserved_usd": _decimal_text(amount),
        }
        for digest, amount in sorted(active.items())
    )
    return total, blockers


def reconcile_budget(
    materialization: CoverageMaterialization,
    *,
    budget_audit_path: Path,
    project_root: Path,
    supplemental_runs: Sequence[SupplementalRun],
    coverage_state: CoverageState,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    cap_usd: Decimal,
    admission_fraction: Decimal,
    external_blockers: Sequence[Mapping[str, Any]] = (),
) -> BudgetState:
    audit, seen = _verify_budget_audit(
        budget_audit_path,
        project_root=project_root,
        cap_usd=cap_usd,
        admission_fraction=admission_fraction,
    )
    global_active, global_blockers = _global_ledger_state(
        ledger_path=global_ledger_path,
        artifact_directory=global_artifact_directory,
        corrections_directory=global_corrections_directory,
        reconciliation_directory=global_reconciliation_directory,
    )
    seen_digests = set(seen)
    supplemental_actual = Decimal(0)
    supplemental_exposure = Decimal(0)
    supplemental_orphan = Decimal(0)
    blockers: list[Mapping[str, Any]] = [*global_blockers, *external_blockers]
    for index, run in enumerate(supplemental_runs):
        accounting = _run_accounting(run, label=f"supplemental_{index + 1}")
        overlap = seen_digests.intersection(accounting.artifact_sha256s)
        if overlap:
            raise IntegrityError("a supplemental source is duplicated in global accounting")
        seen_digests.update(accounting.artifact_sha256s)
        supplemental_actual += accounting.actual_cost_usd
        supplemental_exposure += accounting.exposure_usd
        supplemental_orphan += accounting.orphan_reservation_usd
        blockers.extend(accounting.blockers)
    overlap = seen_digests.intersection(coverage_state.accounting.artifact_sha256s)
    if overlap:
        raise IntegrityError("a coverage source is duplicated in global accounting")
    blockers.extend(coverage_state.accounting.blockers)
    baseline = _decimal(audit.get("current_total_exposure_usd"), field="baseline current exposure")
    current = (
        baseline
        + global_active
        + supplemental_exposure
        + supplemental_orphan
        + coverage_state.accounting.exposure_usd
        + coverage_state.accounting.orphan_reservation_usd
    )
    outstanding = sum(
        (
            cell.forecast.forecast_usd
            for cell in materialization.cells
            if cell.work_item.work_item_id not in coverage_state.accounting.reservations
        ),
        Decimal(0),
    )
    return BudgetState(
        baseline_exposure_usd=baseline,
        global_active_reservation_usd=global_active,
        supplemental_actual_cost_usd=supplemental_actual,
        supplemental_exposure_usd=supplemental_exposure,
        supplemental_orphan_reservation_usd=supplemental_orphan,
        coverage_actual_cost_usd=coverage_state.accounting.actual_cost_usd,
        coverage_exposure_usd=coverage_state.accounting.exposure_usd,
        coverage_orphan_reservation_usd=coverage_state.accounting.orphan_reservation_usd,
        current_total_exposure_usd=current,
        outstanding_repair_forecast_usd=outstanding,
        projected_total_exposure_usd=current + outstanding,
        admission_ceiling_usd=cap_usd * admission_fraction,
        hard_cap_usd=cap_usd,
        blockers=tuple(blockers),
    )


def build_execution_plan(
    materialization: CoverageMaterialization,
    *,
    budget: BudgetState,
    coverage_state: CoverageState,
    budget_audit_sha256: str,
    supplemental_runs: Sequence[SupplementalRun],
    response_envelope_route_gate: Mapping[str, Any],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for cell in materialization.cells:
        work_item_id = cell.work_item.work_item_id
        if work_item_id in coverage_state.accounting.finalizations:
            decision = "skip_finalized_without_provider_call"
        elif work_item_id in coverage_state.accounting.sources:
            decision = "recover_source_without_provider_call"
        elif work_item_id in coverage_state.accounting.reservations:
            decision = "block_reserved_without_source_no_replay"
        elif budget.admission_allowed:
            decision = "admit_sequentially_after_exact_reservation"
        else:
            decision = "block_before_provider_call"
        decisions.append(
            {
                "schedule_cell_id": cell.schedule_cell_id,
                "work_item_id": work_item_id,
                "task_id": cell.work_item.task.public_id,
                "task_family": cell.work_item.task.family,
                "model_id": cell.work_item.candidate.model_id,
                "provider_tag": cell.work_item.candidate.provider_tag,
                "conditions": list(cell.conditions),
                "forecast_usd": _decimal_text(cell.forecast.forecast_usd),
                "decision": decision,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "admissible_dry_run" if budget.admission_allowed else "blocked_dry_run",
        "materialization_sha256": materialization.document["artifact_sha256"],
        "budget_sources": {
            "baseline_audit_sha256": budget_audit_sha256,
            "supplemental_runs": [
                {
                    "source_directory": str(run.source_directory),
                    "ledger_path": str(run.ledger_path),
                }
                for run in supplemental_runs
            ],
        },
        "response_envelope_route_gate": dict(response_envelope_route_gate),
        "budget": {
            "currency": "USD",
            "baseline_exposure_usd": _decimal_text(budget.baseline_exposure_usd),
            "global_active_reservation_usd": _decimal_text(budget.global_active_reservation_usd),
            "supplemental_actual_cost_usd": _decimal_text(budget.supplemental_actual_cost_usd),
            "supplemental_conservative_exposure_usd": _decimal_text(
                budget.supplemental_exposure_usd
            ),
            "supplemental_orphan_reservation_usd": _decimal_text(
                budget.supplemental_orphan_reservation_usd
            ),
            "coverage_actual_cost_usd": _decimal_text(budget.coverage_actual_cost_usd),
            "coverage_conservative_exposure_usd": _decimal_text(budget.coverage_exposure_usd),
            "coverage_orphan_reservation_usd": _decimal_text(
                budget.coverage_orphan_reservation_usd
            ),
            "current_total_exposure_usd": _decimal_text(budget.current_total_exposure_usd),
            "outstanding_repair_worst_case_usd": _decimal_text(
                budget.outstanding_repair_forecast_usd
            ),
            "projected_total_exposure_usd": _decimal_text(budget.projected_total_exposure_usd),
            "admission_ceiling_usd": _decimal_text(budget.admission_ceiling_usd),
            "hard_cap_usd": _decimal_text(budget.hard_cap_usd),
            "budget_within_limits": budget.budget_within_limits,
            "admission_allowed": budget.admission_allowed,
        },
        "blockers": list(budget.blockers),
        "decisions": decisions,
        "provider_calls_made_by_plan": 0,
        "epicure_calls_made_by_plan": 0,
        "synthetic_arms": 0,
        "official": False,
        "rank_eligible": False,
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def _write_artifact(
    document: Mapping[str, Any],
    *,
    directory: Path,
    prefix: str,
) -> Path:
    digest = _require_sha256(document.get("artifact_sha256"), field=f"{prefix} digest")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if sha256_json(payload) != digest:
        raise IntegrityError(f"{prefix} content address is invalid")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise IntegrityError(f"content-addressed output conflicts: {destination}")
        return destination
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=directory,
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


def _reservation_event(
    *,
    run_id: str,
    materialization: CoverageMaterialization,
    cell: CoverageCell,
    total_exposure_before_usd: Decimal,
) -> dict[str, Any]:
    work_item = cell.work_item
    return {
        "event_type": "reservation_created",
        "runner_run_id": run_id,
        "coverage_schedule_sha256": materialization.schedule_sha256,
        "coverage_materialization_sha256": materialization.document["artifact_sha256"],
        "schedule_cell_id": cell.schedule_cell_id,
        "work_item_id": work_item.work_item_id,
        "manifest_sha256": cell.route_manifest_sha256,
        "task_registry_sha256": materialization.task_registry_sha256,
        "task_id": work_item.task.public_id,
        "task_family": work_item.task.family,
        "prompt_sha256": work_item.task.prompt_sha256,
        "model_id": work_item.candidate.model_id,
        "canonical_model_slug": work_item.candidate.canonical_model_slug,
        "provider_tag": work_item.candidate.provider_tag,
        "execution_backend": work_item.candidate.execution_backend,
        "backend_contract_sha256": work_item.candidate.backend_contract_sha256,
        "endpoint_execution_sha256": work_item.endpoint_execution_sha256,
        "execution_policy_sha256": work_item.execution_policy_sha256,
        "conditions": list(cell.conditions),
        "existing_real_conditions_reused": list(cell.existing_conditions),
        "epicure": dict(materialization.epicure),
        "reserved_usd": _decimal_text(cell.forecast.forecast_usd),
        "total_exposure_before_usd": _decimal_text(total_exposure_before_usd),
        "derived_max_price": {
            "prompt_usd_per_mtok": _decimal_text(cell.forecast.price_envelope.prompt_usd_per_mtok),
            "completion_usd_per_mtok": _decimal_text(
                cell.forecast.price_envelope.completion_usd_per_mtok
            ),
        },
    }


def _finalize_source(
    *,
    ledger_path: Path,
    run_id: str,
    reservation: Mapping[str, Any],
    materialization: CoverageMaterialization,
    cell: CoverageCell,
    source: DatasetSource,
    response_directory: Path,
) -> tuple[Mapping[str, Any], list[ResponseArtifact], list[str]]:
    postflight = _source_postflight_issues(
        source,
        cell.work_item,
        expected_conditions=cell.conditions,
        expected_epicure=materialization.epicure,
    )
    hard_issues = sorted(set(postflight) - HARD_POSTFLIGHT_EXEMPTIONS)
    if hard_issues:
        raise IntegrityError(
            f"source cannot be finalized because route/protocol checks failed: {hard_issues}"
        )
    responses, issues = normalise_source_responses(
        source,
        cell.work_item,
        response_directory=response_directory,
        expected_conditions=cell.conditions,
        expected_epicure=materialization.epicure,
    )
    combined_issues = sorted(set(postflight + issues))
    completed_conditions = sorted(response.condition for response in responses)
    exact_complete = completed_conditions == sorted(cell.conditions)
    event = append_dataset_ledger_event(
        ledger_path,
        {
            "event_type": "source_artifact_recorded",
            "runner_run_id": run_id,
            "reservation_entry_sha256": reservation["entry_sha256"],
            "coverage_schedule_sha256": materialization.schedule_sha256,
            "coverage_materialization_sha256": materialization.document["artifact_sha256"],
            "schedule_cell_id": cell.schedule_cell_id,
            "work_item_id": cell.work_item.work_item_id,
            "manifest_sha256": cell.route_manifest_sha256,
            "task_id": cell.work_item.task.public_id,
            "task_family": cell.work_item.task.family,
            "model_id": cell.work_item.candidate.model_id,
            "provider_tag": cell.work_item.candidate.provider_tag,
            "execution_backend": cell.work_item.candidate.execution_backend,
            "execution_policy_sha256": cell.work_item.execution_policy_sha256,
            "conditions": list(cell.conditions),
            "epicure": dict(materialization.epicure),
            "source_artifact_filename": source.path.name,
            "source_artifact_sha256": source.artifact_sha256,
            "source_status": source.exposure.status,
            "source_actual_cost_usd": _decimal_text(source.exposure.actual_cost_usd),
            "source_budget_exposure_usd": _decimal_text(source.exposure.exposure_usd),
            "source_exposure_basis": source.exposure.exposure_basis,
            "response_artifact_sha256s": sorted(response.artifact_sha256 for response in responses),
            "response_conditions": completed_conditions,
            "normalization_issues": combined_issues,
            "complete_required_conditions": exact_complete,
            "safe_to_replay": False,
            "reliability_failure_retained": not exact_complete,
        },
    )
    return event, responses, combined_issues


def _budget_and_plan(
    materialization: CoverageMaterialization,
    *,
    budget_audit_path: Path,
    project_root: Path,
    supplemental_runs: Sequence[SupplementalRun],
    source_directory: Path,
    corrections_directory: Path | None,
    response_directory: Path,
    ledger_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    cap_usd: Decimal,
    admission_fraction: Decimal,
    response_envelope_route_plan_path: Path | None,
    response_envelope_route_audit_path: Path | None,
    response_envelope_route_closure_path: Path | None = None,
) -> tuple[CoverageState, BudgetState, dict[str, Any]]:
    coverage = _coverage_state(
        materialization,
        source_directory=source_directory,
        corrections_directory=corrections_directory,
        response_directory=response_directory,
        ledger_path=ledger_path,
    )
    classifier_binding = materialization.document.get("response_envelope_classifier")
    if not isinstance(classifier_binding, Mapping):
        raise IntegrityError("coverage materialization lacks its classifier binding")
    route_gate, route_gate_blockers = _response_envelope_route_gate(
        route_plan_path=response_envelope_route_plan_path,
        route_audit_path=response_envelope_route_audit_path,
        route_closure_path=response_envelope_route_closure_path,
        classifier_binding=classifier_binding,
        project_root=(
            project_root.parent
            if project_root.name == "flavourbench"
            and (project_root / "src/flavourbench").is_dir()
            else project_root
        ),
    )
    budget = reconcile_budget(
        materialization,
        budget_audit_path=budget_audit_path,
        project_root=project_root,
        supplemental_runs=supplemental_runs,
        coverage_state=coverage,
        global_ledger_path=global_ledger_path,
        global_artifact_directory=global_artifact_directory,
        global_corrections_directory=global_corrections_directory,
        global_reconciliation_directory=global_reconciliation_directory,
        cap_usd=cap_usd,
        admission_fraction=admission_fraction,
        external_blockers=route_gate_blockers,
    )
    audit, digest = _load_content_addressed_artifact(
        budget_audit_path,
        label="frontier budget audit",
        schema_version=BUDGET_AUDIT_SCHEMA_VERSION,
    )
    del audit
    plan = build_execution_plan(
        materialization,
        budget=budget,
        coverage_state=coverage,
        budget_audit_sha256=digest,
        supplemental_runs=supplemental_runs,
        response_envelope_route_gate=route_gate,
    )
    return coverage, budget, plan


def run_coverage_repair(
    *,
    schedule_path: Path,
    arena_path: Path,
    task_validity_path: Path,
    route_manifest_paths: Sequence[Path],
    budget_audit_path: Path,
    supplemental_runs: Sequence[SupplementalRun],
    project_root: Path,
    source_directory: Path,
    corrections_directory: Path | None,
    response_directory: Path,
    ledger_path: Path,
    global_ledger_path: Path,
    global_artifact_directory: Path,
    global_corrections_directory: Path | None,
    global_reconciliation_directory: Path | None,
    output_directory: Path,
    cap_usd: Decimal = AUTHORIZED_TOTAL_CAP_USD,
    admission_fraction: Decimal = DEFAULT_ADMISSION_FRACTION,
    execute: bool = False,
    confirmation: str = "",
    process_timeout_seconds: int = 3_600,
    max_new_cells: int | None = None,
    response_envelope_route_plan_path: Path | None = None,
    response_envelope_route_audit_path: Path | None = None,
    response_envelope_route_closure_path: Path | None = None,
) -> tuple[Mapping[str, Any], Path, Path]:
    """Plan or sequentially execute the exact missing-condition workload."""

    if cap_usd != AUTHORIZED_TOTAL_CAP_USD:
        raise AdmissionDenied(
            f"coverage repair cap must exactly match the shared ${AUTHORIZED_TOTAL_CAP_USD} cap"
        )
    if admission_fraction != DEFAULT_ADMISSION_FRACTION:
        raise AdmissionDenied(
            "coverage repair admission fraction must exactly match the shared 0.85 policy"
        )
    if execute and confirmation != EXECUTION_CONFIRMATION:
        raise AdmissionDenied(f"execution requires --confirm {EXECUTION_CONFIRMATION}")
    if process_timeout_seconds <= 0:
        raise ValueError("process timeout must be positive")
    if max_new_cells is not None and max_new_cells <= 0:
        raise ValueError("max_new_cells must be positive")
    materialization = build_materialization(
        schedule_path=schedule_path,
        arena_path=arena_path,
        task_validity_path=task_validity_path,
        route_manifest_paths=route_manifest_paths,
    )
    materialization_path = _write_artifact(
        materialization.document,
        directory=output_directory,
        prefix="frontier-coverage-materialization",
    )
    run_id = str(uuid.uuid4())
    started = 0
    outcomes: list[Mapping[str, Any]] = []
    with _exclusive_runner_lock(global_ledger_path):
        with _dataset_ledger_lock(ledger_path):
            coverage, budget, plan = _budget_and_plan(
                materialization,
                budget_audit_path=budget_audit_path,
                project_root=project_root,
                supplemental_runs=supplemental_runs,
                source_directory=source_directory,
                corrections_directory=corrections_directory,
                response_directory=response_directory,
                ledger_path=ledger_path,
                global_ledger_path=global_ledger_path,
                global_artifact_directory=global_artifact_directory,
                global_corrections_directory=global_corrections_directory,
                global_reconciliation_directory=global_reconciliation_directory,
                cap_usd=cap_usd,
                admission_fraction=admission_fraction,
                response_envelope_route_plan_path=response_envelope_route_plan_path,
                response_envelope_route_audit_path=response_envelope_route_audit_path,
                response_envelope_route_closure_path=(
                    response_envelope_route_closure_path
                ),
            )
            if not execute:
                outcomes = list(plan["decisions"])
            elif not budget.admission_allowed:
                raise AdmissionDenied("coverage repair is blocked by shared budget state")
            else:
                source_directory.mkdir(parents=True, exist_ok=True)
                response_directory.mkdir(parents=True, exist_ok=True)
                for cell in materialization.cells:
                    coverage, budget, _ = _budget_and_plan(
                        materialization,
                        budget_audit_path=budget_audit_path,
                        project_root=project_root,
                        supplemental_runs=supplemental_runs,
                        source_directory=source_directory,
                        corrections_directory=corrections_directory,
                        response_directory=response_directory,
                        ledger_path=ledger_path,
                        global_ledger_path=global_ledger_path,
                        global_artifact_directory=global_artifact_directory,
                        global_corrections_directory=global_corrections_directory,
                        global_reconciliation_directory=global_reconciliation_directory,
                        cap_usd=cap_usd,
                        admission_fraction=admission_fraction,
                        response_envelope_route_plan_path=(response_envelope_route_plan_path),
                        response_envelope_route_audit_path=(response_envelope_route_audit_path),
                        response_envelope_route_closure_path=(
                            response_envelope_route_closure_path
                        ),
                    )
                    work_item_id = cell.work_item.work_item_id
                    reservation = coverage.accounting.reservations.get(work_item_id)
                    source = coverage.accounting.sources.get(work_item_id)
                    if work_item_id in coverage.accounting.finalizations:
                        outcomes.append(
                            {"work_item_id": work_item_id, "decision": "skip_finalized"}
                        )
                        continue
                    if reservation is not None:
                        if source is None:
                            outcomes.append(
                                {
                                    "work_item_id": work_item_id,
                                    "decision": "stop_reserved_without_source_no_replay",
                                    "reservation_entry_sha256": reservation["entry_sha256"],
                                }
                            )
                            break
                        event, responses, issues = _finalize_source(
                            ledger_path=ledger_path,
                            run_id=run_id,
                            reservation=reservation,
                            materialization=materialization,
                            cell=cell,
                            source=source,
                            response_directory=response_directory,
                        )
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "recovered_existing_source_without_provider_call",
                                "ledger_entry_sha256": event["entry_sha256"],
                                "response_conditions": sorted(
                                    response.condition for response in responses
                                ),
                                "normalization_issues": issues,
                            }
                        )
                        continue
                    if source is not None:
                        raise IntegrityError("coverage source exists without a reservation")
                    if max_new_cells is not None and started >= max_new_cells:
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "stop_execution_batch_limit",
                                "max_new_cells": max_new_cells,
                            }
                        )
                        break
                    if not budget.admission_allowed:
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "stop_shared_budget_not_admissible",
                            }
                        )
                        break
                    reservation = append_dataset_ledger_event(
                        ledger_path,
                        _reservation_event(
                            run_id=run_id,
                            materialization=materialization,
                            cell=cell,
                            total_exposure_before_usd=budget.current_total_exposure_usd,
                        ),
                    )
                    command = _subprocess_command(
                        cell.work_item,
                        forecast=cell.forecast,
                        source_directory=source_directory,
                        manifest_path=cell.route_manifest_path,
                        conditions=cell.conditions,
                        expected_epicure=materialization.epicure,
                    )
                    environment = os.environ.copy()
                    environment.update(materialization.policy.settings_environment())
                    environment["FLAVOURBENCH_OPENROUTER_MAX_PROMPT_PRICE_PER_MTOK"] = (
                        _decimal_text(cell.forecast.price_envelope.prompt_usd_per_mtok)
                    )
                    environment["FLAVOURBENCH_OPENROUTER_MAX_COMPLETION_PRICE_PER_MTOK"] = (
                        _decimal_text(cell.forecast.price_envelope.completion_usd_per_mtok)
                    )
                    started += 1
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
                        incident = append_dataset_ledger_event(
                            ledger_path,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "subprocess_timeout_uncertain_delivery_no_replay",
                                "timeout_seconds": process_timeout_seconds,
                                "output_sha256": _safe_process_hash(str(error.output or "")),
                            },
                        )
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "timeout_reservation_retained_no_replay",
                                "incident_entry_sha256": incident["entry_sha256"],
                            }
                        )
                        break
                    artifact_path = _extract_artifact_path(completed.stdout, source_directory)
                    if artifact_path is None or not artifact_path.exists():
                        incident = append_dataset_ledger_event(
                            ledger_path,
                            {
                                "event_type": "execution_incident",
                                "runner_run_id": run_id,
                                "work_item_id": work_item_id,
                                "reservation_entry_sha256": reservation["entry_sha256"],
                                "incident": "no_verifiable_artifact_reservation_retained_no_replay",
                                "subprocess_returncode": completed.returncode,
                                "stdout_sha256": _safe_process_hash(completed.stdout),
                                "stderr_sha256": _safe_process_hash(completed.stderr),
                            },
                        )
                        outcomes.append(
                            {
                                "work_item_id": work_item_id,
                                "decision": "no_artifact_reservation_retained_no_replay",
                                "incident_entry_sha256": incident["entry_sha256"],
                            }
                        )
                        break
                    refreshed = _coverage_state(
                        materialization,
                        source_directory=source_directory,
                        corrections_directory=corrections_directory,
                        response_directory=response_directory,
                        ledger_path=ledger_path,
                    )
                    source = refreshed.accounting.sources.get(work_item_id)
                    if source is None or source.path.resolve() != artifact_path.resolve():
                        raise IntegrityError(
                            "delegated subprocess artifact is not the reserved source"
                        )
                    event, responses, issues = _finalize_source(
                        ledger_path=ledger_path,
                        run_id=run_id,
                        reservation=reservation,
                        materialization=materialization,
                        cell=cell,
                        source=source,
                        response_directory=response_directory,
                    )
                    complete = sorted(response.condition for response in responses) == sorted(
                        cell.conditions
                    )
                    outcomes.append(
                        {
                            "work_item_id": work_item_id,
                            "decision": (
                                "source_finalized_complete"
                                if complete
                                else "source_finalized_reliability_failure_stop"
                            ),
                            "source_artifact_sha256": source.artifact_sha256,
                            "ledger_entry_sha256": event["entry_sha256"],
                            "response_conditions": sorted(
                                response.condition for response in responses
                            ),
                            "normalization_issues": issues,
                        }
                    )
                    if not complete:
                        break
                coverage, budget, plan = _budget_and_plan(
                    materialization,
                    budget_audit_path=budget_audit_path,
                    project_root=project_root,
                    supplemental_runs=supplemental_runs,
                    source_directory=source_directory,
                    corrections_directory=corrections_directory,
                    response_directory=response_directory,
                    ledger_path=ledger_path,
                    global_ledger_path=global_ledger_path,
                    global_artifact_directory=global_artifact_directory,
                    global_corrections_directory=global_corrections_directory,
                    global_reconciliation_directory=global_reconciliation_directory,
                    cap_usd=cap_usd,
                    admission_fraction=admission_fraction,
                    response_envelope_route_plan_path=response_envelope_route_plan_path,
                    response_envelope_route_audit_path=response_envelope_route_audit_path,
                    response_envelope_route_closure_path=(
                        response_envelope_route_closure_path
                    ),
                )
    result_payload = {
        **{key: value for key, value in plan.items() if key != "artifact_sha256"},
        "status": ("execution_completed_or_safely_stopped" if execute else plan["status"]),
        "runner_run_id": run_id if execute else None,
        "subprocesses_started": started,
        "outcomes": outcomes,
        "generated_at": _utc_now() if execute else None,
    }
    result = {**result_payload, "artifact_sha256": sha256_json(result_payload)}
    plan_path = _write_artifact(
        result,
        directory=output_directory,
        prefix="frontier-coverage-execution-plan",
    )
    return result, materialization_path, plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, action="append", required=True)
    parser.add_argument("--budget-audit", type=Path, required=True)
    parser.add_argument(
        "--supplemental-run-root",
        type=Path,
        action="append",
        default=[],
        help="Run root containing source/, optional corrections/, and ledger.jsonl.",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/source"
        ),
    )
    parser.add_argument(
        "--corrections-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/corrections"
        ),
    )
    parser.add_argument(
        "--response-directory",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/responses"
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1/ledger.jsonl"
        ),
    )
    parser.add_argument(
        "--global-budget-lock-path",
        type=Path,
        default=Path("artifacts/frontier-contract/ledger.jsonl"),
    )
    parser.add_argument(
        "--global-artifact-directory",
        type=Path,
        default=Path("artifacts/live-smoke"),
    )
    parser.add_argument(
        "--global-corrections-directory",
        type=Path,
        default=Path("artifacts/corrections"),
    )
    parser.add_argument(
        "--global-reconciliation-directory",
        type=Path,
        default=Path("artifacts/frontier-contract/reconciliations"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/season1/current-quality-run/frontier-coverage-repair-execution-v1"),
    )
    parser.add_argument("--cap-usd", type=Decimal, default=AUTHORIZED_TOTAL_CAP_USD)
    parser.add_argument(
        "--admission-fraction",
        type=Decimal,
        default=DEFAULT_ADMISSION_FRACTION,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--response-envelope-route-plan",
        type=Path,
        help="Content-addressed v3 route-validation plan with fresh non-replayed IDs.",
    )
    parser.add_argument(
        "--response-envelope-route-audit",
        type=Path,
        help="Content-addressed v3 audit for the supplied route-validation plan.",
    )
    parser.add_argument(
        "--response-envelope-route-closure",
        type=Path,
        help=(
            "Permanent v4 identifier closure; mandatory when the supplied route plan "
            "uses the v4 source-reconstructed protocol."
        ),
    )
    parser.add_argument("--process-timeout-seconds", type=int, default=3_600)
    parser.add_argument("--max-new-cells", type=int)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    supplemental_runs = [
        SupplementalRun(
            source_directory=root / "source",
            corrections_directory=(
                root / "corrections" if (root / "corrections").exists() else None
            ),
            ledger_path=root / "ledger.jsonl",
        )
        for root in arguments.supplemental_run_root
    ]
    result, materialization_path, plan_path = run_coverage_repair(
        schedule_path=arguments.schedule,
        arena_path=arguments.arena,
        task_validity_path=arguments.task_validity,
        route_manifest_paths=arguments.route_manifest,
        budget_audit_path=arguments.budget_audit,
        supplemental_runs=supplemental_runs,
        project_root=arguments.project_root.resolve(),
        source_directory=arguments.source_directory,
        corrections_directory=arguments.corrections_directory,
        response_directory=arguments.response_directory,
        ledger_path=arguments.ledger,
        global_ledger_path=arguments.global_budget_lock_path,
        global_artifact_directory=arguments.global_artifact_directory,
        global_corrections_directory=arguments.global_corrections_directory,
        global_reconciliation_directory=arguments.global_reconciliation_directory,
        output_directory=arguments.output_directory,
        cap_usd=arguments.cap_usd,
        admission_fraction=arguments.admission_fraction,
        execute=arguments.execute,
        confirmation=arguments.confirm,
        process_timeout_seconds=arguments.process_timeout_seconds,
        max_new_cells=arguments.max_new_cells,
        response_envelope_route_plan_path=arguments.response_envelope_route_plan,
        response_envelope_route_audit_path=arguments.response_envelope_route_audit,
        response_envelope_route_closure_path=arguments.response_envelope_route_closure,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "artifact_sha256": result["artifact_sha256"],
                "materialization": str(materialization_path.resolve()),
                "plan": str(plan_path.resolve()),
                "budget": result["budget"],
                "blockers": result["blockers"],
                "subprocesses_started": result["subprocesses_started"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["budget"]["admission_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
