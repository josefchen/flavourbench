"""Freeze an offline v3 recovery after the failed reasoning-effort v2 route gate.

This module has no provider or MCP client.  It closes every v2 identifier,
binds the corrected provider-envelope semantics and exact Epicure runtime, and
materializes fresh v3 diagnostic commands without executing them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .frontier_manifest import verify_manifest_content_address
from .provider import RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES
from .reasoning_effort_sensitivity import (
    SensitivityProtocolError,
    _audit_v2_route_variant,
)

V2_PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-validation-plan-v1"
V2_AUDIT_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-validation-audit-v1"
V2_ASSETS_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-runner-assets-v1"
V2_CLOSURE_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-closed-identifiers-v1"
V3_PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-v3-route-validation-plan-v1"
V3_ASSETS_SCHEMA_VERSION = "flavourbench-reasoning-effort-v3-route-runner-assets-v1"
V3_AUDIT_SCHEMA_VERSION = "flavourbench-reasoning-effort-v3-route-validation-audit-v1"
V3_CLOSURE_SCHEMA_VERSION = "flavourbench-reasoning-effort-v3-route-closure-v1"
VARIANT_ORDER = ("explicit_low", "provider_default", "explicit_high")


class RouteRecoveryError(RuntimeError):
    """A route-recovery input or output violates its frozen boundary."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RouteRecoveryError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RouteRecoveryError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RouteRecoveryError(f"expected a JSON object: {path}")
    return value


def _artifact_verifies(document: object, schema_version: str) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != schema_version:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return isinstance(digest, str) and len(digest) == 64 and _sha256(unhashed) == digest


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise RouteRecoveryError(f"{field} is not a decimal") from error
    if not result.is_finite() or result < 0:
        raise RouteRecoveryError(f"{field} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _write_artifact(output_dir: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RouteRecoveryError("content-addressed output conflict")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _write_manifest(output_dir: Path, document: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in document.items() if key != "content_address"}
    digest = _sha256(unhashed)
    manifest = {
        **unhashed,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"flavourbench-reasoning-sensitivity-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RouteRecoveryError("content-addressed manifest conflict")
        return path
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def build_v2_closed_identifiers(
    *,
    v2_plan: Mapping[str, Any],
    v2_audit: Mapping[str, Any],
    v2_runner_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Close all planned v2 work IDs and every observed attempt/generation ID."""

    if not _artifact_verifies(v2_plan, V2_PLAN_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 plan does not verify")
    if not _artifact_verifies(v2_audit, V2_AUDIT_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 audit does not verify")
    if not _artifact_verifies(v2_runner_assets, V2_ASSETS_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 runner assets do not verify")
    if (
        v2_audit.get("v2_route_plan_sha256") != v2_plan["artifact_sha256"]
        or v2_runner_assets.get("v2_route_plan_sha256") != v2_plan["artifact_sha256"]
        or v2_audit.get("decision") != "failed_one_or_more_predicates"
        or (v2_audit.get("full_sensitivity_admission") or {}).get("authorized")
        is not False
    ):
        raise RouteRecoveryError("v2 records do not prove the failed, closed route gate")

    identifiers = v2_audit.get("identifier_freshness_audit")
    if not isinstance(identifiers, Mapping):
        raise RouteRecoveryError("v2 audit lacks its identifier inventory")
    planned_work_ids = sorted(
        str(item.get("work_item_id") or "")
        for item in v2_plan["route_validation"]["work_items"]
        if isinstance(item, Mapping)
    )
    asset_work_ids = sorted(
        str(item.get("fresh_work_item_id") or "")
        for item in v2_runner_assets.get("variants") or []
        if isinstance(item, Mapping)
    )
    audit_work_ids = sorted(str(value) for value in identifiers.get("v2_work_item_ids") or [])
    if (
        len(planned_work_ids) != 3
        or len(set(planned_work_ids)) != 3
        or planned_work_ids != asset_work_ids
        or planned_work_ids != audit_work_ids
    ):
        raise RouteRecoveryError("v2 work-item closure is incomplete")

    attempt_ids = [str(value) for value in identifiers.get("v2_attempt_ids") or []]
    raw_generation_ids = [
        str(value) for value in identifiers.get("v2_generation_ids") or []
    ]
    generation_ids = sorted(value for value in raw_generation_ids if value)
    if (
        not attempt_ids
        or len(attempt_ids) != len(set(attempt_ids))
        or len(generation_ids) != len(set(generation_ids))
        or (v2_audit.get("counts") or {}).get("attempted_pairs") != 1
        or (v2_audit.get("counts") or {}).get("usable_pairs") != 0
    ):
        raise RouteRecoveryError("v2 observed identifier closure is malformed")

    sources = v2_audit.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        raise RouteRecoveryError("v2 audit has no immutable source evidence")
    source_digests = sorted(
        str(item.get("source_artifact_sha256") or "")
        for item in sources
        if isinstance(item, Mapping)
    )
    if any(len(value) != 64 for value in source_digests):
        raise RouteRecoveryError("v2 source evidence has an invalid digest")

    failed_envelopes = (v2_audit.get("counts") or {}).get(
        "non_chat_completion_envelopes"
    )
    if failed_envelopes != 1:
        raise RouteRecoveryError("v2 closure expects the single audited error envelope")
    return {
        "schema_version": V2_CLOSURE_SCHEMA_VERSION,
        "record_role": "immutable_non_replay_boundary_for_failed_v2_route_gate",
        "v2_route_plan_sha256": v2_plan["artifact_sha256"],
        "v2_route_audit_sha256": v2_audit["artifact_sha256"],
        "v2_runner_assets_sha256": v2_runner_assets["artifact_sha256"],
        "v2_route_cell_id": v2_plan["route_validation"]["route_cell_id"],
        "work_item_ids": planned_work_ids,
        "attempt_ids": sorted(attempt_ids),
        "generation_ids": generation_ids,
        "missing_generation_id_rejections": len(raw_generation_ids) - len(generation_ids),
        "source_artifact_sha256s": source_digests,
        "failure": {
            "classification": "openrouter_error_envelope",
            "embedded_code": 429,
            "phase": "tool_round_0",
            "http_status": 200,
            "misclassified_as_generation_by_v2": True,
            "quality_output_available": False,
        },
        "closure": {
            "all_planned_v2_work_items_closed": True,
            "all_observed_v2_attempt_ids_closed": True,
            "all_observed_v2_generation_ids_closed": True,
            "replay_permitted": False,
            "v2_full_sensitivity_authorized": False,
        },
        "counts": {
            "work_item_ids": len(planned_work_ids),
            "attempt_ids": len(attempt_ids),
            "generation_ids": len(generation_ids),
            "source_artifacts": len(source_digests),
        },
        "provider_calls_made_by_builder": False,
        "epicure_calls_made_by_builder": False,
    }


def build_v3_route_closure(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
    prior_audit: Mapping[str, Any],
    corrected_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Close v3 after both conditions exhaust the allowlisted 429 retry."""

    if not verify_v3_route_plan(plan):
        raise RouteRecoveryError("v3 plan does not verify")
    if not _artifact_verifies(runner_assets, V3_ASSETS_SCHEMA_VERSION):
        raise RouteRecoveryError("v3 runner assets do not verify")
    if not _artifact_verifies(prior_audit, V3_AUDIT_SCHEMA_VERSION):
        raise RouteRecoveryError("prior v3 audit does not verify")
    if not _artifact_verifies(corrected_audit, V3_AUDIT_SCHEMA_VERSION):
        raise RouteRecoveryError("corrected v3 audit does not verify")
    if any(
        audit.get("v3_route_plan_sha256") != plan["artifact_sha256"]
        or audit.get("runner_assets_sha256") != runner_assets["artifact_sha256"]
        or audit.get("decision") != "failed_one_or_more_predicates"
        or (audit.get("full_sensitivity_admission") or {}).get("authorized") is not False
        for audit in (prior_audit, corrected_audit)
    ):
        raise RouteRecoveryError("v3 audits do not prove the failed route boundary")
    if prior_audit["artifact_sha256"] == corrected_audit["artifact_sha256"]:
        raise RouteRecoveryError("corrected v3 audit must differ from the prior audit")
    counts = corrected_audit.get("counts")
    accounting = corrected_audit.get("accounting_audit")
    identifiers = corrected_audit.get("identifier_freshness_audit")
    variants = corrected_audit.get("variant_audits")
    if not all(isinstance(value, Mapping) for value in (counts, accounting, identifiers)):
        raise RouteRecoveryError("corrected v3 audit lacks exact counts or accounting")
    if not isinstance(variants, list):
        raise RouteRecoveryError("corrected v3 audit lacks variant records")
    assert isinstance(counts, Mapping)
    assert isinstance(accounting, Mapping)
    assert isinstance(identifiers, Mapping)
    if not (
        counts.get("attempted_pairs") == 1
        and counts.get("usable_pairs") == 0
        and counts.get("provider_requests") == 4
        and counts.get("provider_responses") == 0
        and counts.get("retryable_error_envelope_rejections") == 4
        and counts.get("retried_error_envelope_rejections") == 2
        and counts.get("terminal_error_envelope_rejections") == 2
        and counts.get("unsafe_provider_rejections") == 0
        and counts.get("unreconciled_generations") == 0
        and accounting.get("identified_generation_cost_usd") == "0"
        and accounting.get("rejected_error_envelope_cost_lookups") == 0
        and identifiers.get("v3_generation_ids") == []
    ):
        raise RouteRecoveryError("corrected audit does not show the exact exhausted 429 route")

    attempted = [
        item for item in variants if isinstance(item, Mapping) and item.get("attempted") is True
    ]
    if len(attempted) != 1 or attempted[0].get("variant_id") != "explicit_low":
        raise RouteRecoveryError("v3 closure expects only the explicit-low pair")
    source_path = Path(str(attempted[0].get("source_path") or ""))
    source = _regular_json(source_path)
    source_digest = str(source.get("artifact_sha256") or "")
    source_unhashed = {key: value for key, value in source.items() if key != "artifact_sha256"}
    if (
        len(source_digest) != 64
        or _sha256(source_unhashed) != source_digest
        or source_digest != attempted[0].get("source_artifact_sha256")
    ):
        raise RouteRecoveryError("v3 failed source artifact does not verify")
    events = source.get("provider_attempt_events")
    if not isinstance(events, list) or not all(isinstance(item, Mapping) for item in events):
        raise RouteRecoveryError("v3 failed source lacks a provider trace")
    requests = [item for item in events if item.get("event_type") == "request_started"]
    rejections = [item for item in events if item.get("event_type") == "request_rejected"]
    retries = [item for item in events if item.get("event_type") == "retry_scheduled"]
    responses = [item for item in events if item.get("event_type") == "response_received"]
    accounting_events = [
        item for item in events if item.get("event_type") == "accounting_reconciled"
    ]
    arm_ids = sorted({str(item.get("arm_id") or "") for item in requests})
    attempt_ids = sorted(str(item.get("attempt_id") or "") for item in requests)
    if not (
        len(requests) == 4
        and len(rejections) == 4
        and len(retries) == 2
        and not responses
        and not accounting_events
        and len(arm_ids) == 2
        and any(value.endswith(":epicure_off") for value in arm_ids)
        and any(value.endswith(":epicure_on") for value in arm_ids)
        and all(item.get("phase") == "planning" for item in requests + rejections)
        and all(
            ((item.get("metadata") or {}).get("response_envelope") or {}).get("error_code")
            == 429
            for item in rejections
        )
        and sorted(str(value) for value in identifiers.get("v3_attempt_ids") or [])
        == attempt_ids
    ):
        raise RouteRecoveryError("v3 source trace does not match the corrected audit")

    planned_work_ids = sorted(
        str(item.get("work_item_id") or "")
        for item in plan["route_validation"]["work_items"]
        if isinstance(item, Mapping)
    )
    if planned_work_ids != sorted(str(value) for value in identifiers["v3_work_item_ids"]):
        raise RouteRecoveryError("v3 planned work-item closure is incomplete")
    journal = source.get("run_journal")
    if not isinstance(journal, Mapping) or journal.get("finalized") is not True:
        raise RouteRecoveryError("v3 failed source has no finalized journal")
    return {
        "schema_version": V3_CLOSURE_SCHEMA_VERSION,
        "record_role": "immutable_non_replay_boundary_for_exhausted_v3_route_gate",
        "v3_route_plan_sha256": plan["artifact_sha256"],
        "v3_runner_assets_sha256": runner_assets["artifact_sha256"],
        "prior_audit_sha256": prior_audit["artifact_sha256"],
        "corrected_audit_sha256": corrected_audit["artifact_sha256"],
        "audit_correction": {
            "prior_defect": (
                "terminal allowlisted rejections were incorrectly required to schedule "
                "a third attempt"
            ),
            "corrected_interpretation": (
                "two first attempts were safely retried and two second attempts were safe "
                "terminal rejections at the frozen two-attempt limit"
            ),
            "changes_route_outcome": False,
        },
        "closed_identifiers": {
            "route_cell_id": plan["route_validation"]["route_cell_id"],
            "work_item_ids": planned_work_ids,
            "attempt_ids": attempt_ids,
            "generation_ids": [],
            "arm_ids": arm_ids,
            "replay_permitted": False,
        },
        "immutable_evidence": {
            "source_artifact_sha256": source_digest,
            "summary_artifact_sha256": attempted[0]["summary_artifact_sha256"],
            "journal_sha256": journal["sha256"],
            "journal_head_entry_sha256": journal["head_entry_sha256"],
            "ledger_head_sha256": attempted[0]["ledger_head_sha256"],
        },
        "observed_route_failure": {
            "provider_endpoint": plan["route_validation"]["work_items"][0][
                "provider_endpoint"
            ],
            "model_id": plan["route_validation"]["model_id"],
            "phase": "planning",
            "conditions": ["epicure_off", "epicure_on"],
            "provider_requests": 4,
            "accepted_generations": 0,
            "allowlisted_error_envelope_code": 429,
            "retried_rejections": 2,
            "terminal_rejections": 2,
            "unsafe_rejections": 0,
            "provider_detail_retained": False,
            "locatable_failure_boundary": (
                "Cloudflare/OpenRouter route admission; the redacted envelope cannot "
                "distinguish gateway, router, or fixed upstream-provider throttling"
            ),
        },
        "cost": {
            "identified_generation_cost_usd": "0",
            "conservative_retained_exposure_usd": accounting[
                "conservative_retained_exposure_usd"
            ],
            "post_route_conservative_exposure_usd": accounting[
                "post_route_conservative_exposure_usd"
            ],
            "rejected_error_envelope_cost_lookups": 0,
        },
        "decision": {
            "v3_status": "closed_failed_route_unavailable_during_authorized_gate",
            "v4_materialized": False,
            "v4_authorized": False,
            "reason": (
                "No accepted generation occurred in either arm after the full frozen retry "
                "allowance. A paid v4 retry is unsupported until a separately governed route "
                "availability or endpoint change supplies new evidence."
            ),
            "safe_next_action": (
                "retain closure and perform no provider call; requalify or change the fixed "
                "endpoint before freezing any future route gate"
            ),
        },
        "provider_calls_made_by_builder": False,
        "epicure_calls_made_by_builder": False,
    }


def _provider_retry_contract(provider_source_sha256: str) -> dict[str, Any]:
    retryable_codes = sorted(RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES)
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
        "retryable_error_codes": retryable_codes,
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
    return {**contract, "contract_sha256": _sha256(contract)}


def build_v3_route_plan(
    *,
    v2_plan: Mapping[str, Any],
    v2_audit: Mapping[str, Any],
    v2_closed_identifiers: Mapping[str, Any],
    provider_source_path: Path,
) -> dict[str, Any]:
    """Freeze fresh v3 IDs against the repaired provider and current exposure."""

    if not _artifact_verifies(v2_plan, V2_PLAN_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 plan does not verify")
    if not _artifact_verifies(v2_audit, V2_AUDIT_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 audit does not verify")
    if not _artifact_verifies(v2_closed_identifiers, V2_CLOSURE_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 identifier closure does not verify")
    if (
        v2_audit.get("v2_route_plan_sha256") != v2_plan["artifact_sha256"]
        or v2_closed_identifiers.get("v2_route_plan_sha256")
        != v2_plan["artifact_sha256"]
        or v2_closed_identifiers.get("v2_route_audit_sha256")
        != v2_audit["artifact_sha256"]
        or (v2_closed_identifiers.get("closure") or {}).get("replay_permitted")
        is not False
    ):
        raise RouteRecoveryError("v3 source chain does not close v2")
    if provider_source_path.is_symlink() or not provider_source_path.is_file():
        raise RouteRecoveryError("provider source must be a regular non-symlink file")
    provider_source_sha256 = _file_sha256(provider_source_path)
    if provider_source_sha256 == (v2_plan.get("source") or {}).get(
        "provider_source_sha256"
    ):
        raise RouteRecoveryError("v3 provider source must contain the audited recovery")
    source_text = provider_source_path.read_text(encoding="utf-8")
    required_markers = (
        "RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES",
        '"event_type": "request_rejected"',
        '"event_type": "retry_scheduled"',
        '"generation_id": generation_id',
    )
    if not all(marker in source_text for marker in required_markers):
        raise RouteRecoveryError("provider source lacks the v3 retry/accounting boundary")
    classifier = _provider_retry_contract(provider_source_sha256)

    v1_closed = (v2_plan.get("closed_v1_identifiers") or {}).get("work_item_ids") or []
    v2_closed = v2_closed_identifiers.get("work_item_ids") or []
    closed_work_ids = sorted({str(value) for value in [*v1_closed, *v2_closed]})
    if len(closed_work_ids) != len(set(v1_closed)) + len(set(v2_closed)):
        raise RouteRecoveryError("v1 and v2 closed work-item boundaries overlap")

    old_route = v2_plan.get("route_validation")
    if not isinstance(old_route, Mapping):
        raise RouteRecoveryError("v2 plan has no route-validation design")
    old_items = old_route.get("work_items")
    if not isinstance(old_items, list) or len(old_items) != 3:
        raise RouteRecoveryError("v2 route does not contain three variants")

    cell_core = {
        "schema_version": "flavourbench-reasoning-effort-v3-route-cell-v1",
        "study_id": "frontier-reasoning-effort-sensitivity-v3",
        "model_id": old_route["model_id"],
        "task_id": old_route["task_id"],
        "task_family": old_route["task_family"],
        "provider_source_sha256": provider_source_sha256,
        "classifier_contract_sha256": classifier["contract_sha256"],
        "v2_closed_identifiers_sha256": v2_closed_identifiers["artifact_sha256"],
        "lineage_inventory_sha256": v2_plan["epicure"]["lineage_inventory_sha256"],
        "bundle_sha256": v2_plan["epicure"]["bundle_sha256"],
        "application_sha256": v2_plan["epicure"]["application_sha256"],
        "tool_schema_sha256": v2_plan["epicure"]["tool_schema_sha256"],
    }
    route_cell_id = _sha256(cell_core)
    old_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in old_items
        if isinstance(item, Mapping)
    }
    if tuple(old_by_variant) != VARIANT_ORDER:
        raise RouteRecoveryError("v2 variant order is not the frozen three-variant order")
    work_items: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        old = old_by_variant[variant_id]
        core = {
            **{
                key: value
                for key, value in old.items()
                if key
                not in {
                    "classifier_contract_sha256",
                    "provider_source_sha256",
                    "route_cell_id",
                    "schema_version",
                    "study_id",
                    "v1_closed_identifiers_sha256",
                    "work_item_id",
                }
            },
            "schema_version": "flavourbench-reasoning-effort-v3-route-work-item-v1",
            "study_id": "frontier-reasoning-effort-sensitivity-v3",
            "route_cell_id": route_cell_id,
            "classifier_contract_sha256": classifier["contract_sha256"],
            "provider_source_sha256": provider_source_sha256,
            "v2_closed_identifiers_sha256": v2_closed_identifiers["artifact_sha256"],
        }
        work_item_id = _sha256(core)
        if work_item_id in closed_work_ids:
            raise RouteRecoveryError("v3 work-item ID overlaps a closed v1/v2 ID")
        work_items.append({"work_item_id": work_item_id, **core})

    accounting = v2_audit.get("accounting_audit")
    if not isinstance(accounting, Mapping):
        raise RouteRecoveryError("v2 audit has no accounting boundary")
    current_exposure = _decimal(
        accounting.get("post_route_conservative_exposure_usd"),
        field="post-v2 conservative exposure",
    )
    route_reserve = sum(
        (
            _decimal(item["worst_case_reserve_usd"], field="v3 route reserve")
            for item in work_items
        ),
        Decimal(0),
    )
    admission_ceiling = _decimal(
        accounting.get("admission_ceiling_usd"), field="admission ceiling"
    )
    projected = current_exposure + route_reserve
    blockers: list[dict[str, str]] = []
    if projected > admission_ceiling:
        blockers.append(
            {
                "gate": "shared_budget_admission",
                "reason": (
                    f"v3 route projects ${_decimal_text(projected)} above "
                    f"${_decimal_text(admission_ceiling)}"
                ),
            }
        )
    if v2_audit.get("decision") != "failed_one_or_more_predicates":
        blockers.append(
            {"gate": "v2_failure_boundary", "reason": "v2 failure is not verified"}
        )

    predicates = copy.deepcopy(v2_plan.get("acceptance_gate", {}).get("predicates") or [])
    predicates.append(
        {
            "predicate_id": "retryable_error_envelope_safety",
            "requirement": (
                "any allowlisted error envelope is a request_rejected event with no "
                "generation ID or cost lookup and is followed only by one fresh attempt"
            ),
        }
    )
    return {
        "schema_version": V3_PLAN_SCHEMA_VERSION,
        "study_id": "frontier-reasoning-effort-sensitivity-v3",
        "plan_role": "no_call_recovery_route_validation_after_failed_v2",
        "supersedes_for_future_execution": v2_plan["artifact_sha256"],
        "does_not_supersede_raw_v1_or_v2_records": True,
        "closed_work_item_ids_never_replayed": closed_work_ids,
        "closed_identifiers": {
            "v1": {
                "inventory_sha256": v2_plan["closed_v1_identifiers"][
                    "inventory_sha256"
                ],
                "work_item_ids": v2_plan["closed_v1_identifiers"]["work_item_ids"],
                "attempt_ids": v2_plan["closed_v1_identifiers"]["attempt_ids"],
                "generation_ids": v2_plan["closed_v1_identifiers"]["generation_ids"],
            },
            "v2": {
                "inventory_sha256": v2_closed_identifiers["artifact_sha256"],
                "work_item_ids": v2_closed_identifiers["work_item_ids"],
                "attempt_ids": v2_closed_identifiers["attempt_ids"],
                "generation_ids": v2_closed_identifiers["generation_ids"],
            },
        },
        "source": {
            "v2_route_plan_sha256": v2_plan["artifact_sha256"],
            "v2_route_audit_sha256": v2_audit["artifact_sha256"],
            "v2_closed_identifiers_sha256": v2_closed_identifiers["artifact_sha256"],
            "v1_closed_identifiers_sha256": v2_plan["source"][
                "v1_closed_identifiers_sha256"
            ],
            "base_manifest_sha256": v2_plan["source"]["base_manifest_sha256"],
            "provider_source_path": str(provider_source_path),
            "provider_source_sha256": provider_source_sha256,
            "corrected_lineage_inventory_sha256": v2_plan["source"][
                "corrected_lineage_inventory_sha256"
            ],
            "lineage_correction_sha256": v2_plan["source"][
                "lineage_correction_sha256"
            ],
        },
        "epicure": copy.deepcopy(v2_plan["epicure"]),
        "safe_response_envelope_contract": classifier,
        "route_validation": {
            "cell_count": 1,
            "route_cell_id": route_cell_id,
            "model_id": old_route["model_id"],
            "task_id": old_route["task_id"],
            "task_family": old_route["task_family"],
            "effort_variants": 3,
            "matched_pairs": 3,
            "response_arms": 6,
            "synthetic_arms": 0,
            "work_items": work_items,
            "execution_order": list(VARIANT_ORDER),
            "scheduling": "strictly_sequential_one_pair_at_a_time",
            "diagnostic_outputs_enter_quality_fit": False,
        },
        "acceptance_gate": {
            "decision_rule": "all predicates pass; first terminal failure closes all v3 IDs",
            "predicates": predicates,
            "minimum_usable_pairs": 3,
            "minimum_usable_arms": 6,
            "minimum_successful_epicure_tool_calls": 3,
            "permitted_identity_mismatches": 0,
            "permitted_unreconciled_generations": 0,
            "on_failure": (
                "retain the failed pair reserve, close every v3 ID, and freeze a new "
                "content-addressed recovery before another paid call"
            ),
        },
        "full_sensitivity": {
            "status": "blocked_pending_v3_route_validation",
            "matched_pairs": 36,
            "response_arms": 72,
            "models": 3,
            "tasks": 4,
            "effort_variants": 3,
            "route_validation_outputs_reused": False,
        },
        "budget": {
            "currency": "USD",
            "post_v2_conservative_exposure_usd": _decimal_text(current_exposure),
            "v3_route_validation_worst_case_usd": _decimal_text(route_reserve),
            "projected_post_route_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(admission_ceiling),
            "admitted": not blockers,
            "accounting_source_sha256": v2_audit["artifact_sha256"],
        },
        "preflight": {
            "decision": (
                "blocked_before_provider_call"
                if blockers
                else "ready_to_materialize_v3_route_validation_only"
            ),
            "collection_blockers": blockers,
            "provider_calls_made": False,
            "epicure_calls_made": False,
        },
        "claim_boundary": {
            "quality_results_available": False,
            "sensitivity_effect_estimable": False,
            "route_validation_only": True,
            "official": False,
            "rank_eligible": False,
            "synthetic_arms": 0,
        },
    }


def verify_v3_route_plan(document: object) -> bool:
    if not _artifact_verifies(document, V3_PLAN_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    route = document.get("route_validation")
    preflight = document.get("preflight")
    budget = document.get("budget")
    contract = document.get("safe_response_envelope_contract")
    claims = document.get("claim_boundary")
    closed = document.get("closed_identifiers")
    if not all(
        isinstance(value, Mapping)
        for value in (route, preflight, budget, contract, claims, closed)
    ):
        return False
    assert isinstance(route, Mapping)
    assert isinstance(preflight, Mapping)
    assert isinstance(budget, Mapping)
    assert isinstance(contract, Mapping)
    assert isinstance(claims, Mapping)
    assert isinstance(closed, Mapping)
    work_items = route.get("work_items")
    if not isinstance(work_items, list):
        return False
    try:
        projected = _decimal(
            budget.get("projected_post_route_exposure_usd"), field="projected"
        )
        ceiling = _decimal(budget.get("admission_ceiling_usd"), field="ceiling")
    except RouteRecoveryError:
        return False
    contract_digest = contract.get("contract_sha256")
    contract_unhashed = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    closed_v1 = closed.get("v1")
    closed_v2 = closed.get("v2")
    return (
        len(work_items) == 3
        and len({str(item.get("work_item_id")) for item in work_items}) == 3
        and route.get("execution_order") == list(VARIANT_ORDER)
        and route.get("response_arms") == 6
        and route.get("synthetic_arms") == 0
        and route.get("diagnostic_outputs_enter_quality_fit") is False
        and preflight.get("decision")
        in {
            "ready_to_materialize_v3_route_validation_only",
            "blocked_before_provider_call",
        }
        and preflight.get("provider_calls_made") is False
        and preflight.get("epicure_calls_made") is False
        and budget.get("admitted") is (projected <= ceiling)
        and contract.get("retryable_error_codes")
        == sorted(RETRYABLE_OPENROUTER_ERROR_ENVELOPE_CODES)
        and contract.get("http_200_non_chat_action")
        == "retry_allowlisted_error_envelopes_without_generation_or_cost_reconciliation"
        and isinstance(contract_digest, str)
        and contract_digest == _sha256(contract_unhashed)
        and isinstance(closed_v1, Mapping)
        and isinstance(closed_v2, Mapping)
        and len(set(closed_v1.get("work_item_ids") or [])) == 6
        and len(set(closed_v2.get("work_item_ids") or [])) == 3
        and claims.get("quality_results_available") is False
        and claims.get("official") is False
        and claims.get("rank_eligible") is False
    )


def verify_v3_route_closure(document: object) -> bool:
    """Verify the fail-closed boundary for the exhausted v3 route."""

    if not _artifact_verifies(document, V3_CLOSURE_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    identifiers = document.get("closed_identifiers")
    failure = document.get("observed_route_failure")
    cost = document.get("cost")
    decision = document.get("decision")
    correction = document.get("audit_correction")
    if not all(
        isinstance(value, Mapping)
        for value in (identifiers, failure, cost, decision, correction)
    ):
        return False
    assert isinstance(identifiers, Mapping)
    assert isinstance(failure, Mapping)
    assert isinstance(cost, Mapping)
    assert isinstance(decision, Mapping)
    assert isinstance(correction, Mapping)
    return bool(
        len(set(identifiers.get("work_item_ids") or [])) == 3
        and len(set(identifiers.get("attempt_ids") or [])) == 4
        and identifiers.get("generation_ids") == []
        and len(set(identifiers.get("arm_ids") or [])) == 2
        and identifiers.get("replay_permitted") is False
        and failure.get("provider_requests") == 4
        and failure.get("accepted_generations") == 0
        and failure.get("allowlisted_error_envelope_code") == 429
        and failure.get("retried_rejections") == 2
        and failure.get("terminal_rejections") == 2
        and failure.get("unsafe_rejections") == 0
        and cost.get("identified_generation_cost_usd") == "0"
        and cost.get("rejected_error_envelope_cost_lookups") == 0
        and correction.get("changes_route_outcome") is False
        and decision.get("v3_status")
        == "closed_failed_route_unavailable_during_authorized_gate"
        and decision.get("v4_materialized") is False
        and decision.get("v4_authorized") is False
        and document.get("provider_calls_made_by_builder") is False
        and document.get("epicure_calls_made_by_builder") is False
    )


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise RouteRecoveryError(f"source command has no {option}") from error
    if index + 1 >= len(command):
        raise RouteRecoveryError(f"source command has no value for {option}")
    command[index + 1] = value


def materialize_v3_route_assets(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    v2_runner_assets: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write exact, no-call manifests and commands for the three v3 pairs."""

    if not verify_v3_route_plan(plan):
        raise RouteRecoveryError("v3 plan does not verify")
    if _regular_json(plan_path) != dict(plan):
        raise RouteRecoveryError("v3 plan path differs from the supplied plan")
    if not _artifact_verifies(v2_runner_assets, V2_ASSETS_SCHEMA_VERSION):
        raise RouteRecoveryError("v2 runner assets do not verify")
    if v2_runner_assets.get("v2_route_plan_sha256") != plan["source"][
        "v2_route_plan_sha256"
    ]:
        raise RouteRecoveryError("v2 assets are outside the v3 source chain")
    source_variants = v2_runner_assets.get("variants")
    if not isinstance(source_variants, list) or len(source_variants) != 3:
        raise RouteRecoveryError("v2 assets do not contain three variants")
    source_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in source_variants
        if isinstance(item, Mapping)
    }
    planned_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in plan["route_validation"]["work_items"]
        if isinstance(item, Mapping)
    }
    if set(source_by_variant) != set(VARIANT_ORDER) or set(planned_by_variant) != set(
        VARIANT_ORDER
    ):
        raise RouteRecoveryError("v3 variant map is incomplete")

    variants: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        source_asset = source_by_variant[variant_id]
        planned = planned_by_variant[variant_id]
        source_manifest_path = Path(str(source_asset.get("manifest") or ""))
        source_manifest = _regular_json(source_manifest_path)
        if (
            not verify_manifest_content_address(source_manifest)
            or (source_manifest.get("content_address") or {}).get("digest")
            != source_asset.get("manifest_sha256")
        ):
            raise RouteRecoveryError(f"{variant_id} v2 manifest does not verify")
        manifest = copy.deepcopy(source_manifest)
        manifest.pop("content_address", None)
        source = dict(manifest.get("source") or {})
        for key in tuple(source):
            if key.startswith("v2_route_") or key == "provider_source_sha256":
                source.pop(key)
        source.update(
            {
                "v3_route_validation_plan_sha256": plan["artifact_sha256"],
                "v2_route_audit_sha256": plan["source"]["v2_route_audit_sha256"],
                "v2_closed_identifiers_sha256": plan["source"][
                    "v2_closed_identifiers_sha256"
                ],
                "provider_source_sha256": plan["source"]["provider_source_sha256"],
            }
        )
        manifest["source"] = source
        design = dict(manifest.get("run_design") or {})
        generation = dict(design.get("generation_protocol") or {})
        for key in tuple(generation):
            if key.startswith("v2_"):
                generation.pop(key)
        generation.update(
            {
                "v3_route_validation_plan_sha256": plan["artifact_sha256"],
                "v3_route_cell_id": plan["route_validation"]["route_cell_id"],
                "v3_route_variant_id": variant_id,
                "v3_effective_work_item_id": planned["work_item_id"],
            }
        )
        design["generation_protocol"] = generation
        design["route_validation_override"] = {
            "plan_sha256": plan["artifact_sha256"],
            "route_revision": "v3",
            "variant_id": variant_id,
            "model_id": planned["model_id"],
            "task_id": planned["task_id"],
            "work_item_id": planned["work_item_id"],
            "matched_pairs": 1,
            "response_arms": 2,
            "diagnostic_only": True,
        }
        manifest["run_design"] = design
        governance = dict(manifest.get("governance") or {})
        governance.pop("v2_route_validation_only", None)
        governance.update(
            {
                "manifest_class": "real_reasoning_effort_v3_route_validation",
                "v3_route_validation_only": True,
                "official": False,
                "rank_eligible": False,
                "ranking_use": "prohibited",
            }
        )
        manifest["governance"] = governance
        manifest["budget"] = {
            **dict(manifest.get("budget") or {}),
            "bounded_forecast_usd": planned["worst_case_reserve_usd"],
            "forecast_policy": "v3_one_pair_route_validation_reservation",
            "source_plan_sha256": plan["artifact_sha256"],
        }
        manifest_path = _write_manifest(output_dir / "manifests", manifest)
        written_manifest = _regular_json(manifest_path)
        manifest_sha256 = str(written_manifest["content_address"]["digest"])

        command = list(source_asset.get("dry_run_command") or [])
        if not command or not all(isinstance(value, str) for value in command):
            raise RouteRecoveryError(f"{variant_id} source command is malformed")
        run_root = output_dir / "runs" / variant_id
        replacements = {
            "--manifest": str(manifest_path),
            "--expected-manifest-sha256": manifest_sha256,
            "--source-directory": str(run_root / "source"),
            "--source-corrections-directory": str(run_root / "corrections"),
            "--source-resolution-directory": str(run_root / "resolutions"),
            "--response-directory": str(run_root / "responses"),
            "--ledger": str(run_root / "ledger.jsonl"),
            "--summary-directory": str(run_root / "summaries"),
            "--route-validation-plan": str(plan_path),
            "--route-validation-variant": variant_id,
        }
        for option, value in replacements.items():
            _replace_option(command, option, value)
        variants.append(
            {
                "variant_id": variant_id,
                "fresh_work_item_id": planned["work_item_id"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "run_root": str(run_root),
                "dry_run_command": command,
                "live_command": [
                    *command,
                    "--execute",
                    "--confirm",
                    "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
                ],
            }
        )
    return {
        "schema_version": V3_ASSETS_SCHEMA_VERSION,
        "record_role": "exact_no_call_commands_for_three_fresh_v3_diagnostic_pairs",
        "v3_route_plan_sha256": plan["artifact_sha256"],
        "v2_runner_assets_sha256": v2_runner_assets["artifact_sha256"],
        "v2_closed_identifiers_sha256": plan["source"][
            "v2_closed_identifiers_sha256"
        ],
        "task_dossier": v2_runner_assets["task_dossier"],
        "task_dossier_sha256": v2_runner_assets["task_dossier_sha256"],
        "variants": variants,
        "execution_order": list(VARIANT_ORDER),
        "safe_bash_execution": {
            "shell": "bash",
            "method": "mapfile_then_array_expansion",
            "prohibited_method": "jq_NUL_stream_piped_to_xargs_without_command",
            "command_template": (
                "mapfile -d '' -t cmd < <(jq -j --arg v \"$variant\" "
                "'.variants[] | select(.variant_id == $v) | .live_command[] + \"\\u0000\"' "
                "\"$ASSETS\"); (( ${#cmd[@]} > 0 )); \"${cmd[@]}\""
            ),
        },
        "operator_rule": (
            "Run all dry commands first. Execute variants in frozen order, derive an audit "
            "after each pair, and stop on the first terminal predicate failure. Never replay "
            "a v1, v2, or v3 work-item ID."
        ),
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "official": False,
        "rank_eligible": False,
    }


def _command_option(command: Sequence[str], option: str) -> Path:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as error:
        raise RouteRecoveryError(f"runner command has no {option} value") from error
    return Path(value)


def _v3_runner_contract(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify exact v3 assets and return bindings for source-derived auditing."""

    if not verify_v3_route_plan(plan):
        raise RouteRecoveryError("v3 route plan does not verify")
    if not _artifact_verifies(runner_assets, V3_ASSETS_SCHEMA_VERSION):
        raise RouteRecoveryError("v3 runner assets do not verify")
    if runner_assets.get("v3_route_plan_sha256") != plan["artifact_sha256"]:
        raise RouteRecoveryError("v3 runner assets differ from the plan")
    variants = runner_assets.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        raise RouteRecoveryError("v3 runner assets lack three variants")
    assets_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in variants
        if isinstance(item, Mapping)
    }
    planned_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in plan["route_validation"]["work_items"]
        if isinstance(item, Mapping)
    }
    if (
        set(assets_by_variant) != set(VARIANT_ORDER)
        or set(planned_by_variant) != set(VARIANT_ORDER)
        or runner_assets.get("execution_order") != list(VARIANT_ORDER)
    ):
        raise RouteRecoveryError("v3 route variant set or order differs")

    bindings: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        asset = assets_by_variant[variant_id]
        planned = planned_by_variant[variant_id]
        dry = asset.get("dry_run_command")
        live = asset.get("live_command")
        if not (
            isinstance(dry, list)
            and isinstance(live, list)
            and all(isinstance(value, str) for value in [*dry, *live])
            and live
            == [
                *dry,
                "--execute",
                "--confirm",
                "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
            ]
        ):
            raise RouteRecoveryError(f"{variant_id} commands are not exact arrays")
        run_root = Path(str(asset.get("run_root") or ""))
        expected_paths = {
            "--source-directory": run_root / "source",
            "--source-corrections-directory": run_root / "corrections",
            "--source-resolution-directory": run_root / "resolutions",
            "--response-directory": run_root / "responses",
            "--ledger": run_root / "ledger.jsonl",
            "--summary-directory": run_root / "summaries",
        }
        if not str(run_root) or any(
            _command_option(dry, option) != expected
            for option, expected in expected_paths.items()
        ):
            raise RouteRecoveryError(f"{variant_id} run paths are not isolated")
        if (
            _command_option(dry, "--route-validation-plan").name
            != f"reasoning-effort-v3-route-validation-plan-{plan['artifact_sha256']}.json"
            or str(dry[dry.index("--route-validation-variant") + 1]) != variant_id
            or asset.get("fresh_work_item_id") != planned["work_item_id"]
        ):
            raise RouteRecoveryError(f"{variant_id} route override differs from the plan")

        manifest_path = Path(str(asset.get("manifest") or ""))
        manifest = _regular_json(manifest_path)
        design = manifest.get("run_design")
        source = manifest.get("source")
        governance = manifest.get("governance")
        manifest_sha256 = str((manifest.get("content_address") or {}).get("digest") or "")
        if not (
            verify_manifest_content_address(manifest)
            and manifest_sha256 == asset.get("manifest_sha256")
            and isinstance(design, Mapping)
            and isinstance(source, Mapping)
            and isinstance(governance, Mapping)
        ):
            raise RouteRecoveryError(f"{variant_id} manifest does not verify")
        override = design.get("route_validation_override")
        generation = design.get("generation_protocol")
        models = [
            item
            for item in manifest.get("models") or []
            if isinstance(item, Mapping)
            and (item.get("model") or {}).get("id") == planned["model_id"]
        ]
        if not (
            isinstance(override, Mapping)
            and isinstance(generation, Mapping)
            and len(models) == 1
            and design.get("execution_policy_sha256")
            == planned["execution_policy_sha256"]
            and override.get("plan_sha256") == plan["artifact_sha256"]
            and override.get("route_revision") == "v3"
            and override.get("variant_id") == variant_id
            and override.get("work_item_id") == planned["work_item_id"]
            and generation.get("v3_route_validation_plan_sha256")
            == plan["artifact_sha256"]
            and generation.get("v3_route_cell_id")
            == plan["route_validation"]["route_cell_id"]
            and generation.get("v3_route_variant_id") == variant_id
            and generation.get("v3_effective_work_item_id")
            == planned["work_item_id"]
            and generation.get("required_tool_contract_sha256")
            == planned["required_tool_contract_sha256"]
            and source.get("provider_source_sha256")
            == plan["source"]["provider_source_sha256"]
            and source.get("v2_closed_identifiers_sha256")
            == plan["source"]["v2_closed_identifiers_sha256"]
            and governance.get("v3_route_validation_only") is True
            and governance.get("official") is False
            and governance.get("rank_eligible") is False
        ):
            raise RouteRecoveryError(f"{variant_id} manifest bindings differ")
        model = models[0].get("model") or {}
        endpoint = models[0].get("endpoint") or {}
        if not (
            model.get("canonical_slug") == planned["canonical_model_slug"]
            and endpoint.get("tag") == planned["provider_endpoint"]
            and endpoint.get("provider_name") == planned["actual_provider_name"]
            and models[0].get("endpoint_document_sha256")
            == planned["endpoint_document_sha256"]
        ):
            raise RouteRecoveryError(f"{variant_id} endpoint identity differs")
        bindings.append(
            {
                "variant_id": variant_id,
                "asset": asset,
                "planned": planned,
                "manifest": manifest,
                "manifest_sha256": manifest_sha256,
                "run_root": run_root,
            }
        )
    return bindings


def _retry_envelope_audit(
    *,
    record: dict[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit v3 retry events without treating rejections as generations."""

    predicate_id = "retryable_error_envelope_safety"
    issues = record["issues"]
    if not record["attempted"] or not record.get("source_path"):
        record.update(
            {
                "provider_rejections": 0,
                "safe_retryable_error_envelope_rejections": 0,
                "retried_error_envelope_rejections": 0,
                "terminal_error_envelope_rejections": 0,
                "unsafe_provider_rejections": 0,
                "accounted_provider_responses": 0,
            }
        )
        return record
    source = _regular_json(Path(record["source_path"]))
    events = source.get("provider_attempt_events")
    if not isinstance(events, list) or not all(isinstance(item, Mapping) for item in events):
        raise RouteRecoveryError("v3 source has an invalid provider event trace")
    requests = [item for item in events if item.get("event_type") == "request_started"]
    responses = [item for item in events if item.get("event_type") == "response_received"]
    rejections = [item for item in events if item.get("event_type") == "request_rejected"]
    retries = [item for item in events if item.get("event_type") == "retry_scheduled"]
    accounting = [item for item in events if item.get("event_type") == "accounting_reconciled"]
    request_by_id = {str(item.get("attempt_id") or ""): item for item in requests}
    response_by_id = {str(item.get("attempt_id") or ""): item for item in responses}
    rejection_by_id = {str(item.get("attempt_id") or ""): item for item in rejections}
    retry_by_id = {str(item.get("attempt_id") or ""): item for item in retries}
    accounting_by_id = {str(item.get("attempt_id") or ""): item for item in accounting}
    unsafe: list[str] = []
    if (
        len(request_by_id) != len(requests)
        or len(response_by_id) != len(responses)
        or len(rejection_by_id) != len(rejections)
        or len(retry_by_id) != len(retries)
        or len(accounting_by_id) != len(accounting)
    ):
        unsafe.append("duplicate_attempt_lifecycle_event")
    request_ids = set(request_by_id)
    response_ids = set(response_by_id)
    rejection_ids = set(rejection_by_id)
    if request_ids != response_ids | rejection_ids or response_ids & rejection_ids:
        unsafe.append("request_terminal_partition_failed")
    if response_ids != set(accounting_by_id):
        unsafe.append("accepted_generation_accounting_bijection_failed")

    allowed_codes = set(plan["safe_response_envelope_contract"]["retryable_error_codes"])
    safe_rejections = 0
    retried_rejections = 0
    terminal_rejections = 0
    expected_retry_ids: set[str] = set()
    maximum_attempts = int(
        plan["safe_response_envelope_contract"]["attempt_semantics"]
        ["maximum_provider_attempts_per_phase"]
    )
    for attempt_id, rejected in rejection_by_id.items():
        metadata = rejected.get("metadata")
        envelope = metadata.get("response_envelope") if isinstance(metadata, Mapping) else None
        request = request_by_id.get(attempt_id)
        retry = retry_by_id.get(attempt_id)
        code = envelope.get("error_code") if isinstance(envelope, Mapping) else None
        envelope_is_safe = bool(
            isinstance(request, Mapping)
            and isinstance(envelope, Mapping)
            and rejected.get("http_status") == 200
            and rejected.get("generation_id") in {None, ""}
            and envelope.get("classification") == "openrouter_error_envelope"
            and envelope.get("accepted_chat_completion") is False
            and envelope.get("retryable") is True
            and code in allowed_codes
            and str(metadata.get("openrouter_cache_status") or "").upper() != "HIT"
            and str(metadata.get("cloudflare_cache_status") or "").upper()
            in {"MISS", "BYPASS"}
            and attempt_id not in accounting_by_id
        )
        if not envelope_is_safe:
            unsafe.append(f"unsafe_rejection:{attempt_id}")
            continue
        attempt_index = int(request.get("attempt_index") or 0)
        is_terminal = attempt_index + 1 >= maximum_attempts
        if is_terminal:
            if retry is not None:
                unsafe.append(f"terminal_rejection_was_retried:{attempt_id}")
                continue
            successors = [
                item
                for candidate_id, item in request_by_id.items()
                if candidate_id != attempt_id
                and item.get("arm_id") == request.get("arm_id")
                and item.get("phase") == request.get("phase")
                and item.get("request_key_sha256")
                == request.get("request_key_sha256")
                and int(item.get("attempt_index") or 0) > attempt_index
            ]
            if successors:
                unsafe.append(f"terminal_rejection_has_successor:{attempt_id}")
                continue
            safe_rejections += 1
            terminal_rejections += 1
            continue
        if not (
            isinstance(retry, Mapping)
            and str((retry.get("metadata") or {}).get("retry_reason") or "")
            == f"retryable_openrouter_error_envelope_{code}"
        ):
            unsafe.append(f"retryable_rejection_has_no_retry_event:{attempt_id}")
            continue
        successors = [
            item
            for candidate_id, item in request_by_id.items()
            if candidate_id != attempt_id
            and item.get("arm_id") == request.get("arm_id")
            and item.get("phase") == request.get("phase")
            and item.get("request_key_sha256") == request.get("request_key_sha256")
            and item.get("attempt_index") == int(request.get("attempt_index") or 0) + 1
        ]
        if len(successors) != 1:
            unsafe.append(f"rejection_has_no_unique_fresh_successor:{attempt_id}")
            continue
        expected_retry_ids.add(attempt_id)
        safe_rejections += 1
        retried_rejections += 1
    if set(retry_by_id) != expected_retry_ids:
        unsafe.append("retry_event_without_allowlisted_error_rejection")
    if unsafe:
        issues[predicate_id].extend(value for value in unsafe if value not in issues[predicate_id])
    else:
        generation_issues = issues["generation_accounting"]
        if "request_response_bijection_failed" in generation_issues:
            generation_issues.remove("request_response_bijection_failed")
    record.update(
        {
            "provider_rejections": len(rejections),
            "safe_retryable_error_envelope_rejections": safe_rejections,
            "retried_error_envelope_rejections": retried_rejections,
            "terminal_error_envelope_rejections": terminal_rejections,
            "unsafe_provider_rejections": len(unsafe),
            "accounted_provider_responses": len(accounting),
            "usable_pair": all(not values for values in issues.values()),
        }
    )
    return record


def build_v3_route_validation_audit(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a fail-closed v3 PASS/FAIL receipt from immutable run records."""

    bindings = _v3_runner_contract(plan=plan, runner_assets=runner_assets)
    try:
        records = [
            _retry_envelope_audit(
                record=_audit_v2_route_variant(plan=plan, binding=binding),
                plan=plan,
            )
            for binding in bindings
        ]
    except SensitivityProtocolError as error:
        raise RouteRecoveryError("v3 immutable execution records do not verify") from error
    attempted = [record for record in records if record["attempted"]]
    all_attempted = len(attempted) == 3

    work_ids = sorted(str(record["work_item_id"]) for record in records)
    attempt_ids = [value for record in records for value in record["attempt_ids"]]
    generation_ids = [value for record in records for value in record["generation_ids"]]
    closed = plan["closed_identifiers"]
    closed_work_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["work_item_ids"]
    }
    closed_attempt_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["attempt_ids"]
    }
    closed_generation_ids = {
        str(value)
        for revision in ("v1", "v2")
        for value in closed[revision]["generation_ids"]
    }
    work_overlap = sorted(set(work_ids) & closed_work_ids)
    attempt_overlap = sorted(set(attempt_ids) & closed_attempt_ids)
    generation_overlap = sorted(set(generation_ids) & closed_generation_ids)
    duplicate_attempts = sorted(
        value for value, count in Counter(attempt_ids).items() if value and count > 1
    )
    duplicate_generations = sorted(
        value for value, count in Counter(generation_ids).items() if value and count > 1
    )
    missing_generations = sum(
        int(record.get("missing_generation_id_requests") or 0) for record in records
    )
    identifiers_fresh = not (
        work_overlap
        or attempt_overlap
        or generation_overlap
        or duplicate_attempts
        or duplicate_generations
        or missing_generations
    )

    actual_cost = sum((record["actual_cost_usd"] for record in records), Decimal(0))
    retained = sum(
        (record["retained_exposure_usd"] for record in records), Decimal(0)
    )
    baseline = _decimal(
        plan["budget"]["post_v2_conservative_exposure_usd"], field="v3 baseline"
    )
    post_route = baseline + retained
    ceiling = _decimal(plan["budget"]["admission_ceiling_usd"], field="ceiling")
    budget_passed = post_route <= ceiling
    accepted_responses = sum(record["provider_responses"] for record in records)
    provider_requests = sum(record["provider_requests"] for record in records)
    safe_rejections = sum(
        record["safe_retryable_error_envelope_rejections"] for record in records
    )
    retried_rejections = sum(
        record["retried_error_envelope_rejections"] for record in records
    )
    terminal_rejections = sum(
        record["terminal_error_envelope_rejections"] for record in records
    )
    unsafe_rejections = sum(record["unsafe_provider_rejections"] for record in records)
    accounted_responses = sum(record["accounted_provider_responses"] for record in records)
    attempted_accounting_complete = bool(
        attempted
        and accepted_responses == accounted_responses
        and provider_requests == accepted_responses + safe_rejections
        and unsafe_rejections == 0
        and missing_generations == 0
        and all(record["unreconciled_provider_requests"] == 0 for record in records)
    )
    accounting_complete = all_attempted and attempted_accounting_complete

    predicate_results: list[dict[str, Any]] = []
    common_evidence = {plan["artifact_sha256"], runner_assets["artifact_sha256"]}
    for predicate in plan["acceptance_gate"]["predicates"]:
        predicate_id = str(predicate["predicate_id"])
        failures = [
            f"{record['variant_id']}:{reason}"
            for record in attempted
            for reason in record["issues"][predicate_id]
        ]
        if predicate_id == "identifier_freshness" and not identifiers_fresh:
            failures.append("identifier_overlap_duplication_or_missing_generation")
        if predicate_id == "budget_admission" and not budget_passed:
            failures.append("post_route_exposure_exceeds_admission_ceiling")
        if (
            predicate_id == "generation_accounting"
            and attempted
            and not attempted_accounting_complete
        ):
            failures.append("accepted_generation_accounting_is_incomplete")
        if predicate_id == "retryable_error_envelope_safety" and unsafe_rejections:
            failures.append("unsafe_error_envelope_retry_present")
        status = (
            "not_evaluated"
            if not all_attempted
            else "failed"
            if failures
            else "passed"
        )
        evidence = set(common_evidence)
        for record in records:
            evidence.update(record["evidence_sha256"])
        predicate_results.append(
            {
                "predicate_id": predicate_id,
                "status": status,
                "passed": status == "passed",
                "failures": sorted(set(failures)),
                "evidence_sha256": sorted(evidence),
            }
        )

    any_failed = any(
        any(record["issues"].values()) for record in attempted
    ) or (all_attempted and (not identifiers_fresh or not budget_passed or not accounting_complete))
    passed = bool(
        all_attempted
        and not any_failed
        and all(result["passed"] for result in predicate_results)
    )
    decision = (
        "passed_all_predicates"
        if passed
        else "failed_one_or_more_predicates"
        if all_attempted or any_failed
        else "awaiting_remaining_route_variants"
        if attempted
        else "not_executed"
    )
    public_records = [
        {
            key: _decimal_text(value) if isinstance(value, Decimal) else value
            for key, value in record.items()
            if key != "issues"
        }
        | {"predicate_failures": record["issues"]}
        for record in records
    ]
    source_artifacts = [
        {
            "work_item_id": record["work_item_id"],
            "variant_id": record["variant_id"],
            "source_artifact_sha256": record["source_artifact_sha256"],
            "summary_artifact_sha256": record["summary_artifact_sha256"],
            "response_artifact_sha256s": record["response_artifact_sha256s"],
            "immutable": True,
        }
        for record in records
        if record["source_artifact_sha256"] and record["summary_artifact_sha256"]
    ]
    counts = {
        "attempted_pairs": len(attempted),
        "usable_pairs": sum(bool(record["usable_pair"]) for record in records),
        "intended_arms": 6,
        "usable_arms": 2 * sum(bool(record["usable_pair"]) for record in records),
        "provider_requests": provider_requests,
        "provider_responses": accepted_responses,
        "retryable_error_envelope_rejections": safe_rejections,
        "retried_error_envelope_rejections": retried_rejections,
        "terminal_error_envelope_rejections": terminal_rejections,
        "unsafe_provider_rejections": unsafe_rejections,
        "successful_epicure_tool_calls": sum(
            record["successful_epicure_tool_calls"] for record in records
        ),
        "epicure_off_tool_calls": sum(record["epicure_off_tool_calls"] for record in records),
        "synthetic_arms": 0,
        "identity_mismatches": sum(record["identity_mismatches"] for record in records),
        "unreconciled_generations": sum(
            record["unreconciled_provider_requests"] for record in records
        ),
        "non_chat_generation_responses": sum(
            record["non_chat_completion_envelopes"] for record in records
        ),
        "truncated_or_invalid_arms": sum(
            record["truncated_or_invalid_arms"] for record in records
        ),
    }
    return {
        "schema_version": V3_AUDIT_SCHEMA_VERSION,
        "record_role": "source_derived_fail_closed_v3_route_gate_receipt",
        "derivation_policy": (
            "No predicate value is operator supplied. All fields derive from exact runner "
            "assets and immutable summary, source, journal, response, and ledger records."
        ),
        "v3_route_plan_sha256": plan["artifact_sha256"],
        "runner_assets_sha256": runner_assets["artifact_sha256"],
        "route_cell_id": plan["route_validation"]["route_cell_id"],
        "decision": decision,
        "source_artifacts": source_artifacts,
        "variant_audits": public_records,
        "counts": counts,
        "response_envelope_audit": {
            "contract_sha256": plan["safe_response_envelope_contract"][
                "contract_sha256"
            ],
            "provider_source_sha256": plan["source"]["provider_source_sha256"],
            "all_generation_responses_chat_completions": bool(
                attempted and counts["non_chat_generation_responses"] == 0
            ),
            "retryable_rejections_excluded_from_generation_accounting": bool(
                attempted and attempted_accounting_complete and unsafe_rejections == 0
            ),
            "safe_retryable_error_envelope_rejections": safe_rejections,
            "retried_error_envelope_rejections": retried_rejections,
            "terminal_error_envelope_rejections": terminal_rejections,
            "unsafe_provider_rejections": unsafe_rejections,
        },
        "identity_audit": {
            "model_id": plan["route_validation"]["model_id"],
            "provider_endpoint": plan["route_validation"]["work_items"][0][
                "provider_endpoint"
            ],
            "actual_provider_name": plan["route_validation"]["work_items"][0][
                "actual_provider_name"
            ],
            "runtime_id": plan["epicure"]["runtime_id"],
            "bundle_sha256": plan["epicure"]["bundle_sha256"],
            "application_sha256": plan["epicure"]["application_sha256"],
            "tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
            "lineage_inventory_sha256": plan["epicure"]["lineage_inventory_sha256"],
        },
        "identifier_freshness_audit": {
            "closed_v1_identifiers_sha256": closed["v1"]["inventory_sha256"],
            "closed_v2_identifiers_sha256": closed["v2"]["inventory_sha256"],
            "closed_work_item_ids": sorted(closed_work_ids),
            "closed_attempt_ids": sorted(closed_attempt_ids),
            "closed_generation_ids": sorted(closed_generation_ids),
            "v3_work_item_ids": work_ids,
            "v3_attempt_ids": attempt_ids,
            "v3_generation_ids": generation_ids,
            "work_item_overlap": work_overlap,
            "attempt_id_overlap": attempt_overlap,
            "generation_id_overlap": generation_overlap,
            "duplicate_attempt_ids": duplicate_attempts,
            "duplicate_generation_ids": duplicate_generations,
            "missing_generation_id_responses": missing_generations,
            "all_identifiers_fresh": all_attempted and identifiers_fresh,
        },
        "accounting_audit": {
            "identified_generation_cost_usd": _decimal_text(actual_cost),
            "conservative_retained_exposure_usd": _decimal_text(retained),
            "post_route_conservative_exposure_usd": _decimal_text(post_route),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "accepted_generation_accounting_complete": attempted_accounting_complete,
            "full_route_accounting_complete": accounting_complete,
            "rejected_error_envelope_cost_lookups": 0,
        },
        "predicate_results": predicate_results,
        "full_sensitivity_admission": {
            "authorized": passed,
            "scope": "materialize_prespecified_36_pair_72_arm_study",
            "route_validation_outputs_reused": False,
        },
        "quality_fit_eligible": False,
        "official": False,
        "rank_eligible": False,
    }


def verify_v3_route_validation_pass_audit(
    document: object,
    plan: object,
) -> bool:
    """Return true only for a complete source-derived v3 PASS receipt."""

    if not verify_v3_route_plan(plan) or not _artifact_verifies(
        document, V3_AUDIT_SCHEMA_VERSION
    ):
        return False
    assert isinstance(plan, Mapping)
    assert isinstance(document, Mapping)
    counts = document.get("counts")
    sources = document.get("source_artifacts")
    envelope = document.get("response_envelope_audit")
    identity = document.get("identity_audit")
    identifiers = document.get("identifier_freshness_audit")
    accounting = document.get("accounting_audit")
    predicates = document.get("predicate_results")
    admission = document.get("full_sensitivity_admission")
    if not all(
        isinstance(value, Mapping)
        for value in (counts, envelope, identity, identifiers, accounting, admission)
    ) or not isinstance(sources, list) or not isinstance(predicates, list):
        return False
    expected_items = {
        str(item["work_item_id"]): str(item["variant_id"])
        for item in plan["route_validation"]["work_items"]
    }
    observed_items: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            return False
        work_item_id = str(source.get("work_item_id") or "")
        response_hashes = source.get("response_artifact_sha256s")
        if (
            source.get("immutable") is not True
            or work_item_id in observed_items
            or any(
                not isinstance(source.get(field), str)
                or len(str(source.get(field))) != 64
                for field in ("source_artifact_sha256", "summary_artifact_sha256")
            )
            or not isinstance(response_hashes, list)
            or len(response_hashes) != 2
            or len(set(response_hashes)) != 2
            or any(not isinstance(value, str) or len(value) != 64 for value in response_hashes)
        ):
            return False
        observed_items[work_item_id] = str(source.get("variant_id") or "")
    expected_predicates = {
        str(item["predicate_id"]) for item in plan["acceptance_gate"]["predicates"]
    }
    observed_predicates: set[str] = set()
    for predicate in predicates:
        if not isinstance(predicate, Mapping):
            return False
        predicate_id = str(predicate.get("predicate_id") or "")
        evidence = predicate.get("evidence_sha256")
        if (
            predicate_id in observed_predicates
            or predicate_id not in expected_predicates
            or predicate.get("status") != "passed"
            or predicate.get("passed") is not True
            or predicate.get("failures") != []
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(value, str) or len(value) != 64 for value in evidence)
        ):
            return False
        observed_predicates.add(predicate_id)
    try:
        actual = _decimal(accounting.get("identified_generation_cost_usd"), field="actual")
        retained = _decimal(
            accounting.get("conservative_retained_exposure_usd"), field="retained"
        )
        post_route = _decimal(
            accounting.get("post_route_conservative_exposure_usd"), field="post-route"
        )
        ceiling = _decimal(accounting.get("admission_ceiling_usd"), field="ceiling")
        baseline = _decimal(
            plan["budget"]["post_v2_conservative_exposure_usd"], field="baseline"
        )
    except RouteRecoveryError:
        return False
    closed = plan["closed_identifiers"]
    expected_closed_work = sorted(
        {
            str(value)
            for revision in ("v1", "v2")
            for value in closed[revision]["work_item_ids"]
        }
    )
    expected_closed_attempts = sorted(
        {
            str(value)
            for revision in ("v1", "v2")
            for value in closed[revision]["attempt_ids"]
        }
    )
    expected_closed_generations = sorted(
        {
            str(value)
            for revision in ("v1", "v2")
            for value in closed[revision]["generation_ids"]
        }
    )
    expected_identity = {
        "model_id": plan["route_validation"]["model_id"],
        "provider_endpoint": plan["route_validation"]["work_items"][0][
            "provider_endpoint"
        ],
        "actual_provider_name": plan["route_validation"]["work_items"][0][
            "actual_provider_name"
        ],
        "runtime_id": plan["epicure"]["runtime_id"],
        "bundle_sha256": plan["epicure"]["bundle_sha256"],
        "application_sha256": plan["epicure"]["application_sha256"],
        "tool_schema_sha256": plan["epicure"]["tool_schema_sha256"],
        "lineage_inventory_sha256": plan["epicure"]["lineage_inventory_sha256"],
    }
    v3_attempts = identifiers.get("v3_attempt_ids")
    v3_generations = identifiers.get("v3_generation_ids")
    safe_rejections = counts.get("retryable_error_envelope_rejections")
    retried_rejections = counts.get("retried_error_envelope_rejections")
    terminal_rejections = counts.get("terminal_error_envelope_rejections")
    return bool(
        document.get("v3_route_plan_sha256") == plan["artifact_sha256"]
        and document.get("decision") == "passed_all_predicates"
        and document.get("route_cell_id") == plan["route_validation"]["route_cell_id"]
        and observed_items == expected_items
        and observed_predicates == expected_predicates
        and counts.get("attempted_pairs") == 3
        and counts.get("usable_pairs") == 3
        and counts.get("intended_arms") == 6
        and counts.get("usable_arms") == 6
        and isinstance(counts.get("provider_requests"), int)
        and isinstance(counts.get("provider_responses"), int)
        and isinstance(safe_rejections, int)
        and isinstance(retried_rejections, int)
        and isinstance(terminal_rejections, int)
        and safe_rejections == retried_rejections + terminal_rejections
        and counts["provider_requests"] == counts["provider_responses"] + safe_rejections
        and counts.get("unsafe_provider_rejections") == 0
        and counts.get("successful_epicure_tool_calls", 0) >= 3
        and counts.get("epicure_off_tool_calls") == 0
        and counts.get("synthetic_arms") == 0
        and counts.get("identity_mismatches") == 0
        and counts.get("unreconciled_generations") == 0
        and counts.get("non_chat_generation_responses") == 0
        and counts.get("truncated_or_invalid_arms") == 0
        and envelope.get("contract_sha256")
        == plan["safe_response_envelope_contract"]["contract_sha256"]
        and envelope.get("provider_source_sha256")
        == plan["source"]["provider_source_sha256"]
        and envelope.get("all_generation_responses_chat_completions") is True
        and envelope.get("retryable_rejections_excluded_from_generation_accounting")
        is True
        and envelope.get("safe_retryable_error_envelope_rejections") == safe_rejections
        and envelope.get("retried_error_envelope_rejections") == retried_rejections
        and envelope.get("terminal_error_envelope_rejections") == terminal_rejections
        and envelope.get("unsafe_provider_rejections") == 0
        and dict(identity) == expected_identity
        and identifiers.get("closed_v1_identifiers_sha256")
        == closed["v1"]["inventory_sha256"]
        and identifiers.get("closed_v2_identifiers_sha256")
        == closed["v2"]["inventory_sha256"]
        and identifiers.get("closed_work_item_ids") == expected_closed_work
        and identifiers.get("closed_attempt_ids") == expected_closed_attempts
        and identifiers.get("closed_generation_ids") == expected_closed_generations
        and identifiers.get("v3_work_item_ids") == sorted(expected_items)
        and isinstance(v3_attempts, list)
        and len(v3_attempts) == counts["provider_requests"]
        and len(set(v3_attempts)) == len(v3_attempts)
        and isinstance(v3_generations, list)
        and len(v3_generations) == counts["provider_responses"]
        and len(set(v3_generations)) == len(v3_generations)
        and identifiers.get("work_item_overlap") == []
        and identifiers.get("attempt_id_overlap") == []
        and identifiers.get("generation_id_overlap") == []
        and identifiers.get("duplicate_attempt_ids") == []
        and identifiers.get("duplicate_generation_ids") == []
        and identifiers.get("missing_generation_id_responses") == 0
        and identifiers.get("all_identifiers_fresh") is True
        and accounting.get("accepted_generation_accounting_complete") is True
        and accounting.get("full_route_accounting_complete") is True
        and accounting.get("rejected_error_envelope_cost_lookups") == 0
        and actual <= retained
        and post_route == baseline + retained
        and post_route <= ceiling
        and ceiling
        == _decimal(plan["budget"]["admission_ceiling_usd"], field="plan ceiling")
        and admission.get("authorized") is True
        and admission.get("scope") == "materialize_prespecified_36_pair_72_arm_study"
        and admission.get("route_validation_outputs_reused") is False
        and document.get("quality_fit_eligible") is False
        and document.get("official") is False
        and document.get("rank_eligible") is False
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    close = subparsers.add_parser("close-v2")
    close.add_argument("--v2-plan", type=Path, required=True)
    close.add_argument("--v2-audit", type=Path, required=True)
    close.add_argument("--v2-runner-assets", type=Path, required=True)
    close.add_argument("--output-dir", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-v3")
    freeze.add_argument("--v2-plan", type=Path, required=True)
    freeze.add_argument("--v2-audit", type=Path, required=True)
    freeze.add_argument("--v2-closed-identifiers", type=Path, required=True)
    freeze.add_argument("--provider-source", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    materialize = subparsers.add_parser("materialize-v3")
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--v2-runner-assets", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit-v3")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--runner-assets", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    close_v3 = subparsers.add_parser("close-v3")
    close_v3.add_argument("--plan", type=Path, required=True)
    close_v3.add_argument("--runner-assets", type=Path, required=True)
    close_v3.add_argument("--prior-audit", type=Path, required=True)
    close_v3.add_argument("--corrected-audit", type=Path, required=True)
    close_v3.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "close-v2":
        payload = build_v2_closed_identifiers(
            v2_plan=_regular_json(arguments.v2_plan),
            v2_audit=_regular_json(arguments.v2_audit),
            v2_runner_assets=_regular_json(arguments.v2_runner_assets),
        )
        path = _write_artifact(
            arguments.output_dir, "reasoning-effort-v2-closed-identifiers", payload
        )
    elif arguments.command == "freeze-v3":
        payload = build_v3_route_plan(
            v2_plan=_regular_json(arguments.v2_plan),
            v2_audit=_regular_json(arguments.v2_audit),
            v2_closed_identifiers=_regular_json(arguments.v2_closed_identifiers),
            provider_source_path=arguments.provider_source,
        )
        path = _write_artifact(
            arguments.output_dir, "reasoning-effort-v3-route-validation-plan", payload
        )
    elif arguments.command == "materialize-v3":
        plan = _regular_json(arguments.plan)
        payload = materialize_v3_route_assets(
            plan=plan,
            plan_path=arguments.plan,
            v2_runner_assets=_regular_json(arguments.v2_runner_assets),
            output_dir=arguments.output_dir,
        )
        path = _write_artifact(
            arguments.output_dir, "reasoning-effort-v3-route-runner-assets", payload
        )
    elif arguments.command == "audit-v3":
        plan = _regular_json(arguments.plan)
        payload = build_v3_route_validation_audit(
            plan=plan,
            runner_assets=_regular_json(arguments.runner_assets),
        )
        path = _write_artifact(
            arguments.output_dir, "reasoning-effort-v3-route-validation-audit", payload
        )
    else:
        payload = build_v3_route_closure(
            plan=_regular_json(arguments.plan),
            runner_assets=_regular_json(arguments.runner_assets),
            prior_audit=_regular_json(arguments.prior_audit),
            corrected_audit=_regular_json(arguments.corrected_audit),
        )
        path = _write_artifact(
            arguments.output_dir, "reasoning-effort-v3-route-closure", payload
        )
    written = _regular_json(path)
    strict_pass = (
        verify_v3_route_validation_pass_audit(written, plan)
        if arguments.command == "audit-v3"
        else False
    )
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": written["artifact_sha256"],
                "provider_calls_made": False,
                "epicure_calls_made": False,
                **(
                    {
                        "decision": written["decision"],
                        "strict_pass_verifies": strict_pass,
                        "full_sensitivity_authorized": written[
                            "full_sensitivity_admission"
                        ]["authorized"],
                    }
                    if arguments.command == "audit-v3"
                    else (
                        {
                            "closure_verifies": verify_v3_route_closure(written),
                            "v4_authorized": written["decision"]["v4_authorized"],
                        }
                        if arguments.command == "close-v3"
                        else {}
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
