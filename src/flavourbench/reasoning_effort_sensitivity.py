"""Freeze and preflight the real-call reasoning-effort sensitivity study.

The protocol compares a contemporaneous explicit-low rerun with an omitted
provider-default control and explicit high effort.  All three variants use the
same exact endpoints, one non-suspect real-human anchor per task family, and
matched Epicure-off/on arms.  Planning and preflight make no provider calls.
Execution remains fail-closed until the content-addressed Epicure release,
route identity, and shared budget gates pass.
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

from .epicure_lineage_correction import verify_correction
from .epicure_lineage_inventory import verify_inventory
from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import IntegrityError, _verify_live_artifact
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256
from .real_dataset_runner import (
    _verify_response_artifact,
    load_dataset_ledger,
    load_development_task_inventory,
    task_registry_sha256,
)
from .run_journal import (
    JournalIntegrityError,
    load_run_journal,
    verify_journal_descriptor,
)
from .tool_contract import required_tool_contract

PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-sensitivity-plan-v1"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-reasoning-effort-sensitivity-preflight-v1"
V2_ROUTE_PLAN_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-validation-plan-v1"
V2_ROUTE_AUDIT_SCHEMA_VERSION = "flavourbench-reasoning-effort-v2-route-validation-audit-v1"
V1_CLOSED_IDENTIFIERS_SCHEMA_VERSION = (
    "flavourbench-reasoning-effort-v1-closed-identifiers-v1"
)
V2_ROUTE_PREDICATES = (
    (
        "complete_arms",
        "all six arms finish normally with non-empty final answers",
    ),
    (
        "chat_completion_envelopes",
        "all provider responses classify as chat_completions",
    ),
    (
        "generation_accounting",
        "every provider response has a generation ID and exact accounting metadata",
    ),
    (
        "model_provider_identity",
        "canonical model and exact provider identities match the frozen endpoint",
    ),
    (
        "epicure_on_treatment",
        "each Epicure-on arm completes at least one successful real MCP tool call",
    ),
    (
        "epicure_off_control",
        "all Epicure-off arms contain zero MCP tool calls",
    ),
    (
        "epicure_runtime_identity",
        "release, bundle, application, and semantic tool hashes match exactly",
    ),
    (
        "cost_reconciliation",
        "all identifiable generation costs reconcile and no unknown HTTP-200 remains",
    ),
    (
        "response_integrity",
        "no arm is truncated, invalid, substituted, cached, or retried after ambiguity",
    ),
    (
        "budget_admission",
        "post-route conservative exposure remains at or below the admission ceiling",
    ),
    (
        "identifier_freshness",
        "work-item, attempt, and generation identifiers are unique and do not overlap v1",
    ),
)
TASK_FAMILIES = ("substitution", "composition", "cookability", "evidence")
DEFAULT_MODEL_IDS = (
    "openai/gpt-5.6-sol-pro",
    "anthropic/claude-sonnet-5",
    "deepseek/deepseek-v4-flash-0731",
)
VARIANTS = (
    {
        "variant_id": "explicit_low",
        "intermediate_reasoning_effort": "low",
        "final_reasoning_effort": "low",
        "request_semantics": "explicit_reasoning_effort",
    },
    {
        "variant_id": "provider_default",
        "intermediate_reasoning_effort": None,
        "final_reasoning_effort": None,
        "request_semantics": "omit_reasoning_effort_parameter",
    },
    {
        "variant_id": "explicit_high",
        "intermediate_reasoning_effort": "high",
        "final_reasoning_effort": "high",
        "request_semantics": "explicit_reasoning_effort",
    },
)


class SensitivityProtocolError(RuntimeError):
    """A sensitivity input or preflight gate failed verification."""


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
    if path.is_symlink() or not path.is_file():
        raise SensitivityProtocolError(f"source input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SensitivityProtocolError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise SensitivityProtocolError(f"{field} must be finite and non-negative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _sha256_text(value: object, *, field: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise SensitivityProtocolError(f"{field} is not a lowercase SHA-256 digest")
    return rendered


def _regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SensitivityProtocolError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SensitivityProtocolError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise SensitivityProtocolError(f"expected an object: {path}")
    return value


def _verified_artifact(path: Path) -> tuple[dict[str, Any], str]:
    value = _regular_json(path)
    artifact_digest = value.get("artifact_sha256")
    if isinstance(artifact_digest, str):
        unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if len(artifact_digest) != 64 or _sha256(unhashed) != artifact_digest:
            raise SensitivityProtocolError(f"artifact content address does not verify: {path}")
        return value, artifact_digest
    content_address = value.get("content_address")
    if isinstance(content_address, Mapping):
        digest = str(content_address.get("digest") or "")
        unhashed = {key: item for key, item in value.items() if key != "content_address"}
        if (
            content_address.get("algorithm") != "sha256"
            or content_address.get("uri") != f"sha256:{digest}"
            or len(digest) != 64
            or _sha256(unhashed) != digest
        ):
            raise SensitivityProtocolError(f"manifest content address does not verify: {path}")
        return value, digest
    raise SensitivityProtocolError(f"input has no supported content address: {path}")


def _artifact_sha256_verifies(document: object, schema_version: str) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != schema_version:
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return isinstance(digest, str) and len(digest) == 64 and _sha256(unhashed) == digest


def verify_v2_route_validation_plan(document: object) -> bool:
    """Verify the content address and closed diagnostic boundary of a v2 route plan."""

    if not _artifact_sha256_verifies(document, V2_ROUTE_PLAN_SCHEMA_VERSION):
        return False
    assert isinstance(document, Mapping)
    route = document.get("route_validation")
    preflight = document.get("preflight")
    full = document.get("full_sensitivity")
    claims = document.get("claim_boundary")
    acceptance = document.get("acceptance_gate")
    if not all(
        isinstance(value, Mapping)
        for value in (route, preflight, full, claims, acceptance)
    ):
        return False
    assert isinstance(route, Mapping)
    assert isinstance(preflight, Mapping)
    assert isinstance(full, Mapping)
    assert isinstance(claims, Mapping)
    assert isinstance(acceptance, Mapping)
    work_items = route.get("work_items")
    predicates = acceptance.get("predicates")
    expected_predicates = [
        {"predicate_id": predicate_id, "requirement": requirement}
        for predicate_id, requirement in V2_ROUTE_PREDICATES
    ]
    return (
        isinstance(work_items, list)
        and len(work_items) == 3
        and len(
            {
                str(item.get("work_item_id") or "")
                for item in work_items
                if isinstance(item, Mapping)
            }
        )
        == 3
        and route.get("matched_pairs") == 3
        and route.get("response_arms") == 6
        and route.get("synthetic_arms") == 0
        and route.get("diagnostic_outputs_enter_quality_fit") is False
        and predicates == expected_predicates
        and preflight.get("decision")
        in {"ready_to_materialize_v2_route_validation_only", "blocked_before_provider_call"}
        and preflight.get("provider_calls_made") is False
        and preflight.get("epicure_calls_made") is False
        and full.get("status") == "blocked_pending_v2_route_validation"
        and full.get("response_arms") == 72
        and claims.get("quality_results_available") is False
        and claims.get("sensitivity_effect_estimable") is False
        and claims.get("route_validation_only") is True
        and claims.get("official") is False
        and claims.get("rank_eligible") is False
    )


def verify_v2_route_validation_pass_audit(
    document: object,
    plan: object,
) -> bool:
    """Return true only for a content-addressed PASS receipt bound to ``plan``."""

    if not verify_v2_route_validation_plan(plan) or not _artifact_sha256_verifies(
        document, V2_ROUTE_AUDIT_SCHEMA_VERSION
    ):
        return False
    assert isinstance(plan, Mapping)
    assert isinstance(document, Mapping)
    route = plan["route_validation"]
    acceptance = plan["acceptance_gate"]
    envelope = plan["safe_response_envelope_contract"]
    epicure = plan["epicure"]
    source = plan["source"]
    counts = document.get("counts")
    sources = document.get("source_artifacts")
    classifier = document.get("response_envelope_audit")
    identity = document.get("identity_audit")
    accounting = document.get("accounting_audit")
    identifiers = document.get("identifier_freshness_audit")
    predicates = document.get("predicate_results")
    admission = document.get("full_sensitivity_admission")
    if not all(
        isinstance(value, Mapping)
        for value in (counts, classifier, identity, accounting, identifiers, admission)
    ) or not isinstance(sources, list) or not isinstance(predicates, list):
        return False
    assert isinstance(counts, Mapping)
    assert isinstance(classifier, Mapping)
    assert isinstance(identity, Mapping)
    assert isinstance(accounting, Mapping)
    assert isinstance(identifiers, Mapping)
    assert isinstance(admission, Mapping)
    expected_items = {
        str(item["work_item_id"]): str(item["variant_id"])
        for item in route["work_items"]
    }
    observed_items: dict[str, str] = {}
    for item in sources:
        if not isinstance(item, Mapping):
            return False
        work_item_id = str(item.get("work_item_id") or "")
        variant_id = str(item.get("variant_id") or "")
        source_digest = str(item.get("source_artifact_sha256") or "")
        summary_digest = str(item.get("summary_artifact_sha256") or "")
        if (
            item.get("immutable") is not True
            or len(source_digest) != 64
            or len(summary_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_digest + summary_digest
            )
            or work_item_id in observed_items
        ):
            return False
        observed_items[work_item_id] = variant_id
    expected_predicate_ids = {
        str(item["predicate_id"]) for item in acceptance["predicates"]
    }
    observed_predicate_ids: set[str] = set()
    for result in predicates:
        if not isinstance(result, Mapping):
            return False
        predicate_id = str(result.get("predicate_id") or "")
        evidence = result.get("evidence_sha256")
        if (
            predicate_id in observed_predicate_ids
            or predicate_id not in expected_predicate_ids
            or result.get("passed") is not True
            or not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in evidence
            )
        ):
            return False
        observed_predicate_ids.add(predicate_id)
    try:
        actual_cost = _decimal(
            accounting.get("identified_generation_cost_usd"), field="v2 actual cost"
        )
        retained = _decimal(
            accounting.get("conservative_retained_exposure_usd"),
            field="v2 retained exposure",
        )
        post_route = _decimal(
            accounting.get("post_route_conservative_exposure_usd"),
            field="v2 post-route exposure",
        )
        ceiling = _decimal(
            accounting.get("admission_ceiling_usd"), field="v2 admission ceiling"
        )
    except SensitivityProtocolError:
        return False
    expected_identity = {
        "model_id": route.get("model_id"),
        "provider_endpoint": route["work_items"][0].get("provider_endpoint"),
        "actual_provider_name": route["work_items"][0].get("actual_provider_name"),
        "runtime_id": epicure.get("runtime_id"),
        "bundle_sha256": epicure.get("bundle_sha256"),
        "application_sha256": epicure.get("application_sha256"),
        "tool_schema_sha256": epicure.get("tool_schema_sha256"),
        "lineage_inventory_sha256": epicure.get("lineage_inventory_sha256"),
    }
    return (
        document.get("v2_route_plan_sha256") == plan.get("artifact_sha256")
        and document.get("decision") == "passed_all_predicates"
        and document.get("route_cell_id") == route.get("route_cell_id")
        and observed_items == expected_items
        and counts.get("attempted_pairs") == 3
        and counts.get("usable_pairs") == 3
        and counts.get("intended_arms") == 6
        and counts.get("usable_arms") == 6
        and isinstance(counts.get("provider_requests"), int)
        and counts.get("provider_requests", 0) >= 6
        and counts.get("provider_responses") == counts.get("provider_requests")
        and counts.get("successful_epicure_tool_calls", 0) >= 3
        and counts.get("epicure_off_tool_calls") == 0
        and counts.get("synthetic_arms") == 0
        and counts.get("identity_mismatches") == 0
        and counts.get("unreconciled_provider_requests") == 0
        and counts.get("non_chat_completion_envelopes") == 0
        and counts.get("truncated_or_invalid_arms") == 0
        and classifier.get("contract_sha256") == envelope.get("contract_sha256")
        and classifier.get("provider_source_sha256")
        == source.get("provider_source_sha256")
        and classifier.get("all_provider_responses_chat_completions") is True
        and classifier.get("unknown_http_200_responses") == 0
        and dict(identity) == expected_identity
        and identifiers.get("v1_closed_identifiers_sha256")
        == plan["closed_v1_identifiers"]["inventory_sha256"]
        and identifiers.get("v1_closed_work_item_ids")
        == plan["closed_v1_identifiers"]["work_item_ids"]
        and identifiers.get("v1_closed_attempt_ids")
        == plan["closed_v1_identifiers"]["attempt_ids"]
        and identifiers.get("v1_closed_generation_ids")
        == plan["closed_v1_identifiers"]["generation_ids"]
        and identifiers.get("v2_work_item_ids") == sorted(expected_items)
        and isinstance(identifiers.get("v2_attempt_ids"), list)
        and len(identifiers.get("v2_attempt_ids")) == counts.get("provider_requests")
        and len(set(identifiers.get("v2_attempt_ids")))
        == len(identifiers.get("v2_attempt_ids"))
        and isinstance(identifiers.get("v2_generation_ids"), list)
        and len(identifiers.get("v2_generation_ids")) == counts.get("provider_responses")
        and len(set(identifiers.get("v2_generation_ids")))
        == len(identifiers.get("v2_generation_ids"))
        and not set(identifiers.get("v2_work_item_ids"))
        & set(identifiers.get("v1_closed_work_item_ids"))
        and not set(identifiers.get("v2_attempt_ids"))
        & set(identifiers.get("v1_closed_attempt_ids"))
        and not set(identifiers.get("v2_generation_ids"))
        & set(identifiers.get("v1_closed_generation_ids"))
        and identifiers.get("work_item_overlap") == []
        and identifiers.get("attempt_id_overlap") == []
        and identifiers.get("generation_id_overlap") == []
        and identifiers.get("duplicate_attempt_ids") == []
        and identifiers.get("duplicate_generation_ids") == []
        and identifiers.get("missing_generation_id_requests") == 0
        and identifiers.get("all_identifiers_fresh") is True
        and accounting.get("generation_accounting_complete") is True
        and actual_cost <= retained
        and post_route <= ceiling
        and ceiling == _decimal(plan["budget"]["admission_ceiling_usd"], field="ceiling")
        and observed_predicate_ids == expected_predicate_ids
        and admission.get("authorized") is True
        and admission.get("scope") == "materialize_prespecified_36_pair_72_arm_study"
        and admission.get("route_validation_outputs_reused") is False
        and document.get("quality_fit_eligible") is False
        and document.get("official") is False
        and document.get("rank_eligible") is False
    )


def _anchor_records(schedule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[object] = []
    for key in ("anchors", "selected_anchors", "anchor_tasks", "tasks"):
        candidates.append(schedule.get(key))
    nested = schedule.get("sensitivity_schedule")
    if isinstance(nested, Mapping):
        for key in ("anchors", "selected_anchors", "anchor_tasks", "tasks"):
            candidates.append(nested.get(key))
    coverage = schedule.get("coverage_repair_schedule")
    if isinstance(coverage, Mapping):
        for key in ("anchors", "entries", "tasks"):
            candidates.append(coverage.get(key))
    for candidate in candidates:
        if isinstance(candidate, list) and candidate and all(
            isinstance(item, Mapping) for item in candidate
        ):
            return list(candidate)  # type: ignore[arg-type]
    by_family = schedule.get("anchors_by_family")
    if isinstance(by_family, Mapping):
        records: list[Mapping[str, Any]] = []
        for family, value in by_family.items():
            if isinstance(value, str):
                records.append({"family": family, "task_id": value})
            elif isinstance(value, Mapping):
                records.append({"family": family, **value})
        if records:
            return records
    raise SensitivityProtocolError("coverage schedule has no recognized anchor list")


def _schedule_quarantine_ids(schedule: Mapping[str, Any]) -> set[str]:
    values: list[object] = [
        schedule.get("quarantined_task_ids"),
        schedule.get("excluded_task_ids"),
        schedule.get("suspect_task_ids"),
    ]
    quarantine = schedule.get("quarantine")
    if isinstance(quarantine, Mapping):
        values.extend(
            [
                quarantine.get("task_ids"),
                quarantine.get("quarantined_task_ids"),
                quarantine.get("excluded_task_ids"),
            ]
        )
    return {
        str(item)
        for value in values
        if isinstance(value, list)
        for item in value
        if isinstance(item, str) and item
    }


def _load_anchors(
    *,
    schedule_path: Path,
    task_validity_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    schedule, schedule_sha256 = _verified_artifact(schedule_path)
    validity, validity_sha256 = _verified_artifact(task_validity_path)
    tasks = validity.get("tasks")
    counts = validity.get("counts")
    claim_boundary = validity.get("claim_boundary")
    if (
        not isinstance(tasks, list)
        or not isinstance(counts, Mapping)
        or counts.get("synthetic_tasks") != 0
        or not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("rank_eligible") is not False
        or claim_boundary.get("supports_official_leaderboard") is not False
        or claim_boundary.get("synthetic_tasks") != 0
    ):
        raise SensitivityProtocolError("task-validity artifact violates development boundary")
    by_id = {
        str(record.get("task_id") or ""): record
        for record in tasks
        if isinstance(record, Mapping) and record.get("task_id")
    }
    quarantine_ids = _schedule_quarantine_ids(schedule)
    selected: list[dict[str, Any]] = []
    for anchor in _anchor_records(schedule):
        task_id = str(anchor.get("task_id") or anchor.get("public_id") or "")
        family = str(anchor.get("family") or "")
        source = by_id.get(task_id)
        if source is None:
            raise SensitivityProtocolError(f"anchor is absent from the task dossier: {task_id}")
        source_family = str(source.get("family") or "")
        if not family:
            family = source_family
        prompt = str(source.get("prompt") or "")
        prompt_sha256 = str(source.get("prompt_sha256") or "")
        anchor_prompt_sha256 = str(anchor.get("prompt_sha256") or prompt_sha256)
        record_quarantined = anchor.get("quarantined") is True or anchor.get("suspect") is True
        if (
            family not in TASK_FAMILIES
            or family != source_family
            or task_id in quarantine_ids
            or record_quarantined
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256
            or anchor_prompt_sha256 != prompt_sha256
            or source.get("rank_eligible") is not False
            or source.get("confirmatory_eligible") is not False
        ):
            raise SensitivityProtocolError(
                f"anchor is not a non-suspect development task: {task_id}"
            )
        selected.append(
            {
                "task_id": task_id,
                "family": family,
                "prompt_sha256": prompt_sha256,
                "task_sha256": source.get("task_sha256"),
                "source_url": source.get("source_url"),
                "source_license": source.get("source_license"),
                "synthetic": False,
                "official": False,
                "rank_eligible": False,
            }
        )
    families = Counter(record["family"] for record in selected)
    if len(selected) != 4 or families != Counter({family: 1 for family in TASK_FAMILIES}):
        raise SensitivityProtocolError("schedule must contain one unique anchor per task family")
    if len({record["task_id"] for record in selected}) != 4:
        raise SensitivityProtocolError("anchor task IDs must be unique")
    return sorted(selected, key=lambda item: TASK_FAMILIES.index(item["family"])), {
        "coverage_schedule_sha256": schedule_sha256,
        "task_validity_sha256": validity_sha256,
    }


def _low_configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    design = manifest.get("run_design")
    if not isinstance(design, Mapping):
        raise SensitivityProtocolError("base manifest has no run design")
    protocol = design.get("generation_protocol")
    policy = design.get("execution_policy")
    if not isinstance(protocol, Mapping) or not isinstance(policy, Mapping):
        raise SensitivityProtocolError("base manifest has no frozen generation policy")
    reasoning = policy.get("reasoning")
    if (
        protocol.get("intermediate_reasoning_effort") != "low"
        or protocol.get("final_reasoning_effort") != "low"
        or not isinstance(reasoning, Mapping)
        or reasoning.get("intermediate_effort") != "low"
        or reasoning.get("final_effort") != "low"
    ):
        raise SensitivityProtocolError("base manifest is not the explicit-low study")
    return {
        "base_execution_policy_sha256": design.get("execution_policy_sha256"),
        "intermediate_reasoning_effort": "low",
        "final_reasoning_effort": "low",
        "provider_response_reasoning_excluded": reasoning.get("exclude_from_provider_response"),
        "decoding": policy.get("decoding"),
        "limits": policy.get("limits"),
        "evidence_protocol": policy.get("evidence_protocol"),
        "final_response_mode": policy.get("final_response_mode"),
        "matched_planning": policy.get("matched_planning"),
    }


def _model_records(
    manifest: Mapping[str, Any],
    model_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], Decimal]:
    entries = manifest.get("models")
    if not isinstance(entries, list):
        raise SensitivityProtocolError("base manifest has no model entries")
    by_id = {
        str((entry.get("model") or {}).get("id") or ""): entry
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("model"), Mapping)
    }
    selected: list[dict[str, Any]] = []
    per_variant_forecast = Decimal(0)
    for model_id in model_ids:
        entry = by_id.get(model_id)
        if not isinstance(entry, Mapping):
            raise SensitivityProtocolError(f"model is absent from the base manifest: {model_id}")
        model = entry.get("model")
        endpoint = entry.get("endpoint")
        route = entry.get("execution_route")
        forecast = entry.get("forecast")
        if not all(isinstance(value, Mapping) for value in (model, endpoint, route, forecast)):
            raise SensitivityProtocolError(f"model contract is incomplete: {model_id}")
        assert isinstance(model, Mapping)
        assert isinstance(endpoint, Mapping)
        assert isinstance(route, Mapping)
        assert isinstance(forecast, Mapping)
        reasoning = model.get("reasoning")
        supported = (
            {str(value) for value in reasoning.get("supported_efforts") or []}
            if isinstance(reasoning, Mapping)
            else set()
        )
        endpoint_parameters = {str(value) for value in endpoint.get("supported_parameters") or []}
        backend = str(route.get("selected_backend") or "openrouter")
        endpoint_document_sha256 = _sha256_text(
            entry.get("endpoint_document_sha256"),
            field=f"{model_id} endpoint document",
        )
        if not str(endpoint.get("tag") or "") or not str(endpoint.get("provider_name") or ""):
            raise SensitivityProtocolError(f"{model_id} exact provider route is incomplete")
        effort_transport_supported = "reasoning" in endpoint_parameters and (
            "reasoning_effort" in endpoint_parameters or backend == "kimi_direct"
        )
        if not {"low", "high"}.issubset(supported) or not effort_transport_supported:
            raise SensitivityProtocolError(
                f"{model_id} does not support both low and high on the frozen route"
            )
        pairs = int(forecast.get("pairs") or 0)
        block_forecast = _decimal(
            forecast.get("model_block_worst_case_usd"),
            field=f"{model_id} model-block forecast",
        )
        if pairs != 4:
            raise SensitivityProtocolError(
                f"{model_id} forecast is not bound to four one-per-family pairs"
            )
        per_variant_forecast += block_forecast
        selected.append(
            {
                "slot_id": str((entry.get("slot") or {}).get("slot_id") or ""),
                "model_id": model_id,
                "canonical_model_slug": model.get("canonical_slug"),
                "model_name": model.get("name"),
                "provider_endpoint": endpoint.get("tag"),
                "actual_provider_name": endpoint.get("provider_name"),
                "execution_backend": backend,
                "endpoint_document_sha256": endpoint_document_sha256,
                "backend_contract_sha256": entry.get("backend_contract_sha256"),
                "supported_efforts": sorted(supported),
                "provider_default_effort": reasoning.get("default_effort"),
                "provider_default_enabled": reasoning.get("default_enabled"),
                "provider_default_mandatory": reasoning.get("mandatory"),
                "per_variant_four_pair_worst_case_usd": _decimal_text(block_forecast),
            }
        )
    return selected, per_variant_forecast


def _budget_state(path: Path) -> tuple[dict[str, Any], str]:
    document, digest = _verified_artifact(path)
    if (
        document.get("schema_version") != "flavourbench-frontier-global-budget-audit-v1"
        or document.get("currency") != "USD"
        or document.get("synthetic_sources") != 0
        or document.get("hard_cap_respected") is not True
    ):
        raise SensitivityProtocolError("budget audit violates the real-call governor contract")
    return document, digest


def build_plan(
    *,
    base_manifest_path: Path,
    lineage_inventory_path: Path,
    coverage_schedule_path: Path,
    task_validity_path: Path,
    budget_audit_path: Path,
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
) -> dict[str, Any]:
    """Freeze the complete no-call sensitivity workload and admission state."""

    manifest = _regular_json(base_manifest_path)
    if not verify_manifest_content_address(manifest):
        raise SensitivityProtocolError("base manifest content address does not verify")
    manifest_sha256 = str((manifest.get("content_address") or {}).get("digest") or "")
    if (
        manifest.get("status") != "unranked_candidate"
        or manifest.get("official_results_authorised") is not False
        or manifest.get("generation_calls_made") != 0
    ):
        raise SensitivityProtocolError("base manifest violates the development-only boundary")
    low_configuration = _low_configuration(manifest)

    lineage = _regular_json(lineage_inventory_path)
    if not verify_inventory(lineage):
        raise SensitivityProtocolError("Epicure lineage inventory does not verify")
    lineage_sha256 = str(lineage["artifact_sha256"])
    anchors, task_sources = _load_anchors(
        schedule_path=coverage_schedule_path,
        task_validity_path=task_validity_path,
    )
    models, per_variant_forecast = _model_records(manifest, model_ids)
    budget, budget_sha256 = _budget_state(budget_audit_path)
    total_forecast = per_variant_forecast * Decimal(len(VARIANTS))
    current_exposure = _decimal(
        budget.get("current_total_exposure_usd"), field="current budget exposure"
    )
    admission_ceiling = _decimal(
        budget.get("admission_ceiling_usd"), field="budget admission ceiling"
    )
    hard_cap = _decimal(budget.get("hard_cap_usd"), field="budget hard cap")
    projected = current_exposure + total_forecast

    collection_blockers: list[dict[str, str]] = []
    officialization_blockers: list[dict[str, str]] = []
    release_gates = lineage.get("release_gates")
    if lineage.get("rank_eligible") is not True:
        officialization_blockers.append(
            {
                "gate": "epicure_rank_eligible_release",
                "reason": "recovered Epicure runtime remains rank_eligible=false",
            }
        )
    if lineage.get("redistributable") is not True:
        officialization_blockers.append(
            {
                "gate": "epicure_redistributable_release",
                "reason": "payload rights attestation is absent",
            }
        )
    if not isinstance(release_gates, Mapping) or release_gates.get(
        "clean_signed_application_release"
    ) is not True:
        officialization_blockers.append(
            {
                "gate": "epicure_immutable_application_release",
                "reason": "runtime uses a dirty development source manifest, not a signed release",
            }
        )
    runtime_attestation = lineage.get("runtime_attestation")
    if not isinstance(runtime_attestation, Mapping) or runtime_attestation.get(
        "matches_recovered_checkout"
    ) is not True:
        collection_blockers.append(
            {
                "gate": "epicure_runtime_parity",
                "reason": (
                    "live private runtime has not attested parity with the recovered checkout"
                ),
            }
        )
    try:
        _sha256_text((lineage.get("bundle") or {}).get("sha256"), field="Epicure bundle")
        _sha256_text(
            (lineage.get("application") or {}).get("sha256"),
            field="Epicure application",
        )
        _sha256_text(
            (lineage.get("tool_contract") or {}).get("semantic_sha256"),
            field="Epicure semantic tool schema",
        )
    except SensitivityProtocolError as error:
        collection_blockers.append(
            {
                "gate": "epicure_content_address_and_tool_schema",
                "reason": str(error),
            }
        )
    if projected > admission_ceiling:
        collection_blockers.append(
            {
                "gate": "shared_budget_admission",
                "reason": (
                    f"projected ${_decimal_text(projected)} exceeds the frozen 85% admission "
                    f"ceiling ${_decimal_text(admission_ceiling)}"
                ),
            }
        )
    if projected > hard_cap:
        collection_blockers.append(
            {
                "gate": "shared_budget_hard_stop",
                "reason": (
                    f"projected ${_decimal_text(projected)} exceeds the hard cap "
                    f"${_decimal_text(hard_cap)}"
                ),
            }
        )

    work_items: list[dict[str, Any]] = []
    for model in models:
        for anchor in anchors:
            for variant in VARIANTS:
                core = {
                    "schema_version": "flavourbench-reasoning-effort-work-item-v1",
                    "base_manifest_sha256": manifest_sha256,
                    "lineage_inventory_sha256": lineage_sha256,
                    "coverage_schedule_sha256": task_sources["coverage_schedule_sha256"],
                    "model_id": model["model_id"],
                    "canonical_model_slug": model["canonical_model_slug"],
                    "provider_endpoint": model["provider_endpoint"],
                    "endpoint_document_sha256": model["endpoint_document_sha256"],
                    "task_id": anchor["task_id"],
                    "task_family": anchor["family"],
                    "prompt_sha256": anchor["prompt_sha256"],
                    "variant_id": variant["variant_id"],
                    "intermediate_reasoning_effort": variant[
                        "intermediate_reasoning_effort"
                    ],
                    "final_reasoning_effort": variant["final_reasoning_effort"],
                    "conditions": ["epicure_off", "epicure_on"],
                    "synthetic": False,
                    "official": False,
                    "rank_eligible": False,
                }
                work_items.append({"work_item_id": _sha256(core), **core})
    work_items.sort(key=lambda item: _sha256({"ordering_seed": "fb-reasoning-v1", **item}))

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "study_id": "frontier-reasoning-effort-sensitivity-v1",
        "study_role": "development_only_non_ranking_sensitivity_and_coverage_repair",
        "base_evidence": {
            "manifest_sha256": manifest_sha256,
            "manifest_filename": base_manifest_path.name,
            "manifest_filename_binds_digest": manifest_sha256 in base_manifest_path.name,
            "verified_explicit_low_configuration": low_configuration,
            "claim": (
                "The exact-frontier records used explicit low effort for intermediate and "
                "final model phases; this is now disclosed as a frozen configuration fact."
            ),
        },
        "epicure": {
            "lineage_inventory_sha256": lineage_sha256,
            "runtime_id": lineage.get("runtime_id"),
            "bundle_sha256": (lineage.get("bundle") or {}).get("sha256"),
            "application_sha256": (lineage.get("application") or {}).get("sha256"),
            "tool_schema_sha256": (lineage.get("tool_contract") or {}).get(
                "semantic_sha256"
            ),
            "rank_eligible": lineage.get("rank_eligible"),
            "redistributable": lineage.get("redistributable"),
        },
        "task_design": {
            **task_sources,
            "source_class": "licensed_real_human_authored_public_questions",
            "one_anchor_per_family": True,
            "anchors": anchors,
            "synthetic_tasks": 0,
            "official": False,
            "rank_eligible": False,
        },
        "model_design": {
            "selection_rule": (
                "prespecified common exact-route subset spanning closed frontier, "
                "frontier generalist, and efficient open-weight families under the shared cap"
            ),
            "model_count": len(models),
            "models": models,
            "all_models_support_low_and_high": True,
        },
        "reasoning_variants": [
            {
                **variant,
                "provider_default_effort_by_model": {
                    model["model_id"]: model["provider_default_effort"] for model in models
                }
                if variant["variant_id"] == "provider_default"
                else None,
            }
            for variant in VARIANTS
        ],
        "execution": {
            "conditions": ["epicure_off", "epicure_on"],
            "pairs": len(work_items),
            "response_arms": len(work_items) * 2,
            "provider_call_lower_bound": len(work_items) * 2,
            "tool_rule": (
                "Every Epicure-on arm must contain at least one successful real Epicure call."
            ),
            "real_calls_only": True,
            "synthetic_arms": 0,
            "append_only": True,
            "pair_arm_scheduling": "concurrent_within_work_item",
            "work_item_order": "sha256_committed_permutation",
            "work_items": work_items,
            "paid_execution_confirmation": "RUN_REAL_REASONING_EFFORT_SENSITIVITY",
        },
        "budget": {
            "audit_sha256": budget_sha256,
            "currency": "USD",
            "current_conservative_exposure_usd": _decimal_text(current_exposure),
            "per_variant_worst_case_usd": _decimal_text(per_variant_forecast),
            "study_worst_case_usd": _decimal_text(total_forecast),
            "projected_conservative_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(admission_ceiling),
            "hard_cap_usd": _decimal_text(hard_cap),
            "admission_allowed_before_lineage_gates": projected <= admission_ceiling,
            "reservation_policy": (
                "Reserve each work item transactionally against the shared global ledger; "
                "reconcile actual provider metadata before admitting the next block."
            ),
        },
        "analysis_contract": {
            "primary_estimands": [
                "within-model provider-default minus explicit-low",
                "within-model explicit-high minus explicit-low",
                "effort-by-Epicure interaction",
            ],
            "quality_source": "new blinded human judgments only",
            "objective_outputs": [
                "normal-finish rate",
                "Epicure treatment success",
                "cost",
                "latency",
                "tool-call count",
            ],
            "inference": (
                "paired task contrasts with task-cluster resampling; never treat the two arms "
                "or repeated use of one generated answer as independent observations"
            ),
            "missingness": (
                "retain failed arms in reliability; no preference ballot if an arm fails"
            ),
            "ranking_use": "prohibited_development_sensitivity_only",
        },
        "preflight": {
            "decision": (
                "blocked_before_provider_call"
                if collection_blockers
                else "eligible_for_live_route_smoke"
            ),
            "collection_blockers": collection_blockers,
            "officialization_blockers": officialization_blockers,
            "provider_calls_made": False,
            "epicure_calls_made": False,
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "coverage_repair": "development_evidence_only",
            "can_supersede_existing_low_records": False,
            "can_enter_current_quality_fit": False,
        },
    }


def verify_plan(document: object) -> bool:
    if not isinstance(document, Mapping):
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return (
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and isinstance(digest, str)
        and len(digest) == 64
        and _sha256(unhashed) == digest
        and (document.get("claim_boundary") or {}).get("official") is False
        and (document.get("execution") or {}).get("synthetic_arms") == 0
    )


def build_preflight_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_plan(plan):
        raise SensitivityProtocolError("sensitivity plan does not verify")
    collection_blockers = list(
        (plan.get("preflight") or {}).get("collection_blockers") or []
    )
    officialization_blockers = list(
        (plan.get("preflight") or {}).get("officialization_blockers") or []
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "record_type": "no_call_admission_preflight",
        "plan_sha256": plan["artifact_sha256"],
        "checks": {
            "base_low_configuration_verified": True,
            "common_exact_endpoint_subset_verified": True,
            "one_non_suspect_anchor_per_family_verified": True,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
            "shared_budget_forecast_within_admission": (plan["budget"])[
                "admission_allowed_before_lineage_gates"
            ],
            "epicure_rank_eligible": plan["epicure"]["rank_eligible"],
            "epicure_redistributable": plan["epicure"]["redistributable"],
        },
        "decision": (
            "blocked_before_provider_call" if collection_blockers else "live_smoke_required"
        ),
        "collection_blockers": collection_blockers,
        "officialization_blockers": officialization_blockers,
        "provider_calls_made": 0,
        "epicure_calls_made": 0,
        "cost_usd": "0",
        "official": False,
        "rank_eligible": False,
    }


def _execution_policy_from_base(
    base_manifest: Mapping[str, Any],
    *,
    intermediate_effort: str | None,
    final_effort: str | None,
) -> ExecutionPolicy:
    design = base_manifest.get("run_design")
    document = design.get("execution_policy") if isinstance(design, Mapping) else None
    if not isinstance(document, Mapping):
        raise SensitivityProtocolError("base execution policy is absent")
    limits = document.get("limits")
    decoding = document.get("decoding")
    forecast = document.get("cost_forecast")
    if not all(isinstance(value, Mapping) for value in (limits, decoding, forecast)):
        raise SensitivityProtocolError("base execution policy is incomplete")
    assert isinstance(limits, Mapping)
    assert isinstance(decoding, Mapping)
    assert isinstance(forecast, Mapping)
    policy = ExecutionPolicy(
        max_output_tokens=int(limits["max_output_tokens"]),
        max_tool_rounds=int(limits["max_tool_rounds"]),
        max_tool_result_bytes=int(limits["max_tool_result_bytes"]),
        max_cumulative_tool_result_bytes=int(limits["max_cumulative_tool_result_bytes"]),
        max_tool_calls_per_round=int(limits["max_tool_calls_per_round"]),
        max_tool_calls_total=int(limits["max_tool_calls_total"]),
        max_provider_attempts=int(limits["max_provider_attempts"]),
        decoding_temperature=float(decoding["temperature"]),
        decoding_top_p=float(decoding["top_p"]),
        decoding_seed=int(decoding["seed"]),
        tool_argument_repair_turns=int(limits["tool_argument_repair_turns"]),
        approximate_non_user_prompt_bytes=int(
            forecast["approximate_non_user_prompt_bytes"]
        ),
        conservative_bytes_per_token=int(forecast["conservative_bytes_per_token"]),
        pair_arm_scheduling=str(document["pair_arm_scheduling"]),
        final_response_mode=str(document["final_response_mode"]),
        max_intermediate_tokens=int(limits["max_intermediate_tokens"]),
        required_tool_contract_max_intermediate_tokens=int(
            limits["required_tool_contract_max_intermediate_tokens"]
        ),
        matched_planning=bool(document["matched_planning"]),
        evidence_protocol=str(document["evidence_protocol"]),
        intermediate_reasoning_effort=intermediate_effort,
        final_reasoning_effort=final_effort,
        required_tool_contract_protocol=str(document["required_tool_contract_protocol"]),
        tool_catalog_bytes_bound=int(forecast["tool_catalog_bytes_bound"]),
        epicure_on_tool_required=bool(document["epicure_on_tool_required"]),
    )
    policy.validate()
    return policy


def _build_runner_task_dossier(
    *,
    plan: Mapping[str, Any],
    source_task_validity: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_ids = {
        str(anchor["task_id"]) for anchor in (plan.get("task_design") or {}).get("anchors") or []
    }
    source_tasks = source_task_validity.get("tasks")
    if not isinstance(source_tasks, list):
        raise SensitivityProtocolError("source task-validity dossier has no tasks")
    selected = [
        copy.deepcopy(record)
        for record in source_tasks
        if isinstance(record, Mapping) and str(record.get("task_id") or "") in anchor_ids
    ]
    if len(selected) != 4 or {str(record["task_id"]) for record in selected} != anchor_ids:
        raise SensitivityProtocolError("could not materialize all sensitivity anchor tasks")
    selected.sort(key=lambda record: str(record["task_id"]))
    payload = copy.deepcopy(dict(source_task_validity))
    payload.pop("artifact_sha256", None)
    payload["artifact_role"] = "reasoning_effort_sensitivity_runner_task_dossier"
    payload["status"] = "surface_clean_source_verified_development_candidate_not_confirmatory"
    payload["candidate_coordinate_sha256"] = str(plan["artifact_sha256"])
    payload["source_task_set_sha256"] = _sha256(
        [
            {
                "task_id": record["task_id"],
                "task_sha256": record.get("task_sha256"),
                "prompt_sha256": record["prompt_sha256"],
            }
            for record in selected
        ]
    )
    payload["selection_policy"] = {
        "method": "content_addressed_non_suspect_common_family_anchors",
        "coverage_schedule_sha256": plan["task_design"]["coverage_schedule_sha256"],
        "sensitivity_plan_sha256": plan["artifact_sha256"],
        "quality_observations_used": 0,
    }
    payload["tasks"] = selected
    payload["counts"] = {
        **dict(payload.get("counts") or {}),
        "selected_development_tasks": 4,
        "selected_recent_tasks": sum(bool(record.get("recent")) for record in selected),
        "per_family": {family: 1 for family in TASK_FAMILIES},
        "synthetic_tasks": 0,
        "independently_human_validated_tasks": 0,
        "tasks_with_human_authored_criterion_packs": 0,
    }
    payload["claim_boundary"] = {
        **dict(payload.get("claim_boundary") or {}),
        "official": False,
        "rank_eligible": False,
        "supports_official_leaderboard": False,
        "supports_confirmatory_task_validity": False,
        "supports_real_current_model_development_runs": True,
        "synthetic_tasks": 0,
        "reasoning_effort_sensitivity_only": True,
    }
    return payload


def _derive_runner_manifest(
    *,
    plan: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    task_dossier_path: Path,
    task_dossier: Mapping[str, Any],
    task_registry_digest: str,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {
        str(model["model_id"]) for model in (plan.get("model_design") or {}).get("models") or []
    }
    source_models = base_manifest.get("models")
    if not isinstance(source_models, list):
        raise SensitivityProtocolError("base manifest has no models")
    models = [
        copy.deepcopy(entry)
        for entry in source_models
        if isinstance(entry, Mapping)
        and str((entry.get("model") or {}).get("id") or "") in selected_ids
    ]
    if len(models) != len(selected_ids):
        raise SensitivityProtocolError("runner manifest model subset is incomplete")
    policy = _execution_policy_from_base(
        base_manifest,
        intermediate_effort=variant.get("intermediate_reasoning_effort"),
        final_effort=variant.get("final_reasoning_effort"),
    )
    tool_contract = required_tool_contract(policy)
    output = copy.deepcopy(dict(base_manifest))
    output.pop("content_address", None)
    output["models"] = models
    output["generation_calls_made"] = 0
    output["generation_spend_usd"] = "0"
    output["status"] = "unranked_candidate"
    output["official_results_authorised"] = False
    output["selection"] = {
        **dict(output.get("selection") or {}),
        "model_count": len(models),
        "quality_observations_used": 0,
        "performance_claim": "none; prespecified development sensitivity subset",
        "route_counts": {
            "kimi_direct": sum(
                (entry.get("execution_route") or {}).get("selected_backend") == "kimi_direct"
                for entry in models
            ),
            "bedrock": sum(
                (entry.get("execution_route") or {}).get("selected_backend") == "bedrock"
                for entry in models
            ),
            "openrouter_fallback": sum(
                (entry.get("execution_route") or {}).get("selected_backend") == "openrouter"
                for entry in models
            ),
        },
    }
    per_variant = _decimal(
        plan["budget"]["per_variant_worst_case_usd"], field="per-variant forecast"
    )
    cap = _decimal(plan["budget"]["hard_cap_usd"], field="hard cap")
    ceiling = _decimal(plan["budget"]["admission_ceiling_usd"], field="admission ceiling")
    output["budget"] = {
        "currency": "USD",
        "cap_usd": _decimal_text(cap),
        "admission_fraction": "0.85",
        "admission_ceiling_usd": _decimal_text(ceiling),
        "bounded_forecast_usd": _decimal_text(per_variant),
        "headroom_to_admission_ceiling_usd": _decimal_text(ceiling - per_variant),
        "within_cap": per_variant <= ceiling,
        "forecast_policy": "reasoning_sensitivity_exact_route_pair_reservation_v1",
        "source_plan_sha256": plan["artifact_sha256"],
    }
    design = dict(output.get("run_design") or {})
    generation = dict(design.get("generation_protocol") or {})
    generation.update(
        {
            "intermediate_reasoning_effort": variant.get(
                "intermediate_reasoning_effort"
            ),
            "final_reasoning_effort": variant.get("final_reasoning_effort"),
            "required_tool_contract": tool_contract,
            "required_tool_contract_sha256": tool_contract["content_address"]["digest"],
            "required_tool_contract_max_intermediate_tokens": (
                policy.required_tool_contract_max_intermediate_tokens
            ),
            "max_intermediate_tokens": policy.max_intermediate_tokens,
            "tool_catalog_bytes_bound": policy.tool_catalog_bytes_bound,
            "evidence_protocol": policy.evidence_protocol,
            "final_response_mode": policy.final_response_mode,
            "matched_planning": policy.matched_planning,
            "epicure_on_tool_required": policy.epicure_on_tool_required,
            "sensitivity_plan_sha256": plan["artifact_sha256"],
            "sensitivity_variant": variant["variant_id"],
            "provider_default_parameter_omitted": variant["variant_id"]
            == "provider_default",
        }
    )
    task_source = {
        "artifact_path": str(task_dossier_path),
        "artifact_sha256": task_dossier["artifact_sha256"],
        "candidate_coordinate_sha256": task_dossier["candidate_coordinate_sha256"],
        "source_task_bank_sha256": task_dossier.get("source_task_bank_sha256"),
        "source_class": "licensed_real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "confirmatory_eligible": False,
        "rank_eligible": False,
    }
    selection_seed = "flavourbench-reasoning-effort-sensitivity-v1-four-anchors"
    design.update(
        {
            "tasks_per_family_in_pool": 1,
            "selected_task_count": 4,
            "assignments_per_model": 4,
            "expected_pairs": len(models) * 4,
            "expected_arms": len(models) * 8,
            "selection_seed": selection_seed,
            "conditions": ["epicure_off", "epicure_on"],
            "execution_policy": policy.document(),
            "execution_policy_sha256": policy.sha256,
            "generation_protocol": generation,
            "task_source": task_source,
        }
    )
    output["run_design"] = design
    source = dict(output.get("source") or {})
    source.update(
        {
            "task_validity_artifact_sha256": task_dossier["artifact_sha256"],
            "task_candidate_coordinate_sha256": task_dossier[
                "candidate_coordinate_sha256"
            ],
            "task_registry_sha256": task_registry_digest,
            "reasoning_sensitivity_plan_sha256": plan["artifact_sha256"],
        }
    )
    output["source"] = source
    output["governance"] = {
        **dict(output.get("governance") or {}),
        "official": False,
        "rank_eligible": False,
        "manifest_class": "real_reasoning_effort_development_sensitivity_candidate",
        "ranking_use": "prohibited",
    }
    digest = _sha256(output)
    return {
        **output,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }


def _write_manifest(output_dir: Path, document: Mapping[str, Any]) -> Path:
    if not verify_manifest_content_address(document):
        raise SensitivityProtocolError("derived runner manifest does not verify")
    digest = str((document.get("content_address") or {}).get("digest") or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"flavourbench-reasoning-sensitivity-{digest}.json"
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise SensitivityProtocolError("content-addressed runner manifest conflict")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as file:
        temporary = Path(file.name)
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _runner_command(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    task_dossier_path: Path,
    variant: Mapping[str, Any],
    run_root: Path,
) -> list[str]:
    design = manifest["run_design"]
    policy = design["execution_policy"]
    limits = policy["limits"]
    decoding = policy["decoding"]
    cost = policy["cost_forecast"]
    command = [
        "flavourbench/.venv/bin/python",
        "-m",
        "flavourbench.real_dataset_runner",
        "--manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        manifest["content_address"]["digest"],
        "--task-pool-per-family",
        "1",
        "--assignments-per-model",
        "4",
        "--selection-seed",
        design["selection_seed"],
        "--task-validity-artifact",
        str(task_dossier_path),
        "--max-output-tokens",
        str(limits["max_output_tokens"]),
        "--max-intermediate-tokens",
        str(limits["max_intermediate_tokens"]),
        "--max-tool-rounds",
        str(limits["max_tool_rounds"]),
        "--max-tool-result-bytes",
        str(limits["max_tool_result_bytes"]),
        "--max-cumulative-tool-result-bytes",
        str(limits["max_cumulative_tool_result_bytes"]),
        "--max-tool-calls-per-round",
        str(limits["max_tool_calls_per_round"]),
        "--max-tool-calls-total",
        str(limits["max_tool_calls_total"]),
        "--max-provider-attempts",
        str(limits["max_provider_attempts"]),
        "--temperature",
        str(decoding["temperature"]),
        "--top-p",
        str(decoding["top_p"]),
        "--seed",
        str(decoding["seed"]),
        "--final-response-mode",
        policy["final_response_mode"],
        "--tool-catalog-bytes-bound",
        str(cost["tool_catalog_bytes_bound"]),
        "--evidence-protocol",
        policy["evidence_protocol"],
        "--require-epicure-call",
        "--prior-artifact-directory",
        "flavourbench/artifacts/live-smoke",
        "--prior-corrections-directory",
        "flavourbench/artifacts/corrections",
        "--source-directory",
        str(run_root / "source"),
        "--source-corrections-directory",
        str(run_root / "corrections"),
        "--source-resolution-directory",
        str(run_root / "resolutions"),
        "--response-directory",
        str(run_root / "responses"),
        "--ledger",
        str(run_root / "ledger.jsonl"),
        "--global-budget-lock-path",
        "flavourbench/artifacts/frontier-contract/ledger.jsonl",
        "--summary-directory",
        str(run_root / "summaries"),
        "--cap-usd",
        manifest["budget"]["cap_usd"],
    ]
    intermediate = variant.get("intermediate_reasoning_effort")
    final = variant.get("final_reasoning_effort")
    if intermediate is not None:
        command.extend(["--intermediate-reasoning-effort", str(intermediate)])
    if final is not None:
        command.extend(["--final-reasoning-effort", str(final)])
    return command


def materialize_runner_assets(
    *,
    plan: Mapping[str, Any],
    base_manifest_path: Path,
    source_task_validity_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write exact manifests and commands for the existing append-only runner."""

    if not verify_plan(plan):
        raise SensitivityProtocolError("sensitivity plan does not verify")
    if (plan.get("preflight") or {}).get("collection_blockers"):
        raise SensitivityProtocolError("collection blockers must be cleared before materialization")
    base_manifest = _regular_json(base_manifest_path)
    if not verify_manifest_content_address(base_manifest):
        raise SensitivityProtocolError("base manifest does not verify")
    if (base_manifest.get("content_address") or {}).get("digest") != plan["base_evidence"][
        "manifest_sha256"
    ]:
        raise SensitivityProtocolError("base manifest differs from the sensitivity plan")
    source_task_validity, source_digest = _verified_artifact(source_task_validity_path)
    if source_digest != plan["task_design"]["task_validity_sha256"]:
        raise SensitivityProtocolError("task-validity dossier differs from the sensitivity plan")

    task_payload = _build_runner_task_dossier(
        plan=plan,
        source_task_validity=source_task_validity,
    )
    task_dossier_path = _write(
        output_dir / "tasks",
        "reasoning-sensitivity-task-dossier",
        task_payload,
    )
    task_dossier = _regular_json(task_dossier_path)
    task_inventory, _ = load_development_task_inventory(task_dossier_path)
    registry_digest = task_registry_sha256(task_inventory)

    variants: list[dict[str, Any]] = []
    for variant in plan["reasoning_variants"]:
        manifest = _derive_runner_manifest(
            plan=plan,
            base_manifest=base_manifest,
            task_dossier_path=task_dossier_path,
            task_dossier=task_dossier,
            task_registry_digest=registry_digest,
            variant=variant,
        )
        manifest_path = _write_manifest(output_dir / "manifests", manifest)
        run_root = output_dir / "runs" / str(variant["variant_id"])
        variants.append(
            {
                "variant_id": variant["variant_id"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest["content_address"]["digest"],
                "dry_run_command": _runner_command(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    task_dossier_path=task_dossier_path,
                    variant=variant,
                    run_root=run_root,
                ),
                "live_command_suffix": [
                    "--execute",
                    "--confirm",
                    "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
                ],
                "smoke_command_suffix": [
                    "--execute",
                    "--confirm",
                    "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
                    "--max-new-pairs",
                    "1",
                ],
            }
        )
    return {
        "schema_version": "flavourbench-reasoning-effort-runner-assets-v1",
        "record_role": "directly_executable_existing_append_only_runner_contract",
        "plan_sha256": plan["artifact_sha256"],
        "task_dossier": str(task_dossier_path),
        "task_dossier_sha256": task_dossier["artifact_sha256"],
        "task_registry_sha256": registry_digest,
        "variants": variants,
        "execution_order": [
            "explicit_low",
            "provider_default",
            "explicit_high",
        ],
        "environment_requirements": {
            "execution_mode": "live",
            "live_authorized": True,
            "openrouter_api_key": "present_but_never_serialized",
            "mcp_token": "present_but_never_serialized",
            "mcp_url": "private_exact_runtime",
            "epicure_identity": plan["epicure"],
        },
        "operator_rule": (
            "Run each dry-run first, then one-pair smoke per variant, regenerate the shared "
            "budget audit, and only then continue the resumable full commands. Never replay an "
            "active reservation."
        ),
        "provider_calls_made": False,
        "epicure_calls_made": False,
        "official": False,
        "rank_eligible": False,
    }


def materialize_v2_route_assets(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    v1_runner_assets: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write exact no-call runner manifests and commands for the three v2 pairs."""

    if not verify_v2_route_validation_plan(plan):
        raise SensitivityProtocolError("v2 route-validation plan does not verify")
    plan_from_path = _regular_json(plan_path)
    if plan_from_path != dict(plan):
        raise SensitivityProtocolError("v2 route plan path differs from the supplied plan")
    if (
        v1_runner_assets.get("plan_sha256") != plan["source"]["v1_plan_sha256"]
        or v1_runner_assets.get("schema_version")
        != "flavourbench-reasoning-effort-runner-assets-v1"
    ):
        raise SensitivityProtocolError("v1 runner assets differ from the v2 source boundary")
    task_dossier_path = Path(str(v1_runner_assets.get("task_dossier") or ""))
    task_dossier, task_dossier_sha256 = _verified_artifact(task_dossier_path)
    if task_dossier_sha256 != v1_runner_assets.get("task_dossier_sha256"):
        raise SensitivityProtocolError("v1 runner task dossier does not verify")
    source_variants = v1_runner_assets.get("variants")
    if not isinstance(source_variants, list) or len(source_variants) != len(VARIANTS):
        raise SensitivityProtocolError("v1 runner assets lack the three effort variants")
    source_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in source_variants
        if isinstance(item, Mapping)
    }
    planned_by_variant = {
        str(item["variant_id"]): item
        for item in plan["route_validation"]["work_items"]
    }
    if set(source_by_variant) != set(planned_by_variant):
        raise SensitivityProtocolError("v2 route variants differ from v1 runner assets")

    variants: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_id = str(variant["variant_id"])
        source_asset = source_by_variant[variant_id]
        planned = planned_by_variant[variant_id]
        source_manifest_path = Path(str(source_asset.get("manifest") or ""))
        source_manifest = _regular_json(source_manifest_path)
        if (
            not verify_manifest_content_address(source_manifest)
            or (source_manifest.get("content_address") or {}).get("digest")
            != source_asset.get("manifest_sha256")
            or (source_manifest.get("run_design") or {}).get("execution_policy_sha256")
            != planned.get("execution_policy_sha256")
        ):
            raise SensitivityProtocolError(f"{variant_id} v1 source manifest does not verify")
        manifest = copy.deepcopy(source_manifest)
        manifest.pop("content_address", None)
        source = dict(manifest.get("source") or {})
        source.update(
            {
                "v2_route_validation_plan_sha256": plan["artifact_sha256"],
                "v1_smoke_audit_sha256": plan["source"]["v1_smoke_audit_sha256"],
                "v1_closed_identifiers_sha256": plan["source"][
                    "v1_closed_identifiers_sha256"
                ],
                "corrected_lineage_inventory_sha256": plan["source"][
                    "corrected_lineage_inventory_sha256"
                ],
                "provider_source_sha256": plan["source"]["provider_source_sha256"],
            }
        )
        manifest["source"] = source
        design = dict(manifest.get("run_design") or {})
        generation = dict(design.get("generation_protocol") or {})
        generation.update(
            {
                "v2_route_validation_plan_sha256": plan["artifact_sha256"],
                "v2_route_cell_id": plan["route_validation"]["route_cell_id"],
                "v2_route_variant_id": variant_id,
                "v2_effective_work_item_id": planned["work_item_id"],
            }
        )
        design["generation_protocol"] = generation
        design["route_validation_override"] = {
            "plan_sha256": plan["artifact_sha256"],
            "variant_id": variant_id,
            "model_id": planned["model_id"],
            "task_id": planned["task_id"],
            "work_item_id": planned["work_item_id"],
            "matched_pairs": 1,
            "response_arms": 2,
            "diagnostic_only": True,
        }
        manifest["run_design"] = design
        manifest["governance"] = {
            **dict(manifest.get("governance") or {}),
            "manifest_class": "real_reasoning_effort_v2_route_validation",
            "v2_route_validation_only": True,
            "official": False,
            "rank_eligible": False,
            "ranking_use": "prohibited",
        }
        manifest["budget"] = {
            **dict(manifest.get("budget") or {}),
            "bounded_forecast_usd": planned["worst_case_reserve_usd"],
            "forecast_policy": "v2_one_pair_route_validation_reservation",
            "source_plan_sha256": plan["artifact_sha256"],
        }
        manifest_digest = _sha256(manifest)
        manifest = {
            **manifest,
            "content_address": {
                "algorithm": "sha256",
                "digest": manifest_digest,
                "uri": f"sha256:{manifest_digest}",
            },
        }
        manifest_path = _write_manifest(output_dir / "manifests", manifest)
        run_root = output_dir / "runs" / variant_id
        command = _runner_command(
            manifest_path=manifest_path,
            manifest=manifest,
            task_dossier_path=task_dossier_path,
            variant=variant,
            run_root=run_root,
        )
        command.extend(
            [
                "--route-validation-plan",
                str(plan_path),
                "--route-validation-variant",
                variant_id,
            ]
        )
        variants.append(
            {
                "variant_id": variant_id,
                "fresh_work_item_id": planned["work_item_id"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_digest,
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
        "schema_version": "flavourbench-reasoning-effort-v2-route-runner-assets-v1",
        "record_role": "exact_no_call_commands_for_three_fresh_v2_diagnostic_pairs",
        "v2_route_plan_sha256": plan["artifact_sha256"],
        "v1_runner_assets_sha256": v1_runner_assets["artifact_sha256"],
        "task_dossier": str(task_dossier_path),
        "task_dossier_sha256": task_dossier_sha256,
        "variants": variants,
        "execution_order": ["explicit_low", "provider_default", "explicit_high"],
        "operator_rule": (
            "Run all three dry runs first. Execute live commands strictly in the recorded order, "
            "regenerate the pass/fail audit after each command, and stop permanently on the first "
            "failed predicate. Never replay a v2 work-item ID."
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
        raise SensitivityProtocolError(f"runner command has no {option} value") from error
    return Path(value)


def _v2_route_runner_contract(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify the no-call assets and return their exact, frozen route bindings."""

    if not verify_v2_route_validation_plan(plan):
        raise SensitivityProtocolError("v2 route-validation plan does not verify")
    if not _artifact_sha256_verifies(
        runner_assets,
        "flavourbench-reasoning-effort-v2-route-runner-assets-v1",
    ):
        raise SensitivityProtocolError("v2 route runner assets do not verify")
    if runner_assets.get("v2_route_plan_sha256") != plan["artifact_sha256"]:
        raise SensitivityProtocolError("v2 route runner assets differ from the plan")
    variants = runner_assets.get("variants")
    if not isinstance(variants, list) or len(variants) != len(VARIANTS):
        raise SensitivityProtocolError("v2 route runner assets lack three variants")
    assets_by_variant = {
        str(item.get("variant_id") or ""): item
        for item in variants
        if isinstance(item, Mapping)
    }
    planned_by_variant = {
        str(item["variant_id"]): item
        for item in plan["route_validation"]["work_items"]
    }
    if (
        set(assets_by_variant) != set(planned_by_variant)
        or runner_assets.get("execution_order")
        != plan["route_validation"]["execution_order"]
    ):
        raise SensitivityProtocolError("v2 route variant set or order differs from the plan")

    bindings: list[dict[str, Any]] = []
    for variant_id in plan["route_validation"]["execution_order"]:
        asset = assets_by_variant[str(variant_id)]
        planned = planned_by_variant[str(variant_id)]
        dry_command = asset.get("dry_run_command")
        live_command = asset.get("live_command")
        if not (
            isinstance(dry_command, list)
            and isinstance(live_command, list)
            and all(isinstance(value, str) for value in dry_command + live_command)
            and live_command
            == [
                *dry_command,
                "--execute",
                "--confirm",
                "RUN_SEQUENTIAL_UNRANKED_REAL_DATASET",
            ]
        ):
            raise SensitivityProtocolError(f"{variant_id} commands are not exact")
        run_root = Path(str(asset.get("run_root") or ""))
        expected_paths = {
            "--source-directory": run_root / "source",
            "--response-directory": run_root / "responses",
            "--ledger": run_root / "ledger.jsonl",
            "--summary-directory": run_root / "summaries",
        }
        if not str(run_root) or any(
            _command_option(dry_command, option) != expected
            for option, expected in expected_paths.items()
        ):
            raise SensitivityProtocolError(f"{variant_id} run paths are not isolated")
        if (
            _command_option(dry_command, "--route-validation-plan").name
            != f"reasoning-effort-v2-route-validation-plan-{plan['artifact_sha256']}.json"
            or str(dry_command[dry_command.index("--route-validation-variant") + 1])
            != variant_id
            or asset.get("fresh_work_item_id") != planned["work_item_id"]
        ):
            raise SensitivityProtocolError(f"{variant_id} route override differs from the plan")

        manifest_path = Path(str(asset.get("manifest") or ""))
        manifest = _regular_json(manifest_path)
        manifest_digest = str((manifest.get("content_address") or {}).get("digest") or "")
        design = manifest.get("run_design")
        source = manifest.get("source")
        governance = manifest.get("governance")
        if not (
            verify_manifest_content_address(manifest)
            and manifest_digest == asset.get("manifest_sha256")
            and isinstance(design, Mapping)
            and isinstance(source, Mapping)
            and isinstance(governance, Mapping)
        ):
            raise SensitivityProtocolError(f"{variant_id} route manifest does not verify")
        override = design.get("route_validation_override")
        generation = design.get("generation_protocol")
        model_entries = manifest.get("models")
        matching_models = [
            entry
            for entry in model_entries or []
            if isinstance(entry, Mapping)
            and (entry.get("model") or {}).get("id") == planned["model_id"]
        ]
        if not (
            isinstance(override, Mapping)
            and isinstance(generation, Mapping)
            and len(matching_models) == 1
            and design.get("execution_policy_sha256")
            == planned["execution_policy_sha256"]
            and override.get("plan_sha256") == plan["artifact_sha256"]
            and override.get("variant_id") == variant_id
            and override.get("model_id") == planned["model_id"]
            and override.get("task_id") == planned["task_id"]
            and override.get("work_item_id") == planned["work_item_id"]
            and generation.get("v2_route_validation_plan_sha256")
            == plan["artifact_sha256"]
            and generation.get("v2_route_cell_id")
            == plan["route_validation"]["route_cell_id"]
            and generation.get("v2_route_variant_id") == variant_id
            and generation.get("v2_effective_work_item_id")
            == planned["work_item_id"]
            and generation.get("required_tool_contract_sha256")
            == planned["required_tool_contract_sha256"]
            and source.get("provider_source_sha256")
            == plan["source"]["provider_source_sha256"]
            and source.get("corrected_lineage_inventory_sha256")
            == plan["epicure"]["lineage_inventory_sha256"]
            and governance.get("v2_route_validation_only") is True
            and governance.get("official") is False
            and governance.get("rank_eligible") is False
        ):
            raise SensitivityProtocolError(f"{variant_id} manifest bindings differ")
        model_entry = matching_models[0]
        model = model_entry.get("model") or {}
        endpoint = model_entry.get("endpoint") or {}
        if not (
            model.get("canonical_slug") == planned["canonical_model_slug"]
            and endpoint.get("tag") == planned["provider_endpoint"]
            and endpoint.get("provider_name") == planned["actual_provider_name"]
            and endpoint_execution_contract_sha256(dict(endpoint))
            == planned["endpoint_execution_sha256"]
            and model_entry.get("endpoint_document_sha256")
            == planned["endpoint_document_sha256"]
        ):
            raise SensitivityProtocolError(f"{variant_id} endpoint binding differs")
        bindings.append(
            {
                "variant_id": variant_id,
                "asset": asset,
                "planned": planned,
                "manifest": manifest,
                "manifest_sha256": manifest_digest,
                "run_root": run_root,
            }
        )
    return bindings


def _v2_route_summary_matches(
    summary: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    variant_id: str,
    work_item_id: str,
) -> bool:
    workload = summary.get("workload")
    override = workload.get("route_validation_override") if isinstance(workload, Mapping) else None
    return bool(
        isinstance(override, Mapping)
        and override.get("plan_sha256") == plan["artifact_sha256"]
        and override.get("route_cell_id") == plan["route_validation"]["route_cell_id"]
        and override.get("variant_id") == variant_id
        and override.get("effective_fresh_work_item_id") == work_item_id
        and override.get("model_id") == plan["route_validation"]["model_id"]
        and override.get("task_id") == plan["route_validation"]["task_id"]
        and override.get("quality_fit_eligible") is False
    )


def _v2_route_journals(source_root: Path) -> list[tuple[Path, list[dict[str, Any]], str]]:
    journals: list[tuple[Path, list[dict[str, Any]], str]] = []
    if not source_root.exists():
        return journals
    paths = sorted(source_root.glob("flavourbench-live-smoke-journal-*.jsonl"))
    paths.extend(sorted(source_root.glob(".*.inprogress.jsonl")))
    for path in paths:
        try:
            entries = load_run_journal(path)
        except JournalIntegrityError as error:
            raise SensitivityProtocolError(f"route journal does not verify: {path}") from error
        journals.append((path, entries, _file_sha256(path)))
    return journals


def _v2_issue(
    issues: dict[str, list[str]],
    predicate_id: str,
    reason: str,
) -> None:
    if reason not in issues[predicate_id]:
        issues[predicate_id].append(reason)


def _audit_v2_route_variant(
    *,
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one variant audit from immutable runner outputs."""

    predicate_ids = [item["predicate_id"] for item in plan["acceptance_gate"]["predicates"]]
    issues: dict[str, list[str]] = {str(predicate_id): [] for predicate_id in predicate_ids}
    variant_id = str(binding["variant_id"])
    planned = binding["planned"]
    asset = binding["asset"]
    run_root = Path(binding["run_root"])
    work_item_id = str(planned["work_item_id"])
    source_root = run_root / "source"
    response_root = run_root / "responses"
    summary_root = run_root / "summaries"
    ledger_path = run_root / "ledger.jsonl"
    evidence = {str(plan["artifact_sha256"]), str(asset.get("manifest_sha256"))}

    summaries: list[tuple[dict[str, Any], str, Path]] = []
    unmatched_execute_summaries = 0
    for path in sorted(summary_root.glob("*.json")) if summary_root.exists() else []:
        summary, digest = _verified_artifact(path)
        if summary.get("mode") == "execute":
            evidence.add(digest)
            if _v2_route_summary_matches(
                summary,
                plan=plan,
                variant_id=variant_id,
                work_item_id=work_item_id,
            ):
                summaries.append((summary, digest, path))
            else:
                unmatched_execute_summaries += 1
    if unmatched_execute_summaries:
        _v2_issue(issues, "identifier_freshness", "unplanned_execute_summary_in_run_root")
        _v2_issue(issues, "response_integrity", "execute_summary_route_binding_mismatch")
    if len(summaries) > 1:
        _v2_issue(issues, "identifier_freshness", "multiple_execute_summaries_for_fresh_id")
        _v2_issue(issues, "response_integrity", "fresh_work_item_was_replayed")
    summary = summaries[0][0] if len(summaries) == 1 else None
    summary_digest = summaries[0][1] if len(summaries) == 1 else None
    summary_path = summaries[0][2] if len(summaries) == 1 else None

    try:
        ledger = load_dataset_ledger(ledger_path)
    except IntegrityError as error:
        raise SensitivityProtocolError(f"{variant_id} route ledger does not verify") from error
    if ledger:
        evidence.add(str(ledger[-1]["entry_sha256"]))
    relevant_ledger = [
        entry for entry in ledger if str(entry.get("work_item_id") or "") == work_item_id
    ]
    if any(
        str(entry.get("work_item_id") or "") not in {"", work_item_id}
        for entry in ledger
    ):
        _v2_issue(issues, "identifier_freshness", "unplanned_ledger_work_item")
        _v2_issue(issues, "budget_admission", "ledger_is_not_route_isolated")
        _v2_issue(issues, "response_integrity", "ledger_is_not_route_isolated")
    reservations = [
        entry for entry in relevant_ledger if entry.get("event_type") == "reservation_created"
    ]
    finalizations = [
        entry for entry in relevant_ledger if entry.get("event_type") == "source_artifact_recorded"
    ]
    incidents = [
        entry for entry in relevant_ledger if entry.get("event_type") == "execution_incident"
    ]

    source_candidates: list[tuple[dict[str, Any], str, Path]] = []
    for path in sorted(source_root.glob("*.json")) if source_root.exists() else []:
        try:
            source, digest = _verify_live_artifact(path)
        except IntegrityError as error:
            raise SensitivityProtocolError(
                f"{variant_id} immutable source does not verify"
            ) from error
        if str(source.get("dataset_work_item_id") or "") == work_item_id:
            source_candidates.append((source, digest, path))
            evidence.add(digest)
        else:
            _v2_issue(issues, "identifier_freshness", "unplanned_source_in_isolated_run_root")
    if len(source_candidates) > 1:
        _v2_issue(issues, "identifier_freshness", "multiple_sources_for_fresh_work_item")
        _v2_issue(issues, "response_integrity", "fresh_work_item_has_multiple_sources")
    source = source_candidates[0][0] if len(source_candidates) == 1 else None
    source_digest = source_candidates[0][1] if len(source_candidates) == 1 else None
    source_path = source_candidates[0][2] if len(source_candidates) == 1 else None

    discovered_journals = _v2_route_journals(source_root)
    matching_journals = []
    for path, entries, digest in discovered_journals:
        started = entries[0].get("payload")
        if isinstance(started, Mapping) and started.get("dataset_work_item_id") == work_item_id:
            matching_journals.append((path, entries, digest))
            evidence.add(digest)
        else:
            _v2_issue(issues, "identifier_freshness", "unplanned_journal_in_run_root")
            _v2_issue(issues, "response_integrity", "journal_is_not_route_isolated")
    attempted = bool(summaries or relevant_ledger or matching_journals or source_candidates)
    if not attempted:
        return {
            "variant_id": variant_id,
            "work_item_id": work_item_id,
            "attempted": False,
            "usable_pair": False,
            "issues": issues,
            "evidence_sha256": sorted(evidence),
            "source_artifact_sha256": None,
            "summary_artifact_sha256": None,
            "journal_sha256": None,
            "ledger_head_sha256": None,
            "response_artifact_sha256s": [],
            "attempt_ids": [],
            "generation_ids": [],
            "provider_requests": 0,
            "provider_responses": 0,
            "unreconciled_provider_requests": 0,
            "non_chat_completion_envelopes": 0,
            "unknown_http_200_responses": 0,
            "successful_epicure_tool_calls": 0,
            "epicure_off_tool_calls": 0,
            "identity_mismatches": 0,
            "truncated_or_invalid_arms": 0,
            "actual_cost_usd": Decimal(0),
            "retained_exposure_usd": Decimal(0),
        }

    if len(summaries) != 1:
        _v2_issue(issues, "complete_arms", "one_execute_summary_required")
        _v2_issue(issues, "response_integrity", "execute_summary_missing_or_duplicated")
    if len(reservations) != 1:
        _v2_issue(issues, "budget_admission", "one_ledger_reservation_required")
        _v2_issue(issues, "identifier_freshness", "reservation_missing_or_duplicated")
    if len(finalizations) != 1:
        _v2_issue(issues, "cost_reconciliation", "one_ledger_finalization_required")
        _v2_issue(issues, "complete_arms", "ledger_finalization_missing_or_duplicated")
    if incidents:
        _v2_issue(issues, "cost_reconciliation", "execution_incident_present")
        _v2_issue(issues, "response_integrity", "execution_incident_present")
    if len(source_candidates) != 1:
        for predicate_id in predicate_ids:
            _v2_issue(issues, str(predicate_id), "one_immutable_source_required")
        retained = sum(
            (_decimal(entry.get("reserved_usd"), field="route reserve") for entry in reservations),
            Decimal(0),
        )
        return {
            "variant_id": variant_id,
            "work_item_id": work_item_id,
            "attempted": True,
            "usable_pair": False,
            "issues": issues,
            "evidence_sha256": sorted(evidence),
            "source_artifact_sha256": None,
            "summary_artifact_sha256": summary_digest,
            "journal_sha256": matching_journals[0][2] if len(matching_journals) == 1 else None,
            "ledger_head_sha256": ledger[-1]["entry_sha256"] if ledger else None,
            "response_artifact_sha256s": [],
            "attempt_ids": [],
            "generation_ids": [],
            "provider_requests": 0,
            "provider_responses": 0,
            "unreconciled_provider_requests": 0,
            "non_chat_completion_envelopes": 0,
            "unknown_http_200_responses": 0,
            "successful_epicure_tool_calls": 0,
            "epicure_off_tool_calls": 0,
            "identity_mismatches": 0,
            "truncated_or_invalid_arms": 2,
            "actual_cost_usd": Decimal(0),
            "retained_exposure_usd": retained,
        }

    assert source is not None and source_digest is not None and source_path is not None
    descriptor = source.get("run_journal")
    if not isinstance(descriptor, Mapping):
        raise SensitivityProtocolError(f"{variant_id} source lacks a journal descriptor")
    try:
        verify_journal_descriptor(source_root, descriptor)
    except JournalIntegrityError as error:
        raise SensitivityProtocolError(f"{variant_id} source journal does not verify") from error
    journal_digest = str(descriptor.get("sha256") or "")
    evidence.add(journal_digest)
    if len(matching_journals) != 1 or matching_journals[0][2] != journal_digest:
        _v2_issue(issues, "identifier_freshness", "journal_missing_duplicated_or_unlinked")
        _v2_issue(issues, "response_integrity", "source_journal_link_not_unique")

    if not (
        source.get("candidate_manifest_sha256") == binding["manifest_sha256"]
        and source.get("dataset_task_id") == planned["task_id"]
        and source.get("category") == planned["task_family"]
        and source.get("prompt_sha256") == planned["prompt_sha256"]
        and source.get("requested_model_id") == planned["model_id"]
        and source.get("requested_provider") == planned["provider_endpoint"]
        and source.get("endpoint_execution_contract_sha256")
        == planned["endpoint_execution_sha256"]
        and source.get("execution_policy_sha256") == planned["execution_policy_sha256"]
    ):
        _v2_issue(issues, "model_provider_identity", "source_route_binding_mismatch")
    protocol = source.get("protocol_bundle")
    core_protocol = protocol.get("core_protocol_bundle") if isinstance(protocol, Mapping) else None
    implementation = (
        core_protocol.get("implementation_sha256")
        if isinstance(core_protocol, Mapping)
        else None
    )
    if not (
        isinstance(implementation, Mapping)
        and implementation.get("provider.py") == plan["source"]["provider_source_sha256"]
    ):
        _v2_issue(issues, "chat_completion_envelopes", "provider_classifier_source_unbound")
        _v2_issue(issues, "response_integrity", "provider_source_identity_mismatch")

    attempts = source.get("provider_attempt_events")
    if not isinstance(attempts, list) or not all(isinstance(event, Mapping) for event in attempts):
        raise SensitivityProtocolError(f"{variant_id} provider trace is invalid")
    request_events = [event for event in attempts if event.get("event_type") == "request_started"]
    response_events = [
        event for event in attempts if event.get("event_type") == "response_received"
    ]
    accounting_events = [
        event for event in attempts if event.get("event_type") == "accounting_reconciled"
    ]
    attempt_ids = [str(event.get("attempt_id") or "") for event in request_events]
    generation_ids = [str(event.get("generation_id") or "") for event in response_events]
    request_counter = Counter(attempt_ids)
    response_counter = Counter(str(event.get("attempt_id") or "") for event in response_events)
    accounting_counter = Counter(
        str(event.get("attempt_id") or "") for event in accounting_events
    )
    missing_generation_requests = sum(not value for value in generation_ids)
    if (
        not request_events
        or len(request_events) != len(response_events)
        or any(not value for value in attempt_ids)
        or any(count != 1 for count in request_counter.values())
        or request_counter != response_counter
    ):
        _v2_issue(issues, "generation_accounting", "request_response_bijection_failed")
    if (
        missing_generation_requests
        or response_counter != accounting_counter
        or any(
            not isinstance(event.get("metadata"), Mapping)
            or event["metadata"].get("reconciled") is not True
            or event["metadata"].get("generation_id") != event.get("generation_id")
            or not isinstance(event["metadata"].get("cost_micros"), int)
            for event in accounting_events
        )
    ):
        _v2_issue(issues, "generation_accounting", "generation_accounting_incomplete")
        _v2_issue(issues, "cost_reconciliation", "generation_accounting_incomplete")

    non_chat = 0
    unknown_http_200 = 0
    cache_or_ambiguity = False
    for event in response_events:
        metadata = event.get("metadata")
        envelope = metadata.get("response_envelope") if isinstance(metadata, Mapping) else None
        accepted = bool(
            isinstance(envelope, Mapping)
            and envelope.get("classification") == "chat_completions"
            and envelope.get("accepted_chat_completion") is True
        )
        if not accepted:
            non_chat += 1
            if event.get("http_status") == 200:
                unknown_http_200 += 1
        if isinstance(metadata, Mapping) and (
            str(metadata.get("openrouter_cache_status") or "").upper() == "HIT"
            or str(metadata.get("cloudflare_cache_status") or "").upper() == "HIT"
        ):
            cache_or_ambiguity = True
    if non_chat:
        _v2_issue(issues, "chat_completion_envelopes", "non_chat_completion_response")
    ambiguous_events = {
        "uncertain_delivery",
        "ambiguous_delivery",
        "ambiguous_delivery_retry",
    }
    if cache_or_ambiguity or any(
        event.get("event_type") in ambiguous_events for event in attempts
    ):
        _v2_issue(issues, "response_integrity", "cached_or_ambiguous_delivery")

    results = source.get("results")
    errors = source.get("errors")
    result_map = results if isinstance(results, Mapping) else {}
    condition_results = {
        condition: result_map.get(condition)
        for condition in ("epicure_off", "epicure_on")
    }
    invalid_arms = 0
    identity_mismatches = 0
    result_generation_ids: list[str] = []
    result_cost_micros = 0
    for condition, result in condition_results.items():
        if not isinstance(result, Mapping):
            invalid_arms += 1
            continue
        finish_reason = str(result.get("finish_reason") or "")
        intermediate = result.get("intermediate_outputs")
        invalid = bool(
            not str(result.get("answer_markdown") or "").strip()
            or finish_reason in {"length", "max_tokens", "content_filter"}
            or not isinstance(intermediate, list)
            or any(
                not isinstance(item, Mapping)
                or item.get("truncated") is not False
                or item.get("finish_reason") in {"length", "max_tokens", "content_filter"}
                for item in intermediate or []
            )
            or result.get("structured_output_valid") is False
        )
        if invalid:
            invalid_arms += 1
        if (
            result.get("actual_model_id") != planned["canonical_model_slug"]
            or result.get("actual_provider") != planned["actual_provider_name"]
        ):
            identity_mismatches += 1
        ids = result.get("generation_ids")
        metadata = result.get("generation_metadata")
        if not isinstance(ids, list) or not isinstance(metadata, list):
            _v2_issue(issues, "generation_accounting", f"{condition}_generation_list_invalid")
            continue
        result_generation_ids.extend(str(value) for value in ids)
        if (
            result.get("generation_id") not in ids
            or result.get("cost_reconciled") is not True
            or any(
                not isinstance(item, Mapping)
                or item.get("reconciled") is not True
                or item.get("generation_id") not in ids
                or item.get("model") != planned["canonical_model_slug"]
                or item.get("provider") != planned["actual_provider_name"]
                for item in metadata
            )
        ):
            _v2_issue(issues, "generation_accounting", f"{condition}_metadata_mismatch")
        cost_micros = result.get("cost_micros")
        if not isinstance(cost_micros, int) or isinstance(cost_micros, bool) or cost_micros < 0:
            _v2_issue(issues, "cost_reconciliation", f"{condition}_cost_invalid")
        else:
            result_cost_micros += cost_micros
    if source.get("status") != "complete" or errors or invalid_arms:
        _v2_issue(issues, "complete_arms", "source_does_not_contain_two_normal_arms")
    if invalid_arms:
        _v2_issue(issues, "response_integrity", "truncated_or_invalid_final_arm")
    if identity_mismatches:
        _v2_issue(issues, "model_provider_identity", "returned_identity_mismatch")
    if sorted(result_generation_ids) != sorted(generation_ids):
        _v2_issue(issues, "generation_accounting", "result_and_trace_generation_ids_differ")
    if source.get("incomplete_generation_metadata"):
        _v2_issue(issues, "generation_accounting", "incomplete_generation_metadata_present")
        _v2_issue(issues, "cost_reconciliation", "incomplete_generation_metadata_present")

    off_result = condition_results["epicure_off"]
    on_result = condition_results["epicure_on"]
    off_trace = off_result.get("tool_trace") if isinstance(off_result, Mapping) else None
    on_trace = on_result.get("tool_trace") if isinstance(on_result, Mapping) else None
    off_tool_calls = len(off_trace) if isinstance(off_trace, list) else 0
    successful_tool_calls = sum(
        isinstance(item, Mapping) and item.get("is_error") is False
        for item in on_trace or []
    ) if isinstance(on_trace, list) else 0
    mcp_events = source.get("mcp_trace_events")
    if not isinstance(mcp_events, list):
        raise SensitivityProtocolError(f"{variant_id} MCP trace is invalid")
    if any(not str(event.get("arm_id") or "").endswith(":epicure_on") for event in mcp_events):
        off_tool_calls += sum(
            not str(event.get("arm_id") or "").endswith(":epicure_on")
            for event in mcp_events
        )
    if successful_tool_calls < 1:
        _v2_issue(issues, "epicure_on_treatment", "no_successful_epicure_on_tool_call")
    if off_tool_calls:
        _v2_issue(issues, "epicure_off_control", "epicure_off_tool_call_present")

    epicure = source.get("epicure")
    frozen = source.get("frozen_generation_contract")
    exact_epicure = bool(
        isinstance(epicure, Mapping)
        and isinstance(frozen, Mapping)
        and epicure.get("bundle_sha256") == plan["epicure"]["bundle_sha256"]
        and epicure.get("application_sha256") == plan["epicure"]["application_sha256"]
        and source.get("epicure_tool_schema_sha256")
        == plan["epicure"]["tool_schema_sha256"]
        and frozen.get("expected_epicure_bundle_sha256")
        == plan["epicure"]["bundle_sha256"]
        and frozen.get("expected_epicure_application_sha256")
        == plan["epicure"]["application_sha256"]
        and frozen.get("expected_epicure_tool_schema_sha256")
        == plan["epicure"]["tool_schema_sha256"]
    )
    if not exact_epicure:
        _v2_issue(issues, "epicure_runtime_identity", "epicure_runtime_hash_mismatch")

    response_artifacts = []
    if response_root.exists():
        for path in sorted(response_root.glob("*.json")):
            try:
                response = _verify_response_artifact(path)
            except IntegrityError as error:
                raise SensitivityProtocolError(
                    f"{variant_id} response artifact does not verify"
                ) from error
            if response.work_item_id == work_item_id:
                response_artifacts.append(response)
                evidence.add(response.artifact_sha256)
            else:
                _v2_issue(issues, "identifier_freshness", "unplanned_response_artifact")
    response_by_condition = {response.condition: response for response in response_artifacts}
    if (
        len(response_artifacts) != 2
        or set(response_by_condition) != {"epicure_off", "epicure_on"}
        or any(
            response.source_artifact_sha256 != source_digest
            or response.task_id != planned["task_id"]
            or response.task_family != planned["task_family"]
            or response.model_id != planned["model_id"]
            or response.provider_tag != planned["provider_endpoint"]
            for response in response_artifacts
        )
    ):
        _v2_issue(issues, "response_integrity", "response_artifact_link_mismatch")
        _v2_issue(issues, "complete_arms", "two_linked_response_artifacts_required")

    budget = source.get("budget")
    source_cost_micros = budget.get("actual_cost_micros") if isinstance(budget, Mapping) else None
    if not (
        isinstance(source_cost_micros, int)
        and not isinstance(source_cost_micros, bool)
        and source_cost_micros >= 0
        and budget.get("all_generation_costs_reconciled") is True
        and source_cost_micros == result_cost_micros
        and source_cost_micros
        == sum(
            int(event["metadata"]["cost_micros"])
            for event in accounting_events
            if isinstance(event.get("metadata"), Mapping)
            and isinstance(event["metadata"].get("cost_micros"), int)
        )
    ):
        _v2_issue(issues, "cost_reconciliation", "source_cost_does_not_reconcile")
    actual_cost = (
        Decimal(source_cost_micros) / Decimal(1_000_000)
        if isinstance(source_cost_micros, int) and not isinstance(source_cost_micros, bool)
        else Decimal(0)
    )

    finalization = finalizations[0] if len(finalizations) == 1 else None
    reservation = reservations[0] if len(reservations) == 1 else None
    response_digests = sorted(response.artifact_sha256 for response in response_artifacts)
    if reservation is not None and not (
        reservation.get("manifest_sha256") == binding["manifest_sha256"]
        and reservation.get("model_id") == planned["model_id"]
        and reservation.get("canonical_model_slug") == planned["canonical_model_slug"]
        and reservation.get("provider_tag") == planned["provider_endpoint"]
        and reservation.get("task_id") == planned["task_id"]
        and reservation.get("task_family") == planned["task_family"]
        and reservation.get("prompt_sha256") == planned["prompt_sha256"]
        and reservation.get("endpoint_execution_sha256")
        == planned["endpoint_execution_sha256"]
        and reservation.get("execution_policy_sha256")
        == planned["execution_policy_sha256"]
        and _decimal(reservation.get("reserved_usd"), field="route reserve")
        == _decimal(planned["worst_case_reserve_usd"], field="planned reserve")
    ):
        _v2_issue(issues, "budget_admission", "reservation_binding_mismatch")
        _v2_issue(issues, "model_provider_identity", "ledger_route_binding_mismatch")
    if finalization is not None and not (
        finalization.get("source_artifact_sha256") == source_digest
        and finalization.get("source_artifact_filename") == source_path.name
        and finalization.get("source_status") == "complete"
        and finalization.get("response_conditions") == ["epicure_off", "epicure_on"]
        and sorted(finalization.get("response_artifact_sha256s") or []) == response_digests
        and finalization.get("all_generation_costs_reconciled") is True
        and finalization.get("provider_cost_exact") is True
        and finalization.get("normalization_issues") == []
        and finalization.get("source_exposure_basis") == "fully_reconciled_actual"
        and _decimal(
            finalization.get("provider_reconciled_actual_cost_usd"),
            field="ledger provider actual",
        )
        == actual_cost
        and _decimal(finalization.get("source_actual_cost_usd"), field="ledger actual")
        == actual_cost
    ):
        _v2_issue(issues, "cost_reconciliation", "ledger_finalization_mismatch")
        _v2_issue(issues, "response_integrity", "ledger_artifact_links_mismatch")

    if summary is not None:
        outcomes = [
            outcome
            for outcome in summary.get("outcomes") or []
            if isinstance(outcome, Mapping) and outcome.get("work_item_id") == work_item_id
        ]
        ledger_descriptor = summary.get("ledger")
        if not (
            summary.get("provider_calls_made") is True
            and summary.get("paid_subprocesses_started") == 1
            and len(outcomes) == 1
            and outcomes[0].get("decision") == "pair_recorded"
            and outcomes[0].get("source_artifact_sha256") == source_digest
            and sorted(outcomes[0].get("response_artifact_sha256s") or [])
            == response_digests
            and outcomes[0].get("subprocess_returncode") == 0
            and isinstance(ledger_descriptor, Mapping)
            and ledger_descriptor.get("entry_count") == len(ledger)
            and ledger_descriptor.get("head_entry_sha256")
            == (ledger[-1]["entry_sha256"] if ledger else None)
        ):
            _v2_issue(issues, "complete_arms", "execute_summary_does_not_confirm_pair")
            _v2_issue(issues, "response_integrity", "execute_summary_links_mismatch")

    retained_exposure = (
        actual_cost
        if finalization is not None and not issues["cost_reconciliation"]
        else sum(
            (_decimal(item.get("reserved_usd"), field="route reserve") for item in reservations),
            Decimal(0),
        )
    )
    usable_pair = all(not values for values in issues.values())
    return {
        "variant_id": variant_id,
        "work_item_id": work_item_id,
        "attempted": True,
        "usable_pair": usable_pair,
        "issues": issues,
        "evidence_sha256": sorted(evidence),
        "source_path": str(source_path),
        "source_artifact_sha256": source_digest,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "summary_artifact_sha256": summary_digest,
        "journal_sha256": journal_digest,
        "ledger_head_sha256": ledger[-1]["entry_sha256"] if ledger else None,
        "response_artifact_sha256s": response_digests,
        "attempt_ids": attempt_ids,
        "generation_ids": generation_ids,
        "missing_generation_id_requests": missing_generation_requests,
        "provider_requests": len(request_events),
        "provider_responses": len(response_events),
        "unreconciled_provider_requests": max(
            len(response_events) - len(accounting_events), 0
        ),
        "non_chat_completion_envelopes": non_chat,
        "unknown_http_200_responses": unknown_http_200,
        "successful_epicure_tool_calls": successful_tool_calls,
        "epicure_off_tool_calls": off_tool_calls,
        "identity_mismatches": identity_mismatches,
        "truncated_or_invalid_arms": invalid_arms,
        "actual_cost_usd": actual_cost,
        "retained_exposure_usd": retained_exposure,
    }


def build_v2_route_validation_audit(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a PASS/FAIL receipt solely from immutable v2 execution records."""

    bindings = _v2_route_runner_contract(plan=plan, runner_assets=runner_assets)
    records = [
        _audit_v2_route_variant(plan=plan, binding=binding) for binding in bindings
    ]
    attempted_records = [record for record in records if record["attempted"]]
    all_attempted = len(attempted_records) == len(records)
    any_failed = any(
        any(record["issues"].values()) for record in attempted_records
    )

    v2_work_ids = sorted(str(record["work_item_id"]) for record in records)
    attempt_ids = [value for record in records for value in record["attempt_ids"]]
    generation_ids = [value for record in records for value in record["generation_ids"]]
    v1 = plan["closed_v1_identifiers"]
    work_overlap = sorted(set(v2_work_ids) & set(v1["work_item_ids"]))
    attempt_overlap = sorted(set(attempt_ids) & set(v1["attempt_ids"]))
    generation_overlap = sorted(set(generation_ids) & set(v1["generation_ids"]))
    duplicate_attempts = sorted(
        value for value, count in Counter(attempt_ids).items() if value and count > 1
    )
    duplicate_generations = sorted(
        value for value, count in Counter(generation_ids).items() if value and count > 1
    )
    missing_generation_ids = sum(
        int(record.get("missing_generation_id_requests") or 0) for record in records
    )
    identifier_fresh = not (
        work_overlap
        or attempt_overlap
        or generation_overlap
        or duplicate_attempts
        or duplicate_generations
        or missing_generation_ids
    )
    if all_attempted and not identifier_fresh:
        any_failed = True

    actual_cost = sum((record["actual_cost_usd"] for record in records), Decimal(0))
    retained_exposure = sum(
        (record["retained_exposure_usd"] for record in records), Decimal(0)
    )
    post_route = _decimal(
        plan["budget"]["post_v1_conservative_exposure_usd"],
        field="post-v1 exposure",
    ) + retained_exposure
    ceiling = _decimal(plan["budget"]["admission_ceiling_usd"], field="ceiling")
    budget_passed = post_route <= ceiling
    if all_attempted and not budget_passed:
        any_failed = True

    predicate_results: list[dict[str, Any]] = []
    base_evidence = {
        str(plan["artifact_sha256"]),
        str(runner_assets["artifact_sha256"]),
    }
    for predicate in plan["acceptance_gate"]["predicates"]:
        predicate_id = str(predicate["predicate_id"])
        failures = [
            f"{record['variant_id']}:{reason}"
            for record in attempted_records
            for reason in record["issues"][predicate_id]
        ]
        if predicate_id == "identifier_freshness" and not identifier_fresh:
            failures.append("cross_variant_or_v1_identifier_overlap")
        if predicate_id == "budget_admission" and not budget_passed:
            failures.append("post_route_exposure_exceeds_admission_ceiling")
        status = (
            "not_evaluated"
            if not all_attempted
            else "failed"
            if failures
            else "passed"
        )
        evidence = set(base_evidence)
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
        if attempted_records
        else "not_executed"
    )
    source_artifacts = [
        {
            "work_item_id": record["work_item_id"],
            "variant_id": record["variant_id"],
            "source_artifact_sha256": record["source_artifact_sha256"],
            "summary_artifact_sha256": record["summary_artifact_sha256"],
            "immutable": True,
        }
        for record in records
        if record["source_artifact_sha256"] and record["summary_artifact_sha256"]
    ]
    counts = {
        "attempted_pairs": len(attempted_records),
        "usable_pairs": sum(bool(record["usable_pair"]) for record in records),
        "intended_arms": 6,
        "usable_arms": 2 * sum(bool(record["usable_pair"]) for record in records),
        "provider_requests": sum(record["provider_requests"] for record in records),
        "provider_responses": sum(record["provider_responses"] for record in records),
        "successful_epicure_tool_calls": sum(
            record["successful_epicure_tool_calls"] for record in records
        ),
        "epicure_off_tool_calls": sum(record["epicure_off_tool_calls"] for record in records),
        "synthetic_arms": 0,
        "identity_mismatches": sum(record["identity_mismatches"] for record in records),
        "unreconciled_provider_requests": sum(
            record["unreconciled_provider_requests"] for record in records
        ),
        "non_chat_completion_envelopes": sum(
            record["non_chat_completion_envelopes"] for record in records
        ),
        "truncated_or_invalid_arms": sum(
            record["truncated_or_invalid_arms"] for record in records
        ),
    }
    public_records = []
    for record in records:
        public_records.append(
            {
                key: (
                    _decimal_text(value)
                    if isinstance(value, Decimal)
                    else value
                )
                for key, value in record.items()
                if key != "issues"
            }
            | {"predicate_failures": record["issues"]}
        )
    return {
        "schema_version": V2_ROUTE_AUDIT_SCHEMA_VERSION,
        "record_role": "source_derived_fail_closed_route_gate_receipt",
        "derivation_policy": (
            "No predicate value is operator supplied; all fields derive from content-addressed "
            "runner assets and immutable summary, source, journal, response, and ledger records."
        ),
        "v2_route_plan_sha256": plan["artifact_sha256"],
        "runner_assets_sha256": runner_assets["artifact_sha256"],
        "route_cell_id": plan["route_validation"]["route_cell_id"],
        "decision": decision,
        "source_artifacts": source_artifacts,
        "variant_audits": public_records,
        "counts": counts,
        "response_envelope_audit": {
            "contract_sha256": plan["safe_response_envelope_contract"]["contract_sha256"],
            "provider_source_sha256": plan["source"]["provider_source_sha256"],
            "all_provider_responses_chat_completions": (
                all_attempted and counts["non_chat_completion_envelopes"] == 0
            ),
            "unknown_http_200_responses": sum(
                record["unknown_http_200_responses"] for record in records
            ),
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
            "v1_closed_identifiers_sha256": v1["inventory_sha256"],
            "v1_closed_work_item_ids": v1["work_item_ids"],
            "v1_closed_attempt_ids": v1["attempt_ids"],
            "v1_closed_generation_ids": v1["generation_ids"],
            "v2_work_item_ids": v2_work_ids,
            "v2_attempt_ids": attempt_ids,
            "v2_generation_ids": generation_ids,
            "work_item_overlap": work_overlap,
            "attempt_id_overlap": attempt_overlap,
            "generation_id_overlap": generation_overlap,
            "duplicate_attempt_ids": duplicate_attempts,
            "duplicate_generation_ids": duplicate_generations,
            "missing_generation_id_requests": missing_generation_ids,
            "all_identifiers_fresh": all_attempted and identifier_fresh,
        },
        "accounting_audit": {
            "identified_generation_cost_usd": _decimal_text(actual_cost),
            "conservative_retained_exposure_usd": _decimal_text(retained_exposure),
            "post_route_conservative_exposure_usd": _decimal_text(post_route),
            "admission_ceiling_usd": _decimal_text(ceiling),
            "generation_accounting_complete": bool(
                all_attempted
                and counts["provider_requests"] == counts["provider_responses"]
                and counts["unreconciled_provider_requests"] == 0
                and missing_generation_ids == 0
            ),
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


def build_smoke_audit(
    *,
    plan: Mapping[str, Any],
    runner_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one immutable paid source per effort variant and block on any failure."""

    if not verify_plan(plan):
        raise SensitivityProtocolError("sensitivity plan does not verify")
    if runner_assets.get("plan_sha256") != plan["artifact_sha256"]:
        raise SensitivityProtocolError("runner assets differ from the sensitivity plan")
    variants = runner_assets.get("variants")
    if not isinstance(variants, list) or len(variants) != len(VARIANTS):
        raise SensitivityProtocolError("runner assets do not contain the three variants")

    records: list[dict[str, Any]] = []
    historical_unclassified_envelopes: list[dict[str, Any]] = []
    retained_exposure = Decimal(0)
    identified_cost = Decimal(0)
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise SensitivityProtocolError("runner variant is invalid")
        variant_id = str(variant.get("variant_id") or "")
        command = variant.get("dry_run_command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise SensitivityProtocolError(f"{variant_id} runner command is invalid")
        source_root = _command_option(command, "--source-directory")
        ledger_path = _command_option(command, "--ledger")
        summary_root = _command_option(command, "--summary-directory")
        source_paths = sorted(source_root.glob("*.json")) if source_root.exists() else []
        if len(source_paths) != 1:
            raise SensitivityProtocolError(
                f"{variant_id} smoke audit requires exactly one paid source artifact"
            )
        try:
            source, source_digest = _verify_live_artifact(source_paths[0])
        except IntegrityError as error:
            raise SensitivityProtocolError(
                f"{variant_id} live source does not verify"
            ) from error
        if source.get("candidate_manifest_sha256") != variant.get("manifest_sha256"):
            raise SensitivityProtocolError(f"{variant_id} source uses a different manifest")
        ledger = load_dataset_ledger(ledger_path)
        work_item_id = str(source.get("dataset_work_item_id") or "")
        reservation = [
            entry
            for entry in ledger
            if entry.get("event_type") == "reservation_created"
            and entry.get("work_item_id") == work_item_id
        ]
        incidents = [
            entry
            for entry in ledger
            if entry.get("event_type") == "execution_incident"
            and entry.get("work_item_id") == work_item_id
        ]
        if len(reservation) != 1 or len(incidents) != 1:
            raise SensitivityProtocolError(
                f"{variant_id} smoke lacks one reservation and one immutable incident"
            )
        reserved = _decimal(reservation[0].get("reserved_usd"), field="smoke reserve")
        retained_exposure += reserved
        budget = source.get("budget")
        if not isinstance(budget, Mapping):
            raise SensitivityProtocolError(f"{variant_id} source has no budget record")
        micros = int(budget.get("actual_cost_micros") or 0)
        if micros < 0:
            raise SensitivityProtocolError("identified provider cost cannot be negative")
        identified_cost += Decimal(micros) / Decimal(1_000_000)
        summaries: list[tuple[dict[str, Any], str, Path]] = []
        for path in sorted(summary_root.glob("*.json")) if summary_root.exists() else []:
            summary, digest = _verified_artifact(path)
            if summary.get("provider_calls_made") is True:
                summaries.append((summary, digest, path))
        if len(summaries) != 1:
            raise SensitivityProtocolError(
                f"{variant_id} smoke audit requires exactly one paid summary"
            )
        summary, summary_digest, summary_path = summaries[0]
        paid_outcomes = [
            outcome
            for outcome in summary.get("outcomes") or []
            if isinstance(outcome, Mapping) and outcome.get("work_item_id") == work_item_id
        ]
        if len(paid_outcomes) != 1:
            raise SensitivityProtocolError(f"{variant_id} paid outcome is missing")
        attempts = source.get("provider_attempt_events") or []
        if not isinstance(attempts, list):
            raise SensitivityProtocolError(f"{variant_id} provider-attempt trace is invalid")
        provider_requests = sum(
            isinstance(event, Mapping) and event.get("event_type") == "request_started"
            for event in attempts
        )
        mcp_call_attempts = sum(
            isinstance(event, Mapping) and event.get("event_type") == "mcp_call_started"
            for event in attempts
        )
        for event in attempts:
            if (
                isinstance(event, Mapping)
                and event.get("event_type") == "response_received"
                and not event.get("generation_id")
            ):
                metadata = event.get("metadata")
                safe_metadata = metadata if isinstance(metadata, Mapping) else {}
                historical_unclassified_envelopes.append(
                    {
                        "variant_id": variant_id,
                        "phase": str(event.get("phase") or "unknown"),
                        "http_status": event.get("http_status"),
                        "payload_sha256": str(event.get("payload_sha256") or ""),
                        "response_model": str(safe_metadata.get("response_model") or ""),
                        "finish_reason": str(safe_metadata.get("finish_reason") or ""),
                        "native_finish_reason": str(
                            safe_metadata.get("native_finish_reason") or ""
                        ),
                    }
                )
        results = source.get("results") or {}
        errors = source.get("errors") or {}
        if not isinstance(results, Mapping) or not isinstance(errors, Mapping):
            raise SensitivityProtocolError(f"{variant_id} source result shape is invalid")
        epicure = source.get("epicure") or {}
        exact_epicure = (
            isinstance(epicure, Mapping)
            and epicure.get("release_id") == "exploratory-unmatched-1790-runtime"
            and epicure.get("bundle_sha256") == plan["epicure"]["bundle_sha256"]
            and epicure.get("application_sha256") == plan["epicure"]["application_sha256"]
            and source.get("epicure_tool_schema_sha256")
            == plan["epicure"]["tool_schema_sha256"]
        )
        if not exact_epicure:
            raise SensitivityProtocolError(f"{variant_id} source Epicure identity differs")
        both_arms_complete = (
            source.get("status") == "complete"
            and {"epicure_off", "epicure_on"}.issubset(results)
            and not errors
            and mcp_call_attempts > 0
        )
        records.append(
            {
                "variant_id": variant_id,
                "source_path": str(source_paths[0]),
                "source_artifact_sha256": source_digest,
                "summary_path": str(summary_path),
                "summary_sha256": summary_digest,
                "dataset_work_item_id": work_item_id,
                "model_id": source.get("requested_model_id"),
                "task_id": source.get("dataset_task_id"),
                "status": source.get("status"),
                "runner_decision": paid_outcomes[0].get("decision"),
                "provider_request_count": provider_requests,
                "identified_generation_cost_usd": _decimal_text(
                    Decimal(micros) / Decimal(1_000_000)
                ),
                "conservative_retained_exposure_usd": _decimal_text(reserved),
                "result_conditions": sorted(str(key) for key in results),
                "errors": {str(key): str(value) for key, value in sorted(errors.items())},
                "runtime_attestation_exact": True,
                "successful_epicure_tool_calls": mcp_call_attempts,
                "both_arms_complete": both_arms_complete,
                "official": False,
                "rank_eligible": False,
            }
        )

    all_passed = all(record["both_arms_complete"] for record in records)
    base_exposure = _decimal(
        plan["budget"]["current_conservative_exposure_usd"],
        field="base conservative exposure",
    )
    fingerprints: list[dict[str, Any]] = []
    for digest in sorted(
        {record["payload_sha256"] for record in historical_unclassified_envelopes}
    ):
        grouped = [
            record
            for record in historical_unclassified_envelopes
            if record["payload_sha256"] == digest
        ]
        fingerprints.append(
            {
                "payload_sha256": digest,
                "count": len(grouped),
                "variants": sorted({record["variant_id"] for record in grouped}),
                "phases": sorted({record["phase"] for record in grouped}),
                "http_statuses": sorted({record["http_status"] for record in grouped}),
                "all_generation_ids_blank": True,
                "all_response_models_blank": all(
                    not record["response_model"] for record in grouped
                ),
                "all_finish_reasons_unknown": all(
                    record["finish_reason"] in {"", "unknown"} for record in grouped
                ),
            }
        )
    return {
        "schema_version": "flavourbench-reasoning-effort-smoke-audit-v1",
        "record_role": "real_paid_smoke_gate_and_conservative_exposure_audit",
        "plan_sha256": plan["artifact_sha256"],
        "runner_assets_sha256": runner_assets["artifact_sha256"],
        "variants": records,
        "counts": {
            "paid_pairs": len(records),
            "usable_pairs": sum(record["both_arms_complete"] for record in records),
            "failed_pairs": sum(not record["both_arms_complete"] for record in records),
            "provider_requests": sum(record["provider_request_count"] for record in records),
            "successful_epicure_tool_calls": sum(
                record["successful_epicure_tool_calls"] for record in records
            ),
            "synthetic_arms": 0,
        },
        "budget": {
            "identified_generation_cost_usd": _decimal_text(identified_cost),
            "conservative_retained_exposure_usd": _decimal_text(retained_exposure),
            "base_audit_conservative_exposure_usd": _decimal_text(base_exposure),
            "post_smoke_conservative_exposure_usd": _decimal_text(
                base_exposure + retained_exposure
            ),
            "admission_ceiling_usd": plan["budget"]["admission_ceiling_usd"],
            "hard_cap_usd": plan["budget"]["hard_cap_usd"],
        },
        "response_envelope_diagnosis": {
            "historical_fingerprints": fingerprints,
            "determination": "indeterminate_historical_non_chat_completion_envelope",
            "cannot_distinguish_from_v1_record": [
                "OpenRouter top-level error object",
                "Cloudflare gateway API envelope",
                "upstream Responses-style schema returned to the chat-completions adapter",
            ],
            "reason": (
                "v1 intentionally retained only a canonical response digest and safe transport "
                "metadata; it did not retain the raw HTTP-200 body"
            ),
            "adapter_observation": (
                "the choice check consumed the decoded upstream object directly, so no local "
                "adapter transformation caused the missing choices; an upstream schema mismatch "
                "remains possible"
            ),
            "v2_guard": (
                "classify and reject non-chat-completion envelopes inside OpenRouterProvider._post "
                "while persisting only classification, code, type, and provider metadata"
            ),
            "v1_replay": "prohibited",
            "required_next_study_id": "frontier-reasoning-effort-sensitivity-v2",
        },
        "decision": (
            "eligible_for_full_development_collection"
            if all_passed
            else "blocked_before_full_study_due_smoke_failures"
        ),
        "claim_boundary": {
            "quality_results_available": False,
            "sensitivity_effect_estimable": False,
            "official": False,
            "rank_eligible": False,
            "synthetic_arms": 0,
        },
    }


def build_v1_closed_identifier_inventory(
    *,
    v1_smoke_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the closed v1 work, attempt, and generation IDs from immutable sources."""

    if not _artifact_sha256_verifies(
        v1_smoke_audit, "flavourbench-reasoning-effort-smoke-audit-v1"
    ):
        raise SensitivityProtocolError("v1 smoke audit content address does not verify")
    audit_digest = str(v1_smoke_audit["artifact_sha256"])
    variants = v1_smoke_audit.get("variants")
    if not isinstance(variants, list) or len(variants) != len(VARIANTS):
        raise SensitivityProtocolError("v1 smoke audit has no complete variant set")
    sources: list[dict[str, Any]] = []
    work_item_ids: set[str] = set()
    attempt_ids: set[str] = set()
    generation_ids: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise SensitivityProtocolError("v1 smoke variant is invalid")
        path = Path(str(variant.get("source_path") or ""))
        try:
            source, source_digest = _verify_live_artifact(path)
        except IntegrityError as error:
            raise SensitivityProtocolError("v1 closed source does not verify") from error
        if (
            source_digest != variant.get("source_artifact_sha256")
            or source.get("dataset_work_item_id") != variant.get("dataset_work_item_id")
        ):
            raise SensitivityProtocolError("v1 closed source differs from the smoke audit")
        events = source.get("provider_attempt_events")
        if not isinstance(events, list) or not events:
            raise SensitivityProtocolError("v1 closed source has no provider-attempt events")
        source_attempts = {
            str(event.get("attempt_id") or "")
            for event in events
            if isinstance(event, Mapping) and event.get("attempt_id")
        }
        source_generations = {
            str(event.get("generation_id") or "")
            for event in events
            if isinstance(event, Mapping) and event.get("generation_id")
        }
        work_item_id = str(source.get("dataset_work_item_id") or "")
        if len(work_item_id) != 64 or not source_attempts:
            raise SensitivityProtocolError("v1 closed source identifiers are incomplete")
        work_item_ids.add(work_item_id)
        attempt_ids.update(source_attempts)
        generation_ids.update(source_generations)
        sources.append(
            {
                "variant_id": variant.get("variant_id"),
                "source_artifact_sha256": source_digest,
                "work_item_id": work_item_id,
                "attempt_ids": sorted(source_attempts),
                "generation_ids": sorted(source_generations),
            }
        )
    if len(work_item_ids) != 3 or not generation_ids:
        raise SensitivityProtocolError("v1 closed identifier inventory is incomplete")
    return {
        "schema_version": V1_CLOSED_IDENTIFIERS_SCHEMA_VERSION,
        "record_role": "immutable_identifier_non_replay_boundary_for_v2",
        "v1_smoke_audit_sha256": audit_digest,
        "sources": sorted(sources, key=lambda item: str(item["variant_id"])),
        "work_item_ids": sorted(work_item_ids),
        "attempt_ids": sorted(attempt_ids),
        "generation_ids": sorted(generation_ids),
        "counts": {
            "sources": len(sources),
            "work_item_ids": len(work_item_ids),
            "attempt_ids": len(attempt_ids),
            "generation_ids": len(generation_ids),
        },
        "provider_calls_made": False,
        "epicure_calls_made": False,
    }


def build_v2_route_validation_plan(
    *,
    v1_plan: Mapping[str, Any],
    v1_smoke_audit: Mapping[str, Any],
    v1_closed_identifiers: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    lineage_inventory: Mapping[str, Any],
    lineage_correction: Mapping[str, Any],
    provider_source_path: Path,
) -> dict[str, Any]:
    """Freeze a fresh no-call, one-cell v2 route gate after the failed v1 smoke."""

    if not verify_plan(v1_plan):
        raise SensitivityProtocolError("v1 sensitivity plan does not verify")
    audit_digest = str(v1_smoke_audit.get("artifact_sha256") or "")
    audit_unhashed = {
        key: value for key, value in v1_smoke_audit.items() if key != "artifact_sha256"
    }
    if (
        len(audit_digest) != 64
        or _sha256(audit_unhashed) != audit_digest
        or v1_smoke_audit.get("plan_sha256") != v1_plan["artifact_sha256"]
        or v1_smoke_audit.get("decision")
        != "blocked_before_full_study_due_smoke_failures"
        or (v1_smoke_audit.get("counts") or {}).get("usable_pairs") != 0
    ):
        raise SensitivityProtocolError("v1 smoke audit is not the verified failed gate")
    if not _artifact_sha256_verifies(
        v1_closed_identifiers, V1_CLOSED_IDENTIFIERS_SCHEMA_VERSION
    ) or v1_closed_identifiers.get("v1_smoke_audit_sha256") != audit_digest:
        raise SensitivityProtocolError("v1 closed-identifier inventory does not verify")
    if not verify_manifest_content_address(base_manifest):
        raise SensitivityProtocolError("base routed manifest does not verify")
    if (base_manifest.get("content_address") or {}).get("digest") != v1_plan[
        "base_evidence"
    ]["manifest_sha256"]:
        raise SensitivityProtocolError("base routed manifest differs from the v1 plan")

    if not verify_inventory(lineage_inventory):
        raise SensitivityProtocolError("corrected Epicure lineage inventory does not verify")
    lineage_sha256 = str(lineage_inventory.get("artifact_sha256") or "")
    old_lineage_sha256 = str(v1_plan["epicure"].get("lineage_inventory_sha256") or "")
    if lineage_sha256 == old_lineage_sha256:
        raise SensitivityProtocolError(
            "v2 must bind the corrected lineage inventory, not the parser-defective v1 record"
        )
    if (
        not verify_correction(lineage_correction)
        or lineage_correction.get("authoritative_inventory_sha256") != lineage_sha256
        or lineage_correction.get("parser_defective_inventory_sha256")
        != old_lineage_sha256
        or (lineage_correction.get("correction") or {}).get(
            "other_inventory_fields_changed"
        )
        != 0
    ):
        raise SensitivityProtocolError("Epicure lineage correction chain does not verify")
    lineage_correction_sha256 = str(lineage_correction["artifact_sha256"])
    lineage_identity = {
        "runtime_id": lineage_inventory.get("runtime_id"),
        "bundle_sha256": (lineage_inventory.get("bundle") or {}).get("sha256"),
        "application_sha256": (lineage_inventory.get("application") or {}).get("sha256"),
        "tool_schema_sha256": (lineage_inventory.get("tool_contract") or {}).get(
            "semantic_sha256"
        ),
    }
    expected_lineage_identity = {
        field: v1_plan["epicure"].get(field) for field in lineage_identity
    }
    if lineage_identity != expected_lineage_identity:
        raise SensitivityProtocolError(
            "corrected Epicure lineage inventory changes the v1 runtime identity"
        )
    if (lineage_inventory.get("runtime_attestation") or {}).get(
        "matches_recovered_checkout"
    ) is not True:
        raise SensitivityProtocolError(
            "corrected Epicure lineage inventory lacks a matching runtime attestation"
        )

    provider_source_sha256 = _file_sha256(provider_source_path)
    try:
        provider_source = provider_source_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise SensitivityProtocolError("provider source is not valid UTF-8") from error
    required_classifier_markers = (
        "openrouter_error_envelope",
        "gateway_api_envelope",
        "responses_api_schema_mismatch",
        "unknown_non_chat_completion_envelope",
        "response_envelope",
    )
    if not all(marker in provider_source for marker in required_classifier_markers):
        raise SensitivityProtocolError("provider source lacks the fail-closed envelope classifier")
    classifier_contract = {
        "schema_version": "flavourbench-safe-provider-envelope-classifier-v1",
        "accepted_classification": "chat_completions",
        "rejected_classifications": [
            "openrouter_error_envelope",
            "gateway_api_envelope",
            "responses_api_schema_mismatch",
            "unknown_non_chat_completion_envelope",
        ],
        "persisted_error_metadata": ["classification", "code", "type", "provider"],
        "prohibited_persistence": [
            "raw response body",
            "provider error message",
            "provider metadata raw field",
            "request prompt",
            "authorization material",
        ],
        "http_200_non_chat_action": "fail_closed_no_automatic_retry",
    }
    classifier_contract_sha256 = _sha256(classifier_contract)

    model_id = "openai/gpt-5.6-sol-pro"
    task_id = "fb-s0-substitution-003"
    model_plan = next(
        (
            model
            for model in v1_plan["model_design"]["models"]
            if model.get("model_id") == model_id
        ),
        None,
    )
    anchor = next(
        (
            task
            for task in v1_plan["task_design"]["anchors"]
            if task.get("task_id") == task_id
        ),
        None,
    )
    base_entry = next(
        (
            entry
            for entry in base_manifest.get("models") or []
            if isinstance(entry, Mapping)
            and (entry.get("model") or {}).get("id") == model_id
        ),
        None,
    )
    if not all(isinstance(value, Mapping) for value in (model_plan, anchor, base_entry)):
        raise SensitivityProtocolError("v2 route-validation cell is absent from the v1 inputs")
    assert isinstance(model_plan, Mapping)
    assert isinstance(anchor, Mapping)
    assert isinstance(base_entry, Mapping)
    endpoint = base_entry.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise SensitivityProtocolError("v2 route-validation endpoint is absent")
    endpoint_execution_sha256 = endpoint_execution_contract_sha256(dict(endpoint))

    audit_variants = v1_smoke_audit.get("variants")
    if not isinstance(audit_variants, list) or len(audit_variants) != len(VARIANTS):
        raise SensitivityProtocolError("v1 smoke audit lacks the three variant reserves")
    reserve_by_variant = {
        str(record.get("variant_id") or ""): _decimal(
            record.get("conservative_retained_exposure_usd"),
            field="v1 route-cell reserve",
        )
        for record in audit_variants
        if isinstance(record, Mapping)
    }
    if set(reserve_by_variant) != {variant["variant_id"] for variant in VARIANTS}:
        raise SensitivityProtocolError("v1 route-cell reserve map is incomplete")

    v1_planned_work_item_ids = {
        str(item.get("work_item_id") or "")
        for item in v1_plan["execution"]["work_items"]
        if item.get("model_id") == model_id and item.get("task_id") == task_id
    }
    v1_executed_work_item_ids = set(v1_closed_identifiers.get("work_item_ids") or [])
    if len(v1_planned_work_item_ids) != 3 or len(v1_executed_work_item_ids) != 3:
        raise SensitivityProtocolError("v1 planned or executed route-cell IDs are incomplete")
    v1_work_item_ids = v1_planned_work_item_ids | v1_executed_work_item_ids
    cell_core = {
        "schema_version": "flavourbench-reasoning-effort-v2-route-cell-v1",
        "study_id": "frontier-reasoning-effort-sensitivity-v2",
        "model_id": model_id,
        "canonical_model_slug": model_plan["canonical_model_slug"],
        "provider_endpoint": model_plan["provider_endpoint"],
        "endpoint_document_sha256": model_plan["endpoint_document_sha256"],
        "endpoint_execution_sha256": endpoint_execution_sha256,
        "task_id": task_id,
        "task_family": anchor["family"],
        "prompt_sha256": anchor["prompt_sha256"],
        "classifier_contract_sha256": classifier_contract_sha256,
        "provider_source_sha256": provider_source_sha256,
        "lineage_inventory_sha256": lineage_sha256,
        "v1_closed_identifiers_sha256": v1_closed_identifiers["artifact_sha256"],
        "lineage_correction_sha256": lineage_correction_sha256,
    }
    route_cell_id = _sha256(cell_core)
    work_items: list[dict[str, Any]] = []
    for variant in VARIANTS:
        policy = _execution_policy_from_base(
            base_manifest,
            intermediate_effort=variant["intermediate_reasoning_effort"],
            final_effort=variant["final_reasoning_effort"],
        )
        core = {
            "schema_version": "flavourbench-reasoning-effort-v2-route-work-item-v1",
            "study_id": "frontier-reasoning-effort-sensitivity-v2",
            "route_cell_id": route_cell_id,
            "model_id": model_id,
            "canonical_model_slug": model_plan["canonical_model_slug"],
            "provider_endpoint": model_plan["provider_endpoint"],
            "endpoint_document_sha256": model_plan["endpoint_document_sha256"],
            "endpoint_execution_sha256": endpoint_execution_sha256,
            "task_id": task_id,
            "task_family": anchor["family"],
            "prompt_sha256": anchor["prompt_sha256"],
            "variant_id": variant["variant_id"],
            "intermediate_reasoning_effort": variant[
                "intermediate_reasoning_effort"
            ],
            "final_reasoning_effort": variant["final_reasoning_effort"],
            "execution_policy_sha256": policy.sha256,
            "required_tool_contract_sha256": required_tool_contract(policy)[
                "content_address"
            ]["digest"],
            "classifier_contract_sha256": classifier_contract_sha256,
            "provider_source_sha256": provider_source_sha256,
            "lineage_inventory_sha256": lineage_sha256,
            "lineage_correction_sha256": lineage_correction_sha256,
            "v1_closed_identifiers_sha256": v1_closed_identifiers["artifact_sha256"],
            "actual_provider_name": model_plan["actual_provider_name"],
            "conditions": ["epicure_off", "epicure_on"],
            "diagnostic_only": True,
            "official": False,
            "rank_eligible": False,
        }
        work_item_id = _sha256(core)
        if work_item_id in v1_work_item_ids:
            raise SensitivityProtocolError("v2 work-item identity reuses a v1 ID")
        work_items.append(
            {
                "work_item_id": work_item_id,
                **core,
                "worst_case_reserve_usd": _decimal_text(
                    reserve_by_variant[variant["variant_id"]]
                ),
            }
        )

    route_reserve = sum(reserve_by_variant.values(), Decimal(0))
    post_v1_exposure = _decimal(
        v1_smoke_audit["budget"]["post_smoke_conservative_exposure_usd"],
        field="post-v1 conservative exposure",
    )
    projected = post_v1_exposure + route_reserve
    admission_ceiling = _decimal(
        v1_plan["budget"]["admission_ceiling_usd"], field="admission ceiling"
    )
    collection_blockers: list[dict[str, str]] = []
    if projected > admission_ceiling:
        collection_blockers.append(
            {
                "gate": "shared_budget_admission",
                "reason": (
                    f"v2 route gate projects ${_decimal_text(projected)} above "
                    f"${_decimal_text(admission_ceiling)}"
                ),
            }
        )
    if (v1_plan.get("preflight") or {}).get("collection_blockers"):
        collection_blockers.append(
            {
                "gate": "inherited_collection_blocker",
                "reason": "the verified v1 plan no longer has a clear collection gate",
            }
        )

    return {
        "schema_version": V2_ROUTE_PLAN_SCHEMA_VERSION,
        "study_id": "frontier-reasoning-effort-sensitivity-v2",
        "plan_role": "no_call_one_model_task_cell_route_validation_before_full_study",
        "supersedes_failed_smoke_audit_sha256": audit_digest,
        "does_not_supersede_raw_v1_records": True,
        "v1_work_item_ids_never_replayed": sorted(v1_work_item_ids),
        "closed_v1_identifiers": {
            "inventory_sha256": v1_closed_identifiers["artifact_sha256"],
            "work_item_ids": sorted(v1_work_item_ids),
            "planned_work_item_ids": sorted(v1_planned_work_item_ids),
            "executed_work_item_ids": sorted(v1_executed_work_item_ids),
            "attempt_ids": v1_closed_identifiers["attempt_ids"],
            "generation_ids": v1_closed_identifiers["generation_ids"],
        },
        "source": {
            "v1_plan_sha256": v1_plan["artifact_sha256"],
            "v1_smoke_audit_sha256": audit_digest,
            "base_manifest_sha256": v1_plan["base_evidence"]["manifest_sha256"],
            "corrected_lineage_inventory_sha256": lineage_sha256,
            "lineage_correction_sha256": lineage_correction_sha256,
            "parser_defective_v1_lineage_inventory_sha256": old_lineage_sha256,
            "v1_closed_identifiers_sha256": v1_closed_identifiers["artifact_sha256"],
            "provider_source_path": str(provider_source_path),
            "provider_source_sha256": provider_source_sha256,
        },
        "epicure": {
            **lineage_identity,
            "lineage_inventory_sha256": lineage_sha256,
            "lineage_correction_sha256": lineage_correction_sha256,
            "runtime_attestation_matches_recovered_checkout": True,
            "rank_eligible": lineage_inventory.get("rank_eligible") is True,
            "redistributable": lineage_inventory.get("redistributable") is True,
        },
        "safe_response_envelope_contract": {
            **classifier_contract,
            "contract_sha256": classifier_contract_sha256,
        },
        "route_validation": {
            "cell_count": 1,
            "route_cell_id": route_cell_id,
            "model_id": model_id,
            "task_id": task_id,
            "task_family": anchor["family"],
            "effort_variants": 3,
            "matched_pairs": 3,
            "response_arms": 6,
            "synthetic_arms": 0,
            "work_items": work_items,
            "execution_order": [
                "explicit_low",
                "provider_default",
                "explicit_high",
            ],
            "scheduling": "strictly_sequential_one_pair_at_a_time",
            "diagnostic_outputs_enter_quality_fit": False,
        },
        "acceptance_gate": {
            "decision_rule": "all predicates must pass; one failure blocks the 72-arm study",
            "predicates": [
                {"predicate_id": predicate_id, "requirement": requirement}
                for predicate_id, requirement in V2_ROUTE_PREDICATES
            ],
            "minimum_usable_pairs": 3,
            "minimum_usable_arms": 6,
            "minimum_successful_epicure_tool_calls": 3,
            "permitted_non_chat_completion_envelopes": 0,
            "permitted_identity_mismatches": 0,
            "permitted_unreconciled_requests": 0,
            "on_failure": (
                "retain the source and full reserve, close the v2 IDs, and freeze v3 before "
                "any further paid retry"
            ),
        },
        "full_sensitivity": {
            "status": "blocked_pending_v2_route_validation",
            "matched_pairs": 36,
            "response_arms": 72,
            "models": 3,
            "tasks": 4,
            "effort_variants": 3,
            "route_validation_outputs_reused": False,
        },
        "budget": {
            "currency": "USD",
            "post_v1_conservative_exposure_usd": _decimal_text(post_v1_exposure),
            "v2_route_validation_worst_case_usd": _decimal_text(route_reserve),
            "projected_post_route_exposure_usd": _decimal_text(projected),
            "admission_ceiling_usd": _decimal_text(admission_ceiling),
            "admitted": not collection_blockers,
        },
        "preflight": {
            "decision": (
                "blocked_before_provider_call"
                if collection_blockers
                else "ready_to_materialize_v2_route_validation_only"
            ),
            "collection_blockers": collection_blockers,
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


def _write(output_dir: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}-{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise SensitivityProtocolError("content-addressed sensitivity output conflict")
        return path
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as file:
        temporary = Path(file.name)
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Freeze a no-call sensitivity plan")
    plan.add_argument("--base-manifest", type=Path, required=True)
    plan.add_argument("--lineage-inventory", type=Path, required=True)
    plan.add_argument("--coverage-schedule", type=Path, required=True)
    plan.add_argument("--task-validity", type=Path, required=True)
    plan.add_argument("--budget-audit", type=Path, required=True)
    plan.add_argument("--model", action="append", default=[])
    plan.add_argument("--output-dir", type=Path, required=True)
    preflight = subparsers.add_parser("preflight", help="Record a no-call gate decision")
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    materialize = subparsers.add_parser(
        "materialize",
        help="Write exact manifests and commands for the existing append-only runner",
    )
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--base-manifest", type=Path, required=True)
    materialize.add_argument("--task-validity", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize_v2 = subparsers.add_parser(
        "materialize-v2-route",
        help="Write exact dry-run and live commands for the three fresh v2 route pairs",
    )
    materialize_v2.add_argument("--plan", type=Path, required=True)
    materialize_v2.add_argument("--v1-runner-assets", type=Path, required=True)
    materialize_v2.add_argument("--output-dir", type=Path, required=True)
    smoke_audit = subparsers.add_parser(
        "audit-smokes",
        help="Audit one paid pair per variant before any full sensitivity collection",
    )
    smoke_audit.add_argument("--plan", type=Path, required=True)
    smoke_audit.add_argument("--runner-assets", type=Path, required=True)
    smoke_audit.add_argument("--output-dir", type=Path, required=True)
    closed_ids = subparsers.add_parser(
        "freeze-v1-closed-identifiers",
        help="Freeze the non-replay work, attempt, and generation identifiers from v1",
    )
    closed_ids.add_argument("--v1-smoke-audit", type=Path, required=True)
    closed_ids.add_argument("--output-dir", type=Path, required=True)
    v2_route = subparsers.add_parser(
        "freeze-v2-route-validation",
        help="Freeze fresh IDs for one no-call v2 model-task route-validation cell",
    )
    v2_route.add_argument("--v1-plan", type=Path, required=True)
    v2_route.add_argument("--v1-smoke-audit", type=Path, required=True)
    v2_route.add_argument("--v1-closed-identifiers", type=Path, required=True)
    v2_route.add_argument("--base-manifest", type=Path, required=True)
    v2_route.add_argument("--lineage-inventory", type=Path, required=True)
    v2_route.add_argument("--lineage-correction", type=Path, required=True)
    v2_route.add_argument("--provider-source", type=Path, required=True)
    v2_route.add_argument("--output-dir", type=Path, required=True)
    v2_audit = subparsers.add_parser(
        "audit-v2-route",
        help="Derive a fail-closed v2 PASS/FAIL receipt from immutable runner records",
    )
    v2_audit.add_argument("--plan", type=Path, required=True)
    v2_audit.add_argument("--runner-assets", type=Path, required=True)
    v2_audit.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        payload = build_plan(
            base_manifest_path=arguments.base_manifest,
            lineage_inventory_path=arguments.lineage_inventory,
            coverage_schedule_path=arguments.coverage_schedule,
            task_validity_path=arguments.task_validity,
            budget_audit_path=arguments.budget_audit,
            model_ids=tuple(arguments.model) or DEFAULT_MODEL_IDS,
        )
        path = _write(arguments.output_dir, "reasoning-effort-sensitivity-plan", payload)
        written = json.loads(path.read_text(encoding="utf-8"))
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "decision": written["preflight"]["decision"],
            "models": written["model_design"]["model_count"],
            "tasks": len(written["task_design"]["anchors"]),
            "pairs": written["execution"]["pairs"],
            "response_arms": written["execution"]["response_arms"],
            "study_worst_case_usd": written["budget"]["study_worst_case_usd"],
            "provider_calls_made": False,
        }
    elif arguments.command == "preflight":
        source = _regular_json(arguments.plan)
        if not verify_plan(source):
            raise SensitivityProtocolError("sensitivity plan does not verify")
        payload = build_preflight_receipt(source)
        path = _write(
            arguments.output_dir,
            "reasoning-effort-sensitivity-preflight",
            payload,
        )
        written = json.loads(path.read_text(encoding="utf-8"))
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "decision": written["decision"],
            "collection_blocker_count": len(written["collection_blockers"]),
            "officialization_blocker_count": len(written["officialization_blockers"]),
            "provider_calls_made": False,
            "cost_usd": "0",
        }
    elif arguments.command == "materialize":
        source = _regular_json(arguments.plan)
        payload = materialize_runner_assets(
            plan=source,
            base_manifest_path=arguments.base_manifest,
            source_task_validity_path=arguments.task_validity,
            output_dir=arguments.output_dir,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-runner-assets",
            payload,
        )
        written = _regular_json(path)
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "task_dossier_sha256": written["task_dossier_sha256"],
            "variant_manifest_sha256s": {
                variant["variant_id"]: variant["manifest_sha256"]
                for variant in written["variants"]
            },
            "provider_calls_made": False,
        }
    elif arguments.command == "materialize-v2-route":
        source = _regular_json(arguments.plan)
        v1_runner_assets, _ = _verified_artifact(arguments.v1_runner_assets)
        payload = materialize_v2_route_assets(
            plan=source,
            plan_path=arguments.plan,
            v1_runner_assets=v1_runner_assets,
            output_dir=arguments.output_dir,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-v2-route-runner-assets",
            payload,
        )
        written = _regular_json(path)
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "fresh_work_item_ids": [
                item["fresh_work_item_id"] for item in written["variants"]
            ],
            "provider_calls_made": False,
            "epicure_calls_made": False,
        }
    elif arguments.command == "audit-smokes":
        plan_document = _regular_json(arguments.plan)
        assets_document, _ = _verified_artifact(arguments.runner_assets)
        payload = build_smoke_audit(
            plan=plan_document,
            runner_assets=assets_document,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-smoke-audit",
            payload,
        )
        written = _regular_json(path)
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "decision": written["decision"],
            "paid_pairs": written["counts"]["paid_pairs"],
            "usable_pairs": written["counts"]["usable_pairs"],
            "identified_generation_cost_usd": written["budget"][
                "identified_generation_cost_usd"
            ],
            "conservative_retained_exposure_usd": written["budget"][
                "conservative_retained_exposure_usd"
            ],
        }
    elif arguments.command == "freeze-v1-closed-identifiers":
        v1_smoke_audit = _regular_json(arguments.v1_smoke_audit)
        payload = build_v1_closed_identifier_inventory(
            v1_smoke_audit=v1_smoke_audit,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-v1-closed-identifiers",
            payload,
        )
        written = _regular_json(path)
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "counts": written["counts"],
            "provider_calls_made": False,
            "epicure_calls_made": False,
        }
    elif arguments.command == "freeze-v2-route-validation":
        v1_plan = _regular_json(arguments.v1_plan)
        v1_smoke_audit = _regular_json(arguments.v1_smoke_audit)
        v1_closed_identifiers = _regular_json(arguments.v1_closed_identifiers)
        base_manifest = _regular_json(arguments.base_manifest)
        lineage_inventory = _regular_json(arguments.lineage_inventory)
        lineage_correction = _regular_json(arguments.lineage_correction)
        payload = build_v2_route_validation_plan(
            v1_plan=v1_plan,
            v1_smoke_audit=v1_smoke_audit,
            v1_closed_identifiers=v1_closed_identifiers,
            base_manifest=base_manifest,
            lineage_inventory=lineage_inventory,
            lineage_correction=lineage_correction,
            provider_source_path=arguments.provider_source,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-v2-route-validation-plan",
            payload,
        )
        written = _regular_json(path)
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "decision": written["preflight"]["decision"],
            "route_cell_id": written["route_validation"]["route_cell_id"],
            "fresh_work_item_ids": [
                item["work_item_id"]
                for item in written["route_validation"]["work_items"]
            ],
            "response_arms": written["route_validation"]["response_arms"],
            "provider_calls_made": False,
        }
    else:
        plan_document = _regular_json(arguments.plan)
        assets_document, _ = _verified_artifact(arguments.runner_assets)
        payload = build_v2_route_validation_audit(
            plan=plan_document,
            runner_assets=assets_document,
        )
        path = _write(
            arguments.output_dir,
            "reasoning-effort-v2-route-validation-audit",
            payload,
        )
        written = _regular_json(path)
        pass_verifies = verify_v2_route_validation_pass_audit(
            written,
            plan_document,
        )
        if written["decision"] == "passed_all_predicates" and not pass_verifies:
            raise SensitivityProtocolError("derived PASS receipt fails its strict verifier")
        result = {
            "output": str(path.resolve()),
            "artifact_sha256": written["artifact_sha256"],
            "decision": written["decision"],
            "strict_pass_verifies": pass_verifies,
            "attempted_pairs": written["counts"]["attempted_pairs"],
            "usable_pairs": written["counts"]["usable_pairs"],
            "identified_generation_cost_usd": written["accounting_audit"][
                "identified_generation_cost_usd"
            ],
            "full_sensitivity_authorized": written["full_sensitivity_admission"][
                "authorized"
            ],
            "provider_calls_made_by_builder": False,
            "epicure_calls_made_by_builder": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
