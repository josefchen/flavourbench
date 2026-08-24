"""Run one immutable Epicure off/on pair through the direct Kimi endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .budget_policy import provider_account_hard_cap_micros, provider_account_scope_sha256
from .config import get_settings
from .execution_policy import (
    GOVERNED_EPICURE_PROTOCOLS,
    MATCHED_EVIDENCE_PROTOCOLS,
    ExecutionPolicy,
    assert_legacy_paid_cli_allowed,
)
from .frontier_contract_runner import load_candidate_manifest, select_candidates
from .live_smoke import (
    CONFIRMATION,
    _epicure_attestation,
    _model_identity_matches,
    _provider_identity_matches,
    _result_payload,
    _sha256,
    _worst_case_cost_usd,
    build_live_protocol_bundle,
    endpoint_execution_contract_sha256,
    frozen_generation_contract,
    verify_expected_epicure_attestation,
)
from .provider import (
    GenerationResult,
    GenerationSpec,
    ProviderAttemptEvent,
    ToolTrace,
    response_schema_sha256,
    system_prompt_sha256,
)
from .run_journal import RunJournal
from .service_kimi import KimiDirectProvider
from .tool_contract import required_tool_contract


def _credential_free_binding(
    *,
    manifest_sha256: str,
    backend_contract_sha256: str,
    scope_sha256: str,
    kind: str,
) -> str:
    return _sha256(
        {
            "schema_version": "flavourbench-provider-binding-v1",
            "kind": kind,
            "manifest_sha256": manifest_sha256,
            "backend_contract_sha256": backend_contract_sha256,
            "provider_account_scope_sha256": scope_sha256,
        }
    )


def _rate_card(endpoint: dict[str, Any]) -> dict[str, Any]:
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, dict):
        raise RuntimeError("direct provider route has no frozen rate card")
    return {
        "prompt_price_per_token": str(pricing.get("prompt") or "0"),
        "completion_price_per_token": str(pricing.get("completion") or "0"),
        "internal_reasoning_price_per_token": str(pricing.get("internal_reasoning") or "0"),
        "request_price": str(pricing.get("request") or "0"),
        "source": pricing.get("source"),
        "status": pricing.get("status"),
    }


KIMI_ACCOUNTING_BASIS = "frozen_rate_card_times_kimi_returned_usage"
DIRECT_PREFLIGHT_SCHEMA_VERSION = "flavourbench-direct-provider-preflight-v1"


def _rate_card_result_is_accounted(
    result: dict[str, Any],
    accounting_basis: str = KIMI_ACCOUNTING_BASIS,
    billing_reconciliation_status: str = "provider_charge_unavailable",
) -> bool:
    metadata = result.get("generation_metadata")
    generation_ids = result.get("generation_ids")
    if (
        result.get("cost_reconciled") is not False
        or result.get("cost_accounting_basis") != accounting_basis
        or result.get("billing_reconciliation_status") != billing_reconciliation_status
        or not isinstance(metadata, list)
        or not isinstance(generation_ids, list)
        or not generation_ids
    ):
        return False
    seen: set[str] = set()
    cost_micros = 0
    for item in metadata:
        if (
            not isinstance(item, dict)
            or item.get("reconciled") is not False
            or item.get("accounting_basis") != accounting_basis
            or item.get("billing_reconciliation_status") != billing_reconciliation_status
        ):
            return False
        generation_id = str(item.get("generation_id") or "")
        item_cost = item.get("cost_micros")
        if (
            not generation_id
            or generation_id in seen
            or not isinstance(item_cost, int)
            or isinstance(item_cost, bool)
            or item_cost < 0
        ):
            return False
        seen.add(generation_id)
        cost_micros += item_cost
    return set(map(str, generation_ids)) == seen and result.get("cost_micros") == cost_micros


async def _run_direct_pair(
    args: argparse.Namespace,
    *,
    execution_backend: str,
    provider_factory: Any,
    credential_attribute: str,
    accounting_basis: str,
    provider_label: str,
    allow_zero_cap: bool = False,
    mutable_alias_accounting_basis: str | None = None,
    mutable_alias_billing_status: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    provenance_url = os.environ.get("FLAVOURBENCH_EPICURE_PROVENANCE_URL")
    if not provenance_url:
        provenance_url = settings.mcp_url.removesuffix("/mcp").rstrip("/") + "/provenance"
    if settings.execution_mode != "live" or not settings.live_authorized:
        raise RuntimeError(f"direct {provider_label} pair requires live authorization")
    if not getattr(settings, credential_attribute, ""):
        raise RuntimeError(f"direct {provider_label} credential is not configured")
    if args.confirm != CONFIRMATION:
        raise RuntimeError(f"pass --confirm {CONFIRMATION} to acknowledge real external calls")
    if args.cap_usd < 0:
        raise RuntimeError(f"direct {provider_label} pair cap cannot be negative")
    if args.cap_usd == 0 and not allow_zero_cap:
        raise RuntimeError(f"direct {provider_label} pair requires a positive bounded cap")
    if allow_zero_cap and execution_backend != "cohere_direct":
        raise RuntimeError("zero-cap admission is restricted to the direct Cohere backend")
    selected_conditions = tuple(getattr(args, "condition", None) or ("epicure_off", "epicure_on"))
    if len(set(selected_conditions)) != len(selected_conditions):
        raise RuntimeError("each selected Epicure condition may appear only once")
    manifest = load_candidate_manifest(
        args.route_manifest,
        expected_digest=args.candidate_manifest_sha256,
    )
    matches = [
        candidate
        for candidate in select_candidates(manifest, (args.model_id,))
        if candidate.provider_tag == args.provider_slug
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"direct {provider_label} route is absent or ambiguous in the frozen manifest"
        )
    candidate = matches[0]
    if candidate.execution_backend != execution_backend:
        raise RuntimeError(f"direct {provider_label} runner received the wrong route")
    if candidate.canonical_model_slug != args.expected_canonical_model_slug:
        raise RuntimeError(f"direct {provider_label} identity differs from the frozen work item")
    if candidate.endpoint_execution_sha256 != args.expected_endpoint_execution_sha256:
        raise RuntimeError(f"direct {provider_label} endpoint differs from the work item")
    if _sha256(candidate.backend_contract) != candidate.backend_contract_sha256:
        raise RuntimeError(f"direct {provider_label} backend contract hash does not verify")
    identity_kind = str(candidate.backend_contract.get("identity_kind") or "")
    mutable_alias = identity_kind == "mutable_alias"
    mutable_alias_opt_in = bool(getattr(args, "allow_mutable_alias_exploratory", False))
    if mutable_alias:
        if (
            execution_backend != "qwencloud_direct"
            or not mutable_alias_opt_in
            or mutable_alias_accounting_basis is None
            or mutable_alias_billing_status is None
            or candidate.backend_contract.get("official") is not False
            or candidate.backend_contract.get("season_eligible") is not False
            or candidate.backend_contract.get("rank_eligible") is not False
            or candidate.backend_contract.get("mutable_alias_execution_requires_explicit_opt_in")
            is not True
        ):
            raise RuntimeError(
                "mutable QwenCloud alias requires explicit permanently exploratory opt-in"
            )
        active_accounting_basis = mutable_alias_accounting_basis
        active_billing_status = mutable_alias_billing_status
    else:
        if mutable_alias_opt_in:
            raise RuntimeError(
                "mutable-alias exploratory opt-in cannot be applied to a frozen release"
            )
        active_accounting_basis = accounting_basis
        active_billing_status = "provider_charge_unavailable"

    execution_policy = ExecutionPolicy.from_settings(
        settings,
        pair_arm_scheduling="sequential" if args.sequential_arms else "concurrent",
        final_response_mode="plain_text" if args.plain_text_final else "structured_json",
        matched_planning=(args.evidence_protocol in MATCHED_EVIDENCE_PROTOCOLS),
        evidence_protocol=args.evidence_protocol,
        intermediate_reasoning_effort=args.intermediate_reasoning_effort,
        final_reasoning_effort=args.final_reasoning_effort,
        tool_catalog_bytes_bound=args.tool_catalog_bytes_bound,
        epicure_on_tool_required=args.require_epicure_call,
    )
    if execution_policy.sha256 != args.expected_execution_policy_sha256:
        raise RuntimeError("runtime settings differ from the frozen execution policy")

    model = {
        "id": candidate.model_id,
        "name": candidate.model_name,
        "canonical_slug": candidate.canonical_model_slug,
        "context_length": candidate.endpoint.get("context_length"),
    }
    endpoint = dict(candidate.endpoint)
    rate_card = _rate_card(endpoint)
    if args.cap_usd == 0:
        zero_rate_status = "frozen_public_api_free_until_rate_limit"
        priced_fields = (
            "prompt_price_per_token",
            "completion_price_per_token",
            "internal_reasoning_price_per_token",
            "request_price",
        )
        if rate_card.get("status") != zero_rate_status or any(
            Decimal(str(rate_card[field])) != 0 for field in priced_fields
        ):
            raise RuntimeError(
                "zero-cap Cohere admission requires the frozen public-free rate card"
            )
    forecast = _worst_case_cost_usd(
        endpoint,
        prompt=args.prompt,
        include_tool_contract=False,
        execution_policy=execution_policy,
        conditions=selected_conditions,
    )
    if mutable_alias:
        pricing_contract = endpoint.get("pricing")
        if not isinstance(pricing_contract, dict):
            raise RuntimeError("mutable QwenCloud alias lacks its budget ceiling")
        operational_ceiling = Decimal(
            str(pricing_contract.get("operational_reservation_ceiling_usd") or "")
        )
        if (
            not operational_ceiling.is_finite()
            or operational_ceiling <= 0
            or pricing_contract.get("provider_rate_known") is not False
            or pricing_contract.get("zero_values_mean") != "unknown_cost_not_free"
            or args.cap_usd != operational_ceiling
        ):
            raise RuntimeError(
                "mutable QwenCloud alias requires its exact full unpriced budget ceiling"
            )
        forecast = operational_ceiling
    if forecast > args.cap_usd:
        raise RuntimeError(f"forecast cost ${forecast:.6f} exceeds pair cap ${args.cap_usd:.6f}")
    provenance, tool_schema_sha256 = await _epicure_attestation()
    verify_expected_epicure_attestation(
        provenance,
        tool_schema_sha256,
        expected_release_id=getattr(args, "expected_epicure_release_id", ""),
        expected_bundle_sha256=getattr(args, "expected_epicure_bundle_sha256", ""),
        expected_application_sha256=getattr(args, "expected_epicure_application_sha256", ""),
        expected_tool_schema_sha256=getattr(args, "expected_epicure_tool_schema_sha256", ""),
    )
    generation_contract = frozen_generation_contract(model, endpoint)
    if mutable_alias:
        generation_contract["allow_mutable_alias_exploratory"] = True
    run_purpose = (
        "epicure_on_off_pair"
        if selected_conditions == ("epicure_off", "epicure_on")
        else "epicure_condition_subset"
    )
    protocol_bundle, protocol_bundle_sha256 = build_live_protocol_bundle(
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        dataset_work_item_id=args.dataset_work_item_id,
        dataset_task_id=args.dataset_task_id,
        prompt=args.prompt,
        category=args.category,
        model=model,
        endpoint=endpoint,
        generation_contract=generation_contract,
        execution_policy=execution_policy,
        provenance=provenance,
        tool_schema_sha256=tool_schema_sha256,
        run_purpose=run_purpose,
        final_response_mode=execution_policy.final_response_mode,
        selected_conditions=(
            selected_conditions if run_purpose == "epicure_condition_subset" else None
        ),
    )
    scope_sha256 = provider_account_scope_sha256(execution_backend)
    generation_contract.update(
        {
            "final_response_mode": execution_policy.final_response_mode,
            "matched_planning": execution_policy.matched_planning,
            "intermediate_max_tokens": execution_policy.max_intermediate_tokens,
            "required_tool_contract_max_intermediate_tokens": (
                execution_policy.required_tool_contract_max_intermediate_tokens
            ),
            "evidence_protocol": execution_policy.evidence_protocol,
            "required_tool_contract_protocol": execution_policy.required_tool_contract_protocol,
            "required_tool_contract_sha256": required_tool_contract(execution_policy)[
                "content_address"
            ]["digest"],
            "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
            "intermediate_reasoning_effort": execution_policy.intermediate_reasoning_effort,
            "final_reasoning_effort": execution_policy.final_reasoning_effort,
            "allow_mutable_alias_exploratory": mutable_alias_opt_in,
            "protocol_bundle_sha256": protocol_bundle_sha256,
            "expected_epicure_release_id": str(provenance["release_id"]),
            "expected_epicure_bundle_sha256": str(provenance["bundle_sha256"]),
            "expected_epicure_application_sha256": str(provenance["application_sha256"]),
            "expected_epicure_tool_schema_sha256": tool_schema_sha256,
            "execution_backend": execution_backend,
            "rate_card_json": rate_card,
            "backend_contract_json": dict(candidate.backend_contract),
            "provider_budget_cap_micros": int(
                (args.cap_usd * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING)
            ),
            "provider_account_budget_cap_micros": provider_account_hard_cap_micros(
                execution_backend
            ),
            "provider_account_scope_sha256": scope_sha256,
            "provider_authorization_envelope_sha256": _credential_free_binding(
                manifest_sha256=args.candidate_manifest_sha256,
                backend_contract_sha256=candidate.backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="pair_budget",
            ),
            "provider_account_authorization_envelope_sha256": _credential_free_binding(
                manifest_sha256=args.candidate_manifest_sha256,
                backend_contract_sha256=candidate.backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="account_budget",
            ),
            "provider_credential_binding_sha256": _credential_free_binding(
                manifest_sha256=args.candidate_manifest_sha256,
                backend_contract_sha256=candidate.backend_contract_sha256,
                scope_sha256=scope_sha256,
                kind="credential_binding",
            ),
            "provider_credential_scope_sha256": scope_sha256,
            "contract_smoke_registry_sha256": str(
                candidate.route_selection.get("evidence", {}).get("compatibility_artifact_sha256")
                or "unresolved"
            ),
        }
    )

    if args.preflight_only:
        preflight: dict[str, Any] = {
            "schema_version": DIRECT_PREFLIGHT_SCHEMA_VERSION,
            "status": "preflight_passed_no_provider_calls",
            "provider_calls_made": False,
            "epicure_attestation_performed": True,
            "epicure_mcp_url": settings.mcp_url,
            "epicure_provenance_url": provenance_url,
            "recorded_at": datetime.now(UTC).isoformat(),
            "official": False,
            "season_eligible": False,
            "rank_eligible": False,
            "research_result": False,
            "model_id": candidate.model_id,
            "canonical_model_slug": candidate.canonical_model_slug,
            "provider_slug": candidate.provider_tag,
            "execution_backend": candidate.execution_backend,
            "candidate_manifest_sha256": args.candidate_manifest_sha256,
            "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            "backend_contract_sha256": candidate.backend_contract_sha256,
            "backend_contract": dict(candidate.backend_contract),
            "execution_policy_sha256": execution_policy.sha256,
            "execution_policy": execution_policy.document(),
            "protocol_bundle_sha256": protocol_bundle_sha256,
            "dataset_work_item_id": args.dataset_work_item_id,
            "dataset_task_id": args.dataset_task_id,
            "category": args.category,
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "conditions": list(selected_conditions),
            "epicure_release_id": provenance["release_id"],
            "epicure_bundle_sha256": provenance["bundle_sha256"],
            "epicure_application_sha256": provenance["application_sha256"],
            "epicure_tool_schema_sha256": tool_schema_sha256,
            "forecast_worst_case_usd": str(forecast),
            "cap_usd": str(args.cap_usd),
            "full_unpriced_budget_ceiling_retained": mutable_alias,
            "provider_cost_known": False if mutable_alias else None,
            "reservation_entry_sha256": getattr(args, "reservation_entry_sha256", None),
            "go_template_sha256": getattr(args, "expected_go_template_sha256", None),
            "model_identity_label": candidate.backend_contract.get("model_identity_label"),
        }
        preflight["artifact_sha256"] = _sha256(preflight)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        preflight_path = output_dir / (f"preflight-{preflight['artifact_sha256']}.json")
        rendered = json.dumps(preflight, indent=2, sort_keys=True) + "\n"
        if preflight_path.exists():
            if preflight_path.read_text(encoding="utf-8") != rendered:
                raise RuntimeError("content-addressed preflight artifact conflict")
        else:
            with preflight_path.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            preflight_path.chmod(0o644)
        return {
            **preflight,
            "artifact": str(preflight_path.resolve()),
        }

    run_id = str(getattr(args, "frozen_run_id", "") or uuid.uuid4())
    frozen_attempt_slots = getattr(args, "frozen_attempt_slots", None)
    attempt_id_factory = None
    if frozen_attempt_slots is not None:
        if not isinstance(frozen_attempt_slots, list) or not frozen_attempt_slots:
            raise RuntimeError("frozen attempt slots must be a non-empty list")
        slot_map: dict[tuple[str, str, int], str] = {}
        frozen_ids: set[str] = set()
        for slot in frozen_attempt_slots:
            if not isinstance(slot, dict):
                raise RuntimeError("frozen attempt slot must be an object")
            key = (
                str(slot.get("arm_id") or ""),
                str(slot.get("phase") or ""),
                int(slot.get("attempt_index", -1)),
            )
            attempt_id = str(slot.get("attempt_id") or "")
            if (
                not all(key[:2])
                or key[2] < 0
                or not attempt_id
                or key in slot_map
                or attempt_id in frozen_ids
            ):
                raise RuntimeError("frozen attempt slots are malformed or duplicated")
            slot_map[key] = attempt_id
            frozen_ids.add(attempt_id)

        def resolve_attempt_id(arm_id: str, phase: str, attempt_index: int) -> str:
            try:
                return slot_map[(arm_id, phase, attempt_index)]
            except KeyError as error:
                raise RuntimeError(
                    "external request has no pre-frozen attempt-ID slot: "
                    f"{arm_id}/{phase}/{attempt_index}"
                ) from error

        attempt_id_factory = resolve_attempt_id
    started_at = datetime.now(UTC)
    output_dir = Path(args.output_dir)
    journal = RunJournal.create(
        output_dir,
        run_id=run_id,
        metadata={
            "run_class": "engineering_live_smoke",
            "execution_backend": execution_backend,
            "candidate_manifest_sha256": args.candidate_manifest_sha256,
            "dataset_work_item_id": args.dataset_work_item_id,
            "dataset_task_id": args.dataset_task_id,
            "requested_model_id": args.model_id,
            "requested_provider": args.provider_slug,
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "category": args.category,
            "contract_only": False,
            "epicure_conditions": list(selected_conditions),
        },
    )
    attempt_events: list[dict[str, Any]] = []
    mcp_trace_events: list[dict[str, Any]] = []

    def attempt_sink(event: ProviderAttemptEvent) -> None:
        payload = asdict(event)
        journal.append("provider_attempt", payload)
        attempt_events.append(payload)

    def tool_sink(arm_id: str, trace: ToolTrace) -> None:
        payload = {
            "arm_id": arm_id,
            "round_index": trace.round_index,
            "name": trace.name,
            "arguments": trace.arguments,
            "result": trace.result,
            "result_sha256": hashlib.sha256(trace.result.encode()).hexdigest(),
            "latency_ms": trace.latency_ms,
            "is_error": trace.is_error,
        }
        journal.append("mcp_trace", payload)
        mcp_trace_events.append(payload)

    provider_kwargs: dict[str, Any] = {
        "attempt_sink": attempt_sink,
        "tool_sink": tool_sink,
    }
    if attempt_id_factory is not None:
        provider_kwargs["attempt_id_factory"] = attempt_id_factory
    provider = provider_factory(**provider_kwargs)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    incomplete_generation_metadata: list[dict[str, Any]] = []
    try:
        specs = [
            GenerationSpec(
                arm_id=f"{run_id}:{condition}",
                battle_id=run_id,
                prompt=args.prompt,
                category=args.category,
                model_id=args.model_id,
                model_name=candidate.model_name,
                provider_slug=args.provider_slug,
                condition=condition,
                idempotency_key=(f"flavourbench-direct-{execution_backend}:{run_id}:{condition}"),
                **{
                    **generation_contract,
                    "epicure_on_tool_required": (
                        condition == "epicure_on" and execution_policy.epicure_on_tool_required
                    ),
                },
            )
            for condition in selected_conditions
        ]
        if args.sequential_arms:
            outcomes: list[GenerationResult | Exception] = []
            for spec in specs:
                try:
                    outcomes.append(await provider.generate(spec))
                except Exception as error:
                    outcomes.append(error)
        else:
            outcomes = await asyncio.gather(
                *(provider.generate(spec) for spec in specs),
                return_exceptions=True,
            )
        for spec, outcome in zip(specs, outcomes, strict=True):
            if isinstance(outcome, GenerationResult):
                result = _result_payload(outcome)
                results[spec.condition] = result
                if not _model_identity_matches(outcome.actual_model_id, model):
                    errors[f"{spec.condition}_model_identity"] = "returned model differs"
                if not _provider_identity_matches(outcome.provider_slug, endpoint):
                    errors[f"{spec.condition}_provider_identity"] = "returned provider differs"
                if outcome.finish_reason not in {"stop", "end_turn"}:
                    errors[f"{spec.condition}_finish_reason"] = outcome.finish_reason
            else:
                errors[spec.condition] = f"{type(outcome).__name__}: {outcome}"
    finally:
        accounted_ids = {
            str(item.get("generation_id") or "")
            for result in results.values()
            for item in result.get("generation_metadata") or []
        }
        received_ids = {
            str(event.get("generation_id") or "")
            for event in attempt_events
            if event.get("event_type") == "response_received"
        }
        for generation_id in sorted(received_ids - accounted_ids - {""}):
            incomplete_generation_metadata.append(
                await provider._generation_cost(generation_id)  # noqa: SLF001
            )
        await provider.aclose()

    total_cost_micros = sum(
        int(result.get("cost_micros") or 0) for result in results.values()
    ) + sum(int(item.get("cost_micros") or 0) for item in incomplete_generation_metadata)
    rate_card_accounted = (
        set(results) == set(selected_conditions)
        and not incomplete_generation_metadata
        and all(
            _rate_card_result_is_accounted(
                result,
                active_accounting_basis,
                active_billing_status,
            )
            for result in results.values()
        )
    )
    complete = not errors and rate_card_accounted
    complete_status = (
        "complete_unpriced_budget_ceiling" if mutable_alias else "complete_rate_card_estimated"
    )
    journal_complete_status = (
        "generation_complete_unpriced_budget_ceiling"
        if mutable_alias
        else "generation_complete_rate_card_estimated"
    )
    journal_descriptor = journal.finalize(
        {
            "status": journal_complete_status if complete else "failed",
            "condition_names": sorted(results),
            "error_keys": sorted(errors),
            "generation_ids": sorted(
                {
                    str(generation_id)
                    for result in results.values()
                    for generation_id in result.get("generation_ids") or []
                    if generation_id
                }
            ),
            "actual_cost_micros": total_cost_micros,
            "all_generation_costs_reconciled": False,
            "all_generation_usage_rate_card_accounted": rate_card_accounted,
            "provider_cost_known": not mutable_alias,
        }
    )
    artifact: dict[str, Any] = {
        "schema_version": "flavourbench-live-smoke-v1",
        "run_id": run_id,
        "status": complete_status if complete else "failed_or_unreconciled",
        "run_class": "engineering_live_smoke",
        "run_purpose": run_purpose,
        "requested_conditions": list(selected_conditions),
        "execution_backend": execution_backend,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "dataset_work_item_id": args.dataset_work_item_id,
        "dataset_task_id": args.dataset_task_id,
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "category": args.category,
        "requested_model_id": args.model_id,
        "requested_provider": args.provider_slug,
        "model_identity_status": candidate.backend_contract.get("model_identity_label"),
        "catalog_observed_at": candidate.backend_contract.get("catalog_observed_at"),
        "mutable_alias_exploratory_opt_in": mutable_alias_opt_in,
        "model_contract": model,
        "endpoint_contract": endpoint,
        "endpoint_contract_sha256": _sha256(endpoint),
        "endpoint_execution_contract_sha256": endpoint_execution_contract_sha256(endpoint),
        "backend_contract": dict(candidate.backend_contract),
        "backend_contract_sha256": candidate.backend_contract_sha256,
        "execution_route": dict(candidate.route_selection),
        "frozen_generation_contract": {
            "supported_parameters": sorted(generation_contract["supported_parameters"]),
            "decoding_parameters": generation_contract["decoding_parameters"],
            "expected_actual_model_id": generation_contract["expected_actual_model_id"],
            "expected_actual_provider_slug": generation_contract["expected_actual_provider_slug"],
            "endpoint_contract_sha256": generation_contract["endpoint_contract_sha256"],
            "execution_backend": execution_backend,
            "backend_contract_sha256": candidate.backend_contract_sha256,
            "final_response_mode": execution_policy.final_response_mode,
            "matched_planning": execution_policy.matched_planning,
            "intermediate_max_tokens": execution_policy.max_intermediate_tokens,
            "required_tool_contract_max_intermediate_tokens": (
                execution_policy.required_tool_contract_max_intermediate_tokens
            ),
            "evidence_protocol": execution_policy.evidence_protocol,
            "required_tool_contract_protocol": execution_policy.required_tool_contract_protocol,
            "required_tool_contract_sha256": required_tool_contract(execution_policy)[
                "content_address"
            ]["digest"],
            "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
            "intermediate_reasoning_effort": execution_policy.intermediate_reasoning_effort,
            "final_reasoning_effort": execution_policy.final_reasoning_effort,
            "allow_mutable_alias_exploratory": mutable_alias_opt_in,
            "protocol_bundle_sha256": protocol_bundle_sha256,
            "expected_epicure_release_id": provenance["release_id"],
            "expected_epicure_bundle_sha256": provenance["bundle_sha256"],
            "expected_epicure_application_sha256": provenance["application_sha256"],
            "expected_epicure_tool_schema_sha256": tool_schema_sha256,
        },
        "protocol_bundle": protocol_bundle,
        "protocol_bundle_sha256": protocol_bundle_sha256,
        "required_tool_contract": required_tool_contract(execution_policy),
        "execution_policy": execution_policy.document(),
        "execution_policy_sha256": execution_policy.sha256,
        "decoding": {
            "temperature": settings.decoding_temperature,
            "top_p": settings.decoding_top_p,
            "seed": settings.decoding_seed,
            "max_output_tokens": settings.max_output_tokens,
            "max_tool_rounds": settings.max_tool_rounds,
            "max_tool_calls_per_round": settings.max_tool_calls_per_round,
            "max_tool_calls_total": settings.max_tool_calls_total,
            "max_tool_result_bytes": settings.max_tool_result_bytes,
            "max_cumulative_tool_result_bytes": settings.max_cumulative_tool_result_bytes,
            "max_provider_attempts": settings.max_provider_attempts,
            "parallel_tool_calls_enforcement": "bounded_sequential_execution",
        },
        "system_prompt_sha256": {
            condition: system_prompt_sha256(
                condition,
                execution_policy.final_response_mode,
                execution_policy.evidence_protocol,
            )
            for condition in selected_conditions
        },
        "response_schema_sha256": response_schema_sha256(execution_policy.final_response_mode),
        "epicure": provenance,
        "epicure_transport": {
            "mcp_url": settings.mcp_url,
            "provenance_url": provenance_url,
        },
        "epicure_tool_schema_sha256": tool_schema_sha256,
        "budget": {
            "cap_usd": str(args.cap_usd),
            "forecast_worst_case_usd": str(forecast),
            "actual_cost_micros": total_cost_micros,
            "all_generation_costs_reconciled": False,
            "all_generation_usage_rate_card_accounted": rate_card_accounted,
            "all_generation_usage_accounted": rate_card_accounted,
            "accounting_basis": (
                "provider_usage_with_unpriced_budget_ceiling"
                if mutable_alias
                else "provider_usage_times_frozen_rate_card"
            ),
            "provider_charge_available": False,
            "provider_rate_available": not mutable_alias,
            "provider_cost_known": False,
            "full_unpriced_budget_ceiling_retained": mutable_alias,
            "retained_exposure_usd": str(args.cap_usd) if mutable_alias else None,
            "zero_recorded_cost_means": "unknown_not_free" if mutable_alias else None,
            "terminal_exposure_basis": (
                "full_unpriced_budget_ceiling_permanently_retained"
                if mutable_alias
                else "provider_usage_times_frozen_rate_card"
            ),
            "provider_account_snapshot_before": "endpoint_not_available",
            "provider_account_snapshot_after": "endpoint_not_available",
        },
        "results": results,
        "errors": errors,
        "provider_attempt_events": attempt_events,
        "mcp_trace_events": mcp_trace_events,
        "incomplete_generation_metadata": incomplete_generation_metadata,
        "run_journal": journal_descriptor.payload(),
        "limitations": [
            "This is an unranked development run and cannot enter an official leaderboard.",
            (
                f"{provider_label} returned per-generation token usage but no provider "
                + (
                    "rate or charged amount for this catalog-observed alias."
                    if mutable_alias
                    else "charged amount."
                )
            ),
            (
                "The full admitted ceiling is retained as exposure; zero recorded cost means "
                "unknown, not free."
                if mutable_alias
                else f"Recorded {provider_label} cost is a frozen public rate-card estimate "
                "and is excluded from exact cost rankings."
            ),
            *(
                [
                    "The requested model is a mutable alias pinned only at one authenticated "
                    "catalog observation, not a frozen model release."
                ]
                if mutable_alias
                else []
            ),
            "The Epicure bundle remains an unmatched exploratory runtime release.",
            "No generation-time provider fallback was permitted.",
        ],
    }
    artifact["artifact_sha256"] = _sha256(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{started_at:%Y%m%dT%H%M%SZ}-{artifact['artifact_sha256'][:12]}.json"
    )
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {output_path}")
    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    output_path.chmod(0o644)
    return {
        "status": artifact["status"],
        "rank_eligible": False,
        "artifact": str(output_path.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "estimated_cost_micros": total_cost_micros,
        "conditions": sorted(results),
        "errors": errors,
    }


async def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    return await _run_direct_pair(
        args,
        execution_backend="kimi_direct",
        provider_factory=KimiDirectProvider,
        credential_attribute="kimi_api_key",
        accounting_basis=KIMI_ACCOUNTING_BASIS,
        provider_label="Kimi",
    )


def _parser(
    description: str | None = None,
    *,
    reasoning_required: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--cap-usd", type=Decimal, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--provider-slug", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--category",
        choices=["substitution", "composition", "cookability", "evidence"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-work-item-id", required=True)
    parser.add_argument("--dataset-task-id", required=True)
    parser.add_argument("--expected-canonical-model-slug", required=True)
    parser.add_argument("--expected-endpoint-execution-sha256", required=True)
    parser.add_argument("--expected-execution-policy-sha256", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        choices=["epicure_off", "epicure_on"],
        help="Run only the selected condition; repeat for an explicit subset.",
    )
    parser.add_argument("--expected-epicure-release-id", default="")
    parser.add_argument("--expected-epicure-bundle-sha256", default="")
    parser.add_argument("--expected-epicure-application-sha256", default="")
    parser.add_argument("--expected-epicure-tool-schema-sha256", default="")
    parser.add_argument("--plain-text-final", action="store_true")
    parser.add_argument("--tool-catalog-bytes-bound", type=int, default=0)
    parser.add_argument(
        "--require-epicure-call",
        action="store_true",
        help="Require at least one successful real Epicure call in the Epicure-on arm.",
    )
    parser.add_argument(
        "--evidence-protocol",
        choices=sorted(GOVERNED_EPICURE_PROTOCOLS),
        required=True,
    )
    parser.add_argument(
        "--intermediate-reasoning-effort",
        choices=["low", "high", "max"],
        required=reasoning_required,
    )
    parser.add_argument(
        "--final-reasoning-effort",
        choices=["low", "high", "max"],
        required=reasoning_required,
    )
    parser.add_argument("--sequential-arms", action="store_true")
    parser.add_argument("--frozen-run-id", default="")
    parser.add_argument("--frozen-attempt-slots-json", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the frozen route and live Epicure attestation without model calls.",
    )
    return parser


def run() -> None:
    assert_legacy_paid_cli_allowed("flavourbench-direct-kimi-pair")
    try:
        args = _parser(reasoning_required=False).parse_args()
        if args.frozen_attempt_slots_json:
            slots = json.loads(args.frozen_attempt_slots_json)
            if not isinstance(slots, list):
                raise RuntimeError("frozen attempt slots JSON must decode to an array")
            args.frozen_attempt_slots = slots
        summary = asyncio.run(run_pair(args))
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] not in {
        "complete_rate_card_estimated",
        "preflight_passed_no_provider_calls",
    }:
        sys.exit(2)


if __name__ == "__main__":
    run()
