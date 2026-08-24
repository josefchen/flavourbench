"""Freeze a provider-first manifest for the real current-model development pilot.

The routing rule is resolved before generation. Kimi K3 uses Moonshot's direct
Kimi Code endpoint. Every other logical model is checked against a frozen
Bedrock catalog. An exact Bedrock route is preferred, while OpenRouter is
selected only when the exact model is absent or a real preflight proves that
the account cannot invoke it. There is no generation-time provider fallback.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .execution_policy import MATCHED_EVIDENCE_PROTOCOL_V2, ExecutionPolicy
from .frontier_contract_runner import ContractCandidate, load_candidate_manifest
from .frontier_manifest import verify_manifest_content_address
from .real_dataset_runner import (
    WorkItem,
    derive_pair_forecast,
    load_development_task_inventory,
    select_balanced_tasks,
    task_registry_sha256,
)
from .real_task_bank import sha256_json
from .tool_contract import required_tool_contract

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
MANIFEST_ROLE = "current_frontier_routed_development_quality_run"
SELECTION_SEED = "flavourbench-current-frontier-quality-v2-matched-evidence"
DEFAULT_CAP_USD = Decimal("100")
KIMI_MODEL_ID = "moonshotai/kimi-k3"
KIMI_PROVIDER = "kimi-code-direct"
KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_PRICE_SOURCE = "https://platform.kimi.com/docs/pricing/chat"

BEDROCK_EXACT_TARGETS = {
    "anthropic/claude-fable-5": "global.anthropic.claude-fable-5",
    "anthropic/claude-opus-5": "global.anthropic.claude-opus-5",
    "anthropic/claude-sonnet-5": "global.anthropic.claude-sonnet-5",
}
BEDROCK_PHYSICAL_REGION_PATTERN = re.compile(
    r"^[a-z]{2}(?:-gov)?-[a-z]+(?:-[a-z]+)*-\d+$"
)


class RoutedManifestError(RuntimeError):
    """A provider route could not be frozen without ambiguity."""


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RoutedManifestError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RoutedManifestError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RoutedManifestError(f"expected a JSON object: {path}")
    return value


def _verified_artifact(path: Path, *, digest_field: str = "artifact_sha256") -> dict[str, Any]:
    value = _read_json(path)
    recorded = str(value.get(digest_field) or "")
    payload = {key: item for key, item in value.items() if key != digest_field}
    if len(recorded) != 64 or sha256_json(payload) != recorded or recorded not in path.name:
        raise RoutedManifestError(f"content address does not verify: {path}")
    return value


def _kimi_entry(
    entry: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    models = catalog.get("models")
    if not isinstance(models, list):
        raise RoutedManifestError("Kimi catalog has no model list")
    matches = [model for model in models if isinstance(model, Mapping) and model.get("id") == "k3"]
    if len(matches) != 1:
        raise RoutedManifestError("Kimi catalog does not contain exactly one k3 identity")
    catalog_entry = dict(matches[0])
    catalog_entry_sha256 = sha256_json(catalog_entry)
    if (
        compatibility.get("status") != "smoke_passed"
        or compatibility.get("requested_model_id") != "k3"
        or compatibility.get("provider") != "kimi_code_direct"
        or compatibility.get("catalog_sha256") != catalog.get("artifact_sha256")
        or compatibility.get("catalog_entry_sha256") != catalog_entry_sha256
        or int(compatibility.get("real_provider_calls") or 0) != 2
        or int(compatibility.get("real_epicure_calls") or 0) != 1
        or compatibility.get("rank_eligible") is not False
    ):
        raise RoutedManifestError("direct Kimi contract evidence does not verify")

    routed = copy.deepcopy(dict(entry))
    routed["model"] = {
        **dict(routed["model"]),
        "canonical_slug": "k3",
        "name": "MoonshotAI: Kimi K3",
        "context_length": int(catalog_entry["context_length"]),
    }
    endpoint = {
        "model_id": KIMI_MODEL_ID,
        "provider_name": KIMI_PROVIDER,
        "tag": KIMI_PROVIDER,
        "quantization": "provider_managed_unpublished",
        "context_length": int(catalog_entry["context_length"]),
        "max_completion_tokens": 8192,
        "pricing": {
            "prompt": "0.000003",
            "completion": "0.000015",
            "input_cache_read": "0.0000003",
            "internal_reasoning": "0",
            "request": "0",
            "currency": "USD",
            "source": KIMI_PRICE_SOURCE,
            "status": "frozen_public_rate_card_estimate",
        },
        "supported_parameters": [
            "max_tokens",
            "reasoning",
            "response_format",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
        ],
    }
    backend_contract = {
        "schema_version": "flavourbench-kimi-direct-endpoint-contract-v1",
        "base_url": KIMI_BASE_URL,
        "requested_model_id": "k3",
        "expected_actual_provider_slug": KIMI_PROVIDER,
        "catalog_sha256": catalog["artifact_sha256"],
        "catalog_entry_sha256": catalog_entry_sha256,
        "allow_fallbacks": False,
        "reasoning_parameter_translation": "reasoning.effort_to_reasoning_effort",
        "cost_accounting": "provider_usage_times_frozen_rate_card",
        "season_eligible": False,
    }
    routed["endpoint"] = endpoint
    routed["endpoint_document_sha256"] = sha256_json(endpoint)
    routed["endpoint_selection"] = {
        "method": "exact direct Kimi model and endpoint frozen before generation",
        "selected_exact_tag": KIMI_PROVIDER,
        "quality_observations_used": 0,
    }
    routed["request_policy"] = {
        "policy_scope": "request_enforced",
        "provider": {
            "only": [KIMI_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        },
        "official_eligibility": "development_only",
    }
    routed["execution_route"] = {
        "policy": "direct_kimi_then_exact_bedrock_then_openrouter_v1",
        "preferred_backend": "kimi_direct",
        "selected_backend": "kimi_direct",
        "fallback_used": False,
        "selection_reason": "exact_direct_kimi_k3_contract_passed",
        "selection_frozen_before_generation": True,
        "generation_time_automatic_fallback": False,
        "evidence": {
            "catalog_sha256": catalog["artifact_sha256"],
            "compatibility_artifact_sha256": compatibility["artifact_sha256"],
        },
    }
    routed["backend_contract"] = backend_contract
    routed["backend_contract_sha256"] = sha256_json(backend_contract)
    routed["cost_accounting_policy"] = "provider_usage_times_frozen_rate_card"
    routed["contract_evidence"] = {
        "source_artifact_sha256": compatibility["artifact_sha256"],
        "real_provider_calls": compatibility["real_provider_calls"],
        "real_epicure_calls": compatibility["real_epicure_calls"],
        "actual_provider": KIMI_PROVIDER,
        "identity_basis": "direct_response_model_and_generation_id",
        "contract_status": "passed_unranked",
        "cost_status": "per_generation_provider_charge_unavailable",
    }
    return routed


def _openrouter_fallback_entry(
    entry: Mapping[str, Any],
    *,
    bedrock_catalog_sha256: str,
    claude_access_receipts: Mapping[str, str],
) -> dict[str, Any]:
    routed = copy.deepcopy(dict(entry))
    model_id = str(routed.get("model", {}).get("id") or "")
    endpoint = routed.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise RoutedManifestError(f"OpenRouter fallback route is malformed: {model_id}")
    exact_bedrock_target = BEDROCK_EXACT_TARGETS.get(model_id)
    if exact_bedrock_target is None:
        reason = "bedrock_exact_model_absent"
        evidence = {
            "bedrock_catalog_sha256": bedrock_catalog_sha256,
            "exact_bedrock_target_id": None,
        }
    else:
        receipt = claude_access_receipts.get(model_id)
        if receipt is None:
            raise RoutedManifestError(f"missing Bedrock access-denial receipt for {model_id}")
        reason = "bedrock_account_access_denied_before_generation"
        evidence = {
            "bedrock_catalog_sha256": bedrock_catalog_sha256,
            "exact_bedrock_target_id": exact_bedrock_target,
            "bedrock_access_receipt_sha256": receipt,
        }
    routed["execution_route"] = {
        "policy": "direct_kimi_then_exact_bedrock_then_openrouter_v1",
        "preferred_backend": "bedrock",
        "selected_backend": "openrouter",
        "fallback_used": True,
        "selection_reason": reason,
        "selection_frozen_before_generation": True,
        "generation_time_automatic_fallback": False,
        "evidence": evidence,
    }
    routed["backend_contract"] = {}
    routed["backend_contract_sha256"] = "unfrozen"
    routed["cost_accounting_policy"] = "provider_generation_metadata"
    return routed


def _candidate(entry: Mapping[str, Any]) -> ContractCandidate:
    model = entry.get("model")
    endpoint = entry.get("endpoint")
    route = entry.get("execution_route")
    if not all(isinstance(value, Mapping) for value in (model, endpoint, route)):
        raise RoutedManifestError("routed entry lacks model, endpoint, or route")
    backend_contract = entry.get("backend_contract") or {}
    return ContractCandidate(
        slot_id=str(entry.get("slot", {}).get("slot_id") or ""),
        model_id=str(model.get("id") or ""),
        canonical_model_slug=str(model.get("canonical_slug") or ""),
        model_name=str(model.get("name") or model.get("id") or ""),
        provider_tag=str(endpoint.get("tag") or ""),
        provider_name=str(endpoint.get("provider_name") or ""),
        endpoint_sha256=sha256_json(endpoint),
        endpoint_execution_sha256="derived_at_runner_load",
        endpoint=dict(endpoint),
        execution_backend=str(route.get("selected_backend") or ""),
        backend_contract=dict(backend_contract),
        backend_contract_sha256=str(entry.get("backend_contract_sha256") or "unfrozen"),
        route_selection=dict(route),
        cost_accounting_policy=str(entry.get("cost_accounting_policy") or ""),
    )


def _access_denial_receipts(
    *,
    unified_path: Path | None,
    fable_path: Path | None,
    claude_eu_path: Path | None,
    expected_bedrock_catalog_sha256: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Verify immutable pre-generation denials for every exact Claude 5 route.

    The unified form is the current contract: all three global inference
    profiles were tested together against one account and one catalog freeze.
    The split form remains readable so older content-addressed manifests and
    their tests stay reproducible.
    """

    if unified_path is not None:
        if fable_path is not None or claude_eu_path is not None:
            raise RoutedManifestError(
                "use either one unified Bedrock receipt or the two legacy receipts"
            )
        receipts = [_verified_artifact(unified_path)]
        expected_targets = set(BEDROCK_EXACT_TARGETS.values())
    else:
        if fable_path is None or claude_eu_path is None:
            raise RoutedManifestError("Bedrock access-denial evidence is incomplete")
        receipts = [
            _verified_artifact(fable_path),
            _verified_artifact(claude_eu_path),
        ]
        expected_targets = {
            "global.anthropic.claude-fable-5",
            "eu.anthropic.claude-opus-5",
            "eu.anthropic.claude-sonnet-5",
        }

    observed: dict[str, str] = {}
    receipt_metadata: list[dict[str, Any]] = []
    for receipt in receipts:
        counts = receipt.get("counts") or {}
        artifacts = receipt.get("artifacts") or []
        if (
            receipt.get("schema_version")
            != "flavourbench-frontier-refresh-contract-summary-v1"
            or receipt.get("status") != "one_or_more_contract_smokes_failed"
            or receipt.get("bedrock_catalog_sha256")
            != expected_bedrock_catalog_sha256
            or not isinstance(artifacts, list)
            or int(counts.get("failed") or 0) != len(artifacts)
            or int(counts.get("real_provider_calls") or 0) != 0
            or int(counts.get("real_epicure_calls") or 0) != 0
            or any(
                not isinstance(item, Mapping)
                or item.get("provider") != "bedrock"
                or item.get("error_type") != "AccessDeniedException"
                or item.get("provider_calls") != 0
                or item.get("epicure_calls") != 0
                or item.get("status") != "failed"
                or not isinstance(item.get("requested_endpoint_id"), str)
                for item in artifacts
            )
        ):
            raise RoutedManifestError("Bedrock access-denial evidence is malformed")
        digest = str(receipt["artifact_sha256"])
        targets = [str(item["requested_endpoint_id"]) for item in artifacts]
        for target in targets:
            if target in observed:
                raise RoutedManifestError("duplicate Bedrock access-denial target")
            observed[target] = digest
        receipt_metadata.append(
            {
                "artifact_sha256": digest,
                "requested_endpoint_ids": sorted(targets),
                "failed_pre_generation": len(targets),
                "real_provider_calls": 0,
                "real_epicure_calls": 0,
            }
        )
    if set(observed) != expected_targets:
        raise RoutedManifestError("Bedrock denial receipts do not cover the frozen targets")

    if unified_path is not None:
        access_receipts = {
            model_id: observed[target_id]
            for model_id, target_id in BEDROCK_EXACT_TARGETS.items()
        }
    else:
        access_receipts = {
            "anthropic/claude-fable-5": observed["global.anthropic.claude-fable-5"],
            "anthropic/claude-opus-5": observed["eu.anthropic.claude-opus-5"],
            "anthropic/claude-sonnet-5": observed["eu.anthropic.claude-sonnet-5"],
        }
    return access_receipts, receipt_metadata


def build_manifest(
    *,
    base_manifest_path: Path,
    expected_base_manifest_sha256: str,
    task_validity_path: Path,
    kimi_catalog_path: Path,
    kimi_compatibility_path: Path,
    bedrock_catalog_path: Path,
    bedrock_fable_receipt_path: Path | None,
    bedrock_claude_eu_receipt_path: Path | None,
    cap_usd: Decimal,
    execution_policy: ExecutionPolicy,
    bedrock_access_receipt_path: Path | None = None,
    target_model_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if cap_usd <= 0 or cap_usd > DEFAULT_CAP_USD:
        raise RoutedManifestError("cap must be in (0, 100]")
    execution_policy.validate()
    base = load_candidate_manifest(
        base_manifest_path,
        expected_digest=expected_base_manifest_sha256,
    )
    if len(base.get("models") or []) != 14:
        raise RoutedManifestError("base frontier manifest must contain exactly 14 models")
    kimi_catalog = _verified_artifact(kimi_catalog_path)
    kimi_compatibility = _verified_artifact(kimi_compatibility_path)
    bedrock_catalog = _verified_artifact(bedrock_catalog_path, digest_field="catalog_sha256")
    bedrock_catalog_region = str(bedrock_catalog.get("region") or "")
    if (
        bedrock_catalog.get("schema_version") != "flavourbench-bedrock-catalog-v2"
        or not BEDROCK_PHYSICAL_REGION_PATTERN.fullmatch(bedrock_catalog_region)
    ):
        raise RoutedManifestError("unexpected Bedrock catalog scope")
    targets = {
        str(target.get("target_id") or "")
        for target in bedrock_catalog.get("targets") or []
        if isinstance(target, Mapping)
    }
    if not set(BEDROCK_EXACT_TARGETS.values()).issubset(targets):
        raise RoutedManifestError("Bedrock catalog omits an expected exact Claude 5 target")
    access_receipts, access_receipt_metadata = _access_denial_receipts(
        unified_path=bedrock_access_receipt_path,
        fable_path=bedrock_fable_receipt_path,
        claude_eu_path=bedrock_claude_eu_receipt_path,
        expected_bedrock_catalog_sha256=str(bedrock_catalog["catalog_sha256"]),
    )

    entries: list[dict[str, Any]] = []
    for source_entry in base["models"]:
        model_id = str(source_entry.get("model", {}).get("id") or "")
        if model_id == KIMI_MODEL_ID:
            entry = _kimi_entry(
                source_entry,
                catalog=kimi_catalog,
                compatibility=kimi_compatibility,
            )
        else:
            entry = _openrouter_fallback_entry(
                source_entry,
                bedrock_catalog_sha256=bedrock_catalog["catalog_sha256"],
                claude_access_receipts=access_receipts,
            )
        entries.append(entry)
    # Put the independently routed lane first so one real direct-Kimi pair is
    # proven before any fallback route can spend. Model order is frozen here,
    # before quality observations, and carries no ranking meaning.
    entries.sort(key=lambda item: str(item["model"]["id"]) != KIMI_MODEL_ID)
    requested_targets = tuple(str(model_id) for model_id in target_model_ids)
    if requested_targets:
        if len(requested_targets) != len(set(requested_targets)) or any(
            not model_id for model_id in requested_targets
        ):
            raise RoutedManifestError("target model IDs must be non-empty and unique")
        available_model_ids = {str(entry["model"]["id"]) for entry in entries}
        unknown = set(requested_targets) - available_model_ids
        if unknown:
            raise RoutedManifestError(
                f"target model IDs are absent from the frozen panel: {sorted(unknown)}"
            )
        target_set = set(requested_targets)
        entries = [
            entry for entry in entries if str(entry["model"]["id"]) in target_set
        ]

    design = base.get("run_design")
    if not isinstance(design, Mapping):
        raise RoutedManifestError("base manifest has no run design")
    tasks_per_family = int(design.get("tasks_per_family_in_pool") or 0)
    assignments_per_model = int(design.get("assignments_per_model") or 0)
    task_inventory, task_source = load_development_task_inventory(task_validity_path)
    selected_tasks, registry_sha = select_balanced_tasks(
        tasks_per_family=tasks_per_family,
        seed=str(design.get("selection_seed") or SELECTION_SEED),
        tasks=task_inventory,
    )
    candidates = [_candidate(entry) for entry in entries]
    total_forecast = Decimal(0)
    model_forecasts: dict[str, Decimal] = {}
    for candidate in candidates:
        candidate_total = Decimal(0)
        for task in selected_tasks[:assignments_per_model]:
            placeholder = WorkItem(
                ordinal=0,
                work_item_id="0" * 64,
                manifest_sha256="0" * 64,
                task_registry_sha256=task_registry_sha256(task_inventory),
                task=task,
                candidate=candidate,
                endpoint_execution_sha256="0" * 64,
                execution_policy_sha256=execution_policy.sha256,
                execution_policy=execution_policy,
            )
            candidate_total += derive_pair_forecast(
                placeholder,
                policy=execution_policy,
            ).forecast_usd
        model_forecasts[candidate.model_id] = candidate_total
        total_forecast += candidate_total
    admission_ceiling = cap_usd * Decimal("0.85")
    if total_forecast > admission_ceiling:
        raise RoutedManifestError("routed workload exceeds the 85 percent admission ceiling")
    for entry in entries:
        model_id = str(entry["model"]["id"])
        entry["forecast"] = {
            "model_block_worst_case_usd": format(model_forecasts[model_id], "f"),
            "pairs": assignments_per_model,
            "conditions_per_pair": 2,
        }

    payload = copy.deepcopy(base)
    payload.pop("content_address", None)
    payload["schema_version"] = SCHEMA_VERSION
    payload["manifest_role"] = MANIFEST_ROLE
    payload["source"] = {
        **dict(payload.get("source") or {}),
        "basis_openrouter_manifest_sha256": expected_base_manifest_sha256,
        "task_validity_artifact_sha256": task_source["artifact_sha256"],
        "task_candidate_coordinate_sha256": task_source["candidate_coordinate_sha256"],
        "task_registry_sha256": registry_sha,
        "kimi_catalog_sha256": kimi_catalog["artifact_sha256"],
        "kimi_contract_smoke_sha256": kimi_compatibility["artifact_sha256"],
        "bedrock_catalog_sha256": bedrock_catalog["catalog_sha256"],
        "bedrock_catalog_region": bedrock_catalog_region,
        "bedrock_access_denial_receipts": access_receipt_metadata,
    }
    backend_counts = {
        "kimi_direct": sum(
            entry["execution_route"]["selected_backend"] == "kimi_direct"
            for entry in entries
        ),
        "bedrock": sum(
            entry["execution_route"]["selected_backend"] == "bedrock"
            for entry in entries
        ),
        "openrouter_fallback": sum(
            entry["execution_route"]["selected_backend"] == "openrouter"
            for entry in entries
        ),
    }
    payload["selection"] = {
        **dict(payload.get("selection") or {}),
        "method": (
            "direct Kimi for K3; exact Bedrock where present and invocable; "
            "predeclared exact OpenRouter fallback otherwise"
        ),
        "model_count": len(entries),
        "quality_observations_used": 0,
        "route_counts": backend_counts,
        "targeting": {
            "method": (
                "operator-frozen operational completion-floor replenishment"
                if requested_targets
                else "complete frozen frontier panel"
            ),
            "selected_model_ids": [str(entry["model"]["id"]) for entry in entries],
            "quality_outcomes_used": False,
        },
    }
    protocol = {
        **dict(design.get("generation_protocol") or {}),
        "schema_version": "flavourbench-live-development-protocol-v10",
        "builder": "flavourbench.live_smoke.build_live_protocol_bundle",
        "final_response_mode": execution_policy.final_response_mode,
        "matched_planning": execution_policy.matched_planning,
        "max_intermediate_tokens": execution_policy.max_intermediate_tokens,
        "required_tool_contract_max_intermediate_tokens": (
            execution_policy.required_tool_contract_max_intermediate_tokens
        ),
        "evidence_protocol": execution_policy.evidence_protocol,
        "required_tool_contract_protocol": execution_policy.required_tool_contract_protocol,
        "required_tool_contract": required_tool_contract(execution_policy),
        "required_tool_contract_sha256": required_tool_contract(execution_policy)[
            "content_address"
        ]["digest"],
        "intermediate_reasoning_effort": execution_policy.intermediate_reasoning_effort,
        "final_reasoning_effort": execution_policy.final_reasoning_effort,
        "tool_catalog_bytes_bound": execution_policy.tool_catalog_bytes_bound,
        "epicure_on_tool_required": execution_policy.epicure_on_tool_required,
        "provider_route_binding": "selected_backend_and_backend_contract_sha256",
    }
    payload["run_design"] = {
        **dict(design),
        "selected_task_count": len(selected_tasks),
        "expected_pairs": len(entries) * assignments_per_model,
        "expected_arms": len(entries) * assignments_per_model * 2,
        "execution_policy": execution_policy.document(),
        "execution_policy_sha256": execution_policy.sha256,
        "generation_protocol": protocol,
    }
    payload["budget"] = {
        "currency": "USD",
        "cap_usd": format(cap_usd, "f"),
        "admission_fraction": "0.85",
        "admission_ceiling_usd": format(admission_ceiling, "f"),
        "bounded_forecast_usd": format(total_forecast, "f"),
        "headroom_to_admission_ceiling_usd": format(admission_ceiling - total_forecast, "f"),
        "within_cap": True,
        "forecast_policy": "multi_backend_pair_reservation_v1",
    }
    payload["models"] = entries
    payload["routing_policy"] = {
        "schema_version": "flavourbench-provider-precedence-v1",
        "precedence": ["kimi_direct_for_kimi_k3", "exact_bedrock", "openrouter_fallback"],
        "resolved_before_generation": True,
        "generation_time_automatic_fallback": False,
        "provider_substitution": "prohibited",
        "openrouter_request_controls": {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "one_exact_provider_endpoint": True,
        },
        "bedrock_result": (
            "No exact model in this 14-model slate was both present and invocable with the "
            "configured account at freeze time. Claude 5 routes were present but denied; "
            "the remaining non-Kimi exact releases were absent."
        ),
        "bedrock_catalog_region": bedrock_catalog_region,
    }
    payload["governance"] = {
        **dict(payload.get("governance") or {}),
        "manifest_class": "real_multi_backend_development_quality_run_candidate",
        "freeze_status": "logical_models_provider_routes_and_workload_frozen_before_generation",
        "data_policy": (
            "direct Kimi or one frozen provider endpoint; no generation-time fallback; "
            "OpenRouter only after immutable Bedrock preflight evidence"
        ),
        "official": False,
        "rank_eligible": False,
    }
    digest = sha256_json(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(payload):
        raise RoutedManifestError("internal routed manifest content address failed")
    return payload


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if not verify_manifest_content_address(payload):
        raise RoutedManifestError("refusing an invalid routed manifest")
    digest = str(payload["content_address"]["digest"])
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"flavourbench-routed-unranked-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise RoutedManifestError("content-addressed output conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--kimi-catalog", type=Path, required=True)
    parser.add_argument("--kimi-compatibility", type=Path, required=True)
    parser.add_argument("--bedrock-catalog", type=Path, required=True)
    receipt_group = parser.add_mutually_exclusive_group(required=True)
    receipt_group.add_argument("--bedrock-access-receipt", type=Path)
    receipt_group.add_argument("--legacy-split-bedrock-receipts", action="store_true")
    parser.add_argument("--bedrock-fable-receipt", type=Path)
    parser.add_argument("--bedrock-claude-eu-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Freeze only this exact model ID for an adaptive replenishment block; "
            "repeat for multiple models. The default freezes the complete panel."
        ),
    )
    parser.add_argument("--cap-usd", type=Decimal, default=DEFAULT_CAP_USD)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-intermediate-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    parser.add_argument("--max-cumulative-tool-result-bytes", type=int, default=98_304)
    parser.add_argument("--max-tool-calls-per-round", type=int, default=16)
    parser.add_argument("--max-tool-calls-total", type=int, default=32)
    parser.add_argument("--max-provider-attempts", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tool-catalog-bytes-bound", type=int, default=24_000)
    args = parser.parse_args(argv)
    policy = ExecutionPolicy(
        max_output_tokens=args.max_output_tokens,
        max_intermediate_tokens=args.max_intermediate_tokens,
        max_tool_rounds=args.max_tool_rounds,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        max_tool_calls_total=args.max_tool_calls_total,
        max_cumulative_tool_result_bytes=args.max_cumulative_tool_result_bytes,
        max_provider_attempts=args.max_provider_attempts,
        decoding_temperature=1.0,
        decoding_top_p=0.95,
        decoding_seed=20260715,
        final_response_mode="plain_text",
        matched_planning=True,
        evidence_protocol=MATCHED_EVIDENCE_PROTOCOL_V2,
        intermediate_reasoning_effort="low",
        final_reasoning_effort="low",
        tool_catalog_bytes_bound=args.tool_catalog_bytes_bound,
        epicure_on_tool_required=True,
    )
    manifest = build_manifest(
        base_manifest_path=args.base_manifest,
        expected_base_manifest_sha256=args.expected_base_manifest_sha256,
        task_validity_path=args.task_validity,
        kimi_catalog_path=args.kimi_catalog,
        kimi_compatibility_path=args.kimi_compatibility,
        bedrock_catalog_path=args.bedrock_catalog,
        bedrock_fable_receipt_path=args.bedrock_fable_receipt,
        bedrock_claude_eu_receipt_path=args.bedrock_claude_eu_receipt,
        bedrock_access_receipt_path=args.bedrock_access_receipt,
        cap_usd=args.cap_usd,
        execution_policy=policy,
        target_model_ids=args.model,
    )
    path = _write(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "manifest_sha256": manifest["content_address"]["digest"],
                "models": manifest["selection"]["model_count"],
                "route_counts": manifest["selection"]["route_counts"],
                "budget": manifest["budget"],
                "provider_calls_made": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
