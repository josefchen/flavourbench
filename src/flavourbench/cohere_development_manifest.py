"""Freeze exact Cohere routes for the real FlavourBench development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cohere_compatibility import (
    ARM_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
)
from .frontier_manifest import verify_manifest_content_address
from .real_task_bank import sha256_json
from .service_cohere import COHERE_REQUIRED_TOOL_INSTRUCTION

COHERE_MODEL_SPECS = (
    {
        "logical_model_id": "cohere/command-a-plus-05-2026",
        "requested_model_id": "command-a-plus-05-2026",
        "display_name": "Cohere Command A Plus",
        "slot_id": "current-frontier-15",
        "max_completion_tokens": 64_000,
        "release_date": "2026-05-20",
        "model_card": "https://docs.cohere.com/docs/command-a-plus",
    },
    {
        "logical_model_id": "cohere/command-a-reasoning-08-2025",
        "requested_model_id": "command-a-reasoning-08-2025",
        "display_name": "Cohere Command A Reasoning",
        "slot_id": "current-frontier-16",
        "max_completion_tokens": 32_000,
        "release_date": "2025-08-21",
        "model_card": "https://docs.cohere.com/docs/command-a-reasoning",
    },
)


class CohereManifestError(RuntimeError):
    """A Cohere catalog, receipt, or frozen manifest failed verification."""


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CohereManifestError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CohereManifestError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise CohereManifestError(f"expected JSON object: {path}")
    return value


def _verified_artifact(path: Path, schema_version: str) -> dict[str, Any]:
    value = _json(path)
    digest = str(value.get("artifact_sha256") or "")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if (
        value.get("schema_version") != schema_version
        or len(digest) != 64
        or sha256_json(unhashed) != digest
        or digest not in path.name
    ):
        raise CohereManifestError(f"content address does not verify: {path}")
    return value


def _entry(
    spec: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    receipt: Mapping[str, Any],
    assignments_per_model: int,
) -> dict[str, Any]:
    requested = str(spec["requested_model_id"])
    models = catalog.get("models")
    catalog_entry = next(
        (
            dict(item)
            for item in models or []
            if isinstance(item, Mapping) and item.get("name") == requested
        ),
        None,
    )
    if catalog_entry is None:
        raise CohereManifestError(f"authenticated catalog omits {requested}")
    catalog_sha = str(catalog.get("artifact_sha256") or "")
    entry_sha = sha256_json(catalog_entry)
    if (
        receipt.get("status") != "smoke_passed"
        or receipt.get("requested_model_id") != requested
        or receipt.get("catalog_sha256") != catalog_sha
        or receipt.get("catalog_entry_sha256") != entry_sha
        or int(receipt.get("real_provider_calls") or 0) < 2
        or int(receipt.get("real_epicure_calls") or 0) < 1
        or receipt.get("official") is not False
        or receipt.get("rank_eligible") is not False
    ):
        raise CohereManifestError(f"live Cohere receipt does not bind {requested}")

    logical_id = str(spec["logical_model_id"])
    receipt_sha = str(receipt["artifact_sha256"])
    endpoint = {
        "context_length": int(catalog_entry["context_length"]),
        "max_completion_tokens": int(spec["max_completion_tokens"]),
        "model_id": logical_id,
        "pricing": {
            "prompt": "0",
            "completion": "0",
            "internal_reasoning": "0",
            "request": "0",
            "currency": "USD",
            "source": spec["model_card"],
            "status": "frozen_public_api_free_until_rate_limit",
        },
        "provider_name": "cohere-direct",
        "quantization": "provider_managed_unpublished",
        "supported_parameters": [
            "max_tokens",
            "reasoning",
            "response_format",
            "seed",
            "structured_outputs",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
        "tag": "cohere-direct",
    }
    backend_contract = {
        "schema_version": "flavourbench-cohere-direct-endpoint-contract-v1",
        "allow_fallbacks": False,
        "base_url": "https://api.cohere.com",
        "catalog_sha256": catalog_sha,
        "catalog_entry_sha256": entry_sha,
        "compatibility_artifact_sha256": receipt_sha,
        "cost_accounting": "provider_usage_times_frozen_rate_card",
        "expected_actual_provider_slug": "cohere-direct",
        "identity_evidence": "authenticated_catalog_exact_request_and_generation_id",
        "message_translation": "openai_internal_protocol_to_cohere_chat_v2",
        "assistant_tool_history": "tool_calls_without_command_a_unsupported_tool_plan",
        "native_tool_choice": "omitted_command_a_rejects_parameter",
        "reasoning_parameter_translation": "effort_to_thinking_token_budget",
        "required_tool_enforcement": (
            "explicit_transport_instruction_plus_client_side_successful_trace_validation"
        ),
        "required_tool_instruction_sha256": hashlib.sha256(
            COHERE_REQUIRED_TOOL_INSTRUCTION.encode()
        ).hexdigest(),
        "requested_model_id": requested,
        "season_eligible": False,
        "tool_schema_projection": "documented_cohere_json_schema_subset",
        "tool_choice_translation": (
            "omit_provider_parameter_and_validate_required_trace_client_side"
        ),
    }
    return {
        "slot": {
            "slot_id": spec["slot_id"],
            "model_id": logical_id,
            "cohort": "current_frontier_development",
            "open_weight_candidate": True if "plus" in requested else None,
            "rationale": (
                "Exact direct Cohere route passed structured-final and real Epicure "
                "contract checks; inclusion is not a quality claim."
            ),
        },
        "model": {
            "id": logical_id,
            "canonical_slug": requested,
            "name": spec["display_name"],
            "description": (
                f"Exact Cohere Chat V2 release {requested}, frozen from the authenticated "
                "model catalog."
            ),
            "context_length": int(catalog_entry["context_length"]),
            "release_date": spec["release_date"],
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "modality": "text->text",
            },
            "reasoning": {
                "default_enabled": True,
                "mandatory": False,
                "supported_efforts": ["low", "medium", "high", "max"],
            },
            "supported_parameters": list(endpoint["supported_parameters"]),
            "catalog_entry": catalog_entry,
        },
        "endpoint": endpoint,
        "endpoint_document_sha256": sha256_json(endpoint),
        "endpoint_selection": {
            "method": "exact authenticated Cohere model frozen before collection",
            "quality_observations_used": 0,
            "selected_exact_tag": "cohere-direct",
        },
        "request_policy": {
            "official_eligibility": "development_only",
            "policy_scope": "request_enforced",
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "only": ["cohere-direct"],
            },
        },
        "execution_route": {
            "policy": "exact_direct_cohere_v1",
            "preferred_backend": "cohere_direct",
            "selected_backend": "cohere_direct",
            "selection_frozen_before_generation": True,
            "selection_reason": "exact_direct_cohere_contract_passed",
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
            "evidence": {
                "catalog_sha256": catalog_sha,
                "compatibility_artifact_sha256": receipt_sha,
            },
        },
        "backend_contract": backend_contract,
        "backend_contract_sha256": sha256_json(backend_contract),
        "cost_accounting_policy": "provider_usage_times_frozen_rate_card",
        "contract_evidence": {
            "actual_provider": "cohere-direct",
            "contract_status": "passed_unranked",
            "cost_status": "public_free_rate_card_provider_charge_unavailable",
            "identity_basis": "authenticated_catalog_exact_request_and_generation_id",
            "real_provider_calls": int(receipt["real_provider_calls"]),
            "real_epicure_calls": int(receipt["real_epicure_calls"]),
            "source_artifact_sha256": receipt_sha,
        },
        "forecast": {
            "pairs": assignments_per_model,
            "conditions_per_pair": 2,
            "model_block_worst_case_usd": "0",
        },
    }


def build_manifest(
    *,
    base_manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    selection_seed: str,
    manifest_amendment_id: str | None = None,
) -> dict[str, Any]:
    if (
        not verify_manifest_content_address(base_manifest)
        or base_manifest.get("manifest_role")
        != "current_frontier_routed_development_quality_run"
        or base_manifest.get("official_results_authorised") is not False
    ):
        raise CohereManifestError("base development manifest does not verify")
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise CohereManifestError("Cohere catalog schema is unsupported")
    if len(receipts) != len(COHERE_MODEL_SPECS):
        raise CohereManifestError("exactly two Cohere receipts are required")
    by_model = {str(item.get("requested_model_id")): item for item in receipts}
    if set(by_model) != {str(item["requested_model_id"]) for item in COHERE_MODEL_SPECS}:
        raise CohereManifestError("Cohere receipt membership changed")

    run_design = deepcopy(base_manifest["run_design"])
    assignments = int(run_design["assignments_per_model"])
    model_entries = [
        _entry(
            spec,
            catalog=catalog,
            receipt=by_model[str(spec["requested_model_id"])],
            assignments_per_model=assignments,
        )
        for spec in COHERE_MODEL_SPECS
    ]
    run_design.update(
        {
            "selection_seed": selection_seed,
            "expected_pairs": assignments * len(model_entries),
            "expected_arms": assignments * len(model_entries) * 2,
        }
    )
    source = {
        key: deepcopy(value)
        for key, value in base_manifest["source"].items()
        if key.startswith("task_")
    }
    source.update(
        {
            "basis_manifest_sha256": base_manifest["content_address"]["digest"],
            "cohere_catalog_sha256": catalog["artifact_sha256"],
            "cohere_contract_smoke_sha256s": {
                model: receipt["artifact_sha256"] for model, receipt in by_model.items()
            },
        }
    )
    if manifest_amendment_id is not None:
        if not manifest_amendment_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in manifest_amendment_id
        ):
            raise CohereManifestError("manifest amendment ID is malformed")
        source["manifest_amendment_id"] = manifest_amendment_id
    payload: dict[str, Any] = {
        "schema_version": "flavourbench-routed-candidate-manifest-v1",
        "status": "unranked_candidate",
        "manifest_role": "current_frontier_routed_development_quality_run",
        "official_results_authorised": False,
        "generation_calls_made": 0,
        "generation_spend_usd": "0",
        "observed_at": "2026-08-03T00:00:00Z",
        "source": source,
        "selection": {
            "method": "exact direct Cohere routes with no provider fallback",
            "model_count": len(model_entries),
            "performance_claim": "none; inclusion is coverage, not a ranking",
            "quality_observations_used": 0,
            "route_counts": {
                "kimi_direct": 0,
                "bedrock": 0,
                "openrouter_fallback": 0,
                "cohere_direct": len(model_entries),
            },
        },
        "run_design": run_design,
        "models": model_entries,
        "routing_policy": {
            "schema_version": "flavourbench-provider-precedence-v1",
            "precedence": ["exact_direct_cohere"],
            "resolved_before_generation": True,
            "generation_time_automatic_fallback": False,
            "provider_substitution": "prohibited",
            "cohere_request_controls": {
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "one_exact_provider_endpoint": True,
            },
        },
        "budget": {
            "currency": "USD",
            "cap_usd": "100",
            "admission_fraction": "0.85",
            "admission_ceiling_usd": "85.00",
            "bounded_forecast_usd": "0",
            "headroom_to_admission_ceiling_usd": "85.00",
            "within_cap": True,
            "forecast_policy": "cohere_public_free_rate_card_pair_reservation_v1",
        },
        "governance": {
            "manifest_class": "real_direct_cohere_development_quality_run_candidate",
            "freeze_status": "exact_models_routes_and_workload_frozen_before_generation",
            "data_policy": "one exact direct Cohere endpoint; no fallback or substitution",
            "official": False,
            "rank_eligible": False,
            "funding_disclosure_required": True,
        },
    }
    digest = sha256_json(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(payload):
        raise CohereManifestError("internal manifest content address failed")
    return payload


def _write(output_dir: Path, manifest: Mapping[str, Any]) -> Path:
    digest = str(manifest["content_address"]["digest"])
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"flavourbench-cohere-unranked-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise CohereManifestError("content-addressed output conflict")
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
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--command-a-plus-receipt", type=Path, required=True)
    parser.add_argument("--command-a-reasoning-receipt", type=Path, required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--manifest-amendment-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    base = _json(args.base_manifest)
    catalog = _verified_artifact(args.catalog, CATALOG_SCHEMA_VERSION)
    receipts = (
        _verified_artifact(args.command_a_plus_receipt, ARM_SCHEMA_VERSION),
        _verified_artifact(args.command_a_reasoning_receipt, ARM_SCHEMA_VERSION),
    )
    manifest = build_manifest(
        base_manifest=base,
        catalog=catalog,
        receipts=receipts,
        selection_seed=args.selection_seed,
        manifest_amendment_id=args.manifest_amendment_id,
    )
    path = _write(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path.resolve()),
                "manifest_sha256": manifest["content_address"]["digest"],
                "models": len(manifest["models"]),
                "generation_calls_made": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
