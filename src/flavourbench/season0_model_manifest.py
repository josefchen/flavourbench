"""Freeze the exact Season 0 models after real provider-and-Epicure smokes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json, sha256_text

SCHEMA_VERSION = "flavourbench-season0-model-manifest-v1"
EXPECTED_ROLES = {
    "closed_family": 4,
    "open_weight": 4,
    "efficiency": 2,
    "reasoning": 2,
}


class ModelManifestError(RuntimeError):
    """The roster could not be frozen into a 12-model Season 0 manifest."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ModelManifestError(f"expected a JSON object: {path}")
    return value


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise ModelManifestError("content-addressed model-manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _artifact_body_sha(document: Mapping[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return sha256_json(body)


def _verified_summary_artifacts(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = summary.get("artifacts")
    if not isinstance(entries, list):
        raise ModelManifestError("compatibility summary has no artifacts")
    output: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("status") != "smoke_passed":
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            raise ModelManifestError("compatibility entry has no artifact path")
        document = _load(Path(path_value))
        claimed = document.get("artifact_sha256")
        if not isinstance(claimed, str) or _artifact_body_sha(document) != claimed:
            raise ModelManifestError(f"compatibility artifact hash is invalid: {path_value}")
        identifier = document.get("requested_target_id") or document.get("requested_model_id")
        if not isinstance(identifier, str) or not identifier:
            raise ModelManifestError("compatibility artifact has no requested identifier")
        output[identifier] = document
    return output


def build_model_manifest(
    *,
    roster: Mapping[str, Any],
    bedrock_catalog: Mapping[str, Any],
    bedrock_compatibility: Mapping[str, Any],
    openrouter_routes: Mapping[str, Any],
    openrouter_compatibility: Mapping[str, Any],
    task_bank: Mapping[str, Any],
    epicure_intervention: Mapping[str, Any],
) -> dict[str, Any]:
    slots = roster.get("slots")
    targets = bedrock_catalog.get("targets")
    routes = openrouter_routes.get("routes")
    if not isinstance(slots, list) or not isinstance(targets, list) or not isinstance(routes, list):
        raise ModelManifestError("roster, Bedrock targets, or OpenRouter routes are invalid")
    if task_bank.get("synthetic_tasks") != 0 or task_bank.get("counts", {}).get("total") != 120:
        raise ModelManifestError("Season 0 model freeze requires the 120-task real-data bank")
    if epicure_intervention.get("intervention_id") != "epicure-mcp-opaque-1790-v1":
        raise ModelManifestError("unexpected Epicure intervention")
    bedrock_by_id = {
        str(target["target_id"]): target
        for target in targets
        if isinstance(target, Mapping) and isinstance(target.get("target_id"), str)
    }
    openrouter_by_id = {
        str(route["model_id"]): route
        for route in routes
        if isinstance(route, Mapping) and isinstance(route.get("model_id"), str)
    }
    bedrock_smokes = _verified_summary_artifacts(bedrock_compatibility)
    openrouter_smokes = _verified_summary_artifacts(openrouter_compatibility)

    models: list[dict[str, Any]] = []
    for index, slot in enumerate(slots, start=1):
        if not isinstance(slot, Mapping):
            raise ModelManifestError("roster contains a non-object slot")
        provider = str(slot.get("provider") or "")
        endpoint_id = str(slot.get("endpoint_id") or "")
        common = {
            "season_model_id": f"fb-s0-model-{index:02d}",
            "slot_role": str(slot.get("slot_role") or ""),
            "display_name": str(slot.get("canonical_name") or ""),
            "provider": provider,
            "requested_endpoint_id": endpoint_id,
            "fallbacks_allowed": False,
            "provider_substitution_rank_eligible": False,
        }
        if provider == "bedrock":
            target = bedrock_by_id.get(endpoint_id)
            smoke = bedrock_smokes.get(endpoint_id)
            if target is None or smoke is None:
                raise ModelManifestError(f"Bedrock slot lacks a passing smoke: {endpoint_id}")
            models.append(
                {
                    **common,
                    "canonical_model_id": str(target["foundation_model_ids"][0]),
                    "canonical_slug": str(target["foundation_model_ids"][0]),
                    "provider_name": str(target.get("provider_name") or "Amazon Bedrock"),
                    "endpoint": dict(target),
                    "compatibility_artifact_sha256": smoke["artifact_sha256"],
                    "compatibility_real_provider_calls": int(smoke["provider_calls"]),
                    "compatibility_real_epicure_calls": int(smoke["real_epicure_calls"]),
                    "cost_accounting": "aws_usage_tokens_plus_frozen_rate_card_estimate",
                }
            )
        elif provider == "openrouter":
            route = openrouter_by_id.get(endpoint_id)
            smoke = openrouter_smokes.get(endpoint_id)
            if route is None or smoke is None:
                raise ModelManifestError(f"OpenRouter slot lacks a passing smoke: {endpoint_id}")
            if route.get("provider_slug") != slot.get("provider_slug"):
                raise ModelManifestError(f"OpenRouter provider route drift: {endpoint_id}")
            if route.get("canonical_slug") != slot.get("canonical_slug"):
                raise ModelManifestError(f"OpenRouter canonical model drift: {endpoint_id}")
            models.append(
                {
                    **common,
                    "canonical_model_id": str(route["canonical_slug"]),
                    "canonical_slug": str(route["canonical_slug"]),
                    "provider_name": str(route["endpoint"].get("provider_name") or ""),
                    "provider_slug": str(route["provider_slug"]),
                    "endpoint": dict(route["endpoint"]),
                    "endpoint_document_sha256": str(route["endpoint_document_sha256"]),
                    "compatibility_artifact_sha256": smoke["artifact_sha256"],
                    "compatibility_real_provider_calls": int(smoke["real_provider_calls"]),
                    "compatibility_real_epicure_calls": int(smoke["real_epicure_calls"]),
                    "compatibility_cost_usd": str(smoke["cost_usd"]),
                    "cost_accounting": "openrouter_generation_metadata_reconciled",
                }
            )
        else:
            raise ModelManifestError(f"unsupported Season 0 provider: {provider}")

    if len(models) != 12:
        raise ModelManifestError(f"Season 0 requires 12 exact models, found {len(models)}")
    if len({model["canonical_model_id"] for model in models}) != 12:
        raise ModelManifestError("Season 0 canonical model identities are not unique")
    roles = Counter(model["slot_role"] for model in models)
    if roles != Counter(EXPECTED_ROLES):
        raise ModelManifestError(f"Season 0 slot roles differ from the frozen design: {roles}")
    if Counter(model["provider"] for model in models) != Counter({"bedrock": 7, "openrouter": 5}):
        raise ModelManifestError("Season 0 provider allocation must be seven Bedrock and five OR")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "season": "Season 0",
        "status": "frozen_compatibility_passed_cost_calibration_pending",
        "official_score_eligibility": "pending_real_cost_calibration_and_pi_task_audit",
        "task_set_sha256": str(task_bank["task_set_sha256"]),
        "task_bank_artifact_sha256": str(task_bank["artifact_sha256"]),
        "epicure_intervention_id": str(epicure_intervention["intervention_id"]),
        "epicure_intervention_artifact_sha256": str(epicure_intervention["artifact_sha256"]),
        "roster_sha256": sha256_json(roster),
        "bedrock_catalog_sha256": str(bedrock_catalog.get("catalog_sha256") or ""),
        "openrouter_route_catalog_sha256": str(openrouter_routes["artifact_sha256"]),
        "execution_contract": {
            "conditions": ["epicure_off", "epicure_on"],
            "normalization_mode": "lossless_client_text_wrapper_v1",
            "provider_structured_output_required": False,
            "final_answer_max_tokens": 2_048,
            "max_tool_rounds": 8,
            "max_tool_calls_per_round": 8,
            "max_tool_calls_total": 32,
            "max_tool_result_bytes": 4_096,
            "max_cumulative_tool_result_bytes": 131_072,
            "temperature": "0.2_when_supported_otherwise_frozen_provider_default",
            "provider_retries": "at_most_two_only_after_explicit_pre_inference_rejection",
            "read_timeout_retry": False,
            "same_prompt_and_answer_limit_across_conditions": True,
            "complete_tool_trace_required": True,
        },
        "budget_contract": {
            "bedrock_hard_cap_usd": "5000",
            "openrouter_hard_cap_usd": "100",
            "admission_stop_fraction": "0.85",
            "drain_fraction": "0.95",
            "cost_calibration": (
                "four real task-family-stratified off/on pairs per model; excluded from scoring; "
                "dense-run reservations use twice the maximum calibrated per-arm cost"
            ),
        },
        "counts": {
            "models": 12,
            "bedrock": 7,
            "openrouter": 5,
            "compatibility_real_epicure_calls": sum(
                int(model["compatibility_real_epicure_calls"]) for model in models
            ),
            "synthetic_models": 0,
            "placeholder_models": 0,
        },
        "model_set_sha256": sha256_json(
            [
                {
                    "season_model_id": model["season_model_id"],
                    "canonical_model_id": model["canonical_model_id"],
                    "provider": model["provider"],
                    "requested_endpoint_id": model["requested_endpoint_id"],
                    "provider_slug": model.get("provider_slug"),
                    "compatibility_artifact_sha256": model["compatibility_artifact_sha256"],
                }
                for model in models
            ]
        ),
        "models": models,
        "manifest_label_sha256": sha256_text("FlavourBench Season 0 exact 12-model panel"),
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--bedrock-catalog", type=Path, required=True)
    parser.add_argument("--bedrock-compatibility", type=Path, required=True)
    parser.add_argument("--openrouter-routes", type=Path, required=True)
    parser.add_argument("--openrouter-compatibility", type=Path, required=True)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--epicure-intervention", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season0/manifests"))
    args = parser.parse_args(argv)
    manifest = build_model_manifest(
        roster=_load(args.roster),
        bedrock_catalog=_load(args.bedrock_catalog),
        bedrock_compatibility=_load(args.bedrock_compatibility),
        openrouter_routes=_load(args.openrouter_routes),
        openrouter_compatibility=_load(args.openrouter_compatibility),
        task_bank=_load(args.task_bank),
        epicure_intervention=_load(args.epicure_intervention),
    )
    path = _atomic_write(args.output_dir, "season0-model-manifest", manifest)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "model_set_sha256": manifest["model_set_sha256"],
                "counts": manifest["counts"],
                "models": [
                    {
                        "season_model_id": model["season_model_id"],
                        "display_name": model["display_name"],
                        "canonical_model_id": model["canonical_model_id"],
                        "provider": model["provider"],
                    }
                    for model in manifest["models"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
