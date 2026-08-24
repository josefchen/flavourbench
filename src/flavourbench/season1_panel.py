"""Verify real provider/Epicure contract receipts and freeze a Season 1 panel.

This command performs no model or MCP calls. It accepts an explicit candidate
specification, verifies every content-addressed live receipt, and writes one
content-addressed 16-endpoint panel manifest. Contract qualification is not a
season freeze and does not make any response rank eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .season_design import SEASON_MODEL_COUNT, SEASON_SLOT_ROLE_COUNTS

SPEC_SCHEMA_VERSION = "flavourbench-season1-panel-candidate-v1"
MANIFEST_SCHEMA_VERSION = "flavourbench-season1-contract-panel-v1"


class Season1PanelError(RuntimeError):
    """The prospective panel or one of its live receipts was invalid."""


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Season1PanelError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise Season1PanelError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise Season1PanelError(f"expected a JSON object: {path}")
    return value


def _verified_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    document = _load_object(path)
    recorded = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    actual = sha256_json(payload)
    if recorded != actual or expected_sha256 != actual:
        raise Season1PanelError(f"content address does not verify: {path}")
    return document


def _non_error_trace(artifact: Mapping[str, Any], model_id: str) -> list[Mapping[str, Any]]:
    trace = artifact.get("complete_epicure_trace")
    if (
        not isinstance(trace, list)
        or len(trace) != 1
        or not isinstance(trace[0], Mapping)
        or trace[0].get("is_error") is not False
        or trace[0].get("name") != "find_pairings"
        or not str(trace[0].get("result_sha256") or "")
    ):
        raise Season1PanelError(f"{model_id} lacks one successful real Epicure trace")
    return trace


def _verify_bedrock(
    entry: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = str(entry["model_id"])
    if (
        artifact.get("requested_target_id") != model_id
        or int(artifact.get("provider_calls") or 0) != 2
        or int(artifact.get("real_epicure_calls") or 0) != 1
    ):
        raise Season1PanelError(f"{model_id} Bedrock call contract does not verify")
    request_ids = artifact.get("request_id_sha256s")
    foundation_ids = artifact.get("expected_foundation_model_ids")
    if (
        not isinstance(request_ids, list)
        or len(request_ids) != 2
        or len(set(request_ids)) != 2
        or any(not isinstance(value, str) or len(value) != 64 for value in request_ids)
        or not isinstance(foundation_ids, list)
        or not foundation_ids
        or any(not isinstance(value, str) or not value for value in foundation_ids)
    ):
        raise Season1PanelError(f"{model_id} Bedrock identity evidence is incomplete")
    expected_canonical = str(entry["canonical_slug"])
    if expected_canonical not in foundation_ids:
        raise Season1PanelError(f"{model_id} Bedrock foundation-model mapping drifted")
    return {
        "requested_endpoint_id": model_id,
        "canonical_model_slug": expected_canonical,
        "execution_backend": "bedrock",
        "provider_endpoint": "amazon-bedrock",
        "identity_basis": "frozen_inference_target_and_foundation_model_mapping",
        "request_identity_sha256s": request_ids,
        "catalog_sha256": str(artifact.get("catalog_sha256") or ""),
        "cost_usd": None,
        "cost_status": "usage_recorded_rate_card_pending_aws_billing_crosscheck",
    }


def _verify_openrouter(
    entry: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = str(entry["model_id"])
    canonical_slug = str(entry["canonical_slug"])
    provider_slug = str(entry["provider_slug"])
    output_json = artifact.get("output_json")
    if (
        artifact.get("requested_model_id") != model_id
        or artifact.get("canonical_slug") != canonical_slug
        or artifact.get("requested_provider_slug") != provider_slug
        or int(artifact.get("real_provider_calls") or 0) != 2
        or int(artifact.get("real_epicure_calls") or 0) != 1
        or artifact.get("finish_reason") != "stop"
        or not isinstance(output_json, Mapping)
        or not str(output_json.get("answer_markdown") or "").strip()
    ):
        raise Season1PanelError(f"{model_id} OpenRouter call contract does not verify")
    generation_ids = artifact.get("generation_ids")
    accounting = artifact.get("generation_accounting")
    returned_provider = str(artifact.get("returned_provider_name") or "")
    if (
        not isinstance(generation_ids, list)
        or len(generation_ids) != 2
        or len(set(generation_ids)) != 2
        or not isinstance(accounting, list)
        or len(accounting) != 2
        or not returned_provider
    ):
        raise Season1PanelError(f"{model_id} OpenRouter identity evidence is incomplete")
    accounted_ids: set[str] = set()
    accounted_cost = Decimal(0)
    for item in accounting:
        if (
            not isinstance(item, Mapping)
            or item.get("model") != canonical_slug
            or item.get("provider_name") != returned_provider
            or item.get("reconciled") is not True
        ):
            raise Season1PanelError(f"{model_id} OpenRouter accounting identity drifted")
        generation_id = str(item.get("generation_id") or "")
        accounted_ids.add(generation_id)
        accounted_cost += Decimal(str(item.get("total_cost_usd") or "0"))
    if accounted_ids != set(generation_ids):
        raise Season1PanelError(f"{model_id} OpenRouter accounting coverage is incomplete")
    recorded_cost = Decimal(str(artifact.get("cost_usd") or "0"))
    if accounted_cost != recorded_cost:
        raise Season1PanelError(f"{model_id} OpenRouter cost does not reconcile")
    return {
        "requested_endpoint_id": model_id,
        "canonical_model_slug": canonical_slug,
        "execution_backend": "openrouter",
        "provider_endpoint": provider_slug,
        "actual_provider": returned_provider,
        "identity_basis": "reconciled_generation_metadata_model_and_provider",
        "request_identity_sha256s": generation_ids,
        "endpoint_document_sha256": str(artifact.get("endpoint_document_sha256") or ""),
        "cost_usd": format(recorded_cost, "f"),
        "cost_status": "openrouter_generation_metadata_reconciled",
        "provider_structured_output_required": bool(
            artifact.get("provider_structured_output_required")
        ),
    }


def build_manifest(spec_path: Path) -> dict[str, Any]:
    spec = _load_object(spec_path)
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise Season1PanelError("unexpected Season 1 panel specification schema")
    entries = spec.get("models")
    if not isinstance(entries, list) or len(entries) != SEASON_MODEL_COUNT:
        raise Season1PanelError(f"Season 1 panel must contain exactly {SEASON_MODEL_COUNT} models")
    model_ids = [
        str(entry.get("model_id") or "") for entry in entries if isinstance(entry, Mapping)
    ]
    canonical_slugs = [
        str(entry.get("canonical_slug") or "") for entry in entries if isinstance(entry, Mapping)
    ]
    if (
        len(model_ids) != SEASON_MODEL_COUNT
        or len(set(model_ids)) != SEASON_MODEL_COUNT
        or len(set(canonical_slugs)) != SEASON_MODEL_COUNT
        or any(not value for value in model_ids + canonical_slugs)
    ):
        raise Season1PanelError("Season 1 model and canonical identities must be unique")
    role_counts = Counter(str(entry.get("slot_role") or "") for entry in entries)
    if role_counts != Counter(SEASON_SLOT_ROLE_COUNTS):
        raise Season1PanelError(f"Season 1 slot roles differ from {SEASON_SLOT_ROLE_COUNTS}")

    prompt_hashes: set[str] = set()
    system_prompt_hashes: set[str] = set()
    tool_hashes: set[str] = set()
    artifact_hashes: set[str] = set()
    request_identities: set[str] = set()
    models: list[dict[str, Any]] = []
    openrouter_cost = Decimal(0)
    provider_calls = 0
    epicure_calls = 0
    for ordinal, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, Mapping):
            raise Season1PanelError("Season 1 panel contains a non-object entry")
        entry = dict(raw_entry)
        backend = str(entry.get("execution_backend") or "")
        artifact_path = Path(str(entry.get("smoke_artifact_path") or ""))
        expected_artifact_sha256 = str(entry.get("smoke_artifact_sha256") or "")
        artifact = _verified_artifact(artifact_path, expected_artifact_sha256)
        model_id = str(entry["model_id"])
        if (
            artifact.get("status") != "smoke_passed"
            or artifact.get("official") is not False
            or artifact.get("rank_eligible") is not False
        ):
            raise Season1PanelError(f"{model_id} receipt is not an unranked passed smoke")
        _non_error_trace(artifact, model_id)
        if expected_artifact_sha256 in artifact_hashes:
            raise Season1PanelError("contract evidence artifacts must be unique")
        artifact_hashes.add(expected_artifact_sha256)
        prompt_hashes.add(str(artifact.get("prompt_sha256") or ""))
        system_prompt_hashes.add(str(artifact.get("system_prompt_sha256") or ""))
        tool_hash = str(
            artifact.get("raw_epicure_tool_schema_sha256")
            or artifact.get("epicure_tool_catalog_sha256")
            or ""
        )
        tool_hashes.add(tool_hash)
        if backend == "bedrock":
            identity = _verify_bedrock(entry, artifact)
            calls = int(artifact.get("provider_calls") or 0)
        elif backend == "openrouter":
            identity = _verify_openrouter(entry, artifact)
            calls = int(artifact.get("real_provider_calls") or 0)
            openrouter_cost += Decimal(str(artifact.get("cost_usd") or "0"))
        else:
            raise Season1PanelError(f"unsupported execution backend for {model_id}: {backend}")
        duplicate_request_identities = request_identities.intersection(
            identity["request_identity_sha256s"]
        )
        if duplicate_request_identities:
            raise Season1PanelError("provider request identities are reused across models")
        request_identities.update(identity["request_identity_sha256s"])
        provider_calls += calls
        epicure_calls += int(artifact.get("real_epicure_calls") or 0)
        models.append(
            {
                "ordinal": ordinal,
                "display_name": str(entry.get("display_name") or ""),
                "slot_role": str(entry["slot_role"]),
                **identity,
                "smoke_artifact_path": str(artifact_path),
                "smoke_artifact_sha256": expected_artifact_sha256,
                "usage": dict(artifact.get("usage") or {}),
                "wall_clock_latency_ms": int(artifact.get("wall_clock_latency_ms") or 0),
                "provider_calls": calls,
                "epicure_calls": 1,
            }
        )
    if (
        len(prompt_hashes) != 1
        or len(system_prompt_hashes) != 1
        or len(tool_hashes) != 1
        or any(
            not value or len(value) != 64
            for value in prompt_hashes | system_prompt_hashes | tool_hashes
        )
    ):
        raise Season1PanelError("panel receipts do not share one frozen smoke protocol")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "snapshot_date": str(spec.get("snapshot_date") or ""),
        "status": "contract_qualified_candidate_not_season_frozen",
        "panel_design": {
            "model_count": SEASON_MODEL_COUNT,
            "slot_role_counts": dict(SEASON_SLOT_ROLE_COUNTS),
            "selection_policy": str(spec.get("selection_policy") or ""),
        },
        "source_specification_path": str(spec_path),
        "source_specification_sha256": sha256_json(spec),
        "protocol": {
            "prompt_sha256": next(iter(prompt_hashes)),
            "system_prompt_sha256": next(iter(system_prompt_hashes)),
            "raw_epicure_tool_catalog_sha256": next(iter(tool_hashes)),
            "contract_requirement": (
                "two real provider generations, one successful real Epicure find_pairings "
                "call, a normally completed non-empty normalized final answer, and exact "
                "route evidence"
            ),
            "production_strict_json_parity": all(
                model.get("execution_backend") != "openrouter"
                or model.get("provider_structured_output_required") is True
                for model in models
            ),
        },
        "counts": {
            "models": len(models),
            "bedrock_models": sum(model["execution_backend"] == "bedrock" for model in models),
            "openrouter_models": sum(
                model["execution_backend"] == "openrouter" for model in models
            ),
            "real_provider_generations": provider_calls,
            "real_epicure_calls": epicure_calls,
        },
        "cost": {
            "openrouter_reconciled_usd": format(openrouter_cost, "f"),
            "bedrock_status": "usage_recorded_rate_card_pending_aws_billing_crosscheck",
        },
        "models": models,
        "claim_boundary": {
            "contract_compatibility_only": True,
            "quality_observations": 0,
            "human_judgments": 0,
            "season_frozen": False,
            "rank_eligible": False,
        },
        "official": False,
        "rank_eligible": False,
    }


def _atomic_write(directory: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"season1-contract-panel-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise Season1PanelError("content-addressed panel manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_manifest(arguments.spec)
    path = _atomic_write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "status": payload["status"],
                "counts": payload["counts"],
                "cost": payload["cost"],
                "inference_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
