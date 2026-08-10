"""Assemble passed frontier contract smokes into one content-addressed evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-frontier-contract-evidence-bundle-v1"
EXPECTED_MODELS = frozenset(
    {
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "command-a-plus-05-2026",
        "command-a-reasoning-08-2025",
        "k3",
    }
)


class FrontierEvidenceError(RuntimeError):
    """Frontier compatibility evidence was missing, inconsistent, or ineligible."""


def _load_verified(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierEvidenceError(f"invalid evidence artifact: {path}") from error
    if not isinstance(document, dict):
        raise FrontierEvidenceError(f"evidence artifact is not an object: {path}")
    recorded = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    actual = sha256_json(payload)
    if recorded != actual:
        raise FrontierEvidenceError(f"artifact content address does not verify: {path}")
    return document


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
            raise FrontierEvidenceError("content-addressed evidence bundle conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _provider_lane(artifact: Mapping[str, Any]) -> str:
    provider = str(artifact.get("provider") or "")
    if provider:
        return provider
    schema = str(artifact.get("schema_version") or "")
    if "openrouter" in schema:
        return "openrouter"
    raise FrontierEvidenceError("could not determine provider lane")


def _identity(artifact: Mapping[str, Any], provider: str) -> dict[str, Any]:
    requested = str(artifact.get("requested_model_id") or "")
    if provider == "openrouter":
        canonical = str(artifact.get("canonical_slug") or "")
        returned_provider = str(artifact.get("returned_provider_name") or "")
        accounting = artifact.get("generation_accounting")
        if not canonical or not returned_provider or not isinstance(accounting, list):
            raise FrontierEvidenceError(f"incomplete OpenRouter identity evidence for {requested}")
        if len(accounting) != 2 or any(
            not isinstance(item, Mapping)
            or item.get("model") != canonical
            or item.get("provider_name") != returned_provider
            for item in accounting
        ):
            raise FrontierEvidenceError(f"OpenRouter identity accounting mismatch for {requested}")
        return {
            "canonical_model_id": canonical,
            "actual_provider": returned_provider,
            "verification": "generation_accounting_model_and_provider",
            "response_returned_model": True,
        }
    if provider == "kimi_code_direct":
        returned = artifact.get("returned_model_ids")
        if (
            not isinstance(returned, list)
            or len(returned) != 2
            or any(item != requested for item in returned)
        ):
            raise FrontierEvidenceError(f"Kimi returned-model mismatch for {requested}")
        return {
            "canonical_model_id": requested,
            "actual_provider": "Kimi Code managed API",
            "verification": "response_returned_exact_catalog_model",
            "response_returned_model": True,
        }
    if provider == "cohere_direct":
        entry = artifact.get("catalog_entry")
        if not isinstance(entry, Mapping) or entry.get("name") != requested:
            raise FrontierEvidenceError(f"Cohere catalog identity mismatch for {requested}")
        return {
            "canonical_model_id": requested,
            "actual_provider": "Cohere direct API",
            "verification": "authenticated_catalog_and_exact_request_only",
            "response_returned_model": False,
        }
    raise FrontierEvidenceError(f"unsupported frontier provider lane: {provider}")


def build_bundle(
    artifact_paths: Sequence[Path],
    *,
    expected_models: frozenset[str] = EXPECTED_MODELS,
) -> dict[str, Any]:
    if not artifact_paths:
        raise FrontierEvidenceError("no frontier artifacts supplied")
    models: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    generation_ids: set[str] = set()
    epicure_hashes: set[str] = set()
    response_schema_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    known_cost = Decimal(0)
    known_cost_models = 0

    for path in artifact_paths:
        artifact = _load_verified(path)
        requested = str(artifact.get("requested_model_id") or "")
        if not requested or requested in seen_models:
            raise FrontierEvidenceError("requested model IDs must be present and unique")
        if artifact.get("status") != "smoke_passed":
            raise FrontierEvidenceError(f"frontier smoke did not pass: {requested}")
        if artifact.get("official") is not False or artifact.get("rank_eligible") is not False:
            raise FrontierEvidenceError(f"frontier smoke has invalid ranking status: {requested}")
        if int(artifact.get("real_provider_calls") or 0) != 2:
            raise FrontierEvidenceError(f"frontier smoke lacks two provider calls: {requested}")
        if int(artifact.get("real_epicure_calls") or 0) != 1:
            raise FrontierEvidenceError(f"frontier smoke lacks one Epicure call: {requested}")
        trace = artifact.get("complete_epicure_trace")
        if (
            not isinstance(trace, list)
            or len(trace) != 1
            or not isinstance(trace[0], Mapping)
            or trace[0].get("is_error") is not False
        ):
            raise FrontierEvidenceError(f"frontier smoke has an invalid Epicure trace: {requested}")
        ids = artifact.get("generation_ids")
        if not isinstance(ids, list) or len(ids) != 2 or any(not str(item) for item in ids):
            raise FrontierEvidenceError(f"frontier smoke has invalid generation IDs: {requested}")
        if generation_ids.intersection(str(item) for item in ids):
            raise FrontierEvidenceError("generation IDs are duplicated across frontier artifacts")
        generation_ids.update(str(item) for item in ids)

        provider = _provider_lane(artifact)
        identity = _identity(artifact, provider)
        epicure_hash = str(artifact.get("epicure_tool_catalog_sha256") or "")
        schema_hash = str(artifact.get("response_schema_sha256") or "")
        prompt_hash = str(artifact.get("prompt_sha256") or "")
        if not all(len(value) == 64 for value in (epicure_hash, schema_hash, prompt_hash)):
            raise FrontierEvidenceError(f"frontier smoke has invalid protocol hashes: {requested}")
        epicure_hashes.add(epicure_hash)
        response_schema_hashes.add(schema_hash)
        prompt_hashes.add(prompt_hash)

        cost_raw = artifact.get("cost_usd")
        cost_usd: str | None = None
        cost_status = str(artifact.get("cost_status") or "")
        if cost_raw is not None:
            cost = Decimal(str(cost_raw))
            if cost < 0:
                raise FrontierEvidenceError(f"negative frontier cost: {requested}")
            known_cost += cost
            known_cost_models += 1
            cost_usd = format(cost, "f")
            cost_status = "generation_metadata_reconciled"
        elif not cost_status:
            raise FrontierEvidenceError(f"frontier cost status is missing: {requested}")

        models.append(
            {
                "requested_model_id": requested,
                **identity,
                "provider_lane": provider,
                "artifact_path": str(path),
                "artifact_sha256": str(artifact["artifact_sha256"]),
                "generation_ids": [str(item) for item in ids],
                "usage": dict(artifact.get("usage") or {}),
                "wall_clock_latency_ms": int(artifact.get("wall_clock_latency_ms") or 0),
                "real_provider_calls": 2,
                "real_epicure_calls": 1,
                "cost_usd": cost_usd,
                "cost_status": cost_status,
            }
        )
        seen_models.add(requested)

    if seen_models != set(expected_models):
        missing = sorted(set(expected_models) - seen_models)
        extra = sorted(seen_models - set(expected_models))
        raise FrontierEvidenceError(
            f"frontier model set mismatch; missing={missing}, extra={extra}"
        )
    if len(epicure_hashes) != 1 or len(response_schema_hashes) != 1 or len(prompt_hashes) != 1:
        raise FrontierEvidenceError("frontier artifacts do not share one frozen task contract")
    models.sort(key=lambda item: item["requested_model_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": "2026-07-28",
        "scope": "post_freeze_frontier_contract_evidence",
        "status": "all_selected_contract_smokes_passed",
        "epicure_tool_catalog_sha256": next(iter(epicure_hashes)),
        "response_schema_sha256": next(iter(response_schema_hashes)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "counts": {
            "models": len(models),
            "provider_lanes": len({item["provider_lane"] for item in models}),
            "provider_generations": sum(item["real_provider_calls"] for item in models),
            "epicure_calls": sum(item["real_epicure_calls"] for item in models),
            "response_returned_model_verified": sum(
                bool(item["response_returned_model"]) for item in models
            ),
            "known_cost_models": known_cost_models,
        },
        "known_reconciled_cost_usd": format(known_cost, "f"),
        "models": models,
        "interpretation": {
            "performance_measurement": False,
            "season0_member": False,
            "leaderboard_observation": False,
            "claim": (
                "These records establish endpoint, tool-call, Epicure, and output-contract "
                "compatibility only."
            ),
        },
        "official": False,
        "rank_eligible": False,
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_bundle(arguments.artifact)
    path = _atomic_write(arguments.output_dir, "frontier-contract-evidence", payload)
    print(
        json.dumps(
            {
                "bundle": str(path),
                "status": payload["status"],
                "counts": payload["counts"],
                "known_reconciled_cost_usd": payload["known_reconciled_cost_usd"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
