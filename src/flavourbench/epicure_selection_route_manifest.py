"""Freeze an exact route successor after bounded paid calibration."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v5"
ENDPOINT_REPLACEMENTS = {
    "google/gemini-3.6-flash": "google-vertex/us",
    "nvidia/nemotron-3-ultra-550b-a55b": "baseten/fp4",
}
REQUIRED_PARAMETERS = frozenset({"max_tokens"})


class SelectionRouteManifestError(RuntimeError):
    """The exact powered route successor could not be frozen."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRouteManifestError("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestError("source manifest content address is invalid")
    return value


async def _fetch_endpoints(model_id: str) -> list[dict[str, Any]]:
    author, slug = model_id.split("/", 1)
    url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
    endpoints = (payload.get("data") or {}).get("endpoints")
    if not isinstance(endpoints, list):
        raise SelectionRouteManifestError(f"endpoint catalog missing for {model_id}")
    return [dict(value) for value in endpoints if isinstance(value, Mapping)]


def _select_exact(
    *, model_id: str, tag: str, endpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    matches = [dict(value) for value in endpoints if value.get("tag") == tag]
    if len(matches) != 1:
        raise SelectionRouteManifestError(f"{model_id} exact endpoint {tag} is not unique")
    endpoint = matches[0]
    supported = frozenset(str(value) for value in endpoint.get("supported_parameters") or [])
    if (
        endpoint.get("model_id") != model_id
        or endpoint.get("status") != 0
        or not REQUIRED_PARAMETERS <= supported
        or not endpoint.get("provider_name")
    ):
        raise SelectionRouteManifestError(f"{model_id} replacement endpoint is not executable")
    endpoint["supported_parameters"] = sorted(supported)
    return endpoint


def _calibration_commitment(directory: Path) -> dict[str, Any]:
    responses = sorted(directory.glob("responses/primary/*/response-*.json"))
    digests: list[str] = []
    spend_micros = 0
    for path in responses:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(value)
        recorded = str(payload.pop("artifact_sha256", ""))
        if recorded != _sha256(payload):
            raise SelectionRouteManifestError("calibration response failed verification")
        digests.append(recorded)
        spend_micros += int((value.get("generation") or {}).get("cost_micros") or 0)
    journal = directory / "attempts/provider-attempts.jsonl"
    if len(responses) != 103 or not journal.is_file() or journal.is_symlink():
        raise SelectionRouteManifestError("calibration run is incomplete")
    return {
        "response_count": len(responses),
        "response_artifact_set_sha256": _sha256(sorted(digests)),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "spend_micros": spend_micros,
        "used_as_primary_data": False,
        "interrupted_after_route_and_parser_findings": True,
    }


def _block_envelope_usd(endpoint: Mapping[str, Any]) -> str:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * 4096
    completion = Decimal(str(pricing.get("completion") or 0)) * 8192
    reasoning = Decimal(str(pricing.get("internal_reasoning") or 0)) * 8192
    value = (prompt + completion + reasoning) * 704
    return format(value, "f")


async def build(source_path: Path, calibration_directory: Path) -> dict[str, Any]:
    source = _load(source_path)
    if source.get("schema_version") != SCHEMA_VERSION or len(source.get("models") or []) != 20:
        raise SelectionRouteManifestError("source is not the exact 20-model routed manifest")
    endpoint_sets = await asyncio.gather(
        *(_fetch_endpoints(model_id) for model_id in ENDPOINT_REPLACEMENTS)
    )
    catalogs = dict(zip(ENDPOINT_REPLACEMENTS, endpoint_sets, strict=True))
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    prior: dict[str, Any] = {}
    replaced = 0
    for entry in document["models"]:
        model_id = str(entry["model"]["id"])
        if model_id not in ENDPOINT_REPLACEMENTS:
            continue
        replaced += 1
        endpoints = catalogs[model_id]
        endpoint = _select_exact(
            model_id=model_id,
            tag=ENDPOINT_REPLACEMENTS[model_id],
            endpoints=endpoints,
        )
        old_endpoint = dict(entry["endpoint"])
        prior[model_id] = {
            "tag": old_endpoint["tag"],
            "endpoint_execution_sha256": endpoint_execution_contract_sha256(old_endpoint),
            "pilot_outcome": (
                "two_of_four requests failed"
                if model_id == "google/gemini-3.6-flash"
                else "two_of_four requests failed after the selected route became unhealthy"
            ),
        }
        entry["endpoint"] = endpoint
        entry["endpoint_document_sha256"] = _sha256(endpoint)
        entry["endpoint_selection"] = {
            "method": "failed_final_pilot_route_replaced_before_primary_collection",
            "selected_exact_tag": endpoint["tag"],
            "eligible_endpoint_count": sum(
                value.get("status") == 0
                and REQUIRED_PARAMETERS
                <= frozenset(str(item) for item in value.get("supported_parameters") or [])
                for value in endpoints
            ),
            "observed_at": observed_at,
            "automatic_fallback": False,
        }
        entry["request_policy"]["provider"]["only"] = [endpoint["tag"]]
        entry["execution_route"].update(
            {
                "fallback_used": False,
                "generation_time_automatic_fallback": False,
                "selection_frozen_before_generation": True,
                "selection_reason": "preprimary_route_reliability_repair",
            }
        )
        entry["contract_evidence"] = {
            "status": "live_endpoint_repaired_before_successor_pilot",
            "generation_calls": 0,
            "quality_observations": 0,
            "requires_successful_successor_pilot": True,
        }
        entry["backend_contract"] = {}
        entry["backend_contract_sha256"] = "unfrozen"
        entry["forecast"] = {
            "primary_tasks": 640,
            "repeat_tasks": 64,
            "model_block_worst_case_usd": _block_envelope_usd(endpoint),
        }
    if replaced != len(ENDPOINT_REPLACEMENTS):
        raise SelectionRouteManifestError("source endpoint replacements were not unique")
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v5",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "calibration_plan_sha256": (
                    "5c219afba91ac35f3c3598bdaab1a53422e1c72cff23bd9802235de66afc720d"
                ),
                "calibration": _calibration_commitment(calibration_directory),
                "calibration_provider_calls": 103,
                "candidate_specific_calibration_provider_calls": 8,
                "calibration_used_as_primary_data": False,
                "current_endpoint_network_reads": 2,
                "successor_provider_calls": 0,
                "prior_routes": prior,
                "replacement_routes": {
                    model_id: {
                        "tag": next(
                            entry["endpoint"]["tag"]
                            for entry in document["models"]
                            if entry["model"]["id"] == model_id
                        ),
                        "provider_name": next(
                            entry["endpoint"]["provider_name"]
                            for entry in document["models"]
                            if entry["model"]["id"] == model_id
                        ),
                        "endpoint_execution_sha256": endpoint_execution_contract_sha256(
                            next(
                                entry["endpoint"]
                                for entry in document["models"]
                                if entry["model"]["id"] == model_id
                            )
                        ),
                    }
                    for model_id in ENDPOINT_REPLACEMENTS
                },
                "all_other_routes_byte_preserved": True,
                "automatic_fallback": False,
            },
        }
    )
    digest = _sha256(document)
    document["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest_content_address(document):
        raise SelectionRouteManifestError("successor route manifest failed verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestError("content-addressed route manifest conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--calibration-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        _write(
            asyncio.run(build(args.source_manifest, args.calibration_directory)),
            args.output_directory,
        )
    )


if __name__ == "__main__":
    run()
