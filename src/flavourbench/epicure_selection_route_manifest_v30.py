"""Replace unavailable Nemotron Ultra with dated Nemotron 3.5 Lightning."""

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
from pathlib import Path
from typing import Any

import httpx

from .epicure_selection_powered_plan import run_commitment
from .epicure_selection_route_manifest import (
    _block_envelope_usd,
    _fetch_endpoints,
    _select_exact,
)
from .epicure_selection_route_manifest_v29 import NEMOTRON_MODEL_ID as ULTRA_MODEL_ID
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v14"
LIGHTNING_MODEL_ID = "nvidia/nemotron-3.5-lightning"
LIGHTNING_CANONICAL_ID = "nvidia/nemotron-3.5-lightning-20260807"
REPLACEMENT_TAG = "coreweave/bf16"
REPLACEMENT_PROVIDER = "CoreWeave"
MODEL_FIELDS = (
    "architecture",
    "canonical_slug",
    "context_length",
    "created",
    "hugging_face_id",
    "id",
    "name",
    "pricing",
    "supported_parameters",
    "top_provider",
)


class SelectionRouteManifestV30Error(RuntimeError):
    """The Nemotron Lightning successor failed verification."""


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
        raise SelectionRouteManifestV30Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV30Error("source manifest failed content verification")
    return value


async def _fetch_model(model_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/models", headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        rows = response.json().get("data") or []
    matches = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("id") == model_id]
    if len(matches) != 1:
        raise SelectionRouteManifestV30Error("Nemotron Lightning catalog row is not unique")
    return {field: matches[0].get(field) for field in MODEL_FIELDS}


async def build(*, source_path: Path, calibration_v29_directory: Path) -> dict[str, Any]:
    source = _load(source_path)
    model, endpoints = await asyncio.gather(
        _fetch_model(LIGHTNING_MODEL_ID), _fetch_endpoints(LIGHTNING_MODEL_ID)
    )
    if model.get("canonical_slug") != LIGHTNING_CANONICAL_ID:
        raise SelectionRouteManifestV30Error("Nemotron Lightning dated identity changed")
    endpoint = _select_exact(
        model_id=LIGHTNING_MODEL_ID,
        tag=REPLACEMENT_TAG,
        endpoints=endpoints,
    )
    if endpoint.get("provider_name") != REPLACEMENT_PROVIDER:
        raise SelectionRouteManifestV30Error("CoreWeave endpoint identity changed")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    matches = [
        (index, entry)
        for index, entry in enumerate(document["models"])
        if entry["model"]["id"] == ULTRA_MODEL_ID
    ]
    if len(matches) != 1 or any(
        entry["model"]["id"] == LIGHTNING_MODEL_ID for entry in document["models"]
    ):
        raise SelectionRouteManifestV30Error("unexpected predecessor roster")
    index, prior = matches[0]
    entry = copy.deepcopy(prior)
    slot_id = prior["slot"]["slot_id"]
    entry["slot"] = {
        "cohort": "nvidia",
        "model_id": LIGHTNING_MODEL_ID,
        "open_weight_candidate": False,
        "rationale": "Nemotron 3.5 Lightning.",
        "slot_id": slot_id,
    }
    entry["model"] = model
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    entry["endpoint"] = endpoint
    entry["endpoint_document_sha256"] = _sha256(endpoint)
    entry["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    entry["endpoint_selection"] = {
        "method": "newest_dated_nvidia_successor_on_high_uptime_exact_bf16_endpoint",
        "selected_exact_tag": endpoint["tag"],
        "eligible_endpoint_count": sum(
            value.get("status") == 0
            and "max_tokens" in set(value.get("supported_parameters") or [])
            for value in endpoints
        ),
        "observed_at": observed_at,
        "automatic_fallback": False,
    }
    entry["request_policy"]["provider"]["only"] = [endpoint["tag"]]
    entry["execution_route"].update(
        {
            "policy": "exact_openrouter_endpoint_v1",
            "preferred_backend": "openrouter",
            "selected_backend": "openrouter",
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
            "selection_frozen_before_generation": True,
            "selection_reason": "replace unavailable Ultra routes with dated Lightning BF16",
        }
    )
    entry["contract_evidence"] = {
        "status": "live_endpoint_selected_before_successor_primary",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_successful_eight_cell_successor_check": True,
    }
    entry["backend_contract"] = {}
    entry["backend_contract_sha256"] = "unfrozen"
    entry["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "model_block_worst_case_usd": _block_envelope_usd(endpoint),
    }
    document["models"][index] = entry
    calibration = run_commitment(calibration_v29_directory, expected_responses=6)
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v14",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "calibration_v29": calibration,
                "calibration_used_as_primary_data": False,
                "current_catalog_network_reads": 1,
                "current_endpoint_network_reads": 1,
                "successor_provider_calls": 0,
                "removed_model_id": ULTRA_MODEL_ID,
                "removed_endpoint_execution_sha256": endpoint_execution_contract_sha256(
                    prior["endpoint"]
                ),
                "replacement_model_id": LIGHTNING_MODEL_ID,
                "replacement_canonical_model_id": LIGHTNING_CANONICAL_ID,
                "replacement_route": {
                    "tag": endpoint["tag"],
                    "provider_name": endpoint["provider_name"],
                    "execution_backend": "openrouter",
                    "endpoint_execution_sha256": endpoint_execution_contract_sha256(endpoint),
                },
                "preserved_slot_id": slot_id,
                "all_other_model_entries_byte_preserved": True,
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
        raise SelectionRouteManifestV30Error("route manifest failed content verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV30Error("content-addressed manifest conflict")
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
    parser.add_argument("--calibration-v29-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_path=args.source_manifest,
            calibration_v29_directory=args.calibration_v29_directory,
        )
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
