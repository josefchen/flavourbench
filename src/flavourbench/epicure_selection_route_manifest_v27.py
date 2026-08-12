"""Freeze DeepSeek V4 Pro to Novita FP8 after native-route HTTP 404s."""

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

from .epicure_selection_powered_plan import run_commitment
from .epicure_selection_route_manifest import (
    _block_envelope_usd,
    _fetch_endpoints,
    _select_exact,
)
from .epicure_selection_route_manifest_v26 import (
    DEEPSEEK_PRO_MODEL_ID,
    EXPECTED_ACTUAL_MODEL_ID,
)
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v11"
REPLACEMENT_TAG = "novita/fp8"
REPLACEMENT_PROVIDER = "Novita"


class SelectionRouteManifestV27Error(RuntimeError):
    """The Novita DeepSeek route successor failed verification."""


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
        raise SelectionRouteManifestV27Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV27Error("source manifest content address is invalid")
    return value


async def build(*, source_path: Path, calibration_v26_directory: Path) -> dict[str, Any]:
    source = _load(source_path)
    endpoints = await _fetch_endpoints(DEEPSEEK_PRO_MODEL_ID)
    endpoint = _select_exact(
        model_id=DEEPSEEK_PRO_MODEL_ID,
        tag=REPLACEMENT_TAG,
        endpoints=endpoints,
    )
    if endpoint.get("provider_name") != REPLACEMENT_PROVIDER:
        raise SelectionRouteManifestV27Error("Novita endpoint identity changed")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    matches = [
        entry for entry in document["models"] if entry["model"]["id"] == DEEPSEEK_PRO_MODEL_ID
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV27Error("DeepSeek V4 Pro slot is not unique")
    entry = matches[0]
    if (
        entry["endpoint"]["tag"] != "deepseek"
        or entry["model"]["canonical_slug"] != EXPECTED_ACTUAL_MODEL_ID
    ):
        raise SelectionRouteManifestV27Error("unexpected v26 DeepSeek predecessor")
    prior = {
        "tag": entry["endpoint"]["tag"],
        "provider_name": entry["endpoint"]["provider_name"],
        "endpoint_execution_sha256": endpoint_execution_contract_sha256(entry["endpoint"]),
    }
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    entry["endpoint"] = endpoint
    entry["endpoint_document_sha256"] = _sha256(endpoint)
    entry["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    entry["endpoint_selection"] = {
        "method": "native_exact_route_http_404_replaced_before_clean_primary_collection",
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
            "fallback_used": False,
            "generation_time_automatic_fallback": False,
            "selection_frozen_before_generation": True,
            "selection_reason": "replace unavailable native route with exact Novita FP8 endpoint",
        }
    )
    entry["contract_evidence"] = {
        "status": "live_endpoint_repaired_before_successor_primary",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_successful_successor_check": True,
    }
    entry["backend_contract"] = {}
    entry["backend_contract_sha256"] = "unfrozen"
    entry["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "model_block_worst_case_usd": _block_envelope_usd(endpoint),
    }
    calibration = run_commitment(calibration_v26_directory, expected_responses=4)
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v11",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "calibration_v26": calibration,
                "calibration_used_as_primary_data": False,
                "current_endpoint_network_reads": 1,
                "successor_provider_calls": 0,
                "model_id": DEEPSEEK_PRO_MODEL_ID,
                "prior_route": prior,
                "replacement_route": {
                    "tag": endpoint["tag"],
                    "provider_name": endpoint["provider_name"],
                    "execution_backend": "openrouter",
                    "endpoint_execution_sha256": endpoint_execution_contract_sha256(endpoint),
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
        raise SelectionRouteManifestV27Error("route manifest failed content verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV27Error("content-addressed manifest conflict")
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
    parser.add_argument("--calibration-v26-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_path=args.source_manifest,
            calibration_v26_directory=args.calibration_v26_directory,
        )
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
