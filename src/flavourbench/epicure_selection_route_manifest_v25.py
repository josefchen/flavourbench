"""Bind Kimi's exact dated OpenRouter identity after live Morph responses."""

from __future__ import annotations

import argparse
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
from .epicure_selection_route_manifest_v23 import KIMI_MODEL_ID
from .frontier_manifest import verify_manifest_content_address

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v9"
EXPECTED_ACTUAL_MODEL_ID = "moonshotai/kimi-k3-20260715"


class SelectionRouteManifestV25Error(RuntimeError):
    """The dated Kimi identity successor failed verification."""


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
        raise SelectionRouteManifestV25Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV25Error("source manifest content address is invalid")
    return value


def build(*, source_path: Path, calibration_v24_directory: Path) -> dict[str, Any]:
    source = _load(source_path)
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    matches = [entry for entry in document["models"] if entry["model"]["id"] == KIMI_MODEL_ID]
    if len(matches) != 1:
        raise SelectionRouteManifestV25Error("Kimi slot is not unique")
    entry = matches[0]
    prior_slug = entry["model"]["canonical_slug"]
    if prior_slug != "k3" or entry["endpoint"]["tag"] != "morph":
        raise SelectionRouteManifestV25Error("unexpected Kimi predecessor identity or route")
    entry["model"]["canonical_slug"] = EXPECTED_ACTUAL_MODEL_ID
    entry["contract_evidence"] = {
        "status": "live_identity_repaired_before_successor_primary",
        "generation_calls": 2,
        "quality_observations": 0,
        "response_envelope_model": KIMI_MODEL_ID,
        "accounting_model": EXPECTED_ACTUAL_MODEL_ID,
        "actual_provider": "Morph",
        "responses_used_as_primary_data": False,
    }
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    calibration = run_commitment(calibration_v24_directory, expected_responses=2)
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v9",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "calibration_v24": calibration,
                "calibration_used_as_primary_data": False,
                "current_endpoint_network_reads": 0,
                "successor_provider_calls": 0,
                "model_id": KIMI_MODEL_ID,
                "route_unchanged": "morph",
                "prior_expected_actual_model_id": prior_slug,
                "replacement_expected_actual_model_id": EXPECTED_ACTUAL_MODEL_ID,
                "all_other_route_and_model_bytes_preserved": True,
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
        raise SelectionRouteManifestV25Error("route manifest failed content verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV25Error("content-addressed manifest conflict")
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
    parser.add_argument("--calibration-v24-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build(
        source_path=args.source_manifest,
        calibration_v24_directory=args.calibration_v24_directory,
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
