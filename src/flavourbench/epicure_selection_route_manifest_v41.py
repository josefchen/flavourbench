"""Freeze Claude Fable 5 on the exact Azure OpenRouter endpoint."""

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

from .epicure_selection_route_manifest import _block_envelope_usd, _fetch_endpoints
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-refresh-route-v41"
FABLE_MODEL_ID = "anthropic/claude-fable-5"
FABLE_CANONICAL_ID = "anthropic/claude-5-fable-20260609"
SELECTED_TAG = "azure"
SELECTED_PROVIDER = "Azure"


class SelectionRouteManifestV41Error(RuntimeError):
    """The exact Azure Fable route could not be frozen."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRouteManifestV41Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV41Error("source manifest failed content verification")
    return value


def _select_azure(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [
        dict(endpoint)
        for endpoint in endpoints
        if endpoint.get("tag") == SELECTED_TAG
        and endpoint.get("provider_name") == SELECTED_PROVIDER
        and endpoint.get("status") == 0
        and {"max_tokens", "tools"} <= set(endpoint.get("supported_parameters") or [])
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV41Error("exact Azure Fable endpoint is not uniquely healthy")
    return matches[0]


async def build(*, source_path: Path) -> dict[str, Any]:
    source = _load(source_path)
    endpoints = await _fetch_endpoints(FABLE_MODEL_ID)
    endpoint = _select_azure(endpoints)
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    matches = [
        (index, row)
        for index, row in enumerate(document["models"])
        if row["model"]["id"] == FABLE_MODEL_ID
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV41Error("source Fable row is not unique")
    index, source_row = matches[0]
    if source_row["model"]["canonical_slug"] != FABLE_CANONICAL_ID:
        raise SelectionRouteManifestV41Error("Fable canonical identity changed")
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    row = copy.deepcopy(source_row)
    row["endpoint"] = endpoint
    row["endpoint_document_sha256"] = _sha256(endpoint)
    row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    row["endpoint_selection"] = {
        "method": "first healthy exact non-Anthropic non-AWS provider after transport-only failure",
        "selected_exact_tag": SELECTED_TAG,
        "eligible_endpoint_count": 1,
        "observed_at": observed_at,
        "automatic_fallback": False,
        "quality_observations_used": 0,
    }
    row["request_policy"]["provider"]["only"] = [SELECTED_TAG]
    row["execution_route"] = {
        "policy": "exact_openrouter_provider_only_v1",
        "preferred_backend": "openrouter",
        "selected_backend": "openrouter",
        "fallback_used": False,
        "generation_time_automatic_fallback": False,
        "selection_frozen_before_generation": True,
        "selection_reason": "v40 Anthropic content-filter transport failure",
    }
    row["contract_evidence"] = {
        "status": "route_frozen_before_complete_successor_block",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_complete_primary_and_repeat_blocks": True,
    }
    row["backend_contract"] = {}
    row["backend_contract_sha256"] = "unfrozen"
    row["forecast"] = {
        **dict(row["forecast"]),
        "model_block_worst_case_usd": _block_envelope_usd(endpoint),
        "new_provider_calls": 704,
    }
    row["model"]["description"] = "Claude Fable 5 on the exact Azure route."
    row["slot"]["rationale"] = "Claude Fable 5 on the exact Azure route."
    document["models"][index] = row
    document.update(
        {
            "schema_version": SCHEMA_VERSION,
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_fable_azure_successor_v41",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh_v41": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "current_endpoint_network_reads": 1,
                "current_endpoint_count": len(endpoints),
                "successor_provider_calls": 0,
                "changed_model_ids": [FABLE_MODEL_ID],
                "selected_exact_tag": SELECTED_TAG,
                "selected_provider_name": SELECTED_PROVIDER,
                "canonical_model_slug": FABLE_CANONICAL_ID,
                "automatic_fallback": False,
                "all_other_model_entries_byte_preserved": True,
                "scores_or_selections_used": False,
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
        raise SelectionRouteManifestV41Error("v41 manifest failed content verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        rows = [row for row in document["models"] if row["model"]["id"] == FABLE_MODEL_ID]
        refresh = document["route_refresh_v41"]
    except (KeyError, TypeError):
        return False
    if len(rows) != 1:
        return False
    row = rows[0]
    return bool(
        verify_manifest_content_address(document)
        and document.get("schema_version") == SCHEMA_VERSION
        and len(document.get("models", [])) == 26
        and row["model"].get("canonical_slug") == FABLE_CANONICAL_ID
        and row["endpoint"].get("tag") == SELECTED_TAG
        and row["endpoint"].get("provider_name") == SELECTED_PROVIDER
        and row["endpoint"].get("status") == 0
        and row["request_policy"]["provider"].get("only") == [SELECTED_TAG]
        and row["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("changed_model_ids") == [FABLE_MODEL_ID]
        and refresh.get("automatic_fallback") is False
        and refresh.get("scores_or_selections_used") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and document.get("generation_calls_made") == 0
        and document.get("official_results_authorised") is False
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-frontier-refresh-26-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV41Error("content-addressed manifest conflict")
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
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(build(source_path=args.source_manifest))
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
