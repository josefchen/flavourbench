"""Freeze the Fable 5 Amazon Bedrock route before any replacement scoring.

The predecessor already contains the complete Qwen/Alibaba repair.  This
successor changes only Fable's exact OpenRouter provider from the failing
Anthropic route to OpenRouter's Amazon Bedrock route.  The decision uses
normal-completion metadata and the public endpoint contract, never model
selections or Epicure scores.
"""

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

from .epicure_selection_route_manifest import _fetch_endpoints
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID
from .epicure_selection_route_manifest_v46 import verify_manifest as verify_manifest_v46
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-fable-bedrock-route-v49"
FABLE_BEDROCK_SPEC = {
    "tag": "amazon-bedrock/claude-on-aws",
    "provider": "Amazon Bedrock",
    "max_output_tokens": 2_048,
    "reasoning_effort": "minimal",
}


class SelectionRouteManifestV49Error(RuntimeError):
    """The Fable Bedrock route successor could not be frozen."""


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
        raise SelectionRouteManifestV49Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRouteManifestV49Error("manifest input is not a JSON object")
    return value


def _block_envelope(endpoint: Mapping[str, Any]) -> str:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * 4_096
    completion = Decimal(str(pricing.get("completion") or 0)) * int(
        FABLE_BEDROCK_SPEC["max_output_tokens"]
    )
    reasoning = Decimal(str(pricing.get("internal_reasoning") or 0)) * int(
        FABLE_BEDROCK_SPEC["max_output_tokens"]
    )
    request = Decimal(str(pricing.get("request") or 0))
    return format((prompt + completion + reasoning + request) * 704, "f")


async def build(*, source_path: Path) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v46(source):
        raise SelectionRouteManifestV49Error("v49 requires the exact v46 route predecessor")
    endpoints = await _fetch_endpoints(FABLE_MODEL_ID)
    matches = [
        copy.deepcopy(endpoint)
        for endpoint in endpoints
        if endpoint.get("tag") == FABLE_BEDROCK_SPEC["tag"]
        and endpoint.get("provider_name") == FABLE_BEDROCK_SPEC["provider"]
        and endpoint.get("status") == 0
        and {"max_tokens", "tools"} <= set(endpoint.get("supported_parameters") or [])
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV49Error("Fable Bedrock route is not uniquely executable")
    endpoint = matches[0]
    endpoint["supported_parameters"] = sorted(
        str(value) for value in endpoint.get("supported_parameters") or []
    )
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    if FABLE_MODEL_ID not in rows:
        raise SelectionRouteManifestV49Error("Fable is absent from the predecessor roster")
    row = rows[FABLE_MODEL_ID]
    row["endpoint"] = endpoint
    row["endpoint_document_sha256"] = _sha256(endpoint)
    row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    row["endpoint_selection"] = {
        "method": "score-blind normal-completion recovery plus current public route contract",
        "selected_exact_tag": FABLE_BEDROCK_SPEC["tag"],
        "eligible_endpoint_count": 1,
        "observed_at": observed_at,
        "automatic_fallback": False,
        "quality_scores_or_selections_used": False,
    }
    row["request_policy"]["provider"] = {
        "only": [FABLE_BEDROCK_SPEC["tag"]],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "require_parameters": True,
    }
    row["execution_route"] = {
        "policy": "exact_openrouter_provider_only_v1",
        "preferred_backend": "openrouter",
        "selected_backend": "openrouter",
        "fallback_used": False,
        "generation_time_automatic_fallback": False,
        "selection_frozen_before_generation": True,
        "selection_reason": (
            "score-blind recovery after Anthropic and Google routes showed high refusal rates"
        ),
    }
    row["contract_evidence"] = {
        "status": "route_frozen_before_transport_pilot_and_complete_replacement_blocks",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_complete_primary_and_repeat_blocks": True,
    }
    row["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "new_provider_calls_per_panel": 704,
        "replacement_panel_count": 2,
        "prompt_token_bound": 4_096,
        "route_max_output_tokens": FABLE_BEDROCK_SPEC["max_output_tokens"],
        "per_panel_worst_case_usd": _block_envelope(endpoint),
    }
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_fable_bedrock_successor_v49",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "fable_route_v49": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "changed_model_ids": [FABLE_MODEL_ID],
                "selected_exact_tag": FABLE_BEDROCK_SPEC["tag"],
                "selected_provider": FABLE_BEDROCK_SPEC["provider"],
                "reasoning_effort": FABLE_BEDROCK_SPEC["reasoning_effort"],
                "automatic_fallback": False,
                "selection_uses_status_and_finish_metadata_only": True,
                "quality_scores_or_selections_used": False,
                "selective_failed_cell_retry": False,
                "complete_replacement_blocks_required": True,
                "first_panel_successes_reused": False,
                "all_other_model_entries_byte_preserved": True,
            },
        }
    )
    digest = _sha256(document)
    document["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    if not verify_manifest(document):
        raise SelectionRouteManifestV49Error("constructed v49 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["fable_route_v49"]
        rows = {str(row["model"]["id"]): row for row in document["models"]}
        fable = rows[FABLE_MODEL_ID]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 26
        and fable["endpoint"].get("tag") == FABLE_BEDROCK_SPEC["tag"]
        and fable["endpoint"].get("provider_name") == FABLE_BEDROCK_SPEC["provider"]
        and fable["request_policy"]["provider"].get("only") == [FABLE_BEDROCK_SPEC["tag"]]
        and fable["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("changed_model_ids") == [FABLE_MODEL_ID]
        and refresh.get("selected_exact_tag") == FABLE_BEDROCK_SPEC["tag"]
        and refresh.get("selected_provider") == FABLE_BEDROCK_SPEC["provider"]
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_status_and_finish_metadata_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("complete_replacement_blocks_required") is True
        and refresh.get("first_panel_successes_reused") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and isinstance((refresh.get("source_manifest") or {}).get("semantic_sha256"), str)
        and isinstance((refresh.get("source_manifest") or {}).get("physical_sha256"), str)
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
            raise SelectionRouteManifestV49Error("content-addressed manifest conflict")
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
    print(_write(asyncio.run(build(source_path=args.source_manifest)), args.output_directory))


if __name__ == "__main__":
    run()
