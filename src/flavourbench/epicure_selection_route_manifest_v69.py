"""Freeze a score-blind DeepSeek price-contract refresh for the 27-model study.

The previously pinned OpenRouter requests capped the DeepSeek provider price at
the value observed before collection.  After the endpoint price changed,
OpenRouter rejected otherwise valid exact-provider requests with HTTP 404.  The
model, prompt, decoder, scorer, and exact provider remain independent of all
quality outcomes; this successor refreshes only the live endpoint contract and
requires two new complete 640-primary + 64-repeat blocks.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest import _fetch_endpoints
from .epicure_selection_route_manifest_v54 import (
    DEEPSEEK_PRO_MODEL_ID,
    _block_envelope,
    _load,
    _max_output_tokens,
    _sha256,
    _sha256_file,
)
from .epicure_selection_route_manifest_v65 import verify_manifest as verify_manifest_v65
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-deepseek-price-contract-refresh-v69"
ROUTE_TAG = "deepseek"
PROVIDER_NAME = "DeepSeek"
EXPECTED_CELLS_PER_PANEL = 704


class SelectionRouteManifestV69Error(RuntimeError):
    """The DeepSeek price-contract successor could not be frozen."""


async def build(*, source_path: Path) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v65(source):
        raise SelectionRouteManifestV69Error("v69 requires the exact v65 predecessor")

    endpoints = await _fetch_endpoints(DEEPSEEK_PRO_MODEL_ID)
    matches = [
        copy.deepcopy(endpoint)
        for endpoint in endpoints
        if endpoint.get("tag") == ROUTE_TAG
        and endpoint.get("provider_name") == PROVIDER_NAME
        and endpoint.get("status") == 0
        and "max_tokens" in set(endpoint.get("supported_parameters") or [])
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV69Error("official DeepSeek route is not uniquely executable")
    endpoint = matches[0]
    endpoint["supported_parameters"] = sorted(
        str(value) for value in endpoint.get("supported_parameters") or []
    )

    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    source_rows = {str(row["model"]["id"]): row for row in source["models"]}
    row = rows[DEEPSEEK_PRO_MODEL_ID]
    old_endpoint = copy.deepcopy(row["endpoint"])
    old_execution_sha256 = str(row["endpoint_execution_sha256"])
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    max_output_tokens = _max_output_tokens(row)

    row["endpoint"] = endpoint
    row["endpoint_document_sha256"] = _sha256(endpoint)
    row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    row["endpoint_selection"] = {
        "method": "score-blind exact-provider live price-contract refresh",
        "selected_exact_tag": ROUTE_TAG,
        "superseded_exact_tags": ["gmicloud/fp8"],
        "eligible_endpoint_count": 1,
        "observed_at": observed_at,
        "automatic_fallback": False,
        "quality_scores_or_selections_used": False,
    }
    row["request_policy"]["provider"] = {
        "only": [ROUTE_TAG],
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
        "selection_reason": "score-blind current contract and first-party provider identity",
    }
    row["contract_evidence"] = {
        "status": "route_and_price_contract_frozen_before_complete_replacement_blocks",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_complete_primary_and_repeat_blocks": True,
    }
    row["forecast"] = {
        "panels": 2,
        "primary_tasks_per_panel": 640,
        "repeat_tasks_per_panel": 64,
        "new_provider_calls": EXPECTED_CELLS_PER_PANEL * 2,
        "prompt_token_bound": 4_096,
        "route_max_output_tokens": max_output_tokens,
        "worst_case_usd_per_panel": _block_envelope(endpoint, max_output_tokens=max_output_tokens),
    }

    if any(
        rows[model_id] != source_row
        for model_id, source_row in source_rows.items()
        if model_id != DEEPSEEK_PRO_MODEL_ID
    ):
        raise SelectionRouteManifestV69Error("v69 changed a non-DeepSeek model")

    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_27_deepseek_price_contract_refresh_v69",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "deepseek_price_contract_refresh_v69": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "changed_model_ids": [DEEPSEEK_PRO_MODEL_ID],
                "selected_exact_tag": ROUTE_TAG,
                "selected_provider_name": PROVIDER_NAME,
                "superseded_exact_tags": ["gmicloud/fp8"],
                "prior_endpoint_document_sha256": _sha256(old_endpoint),
                "prior_endpoint_execution_sha256": old_execution_sha256,
                "current_endpoint_document_sha256": row["endpoint_document_sha256"],
                "current_endpoint_execution_sha256": row["endpoint_execution_sha256"],
                "automatic_fallback": False,
                "selection_uses_quality_scores_or_selections": False,
                "complete_two_panel_blocks_required": True,
                "selective_failed_cell_retry": False,
                "superseded_responses_used": False,
                "cross_route_response_pooling": False,
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
        raise SelectionRouteManifestV69Error("constructed v69 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["deepseek_price_contract_refresh_v69"]
        rows = document["models"]
        row = next(value for value in rows if value["model"]["id"] == DEEPSEEK_PRO_MODEL_ID)
    except (KeyError, StopIteration, TypeError):
        return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 27
        and any(value["model"]["id"] == "z-ai/glm-5.3" for value in rows)
        and row["endpoint"].get("tag") == ROUTE_TAG
        and row["endpoint"].get("provider_name") == PROVIDER_NAME
        and row["request_policy"]["provider"].get("only") == [ROUTE_TAG]
        and row["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("changed_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and refresh.get("selected_exact_tag") == ROUTE_TAG
        and refresh.get("selected_provider_name") == PROVIDER_NAME
        and refresh.get("superseded_exact_tags") == ["gmicloud/fp8"]
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_quality_scores_or_selections") is False
        and refresh.get("complete_two_panel_blocks_required") is True
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("superseded_responses_used") is False
        and refresh.get("cross_route_response_pooling") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and document.get("generation_calls_made") == 0
        and document.get("official_results_authorised") is False
    )


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-frontier-refresh-27-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV69Error("content-addressed manifest conflict")
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(build(source_path=args.source))
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
