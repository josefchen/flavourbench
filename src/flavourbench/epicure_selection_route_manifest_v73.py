"""Freeze the executable GMICloud DeepSeek contract at its current price.

The prior exact GMICloud route remains available, but its advertised price rose
above the frozen OpenRouter ``max_price`` ceiling.  A non-scored compatibility
probe verified that the same exact provider still serves the dated DeepSeek
identity under the benchmark's no-training data policy.  This successor binds
the current endpoint bytes and requires two new complete blocks; no prior
DeepSeek responses are pooled into the final estimate.
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
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-deepseek-gmicloud-contract-refresh-v73"
ROUTE_TAG = "gmicloud/fp8"
PROVIDER_NAME = "GMICloud"
EXPECTED_CELLS_PER_PANEL = 704


class SelectionRouteManifestV73Error(RuntimeError):
    """The current-price GMICloud successor could not be frozen."""


async def build(*, source_path: Path) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v65(source):
        raise SelectionRouteManifestV73Error("v73 requires the exact v65 predecessor")
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
        raise SelectionRouteManifestV73Error("GMICloud route is not uniquely executable")
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
        "method": "score-blind same-provider live price-contract refresh",
        "selected_exact_tag": ROUTE_TAG,
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
        "selection_reason": "same exact provider, current price contract, no-training policy",
    }
    row["contract_evidence"] = {
        "status": "compatibility_verified_before_complete_replacement_blocks",
        "compatibility_generation_calls": 1,
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
        raise SelectionRouteManifestV73Error("v73 changed a non-DeepSeek model")

    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_27_deepseek_gmicloud_contract_refresh_v73",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "deepseek_gmicloud_contract_refresh_v73": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "changed_model_ids": [DEEPSEEK_PRO_MODEL_ID],
                "selected_exact_tag": ROUTE_TAG,
                "selected_provider_name": PROVIDER_NAME,
                "prior_endpoint_document_sha256": _sha256(old_endpoint),
                "prior_endpoint_execution_sha256": old_execution_sha256,
                "current_endpoint_document_sha256": row["endpoint_document_sha256"],
                "current_endpoint_execution_sha256": row["endpoint_execution_sha256"],
                "compatibility_probe": {
                    "provider_http_status": 200,
                    "returned_model_id": DEEPSEEK_PRO_MODEL_ID,
                    "returned_provider": PROVIDER_NAME,
                    "scored_as_benchmark_evidence": False,
                    "quality_values_read": False,
                },
                "automatic_fallback": False,
                "selection_uses_quality_scores_or_selections": False,
                "complete_two_panel_blocks_required": True,
                "selective_failed_cell_retry": False,
                "superseded_responses_used": False,
                "cross_contract_response_pooling": False,
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
        raise SelectionRouteManifestV73Error("constructed v73 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["deepseek_gmicloud_contract_refresh_v73"]
        rows = document["models"]
        row = next(value for value in rows if value["model"]["id"] == DEEPSEEK_PRO_MODEL_ID)
    except (KeyError, StopIteration, TypeError):
        return False
    probe = refresh.get("compatibility_probe") or {}
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
        and probe.get("provider_http_status") == 200
        and probe.get("returned_model_id") == DEEPSEEK_PRO_MODEL_ID
        and probe.get("returned_provider") == PROVIDER_NAME
        and probe.get("scored_as_benchmark_evidence") is False
        and probe.get("quality_values_read") is False
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_quality_scores_or_selections") is False
        and refresh.get("complete_two_panel_blocks_required") is True
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("superseded_responses_used") is False
        and refresh.get("cross_contract_response_pooling") is False
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
            raise SelectionRouteManifestV73Error("content-addressed manifest conflict")
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
    print(_write(asyncio.run(build(source_path=args.source)), args.output_directory))


if __name__ == "__main__":
    run()
