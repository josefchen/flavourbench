"""Freeze a score-blind GMICloud successor for the DeepSeek full blocks.

The direct DeepSeek repair returned no completed cells in either panel and the
subsequent BaseTen experiment was dominated by HTTP 429 failures.  Historical
GMICloud blocks completed 699/704 and 704/704 cells.  This successor observes
only transport status, route identity, plan identity, and cell identity; it
never reads an Epicure score or model selection.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .epicure_selection_route_manifest import _fetch_endpoints
from .epicure_selection_route_manifest_v54 import DEEPSEEK_PRO_MODEL_ID
from .epicure_selection_route_manifest_v57 import (
    _block_envelope,
    _load,
    _max_output_tokens,
    _sha256,
    _sha256_file,
)
from .epicure_selection_route_manifest_v57 import verify_manifest as verify_manifest_v57
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-deepseek-complete-block-repair-v61"
ROUTE_TAG = "gmicloud/fp8"
PROVIDER_NAME = "GMICloud"
EXPECTED_CELLS_PER_PANEL = 704
BASE_TEN_ROUTE = "baseten/fp4"


class SelectionRouteManifestV61Error(RuntimeError):
    """The GMICloud complete-block route successor could not be frozen."""


def _transport_projection(
    run_directory: Path,
    *,
    expected_plan_sha256: str,
    expected_route: str,
    expected_completed: int,
    expected_failed: int,
) -> dict[str, Any]:
    responses = run_directory / "responses"
    if responses.is_symlink() or not responses.is_dir():
        raise SelectionRouteManifestV61Error("transport response directory is unavailable")
    records: list[dict[str, Any]] = []
    for path in sorted(responses.rglob("*.json")):
        value = _load(path)
        if value.get("model_id") == DEEPSEEK_PRO_MODEL_ID:
            records.append(value)
    statuses = Counter(str(value.get("status")) for value in records)
    if (
        len(records) != EXPECTED_CELLS_PER_PANEL
        or statuses != Counter({"completed": expected_completed, "failed": expected_failed})
        or any(
            value.get("provider_route") != expected_route
            or value.get("plan_sha256") != expected_plan_sha256
            for value in records
        )
    ):
        raise SelectionRouteManifestV61Error("transport projection differs")
    cell_ids = sorted(str(value["cell_id"]) for value in records)
    if len(set(cell_ids)) != EXPECTED_CELLS_PER_PANEL:
        raise SelectionRouteManifestV61Error("transport projection contains duplicate cells")
    return {
        "plan_sha256": expected_plan_sha256,
        "model_id": DEEPSEEK_PRO_MODEL_ID,
        "provider_tag": expected_route,
        "scheduled_cells": EXPECTED_CELLS_PER_PANEL,
        "completed_response_artifacts": expected_completed,
        "failed_response_artifacts": expected_failed,
        "quality_scores_or_selections_read": False,
        "cell_id_set_sha256": hashlib.sha256("\n".join(cell_ids).encode()).hexdigest(),
    }


async def build(
    *,
    source_path: Path,
    baseten_panel_1_run: Path,
    baseten_panel_1_plan_sha256: str,
    baseten_panel_2_run: Path,
    baseten_panel_2_plan_sha256: str,
    historical_panel_1_run: Path,
    historical_panel_1_plan_sha256: str,
    historical_panel_2_run: Path,
    historical_panel_2_plan_sha256: str,
) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v57(source):
        raise SelectionRouteManifestV61Error("v61 requires the exact v57 predecessor")
    baseten = {
        "panel_1": _transport_projection(
            baseten_panel_1_run,
            expected_plan_sha256=baseten_panel_1_plan_sha256,
            expected_route=BASE_TEN_ROUTE,
            expected_completed=70,
            expected_failed=634,
        ),
        "panel_2": _transport_projection(
            baseten_panel_2_run,
            expected_plan_sha256=baseten_panel_2_plan_sha256,
            expected_route=BASE_TEN_ROUTE,
            expected_completed=72,
            expected_failed=632,
        ),
    }
    historical = {
        "panel_1": _transport_projection(
            historical_panel_1_run,
            expected_plan_sha256=historical_panel_1_plan_sha256,
            expected_route=ROUTE_TAG,
            expected_completed=699,
            expected_failed=5,
        ),
        "panel_2": _transport_projection(
            historical_panel_2_run,
            expected_plan_sha256=historical_panel_2_plan_sha256,
            expected_route=ROUTE_TAG,
            expected_completed=704,
            expected_failed=0,
        ),
    }
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
        raise SelectionRouteManifestV61Error("GMICloud route is not uniquely executable")
    endpoint = matches[0]
    endpoint["supported_parameters"] = sorted(
        str(value) for value in endpoint.get("supported_parameters") or []
    )

    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    source_rows = {str(row["model"]["id"]): row for row in source["models"]}
    row = rows[DEEPSEEK_PRO_MODEL_ID]
    max_output_tokens = _max_output_tokens(row)
    row["endpoint"] = endpoint
    row["endpoint_document_sha256"] = _sha256(endpoint)
    row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    row["endpoint_selection"] = {
        "method": "score-blind complete-block transport selection",
        "selected_exact_tag": ROUTE_TAG,
        "superseded_exact_tags": ["deepseek", BASE_TEN_ROUTE],
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
        "selection_reason": "score-blind full-block transport reliability",
    }
    row["contract_evidence"] = {
        "status": "route_frozen_before_complete_replacement_blocks",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_complete_primary_and_repeat_blocks": True,
    }
    row["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "new_provider_calls": EXPECTED_CELLS_PER_PANEL,
        "prompt_token_bound": 4_096,
        "route_max_output_tokens": max_output_tokens,
        "worst_case_usd": _block_envelope(endpoint, max_output_tokens=max_output_tokens),
    }
    if any(
        rows[model_id] != source_row
        for model_id, source_row in source_rows.items()
        if model_id != DEEPSEEK_PRO_MODEL_ID
    ):
        raise SelectionRouteManifestV61Error("v61 changed a non-DeepSeek model")

    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_deepseek_complete_block_repair_v61",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "deepseek_complete_block_repair_v61": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "baseten_transport_projections": baseten,
                "historical_gmicloud_transport_projections": historical,
                "changed_model_ids": [DEEPSEEK_PRO_MODEL_ID],
                "selected_exact_tag": ROUTE_TAG,
                "superseded_exact_tags": ["deepseek", BASE_TEN_ROUTE],
                "automatic_fallback": False,
                "selection_uses_transport_status_only": True,
                "quality_scores_or_selections_used": False,
                "selective_failed_cell_retry": False,
                "complete_replacement_blocks_required": True,
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
        raise SelectionRouteManifestV61Error("constructed v61 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["deepseek_complete_block_repair_v61"]
        row = next(
            value for value in document["models"] if value["model"]["id"] == DEEPSEEK_PRO_MODEL_ID
        )
        baseten = refresh["baseten_transport_projections"]
        historical = refresh["historical_gmicloud_transport_projections"]
    except (KeyError, StopIteration, TypeError):
        return False
    expected = {
        ("baseten", "panel_1"): (BASE_TEN_ROUTE, 70, 634),
        ("baseten", "panel_2"): (BASE_TEN_ROUTE, 72, 632),
        ("historical", "panel_1"): (ROUTE_TAG, 699, 5),
        ("historical", "panel_2"): (ROUTE_TAG, 704, 0),
    }
    projections = {"baseten": baseten, "historical": historical}
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(document.get("models") or []) == 26
        and row["endpoint"].get("tag") == ROUTE_TAG
        and row["endpoint"].get("provider_name") == PROVIDER_NAME
        and row["request_policy"]["provider"].get("only") == [ROUTE_TAG]
        and row["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("changed_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and refresh.get("selected_exact_tag") == ROUTE_TAG
        and refresh.get("superseded_exact_tags") == ["deepseek", BASE_TEN_ROUTE]
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_transport_status_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("complete_replacement_blocks_required") is True
        and refresh.get("superseded_responses_used") is False
        and refresh.get("cross_route_response_pooling") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and all(
            projections[group][panel].get("provider_tag") == route
            and projections[group][panel].get("scheduled_cells") == EXPECTED_CELLS_PER_PANEL
            and projections[group][panel].get("completed_response_artifacts") == completed
            and projections[group][panel].get("failed_response_artifacts") == failed
            and projections[group][panel].get("quality_scores_or_selections_read") is False
            for (group, panel), (route, completed, failed) in expected.items()
        )
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
            raise SelectionRouteManifestV61Error("content-addressed manifest conflict")
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
    parser.add_argument("--baseten-panel-1-run", type=Path, required=True)
    parser.add_argument("--baseten-panel-1-plan-sha256", required=True)
    parser.add_argument("--baseten-panel-2-run", type=Path, required=True)
    parser.add_argument("--baseten-panel-2-plan-sha256", required=True)
    parser.add_argument("--historical-panel-1-run", type=Path, required=True)
    parser.add_argument("--historical-panel-1-plan-sha256", required=True)
    parser.add_argument("--historical-panel-2-run", type=Path, required=True)
    parser.add_argument("--historical-panel-2-plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_path=args.source,
            baseten_panel_1_run=args.baseten_panel_1_run,
            baseten_panel_1_plan_sha256=args.baseten_panel_1_plan_sha256,
            baseten_panel_2_run=args.baseten_panel_2_run,
            baseten_panel_2_plan_sha256=args.baseten_panel_2_plan_sha256,
            historical_panel_1_run=args.historical_panel_1_run,
            historical_panel_1_plan_sha256=args.historical_panel_1_plan_sha256,
            historical_panel_2_run=args.historical_panel_2_run,
            historical_panel_2_plan_sha256=args.historical_panel_2_plan_sha256,
        )
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
