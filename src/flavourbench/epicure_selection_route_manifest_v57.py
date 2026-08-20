"""Freeze the score-blind DeepSeek complete-block transport successor.

The v54 exact ``deepseek`` route returned a failed response artifact for every
scheduled DeepSeek V4 Pro cell in both panels.  This successor changes only
that route to the independently executable BaseTen FP4 endpoint.  It is frozen
before replacement generation and never reads Epicure scores or selections.
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
from .epicure_selection_route_manifest_v54 import (
    DEEPSEEK_PRO_MODEL_ID,
)
from .epicure_selection_route_manifest_v54 import (
    verify_manifest as verify_manifest_v54,
)
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-deepseek-complete-block-repair-v57"
ROUTE_TAG = "baseten/fp4"
PROVIDER_NAME = "BaseTen"
SUPERSEDED_TAG = "deepseek"
EXPECTED_CELLS_PER_PANEL = 704


class SelectionRouteManifestV57Error(RuntimeError):
    """The DeepSeek complete-block route successor could not be frozen."""


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
        raise SelectionRouteManifestV57Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRouteManifestV57Error("manifest input is not a JSON object")
    return value


def _failed_block_projection(run_directory: Path, *, expected_plan_sha256: str) -> dict[str, Any]:
    responses = run_directory / "responses"
    if responses.is_symlink() or not responses.is_dir():
        raise SelectionRouteManifestV57Error("repair response directory is unavailable")
    records: list[dict[str, Any]] = []
    for path in sorted(responses.rglob("*.json")):
        value = _load(path)
        if value.get("model_id") == DEEPSEEK_PRO_MODEL_ID:
            records.append(value)
    if len(records) != EXPECTED_CELLS_PER_PANEL:
        raise SelectionRouteManifestV57Error("DeepSeek failed block is not complete")
    if any(
        value.get("status") != "failed"
        or value.get("provider_route") != SUPERSEDED_TAG
        or value.get("plan_sha256") != expected_plan_sha256
        for value in records
    ):
        raise SelectionRouteManifestV57Error("DeepSeek failure projection differs")
    cell_ids = sorted(str(value["cell_id"]) for value in records)
    if len(set(cell_ids)) != EXPECTED_CELLS_PER_PANEL:
        raise SelectionRouteManifestV57Error("DeepSeek failure block has duplicate cells")
    return {
        "plan_sha256": expected_plan_sha256,
        "model_id": DEEPSEEK_PRO_MODEL_ID,
        "provider_tag": SUPERSEDED_TAG,
        "scheduled_cells": EXPECTED_CELLS_PER_PANEL,
        "failed_response_artifacts": EXPECTED_CELLS_PER_PANEL,
        "completed_response_artifacts": 0,
        "quality_scores_or_selections_read": False,
        "cell_id_set_sha256": hashlib.sha256("\n".join(cell_ids).encode()).hexdigest(),
    }


def _max_output_tokens(row: Mapping[str, Any]) -> int:
    forecast = row.get("forecast") or {}
    value = forecast.get("route_max_output_tokens")
    return int(value) if isinstance(value, int) and value > 0 else 16_384


def _block_envelope(endpoint: Mapping[str, Any], *, max_output_tokens: int) -> str:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * 4_096
    completion = Decimal(str(pricing.get("completion") or 0)) * max_output_tokens
    reasoning = Decimal(str(pricing.get("internal_reasoning") or 0)) * max_output_tokens
    request = Decimal(str(pricing.get("request") or 0))
    return format((prompt + completion + reasoning + request) * EXPECTED_CELLS_PER_PANEL, "f")


async def build(
    *,
    source_path: Path,
    panel_1_run_directory: Path,
    panel_1_plan_sha256: str,
    panel_2_run_directory: Path,
    panel_2_plan_sha256: str,
) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v54(source):
        raise SelectionRouteManifestV57Error("v57 requires the exact v54 route predecessor")
    panel_1_failure = _failed_block_projection(
        panel_1_run_directory, expected_plan_sha256=panel_1_plan_sha256
    )
    panel_2_failure = _failed_block_projection(
        panel_2_run_directory, expected_plan_sha256=panel_2_plan_sha256
    )
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
        raise SelectionRouteManifestV57Error("BaseTen route is not uniquely executable")
    endpoint = matches[0]
    endpoint["supported_parameters"] = sorted(
        str(value) for value in endpoint.get("supported_parameters") or []
    )

    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    row = rows[DEEPSEEK_PRO_MODEL_ID]
    max_output_tokens = _max_output_tokens(row)
    row["endpoint"] = endpoint
    row["endpoint_document_sha256"] = _sha256(endpoint)
    row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    row["endpoint_selection"] = {
        "method": "score-blind all-cell replacement after deterministic route failure",
        "selected_exact_tag": ROUTE_TAG,
        "superseded_exact_tags": [SUPERSEDED_TAG],
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
        "selection_reason": "score-blind deterministic complete-block transport repair",
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

    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_deepseek_complete_block_repair_v57",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "deepseek_complete_block_repair_v57": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "failed_block_projections": {
                    "panel_1": panel_1_failure,
                    "panel_2": panel_2_failure,
                },
                "changed_model_ids": [DEEPSEEK_PRO_MODEL_ID],
                "selected_exact_tag": ROUTE_TAG,
                "superseded_exact_tag": SUPERSEDED_TAG,
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
        raise SelectionRouteManifestV57Error("constructed v57 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["deepseek_complete_block_repair_v57"]
        rows = {str(row["model"]["id"]): row for row in document["models"]}
        row = rows[DEEPSEEK_PRO_MODEL_ID]
        failures = refresh["failed_block_projections"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 26
        and row["endpoint"].get("tag") == ROUTE_TAG
        and row["endpoint"].get("provider_name") == PROVIDER_NAME
        and row["request_policy"]["provider"].get("only") == [ROUTE_TAG]
        and row["request_policy"]["provider"].get("allow_fallbacks") is False
        and refresh.get("changed_model_ids") == [DEEPSEEK_PRO_MODEL_ID]
        and refresh.get("selected_exact_tag") == ROUTE_TAG
        and refresh.get("superseded_exact_tag") == SUPERSEDED_TAG
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_transport_status_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("complete_replacement_blocks_required") is True
        and refresh.get("superseded_responses_used") is False
        and refresh.get("cross_route_response_pooling") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and all(
            failures[panel].get("scheduled_cells") == EXPECTED_CELLS_PER_PANEL
            and failures[panel].get("failed_response_artifacts") == EXPECTED_CELLS_PER_PANEL
            and failures[panel].get("completed_response_artifacts") == 0
            and failures[panel].get("quality_scores_or_selections_read") is False
            for panel in ("panel_1", "panel_2")
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
            raise SelectionRouteManifestV57Error("content-addressed manifest conflict")
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
    parser.add_argument("--panel-1-run-directory", type=Path, required=True)
    parser.add_argument("--panel-1-plan-sha256", required=True)
    parser.add_argument("--panel-2-run-directory", type=Path, required=True)
    parser.add_argument("--panel-2-plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_path=args.source,
            panel_1_run_directory=args.panel_1_run_directory,
            panel_1_plan_sha256=args.panel_1_plan_sha256,
            panel_2_run_directory=args.panel_2_run_directory,
            panel_2_plan_sha256=args.panel_2_plan_sha256,
        )
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
