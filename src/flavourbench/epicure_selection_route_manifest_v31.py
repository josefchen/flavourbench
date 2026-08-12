"""Replace the failed Novita DeepSeek route with exact CoreWeave FP8.

The predecessor DeepSeek block is retained as transport-failure evidence but is
not pooled with the successor block.  The successor must rerun all 640 primary
tasks and all 64 repeat tasks on one exact provider route.
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

from .epicure_native_powered_runner import _semantic_valid
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
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v15"
PREDECESSOR_TAG = "novita/fp8"
PREDECESSOR_PROVIDER = "Novita"
REPLACEMENT_TAG = "coreweave/fp8"
REPLACEMENT_PROVIDER = "CoreWeave"
EXPECTED_PREDECESSOR_RESPONSES = 560


class SelectionRouteManifestV31Error(RuntimeError):
    """The CoreWeave DeepSeek route successor failed verification."""


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
        raise SelectionRouteManifestV31Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV31Error("source manifest content address is invalid")
    return value


def deepseek_failure_commitment(
    run_directory: Path,
    *,
    expected_plan_sha256: str,
    expected_responses: int = EXPECTED_PREDECESSOR_RESPONSES,
) -> dict[str, Any]:
    """Commit only the failed-route DeepSeek block from an append-only run."""

    responses: list[dict[str, Any]] = []
    for path in sorted((run_directory / "responses/primary").glob("*/response-*.json")):
        if path.is_symlink() or not path.is_file():
            raise SelectionRouteManifestV31Error("response input is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not _semantic_valid(value):
            raise SelectionRouteManifestV31Error("predecessor response failed integrity")
        if value.get("model_id") == DEEPSEEK_PRO_MODEL_ID:
            responses.append(value)
    if len(responses) != expected_responses:
        raise SelectionRouteManifestV31Error("unexpected predecessor DeepSeek response count")
    if len({str(row.get("arm_id")) for row in responses}) != expected_responses:
        raise SelectionRouteManifestV31Error("predecessor DeepSeek arm IDs are not unique")
    if len({str(row.get("task_id")) for row in responses}) != expected_responses:
        raise SelectionRouteManifestV31Error("predecessor DeepSeek task IDs are not unique")
    if not all(
        row.get("plan_sha256") == expected_plan_sha256
        and row.get("panel") == "primary"
        and row.get("provider_route") == PREDECESSOR_TAG
        and row.get("execution_backend") == "openrouter"
        and row.get("status") in {"completed", "failed"}
        for row in responses
    ):
        raise SelectionRouteManifestV31Error("predecessor DeepSeek response scope changed")

    arm_ids = {str(row["arm_id"]) for row in responses}
    journal_path = run_directory / "attempts/provider-attempts.jsonl"
    if journal_path.is_symlink() or not journal_path.is_file():
        raise SelectionRouteManifestV31Error("attempt journal is unavailable")
    events_by_arm: dict[str, list[dict[str, Any]]] = {arm_id: [] for arm_id in arm_ids}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        payload = dict(row)
        recorded = str(payload.pop("event_sha256", ""))
        if recorded != _sha256(payload):
            raise SelectionRouteManifestV31Error("attempt journal failed integrity")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise SelectionRouteManifestV31Error("attempt journal event is malformed")
        arm_id = str(event.get("arm_id") or "")
        if arm_id in events_by_arm:
            if payload.get("plan_sha256") != expected_plan_sha256:
                raise SelectionRouteManifestV31Error("DeepSeek attempt plan binding changed")
            events_by_arm[arm_id].append(row)

    response_event_hashes: list[str] = []
    journal_event_hashes: list[str] = []
    event_types: Counter[str] = Counter()
    rejection_statuses: Counter[str] = Counter()
    for response in responses:
        arm_id = str(response["arm_id"])
        expected = [str(value) for value in response.get("attempt_event_sha256s") or []]
        observed = [str(row["event_sha256"]) for row in events_by_arm[arm_id]]
        if not expected or observed != expected:
            raise SelectionRouteManifestV31Error("response-to-attempt lineage changed")
        response_event_hashes.extend(expected)
        journal_event_hashes.extend(observed)
        for row in events_by_arm[arm_id]:
            event = row["event"]
            kind = str(event["event_type"])
            event_types[kind] += 1
            if kind == "request_rejected":
                rejection_statuses[str(event.get("http_status"))] += 1

    statuses = Counter(str(row["status"]) for row in responses)
    failed_task_ids = sorted(str(row["task_id"]) for row in responses if row["status"] == "failed")
    response_artifacts = sorted(str(row["artifact_sha256"]) for row in responses)
    task_ids = sorted(str(row["task_id"]) for row in responses)
    spend_micros = sum(
        int((row.get("generation") or {}).get("cost_micros") or 0) for row in responses
    )
    return {
        "model_id": DEEPSEEK_PRO_MODEL_ID,
        "plan_sha256": expected_plan_sha256,
        "response_count": len(responses),
        "completed_count": statuses["completed"],
        "failed_count": statuses["failed"],
        "failed_task_ids": failed_task_ids,
        "response_artifact_set_sha256": _sha256(response_artifacts),
        "task_id_set_sha256": _sha256(task_ids),
        "attempt_event_count": len(journal_event_hashes),
        "attempt_event_sequence_sha256": _sha256(journal_event_hashes),
        "response_attempt_sequence_sha256": _sha256(response_event_hashes),
        "event_type_counts": dict(sorted(event_types.items())),
        "request_rejection_http_status_counts": dict(sorted(rejection_statuses.items())),
        "spend_micros": spend_micros,
        "used_as_successor_score_data": False,
        "excluded_from_all_model_score_and_rank_inference": True,
    }


async def build(
    *,
    source_path: Path,
    failed_run_directory: Path,
    failed_plan_sha256: str,
) -> dict[str, Any]:
    source = _load(source_path)
    endpoints = await _fetch_endpoints(DEEPSEEK_PRO_MODEL_ID)
    endpoint = _select_exact(
        model_id=DEEPSEEK_PRO_MODEL_ID,
        tag=REPLACEMENT_TAG,
        endpoints=endpoints,
    )
    if endpoint.get("provider_name") != REPLACEMENT_PROVIDER:
        raise SelectionRouteManifestV31Error("CoreWeave endpoint identity changed")
    failure = deepseek_failure_commitment(
        failed_run_directory,
        expected_plan_sha256=failed_plan_sha256,
    )
    if failure["completed_count"] != 546 or failure["failed_count"] != 14:
        raise SelectionRouteManifestV31Error("DeepSeek failure boundary changed")

    document = copy.deepcopy(source)
    document.pop("content_address", None)
    matches = [
        entry for entry in document["models"] if entry["model"]["id"] == DEEPSEEK_PRO_MODEL_ID
    ]
    if len(matches) != 1:
        raise SelectionRouteManifestV31Error("DeepSeek V4 Pro slot is not unique")
    entry = matches[0]
    if (
        entry["endpoint"]["tag"] != PREDECESSOR_TAG
        or entry["endpoint"]["provider_name"] != PREDECESSOR_PROVIDER
        or entry["model"]["canonical_slug"] != EXPECTED_ACTUAL_MODEL_ID
    ):
        raise SelectionRouteManifestV31Error("unexpected Novita predecessor route")
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
        "method": "replace_systematic_novita_http_404_route_before_full_block_rerun",
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
            "selection_reason": (
                "replace the failed Novita route with exact CoreWeave FP8 and rerun the full "
                "DeepSeek block without cross-provider pooling"
            ),
        }
    )
    entry["contract_evidence"] = {
        "status": "route_selected_before_any_coreweave_successor_response",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_successful_eight_cell_transport_check": True,
        "requires_full_primary_and_repeat_block_rerun": True,
    }
    entry["backend_contract"] = {}
    entry["backend_contract_sha256"] = "unfrozen"
    entry["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "model_block_worst_case_usd": _block_envelope_usd(endpoint),
    }
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v15",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "failed_route_observation": failure,
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
                "full_successor_primary_cells": 640,
                "full_successor_repeat_cells": 64,
                "prior_route_responses_excluded": True,
                "cross_provider_score_pooling": False,
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
        raise SelectionRouteManifestV31Error("route manifest failed content verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV31Error("content-addressed manifest conflict")
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
    parser.add_argument("--failed-run-directory", type=Path, required=True)
    parser.add_argument("--failed-plan-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = asyncio.run(
        build(
            source_path=args.source_manifest,
            failed_run_directory=args.failed_run_directory,
            failed_plan_sha256=args.failed_plan_sha256,
        )
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
