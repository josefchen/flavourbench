"""Replace quota-capped direct Cohere routes with exact OpenRouter routes.

The two direct-Cohere primary blocks remain immutable transport evidence.  They
are never pooled with the successor data.  Command A and Command R+ are rerun
from zero on one exact Cohere endpoint each through OpenRouter.
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
from decimal import Decimal
from pathlib import Path
from typing import Any

from .epicure_native_powered_runner import _semantic_valid
from .epicure_selection_route_manifest import _fetch_endpoints, _select_exact
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-selection-route-refresh-v16"
DIRECT_PROVIDER_TAG = "cohere-direct"
DIRECT_BACKEND = "cohere_direct"
OPENROUTER_PROVIDER_TAG = "cohere"
OPENROUTER_PROVIDER_NAME = "Cohere"
ROUTE_MAX_OUTPUT_TOKENS = 1_800
REQUESTS_PER_BLOCK = 704
PROMPT_TOKEN_BOUND = 4_096

REPLACEMENTS = {
    "cohere/command-a-plus-05-2026": {
        "model_id": "cohere/command-a",
        "canonical_slug": "cohere/command-a-03-2025",
        "expected_responses": 572,
        "expected_completed": 550,
        "expected_failed": 22,
    },
    "cohere/command-a-reasoning-08-2025": {
        "model_id": "cohere/command-r-plus-08-2024",
        "canonical_slug": "cohere/command-r-plus-08-2024",
        "expected_responses": 590,
        "expected_completed": 579,
        "expected_failed": 11,
    },
}


class SelectionRouteManifestV32Error(RuntimeError):
    """The OpenRouter Cohere route successor failed verification."""


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
        raise SelectionRouteManifestV32Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV32Error("source manifest content address is invalid")
    return value


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _block_envelope(endpoint: Mapping[str, Any]) -> Decimal:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * PROMPT_TOKEN_BOUND
    completion = Decimal(str(pricing.get("completion") or 0)) * ROUTE_MAX_OUTPUT_TOKENS
    return (prompt + completion) * REQUESTS_PER_BLOCK


def direct_block_commitment(
    run_directory: Path,
    *,
    model_id: str,
    expected_plan_sha256: str,
    expected_responses: int,
) -> dict[str, Any]:
    """Bind one quota-capped direct block and its exact attempt lineage."""

    responses: list[dict[str, Any]] = []
    for path in sorted((run_directory / "responses/primary").glob("*/response-*.json")):
        if path.is_symlink() or not path.is_file():
            raise SelectionRouteManifestV32Error("response input is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not _semantic_valid(value):
            raise SelectionRouteManifestV32Error("direct Cohere response failed integrity")
        if value.get("model_id") == model_id:
            responses.append(value)
    if len(responses) != expected_responses:
        raise SelectionRouteManifestV32Error("unexpected direct Cohere response count")
    if len({str(row.get("arm_id")) for row in responses}) != expected_responses:
        raise SelectionRouteManifestV32Error("direct Cohere arm IDs are not unique")
    if len({str(row.get("task_id")) for row in responses}) != expected_responses:
        raise SelectionRouteManifestV32Error("direct Cohere task IDs are not unique")
    if not all(
        row.get("plan_sha256") == expected_plan_sha256
        and row.get("panel") == "primary"
        and row.get("provider_route") == DIRECT_PROVIDER_TAG
        and row.get("execution_backend") == DIRECT_BACKEND
        and row.get("status") in {"completed", "failed"}
        for row in responses
    ):
        raise SelectionRouteManifestV32Error("direct Cohere response scope changed")

    arm_ids = {str(row["arm_id"]) for row in responses}
    journal_path = run_directory / "attempts/provider-attempts.jsonl"
    if journal_path.is_symlink() or not journal_path.is_file():
        raise SelectionRouteManifestV32Error("attempt journal is unavailable")
    events_by_arm: dict[str, list[dict[str, Any]]] = {arm_id: [] for arm_id in arm_ids}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        payload = dict(row)
        recorded = str(payload.pop("event_sha256", ""))
        if recorded != _sha256(payload):
            raise SelectionRouteManifestV32Error("attempt journal failed integrity")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise SelectionRouteManifestV32Error("attempt journal event is malformed")
        arm_id = str(event.get("arm_id") or "")
        if arm_id in events_by_arm:
            if payload.get("plan_sha256") != expected_plan_sha256:
                raise SelectionRouteManifestV32Error("direct Cohere attempt plan changed")
            events_by_arm[arm_id].append(row)

    event_hashes: list[str] = []
    event_types: Counter[str] = Counter()
    rejection_statuses: Counter[str] = Counter()
    for response in responses:
        observed = events_by_arm[str(response["arm_id"])]
        expected_hashes = [str(value) for value in response.get("attempt_event_sha256s") or []]
        observed_hashes = [str(row["event_sha256"]) for row in observed]
        if not expected_hashes or observed_hashes != expected_hashes:
            raise SelectionRouteManifestV32Error("response-to-attempt lineage changed")
        event_hashes.extend(observed_hashes)
        for row in observed:
            event = row["event"]
            kind = str(event["event_type"])
            event_types[kind] += 1
            if kind == "request_rejected":
                rejection_statuses[str(event.get("http_status"))] += 1

    statuses = Counter(str(row["status"]) for row in responses)
    errors = Counter(
        str((row.get("error") or {}).get("type") or "none")
        for row in responses
        if row["status"] == "failed"
    )
    return {
        "model_id": model_id,
        "plan_sha256": expected_plan_sha256,
        "response_count": len(responses),
        "completed_count": statuses["completed"],
        "failed_count": statuses["failed"],
        "failed_task_ids": sorted(
            str(row["task_id"]) for row in responses if row["status"] == "failed"
        ),
        "failure_type_counts": dict(sorted(errors.items())),
        "response_artifact_set_sha256": _sha256(
            sorted(str(row["artifact_sha256"]) for row in responses)
        ),
        "task_id_set_sha256": _sha256(sorted(str(row["task_id"]) for row in responses)),
        "attempt_event_count": len(event_hashes),
        "attempt_event_sequence_sha256": _sha256(event_hashes),
        "event_type_counts": dict(sorted(event_types.items())),
        "request_rejection_http_status_counts": dict(sorted(rejection_statuses.items())),
        "spend_micros": sum(
            int((row.get("generation") or {}).get("cost_micros") or 0) for row in responses
        ),
        "used_as_successor_score_data": False,
        "excluded_from_all_model_score_and_rank_inference": True,
        "exclusion_reason": "provider monthly request cap prevented a complete direct block",
    }


def _replacement_entry(
    source: Mapping[str, Any],
    *,
    replacement: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    value = copy.deepcopy(source)
    new_model_id = str(replacement["model_id"])
    canonical_slug = str(replacement["canonical_slug"])
    endpoint = dict(endpoint)
    if (
        endpoint.get("model_id") != new_model_id
        or endpoint.get("provider_name") != OPENROUTER_PROVIDER_NAME
        or endpoint.get("tag") != OPENROUTER_PROVIDER_TAG
        or canonical_slug not in str(endpoint.get("name") or "")
    ):
        raise SelectionRouteManifestV32Error("OpenRouter Cohere endpoint identity changed")
    supported = sorted(str(item) for item in endpoint.get("supported_parameters") or [])
    value["slot"].update(
        {
            "model_id": new_model_id,
            "open_weight_candidate": True,
            "rationale": (
                "Exact Cohere endpoint selected through OpenRouter after the direct account hit "
                "its monthly request limit; the full block is rerun without response pooling."
            ),
        }
    )
    value["model"] = {
        "id": new_model_id,
        "canonical_slug": canonical_slug,
        "name": str(endpoint["model_name"]),
        "description": (
            "Exact Cohere release served by the sole healthy Cohere endpoint exposed by "
            "OpenRouter at route freeze."
        ),
        "context_length": int(endpoint["context_length"]),
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "modality": "text->text",
        },
        "supported_parameters": supported,
    }
    value["endpoint"] = endpoint
    value["endpoint_document_sha256"] = _sha256(endpoint)
    value["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
    value["endpoint_selection"] = {
        "method": "sole exact Cohere endpoint selected before any successor generation",
        "selected_exact_tag": OPENROUTER_PROVIDER_TAG,
        "eligible_endpoint_count": 1,
        "quality_observations_used": 0,
        "observed_at": observed_at,
        "automatic_fallback": False,
    }
    value["request_policy"] = {
        "official_eligibility": "development_only",
        "policy_scope": "request_enforced",
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "only": [OPENROUTER_PROVIDER_TAG],
            "require_parameters": True,
        },
    }
    value["execution_route"] = {
        "preferred_backend": "openrouter",
        "selected_backend": "openrouter",
        "policy": "exact_openrouter_provider_only_v1",
        "fallback_used": False,
        "generation_time_automatic_fallback": False,
        "selection_frozen_before_generation": True,
        "selection_reason": "direct Cohere monthly cap; full clean OpenRouter successor block",
    }
    value["contract_evidence"] = {
        "status": "route_selected_before_any_successor_response",
        "generation_calls": 0,
        "quality_observations": 0,
        "requires_successful_eight_cell_transport_check": True,
        "requires_full_primary_and_repeat_block_rerun": True,
    }
    value["backend_contract"] = {}
    value["backend_contract_sha256"] = "unfrozen"
    value["cost_accounting_policy"] = "provider_generation_metadata"
    envelope = _block_envelope(endpoint)
    value["forecast"] = {
        "primary_tasks": 640,
        "repeat_tasks": 64,
        "prompt_token_bound": PROMPT_TOKEN_BOUND,
        "route_max_output_tokens": ROUTE_MAX_OUTPUT_TOKENS,
        "model_block_worst_case_usd": _decimal_text(envelope),
    }
    return value


async def build(
    *,
    source_path: Path,
    failed_run_directory: Path,
    failed_plan_sha256: str,
) -> dict[str, Any]:
    source = _load(source_path)
    if source.get("schema_version") != SCHEMA_VERSION or len(source.get("models") or []) != 20:
        raise SelectionRouteManifestV32Error("source is not the exact 20-model routed manifest")
    endpoint_sets = await asyncio.gather(
        *(_fetch_endpoints(str(spec["model_id"])) for spec in REPLACEMENTS.values())
    )
    endpoints = {
        str(spec["model_id"]): _select_exact(
            model_id=str(spec["model_id"]),
            tag=OPENROUTER_PROVIDER_TAG,
            endpoints=rows,
        )
        for spec, rows in zip(REPLACEMENTS.values(), endpoint_sets, strict=True)
    }
    failures = {
        old_id: direct_block_commitment(
            failed_run_directory,
            model_id=old_id,
            expected_plan_sha256=failed_plan_sha256,
            expected_responses=int(spec["expected_responses"]),
        )
        for old_id, spec in REPLACEMENTS.items()
    }
    for old_id, spec in REPLACEMENTS.items():
        failure = failures[old_id]
        if (
            failure["completed_count"] != spec["expected_completed"]
            or failure["failed_count"] != spec["expected_failed"]
        ):
            raise SelectionRouteManifestV32Error("direct Cohere failure boundary changed")

    document = copy.deepcopy(source)
    document.pop("content_address", None)
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    replaced = 0
    for index, entry in enumerate(document["models"]):
        old_id = str(entry["model"]["id"])
        spec = REPLACEMENTS.get(old_id)
        if spec is None:
            continue
        if (
            entry["execution_route"]["selected_backend"] != DIRECT_BACKEND
            or entry["endpoint"]["tag"] != DIRECT_PROVIDER_TAG
        ):
            raise SelectionRouteManifestV32Error("unexpected direct Cohere predecessor route")
        document["models"][index] = _replacement_entry(
            entry,
            replacement=spec,
            endpoint=endpoints[str(spec["model_id"])],
            observed_at=observed_at,
        )
        replaced += 1
    if replaced != len(REPLACEMENTS):
        raise SelectionRouteManifestV32Error("direct Cohere slots were not uniquely replaced")

    budget_rows = dict(source["budget"]["per_model_worst_case_usd"])
    for model_id in (
        *REPLACEMENTS,
        "anthropic/claude-fable-5",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ):
        budget_rows.pop(model_id, None)
    for model_id in ("meta-llama/llama-4-maverick", "nvidia/nemotron-3.5-lightning"):
        entry = next(row for row in document["models"] if row["model"]["id"] == model_id)
        budget_rows[model_id] = str(entry["forecast"]["model_block_worst_case_usd"])
    for spec in REPLACEMENTS.values():
        model_id = str(spec["model_id"])
        budget_rows[model_id] = _decimal_text(_block_envelope(endpoints[model_id]))
    if len(budget_rows) != 20:
        raise SelectionRouteManifestV32Error("successor budget roster is not exact")
    bounded = sum((Decimal(value) for value in budget_rows.values()), Decimal(0))
    admission_ceiling = Decimal("85")
    if bounded > admission_ceiling:
        raise SelectionRouteManifestV32Error("successor route forecast exceeds admission ceiling")
    document["budget"].update(
        {
            "per_model_worst_case_usd": dict(sorted(budget_rows.items())),
            "bounded_forecast_usd": _decimal_text(bounded),
            "headroom_to_admission_ceiling_usd": _decimal_text(admission_ceiling - bounded),
            "within_cap": True,
        }
    )
    prior_routes = {
        old_id: {
            "provider_tag": DIRECT_PROVIDER_TAG,
            "execution_backend": DIRECT_BACKEND,
            "response_commitment": failures[old_id],
        }
        for old_id in REPLACEMENTS
    }
    replacement_routes = {
        str(spec["model_id"]): {
            "provider_tag": OPENROUTER_PROVIDER_TAG,
            "provider_name": OPENROUTER_PROVIDER_NAME,
            "execution_backend": "openrouter",
            "canonical_model_slug": spec["canonical_slug"],
            "route_max_output_tokens": ROUTE_MAX_OUTPUT_TOKENS,
            "endpoint_execution_sha256": endpoint_execution_contract_sha256(
                endpoints[str(spec["model_id"])]
            ),
        }
        for spec in REPLACEMENTS.values()
    }
    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "epicure_selection_powered_route_successor_v16",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "route_refresh": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": _sha256_file(source_path),
                "direct_quota_failure_observations": prior_routes,
                "current_endpoint_network_reads": 2,
                "successor_provider_calls": 0,
                "replacement_routes": replacement_routes,
                "full_successor_primary_cells_per_model": 640,
                "full_successor_repeat_cells_per_model": 64,
                "direct_responses_excluded": True,
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
        raise SelectionRouteManifestV32Error("route manifest failed content verification")
    return document


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["content_address"]["digest"])
    destination = directory / f"flavourbench-selection-route-manifest-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionRouteManifestV32Error("content-addressed manifest conflict")
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
