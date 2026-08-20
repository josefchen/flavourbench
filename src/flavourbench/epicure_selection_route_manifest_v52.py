"""Freeze score-blind replication-2 route repairs for Luna and DeepSeek Flash.

The running v49 replication established transport failures before any joint
analysis: OpenAI Flex rejected the Luna block and DeepInfra's former fp4 tag
returned 404 after the provider replaced it with fp8.  This successor changes
only those two exact provider routes and requires complete replacement blocks.
No Epicure score or model selection is read by this module.
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
from .epicure_selection_route_manifest_v49 import verify_manifest as verify_manifest_v49
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-replication-2-route-repair-v52"
LUNA_MODEL_ID = "openai/gpt-5.6-luna-pro"
DEEPSEEK_FLASH_MODEL_ID = "deepseek/deepseek-v4-flash-0731"
REPLACEMENT_MODEL_IDS = [LUNA_MODEL_ID, DEEPSEEK_FLASH_MODEL_ID]
ROUTE_SPECS: dict[str, dict[str, Any]] = {
    LUNA_MODEL_ID: {
        "tag": "openai",
        "provider": "OpenAI",
        "reasoning_effort": "minimal",
        "max_output_tokens": 2_048,
        "superseded_tag": "openai/flex",
        "transport_finding": "quota_rejection",
    },
    DEEPSEEK_FLASH_MODEL_ID: {
        "tag": "deepinfra/fp8",
        "provider": "DeepInfra",
        "reasoning_effort": "none",
        "max_output_tokens": 2_048,
        "superseded_tag": "deepinfra/fp4",
        "transport_finding": "endpoint_not_found",
    },
}


class SelectionRouteManifestV52Error(RuntimeError):
    """The replication-2 route repair manifest could not be frozen."""


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
        raise SelectionRouteManifestV52Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionRouteManifestV52Error("manifest input is not a JSON object")
    return value


def _block_envelope(endpoint: Mapping[str, Any], *, max_output_tokens: int) -> str:
    pricing = endpoint.get("pricing") or {}
    prompt = Decimal(str(pricing.get("prompt") or 0)) * 4_096
    completion = Decimal(str(pricing.get("completion") or 0)) * max_output_tokens
    reasoning = Decimal(str(pricing.get("internal_reasoning") or 0)) * max_output_tokens
    request = Decimal(str(pricing.get("request") or 0))
    return format((prompt + completion + reasoning + request) * 704, "f")


async def build(*, source_path: Path) -> dict[str, Any]:
    source = _load(source_path)
    if not verify_manifest_v49(source):
        raise SelectionRouteManifestV52Error("v52 requires the exact v49 route predecessor")
    fetched = await asyncio.gather(
        *(_fetch_endpoints(model_id) for model_id in REPLACEMENT_MODEL_IDS)
    )
    selected: dict[str, dict[str, Any]] = {}
    for model_id, endpoints in zip(REPLACEMENT_MODEL_IDS, fetched, strict=True):
        spec = ROUTE_SPECS[model_id]
        matches = [
            copy.deepcopy(endpoint)
            for endpoint in endpoints
            if endpoint.get("tag") == spec["tag"]
            and endpoint.get("provider_name") == spec["provider"]
            and endpoint.get("status") == 0
            and {"max_tokens", "tools"} <= set(endpoint.get("supported_parameters") or [])
        ]
        if len(matches) != 1:
            raise SelectionRouteManifestV52Error(
                f"replacement route is not uniquely executable for {model_id}"
            )
        endpoint = matches[0]
        endpoint["supported_parameters"] = sorted(
            str(value) for value in endpoint.get("supported_parameters") or []
        )
        selected[model_id] = endpoint

    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    rows = {str(row["model"]["id"]): row for row in document["models"]}
    if not set(REPLACEMENT_MODEL_IDS) <= set(rows):
        raise SelectionRouteManifestV52Error("replacement models are absent from predecessor")
    for model_id in REPLACEMENT_MODEL_IDS:
        spec = ROUTE_SPECS[model_id]
        endpoint = selected[model_id]
        row = rows[model_id]
        row["endpoint"] = endpoint
        row["endpoint_document_sha256"] = _sha256(endpoint)
        row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
        row["endpoint_selection"] = {
            "method": (
                "score-blind transport-failure replacement plus current public endpoint contract"
            ),
            "selected_exact_tag": spec["tag"],
            "superseded_exact_tag": spec["superseded_tag"],
            "transport_finding": spec["transport_finding"],
            "eligible_endpoint_count": 1,
            "observed_at": observed_at,
            "automatic_fallback": False,
            "quality_scores_or_selections_used": False,
        }
        row["request_policy"]["provider"] = {
            "only": [spec["tag"]],
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
            "selection_reason": "score-blind transport availability repair",
        }
        row["contract_evidence"] = {
            "status": "route_frozen_before_complete_replacement_block",
            "generation_calls": 0,
            "quality_observations": 0,
            "requires_complete_primary_and_repeat_blocks": True,
        }
        row["forecast"] = {
            "primary_tasks": 640,
            "repeat_tasks": 64,
            "new_provider_calls": 704,
            "prompt_token_bound": 4_096,
            "route_max_output_tokens": spec["max_output_tokens"],
            "worst_case_usd": _block_envelope(
                endpoint, max_output_tokens=int(spec["max_output_tokens"])
            ),
        }

    document.update(
        {
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_replication_2_route_repair_v52",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "replication_route_repair_v52": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest": {
                    "semantic_sha256": source["content_address"]["digest"],
                    "physical_sha256": _sha256_file(source_path),
                },
                "changed_model_ids": REPLACEMENT_MODEL_IDS,
                "selected_exact_tags": {
                    model_id: ROUTE_SPECS[model_id]["tag"] for model_id in REPLACEMENT_MODEL_IDS
                },
                "superseded_exact_tags": {
                    model_id: ROUTE_SPECS[model_id]["superseded_tag"]
                    for model_id in REPLACEMENT_MODEL_IDS
                },
                "automatic_fallback": False,
                "selection_uses_transport_status_only": True,
                "quality_scores_or_selections_used": False,
                "selective_failed_cell_retry": False,
                "complete_replacement_blocks_required": True,
                "superseded_responses_used": False,
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
        raise SelectionRouteManifestV52Error("constructed v52 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["replication_route_repair_v52"]
        rows = {str(row["model"]["id"]): row for row in document["models"]}
    except (KeyError, TypeError):
        return False
    model_checks = []
    for model_id in REPLACEMENT_MODEL_IDS:
        try:
            row = rows[model_id]
            spec = ROUTE_SPECS[model_id]
            model_checks.append(
                row["endpoint"].get("tag") == spec["tag"]
                and row["endpoint"].get("provider_name") == spec["provider"]
                and row["request_policy"]["provider"].get("only") == [spec["tag"]]
                and row["request_policy"]["provider"].get("allow_fallbacks") is False
            )
        except (KeyError, TypeError):
            return False
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("status") == "unranked_candidate"
        and verify_manifest_content_address(document)
        and len(rows) == 26
        and all(model_checks)
        and refresh.get("changed_model_ids") == REPLACEMENT_MODEL_IDS
        and refresh.get("selected_exact_tags")
        == {model_id: ROUTE_SPECS[model_id]["tag"] for model_id in REPLACEMENT_MODEL_IDS}
        and refresh.get("superseded_exact_tags")
        == {model_id: ROUTE_SPECS[model_id]["superseded_tag"] for model_id in REPLACEMENT_MODEL_IDS}
        and refresh.get("automatic_fallback") is False
        and refresh.get("selection_uses_transport_status_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("complete_replacement_blocks_required") is True
        and refresh.get("superseded_responses_used") is False
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
            raise SelectionRouteManifestV52Error("content-addressed manifest conflict")
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
