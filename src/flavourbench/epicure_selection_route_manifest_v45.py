"""Freeze transport-recovery routes for Fable 5 and Qwen3.8 A95B.

The transformation is based only on completion/failure metadata from the
anchor-free collection and the current public endpoint catalogue.  It does not
inspect model selections or Epicure scores.  Both affected models require full
replacement blocks; selective successful-cell pooling is prohibited.
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
from .frontier_manifest import verify_manifest_content_address
from .live_smoke import endpoint_execution_contract_sha256

SCHEMA_VERSION = "flavourbench-routed-candidate-manifest-v1"
REFRESH_SCHEMA_VERSION = "flavourbench-frontier-completion-route-v45"
FABLE_MODEL_ID = "anthropic/claude-fable-5"
QWEN_MODEL_ID = "qwen/qwen3.8-2.4t-a95b"
ROUTE_SPECS = {
    FABLE_MODEL_ID: {
        "tag": "google-vertex/global",
        "provider": "Google",
        "max_output_tokens": 2_048,
        "reasoning_effort": "minimal",
        "reason": "highest current route availability after cross-route refusal diagnostics",
    },
    QWEN_MODEL_ID: {
        "tag": "alibaba",
        "provider": "Alibaba",
        "max_output_tokens": 16_384,
        "reasoning_effort": "low",
        "reason": "first-party route with supported low reasoning and a larger output ceiling",
    },
}


class SelectionRouteManifestV45Error(RuntimeError):
    """The completion-route manifest could not be frozen."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionRouteManifestV45Error("source manifest is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise SelectionRouteManifestV45Error("source manifest failed content verification")
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
    endpoint_sets = await asyncio.gather(*(_fetch_endpoints(model_id) for model_id in ROUTE_SPECS))
    catalogs = dict(zip(ROUTE_SPECS, endpoint_sets, strict=True))
    document = copy.deepcopy(source)
    document.pop("content_address", None)
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    changed: list[str] = []
    route_records: dict[str, Any] = {}
    for row in document["models"]:
        model_id = str(row["model"]["id"])
        if model_id not in ROUTE_SPECS:
            continue
        spec = ROUTE_SPECS[model_id]
        matches = [
            dict(endpoint)
            for endpoint in catalogs[model_id]
            if endpoint.get("tag") == spec["tag"]
            and endpoint.get("provider_name") == spec["provider"]
            and endpoint.get("status") == 0
            and {"max_tokens", "tools"} <= set(endpoint.get("supported_parameters") or [])
        ]
        if len(matches) != 1:
            raise SelectionRouteManifestV45Error(
                f"{model_id} completion route is not uniquely executable"
            )
        endpoint = matches[0]
        endpoint["supported_parameters"] = sorted(
            str(value) for value in endpoint.get("supported_parameters") or []
        )
        row["endpoint"] = endpoint
        row["endpoint_document_sha256"] = _sha256(endpoint)
        row["endpoint_execution_sha256"] = endpoint_execution_contract_sha256(endpoint)
        row["endpoint_selection"] = {
            "method": "score-blind normal-completion recovery plus current public route contract",
            "selected_exact_tag": spec["tag"],
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
            "selection_reason": spec["reason"],
        }
        row["contract_evidence"] = {
            "status": "route_frozen_before_transport_pilot_and_complete_replacement_block",
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
            "model_block_worst_case_usd": _block_envelope(
                endpoint, max_output_tokens=int(spec["max_output_tokens"])
            ),
        }
        changed.append(model_id)
        route_records[model_id] = {
            "provider_tag": spec["tag"],
            "provider_name": spec["provider"],
            "endpoint_sha256": _sha256(endpoint),
            "endpoint_execution_sha256": row["endpoint_execution_sha256"],
            "final_max_output_tokens": spec["max_output_tokens"],
            "final_reasoning_effort": spec["reasoning_effort"],
        }
    if changed != [FABLE_MODEL_ID, QWEN_MODEL_ID]:
        raise SelectionRouteManifestV45Error("source manifest model order changed")
    document.update(
        {
            "schema_version": SCHEMA_VERSION,
            "observed_at": observed_at,
            "manifest_role": "frontier_refresh_26_completion_route_successor_v45",
            "generation_calls_made": 0,
            "generation_spend_usd": "0",
            "official_results_authorised": False,
            "status": "unranked_candidate",
            "completion_route_v45": {
                "schema_version": REFRESH_SCHEMA_VERSION,
                "source_manifest_semantic_sha256": source["content_address"]["digest"],
                "source_manifest_physical_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "changed_model_ids": changed,
                "routes": route_records,
                "selection_uses_status_and_finish_metadata_only": True,
                "quality_scores_or_selections_used": False,
                "selective_failed_cell_retry": False,
                "complete_replacement_blocks_required": True,
                "automatic_fallback": False,
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
        raise SelectionRouteManifestV45Error("v45 manifest failed verification")
    return document


def verify_manifest(document: Mapping[str, Any]) -> bool:
    try:
        refresh = document["completion_route_v45"]
        rows = {row["model"]["id"]: row for row in document["models"]}
    except (KeyError, TypeError):
        return False
    if set(ROUTE_SPECS) - rows.keys():
        return False
    return bool(
        verify_manifest_content_address(document)
        and len(document.get("models", [])) == 26
        and refresh.get("changed_model_ids") == [FABLE_MODEL_ID, QWEN_MODEL_ID]
        and refresh.get("selection_uses_status_and_finish_metadata_only") is True
        and refresh.get("quality_scores_or_selections_used") is False
        and refresh.get("selective_failed_cell_retry") is False
        and refresh.get("complete_replacement_blocks_required") is True
        and refresh.get("automatic_fallback") is False
        and refresh.get("all_other_model_entries_byte_preserved") is True
        and all(
            rows[model_id]["endpoint"].get("tag") == spec["tag"]
            and rows[model_id]["endpoint"].get("provider_name") == spec["provider"]
            and rows[model_id]["request_policy"]["provider"].get("only") == [spec["tag"]]
            and rows[model_id]["request_policy"]["provider"].get("allow_fallbacks") is False
            for model_id, spec in ROUTE_SPECS.items()
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
            raise SelectionRouteManifestV45Error("content-addressed manifest conflict")
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
