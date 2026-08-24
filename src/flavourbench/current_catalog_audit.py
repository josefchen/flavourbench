"""Audit a frozen development roster against the live OpenRouter catalog without generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .frontier_manifest import verify_manifest_content_address
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-current-model-catalog-audit-v2"
REQUIRED_PARAMETERS = (
    "max_tokens",
    "reasoning",
    "response_format",
    "structured_outputs",
    "tool_choice",
    "tools",
)


class CurrentCatalogAuditError(RuntimeError):
    """The current-model catalog audit could not be verified."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CurrentCatalogAuditError(f"manifest must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurrentCatalogAuditError("manifest is not valid JSON") from exc
    if not isinstance(value, dict) or not verify_manifest_content_address(value):
        raise CurrentCatalogAuditError("manifest content address does not verify")
    return value


def build_catalog_audit(
    *,
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    endpoint_documents: Mapping[str, Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    if not verify_manifest_content_address(manifest):
        raise CurrentCatalogAuditError("manifest content address does not verify")
    models = manifest.get("models")
    catalog_rows = catalog.get("data")
    if not isinstance(models, list) or not isinstance(catalog_rows, list):
        raise CurrentCatalogAuditError("manifest or catalog has no model rows")
    catalog_by_id: dict[str, Mapping[str, Any]] = {}
    for row in catalog_rows:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("id") or "")
        if not model_id:
            continue
        if model_id in catalog_by_id:
            raise CurrentCatalogAuditError(f"catalog repeats model id: {model_id}")
        catalog_by_id[model_id] = row

    audited: list[dict[str, Any]] = []
    seen_manifest_ids: set[str] = set()
    for entry in models:
        if not isinstance(entry, Mapping):
            raise CurrentCatalogAuditError("manifest contains a non-object model entry")
        model = entry.get("model")
        endpoint = entry.get("endpoint")
        if not isinstance(model, Mapping) or not isinstance(endpoint, Mapping):
            raise CurrentCatalogAuditError("manifest entry lacks model or endpoint identity")
        model_id = str(model.get("id") or "")
        expected_canonical = str(model.get("canonical_slug") or "")
        expected_tag = str(endpoint.get("tag") or "")
        expected_provider_name = str(endpoint.get("provider_name") or "")
        if (
            not model_id
            or not expected_canonical
            or not expected_tag
            or not expected_provider_name
        ):
            raise CurrentCatalogAuditError("manifest entry has an incomplete frozen identity")
        if model_id in seen_manifest_ids:
            raise CurrentCatalogAuditError(f"manifest repeats model id: {model_id}")
        seen_manifest_ids.add(model_id)
        current_model = catalog_by_id.get(model_id)
        endpoint_document = endpoint_documents.get(model_id)
        endpoint_document_identity_matches = bool(
            isinstance(endpoint_document, Mapping)
            and isinstance(endpoint_document.get("data"), Mapping)
            and str(endpoint_document["data"].get("id") or "") == model_id
        )
        endpoints = (
            endpoint_document.get("data", {}).get("endpoints", [])
            if isinstance(endpoint_document, Mapping)
            and isinstance(endpoint_document.get("data"), Mapping)
            else []
        )
        matches = [
            row
            for row in endpoints
            if isinstance(row, Mapping) and str(row.get("tag") or "") == expected_tag
        ]
        current_endpoint = matches[0] if len(matches) == 1 else None
        model_parameters = {
            str(value)
            for value in ((current_model or {}).get("supported_parameters") or [])
        }
        endpoint_parameters = {
            str(value)
            for value in ((current_endpoint or {}).get("supported_parameters") or [])
        }
        model_missing = sorted(set(REQUIRED_PARAMETERS) - model_parameters)
        endpoint_missing = sorted(set(REQUIRED_PARAMETERS) - endpoint_parameters)
        canonical_matches = bool(
            current_model is not None
            and str(current_model.get("canonical_slug") or "") == expected_canonical
        )
        endpoint_route_identity_matches = bool(
            current_endpoint is not None
            and str(current_endpoint.get("model_id") or "") == model_id
        )
        endpoint_canonical_name_matches = bool(
            current_endpoint is not None
            and str(current_endpoint.get("name") or "").rsplit(" | ", 1)[-1]
            == expected_canonical
        )
        endpoint_provider_name_matches = bool(
            current_endpoint is not None
            and str(current_endpoint.get("provider_name") or "") == expected_provider_name
        )
        passed = bool(
            current_model is not None
            and len(matches) == 1
            and canonical_matches
            and endpoint_document_identity_matches
            and endpoint_route_identity_matches
            and endpoint_canonical_name_matches
            and endpoint_provider_name_matches
            and not model_missing
            and not endpoint_missing
            and current_model.get("expiration_date") is None
        )
        audited.append(
            {
                "model_id": model_id,
                "display_name": model.get("name"),
                "expected_canonical_slug": expected_canonical,
                "observed_canonical_slug": (
                    current_model.get("canonical_slug") if current_model is not None else None
                ),
                "expected_provider_tag": expected_tag,
                "expected_provider_name": expected_provider_name,
                "observed_provider_name": (
                    current_endpoint.get("provider_name")
                    if current_endpoint is not None
                    else None
                ),
                "model_discovered": current_model is not None,
                "exact_provider_endpoint_matches": len(matches) == 1,
                "canonical_identity_matches": canonical_matches,
                "endpoint_document_identity_matches": endpoint_document_identity_matches,
                "endpoint_route_identity_matches": endpoint_route_identity_matches,
                "endpoint_canonical_name_matches": endpoint_canonical_name_matches,
                "endpoint_provider_name_matches": endpoint_provider_name_matches,
                "model_missing_required_parameters": model_missing,
                "endpoint_missing_required_parameters": endpoint_missing,
                "expiration_date": (
                    current_model.get("expiration_date") if current_model is not None else None
                ),
                "stable_endpoint_projection_sha256": (
                    sha256_json(
                        {
                            "name": current_endpoint.get("name"),
                            "model_id": current_endpoint.get("model_id"),
                            "provider_name": current_endpoint.get("provider_name"),
                            "tag": current_endpoint.get("tag"),
                            "supported_parameters": sorted(endpoint_parameters),
                            "context_length": current_endpoint.get("context_length"),
                            "max_completion_tokens": current_endpoint.get(
                                "max_completion_tokens"
                            ),
                        }
                    )
                    if current_endpoint is not None
                    else None
                ),
                "status": "freshness_contract_passed" if passed else "freshness_contract_failed",
                "quality_observations": 0,
            }
        )

    passed_count = sum(row["status"] == "freshness_contract_passed" for row in audited)
    manifest_digest = str(manifest.get("content_address", {}).get("digest") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at": observed_at,
        "source_manifest_sha256": manifest_digest,
        "catalog_source": {
            "models_url": "https://openrouter.ai/api/v1/models?output_modalities=text",
            "endpoint_url_template": (
                "https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
            ),
            "catalog_projection_sha256": sha256_json(
                [
                    {
                        "id": row.get("id"),
                        "canonical_slug": row.get("canonical_slug"),
                        "supported_parameters": sorted(row.get("supported_parameters") or []),
                        "expiration_date": row.get("expiration_date"),
                    }
                    for row in catalog_rows
                    if isinstance(row, Mapping)
                ]
            ),
        },
        "required_parameters": list(REQUIRED_PARAMETERS),
        "counts": {
            "manifest_models": len(audited),
            "models_discovered": sum(row["model_discovered"] for row in audited),
            "exact_provider_endpoints_matched": sum(
                row["exact_provider_endpoint_matches"] for row in audited
            ),
            "freshness_contract_passed": passed_count,
            "freshness_contract_failed": len(audited) - passed_count,
            "quality_observations": 0,
            "provider_generations": 0,
            "spend_usd": "0",
        },
        "models": audited,
        "claim_boundary": {
            "live_catalog_network_requests": 1 + len(audited),
            "provider_generation_requests": 0,
            "epicure_calls": 0,
            "quality_observations": 0,
            "rank_eligible": False,
            "catalog_presence_is_not_execution_compatibility": True,
            "catalog_presence_is_not_model_quality": True,
        },
    }


def verify_catalog_audit_content_address(document: Mapping[str, Any]) -> bool:
    """Verify the artifact address without treating its claims as quality evidence."""

    digest = document.get("artifact_sha256")
    unhashed = dict(document)
    unhashed.pop("artifact_sha256", None)
    return (
        document.get("schema_version") == SCHEMA_VERSION
        and isinstance(digest, str)
        and digest == sha256_json(unhashed)
    )


async def collect_catalog_audit(
    *,
    manifest: Mapping[str, Any],
    base_url: str = "https://openrouter.ai",
) -> dict[str, Any]:
    models = manifest.get("models")
    if not isinstance(models, list):
        raise CurrentCatalogAuditError("manifest has no model rows")
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        catalog_response = await client.get(
            "/api/v1/models",
            params={"output_modalities": "text"},
        )
        catalog_response.raise_for_status()
        catalog = catalog_response.json()

        async def endpoint_document(model_id: str) -> tuple[str, Mapping[str, Any]]:
            response = await client.get(f"/api/v1/models/{model_id}/endpoints")
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, Mapping):
                raise CurrentCatalogAuditError("endpoint catalog returned a non-object")
            return model_id, document

        pairs = await asyncio.gather(
            *(
                endpoint_document(str(entry.get("model", {}).get("id") or ""))
                for entry in models
                if isinstance(entry, Mapping) and isinstance(entry.get("model"), Mapping)
            )
        )
    if not isinstance(catalog, Mapping):
        raise CurrentCatalogAuditError("model catalog returned a non-object")
    return build_catalog_audit(
        manifest=manifest,
        catalog=catalog,
        endpoint_documents=dict(pairs),
        observed_at=datetime.now(UTC).isoformat(),
    )


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"current-model-catalog-audit-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise CurrentCatalogAuditError("content-addressed catalog audit conflicts")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai")
    arguments = parser.parse_args(argv)
    manifest = _load(arguments.manifest)
    payload = asyncio.run(
        collect_catalog_audit(manifest=manifest, base_url=arguments.base_url)
    )
    path = _write(arguments.output_dir, payload)
    print(path)


if __name__ == "__main__":
    run()
