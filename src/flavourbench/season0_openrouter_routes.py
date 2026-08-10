"""Snapshot exact OpenRouter routes selected by the Season 0 roster."""

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
from urllib.parse import quote

import httpx

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season0-openrouter-route-catalog-v1"
OPENROUTER_API = "https://openrouter.ai/api/v1"


class RouteCatalogError(RuntimeError):
    """The requested exact OpenRouter route could not be frozen."""


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise RouteCatalogError("content-addressed route catalog conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def build_route_catalog(
    roster: Mapping[str, Any],
    model_document: Mapping[str, Any],
    endpoint_documents: Mapping[str, Mapping[str, Any]],
    *,
    observed_at: str,
) -> dict[str, Any]:
    slots = roster.get("slots")
    models = model_document.get("data")
    if not isinstance(slots, list) or not isinstance(models, list):
        raise RouteCatalogError("roster or OpenRouter model catalog is invalid")
    models_by_id = {
        str(model["id"]): model
        for model in models
        if isinstance(model, Mapping) and isinstance(model.get("id"), str)
    }
    routes: list[dict[str, Any]] = []
    for slot in slots:
        if not isinstance(slot, Mapping) or slot.get("provider") != "openrouter":
            continue
        model_id = str(slot.get("endpoint_id") or "")
        provider_slug = str(slot.get("provider_slug") or "")
        canonical_slug = str(slot.get("canonical_slug") or "")
        model = models_by_id.get(model_id)
        if model is None:
            raise RouteCatalogError(f"model absent from OpenRouter catalog: {model_id}")
        if model.get("canonical_slug") != canonical_slug:
            raise RouteCatalogError(f"canonical slug drift for {model_id}")
        endpoint_document = endpoint_documents.get(model_id)
        if not isinstance(endpoint_document, Mapping):
            raise RouteCatalogError(f"endpoint document absent for {model_id}")
        data = endpoint_document.get("data")
        endpoints = data.get("endpoints") if isinstance(data, Mapping) else None
        if not isinstance(endpoints, list):
            raise RouteCatalogError(f"endpoint collection invalid for {model_id}")
        matching = [
            endpoint
            for endpoint in endpoints
            if isinstance(endpoint, Mapping) and endpoint.get("tag") == provider_slug
        ]
        if len(matching) != 1:
            raise RouteCatalogError(
                f"expected one {provider_slug} endpoint for {model_id}; found {len(matching)}"
            )
        endpoint = matching[0]
        supported = set(endpoint.get("supported_parameters") or [])
        required = {"tools", "tool_choice"}
        if not required.issubset(supported) or not {
            "max_tokens",
            "max_completion_tokens",
        }.intersection(supported):
            raise RouteCatalogError(f"route lacks bounded tool calling: {model_id}")
        routes.append(
            {
                "slot_role": str(slot.get("slot_role") or ""),
                "display_name": str(slot.get("canonical_name") or model.get("name") or model_id),
                "model_id": model_id,
                "canonical_slug": canonical_slug,
                "provider_slug": provider_slug,
                "model": dict(model),
                "endpoint": dict(endpoint),
                "endpoint_document_sha256": sha256_json(endpoint_document),
            }
        )
    if not routes:
        raise RouteCatalogError("roster contains no OpenRouter routes")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_exact_routes_pending_live_smoke",
        "observed_at": observed_at,
        "source": {
            "models_url": f"{OPENROUTER_API}/models",
            "endpoint_url_template": f"{OPENROUTER_API}/models/:author/:slug/endpoints",
            "models_document_sha256": sha256_json(model_document),
        },
        "roster_sha256": sha256_json(roster),
        "routing_policy": {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "provider_substitution_rank_eligible": False,
            "provider_structured_output_required": False,
            "normalization_mode": "lossless_client_text_wrapper_v1",
        },
        "counts": {"routes": len(routes)},
        "routes": routes,
    }


async def snapshot(roster_path: Path) -> dict[str, Any]:
    roster = json.loads(roster_path.read_bytes())
    if not isinstance(roster, Mapping):
        raise RouteCatalogError("roster must be a JSON object")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "FLAVOURBENCH_OPENROUTER_API_KEY"
    )
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(
        base_url=OPENROUTER_API + "/", headers=headers, timeout=60
    ) as client:
        model_response = await client.get("models")
        model_response.raise_for_status()
        model_document = model_response.json()
        slots = roster.get("slots")
        if not isinstance(slots, list):
            raise RouteCatalogError("roster slots are invalid")

        async def fetch(model_id: str) -> tuple[str, Mapping[str, Any]]:
            author, slug = model_id.split("/", 1)
            path = f"models/{quote(author, safe='')}/{quote(slug, safe=':~._-')}/endpoints"
            response = await client.get(path)
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, Mapping):
                raise RouteCatalogError(f"endpoint response is invalid for {model_id}")
            return model_id, document

        model_ids = [
            str(slot.get("endpoint_id") or "")
            for slot in slots
            if isinstance(slot, Mapping) and slot.get("provider") == "openrouter"
        ]
        fetched = await asyncio.gather(*(fetch(model_id) for model_id in model_ids))
    if not isinstance(model_document, Mapping):
        raise RouteCatalogError("model response is invalid")
    return build_route_catalog(
        roster,
        model_document,
        dict(fetched),
        observed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season0/routes"))
    args = parser.parse_args(argv)
    catalog = asyncio.run(snapshot(args.roster))
    path = _atomic_write(args.output_dir, "openrouter-route-catalog", catalog)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "counts": catalog["counts"],
                "routes": [
                    {
                        "display_name": route["display_name"],
                        "model_id": route["model_id"],
                        "canonical_slug": route["canonical_slug"],
                        "provider_slug": route["provider_slug"],
                    }
                    for route in catalog["routes"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
