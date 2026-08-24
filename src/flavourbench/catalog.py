from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import CatalogModel

CATALOG_SOURCE = "openrouter"
GOVERNED_STATUSES = {"smoke_passed", "season_eligible"}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def fetch_openrouter_catalog() -> list[dict]:
    settings = get_settings()
    headers = {"Accept": "application/json"}
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter model catalog returned an invalid payload")
    return [item for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


def sync_catalog(session: Session, items: list[dict]) -> dict[str, int]:
    now = datetime.now(UTC)
    seen: set[str] = set()
    counts = {"discovered": 0, "compatible": 0, "retired": 0}
    for item in items:
        model_id = item["id"]
        seen.add(model_id)
        supported = set(item.get("supported_parameters") or [])
        tools = "tools" in supported
        structured = "structured_outputs" in supported
        status = "compatible" if tools and structured else "discovered"
        family = model_id.split("/", 1)[0]
        model = session.get(CatalogModel, model_id)
        if model is None:
            model = CatalogModel(
                model_id=model_id,
                canonical_slug=item.get("canonical_slug") or model_id,
                catalog_source=CATALOG_SOURCE,
            )
            session.add(model)
        elif model.catalog_source != CATALOG_SOURCE:
            # Provider registries are isolated. A manually admitted Bedrock or
            # private endpoint must not be rewritten by an OpenRouter sync.
            continue
        prior_status = model.status
        canonical_slug = item.get("canonical_slug") or model_id
        top_provider = (
            item.get("top_provider") if isinstance(item.get("top_provider"), dict) else {}
        )
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        context_length = (
            item.get("context_length") if isinstance(item.get("context_length"), int) else None
        )
        discovery = {
            "source": CATALOG_SOURCE,
            "canonical_slug": canonical_slug,
            "supported_parameters": sorted(supported),
            "context_length": context_length,
            "pricing": pricing,
            "top_provider": top_provider,
            "hugging_face_id": item.get("hugging_face_id"),
        }
        discovery_sha256 = _canonical_sha256(discovery)
        governed_endpoint = dict(model.endpoint_json or {})
        bound_discovery = governed_endpoint.get("smoke_catalog_discovery_sha256")
        discovery_drifted = (
            prior_status in GOVERNED_STATUSES and bound_discovery != discovery_sha256
        )
        model.canonical_slug = canonical_slug
        model.name = item.get("name") or model_id
        model.family = family
        hugging_face_id = item.get("hugging_face_id")
        model.open_weight = isinstance(hugging_face_id, str) and bool(hugging_face_id.strip())
        model.open_weight_evidence_json = (
            {
                "source": "openrouter_model_catalog",
                "hugging_face_id": hugging_face_id,
                "classification": "open_weight",
            }
            if model.open_weight
            else {
                "source": "openrouter_model_catalog",
                "classification": "unknown_fail_closed",
            }
        )
        model.status = (
            status
            if discovery_drifted
            else prior_status
            if prior_status in GOVERNED_STATUSES
            else status
        )
        model.supports_tools = tools
        model.supports_structured_outputs = structured
        model.context_length = context_length
        model.pricing_json = pricing
        governed_endpoint["catalog_discovery"] = discovery
        governed_endpoint["catalog_discovery_sha256"] = discovery_sha256
        if discovery_drifted:
            governed_endpoint["smoke_invalidated_by_catalog_drift"] = True
        model.endpoint_json = governed_endpoint
        model.last_seen_at = now
        model.retired_at = None
        counts[model.status if model.status in counts else status] += 1

    existing_ids = session.scalars(
        select(CatalogModel.model_id).where(
            CatalogModel.catalog_source == CATALOG_SOURCE,
            ~CatalogModel.model_id.like("flavourbench/mock-%"),
        )
    ).all()
    for model_id in existing_ids:
        if model_id not in seen:
            model = session.get(CatalogModel, model_id)
            if model is not None:
                model.status = "retired"
                model.retired_at = now
                counts["retired"] += 1
    return counts
