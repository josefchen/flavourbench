from __future__ import annotations

import pytest

from flavourbench.season0_openrouter_routes import RouteCatalogError, build_route_catalog


def _inputs(provider_slug: str = "provider/eu") -> tuple[dict, dict, dict]:
    roster = {
        "slots": [
            {
                "slot_role": "closed_family",
                "canonical_name": "Lab Model",
                "provider": "openrouter",
                "endpoint_id": "lab/model",
                "canonical_slug": "lab/model-20260701",
                "provider_slug": provider_slug,
            }
        ]
    }
    models = {
        "data": [
            {
                "id": "lab/model",
                "canonical_slug": "lab/model-20260701",
                "name": "Lab Model",
            }
        ]
    }
    endpoints = {
        "lab/model": {
            "data": {
                "endpoints": [
                    {
                        "tag": "provider/eu",
                        "provider_name": "Provider",
                        "supported_parameters": [
                            "max_completion_tokens",
                            "tools",
                            "tool_choice",
                        ],
                        "pricing": {"prompt": "0.1", "completion": "0.2"},
                    }
                ]
            }
        }
    }
    return roster, models, endpoints


def test_build_route_catalog_freezes_exact_endpoint() -> None:
    roster, models, endpoints = _inputs()
    catalog = build_route_catalog(
        roster, models, endpoints, observed_at="2026-07-16T00:00:00Z"
    )
    assert catalog["counts"] == {"routes": 1}
    assert catalog["routes"][0]["provider_slug"] == "provider/eu"
    assert catalog["routing_policy"]["allow_fallbacks"] is False


def test_build_route_catalog_refuses_provider_substitution() -> None:
    roster, models, endpoints = _inputs(provider_slug="wrong")
    with pytest.raises(RouteCatalogError, match="expected one"):
        build_route_catalog(roster, models, endpoints, observed_at="2026-07-16T00:00:00Z")
