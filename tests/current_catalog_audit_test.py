from __future__ import annotations

from copy import deepcopy

import pytest

from flavourbench.current_catalog_audit import (
    REQUIRED_PARAMETERS,
    CurrentCatalogAuditError,
    _write,
    build_catalog_audit,
    verify_catalog_audit_content_address,
)
from flavourbench.real_task_bank import sha256_json

MODEL_ID = "vendor/model-v1"
CANONICAL_SLUG = "vendor/model-v1-20260801"
PROVIDER_TAG = "provider/fixed"


def _manifest() -> dict:
    payload = {
        "schema_version": "flavourbench-openrouter-candidate-manifest-v1",
        "models": [
            {
                "model": {
                    "id": MODEL_ID,
                    "canonical_slug": CANONICAL_SLUG,
                    "name": "Vendor Model V1",
                },
                "endpoint": {
                    "model_id": MODEL_ID,
                    "name": f"Provider | {CANONICAL_SLUG}",
                    "provider_name": "Provider",
                    "tag": PROVIDER_TAG,
                },
            }
        ],
    }
    digest = sha256_json(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    return payload


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": MODEL_ID,
                "canonical_slug": CANONICAL_SLUG,
                "supported_parameters": list(REQUIRED_PARAMETERS),
                "expiration_date": None,
            }
        ]
    }


def _endpoint_documents() -> dict:
    return {
        MODEL_ID: {
            "data": {
                "id": MODEL_ID,
                "endpoints": [
                    {
                        "name": f"Provider | {CANONICAL_SLUG}",
                        "model_id": MODEL_ID,
                        "provider_name": "Provider",
                        "tag": PROVIDER_TAG,
                        "supported_parameters": list(REQUIRED_PARAMETERS),
                        "context_length": 128_000,
                        "max_completion_tokens": 16_000,
                    }
                ],
            }
        }
    }


def test_exact_route_provider_and_canonical_release_pass_without_quality_claim() -> None:
    audit = build_catalog_audit(
        manifest=_manifest(),
        catalog=_catalog(),
        endpoint_documents=_endpoint_documents(),
        observed_at="2026-08-02T00:00:00Z",
    )

    assert audit["counts"] == {
        "manifest_models": 1,
        "models_discovered": 1,
        "exact_provider_endpoints_matched": 1,
        "freshness_contract_passed": 1,
        "freshness_contract_failed": 0,
        "quality_observations": 0,
        "provider_generations": 0,
        "spend_usd": "0",
    }
    assert audit["models"][0]["endpoint_document_identity_matches"] is True
    assert audit["models"][0]["endpoint_route_identity_matches"] is True
    assert audit["models"][0]["endpoint_canonical_name_matches"] is True
    assert audit["models"][0]["endpoint_provider_name_matches"] is True
    assert audit["claim_boundary"]["rank_eligible"] is False
    assert audit["claim_boundary"]["catalog_presence_is_not_model_quality"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_field"),
    [
        ("canonical", "endpoint_canonical_name_matches"),
        ("route", "endpoint_route_identity_matches"),
        ("provider", "endpoint_provider_name_matches"),
        ("capability", "endpoint_missing_required_parameters"),
    ],
)
def test_identity_or_capability_drift_fails_closed(
    mutation: str,
    failed_field: str,
) -> None:
    endpoints = _endpoint_documents()
    endpoint = endpoints[MODEL_ID]["data"]["endpoints"][0]
    if mutation == "canonical":
        endpoint["name"] = "Provider | vendor/model-v1-elsewhere"
    elif mutation == "route":
        endpoint["model_id"] = "vendor/different-route"
    elif mutation == "provider":
        endpoint["provider_name"] = "Different Provider"
    else:
        endpoint["supported_parameters"].remove("tools")

    audit = build_catalog_audit(
        manifest=_manifest(),
        catalog=_catalog(),
        endpoint_documents=endpoints,
        observed_at="2026-08-02T00:00:00Z",
    )

    assert audit["counts"]["freshness_contract_failed"] == 1
    assert audit["models"][0]["status"] == "freshness_contract_failed"
    if failed_field == "endpoint_missing_required_parameters":
        assert audit["models"][0][failed_field] == ["tools"]
    else:
        assert audit["models"][0][failed_field] is False


def test_duplicate_catalog_identity_is_rejected() -> None:
    catalog = _catalog()
    catalog["data"].append(deepcopy(catalog["data"][0]))

    with pytest.raises(CurrentCatalogAuditError, match="catalog repeats model id"):
        build_catalog_audit(
            manifest=_manifest(),
            catalog=catalog,
            endpoint_documents=_endpoint_documents(),
            observed_at="2026-08-02T00:00:00Z",
        )


def test_written_audit_is_content_addressed_and_tamper_evident(tmp_path) -> None:
    payload = build_catalog_audit(
        manifest=_manifest(),
        catalog=_catalog(),
        endpoint_documents=_endpoint_documents(),
        observed_at="2026-08-02T00:00:00Z",
    )
    path = _write(tmp_path, payload)
    document = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert path.name == f"current-model-catalog-audit-{document['artifact_sha256']}.json"
    assert verify_catalog_audit_content_address(document)
    document["counts"]["quality_observations"] = 1
    assert not verify_catalog_audit_content_address(document)
