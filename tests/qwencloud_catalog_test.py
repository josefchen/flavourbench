from __future__ import annotations

import json

import httpx
import pytest

from flavourbench.config import Settings
from flavourbench.frontier_contract_runner import load_candidate_manifest, select_candidates
from flavourbench.frontier_manifest import verify_manifest_content_address
from flavourbench.qwencloud_catalog import (
    CANDIDATE_SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    DEFAULT_CANDIDATES,
    QwenCloudCatalogError,
    _atomic_write,
    build_candidate_extension,
    build_catalog_artifact,
    build_unranked_qwen37_route_manifest,
    build_unranked_qwen38_alias_route_manifest,
    fetch_authenticated_catalog,
    verify_content_address,
    write_qwencloud_route_manifest,
)


def _provider_catalog() -> dict:
    ids = [
        "qwen3.8-max",
        "qwen3.7-max",
        *DEFAULT_CANDIDATES,
        "deepseek-v4-pro",
    ]
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 1, "owned_by": "system"}
            for model_id in reversed(ids)
        ],
    }


@pytest.mark.asyncio
async def test_authenticated_catalog_fetch_is_models_only_and_sorted() -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert request.headers["Authorization"].startswith("Bearer sk-ws-")
        return httpx.Response(
            200,
            json=_provider_catalog(),
            headers={"Date": "Sat, 08 Aug 2026 12:00:00 GMT"},
        )

    catalog, response_date = await fetch_authenticated_catalog(
        api_key="sk-ws-test-credential-not-real",
        transport=httpx.MockTransport(handler),
    )

    assert requested_paths == ["/compatible-mode/v1/models"]
    assert [row["id"] for row in catalog["data"]] == sorted(
        row["id"] for row in _provider_catalog()["data"]
    )
    assert response_date == "Sat, 08 Aug 2026 12:00:00 GMT"


@pytest.mark.asyncio
async def test_authenticated_catalog_accepts_legacy_pay_as_you_go_key() -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer sk-test-legacy-payg-not-real"
        return httpx.Response(200, json=_provider_catalog())

    catalog, _response_date = await fetch_authenticated_catalog(
        api_key="sk-test-legacy-payg-not-real",
        transport=httpx.MockTransport(handler),
    )

    assert requested_paths == ["/compatible-mode/v1/models"]
    assert len(catalog["data"]) == len(_provider_catalog()["data"])


@pytest.mark.asyncio
async def test_catalog_rejects_unapproved_credential_destination() -> None:
    with pytest.raises(QwenCloudCatalogError, match="approved HTTPS host"):
        await fetch_authenticated_catalog(
            api_key="sk-ws-test-credential-not-real",
            base_url="https://attacker.invalid/v1",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
        )


@pytest.mark.asyncio
async def test_catalog_rejects_token_plan_endpoint() -> None:
    with pytest.raises(QwenCloudCatalogError, match="pay-as-you-go"):
        await fetch_authenticated_catalog(
            api_key="sk-ws-test-credential-not-real",
            base_url=("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        )


@pytest.mark.asyncio
async def test_catalog_rejects_token_plan_credential() -> None:
    with pytest.raises(QwenCloudCatalogError, match="pay-as-you-go key"):
        await fetch_authenticated_catalog(
            api_key="sk-sp-test-credential-not-real",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        )


def _catalog_document() -> dict:
    payload = build_catalog_artifact(
        catalog=_provider_catalog(),
        observed_at="2026-08-08T12:00:00Z",
        response_date="Sat, 08 Aug 2026 12:00:00 GMT",
    )
    payload["artifact_sha256"] = __import__(
        "flavourbench.real_task_bank", fromlist=["sha256_json"]
    ).sha256_json(payload)
    return payload


def test_candidate_extension_has_zero_observations_and_keeps_38_alias_unranked() -> None:
    extension = build_candidate_extension(catalog_artifact=_catalog_document())

    assert extension["counts"] == {
        "discovery_aliases": 2,
        "frozen_candidates": 4,
        "observed_arms": 0,
        "provider_generation_calls": 0,
        "epicure_calls": 0,
        "quality_judgments": 0,
        "rankable_comparisons": 0,
    }
    assert {row["model_id"] for row in extension["candidates"]} == set(DEFAULT_CANDIDATES)
    assert all(row["rank_eligible"] is False for row in extension["candidates"])
    qwen38 = next(row for row in extension["discovery_aliases"] if row["model_id"] == "qwen3.8-max")
    assert qwen38["identity_kind"] == "mutable_alias"
    assert qwen38["observed_arms"] == 0
    assert qwen38["rank_eligible"] is False
    assert extension["requested_user_label"]["frozen_execution_candidate"] is False


def test_38_is_not_inferred_when_authenticated_catalog_lacks_it() -> None:
    catalog = _catalog_document()
    catalog["models"] = [row for row in catalog["models"] if row["id"] != "qwen3.8-max"]
    payload = {key: value for key, value in catalog.items() if key != "artifact_sha256"}
    payload["model_count"] = len(payload["models"])
    payload["artifact_sha256"] = __import__(
        "flavourbench.real_task_bank", fromlist=["sha256_json"]
    ).sha256_json({key: value for key, value in payload.items() if key != "artifact_sha256"})

    with pytest.raises(QwenCloudCatalogError, match="absent from the authenticated catalog"):
        build_candidate_extension(catalog_artifact=payload)


def test_artifacts_are_content_addressed_and_tamper_evident(tmp_path) -> None:
    catalog_path = _atomic_write(
        tmp_path,
        "qwencloud-model-catalog",
        {key: value for key, value in _catalog_document().items() if key != "artifact_sha256"},
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert verify_content_address(catalog, CATALOG_SCHEMA_VERSION)

    extension_payload = build_candidate_extension(catalog_artifact=catalog)
    extension_path = _atomic_write(tmp_path, "qwencloud-candidate-extension", extension_payload)
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    assert verify_content_address(extension, CANDIDATE_SCHEMA_VERSION)
    extension["counts"]["observed_arms"] = 1
    assert not verify_content_address(extension, CANDIDATE_SCHEMA_VERSION)


def test_settings_accept_official_dashscope_variable_without_leaking_to_api(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ws-test-credential-not-real")
    monkeypatch.setenv("MCP_API_TOKEN", "test-mcp-token-not-real")
    settings = Settings(_env_file=None)
    assert settings.qwencloud_api_key.startswith("sk-ws-")
    assert settings.mcp_token == "test-mcp-token-not-real"

    with pytest.raises(ValueError, match="production API must not receive provider"):
        Settings(
            _env_file=None,
            environment="production",
            service_role="api",
            execution_mode="live",
            live_authorized=True,
            database_url="postgresql://example",
            auto_create_schema=False,
            service_token="s" * 32,
            admin_token="a" * 32,
            expert_token="e" * 32,
            pseudonym_secret="p" * 32,
            task_validator_identity_hmac_secret="t" * 32,
            reviewer_identity_hmac_secret="i" * 32,
            reviewer_credential_hmac_secret="c" * 32,
            organization_api_key_hmac_secret="o" * 32,
            run_card_signing_secret="r" * 32,
            budget_authorization_signing_secret="b" * 32,
        )


def test_dated_qwen37_route_is_content_bound_and_permanently_unranked(tmp_path) -> None:
    manifest = build_unranked_qwen37_route_manifest(
        catalog_artifact=_catalog_document(),
        cap_usd="2",
    )

    assert verify_manifest_content_address(manifest)
    assert manifest["official_results_authorised"] is False
    assert manifest["budget"]["bounded_forecast_usd"] == "1.82592"
    entry = manifest["models"][0]
    assert entry["model"]["canonical_slug"] == "qwen3.7-max-2026-06-08"
    assert entry["backend_contract"]["identity_kind"] == "immutable_dated_release"
    assert entry["backend_contract"]["structured_outputs_supported"] is False
    assert entry["backend_contract"]["cost_reconciliation"] == ("provider_charge_unavailable")
    assert entry["backend_contract"]["rank_eligible"] is False
    assert "response_format" not in entry["endpoint"]["supported_parameters"]
    assert "structured_outputs" not in entry["endpoint"]["supported_parameters"]
    assert "separate strata" in manifest["governance"]["cross_provider_identity_policy"]

    path = write_qwencloud_route_manifest(manifest, tmp_path)
    loaded = load_candidate_manifest(
        path,
        expected_digest=manifest["content_address"]["digest"],
    )
    candidates = select_candidates(loaded)
    assert len(candidates) == 1
    assert candidates[0].execution_backend == "qwencloud_direct"
    assert candidates[0].cost_accounting_policy == ("provider_usage_times_frozen_rate_card")


def test_mutable_qwen38_alias_requires_explicit_exploratory_opt_in() -> None:
    with pytest.raises(QwenCloudCatalogError, match="explicit mutable-alias"):
        build_unranked_qwen37_route_manifest(
            catalog_artifact=_catalog_document(),
            model_id="qwen3.8-max",
        )


def test_mutable_qwen38_alias_route_is_observation_pinned_and_never_rankable(
    tmp_path,
) -> None:
    manifest = build_unranked_qwen38_alias_route_manifest(
        catalog_artifact=_catalog_document(),
        cap_usd="2",
        allow_mutable_alias_exploratory=True,
    )

    assert verify_manifest_content_address(manifest)
    assert manifest["official_results_authorised"] is False
    assert manifest["budget"]["bounded_forecast_usd"] == "2"
    entry = manifest["models"][0]
    contract = entry["backend_contract"]
    assert entry["model"]["canonical_slug"] == "qwen3.8-max"
    assert contract["catalog_sha256"] == _catalog_document()["artifact_sha256"]
    assert contract["catalog_observed_at"] == "2026-08-08T12:00:00Z"
    assert contract["catalog_entry_sha256"]
    assert contract["identity_kind"] == "mutable_alias"
    assert contract["catalog_pinned_at_observation"] is True
    assert contract["model_identity_label"] == ("catalog_pinned_at_observation_not_a_frozen_model")
    assert contract["official"] is False
    assert contract["season_eligible"] is False
    assert contract["rank_eligible"] is False
    assert contract["mutable_alias_execution_requires_explicit_opt_in"] is True
    assert contract["cost_reconciliation"] == ("provider_rate_and_charge_unavailable")
    assert entry["cost_accounting_policy"] == ("provider_usage_with_unpriced_budget_ceiling")
    assert entry["endpoint"]["pricing"]["provider_rate_known"] is False
    assert entry["endpoint"]["pricing"]["zero_values_mean"] == ("unknown_cost_not_free")
    assert "separate strata" in manifest["governance"]["cross_provider_identity_policy"]

    path = write_qwencloud_route_manifest(manifest, tmp_path)
    loaded = load_candidate_manifest(
        path,
        expected_digest=manifest["content_address"]["digest"],
    )
    candidate = select_candidates(loaded)[0]
    assert candidate.model_id == "qwen3.8-max"
    assert candidate.cost_accounting_policy == ("provider_usage_with_unpriced_budget_ceiling")


def test_qwen37_route_fails_closed_below_worst_case_reservation() -> None:
    with pytest.raises(QwenCloudCatalogError, match="exceeds cap"):
        build_unranked_qwen37_route_manifest(
            catalog_artifact=_catalog_document(),
            cap_usd="1",
        )


def test_qwen38_tool_auto_successor_binds_prior_failure_and_remains_unranked() -> None:
    predecessor = build_unranked_qwen38_alias_route_manifest(
        catalog_artifact=_catalog_document(),
        cap_usd="2",
        allow_mutable_alias_exploratory=True,
    )
    successor = build_unranked_qwen38_alias_route_manifest(
        catalog_artifact=_catalog_document(),
        cap_usd="2",
        allow_mutable_alias_exploratory=True,
        tool_auto_successor_failure_sha256="f" * 64,
    )

    assert verify_manifest_content_address(successor)
    assert successor["content_address"]["digest"] != predecessor["content_address"]["digest"]
    entry = successor["models"][0]
    contract = entry["backend_contract"]
    assert contract["tool_choice_transport_mode"] == ("auto_with_required_success_postcondition")
    assert contract["tool_choice_required_supported"] is False
    assert contract["predecessor_failure_artifact_sha256"] == "f" * 64
    assert entry["request_policy"]["tool_choice_transport"] == "auto"
    assert entry["request_policy"]["required_tool_success_enforced_after_response"] is True
    assert contract["official"] is False
    assert contract["rank_eligible"] is False
