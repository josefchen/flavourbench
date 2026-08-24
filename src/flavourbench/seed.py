from __future__ import annotations

import hashlib
import json

import sqlalchemy as sa
from sqlalchemy import select

from .config import get_settings
from .database import init_database, session_scope
from .endpoint_contract import endpoint_contract_sha256
from .models import CatalogModel, Season, SeasonModel, Task
from .protocol_contract import build_protocol_bundle
from .tasks import candidate_tasks

MOCK_PANEL = [
    ("flavourbench/mock-openai-flagship", "OpenAI fixture", "openai", False, "closed_flagship"),
    (
        "flavourbench/mock-anthropic-flagship",
        "Anthropic fixture",
        "anthropic",
        False,
        "closed_flagship",
    ),
    ("flavourbench/mock-google-flagship", "Google fixture", "google", False, "closed_flagship"),
    ("flavourbench/mock-xai-flagship", "xAI fixture", "xai", False, "closed_flagship"),
    ("flavourbench/mock-meta-open", "Meta fixture", "meta-llama", True, "open_weight"),
    ("flavourbench/mock-deepseek-open", "DeepSeek fixture", "deepseek", True, "open_weight"),
    ("flavourbench/mock-qwen-open", "Qwen fixture", "qwen", True, "open_weight"),
    ("flavourbench/mock-mistral-open", "Mistral fixture", "mistral", True, "open_weight"),
    ("flavourbench/mock-efficient-a", "Efficiency fixture A", "efficient-a", True, "efficiency"),
    ("flavourbench/mock-efficient-b", "Efficiency fixture B", "efficient-b", False, "efficiency"),
    ("flavourbench/mock-reasoning-a", "Reasoning fixture A", "reasoning-a", False, "reasoning"),
    ("flavourbench/mock-reasoning-b", "Reasoning fixture B", "reasoning-b", True, "reasoning"),
]


def seed_database() -> dict[str, int | str]:
    settings = get_settings()
    if settings.environment == "production":
        # This module contains only development fixtures. Production seasons,
        # models, and tasks must enter through reviewed provisioning/import
        # endpoints; startup must never synthesize benchmark observations.
        return {
            "season": settings.default_season_slug,
            "tasks": 0,
            "models": 0,
            "status": "development_fixtures_disabled",
        }
    mock_supported_parameters = [
        "max_tokens",
        "response_format",
        "seed",
        "structured_outputs",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    ]
    mock_decoding = {
        "max_tokens": settings.max_output_tokens,
        "temperature": settings.decoding_temperature,
        "top_p": settings.decoding_top_p,
        "seed": settings.decoding_seed,
    }
    mock_endpoint_document_sha256 = hashlib.sha256(b"flavourbench-mock-endpoint-v1").hexdigest()
    mock_rate_card = {
        "schema_version": "flavourbench-endpoint-rate-card-v3",
        "currency": "USD",
        "unit": "per_token_unless_request",
        "prompt_price_per_token": "0",
        "completion_price_per_token": "0",
        "request_price": "0",
        "internal_reasoning_price_per_token": "0",
        "input_cache_read_price_per_token": "0",
        "input_cache_write_price_per_token": "0",
        "input_cache_write_1h_price_per_token": "0",
        "image_price_per_unit": "0",
        "web_search_price_per_request": "0",
        "context_length": 0,
        "pricing_source_uri": "mock://catalog",
        "pricing_source_document_sha256": "0" * 64,
        "pricing_observed_at": "mock",
        "maximum_provider_requests_per_arm": settings.max_tool_rounds + 1,
        "maximum_completion_tokens_per_request": settings.max_output_tokens,
        "maximum_images_per_request": 0,
        "maximum_web_searches_per_request": 0,
        "calculation": (
            "full_context_at_maximum_input_or_cache_write_rate_plus_"
            "max_completion_and_reasoning_each_request"
        ),
    }
    mock_rate_card_sha256 = hashlib.sha256(
        json.dumps(mock_rate_card, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    init_database()
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == settings.default_season_slug))
        if season is None:
            season = Season(
                slug=settings.default_season_slug,
                name="Season 0 · governed pilot",
                status="draft",
                official=False,
                epicure_release_id=settings.epicure_release_id,
                epicure_bundle_sha256=settings.epicure_bundle_sha256,
                epicure_application_sha256=settings.epicure_application_sha256,
            )
            session.add(season)
            session.flush()
        elif season.frozen_at is not None or season.status != "draft":
            task_count = session.scalar(
                select(sa.func.count(Task.id)).where(Task.season_id == season.id)
            )
            model_count = session.scalar(
                select(sa.func.count(SeasonModel.id)).where(SeasonModel.season_id == season.id)
            )
            return {
                "season": season.slug,
                "tasks": int(task_count or 0),
                "models": int(model_count or 0),
            }

        for model_id, name, family, open_weight, role in MOCK_PANEL:
            model = session.get(CatalogModel, model_id)
            if model is None:
                model = CatalogModel(
                    model_id=model_id,
                    canonical_slug=model_id,
                    name=name,
                    family=family,
                    catalog_source="mock",
                    open_weight=open_weight,
                    open_weight_evidence_json={
                        "source": "development_fixture",
                        "classification": "open_weight" if open_weight else "closed",
                    },
                    status="smoke_passed",
                    supports_tools=True,
                    supports_structured_outputs=True,
                    pricing_json={"prompt": "0", "completion": "0"},
                    endpoint_json={
                        "provider": "mock",
                        "execution_backend": "mock",
                        "fallbacks": False,
                    },
                )
                session.add(model)
            existing = session.scalar(
                select(SeasonModel).where(
                    SeasonModel.season_id == season.id,
                    SeasonModel.model_id == model_id,
                )
            )
            endpoint_contract = {
                "execution_backend": "mock",
                "provider_slug": "mock",
                "expected_actual_model_id": model_id,
                "expected_actual_provider_slug": "mock",
                "supported_parameters_json": mock_supported_parameters,
                "decoding_json": mock_decoding,
                "endpoint_max_completion_tokens": settings.max_output_tokens,
                "endpoint_document_sha256": mock_endpoint_document_sha256,
                "endpoint_contract_sha256": endpoint_contract_sha256(
                    model_id=model_id,
                    provider_slug="mock",
                    expected_actual_model_id=model_id,
                    expected_actual_provider_slug="mock",
                    supported_parameters=mock_supported_parameters,
                    decoding=mock_decoding,
                    endpoint_max_completion_tokens=settings.max_output_tokens,
                    endpoint_document_sha256=mock_endpoint_document_sha256,
                ),
                "backend_contract_json": {},
                "backend_contract_sha256": hashlib.sha256(b"{}").hexdigest(),
                "rate_card_json": mock_rate_card,
                "rate_card_sha256": mock_rate_card_sha256,
            }
            if existing is None:
                session.add(
                    SeasonModel(
                        season_id=season.id,
                        model_id=model_id,
                        slot_role=role,
                        **endpoint_contract,
                        eligible=True,
                        worst_case_cost_micros=0,
                    )
                )
            elif existing.manifest_sha256 in {"", "unfrozen", "unresolved"}:
                for field, value in endpoint_contract.items():
                    setattr(existing, field, value)

        tasks = candidate_tasks()
        for candidate in tasks:
            existing_task = session.scalar(
                select(Task).where(
                    Task.season_id == season.id,
                    Task.public_id == candidate.public_id,
                )
            )
            if existing_task is None:
                session.add(
                    Task(
                        public_id=candidate.public_id,
                        season_id=season.id,
                        family=candidate.family,
                        prompt=candidate.prompt,
                        prompt_sha256=candidate.prompt_sha256,
                        split="development",
                        review_status=candidate.review_status,
                        provenance_json={
                            "status": "candidate_engineering_fixture",
                            "source_split": candidate.split,
                            "human_reviewed": False,
                            "confirmatory_eligible": False,
                        },
                    )
                )

        task_registry = [
            {"id": task.public_id, "family": task.family, "sha256": task.prompt_sha256}
            for task in tasks
        ]
        season.prompt_registry_sha256 = hashlib.sha256(
            json.dumps(task_registry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        season.tool_registry_sha256 = hashlib.sha256(
            b"flavourbench-mock-tool-registry-v1"
        ).hexdigest()
        season.epicure_release_id = "flavourbench-mock-epicure-v1"
        season.epicure_bundle_sha256 = hashlib.sha256(
            b"flavourbench-mock-epicure-bundle-v1"
        ).hexdigest()
        season.epicure_application_sha256 = hashlib.sha256(
            b"flavourbench-mock-epicure-application-v1"
        ).hexdigest()
        season.analysis_plan_sha256 = hashlib.sha256(
            b"flavourbench-mock-analysis-plan-v1"
        ).hexdigest()
        protocol_bundle, protocol_bundle_sha256 = build_protocol_bundle(
            tool_registry_sha256=season.tool_registry_sha256,
            epicure_release_id=season.epicure_release_id,
            epicure_bundle_sha256=season.epicure_bundle_sha256,
            epicure_application_sha256=season.epicure_application_sha256,
            analysis_plan_sha256=season.analysis_plan_sha256,
        )
        season.protocol_bundle_json = protocol_bundle
        season.protocol_bundle_sha256 = protocol_bundle_sha256
        return {"season": season.slug, "tasks": len(tasks), "models": len(MOCK_PANEL)}


def run() -> None:
    print(json.dumps(seed_database(), sort_keys=True))


if __name__ == "__main__":
    run()
