from __future__ import annotations

import asyncio
import copy
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import delete, select, text

import flavourbench.provider as provider_module
from flavourbench.catalog import sync_catalog
from flavourbench.config import Settings
from flavourbench.database import init_database, session_scope
from flavourbench.engine import (
    LostJobLease,
    _persist_provider_attempt,
    _record_known_zero_cost,
    claim_job,
    process_job,
    reconcile_battle_cost,
    recover_stale_jobs,
)
from flavourbench.mcp_client import McpSession, _decode_response, _read_result
from flavourbench.models import (
    Battle,
    CatalogModel,
    ControlledRun,
    GenerationAttempt,
    Job,
    ResponseArm,
    Season,
)
from flavourbench.provider import (
    GenerationSpec,
    MockProvider,
    OpenRouterProvider,
    ProviderAttemptEvent,
    ProviderError,
    UncertainDeliveryError,
    _extract_content,
    _money_to_micros,
    _parse_final,
    _verified_openrouter_generation_identity,
    system_prompt_sha256,
)
from flavourbench.security import (
    contains_identity_leak,
    request_pseudonym,
    sanitize_for_release,
)
from flavourbench.seed import seed_database
from flavourbench.smoke import smoke
from flavourbench.validators import validate_output


def _queued_generation_fixture(
    session,
    *,
    season: Season,
    battle_id: str,
    left_arm_id: str,
    job_id: str,
    prompt: str,
    prompt_sha256: str,
    controlled_run_id: str | None = None,
    reserved_cost_micros: int = 0,
) -> tuple[Battle, ResponseArm, ResponseArm, Job]:
    battle = Battle(
        id=battle_id,
        season_id=season.id,
        controlled_run_id=controlled_run_id,
        data_stratum="controlled" if controlled_run_id else "public_freeform",
        track="epicure_uplift",
        category="composition",
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        client_nonce_sha256=hashlib.sha256(f"nonce:{battle_id}".encode()).hexdigest(),
        requester_pseudonym=hashlib.sha256(f"requester:{battle_id}".encode()).hexdigest(),
        status="queued",
        reserved_cost_micros=reserved_cost_micros,
        retention_until=datetime.now(UTC) + timedelta(days=30),
    )
    session.add(battle)
    session.flush()
    left = ResponseArm(
        id=left_arm_id,
        battle_id=battle_id,
        side="left",
        condition="epicure_on",
        model_id="flavourbench/mock-openai-flagship",
        provider_slug="openrouter",
        status="queued",
        prompt_sha256=prompt_sha256,
        schema_sha256="4" * 64,
        tool_schema_sha256="5" * 64,
        epicure_release_id="qa-release",
        epicure_bundle_sha256="6" * 64,
    )
    right = ResponseArm(
        id=f"{left_arm_id}-peer",
        battle_id=battle_id,
        side="right",
        condition="epicure_off",
        model_id="flavourbench/mock-openai-flagship",
        provider_slug="openrouter",
        status="queued",
        prompt_sha256=prompt_sha256,
        schema_sha256="4" * 64,
        tool_schema_sha256="5" * 64,
        epicure_release_id="qa-release",
        epicure_bundle_sha256="6" * 64,
    )
    job = Job(
        id=job_id,
        kind="generate_battle",
        battle_id=battle_id,
        status="queued",
        max_attempts=3,
    )
    session.add_all([left, right, job])
    session.flush()
    battle.left_arm_id = left.id
    battle.right_arm_id = right.id
    session.flush()
    return battle, left, right, job


def test_openrouter_identity_requires_exact_coverage_for_every_round() -> None:
    spec = GenerationSpec(
        arm_id="identity-arm",
        battle_id="identity-battle",
        prompt="Compose a dish.",
        category="composition",
        model_id="vendor/model-alias",
        model_name="Identity model",
        provider_slug="provider-route",
        condition="epicure_on",
        idempotency_key="identity-test",
        expected_actual_model_id="vendor/model-versioned",
        expected_actual_provider_slug="Exact Provider",
    )
    valid = [
        {
            "generation_id": generation_id,
            "model": "vendor/model-versioned",
            "provider": "Exact Provider",
        }
        for generation_id in ("tool-generation", "final-generation")
    ]
    assert _verified_openrouter_generation_identity(
        spec,
        ["tool-generation", "final-generation"],
        valid,
    ) == ("vendor/model-versioned", "Exact Provider")

    invalid_cases = [
        [
            {**valid[0], "model": "vendor/substituted"},
            valid[1],
        ],
        [
            {**valid[0], "provider": "Substituted Provider"},
            valid[1],
        ],
        [
            {**valid[0], "model": "unknown"},
            valid[1],
        ],
        [
            valid[0],
            {**valid[1], "generation_id": "tool-generation"},
        ],
    ]
    for metadata in invalid_cases:
        with pytest.raises(ProviderError):
            _verified_openrouter_generation_identity(
                spec,
                ["tool-generation", "final-generation"],
                metadata,
            )
    with pytest.raises(ProviderError, match="coverage"):
        _verified_openrouter_generation_identity(
            spec,
            ["tool-generation", "tool-generation"],
            valid,
        )
    with pytest.raises(ProviderError, match="coverage"):
        _verified_openrouter_generation_identity(
            spec,
            ["tool-generation", "final-generation"],
            valid[:1],
        )


def test_settings_fail_closed_for_live_and_production_boundaries() -> None:
    with pytest.raises(ValidationError, match="live execution requires"):
        Settings(execution_mode="live", live_authorized=False, openrouter_api_key="")
    with pytest.raises(ValidationError, match="provider credential"):
        Settings(
            execution_mode="live",
            live_authorized=True,
            service_role="worker",
            openrouter_api_key="",
        )
    direct_cohere = Settings(
        execution_mode="live",
        live_authorized=True,
        service_role="worker",
        cohere_api_key="c" * 40,
    )
    assert direct_cohere.cohere_api_key
    production = {
        "environment": "production",
        "execution_mode": "live",
        "live_authorized": True,
        "database_url": "postgresql://flavourbench:test@db/flavourbench",
        "auto_create_schema": False,
        "service_token": "s" * 40,
        "admin_token": "a" * 40,
        "expert_token": "e" * 40,
        "pseudonym_secret": "p" * 40,
        "reviewer_identity_hmac_secret": "i" * 40,
        "reviewer_credential_hmac_secret": "c" * 40,
        "organization_api_key_hmac_secret": "o" * 40,
        "run_card_signing_secret": "r" * 40,
        "budget_authorization_signing_secret": "b" * 40,
    }
    with pytest.raises(ValidationError, match="unique service token"):
        Settings(**{**production, "service_token": "development-only-service-token"})
    with pytest.raises(ValidationError, match="unique pseudonym secret"):
        Settings(
            **{
                **production,
                "pseudonym_secret": "development-only-pseudonym-secret",
            }
        )
    with pytest.raises(ValidationError, match="unique run-card signing secret"):
        Settings(
            **{
                **production,
                "run_card_signing_secret": "development-only-run-card-signing-secret",
            }
        )
    settings = Settings(**production)
    assert settings.execution_mode == "live"


def test_security_pseudonyms_redaction_and_identity_detection_are_deterministic() -> None:
    supplied = "a" * 64
    assert request_pseudonym(supplied, "fallback") == supplied
    generated = request_pseudonym("invalid", "198.51.100.8")
    assert generated == request_pseudonym(None, "198.51.100.8")
    assert len(generated) == 64 and generated != request_pseudonym(None, "198.51.100.9")
    sanitized = sanitize_for_release("Email chef@example.com or call +34 612 345 678.")
    assert sanitized == "Email [EMAIL REDACTED] or call [PHONE REDACTED]."
    assert contains_identity_leak("I am Claude and this is my answer")
    assert contains_identity_leak("Prepared by Frontier Culinary Model", "Frontier Culinary Model")
    assert contains_identity_leak("I had access to Epicure and used it.")
    assert contains_identity_leak(
        "The tool returned a pairing score.",
        prompt="Interpret Epicure evidence.",
    )
    assert not contains_identity_leak(
        "Epicure evidence is suggestive, not causal.",
        prompt="Interpret Epicure evidence.",
    )
    assert not contains_identity_leak("Roast until mahogany at the edges")


def test_catalog_sync_discovers_compatibility_updates_and_retires_missing_models() -> None:
    init_database()
    first_id = "qa-provider/catalog-compatible"
    second_id = "qa-provider/catalog-retire"
    with session_scope() as session:
        session.add(
            CatalogModel(
                model_id=second_id,
                canonical_slug=second_id,
                name="Retire me",
                family="qa-provider",
            )
        )
    with session_scope() as session:
        counts = sync_catalog(
            session,
            [
                {
                    "id": first_id,
                    "canonical_slug": "qa-provider/catalog-compatible:stable",
                    "name": "Catalog compatible",
                    "supported_parameters": ["tools", "structured_outputs"],
                    "context_length": 131072,
                    "pricing": {"prompt": "0.000001"},
                    "top_provider": {"context_length": 131072},
                }
            ],
        )
        assert counts["compatible"] == 1
        assert counts["retired"] >= 1
    with session_scope() as session:
        compatible = session.get(CatalogModel, first_id)
        retired = session.get(CatalogModel, second_id)
        assert compatible and compatible.status == "compatible"
        assert compatible.supports_tools and compatible.supports_structured_outputs
        assert compatible.context_length == 131072
        assert retired and retired.status == "retired" and retired.retired_at is not None


def test_provider_parsers_and_prompt_hashes_fail_closed() -> None:
    assert _money_to_micros("0.012345") == 12_345
    assert _money_to_micros("0.0000001") == 1
    for invalid in ("not-money", "NaN", "Infinity", "-1"):
        with pytest.raises(ValueError):
            _money_to_micros(invalid)
    assert _extract_content({"content": "plain"}) == "plain"
    assert (
        _extract_content(
            {
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "image", "text": "skip"},
                    {"type": "output_text", "text": "two"},
                ]
            }
        )
        == "onetwo"
    )
    parsed = _parse_final(
        '```json\n{"answer_markdown":"Useful","ingredient_mentions":[],"constraints_addressed":[],"uncertainties":[]}\n```'
    )
    assert parsed["answer_markdown"] == "Useful"
    for malformed in ["not json", "[]", '{"answer_markdown":"x"}']:
        with pytest.raises(ProviderError):
            _parse_final(malformed)
    assert system_prompt_sha256("epicure_on") != system_prompt_sha256("epicure_off")
    assert len(system_prompt_sha256("epicure_on")) == 64
    matched_v1 = system_prompt_sha256(
        "epicure_on", "plain_text", "matched_evidence_v1"
    )
    matched_v2 = system_prompt_sha256(
        "epicure_on", "plain_text", "matched_evidence_v2"
    )
    assert matched_v1 == system_prompt_sha256(
        "epicure_off", "plain_text", "matched_evidence_v1"
    )
    assert matched_v2 == system_prompt_sha256(
        "epicure_off", "plain_text", "matched_evidence_v2"
    )
    assert matched_v2 != matched_v1


def test_mock_provider_is_deterministic_and_condition_specific() -> None:
    base = dict(
        arm_id="mock-arm",
        battle_id="mock-battle",
        prompt="Replace anchovy while retaining salinity and savoury depth.",
        category="substitution",
        model_id="flavourbench/mock-efficient-a",
        model_name="Mock efficient A",
        provider_slug="mock",
        idempotency_key="qa-mock-provider",
    )
    off = asyncio.run(MockProvider().generate(GenerationSpec(**base, condition="epicure_off")))
    on = asyncio.run(MockProvider().generate(GenerationSpec(**base, condition="epicure_on")))
    assert off.actual_model_id == on.actual_model_id == base["model_id"]
    assert off.tool_traces == []
    assert len(on.tool_traces) == 1 and on.tool_traces[0].name == "find_pairings"
    assert off.output_json["constraints_addressed"] == ["user-stated constraint"]


def test_openrouter_provider_executes_bounded_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used",
        openrouter_http_referer="https://epicure.kaikaku.ai/flavourbench",
        openrouter_title="Epicure FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=False,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_output_tokens=400,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=7,
        execution_mode="live",
        max_tool_rounds=8,
        max_tool_calls_per_round=4,
        max_tool_calls_total=16,
        max_tool_result_bytes=32_768,
        max_cumulative_tool_result_bytes=98_304,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    class FakeMcpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "find_pairings",
                    "description": "Find pairing evidence",
                    "inputSchema": {"type": "object"},
                }
            ]

        async def attest_runtime(self, *, expected, tools):
            return {
                **expected,
                "ingredient_count": 1,
                "embedding_dimensions": 1,
                "tool_count": len(tools),
                "mcp_protocol_version": "2025-06-18",
            }

        async def call_tool(self, name: str, arguments: dict[str, object]):
            return SimpleNamespace(
                text=f"evidence:{name}:{arguments['ingredient']}",
                structured={},
                is_error=False,
                latency_ms=1,
            )

    monkeypatch.setattr(provider_module, "McpSession", FakeMcpSession)

    async def scenario() -> None:
        instance = OpenRouterProvider()
        responses = iter(
            [
                {
                    "id": "tool-generation",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-a",
                                        "function": {
                                            "name": "find_pairings",
                                            "arguments": '{"ingredient":"pear"}',
                                        },
                                    },
                                    {
                                        "id": "call-b",
                                        "function": {
                                            "name": "find_pairings",
                                            "arguments": '{"ingredient":"miso"}',
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                },
                {
                    "id": "final-generation",
                    "model": "qa/model-20260715",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"answer_markdown":"Cook it","ingredient_mentions":[],'
                                    '"constraints_addressed":[],"uncertainties":[]}'
                                )
                            },
                        }
                    ],
                },
            ]
        )

        async def fake_post(*_args: object, **_kwargs: object) -> dict[str, object]:
            return next(responses)

        async def fake_cost(generation_id: str) -> dict[str, object]:
            return {
                "generation_id": generation_id,
                "cost_micros": 1,
                "provider": "QA",
                "model": "qa/model-20260715",
                "reconciled": True,
            }

        monkeypatch.setattr(instance, "_post", fake_post)
        monkeypatch.setattr(instance, "_generation_cost", fake_cost)
        result = await instance.generate(
            GenerationSpec(
                arm_id="arm",
                battle_id="battle",
                prompt="Pair pear and miso.",
                category="composition",
                model_id="qa/model",
                model_name="QA model",
                provider_slug="qa",
                condition="epicure_on",
                idempotency_key="parallel-tool-test",
                intermediate_max_tokens=400,
                supported_parameters=frozenset(
                    {
                        "max_tokens",
                        "response_format",
                        "structured_outputs",
                        "tool_choice",
                        "tools",
                    }
                ),
                decoding_parameters={"max_tokens": 400},
                expected_actual_model_id="qa/model-20260715",
                expected_actual_provider_slug="QA",
                endpoint_contract_sha256="f" * 64,
                protocol_bundle_sha256="b" * 64,
                expected_epicure_release_id="qa-release",
                expected_epicure_bundle_sha256="e" * 64,
                expected_epicure_application_sha256="a" * 64,
                expected_epicure_tool_schema_sha256="d" * 64,
            )
        )
        assert [trace.arguments["ingredient"] for trace in result.tool_traces] == [
            "pear",
            "miso",
        ]
        assert result.cost_micros == 2
        await instance.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("evidence_protocol", ("matched_evidence_v1", "matched_evidence_v2"))
def test_matched_evidence_protocol_changes_only_tool_availability_between_arms(
    monkeypatch: pytest.MonkeyPatch,
    evidence_protocol: str,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used",
        openrouter_http_referer="https://epicure.example/flavourbench",
        openrouter_title="FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=False,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_output_tokens=2048,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=7,
        execution_mode="live",
        max_tool_rounds=8,
        max_tool_calls_per_round=4,
        max_tool_calls_total=16,
        max_tool_result_bytes=32_768,
        max_cumulative_tool_result_bytes=98_304,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    class FakeMcpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "find_pairings",
                    "description": "Find pairing evidence",
                    "inputSchema": {"type": "object"},
                }
            ]

        async def attest_runtime(self, *, expected, tools):
            return {**expected, "tool_count": len(tools)}

        async def call_tool(self, name: str, arguments: dict[str, object]):
            return SimpleNamespace(
                text=f"evidence:{name}:{arguments['ingredient']}",
                structured={"pairing": "pear-miso"},
                is_error=False,
                latency_ms=1,
            )

    monkeypatch.setattr(provider_module, "McpSession", FakeMcpSession)

    async def scenario() -> None:
        provider = OpenRouterProvider()
        payloads: list[dict[str, object]] = []
        responses = iter(
            [
                {
                    "id": "off-planning",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_details": [{"type": "reasoning.encrypted"}],
                            },
                        }
                    ],
                },
                {
                    "id": "off-evidence",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_details": [{"type": "reasoning.encrypted"}],
                            },
                        }
                    ],
                },
                {
                    "id": "off-final",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "Cook the pear."}}
                    ],
                },
                {
                    "id": "on-planning",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_details": [{"type": "reasoning.encrypted"}],
                            },
                        }
                    ],
                },
                {
                    "id": "on-evidence",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-pairing",
                                        "function": {
                                            "name": "find_pairings",
                                            "arguments": '{"ingredient":"pear"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "id": "on-final",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "Cook the pear."}}
                    ],
                },
            ]
        )

        async def fake_post(payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            return next(responses)

        async def fake_cost(generation_id: str) -> dict[str, object]:
            return {
                "generation_id": generation_id,
                "cost_micros": 1,
                "provider": "QA",
                "model": "qa/model-20260801",
                "reconciled": True,
            }

        monkeypatch.setattr(provider, "_post", fake_post)
        monkeypatch.setattr(provider, "_generation_cost", fake_cost)
        common = {
            "battle_id": "battle",
            "prompt": "Compose pear with miso.",
            "category": "composition",
            "model_id": "qa/model",
            "model_name": "QA model",
            "provider_slug": "qa",
            "final_response_mode": "plain_text",
            "matched_planning": True,
            "intermediate_max_tokens": 1024,
            "evidence_protocol": evidence_protocol,
            "intermediate_reasoning_effort": "minimal",
            "final_reasoning_effort": "low",
            "required_tool_contract_sha256": "c" * 64,
            "supported_parameters": frozenset(
                {
                    "max_tokens",
                    "reasoning",
                    "response_format",
                    "structured_outputs",
                    "tool_choice",
                    "tools",
                }
            ),
            "decoding_parameters": {"max_tokens": 2048},
            "expected_actual_model_id": "qa/model-20260801",
            "expected_actual_provider_slug": "QA",
            "endpoint_contract_sha256": "f" * 64,
            "protocol_bundle_sha256": "b" * 64,
            "expected_epicure_release_id": "qa-release",
            "expected_epicure_bundle_sha256": "e" * 64,
            "expected_epicure_application_sha256": "a" * 64,
            "expected_epicure_tool_schema_sha256": "d" * 64,
        }
        off = await provider.generate(
            GenerationSpec(
                arm_id="off",
                condition="epicure_off",
                idempotency_key="matched-off",
                **common,
            )
        )
        on = await provider.generate(
            GenerationSpec(
                arm_id="on",
                condition="epicure_on",
                idempotency_key="matched-on",
                **common,
            )
        )

        off_planning, off_evidence, off_final, on_planning, on_evidence, on_final = payloads
        assert off_planning["messages"][0] == on_planning["messages"][0]
        assert off_planning["messages"][-1] == on_planning["messages"][-1]
        assert off_evidence["messages"][-1] == on_evidence["messages"][-1]
        assert off_evidence["messages"][-2]["reasoning_details"] == [
            {"type": "reasoning.encrypted"}
        ]
        assert off_final["messages"][-2]["reasoning_details"] == [
            {"type": "reasoning.encrypted"}
        ]
        assert off_final["messages"][-1] == on_final["messages"][-1]
        if evidence_protocol == "matched_evidence_v2":
            assert "Never infer binding, thickening, sweetness" in str(
                off_final["messages"][-1]["content"]
            )
            assert "data proves or confirms" in str(off_final["messages"][-1]["content"])
        assert "tools" not in off_evidence and on_evidence["tools"]
        assert (
            off_planning["reasoning"]
            == on_planning["reasoning"]
            == {
                "effort": "minimal",
                "exclude": True,
            }
        )
        assert off_evidence["reasoning"] == on_evidence["reasoning"]
        assert (
            off_final["reasoning"]
            == on_final["reasoning"]
            == {
                "effort": "low",
                "exclude": True,
            }
        )
        assert not off.tool_traces and len(on.tool_traces) == 1
        assert [item["phase"] for item in off.intermediate_outputs] == [
            "planning",
            "evidence_decision",
        ]
        assert [item["phase"] for item in on.intermediate_outputs] == [
            "planning",
            "tool_selection",
        ]
        assert off.intermediate_outputs[0]["visible_content_status"] == (
            "reasoning_only_or_suppressed"
        )
        await provider.aclose()

    asyncio.run(scenario())


def test_required_tool_contract_is_a_direct_tool_first_unranked_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used",
        openrouter_http_referer="https://epicure.example/flavourbench",
        openrouter_title="FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=False,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_output_tokens=4096,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=7,
        execution_mode="live",
        max_tool_rounds=2,
        max_tool_calls_per_round=4,
        max_tool_calls_total=8,
        max_tool_result_bytes=32_768,
        max_cumulative_tool_result_bytes=98_304,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    class FakeMcpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "find_pairings",
                    "description": "Find pairing evidence",
                    "inputSchema": {"type": "object"},
                }
            ]

        async def attest_runtime(self, *, expected, tools):
            return {**expected, "tool_count": len(tools)}

        async def call_tool(self, name: str, arguments: dict[str, object]):
            return SimpleNamespace(
                text=f"evidence:{name}:{arguments['ingredient']}",
                structured={"pairing": "pear-miso"},
                is_error=False,
                latency_ms=1,
            )

    monkeypatch.setattr(provider_module, "McpSession", FakeMcpSession)

    async def scenario() -> None:
        provider = OpenRouterProvider()
        payloads: list[dict[str, object]] = []
        responses = iter(
            [
                {
                    "id": "contract-tool",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-pairing",
                                        "function": {
                                            "name": "find_pairings",
                                            "arguments": '{"ingredient":"pear"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "id": "contract-final",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "Cook the pear."}}
                    ],
                },
            ]
        )

        async def fake_post(payload, *_args, **_kwargs):
            payloads.append(copy.deepcopy(payload))
            return next(responses)

        async def fake_cost(generation_id: str) -> dict[str, object]:
            return {
                "generation_id": generation_id,
                "cost_micros": 1,
                "provider": "QA",
                "model": "qa/model-20260801",
                "reconciled": True,
            }

        monkeypatch.setattr(provider, "_post", fake_post)
        monkeypatch.setattr(provider, "_generation_cost", fake_cost)
        result = await provider.generate(
            GenerationSpec(
                arm_id="tool-contract",
                battle_id="battle",
                prompt="Use find_pairings before answering.",
                category="evidence",
                model_id="qa/model",
                model_name="QA model",
                provider_slug="qa",
                condition="epicure_on",
                idempotency_key="direct-tool-contract",
                final_response_mode="plain_text",
                matched_planning=True,
                intermediate_max_tokens=4096,
                required_tool_contract_max_intermediate_tokens=1024,
                evidence_protocol="matched_evidence_v1",
                intermediate_reasoning_effort="low",
                final_reasoning_effort="low",
                required_tool_contract_sha256="c" * 64,
                tool_choice="required",
                tool_contract_diagnostic=True,
                supported_parameters=frozenset(
                    {
                        "max_tokens",
                        "reasoning",
                        "response_format",
                        "structured_outputs",
                        "tool_choice",
                        "tools",
                    }
                ),
                decoding_parameters={"max_tokens": 4096},
                expected_actual_model_id="qa/model-20260801",
                expected_actual_provider_slug="QA",
                endpoint_contract_sha256="f" * 64,
                protocol_bundle_sha256="b" * 64,
                expected_epicure_release_id="qa-release",
                expected_epicure_bundle_sha256="e" * 64,
                expected_epicure_application_sha256="a" * 64,
                expected_epicure_tool_schema_sha256="d" * 64,
            )
        )

        assert len(payloads) == 2
        first, final = payloads
        assert first["tool_choice"] == "required"
        assert first["max_tokens"] == 1024
        assert first["tools"]
        assert first["messages"][-1] == {
            "role": "user",
            "content": "Use find_pairings before answering.",
        }
        assert all("Draft a compact checklist" not in str(message) for message in first["messages"])
        assert final["messages"][-1]["content"].startswith("Return only the final")
        assert [item["phase"] for item in result.intermediate_outputs] == ["tool_selection"]
        assert result.tool_traces[0].name == "find_pairings"
        await provider.aclose()

    asyncio.run(scenario())


def test_openrouter_retry_preferences_and_generation_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used-by-mock-transport",
        openrouter_http_referer="https://epicure.kaikaku.ai/flavourbench",
        openrouter_title="Epicure FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=True,
        openrouter_accounting_attempts=1,
        openrouter_accounting_initial_delay_seconds=0,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_provider_attempts=3,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    calls = {"post": 0, "get": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.headers["x-openrouter-cache"] == "false"
            calls["post"] += 1
            if calls["post"] == 1:
                return httpx.Response(503, json={"error": "transient"})
            return httpx.Response(200, json={"id": "generation-qa", "choices": []})
        calls["get"] += 1
        return httpx.Response(
            200,
            json={"data": {"total_cost": "0.001234", "provider_name": "QA Provider"}},
        )

    async def scenario() -> None:
        events: list[ProviderAttemptEvent] = []
        instance = OpenRouterProvider(attempt_sink=events.append)
        await instance.client.aclose()
        instance.client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1/",
            headers=instance.client.headers,
            transport=httpx.MockTransport(handler),
        )
        await instance.accounting_client.aclose()
        instance.accounting_client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1/",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(UncertainDeliveryError, match="ambiguous failure"):
            await instance._post({"model": "qa"}, "stable-key")
        assert instance._provider_preferences("anthropic") == {
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "only": ["anthropic"],
            "zdr": True,
        }
        assert await instance._generation_cost("generation-qa") == {
            "generation_id": "generation-qa",
            "cost_micros": 1_234,
            "provider": "QA Provider",
            "model": "unknown",
            "reconciled": True,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "native_tokens_prompt": 0,
            "native_tokens_completion": 0,
            "generation_time_ms": 0,
            "upstream_latency_ms": 0,
        }
        assert [event.event_type for event in events] == [
            "request_started",
            "uncertain_delivery",
        ]
        assert all(len(event.payload_sha256) == 64 for event in events[:2])
        await instance.aclose()

    asyncio.run(scenario())
    assert calls == {"post": 1, "get": 1}


def test_openrouter_retries_only_explicit_pre_acceptance_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used-by-mock-transport",
        openrouter_http_referer="https://epicure.example/flavourbench",
        openrouter_title="FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=False,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_provider_attempts=2,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    delays: list[float] = []

    async def no_delay(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(provider_module.asyncio, "sleep", no_delay)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})
        return httpx.Response(
            200,
            json={
                "id": "generation-after-429",
                "model": "qa/model",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    async def scenario() -> None:
        events: list[ProviderAttemptEvent] = []
        instance = OpenRouterProvider(attempt_sink=events.append)
        await instance.client.aclose()
        instance.client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1/",
            headers=instance.client.headers,
            transport=httpx.MockTransport(handler),
        )
        response = await instance._post(
            {"model": "qa/model"},
            "stable-429-key",
            arm_id="arm-429",
            phase="planning",
        )
        assert response["id"] == "generation-after-429"
        assert response["_flavourbench_retries"] == 1
        assert [event.event_type for event in events] == [
            "request_started",
            "request_rejected",
            "retry_scheduled",
            "request_started",
            "response_received",
        ]
        assert len({event.request_key_sha256 for event in events}) == 1
        await instance.aclose()

    asyncio.run(scenario())
    assert calls == 2
    assert len(delays) == 1 and 0 < delays[0] <= 30


def test_openrouter_rejects_cached_responses_and_reconciles_accepted_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        openrouter_api_key="not-used-by-mock-transport",
        openrouter_http_referer="https://epicure.kaikaku.ai/flavourbench",
        openrouter_title="Epicure FlavourBench",
        openrouter_base_url="https://openrouter.test/api/v1",
        openrouter_accounting_base_url="https://openrouter.test/api/v1",
        openrouter_timeout_seconds=10,
        cloudflare_ai_gateway_token="",
        openrouter_zdr=True,
        openrouter_accounting_attempts=1,
        openrouter_accounting_initial_delay_seconds=0,
        openrouter_max_prompt_price_per_mtok=None,
        openrouter_max_completion_price_per_mtok=None,
        max_provider_attempts=1,
        max_output_tokens=512,
        decoding_temperature=0.2,
        decoding_top_p=0.95,
        decoding_seed=7,
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-openrouter-cache"] == "false"
        return httpx.Response(
            200,
            headers={"X-OpenRouter-Cache-Status": "HIT"},
            json={
                "id": "cached-generation",
                "model": "qa/cached",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "cached"}}
                ],
            },
        )

    async def scenario() -> None:
        events: list[ProviderAttemptEvent] = []
        instance = OpenRouterProvider(attempt_sink=events.append)
        await instance.client.aclose()
        instance.client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1/",
            headers=instance.client.headers,
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(ProviderError):
            await instance._post(
                {"model": "qa/cached"},
                "cache-test-key",
                arm_id="cache-test-arm",
                phase="final",
            )
        assert [event.event_type for event in events] == [
            "request_started",
            "response_received",
            "invalid_response",
        ]
        assert events[1].metadata["openrouter_cache_status"] == "HIT"

        async def fake_cost(generation_id: str) -> dict[str, object]:
            assert generation_id == "cached-generation"
            return {
                "generation_id": generation_id,
                "cost_micros": 321,
                "provider": "QA Provider",
                "model": "qa/cached-versioned",
                "reconciled": True,
                "tokens_prompt": 11,
                "tokens_completion": 7,
            }

        monkeypatch.setattr(instance, "_generation_cost", fake_cost)
        failure = await instance.reconcile_failure(
            GenerationSpec(
                arm_id="cache-test-arm",
                battle_id="cache-test-battle",
                prompt="Compose a pear dish.",
                category="composition",
                model_id="qa/cached",
                model_name="QA cached",
                provider_slug="qa",
                condition="epicure_off",
                idempotency_key="cache-test-key",
                decoding_parameters={
                    "max_tokens": 512,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "seed": 7,
                },
            ),
            ProviderError("cached provider response"),
        )
        assert failure is not None
        assert failure.generation_ids == ["cached-generation"]
        assert failure.cost_micros == 321 and failure.cost_reconciled is True
        assert failure.prompt_tokens == 11 and failure.completion_tokens == 7
        assert failure.actual_model_id == "qa/cached-versioned"
        assert failure.provider_slug == "QA Provider"
        await instance.aclose()

    asyncio.run(scenario())


def test_mcp_envelope_decoding_and_error_contracts() -> None:
    envelope = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    assert _decode_response('{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}') == envelope
    sse = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n'
    assert _decode_response(sse) == envelope
    assert _read_result(envelope) == {"tools": []}
    with pytest.raises(RuntimeError, match="unsupported response"):
        _decode_response("event: ping\n")
    with pytest.raises(RuntimeError, match="protocol failed"):
        _read_result({"error": {"message": "protocol failed"}})
    with pytest.raises(RuntimeError, match="missing a result"):
        _read_result({"jsonrpc": "2.0", "id": 1})


def test_mcp_tool_result_prefers_text_then_structured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session = McpSession()

        async def text_post(*_args: object, **_kwargs: object):
            return (
                {
                    "result": {
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "text", "text": " second"},
                        ],
                        "structuredContent": {"score": 0.42},
                        "isError": False,
                    }
                },
                httpx.Headers(),
            )

        monkeypatch.setattr(session, "_post", text_post)
        result = await session.call_tool("find_pairings", {"ingredients": ["pear"]})
        assert result.text == "first second" and result.structured == {"score": 0.42}

        async def structured_post(*_args: object, **_kwargs: object):
            return ({"result": {"content": [], "structuredContent": {"ok": True}}}, httpx.Headers())

        monkeypatch.setattr(session, "_post", structured_post)
        fallback = await session.call_tool("health", {})
        assert fallback.text == '{"ok": true}'
        await session.client.aclose()

    asyncio.run(scenario())


def test_validators_keep_schema_identity_constraints_and_tools_separate() -> None:
    answer = (
        "Roast the tofu until the edges are deeply golden, then season while hot. "
        "Keep the pieces well spaced so steam can escape, and test one piece before "
        "serving the whole tray."
    )
    output = {
        "answer_markdown": answer,
        "ingredient_mentions": ["tofu"],
        "constraints_addressed": ["without deep frying"],
        "uncertainties": ["Oven performance varies"],
    }
    results = validate_output(
        prompt="Make crisp tofu without deep frying in under 40 minutes.",
        output=output,
        answer=answer,
        model_name="QA Model",
        tool_errors=1,
        tool_calls=2,
    )
    by_name = {item.name: item for item in results}
    assert by_name["structured_response"].status == "pass"
    assert by_name["identity_blinding"].status == "pass"
    assert by_name["semantic_completion"].status == "pass"
    assert by_name["evidence_claim_boundary"].status == "pass"
    assert by_name["task_surface_integrity"].status == "pass"
    assert by_name["constraint_acknowledgement"].status == "pass"
    assert by_name["tool_execution"].status == "warn"
    assert by_name["tool_execution"].score_milli == 500


@pytest.mark.parametrize(
    ("answer", "reason"),
    (
        ("## Recommendation", "fewer_than_120_characters"),
        (
            "Use a restrained amount of acid, then taste the sauce and adjust because",
            "dangling_terminal_clause",
        ),
        (
            "Use the following measured test before scaling the recipe.\n\n```text\n"
            "20 g sauce, 0.2 g salt",
            "unclosed_code_fence",
        ),
        (
            "Start with a small test portion, keep the heat low, and taste before changing "
            "the entire batch. This gives enough evidence to adjust salt, acid, and texture "
            "without wasting the original preparation.\n\n## Final dosage",
            "ends_with_markdown_heading",
        ),
    ),
)
def test_semantic_completion_rejects_strong_incompletion_signals(
    answer: str,
    reason: str,
) -> None:
    output = {
        "answer_markdown": answer,
        "ingredient_mentions": [],
        "constraints_addressed": ["stated constraint"],
        "uncertainties": [],
    }
    result = {
        item.name: item
        for item in validate_output(
            prompt="Give a practical culinary recommendation.",
            output=output,
            answer=answer,
            model_name="QA Model",
            tool_errors=0,
            tool_calls=0,
        )
    }["semantic_completion"]
    assert result.status == "fail"
    assert reason in result.detail["failure_reasons"]


def test_semantic_completion_rejects_provider_length_finish() -> None:
    answer = (
        "Build the sauce gradually, tasting after each addition. Keep the heat moderate so the "
        "emulsion remains stable, and reserve a little cooking liquid for the final adjustment."
    )
    result = {
        item.name: item
        for item in validate_output(
            prompt="Give a practical culinary recommendation.",
            output={
                "answer_markdown": answer,
                "ingredient_mentions": [],
                "constraints_addressed": [],
                "uncertainties": [],
            },
            answer=answer,
            model_name="QA Model",
            tool_errors=0,
            tool_calls=0,
            finish_reason="length",
        )
    }["semantic_completion"]
    assert result.status == "fail"
    assert "non_normal_finish_reason" in result.detail["failure_reasons"]


@pytest.mark.parametrize(
    "answer",
    (
        "The similarity data confirms that guar gum is a functional binder for this gnocchi.",
        "The embedding score therefore demonstrates the mechanism responsible for sweetness.",
        "The pairing network proves this dosage is food-safe for every diner.",
    ),
)
def test_evidence_claim_boundary_flags_unsupported_inference_language(answer: str) -> None:
    result = {
        item.name: item
        for item in validate_output(
            prompt="Interpret the culinary evidence and recommend a practical next step.",
            output={
                "answer_markdown": answer,
                "ingredient_mentions": [],
                "constraints_addressed": [],
                "uncertainties": [],
            },
            answer=answer,
            model_name="QA Model",
            tool_errors=0,
            tool_calls=1,
        )
    }["evidence_claim_boundary"]
    assert result.status == "warn"
    assert result.detail["warning_count"] == 1
    assert "sentence_sha256" in result.detail["findings"][0]
    assert "sentence" not in result.detail["findings"][0]


def test_evidence_claim_boundary_allows_calibrated_interpretation() -> None:
    answer = (
        "The pairing score suggests cumin as a candidate to test with chicken, but it does not "
        "prove that cumin belongs in this Thai curry or establish a culinary mechanism."
    )
    result = {
        item.name: item
        for item in validate_output(
            prompt="Interpret the pairing evidence.",
            output={
                "answer_markdown": answer,
                "ingredient_mentions": ["cumin", "chicken"],
                "constraints_addressed": [],
                "uncertainties": ["Pairing scores are associative"],
            },
            answer=answer,
            model_name="QA Model",
            tool_errors=0,
            tool_calls=1,
        )
    }["evidence_claim_boundary"]
    assert result.status == "pass"


def test_task_surface_integrity_warns_without_deciding_construct_validity() -> None:
    answer = (
        "Describe the observable texture in words, then run a small controlled batch before "
        "changing the full recipe. Record time, temperature, and ingredient mass."
    )
    result = {
        item.name: item
        for item in validate_output(
            prompt="Use the attached photo and https://example.test/context to diagnose the cake.",
            output={
                "answer_markdown": answer,
                "ingredient_mentions": [],
                "constraints_addressed": [],
                "uncertainties": ["The photo is unavailable"],
            },
            answer=answer,
            model_name="QA Model",
            tool_errors=0,
            tool_calls=0,
        )
    }["task_surface_integrity"]
    assert result.status == "warn"
    assert result.detail["failure_reasons"] == [
        "external_url_dependency_signal",
        "visual_context_dependency_signal",
    ]


def test_job_claim_recovery_and_unsupported_kind() -> None:
    init_database()
    stale_id = "qa-stale-job"
    unsupported_id = "qa-unsupported-job"
    with session_scope() as session:
        session.execute(delete(Job))
        stale = Job(
            id=stale_id,
            kind="maintenance",
            status="queued",
        )
        session.add_all([stale, Job(id=unsupported_id, kind="unknown-qa-kind", status="queued")])
        session.flush()
        stale.status = "running"
        stale.claimed_by = "dead-worker"
        stale.claimed_at = datetime.now(UTC) - timedelta(hours=1)
    with session_scope() as session:
        assert recover_stale_jobs(session) >= 1
        stale = session.get(Job, stale_id)
        assert stale and stale.status == "queued" and stale.claimed_by is None
        first_lease = claim_job(session, "qa-worker")
        assert first_lease is not None
    asyncio.run(process_job(*first_lease))
    with session_scope() as session:
        second_lease = claim_job(session, "qa-worker")
    if second_lease is not None:
        asyncio.run(process_job(*second_lease))
    with session_scope() as session:
        unsupported = session.get(Job, unsupported_id)
        assert unsupported and unsupported.status == "failed"
        assert unsupported.last_error == "unsupported job kind: unknown-qa-kind"


def test_stale_provider_attempt_is_not_replayed_and_keeps_budget_reserved() -> None:
    init_database()
    seed_database()
    battle_id = "qa-uncertain-battle"
    arm_id = "qa-uncertain-arm"
    job_id = "qa-uncertain-job"
    with session_scope() as session:
        season = Season(
            slug="qa-uncertain-season",
            name="Uncertain delivery isolation fixture",
            epicure_release_id="qa-release",
        )
        session.add(season)
        session.flush()
        battle, left, right, job = _queued_generation_fixture(
            session,
            season=season,
            battle_id=battle_id,
            left_arm_id=arm_id,
            job_id=job_id,
            prompt="Design a practical tomato starter with enough detail.",
            prompt_sha256="a" * 64,
            reserved_cost_micros=250_000,
        )
        battle.status = "running"
        left.status = "running"
        right.status = "running"
        job.status = "running"
        job.claimed_by = "dead-worker"
        job.claimed_at = datetime.now(UTC) - timedelta(hours=1)
        session.flush()
        session.add(
            GenerationAttempt(
                attempt_id="00000000-0000-0000-0000-000000000001",
                arm_id=arm_id,
                request_key_sha256="1" * 64,
                phase="final",
                attempt_index=0,
                event_type="request_started",
                payload_sha256="2" * 64,
            )
        )

    with session_scope() as session:
        assert recover_stale_jobs(session) >= 1
    with session_scope() as session:
        job = session.get(Job, job_id)
        battle = session.get(Battle, battle_id)
        arm = session.get(ResponseArm, arm_id)
        assert job and job.status == "uncertain" and job.claimed_by is None
        assert battle and battle.status == "failed"
        assert battle.reserved_cost_micros == 250_000
        assert arm and arm.status == "uncertain"


def test_superseded_worker_cannot_start_paid_work_but_can_append_terminal_evidence() -> None:
    init_database()
    seed_database()
    battle_id = "qa-fenced-battle"
    arm_id = "qa-fenced-arm"
    job_id = "qa-fenced-job"
    with session_scope() as session:
        session.execute(delete(Job))
        season = session.scalar(select(Season).where(Season.slug == "season-0"))
        assert season is not None
        _queued_generation_fixture(
            session,
            season=season,
            battle_id=battle_id,
            left_arm_id=arm_id,
            job_id=job_id,
            prompt="Create a fennel dish with a practical sequence.",
            prompt_sha256="1" * 64,
        )
    with session_scope() as session:
        first = claim_job(session, "worker-a")
        assert first is not None
    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.claimed_at = datetime.now(UTC) - timedelta(hours=1)
    with session_scope() as session:
        assert recover_stale_jobs(session) == 1
    with session_scope() as session:
        second = claim_job(session, "worker-b")
        assert second is not None and second[0] == job_id

    started = ProviderAttemptEvent(
        attempt_id="00000000-0000-0000-0000-000000000101",
        arm_id=arm_id,
        request_key_sha256="7" * 64,
        phase="final",
        attempt_index=0,
        event_type="request_started",
        payload_sha256="8" * 64,
    )
    with pytest.raises(LostJobLease):
        _persist_provider_attempt(
            started,
            job_id=first[0],
            claimed_by=first[1],
            claim_attempt=first[2],
        )
    _persist_provider_attempt(
        started,
        job_id=second[0],
        claimed_by=second[1],
        claim_attempt=second[2],
    )
    duplicate_dispatch = ProviderAttemptEvent(
        **{
            **started.__dict__,
            "attempt_id": "00000000-0000-0000-0000-000000000102",
        }
    )
    with pytest.raises(ProviderError, match="duplicate external-work dispatch"):
        _persist_provider_attempt(
            duplicate_dispatch,
            job_id=second[0],
            claimed_by=second[1],
            claim_attempt=second[2],
        )
    with session_scope() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.claimed_at = datetime.now(UTC) - timedelta(hours=1)
    with session_scope() as session:
        assert recover_stale_jobs(session) == 1
        job = session.get(Job, job_id)
        assert job is not None and job.status == "uncertain"

    received = ProviderAttemptEvent(
        **{
            **started.__dict__,
            "event_type": "response_received",
            "generation_id": "generation-after-lease-loss",
            "payload_sha256": "9" * 64,
        }
    )
    _persist_provider_attempt(
        received,
        job_id=second[0],
        claimed_by=second[1],
        claim_attempt=second[2],
    )
    with session_scope() as session:
        events = session.scalars(
            select(GenerationAttempt).where(GenerationAttempt.attempt_id == started.attempt_id)
        ).all()
        assert [event.event_type for event in events] == [
            "request_started",
            "response_received",
        ]
        # This fixture deliberately leaves an accepted provider request without
        # accounting evidence. Remove it after the lease-fencing assertion so
        # later integration tests can audit the shared test season strictly.
        session.execute(
            delete(GenerationAttempt).where(
                GenerationAttempt.arm_id.in_(
                    [
                        arm_id,
                        f"{arm_id}-peer",
                    ]
                )
            )
        )
        # Battle-to-arm and arm-to-battle foreign keys form an intentional
        # cycle.  Defer SQLite's checks until both fixture sides are gone.
        session.execute(text("PRAGMA defer_foreign_keys = ON"))
        dependent_parameters = {
            "battle_id": battle_id,
            "left_arm_id": arm_id,
            "right_arm_id": f"{arm_id}-peer",
        }
        for statement in (
            "DELETE FROM tool_calls WHERE arm_id IN (:left_arm_id, :right_arm_id)",
            "DELETE FROM validator_results WHERE arm_id IN (:left_arm_id, :right_arm_id)",
            "DELETE FROM bedrock_billing_crosscheck_arms "
            "WHERE arm_id IN (:left_arm_id, :right_arm_id)",
            "DELETE FROM cost_events WHERE battle_id = :battle_id "
            "OR arm_id IN (:left_arm_id, :right_arm_id)",
            "DELETE FROM votes WHERE battle_id = :battle_id",
            "DELETE FROM controlled_run_assignments WHERE battle_id = :battle_id",
            "DELETE FROM jobs WHERE battle_id = :battle_id",
        ):
            session.execute(text(statement), dependent_parameters)
        session.execute(delete(ResponseArm).where(ResponseArm.battle_id == battle_id))
        session.execute(delete(Battle).where(Battle.id == battle_id))


def test_revoked_controlled_run_cannot_start_provider_or_mcp_work() -> None:
    init_database()
    seed_database()
    battle_id = "qa-revoked-controlled-battle"
    arm_id = "qa-revoked-controlled-arm"
    job_id = "qa-revoked-controlled-job"
    run_id = "qa-revoked-controlled-run"
    with session_scope() as session:
        session.execute(delete(Job))
        season = session.scalar(select(Season).where(Season.slug == "season-0"))
        assert season is not None
        controlled_run = ControlledRun(
            id=run_id,
            season_id=season.id,
            organization_reference_sha256="a" * 64,
            access_token_sha256="b" * 64,
            status="active",
            protocol_version="flavourbench-controlled-run-v1",
            rater_plan_sha256="c" * 64,
            analysis_plan_sha256="d" * 64,
            budget_cap_micros=1_000_000,
            run_card_json={"fixture": True},
            run_card_sha256="e" * 64,
            run_card_signature="f" * 64,
        )
        session.add(controlled_run)
        session.flush()
        _queued_generation_fixture(
            session,
            season=season,
            battle_id=battle_id,
            left_arm_id=arm_id,
            job_id=job_id,
            prompt="Create a practical fennel dish.",
            prompt_sha256="1" * 64,
            controlled_run_id=run_id,
        )
    with session_scope() as session:
        claim = claim_job(session, "worker-controlled")
        assert claim is not None and claim[0] == job_id
    with session_scope() as session:
        controlled_run = session.get(ControlledRun, run_id)
        assert controlled_run is not None
        controlled_run.status = "revoked"
        controlled_run.revoked_at = datetime.now(UTC)

    for index, event_type in enumerate(
        ("request_started", "mcp_session_started", "mcp_call_started"), start=1
    ):
        event = ProviderAttemptEvent(
            attempt_id=f"00000000-0000-0000-0000-{index:012d}",
            arm_id=arm_id,
            request_key_sha256="7" * 64,
            phase="final" if event_type == "request_started" else "tool",
            attempt_index=0,
            event_type=event_type,
            payload_sha256="8" * 64,
        )
        with pytest.raises(ProviderError, match="controlled run was revoked"):
            _persist_provider_attempt(
                event,
                job_id=claim[0],
                claimed_by=claim[1],
                claim_attempt=claim[2],
            )

    with session_scope() as session:
        assert (
            session.scalar(select(GenerationAttempt).where(GenerationAttempt.arm_id == arm_id))
            is None
        )
        battle = session.get(Battle, battle_id)
        job = session.get(Job, job_id)
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle_id)).all()
        assert battle is not None and job is not None and len(arms) == 2
        arm_completed_at = datetime.now(UTC) + timedelta(milliseconds=10)
        for arm in arms:
            arm.status = "failed"
            arm.error_code = "ControlledRunRevoked"
            arm.error_detail = "Controlled run was revoked after the worker claim."
            arm.completed_at = arm_completed_at
            _record_known_zero_cost(
                session,
                battle,
                arm,
                reason="revoked_controlled_run_fixture_cleanup",
            )
        session.flush()
        battle.status = "failed"
        battle.completed_at = arm_completed_at + timedelta(milliseconds=1)
        job.status = "failed"
        job.completed_at = battle.completed_at
        session.flush()
        reconcile_battle_cost(session, battle)


def test_bundled_smoke_run_completes_three_unranked_battles() -> None:
    result = asyncio.run(smoke())
    assert result["mode"] == "mock" and result["ranked"] is False
    assert len(result["battles"]) == 3
    assert all(item["status"] == "complete" for item in result["battles"])
    assert all(1 <= len(item["models"]) <= 2 for item in result["battles"])
    assert all(
        set(item["conditions"]) in ({"epicure_on"}, {"epicure_on", "epicure_off"})
        for item in result["battles"]
    )
