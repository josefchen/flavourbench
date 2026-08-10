from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import delete, select

import flavourbench.engine as engine_module
import flavourbench.live_smoke as live_smoke_module
from flavourbench.database import init_database, session_scope
from flavourbench.endpoint_contract import endpoint_contract_sha256
from flavourbench.engine import _generation_spec, _record_result
from flavourbench.live_smoke import (
    _model_identity_matches,
    _provider_identity_matches,
    frozen_generation_contract,
)
from flavourbench.main import admin_model_smoke
from flavourbench.models import (
    Battle,
    CatalogModel,
    CostEvent,
    GenerationAttempt,
    Incident,
    Job,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    ToolCall,
    ValidatorResult,
    Vote,
)
from flavourbench.protocol_contract import build_protocol_bundle
from flavourbench.provider import (
    FINAL_SCHEMA_SHA256,
    GenerationResult,
    ProviderError,
    system_prompt_sha256,
)
from flavourbench.schemas import (
    ManifestEntry,
    ModelSmokeArtifactCreate,
    ModelSmokeCreate,
)

SUPPORTED = [
    "max_tokens",
    "response_format",
    "seed",
    "structured_outputs",
    "tool_choice",
    "tools",
]
DECODING = {"max_tokens": 600, "seed": 20260715}
ENDPOINT_DOCUMENT_SHA256 = hashlib.sha256(b"endpoint-document-v1").hexdigest()


def _smoke_evidence(
    actual_model_id: str,
    actual_provider_slug: str,
) -> tuple[ModelSmokeArtifactCreate, str]:
    artifact = ModelSmokeArtifactCreate(
        requestSha256="a" * 64,
        responseSha256="b" * 64,
        providerRequestIdSha256="c" * 64,
        generationId="generation-smoke-contract",
        actualModelId=actual_model_id,
        actualProviderSlug=actual_provider_slug,
        toolsPassed=True,
        structuredOutputPassed=True,
        dataCollectionDenied=True,
        schemaSha256=FINAL_SCHEMA_SHA256,
        toolSchemaSha256="1" * 64,
        toolTraceSha256="d" * 64,
        structuredOutputSha256="e" * 64,
        costMicros=100,
        costReconciled=True,
        completedAt="2026-07-21T00:00:00+00:00",
    )
    payload = artifact.model_dump(mode="json")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return artifact, digest


@pytest.fixture(autouse=True)
def clean_endpoint_contract_test_records():
    init_database()
    yield
    with session_scope() as session:
        season_ids = list(
            session.scalars(
                select(Season.id).where(Season.slug.like("endpoint-contract-season-%"))
            ).all()
        )
        battle_ids = (
            list(
                session.scalars(
                    select(Battle.id).where(Battle.season_id.in_(season_ids))
                ).all()
            )
            if season_ids
            else []
        )
        arm_ids = (
            list(
                session.scalars(
                    select(ResponseArm.id).where(ResponseArm.battle_id.in_(battle_ids))
                ).all()
            )
            if battle_ids
            else []
        )
        if arm_ids:
            session.execute(delete(ValidatorResult).where(ValidatorResult.arm_id.in_(arm_ids)))
            session.execute(delete(ToolCall).where(ToolCall.arm_id.in_(arm_ids)))
            session.execute(
                delete(GenerationAttempt).where(GenerationAttempt.arm_id.in_(arm_ids))
            )
        if battle_ids:
            session.execute(delete(Vote).where(Vote.battle_id.in_(battle_ids)))
            session.execute(delete(Job).where(Job.battle_id.in_(battle_ids)))
            session.execute(delete(CostEvent).where(CostEvent.battle_id.in_(battle_ids)))
            session.execute(delete(Incident).where(Incident.battle_id.in_(battle_ids)))
            session.execute(delete(ResponseArm).where(ResponseArm.battle_id.in_(battle_ids)))
            session.execute(delete(Battle).where(Battle.id.in_(battle_ids)))
        if season_ids:
            session.execute(delete(SeasonModel).where(SeasonModel.season_id.in_(season_ids)))
            session.execute(delete(CostEvent).where(CostEvent.season_id.in_(season_ids)))
            session.execute(delete(Season).where(Season.id.in_(season_ids)))
        session.execute(
            delete(RunEvent).where(RunEvent.entity_id.like("endpoint-contract-%"))
        )
        session.execute(
            delete(CatalogModel).where(CatalogModel.model_id.like("endpoint-contract/%"))
        )


def _contract_digest(model_id: str) -> str:
    return endpoint_contract_sha256(
        model_id=model_id,
        provider_slug="openai/flex",
        expected_actual_model_id=f"{model_id}-20260709",
        expected_actual_provider_slug="OpenAI",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=4096,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
    )


def _create_live_arm(session, suffix: str) -> tuple[Battle, ResponseArm, SeasonModel]:
    model_id = f"endpoint-contract/model-{suffix}"
    manifest = hashlib.sha256(f"manifest:{suffix}".encode()).hexdigest()
    model = CatalogModel(
        model_id=model_id,
        canonical_slug=f"{model_id}-20260709",
        name=f"Endpoint contract model {suffix}",
        family="endpoint-contract",
        status="smoke_passed",
        supports_tools=True,
        supports_structured_outputs=True,
    )
    season = Season(
        slug=f"endpoint-contract-season-{suffix}",
        name=f"Endpoint contract season {suffix}",
        status="pilot",
        manifest_sha256=manifest,
        tool_registry_sha256="1" * 64,
        epicure_release_id="endpoint-contract-release",
        epicure_bundle_sha256="2" * 64,
        epicure_application_sha256="3" * 64,
        analysis_plan_sha256="6" * 64,
    )
    session.add_all([model, season])
    session.flush()
    protocol_bundle, protocol_bundle_sha256 = build_protocol_bundle(
        tool_registry_sha256=season.tool_registry_sha256,
        epicure_release_id=season.epicure_release_id,
        epicure_bundle_sha256=season.epicure_bundle_sha256,
        epicure_application_sha256=season.epicure_application_sha256,
        analysis_plan_sha256=season.analysis_plan_sha256,
    )
    season.protocol_bundle_json = protocol_bundle
    season.protocol_bundle_sha256 = protocol_bundle_sha256
    slot = SeasonModel(
        season_id=season.id,
        model_id=model_id,
        slot_role="closed_family",
        provider_slug="openai/flex",
        expected_actual_model_id=f"{model_id}-20260709",
        expected_actual_provider_slug="OpenAI",
        supported_parameters_json=SUPPORTED,
        decoding_json=DECODING,
        endpoint_max_completion_tokens=4096,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
        endpoint_contract_sha256=_contract_digest(model_id),
        manifest_sha256=manifest,
        worst_case_cost_micros=100_000,
    )
    prompt = "Design a practical savoury starter with exact culinary constraints."
    battle = Battle(
        season_id=season.id,
        run_class="pilot",
        rank_eligible=False,
        # Endpoint-contract fixtures exercise provider identity only; they are
        # development evidence and therefore must not claim a ControlledRun.
        data_stratum="development",
        manifest_sha256=manifest,
        protocol_bundle_sha256=protocol_bundle_sha256,
        scheduler_version="endpoint-contract-test-v1",
        assignment_seed="4" * 64,
        track_assignment_probability="1/2",
        model_assignment_probability="1/1",
        side_assignment_probability="1/2",
        track="epicure_uplift",
        category="composition",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        client_nonce_sha256=hashlib.sha256(f"nonce:{suffix}".encode()).hexdigest(),
        requester_pseudonym=hashlib.sha256(f"rater:{suffix}".encode()).hexdigest(),
        status="queued",
        retention_until=datetime.now(UTC) + timedelta(days=30),
    )
    session.add_all([slot, battle])
    session.flush()
    arm = ResponseArm(
        battle_id=battle.id,
        side="left",
        condition="epicure_off",
        model_id=model_id,
        provider_slug="openai/flex",
        status="queued",
        prompt_sha256=battle.prompt_sha256,
        system_prompt_sha256=system_prompt_sha256("epicure_off"),
        schema_sha256=FINAL_SCHEMA_SHA256,
        tool_schema_sha256="1" * 64,
        decoding_json={
            "max_tokens": 600,
            "seed": 20260715,
            "structured_output": True,
            "temperature": "provider_fixed_unsupported",
            "top_p": "provider_fixed_unsupported",
            "max_tool_rounds": 8,
        },
        protocol_bundle_sha256=protocol_bundle_sha256,
        epicure_release_id=season.epicure_release_id,
        epicure_bundle_sha256=season.epicure_bundle_sha256,
        epicure_application_sha256=season.epicure_application_sha256,
    )
    session.add(arm)
    session.flush()
    return battle, arm, slot


def _result(slot: SeasonModel, *, model_id: str | None = None, provider: str | None = None):
    output = {
        "answer_markdown": "Roast the vegetables and balance the finish with measured acid.",
        "ingredient_mentions": ["vegetables"],
        "constraints_addressed": ["practical preparation"],
        "uncertainties": ["Seasoning needs tasting."],
    }
    generation_id = f"generation-{slot.id}"
    actual_model = model_id or slot.expected_actual_model_id
    actual_provider = provider or slot.expected_actual_provider_slug
    return GenerationResult(
        answer_markdown=output["answer_markdown"],
        output_json=output,
        actual_model_id=actual_model,
        provider_slug=actual_provider,
        generation_id=generation_id,
        generation_ids=[generation_id],
        cost_micros=321,
        cost_reconciled=True,
        decoding_json={
            "max_tokens": 600,
            "temperature": "provider_fixed_unsupported",
            "top_p": "provider_fixed_unsupported",
            "seed": 20260715,
        },
        generation_metadata=[
            {
                "generation_id": generation_id,
                "model": actual_model,
                "provider": actual_provider,
                "cost_micros": 321,
                "reconciled": True,
            }
        ],
        backend_response_schema_sha256=FINAL_SCHEMA_SHA256,
        backend_tool_schema_sha256="1" * 64,
        cost_accounting_basis="openrouter_generation_metadata",
        billing_reconciliation_status="provider_generation_metadata",
    )


def test_manifest_schema_requires_an_executable_exact_endpoint_contract() -> None:
    entry = ManifestEntry(
        model_id="openai/model",
        provider_slug="openai/flex",
        expected_actual_model_id="openai/model-20260709",
        expected_actual_provider_slug="OpenAI",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=4096,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
        slot_role="closed_family",
        worst_case_cost_micros=500_000,
    )
    assert entry.supported_parameters == sorted(SUPPORTED)
    with pytest.raises(ValidationError, match="missing required parameters"):
        ManifestEntry.model_validate(
            {
                **entry.model_dump(),
                "supported_parameters": [
                    item for item in SUPPORTED if item != "structured_outputs"
                ],
            }
        )
    with pytest.raises(ValidationError, match="completion limit"):
        ManifestEntry.model_validate(
            {
                **entry.model_dump(),
                "decoding": {"max_tokens": 5000},
            }
        )
    with pytest.raises(ValidationError):
        ManifestEntry.model_validate(
            {
                **entry.model_dump(),
                "decoding": {"max_tokens": True},
            }
        )


def test_admin_smoke_persists_the_exact_evidence_contract() -> None:
    init_database()
    model_id = "endpoint-contract/admin-smoke"
    artifact, artifact_sha256 = _smoke_evidence(
        f"{model_id}-20260709",
        "OpenAI",
    )
    request = ModelSmokeCreate(
        provider_slug="openai/flex",
        expected_actual_model_id=f"{model_id}-20260709",
        expected_actual_provider_slug="OpenAI",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=4096,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
        tools_passed=True,
        structured_output_passed=True,
        data_collection_denied=True,
        evidence_reference="artifact:paid-contract-smoke",
        evidence_artifact=artifact,
        evidence_artifact_sha256=artifact_sha256,
    )
    with session_scope() as session:
        session.add(
            CatalogModel(
                model_id=model_id,
                canonical_slug=f"{model_id}-20260709",
                name="Admin smoke model",
                family="endpoint-contract",
                status="compatible",
            )
        )
    with session_scope() as session:
        response = admin_model_smoke(model_id, request, session)
        assert response["status"] == "smoke_passed"
    with session_scope() as session:
        model = session.get(CatalogModel, model_id)
        assert model is not None
        contract = model.endpoint_json["smoke_endpoint_contract"]
        assert contract["expected_actual_model_id"] == f"{model_id}-20260709"
        assert model.endpoint_json["smoke_endpoint_contract_sha256"] == (
            endpoint_contract_sha256(**contract)
        )


def test_worker_passes_the_frozen_contract_and_accepts_only_exact_dated_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_database()
    monkeypatch.setattr(
        engine_module,
        "get_settings",
        lambda: SimpleNamespace(
            execution_mode="live",
            max_tool_rounds=8,
            build_image_digest="sha256:" + ("d" * 64),
        ),
    )
    with session_scope() as session:
        battle, arm, slot = _create_live_arm(session, "accepted")
        spec = _generation_spec(session, battle, arm)
        assert spec.model_id == slot.model_id
        assert spec.expected_actual_model_id == slot.expected_actual_model_id
        assert spec.expected_actual_provider_slug == "OpenAI"
        assert spec.supported_parameters == frozenset(SUPPORTED)
        assert spec.decoding_parameters == DECODING
        _record_result(session, battle, arm, _result(slot))
        assert arm.status == "complete"
        assert arm.actual_model_id.endswith("-20260709")

    with session_scope() as session:
        battle, arm, slot = _create_live_arm(session, "substitution")
        with pytest.raises(ProviderError, match="outside the frozen endpoint contract"):
            _record_result(session, battle, arm, _result(slot, model_id=slot.model_id))
    with session_scope() as session:
        incident = session.scalar(
            select(Incident).where(
                Incident.battle_id == battle.id,
                Incident.code == "provider_model_substitution",
            )
        )
        assert incident is not None

    with session_scope() as session:
        battle, arm, slot = _create_live_arm(session, "provider-substitution")
        with pytest.raises(ProviderError, match="provider outside"):
            _record_result(session, battle, arm, _result(slot, provider="Other Provider"))
    with session_scope() as session:
        incident = session.scalar(
            select(Incident).where(
                Incident.battle_id == battle.id,
                Incident.code == "provider_endpoint_substitution",
            )
        )
        assert incident is not None


def test_frozen_season_endpoint_contract_is_immutable() -> None:
    init_database()
    with session_scope() as session:
        _battle, _arm, slot = _create_live_arm(session, "immutable")
        slot_id = slot.id
    with pytest.raises(ValueError, match="endpoint contract is immutable"):
        with session_scope() as session:
            slot = session.get(SeasonModel, slot_id)
            assert slot is not None
            slot.expected_actual_provider_slug = "Substituted Provider"


def test_live_smoke_builds_complete_contract_before_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_smoke_module,
        "get_settings",
        lambda: SimpleNamespace(
            max_output_tokens=600,
            decoding_temperature=0.2,
            decoding_top_p=0.95,
            decoding_seed=20260715,
        ),
    )
    model = {
        "id": "openai/model",
        "canonical_slug": "openai/model-20260709",
    }
    endpoint = {
        "model_id": "openai/model",
        "provider_name": "OpenAI",
        "tag": "openai/flex",
        "quantization": "unknown",
        "context_length": 100_000,
        "max_completion_tokens": 4096,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": SUPPORTED,
    }
    contract = frozen_generation_contract(model, endpoint)
    assert contract["decoding_parameters"] == DECODING
    assert contract["expected_actual_model_id"] == "openai/model-20260709"
    assert contract["expected_actual_provider_slug"] == "OpenAI"
    assert contract["endpoint_contract_sha256"] not in {"", "unfrozen"}
    assert _model_identity_matches("openai/model-20260709", model)
    assert not _model_identity_matches("openai/model", model)
    assert _provider_identity_matches("OpenAI", endpoint)
    assert not _provider_identity_matches("OpenAI: Flex", endpoint)


def test_unknown_endpoint_completion_metadata_keeps_client_max_tokens_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_smoke_module,
        "get_settings",
        lambda: SimpleNamespace(
            max_output_tokens=600,
            decoding_temperature=0.2,
            decoding_top_p=0.95,
            decoding_seed=20260715,
        ),
    )
    model = {
        "id": "nvidia/model",
        "canonical_slug": "nvidia/model-20260715",
    }
    endpoint = {
        "model_id": "nvidia/model",
        "provider_name": "Together",
        "tag": "together",
        "quantization": "unknown",
        "context_length": 100_000,
        "max_completion_tokens": None,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "supported_parameters": SUPPORTED,
    }
    contract = frozen_generation_contract(model, endpoint)
    assert contract["decoding_parameters"]["max_tokens"] == 600
    assert contract["endpoint_contract_sha256"] == endpoint_contract_sha256(
        model_id="nvidia/model",
        provider_slug="together",
        expected_actual_model_id="nvidia/model-20260715",
        expected_actual_provider_slug="Together",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=None,
        endpoint_document_sha256=live_smoke_module.endpoint_execution_contract_sha256(
            endpoint
        ),
    )

    manifest_entry = ManifestEntry(
        model_id="nvidia/model",
        provider_slug="together",
        expected_actual_model_id="nvidia/model-20260715",
        expected_actual_provider_slug="Together",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=0,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
        slot_role="open_weight",
        worst_case_cost_micros=100_000,
    )
    assert manifest_entry.endpoint_max_completion_tokens == 0
    artifact, artifact_sha256 = _smoke_evidence(
        "nvidia/model-20260715",
        "Together",
    )
    smoke = ModelSmokeCreate(
        provider_slug="together",
        expected_actual_model_id="nvidia/model-20260715",
        expected_actual_provider_slug="Together",
        supported_parameters=SUPPORTED,
        decoding=DECODING,
        endpoint_max_completion_tokens=0,
        endpoint_document_sha256=ENDPOINT_DOCUMENT_SHA256,
        tools_passed=True,
        structured_output_passed=True,
        data_collection_denied=True,
        evidence_reference="artifact:unknown-upstream-completion-limit",
        evidence_artifact=artifact,
        evidence_artifact_sha256=artifact_sha256,
    )
    assert smoke.endpoint_max_completion_tokens == 0
