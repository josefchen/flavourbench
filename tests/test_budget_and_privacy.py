from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

import flavourbench.arena as arena
from flavourbench.database import init_database, session_scope
from flavourbench.engine import redact_expired
from flavourbench.models import (
    TOOL_CALL_REDACTION_JSON,
    TOOL_CALL_REDACTION_SENTINEL,
    Battle,
    CostEvent,
    ResponseArm,
    Season,
    SeasonModel,
    ToolCall,
    ValidatorResult,
)
from flavourbench.schemas import BedrockBillingCrosscheckCreate
from flavourbench.security import contains_identity_leak, sanitize_for_release
from flavourbench.seed import seed_database


@pytest.mark.parametrize(
    "source_artifact_uri",
    ["s3://billing/café", "s3://billing/has space"],
)
def test_bedrock_billing_uri_requires_ascii_uri_encoding(
    source_artifact_uri: str,
) -> None:
    with pytest.raises(ValidationError):
        BedrockBillingCrosscheckCreate(
            arm_ids=["arm-1"],
            source_kind="aws_cur",
            source_artifact_uri=source_artifact_uri,
            source_artifact_sha256="a" * 64,
            statement_sha256="b" * 64,
            generation_request_map_sha256="c" * 64,
            coverage_start=datetime(2026, 7, 1, tzinfo=UTC),
            coverage_end=datetime(2026, 8, 1, tzinfo=UTC),
            billed_usage_micros=100,
            credits_policy="gross_usage_before_credits_excluding_tax",
            authorization_reference_sha256="d" * 64,
        )


def test_budget_admission_closes_at_85_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arena, "get_settings", lambda: SimpleNamespace(execution_mode="live"))
    season = Season(
        slug="budget-test",
        name="Budget",
        status="pilot",
        manifest_sha256="a" * 64,
        epicure_release_id="release",
        epicure_bundle_sha256="b" * 64,
        budget_cap_micros=1_000_000,
        budget_used_micros=840_000,
    )
    model = SeasonModel(
        season_id="season",
        model_id="model",
        slot_role="test",
        worst_case_cost_micros=10_000,
    )
    with pytest.raises(HTTPException) as exc:
        arena._reserve_budget(SimpleNamespace(), season, [model])
    assert exc.value.status_code == 503


def test_retention_redacts_content_but_keeps_hashes() -> None:
    init_database()
    seed_database()
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == "season-0"))
        assert season
        battle = Battle(
            season_id=season.id,
            track="model_arena",
            category="composition",
            prompt="private prompt person@example.com",
            prompt_sha256="c" * 64,
            client_nonce_sha256="n" * 64,
            research_consent=False,
            requester_pseudonym="d" * 64,
            status="queued",
            retention_until=datetime.now(UTC) - timedelta(days=1),
        )
        session.add(battle)
        session.flush()
        arms = []
        for side in ("left", "right"):
            arm = ResponseArm(
                battle_id=battle.id,
                side=side,
                condition="epicure_on",
                model_id="flavourbench/mock-openai-flagship",
                provider_slug="mock",
                status="queued",
                prompt_sha256=battle.prompt_sha256,
                schema_sha256="s" * 64,
                tool_schema_sha256="t" * 64,
                epicure_release_id="dev",
                epicure_bundle_sha256="u" * 64,
            )
            session.add(arm)
            arms.append(arm)
        session.flush()
        completed_at = datetime.now(UTC) + timedelta(milliseconds=1)
        for arm in arms:
            arm.status = "complete"
            arm.actual_provider_slug = "mock"
            arm.actual_model_id = arm.model_id
            arm.generation_id = f"retention-{arm.side}"
            arm.provider_generation_ids_json = [arm.generation_id]
            arm.finish_reason = "stop"
            arm.answer_markdown = f"private {arm.side} answer"
            arm.output_json = {"answer_markdown": arm.answer_markdown}
            arm.cost_reconciled = True
            arm.cost_accounting_basis = "known_zero_no_provider_acceptance"
            arm.billing_reconciliation_status = "known_zero_no_provider_acceptance"
            arm.completed_at = completed_at
            session.add(
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    arm_id=arm.id,
                    kind="actual",
                    amount_micros=0,
                    provider="mock",
                    generation_id=arm.generation_id,
                    accounting_json={"basis": "known_zero_no_provider_acceptance"},
                )
            )
        session.flush()
        battle.left_arm_id = arms[0].id
        battle.right_arm_id = arms[1].id
        session.flush()
        battle.status = "complete"
        battle.completed_at = completed_at + timedelta(milliseconds=1)
        session.flush()
        session.add(
            ToolCall(
                arm_id=arms[0].id,
                round_index=0,
                call_index=0,
                tool_call_id="retention-call",
                tool_name="find_pairings",
                arguments_json={"ingredients": ["private ingredient"]},
                result_text="private result",
                structured_content_json={"pairings": ["private result"]},
                result_sha256=hashlib.sha256(b"private result").hexdigest(),
            )
        )
        session.add(
            ValidatorResult(
                arm_id=arm.id,
                validator_name="identity_blinding",
                validator_version="retention-v1",
                status="pass",
                score_milli=1000,
                detail_json={"matched_identity": "private"},
            )
        )
    with session_scope() as session:
        call = session.scalar(select(ToolCall).where(ToolCall.tool_call_id == "retention-call"))
        assert call and call.arguments_sha256 and call.structured_content_sha256
        original_digests = (
            call.arguments_sha256,
            call.result_sha256,
            call.structured_content_sha256,
        )
        validation = session.scalar(
            select(ValidatorResult).where(
                ValidatorResult.validator_version == "retention-v1"
            )
        )
        assert validation and validation.detail_sha256
        original_validation_digest = validation.detail_sha256
    with session_scope() as session:
        assert redact_expired(session) == 1
    with session_scope() as session:
        stored = session.scalar(select(Battle).where(Battle.prompt_sha256 == "c" * 64))
        assert stored and stored.prompt is None and stored.prompt_redacted
        assert stored.prompt_sha256 == "c" * 64
        call = session.scalar(select(ToolCall).where(ToolCall.tool_call_id == "retention-call"))
        assert call
        assert call.arguments_json == TOOL_CALL_REDACTION_JSON
        assert call.result_text == TOOL_CALL_REDACTION_SENTINEL
        assert call.structured_content_json == TOOL_CALL_REDACTION_JSON
        assert (
            call.arguments_sha256,
            call.result_sha256,
            call.structured_content_sha256,
        ) == original_digests
        validation = session.scalar(
            select(ValidatorResult).where(
                ValidatorResult.validator_version == "retention-v1"
            )
        )
        assert validation
        assert validation.detail_json == TOOL_CALL_REDACTION_JSON
        assert validation.detail_sha256 == original_validation_digest


def test_release_sanitizer_and_identity_filter() -> None:
    assert "[EMAIL REDACTED]" in sanitize_for_release("write to person@example.com")
    assert contains_identity_leak("I am Claude and I suggest roasting it.")
    assert not contains_identity_leak("Roast until the edges are deeply browned.")
