from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select

import flavourbench.arena as arena_module
import flavourbench.budget_integrity as budget_integrity_module
from flavourbench.budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from flavourbench.database import init_database, session_scope
from flavourbench.engine import reconcile_battle_cost
from flavourbench.main import (
    _verified_official_epicure_release,
    admin_register_epicure_release,
    admin_season_officialize,
)
from flavourbench.models import (
    Battle,
    CatalogModel,
    CostEvent,
    EpicureRelease,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
    ResponseArm,
    Season,
    SeasonModel,
    SeasonProviderBudget,
)
from flavourbench.schemas import EpicureReleaseRegisterCreate, SeasonOfficializeCreate

BUDGET_SIGNING_SECRET = "test-provider-budget-signing-secret-0001"


def test_officialization_requires_the_production_postgresql_service() -> None:
    init_database()
    request = SeasonOfficializeCreate(
        gate_a_decision_reference="gate-a-test",
        privacy_review_reference="privacy-test",
        security_review_reference="security-test",
        expert_access_reference="expert-test",
        task_registry_manifest_sha256="1" * 64,
        analysis_plan_sha256="2" * 64,
        statistical_approval_reference="statistics-test",
        reproducibility_approval_reference="reproducibility-test",
        data_steward_approval_reference="steward-test",
    )
    with session_scope() as session:
        with pytest.raises(HTTPException, match="production PostgreSQL"):
            admin_season_officialize("season-0", request, session)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _release_request(release_id: str, *, public_match: bool = True):
    values = {
        "release_id": release_id,
        "bundle_sha256": "b" * 64,
        "application_sha256": "a" * 64,
        "public_release_uri": f"https://releases.epicure.example/{release_id}",
        "release_artifact_sha256": "c" * 64,
        "rights_clearance_sha256": "d" * 64,
        "verification_report_sha256": "e" * 64,
        "public_release_match": public_match,
        "redistribution_rights_cleared": True,
        "reproducibility_verified": True,
    }
    manifest = {
        "schema_version": "flavourbench-epicure-release-lineage-v1",
        **values,
    }
    return EpicureReleaseRegisterCreate(
        **values,
        lineage_manifest_sha256=_sha(manifest),
    )


def test_official_lineage_registry_is_content_addressed_and_fail_closed() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    unmatched_id = f"epicure-unmatched-{suffix}"
    eligible_id = f"epicure-core-{suffix}"

    invalid = _release_request(eligible_id)
    invalid.lineage_manifest_sha256 = "0" * 64
    with session_scope() as session:
        with pytest.raises(HTTPException, match="manifest hash"):
            admin_register_epicure_release(invalid, session)

    with session_scope() as session:
        result = admin_register_epicure_release(
            _release_request(unmatched_id),
            session,
        )
        assert result["officialEligible"] is False
    with session_scope() as session:
        season = Season(
            slug=f"lineage-unmatched-{suffix}",
            name="Unmatched lineage fixture",
            status="pilot",
            epicure_release_id=unmatched_id,
            epicure_bundle_sha256="b" * 64,
            epicure_application_sha256="a" * 64,
        )
        session.add(season)
        session.flush()
        with pytest.raises(HTTPException, match="eligible public Epicure release"):
            _verified_official_epicure_release(session, season)

    with session_scope() as session:
        result = admin_register_epicure_release(
            _release_request(eligible_id),
            session,
        )
        assert result["officialEligible"] is True
    with session_scope() as session:
        season = Season(
            slug=f"lineage-eligible-{suffix}",
            name="Eligible lineage fixture",
            status="pilot",
            epicure_release_id=eligible_id,
            epicure_bundle_sha256="b" * 64,
            epicure_application_sha256="a" * 64,
        )
        session.add(season)
        session.flush()
        assert _verified_official_epicure_release(session, season).release_id == eligible_id

    with pytest.raises(ValueError, match="append-only"):
        with session_scope() as session:
            release = session.get(EpicureRelease, eligible_id)
            assert release is not None
            release.public_release_uri = "https://releases.epicure.example/rewritten"


def _provider_budget(
    season: Season,
    backend: str,
    cap: int,
    suffix: str,
    account_authorization_envelope_sha256: str,
) -> SeasonProviderBudget:
    valid_until = datetime.now(UTC) + timedelta(days=90)
    envelope = {
        "schema_version": "flavourbench-provider-budget-authorization-v1",
        "season_slug": season.slug,
        "execution_backend": backend,
        "currency": "USD",
        "budget_cap_micros": cap,
        "account_scope_sha256": provider_account_scope_sha256(backend),
        "authorization_reference_sha256": hashlib.sha256(
            f"authorization:{backend}:{suffix}".encode()
        ).hexdigest(),
        "account_authorization_envelope_sha256": (account_authorization_envelope_sha256),
        "valid_until": valid_until.isoformat(),
    }
    return SeasonProviderBudget(
        season_id=season.id,
        execution_backend=backend,
        currency="USD",
        budget_cap_micros=cap,
        account_scope_sha256=envelope["account_scope_sha256"],
        authorization_reference_sha256=envelope["authorization_reference_sha256"],
        account_authorization_envelope_sha256=(account_authorization_envelope_sha256),
        authorization_envelope_json=envelope,
        authorization_envelope_sha256=_sha(envelope),
        valid_until=valid_until,
    )


def _account_budget(
    backend: str,
    suffix: str,
) -> tuple[ProviderAccountBudget, ProviderAccountAuthorization]:
    valid_until = datetime.now(UTC) + timedelta(days=90)
    scope = provider_account_scope_sha256(backend)
    opening = {
        "schema_version": "flavourbench-provider-opening-balance-v1",
        "execution_backend": backend,
        "sources": [],
        "governed_used_micros": 0,
        "governed_reserved_micros": 0,
    }
    binding = {
        "binding_kind": f"{backend}_test_binding_v1",
        "credential_scope_sha256": scope,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    ledger_id = str(uuid.uuid4())
    exposure_attestation_sha256 = _sha(opening)
    envelope = {
        "schema_version": "flavourbench-provider-account-authorization-v3",
        "provider_account_budget_id": ledger_id,
        "execution_backend": backend,
        "currency": "USD",
        "budget_cap_micros": provider_account_hard_cap_micros(backend),
        "account_scope_sha256": scope,
        "authorization_reference_sha256": hashlib.sha256(
            f"account-authorization:{backend}:{suffix}".encode()
        ).hexdigest(),
        "ledger_opening_balance_sha256": _sha(opening),
        "exposure_attestation_sha256": exposure_attestation_sha256,
        "cumulative_used_micros": 0,
        "cumulative_reserved_micros": 0,
        "credential_binding_sha256": _sha(binding),
        "supersedes_authorization_envelope_sha256": None,
        "valid_until": valid_until.isoformat(),
    }
    envelope_sha256 = _sha(envelope)
    authorization_hmac = hmac.new(
        BUDGET_SIGNING_SECRET.encode(),
        envelope_sha256.encode(),
        hashlib.sha256,
    ).hexdigest()
    ledger = ProviderAccountBudget(
        id=ledger_id,
        execution_backend=backend,
        currency="USD",
        status="active",
        budget_cap_micros=provider_account_hard_cap_micros(backend),
        budget_used_micros=0,
        budget_reserved_micros=0,
        opening_used_micros=0,
        opening_reserved_micros=0,
        account_scope_sha256=scope,
        authorization_reference_sha256=envelope["authorization_reference_sha256"],
        opening_balance_json=opening,
        opening_balance_sha256=_sha(opening),
        credential_binding_json=binding,
        credential_binding_sha256=_sha(binding),
        authorization_envelope_json=envelope,
        authorization_envelope_sha256=envelope_sha256,
        authorization_hmac_sha256=authorization_hmac,
        valid_until=valid_until,
    )
    authorization = ProviderAccountAuthorization(
        provider_account_budget_id=ledger_id,
        execution_backend=backend,
        account_scope_sha256=scope,
        status="active",
        authorization_reference_sha256=envelope["authorization_reference_sha256"],
        exposure_attestation_json=opening,
        exposure_attestation_sha256=exposure_attestation_sha256,
        authorized_used_micros=0,
        authorized_reserved_micros=0,
        credential_binding_json=binding,
        credential_binding_sha256=_sha(binding),
        authorization_envelope_json=envelope,
        authorization_envelope_sha256=envelope_sha256,
        authorization_hmac_sha256=authorization_hmac,
        valid_until=valid_until,
    )
    return ledger, authorization


def test_provider_caps_reserve_and_reconcile_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_database()
    suffix = uuid.uuid4().hex
    monkeypatch.setattr(
        arena_module,
        "get_settings",
        lambda: SimpleNamespace(
            execution_mode="live",
            budget_authorization_signing_secret=BUDGET_SIGNING_SECRET,
        ),
    )
    monkeypatch.setattr(
        budget_integrity_module,
        "get_settings",
        lambda: SimpleNamespace(
            budget_authorization_signing_secret=BUDGET_SIGNING_SECRET,
        ),
    )
    with session_scope() as session:
        season = Season(
            slug=f"provider-budget-{suffix}",
            name="Provider budget fixture",
            status="pilot",
            manifest_sha256="f" * 64,
            tool_registry_sha256="t" * 64,
            epicure_release_id="epicure-core-provider-budget",
            epicure_bundle_sha256="b" * 64,
            epicure_application_sha256="a" * 64,
            budget_cap_micros=10_000_000,
        )
        session.add(season)
        session.flush()
        models: list[SeasonModel] = []
        for backend, amount in (("openrouter", 100_000), ("bedrock", 200_000)):
            model_id = f"provider-budget/{backend}-{suffix}"
            session.add(
                CatalogModel(
                    model_id=model_id,
                    canonical_slug=model_id,
                    name=model_id,
                    family="provider-budget",
                )
            )
            slot = SeasonModel(
                season_id=season.id,
                model_id=model_id,
                slot_role="test",
                execution_backend=backend,
                provider_slug=f"{backend}/fixed",
                worst_case_cost_micros=amount,
            )
            session.add(slot)
            models.append(slot)
        openrouter_account, openrouter_authorization = _account_budget("openrouter", suffix)
        bedrock_account, bedrock_authorization = _account_budget("bedrock", suffix)
        session.add_all([openrouter_account, bedrock_account])
        session.flush()
        session.add_all([openrouter_authorization, bedrock_authorization])
        session.flush()
        openrouter_budget = _provider_budget(
            season,
            "openrouter",
            150_000,
            suffix,
            openrouter_authorization.authorization_envelope_sha256,
        )
        bedrock_budget = _provider_budget(
            season,
            "bedrock",
            5_000_000,
            suffix,
            bedrock_authorization.authorization_envelope_sha256,
        )
        session.add_all([openrouter_budget, bedrock_budget])
        session.flush()

        assert arena_module._reserve_budget(session, season, models) == 300_000
        assert season.budget_reserved_micros == 300_000
        assert openrouter_budget.budget_reserved_micros == 100_000
        assert bedrock_budget.budget_reserved_micros == 200_000
        assert openrouter_account.budget_reserved_micros == 100_000
        assert bedrock_account.budget_reserved_micros == 200_000

        battle = Battle(
            season_id=season.id,
            track="model_arena",
            category="composition",
            prompt="Create a composed dish.",
            prompt_sha256=hashlib.sha256(b"provider-budget-prompt").hexdigest(),
            client_nonce_sha256=hashlib.sha256(
                f"provider-budget-nonce:{suffix}".encode()
            ).hexdigest(),
            requester_pseudonym=hashlib.sha256(
                f"provider-budget-rater:{suffix}".encode()
            ).hexdigest(),
            status="queued",
            reserved_cost_micros=300_000,
            provider_reservations_json={
                "openrouter": 100_000,
                "bedrock": 200_000,
            },
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        session.add(battle)
        session.flush()
        session.add_all(
            [
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="reserve",
                    amount_micros=300_000,
                    provider="governor",
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reserve",
                    amount_micros=100_000,
                    provider="openrouter",
                    accounting_json={"budget_scope": "provider"},
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_reserve",
                    amount_micros=100_000,
                    provider="openrouter",
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": openrouter_account.account_scope_sha256,
                    },
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reserve",
                    amount_micros=200_000,
                    provider="bedrock",
                    accounting_json={"budget_scope": "provider"},
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_reserve",
                    amount_micros=200_000,
                    provider="bedrock",
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": bedrock_account.account_scope_sha256,
                    },
                ),
            ]
        )
        session.flush()
        arms = []
        for side, slot, cost in zip(
            ("left", "right"),
            models,
            (50_000, 100_000),
            strict=True,
        ):
            arm = ResponseArm(
                battle_id=battle.id,
                side=side,
                condition="epicure_on",
                model_id=slot.model_id,
                execution_backend=slot.execution_backend,
                provider_slug=slot.provider_slug,
                status="queued",
                prompt_sha256=battle.prompt_sha256,
                schema_sha256="s" * 64,
                tool_schema_sha256="t" * 64,
                epicure_release_id=season.epicure_release_id,
                epicure_bundle_sha256=season.epicure_bundle_sha256,
            )
            session.add(arm)
            arms.append((arm, cost))
        session.flush()
        arm_completed_at = datetime.now(UTC) + timedelta(milliseconds=10)
        for arm, cost in arms:
            generation_id = f"provider-budget-{arm.side}-{suffix}"
            arm.status = "complete"
            arm.actual_provider_slug = arm.provider_slug
            arm.actual_model_id = arm.model_id
            arm.generation_id = generation_id
            arm.provider_generation_ids_json = [generation_id]
            arm.finish_reason = "stop"
            arm.answer_markdown = f"Reconciled {arm.side} answer"
            arm.output_json = {"answer_markdown": arm.answer_markdown}
            arm.cost_micros = cost
            arm.cost_reconciled = True
            arm.cost_accounting_basis = "provider_budget_test_reconciliation"
            arm.billing_reconciliation_status = "complete"
            arm.completed_at = arm_completed_at
            session.add(
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    arm_id=arm.id,
                    kind="actual",
                    amount_micros=cost,
                    provider=arm.provider_slug,
                    generation_id=generation_id,
                    accounting_json={
                        "generation_ids": [generation_id],
                        "reconciled": True,
                        "cost_accounting_basis": (
                            "provider_budget_test_reconciliation"
                        ),
                        "billing_reconciliation_status": "complete",
                    },
                )
            )
        session.flush()
        battle.left_arm_id = arms[0][0].id
        battle.right_arm_id = arms[1][0].id
        session.flush()
        battle.status = "complete"
        battle.completed_at = arm_completed_at + timedelta(milliseconds=1)
        session.flush()
        reconcile_battle_cost(session, battle)
        assert season.budget_reserved_micros == 0
        assert season.budget_used_micros == 150_000
        assert openrouter_budget.budget_reserved_micros == 0
        assert openrouter_budget.budget_used_micros == 50_000
        assert bedrock_budget.budget_reserved_micros == 0
        assert bedrock_budget.budget_used_micros == 100_000
        assert openrouter_account.budget_reserved_micros == 0
        assert openrouter_account.budget_used_micros == 50_000
        assert bedrock_account.budget_reserved_micros == 0
        assert bedrock_account.budget_used_micros == 100_000

        with pytest.raises(HTTPException, match="openrouter provider budget admission"):
            arena_module._reserve_budget(session, season, [models[0]])
        assert openrouter_budget.budget_reserved_micros == 0
        assert season.budget_reserved_micros == 0

    with session_scope() as session:
        budgets = session.scalars(
            select(SeasonProviderBudget).where(SeasonProviderBudget.season_id == season.id)
        ).all()
        assert {budget.execution_backend for budget in budgets} == {
            "openrouter",
            "bedrock",
        }
