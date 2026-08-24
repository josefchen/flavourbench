from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update

from flavourbench.account_authority import account_authorization_chain_valid
from flavourbench.budget_integrity import (
    BudgetIntegrityError,
    assert_budget_integrity,
    decrement_reservation,
)
from flavourbench.config import get_settings
from flavourbench.database import init_database, session_scope
from flavourbench.models import (
    Battle,
    BedrockBillingCrosscheck,
    BedrockBillingCrosscheckArm,
    CatalogModel,
    ControlledRun,
    CostEvent,
    GenerationAttempt,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
    ResponseArm,
    RunEvent,
    Season,
    SeasonProviderBudget,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _authorization_material(
    account: ProviderAccountBudget,
    *,
    suffix: str,
    used_micros: int,
    reserved_micros: int,
    valid_until: datetime,
    supersedes_envelope_sha256: str | None = None,
) -> dict[str, object]:
    reference_sha256 = _sha(f"account-epoch-reference:{suffix}")
    exposure = {
        "schema_version": "budget-integrity-exposure-test-v1",
        "used_micros": used_micros,
        "reserved_micros": reserved_micros,
    }
    exposure_sha256 = _canonical_sha(exposure)
    binding = {
        "schema_version": "budget-integrity-binding-test-v1",
        "credential_scope_sha256": account.account_scope_sha256,
    }
    binding_sha256 = _canonical_sha(binding)
    envelope = {
        "schema_version": "flavourbench-provider-account-authorization-v3",
        "provider_account_budget_id": account.id,
        "execution_backend": account.execution_backend,
        "currency": account.currency,
        "budget_cap_micros": account.budget_cap_micros,
        "account_scope_sha256": account.account_scope_sha256,
        "authorization_reference_sha256": reference_sha256,
        "ledger_opening_balance_sha256": account.opening_balance_sha256,
        "exposure_attestation_sha256": exposure_sha256,
        "cumulative_used_micros": used_micros,
        "cumulative_reserved_micros": reserved_micros,
        "credential_binding_sha256": binding_sha256,
        "supersedes_authorization_envelope_sha256": supersedes_envelope_sha256,
        "valid_until": valid_until.isoformat(),
    }
    envelope_sha256 = _canonical_sha(envelope)
    signature = hmac.new(
        get_settings().budget_authorization_signing_secret.encode(),
        envelope_sha256.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        "authorization_reference_sha256": reference_sha256,
        "exposure_attestation_json": exposure,
        "exposure_attestation_sha256": exposure_sha256,
        "credential_binding_json": binding,
        "credential_binding_sha256": binding_sha256,
        "authorization_envelope_json": envelope,
        "authorization_envelope_sha256": envelope_sha256,
        "authorization_hmac_sha256": signature,
    }


def _seed_reservations(
    session,
    *,
    suffix: str,
    amounts: tuple[int, ...] = (100, 100),
    opening_reserved: int = 25,
    controlled: bool = False,
    backend: str = "openrouter",
) -> tuple[Season, SeasonProviderBudget, ProviderAccountBudget, list[Battle], ControlledRun | None]:
    scope = _sha(f"account-scope:{suffix}")
    total = sum(amounts)
    season = Season(
        slug=f"budget-ledger-{suffix}",
        name="Budget ledger integrity test",
        status="active",
        official=True,
        epicure_release_id="budget-ledger-test-release",
        budget_cap_micros=10_000,
        budget_reserved_micros=total,
    )
    session.add(season)
    session.flush()
    model_id = f"budget-ledger/model-{suffix}"
    session.add(
        CatalogModel(
            model_id=model_id,
            canonical_slug=model_id,
            name="Budget ledger model",
            family="budget-ledger-test",
        )
    )
    session.flush()
    run = None
    if controlled:
        run = ControlledRun(
            season_id=season.id,
            organization_reference_sha256=_sha(f"organization:{suffix}"),
            access_token_sha256=_sha(f"access:{suffix}"),
            status="active",
            protocol_version="budget-ledger-test-v1",
            rater_plan_sha256=_sha(f"raters:{suffix}"),
            analysis_plan_sha256=_sha(f"analysis:{suffix}"),
            budget_cap_micros=10_000,
            budget_reserved_micros=total,
            run_card_json={"test": True},
            run_card_sha256=_sha(f"run-card:{suffix}"),
            run_card_signature=_sha(f"signature:{suffix}"),
        )
        session.add(run)
        session.flush()
    valid_until = datetime.now(UTC) + timedelta(days=30)
    opening_balance = {"reserved": opening_reserved, "test_scope": suffix}
    opening_balance_sha256 = _canonical_sha(opening_balance)
    placeholder_binding = {"credential_scope_sha256": scope}
    placeholder_envelope = {"pending": True, "suffix": suffix}
    account = ProviderAccountBudget(
        id=str(uuid.uuid4()),
        execution_backend=backend,
        currency="USD",
        status="active",
        budget_cap_micros=10_000,
        budget_reserved_micros=opening_reserved + total,
        opening_used_micros=0,
        opening_reserved_micros=opening_reserved,
        account_scope_sha256=scope,
        authorization_reference_sha256=_sha(f"account-authorization:{suffix}"),
        opening_balance_json=opening_balance,
        opening_balance_sha256=opening_balance_sha256,
        credential_binding_json=placeholder_binding,
        credential_binding_sha256=_canonical_sha(placeholder_binding),
        authorization_envelope_json=placeholder_envelope,
        authorization_envelope_sha256=_canonical_sha(placeholder_envelope),
        authorization_hmac_sha256="0" * 64,
        valid_until=valid_until,
    )
    material = _authorization_material(
        account,
        suffix=suffix,
        used_micros=0,
        reserved_micros=opening_reserved,
        valid_until=valid_until,
    )
    account.authorization_reference_sha256 = str(
        material["authorization_reference_sha256"]
    )
    account.credential_binding_json = dict(material["credential_binding_json"])
    account.credential_binding_sha256 = str(material["credential_binding_sha256"])
    account.authorization_envelope_json = dict(material["authorization_envelope_json"])
    account.authorization_envelope_sha256 = str(
        material["authorization_envelope_sha256"]
    )
    account.authorization_hmac_sha256 = str(material["authorization_hmac_sha256"])
    provider_envelope = {"test": True, "suffix": suffix}
    provider = SeasonProviderBudget(
        season_id=season.id,
        execution_backend=backend,
        currency="USD",
        budget_cap_micros=10_000,
        budget_reserved_micros=total,
        account_scope_sha256=scope,
        authorization_reference_sha256=_sha(f"provider-authorization:{suffix}"),
        account_authorization_envelope_sha256=str(
            material["authorization_envelope_sha256"]
        ),
        authorization_envelope_json=provider_envelope,
        authorization_envelope_sha256=_canonical_sha(provider_envelope),
        valid_until=valid_until,
    )
    session.add_all([account, provider])
    session.flush()
    authorization = ProviderAccountAuthorization(
        provider_account_budget_id=account.id,
        execution_backend=backend,
        account_scope_sha256=scope,
        status="active",
        authorization_reference_sha256=str(material["authorization_reference_sha256"]),
        exposure_attestation_json=dict(material["exposure_attestation_json"]),
        exposure_attestation_sha256=str(material["exposure_attestation_sha256"]),
        authorized_used_micros=0,
        authorized_reserved_micros=opening_reserved,
        credential_binding_json=dict(material["credential_binding_json"]),
        credential_binding_sha256=str(material["credential_binding_sha256"]),
        authorization_envelope_json=dict(material["authorization_envelope_json"]),
        authorization_envelope_sha256=str(material["authorization_envelope_sha256"]),
        authorization_hmac_sha256=str(material["authorization_hmac_sha256"]),
        valid_until=valid_until,
    )
    session.add(authorization)
    session.flush()

    battles: list[Battle] = []
    for index, amount in enumerate(amounts):
        battle = Battle(
            season_id=season.id,
            controlled_run_id=run.id if run is not None else None,
            run_class="official",
            rank_eligible=True,
            data_stratum="controlled" if run is not None else "public_freeform",
            track="model_arena",
            category="composition",
            prompt=f"Reservation integrity prompt {suffix} {index}",
            prompt_sha256=_sha(f"prompt:{suffix}:{index}"),
            client_nonce_sha256=_sha(f"nonce:{suffix}:{index}"),
            requester_pseudonym=_sha(f"rater:{suffix}:{index}"),
            status="queued",
            reserved_cost_micros=amount,
            provider_reservations_json={backend: amount},
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        session.add(battle)
        session.flush()
        session.add_all(
            [
                ResponseArm(
                    battle_id=battle.id,
                    side=side,
                    condition="epicure_on",
                    model_id=model_id,
                    execution_backend=backend,
                    provider_slug=f"{backend}-test-route",
                    status="queued",
                    prompt_sha256=battle.prompt_sha256,
                    schema_sha256=_sha(f"schema:{suffix}"),
                    tool_schema_sha256=_sha(f"tools:{suffix}"),
                    epicure_release_id=season.epicure_release_id,
                    epicure_bundle_sha256=_sha(f"bundle:{suffix}"),
                )
                for side in ("left", "right")
            ]
        )
        session.add_all(
            [
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="reserve",
                    amount_micros=amount,
                    provider="governor",
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reserve",
                    amount_micros=amount,
                    provider=backend,
                    accounting_json={"budget_scope": "provider"},
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_reserve",
                    amount_micros=amount,
                    provider=backend,
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": scope,
                    },
                ),
            ]
        )
        battles.append(battle)
    session.flush()
    return season, provider, account, battles, run


def _settle(
    session,
    season: Season,
    provider: SeasonProviderBudget,
    account: ProviderAccountBudget,
    battles: list[Battle],
    run: ControlledRun | None,
    *,
    reconcile_offset: int = 0,
    account_used_offset: int = 0,
    include_receipts: bool = True,
) -> None:
    total_actual = sum(battle.reserved_cost_micros // 2 + reconcile_offset for battle in battles)
    season.budget_reserved_micros = 0
    season.budget_used_micros = total_actual
    provider.budget_reserved_micros = 0
    provider.budget_used_micros = total_actual
    account.budget_reserved_micros = account.opening_reserved_micros
    account.budget_used_micros = account_used_offset + total_actual
    if run is not None:
        run.budget_reserved_micros = 0
        run.budget_used_micros = total_actual
    for battle in battles:
        amount = battle.reserved_cost_micros
        governed_actual = amount // 2 + reconcile_offset
        arms = list(
            session.scalars(
                select(ResponseArm)
                .where(ResponseArm.battle_id == battle.id)
                .order_by(ResponseArm.side)
            ).all()
        )
        assert len(arms) == 2
        arm_costs = (amount // 4, amount // 2 - amount // 4)
        for arm_index, (arm, cost) in enumerate(zip(arms, arm_costs, strict=True)):
            generation_id = f"receipt-{battle.id}-{arm_index}"
            arm.cost_micros = cost
            arm.cost_reconciled = True
            arm.cost_accounting_basis = "test_endpoint_receipt"
            arm.billing_reconciliation_status = "complete"
            arm.status = "failed"
            arm.actual_provider_slug = arm.provider_slug
            arm.actual_model_id = arm.model_id
            arm.generation_id = generation_id
            arm.provider_generation_ids_json = [generation_id]
            arm.completed_at = datetime.now(UTC)
            if include_receipts:
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
                            "cost_accounting_basis": "test_endpoint_receipt",
                            "billing_reconciliation_status": "complete",
                        },
                    )
                )
        battle.reserved_cost_micros = 0
        session.add_all(
            [
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="release",
                    amount_micros=-amount,
                    provider="governor",
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="reconcile",
                    amount_micros=governed_actual,
                    provider="governor",
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_release",
                    amount_micros=-amount,
                    provider=provider.execution_backend,
                    accounting_json={"budget_scope": "provider"},
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reconcile",
                    amount_micros=governed_actual,
                    provider=provider.execution_backend,
                    accounting_json={"budget_scope": "provider"},
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_release",
                    amount_micros=-amount,
                    provider=provider.execution_backend,
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": account.account_scope_sha256,
                    },
                ),
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_reconcile",
                    amount_micros=governed_actual,
                    provider=provider.execution_backend,
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": account.account_scope_sha256,
                    },
                ),
            ]
        )
    session.flush()


def _add_valid_bedrock_crosscheck(
    session,
    *,
    season: Season,
    provider: SeasonProviderBudget,
    account: ProviderAccountBudget,
    battles: list[Battle],
    suffix: str,
    ledger_delta_micros: int = 10,
    event_scope_sha256: str | None = None,
    supersedes_crosscheck: BedrockBillingCrosscheck | None = None,
) -> tuple[BedrockBillingCrosscheck, CostEvent]:
    arms = list(
        session.scalars(
            select(ResponseArm)
            .where(ResponseArm.battle_id.in_([battle.id for battle in battles]))
            .order_by(ResponseArm.id)
        ).all()
    )
    authorization = session.scalar(
        select(ProviderAccountAuthorization).where(
            ProviderAccountAuthorization.provider_account_budget_id == account.id,
            ProviderAccountAuthorization.status == "active",
        )
    )
    assert authorization is not None
    generation_map: list[dict[str, object]] = []
    for index, arm in enumerate(arms):
        generation_ids = sorted(set(arm.provider_generation_ids_json or []))
        assert generation_ids
        generation_set_sha256 = _canonical_sha({"generation_ids": generation_ids})
        generation_map.append(
            {
                "arm_id": arm.id,
                "generation_ids": generation_ids,
                "account_authorization_envelope_sha256": (
                    authorization.authorization_envelope_sha256
                ),
                "generation_set_sha256": generation_set_sha256,
            }
        )
        session.add(
            GenerationAttempt(
                attempt_id=str(uuid.uuid4()),
                arm_id=arm.id,
                request_key_sha256=_sha(f"request-key:{suffix}:{index}"),
                phase="generation",
                attempt_index=1,
                event_type="request_started",
                payload_sha256=_sha(f"request-payload:{suffix}:{index}"),
                metadata_json={
                    "verified_provider_account_authorization_envelope_sha256": (
                        authorization.authorization_envelope_sha256
                    )
                },
            )
        )
    arm_ids = [arm.id for arm in arms]
    rate_card_estimated_micros = sum(arm.cost_micros for arm in arms)
    billing_difference_micros = ledger_delta_micros + (
        supersedes_crosscheck.billing_difference_micros
        if supersedes_crosscheck is not None
        else 0
    )
    billed_usage_micros = rate_card_estimated_micros + billing_difference_micros
    tolerance_micros = max(10_000, (rate_card_estimated_micros + 49) // 50)
    status = (
        "accepted"
        if abs(billing_difference_micros) <= tolerance_micros
        else "discrepant"
    )
    governed_delta = max(0, ledger_delta_micros)
    coverage_start = datetime.now(UTC) - timedelta(days=1)
    coverage_end = datetime.now(UTC)
    arm_set_sha256 = _canonical_sha({"arm_ids": arm_ids})
    generation_request_map_sha256 = _canonical_sha({"arms": generation_map})
    evidence = {
        "schema_version": "flavourbench-bedrock-billing-crosscheck-v1",
        "season_slug": season.slug,
        "account_scope_sha256": account.account_scope_sha256,
        "source_kind": "aws_cur",
        "source_artifact_uri": f"s3://billing-test/{suffix}",
        "source_artifact_sha256": _sha(f"source:{suffix}"),
        "statement_sha256": _sha(f"statement:{suffix}"),
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "arm_set_sha256": arm_set_sha256,
        "generation_request_map_sha256": generation_request_map_sha256,
        "rate_card_estimated_micros": rate_card_estimated_micros,
        "billed_usage_micros": billed_usage_micros,
        "billing_difference_micros": billing_difference_micros,
        "crosscheck_status": status,
        "supersedes_crosscheck_id": (
            supersedes_crosscheck.id if supersedes_crosscheck is not None else None
        ),
        "tolerance_micros": tolerance_micros,
        "credits_policy": "gross_usage_before_credits_excluding_tax",
        "authorization_reference_sha256": (
            authorization.authorization_reference_sha256
        ),
        "ledger_delta_micros": ledger_delta_micros,
        "governed_budget_delta_micros": governed_delta,
    }
    evidence_sha256 = _canonical_sha(evidence)
    crosscheck = BedrockBillingCrosscheck(
        season_id=season.id,
        provider_account_budget_id=account.id,
        status=status,
        supersedes_crosscheck_id=(
            supersedes_crosscheck.id if supersedes_crosscheck is not None else None
        ),
        source_kind="aws_cur",
        source_artifact_uri=f"s3://billing-test/{suffix}",
        source_artifact_sha256=evidence["source_artifact_sha256"],
        statement_sha256=evidence["statement_sha256"],
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        arm_set_sha256=arm_set_sha256,
        generation_request_map_sha256=generation_request_map_sha256,
        rate_card_estimated_micros=rate_card_estimated_micros,
        billed_usage_micros=billed_usage_micros,
        billing_difference_micros=billing_difference_micros,
        ledger_delta_micros=ledger_delta_micros,
        tolerance_micros=tolerance_micros,
        credits_policy="gross_usage_before_credits_excluding_tax",
        authorization_reference_sha256=authorization.authorization_reference_sha256,
        evidence_json=evidence,
        evidence_sha256=evidence_sha256,
    )
    session.add(crosscheck)
    session.flush()
    session.add_all(
        [
            BedrockBillingCrosscheckArm(
                crosscheck_id=crosscheck.id,
                arm_id=item["arm_id"],
                generation_set_sha256=item["generation_set_sha256"],
            )
            for item in generation_map
        ]
    )
    event = CostEvent(
        season_id=season.id,
        kind="bedrock_billing_adjustment",
        amount_micros=ledger_delta_micros,
        provider="bedrock",
        accounting_json={
            "crosscheck_id": crosscheck.id,
            "evidence_sha256": evidence_sha256,
            "arm_set_sha256": arm_set_sha256,
            "account_scope_sha256": (
                event_scope_sha256 or account.account_scope_sha256
            ),
            "governed_budget_delta_micros": governed_delta,
        },
    )
    session.add(event)
    season.budget_used_micros += governed_delta
    provider.budget_used_micros += governed_delta
    account.budget_used_micros += governed_delta
    session.flush()
    return crosscheck, event


def _seed_shared_account_season(
    session,
    *,
    account: ProviderAccountBudget,
    account_root_sha256: str,
    model_id: str,
    suffix: str,
    amount: int = 100,
) -> tuple[Season, SeasonProviderBudget, list[Battle]]:
    season = Season(
        slug=f"budget-ledger-shared-{suffix}",
        name="Shared provider account integrity test",
        status="active",
        official=True,
        epicure_release_id="budget-ledger-test-release",
        budget_cap_micros=10_000,
        budget_reserved_micros=amount,
    )
    session.add(season)
    session.flush()
    provider_envelope = {"test": True, "shared_suffix": suffix}
    provider = SeasonProviderBudget(
        season_id=season.id,
        execution_backend="bedrock",
        currency="USD",
        budget_cap_micros=10_000,
        budget_reserved_micros=amount,
        account_scope_sha256=account.account_scope_sha256,
        authorization_reference_sha256=_sha(f"shared-provider-auth:{suffix}"),
        account_authorization_envelope_sha256=account_root_sha256,
        authorization_envelope_json=provider_envelope,
        authorization_envelope_sha256=_canonical_sha(provider_envelope),
        valid_until=datetime.now(UTC) + timedelta(days=30),
    )
    battle = Battle(
        season_id=season.id,
        run_class="official",
        rank_eligible=True,
        data_stratum="public_freeform",
        track="model_arena",
        category="composition",
        prompt=f"Shared account prompt {suffix}",
        prompt_sha256=_sha(f"shared-prompt:{suffix}"),
        client_nonce_sha256=_sha(f"shared-nonce:{suffix}"),
        requester_pseudonym=_sha(f"shared-rater:{suffix}"),
        status="queued",
        reserved_cost_micros=amount,
        provider_reservations_json={"bedrock": amount},
        retention_until=datetime.now(UTC) + timedelta(days=30),
    )
    session.add_all([provider, battle])
    session.flush()
    session.add_all(
        [
            ResponseArm(
                battle_id=battle.id,
                side=side,
                condition="epicure_on",
                model_id=model_id,
                execution_backend="bedrock",
                provider_slug="bedrock-test-route",
                status="queued",
                prompt_sha256=battle.prompt_sha256,
                schema_sha256=_sha(f"shared-schema:{suffix}"),
                tool_schema_sha256=_sha(f"shared-tools:{suffix}"),
                epicure_release_id=season.epicure_release_id,
                epicure_bundle_sha256=_sha(f"shared-bundle:{suffix}"),
            )
            for side in ("left", "right")
        ]
    )
    session.add_all(
        [
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                kind="reserve",
                amount_micros=amount,
                provider="governor",
            ),
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                kind="provider_reserve",
                amount_micros=amount,
                provider="bedrock",
                accounting_json={"budget_scope": "provider"},
            ),
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                kind="provider_account_reserve",
                amount_micros=amount,
                provider="bedrock",
                accounting_json={
                    "budget_scope": "provider_account",
                    "account_scope_sha256": account.account_scope_sha256,
                },
            ),
        ]
    )
    account.budget_reserved_micros += amount
    session.flush()
    return season, provider, [battle]


def test_active_reservations_match_all_ledger_views_and_preserve_opening_balance() -> None:
    init_database()
    with session_scope() as session:
        season, _, account, _, _ = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
        )
        report = assert_budget_integrity(session, season.id)
        assert report.ok
        assert report.season_reserved_micros == 200
        assert next(iter(report.account_reserved_micros.values())) == 225
        assert account.opening_reserved_micros == 25


def test_lowered_aggregate_beneath_two_reservations_fails_closed() -> None:
    init_database()
    with session_scope() as session:
        season, _, _, _, _ = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
        )
        season.budget_reserved_micros = 100
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any("season counter 100" in item for item in failure.value.report.violations)


def test_missing_release_and_controlled_run_counter_mismatch_are_detected() -> None:
    init_database()
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
            amounts=(100,),
            controlled=True,
        )
        assert run is not None
        battle = battles[0]
        battle.reserved_cost_micros = 0
        season.budget_reserved_micros = 0
        provider.budget_reserved_micros = 0
        account.budget_reserved_micros = account.opening_reserved_micros
        # Deliberately retain the run counter and omit every release event.
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        message = " | ".join(failure.value.report.violations)
        assert "governor release" in message
        assert "controlled run" in message


def test_fully_settled_reservations_have_zero_net_without_rewriting_contracts() -> None:
    init_database()
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
            controlled=True,
        )
        frozen_contracts = [dict(battle.provider_reservations_json) for battle in battles]
        _settle(session, season, provider, account, battles, run)
        report = assert_budget_integrity(session, season.id)
        assert report.ok
        assert report.battle_reserved_micros == 0
        assert report.governor_event_net_micros == 0
        assert report.provider_event_net_micros == {"openrouter": 0}
        assert [battle.provider_reservations_json for battle in battles] == frozen_contracts


def test_settled_budget_requires_endpoint_generation_receipts() -> None:
    init_database()
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
            controlled=True,
        )
        _settle(
            session,
            season,
            provider,
            account,
            battles,
            run,
            include_receipts=False,
        )
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "lacks an endpoint-generation cost receipt" in item
            for item in failure.value.report.violations
        )


def test_nonzero_terminal_arm_cannot_hide_behind_an_empty_contract() -> None:
    init_database()
    with session_scope() as session:
        season, _, _, battles, _ = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
            amounts=(100,),
        )
        battle = battles[0]
        season.budget_reserved_micros = 0
        arm = session.scalar(
            select(ResponseArm)
            .where(ResponseArm.battle_id == battle.id)
            .order_by(ResponseArm.side)
            .limit(1)
        )
        assert arm is not None
        arm.status = "failed"
        arm.completed_at = datetime.now(UTC)
        arm.cost_micros = 50
        arm.cost_reconciled = True
        arm.cost_accounting_basis = "test_endpoint_receipt"
        arm.billing_reconciliation_status = "complete"
        arm.actual_provider_slug = arm.provider_slug
        arm.generation_id = f"zero-contract-{arm.id}"
        session.add(
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                arm_id=arm.id,
                kind="actual",
                amount_micros=50,
                provider=arm.provider_slug,
                generation_id=arm.generation_id,
                accounting_json={"reconciled": True},
            )
        )
        governed_ids = session.scalars(
            select(CostEvent.id).where(
                CostEvent.battle_id == battle.id,
                CostEvent.kind.in_(
                    ["reserve", "provider_reserve", "provider_account_reserve"]
                ),
            )
        ).all()
        session.query(CostEvent).filter(CostEvent.id.in_(governed_ids)).delete(
            synchronize_session=False
        )
        session.execute(
            update(Battle)
            .where(Battle.id == battle.id)
            .values(provider_reservations_json={}, reserved_cost_micros=0)
        )
        session.expire(battle)
        session.flush()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "without a frozen provider reservation contract" in item
            for item in failure.value.report.violations
        )


def test_reservation_underflow_is_never_clamped() -> None:
    with pytest.raises(ValueError, match="underflow"):
        decrement_reservation(50, 51, label="test budget")
    assert decrement_reservation(50, 50, label="test budget") == 0


def test_used_spend_counter_cannot_diverge_from_governed_events() -> None:
    init_database()
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
        )
        _settle(session, season, provider, account, battles, run)
        season.budget_used_micros += 1
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any("season used counter" in item for item in failure.value.report.violations)


def test_reconcile_events_cannot_disagree_with_response_arm_costs() -> None:
    init_database()
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=uuid.uuid4().hex,
            amounts=(100,),
        )
        _settle(
            session,
            season,
            provider,
            account,
            battles,
            run,
            reconcile_offset=1,
        )
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "does not match response-arm cost" in item for item in failure.value.report.violations
        )


def test_active_account_authorization_is_a_nonduplicating_checkpoint() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, _, account, _, _ = _seed_reservations(
            session,
            suffix=suffix,
        )
        prior = session.scalar(
            select(ProviderAccountAuthorization).where(
                ProviderAccountAuthorization.provider_account_budget_id == account.id,
                ProviderAccountAuthorization.status == "active",
            )
        )
        assert prior is not None
        prior.status = "revoked"
        prior.revoked_at = datetime.now(UTC)
        session.flush()
        # Historical reconciliation remains auditable after credential revocation.
        assert assert_budget_integrity(session, season.id).ok
        successor_valid_until = datetime.now(UTC) + timedelta(days=30)
        successor_material = _authorization_material(
            account,
            suffix=f"successor:{suffix}",
            used_micros=account.budget_used_micros,
            reserved_micros=account.budget_reserved_micros,
            valid_until=successor_valid_until,
            supersedes_envelope_sha256=prior.authorization_envelope_sha256,
        )
        successor = ProviderAccountAuthorization(
            provider_account_budget_id=account.id,
            execution_backend=account.execution_backend,
            account_scope_sha256=account.account_scope_sha256,
            status="active",
            supersedes_authorization_id=prior.id,
            authorization_reference_sha256=str(
                successor_material["authorization_reference_sha256"]
            ),
            exposure_attestation_json=dict(
                successor_material["exposure_attestation_json"]
            ),
            exposure_attestation_sha256=str(
                successor_material["exposure_attestation_sha256"]
            ),
            authorized_used_micros=account.budget_used_micros,
            authorized_reserved_micros=account.budget_reserved_micros,
            credential_binding_json=dict(successor_material["credential_binding_json"]),
            credential_binding_sha256=str(
                successor_material["credential_binding_sha256"]
            ),
            authorization_envelope_json=dict(
                successor_material["authorization_envelope_json"]
            ),
            authorization_envelope_sha256=str(
                successor_material["authorization_envelope_sha256"]
            ),
            authorization_hmac_sha256=str(
                successor_material["authorization_hmac_sha256"]
            ),
            valid_until=successor_valid_until,
        )
        session.add(successor)
        session.flush()
        report = assert_budget_integrity(session, season.id)
        assert report.ok
        assert next(iter(report.account_event_net_micros.values())) == 225


def test_bedrock_adjustment_must_resolve_to_immutable_crosscheck() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)

        _, event = _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=battles,
            suffix=suffix,
        )
        assert assert_budget_integrity(session, season.id).ok

        session.execute(
            update(CostEvent)
            .where(CostEvent.id == event.id)
            .values(
                accounting_json={
                    **event.accounting_json,
                    "evidence_sha256": "0" * 64,
                }
            )
        )
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "disagrees with immutable crosscheck" in item
            for item in failure.value.report.violations
        )


def test_bedrock_adjustment_cannot_hide_behind_a_forged_account_scope() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)
        _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=battles,
            suffix=suffix,
            event_scope_sha256="f" * 64,
        )
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "but crosscheck" in item and account.account_scope_sha256 in item
            for item in failure.value.report.violations
        )


def test_crosscheck_semantics_are_reconstructed_instead_of_trusting_its_hash() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)
        crosscheck, event = _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=battles,
            suffix=suffix,
        )
        forged_evidence = {
            **crosscheck.evidence_json,
            "billing_difference_micros": 11,
            "ledger_delta_micros": 11,
            "governed_budget_delta_micros": 11,
        }
        forged_hash = _canonical_sha(forged_evidence)
        session.execute(
            update(BedrockBillingCrosscheck)
            .where(BedrockBillingCrosscheck.id == crosscheck.id)
            .values(
                billing_difference_micros=11,
                ledger_delta_micros=11,
                evidence_json=forged_evidence,
                evidence_sha256=forged_hash,
            )
        )
        session.execute(
            update(CostEvent)
            .where(CostEvent.id == event.id)
            .values(
                amount_micros=11,
                accounting_json={
                    **event.accounting_json,
                    "evidence_sha256": forged_hash,
                    "governed_budget_delta_micros": 11,
                },
            )
        )
        season.budget_used_micros += 1
        provider.budget_used_micros += 1
        account.budget_used_micros += 1
        session.flush()
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "cannot be reconstructed from relational cost evidence" in item
            for item in failure.value.report.violations
        )


def test_billing_correction_cannot_subtract_an_unrelated_arm_set() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100, 100),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)
        prior, _ = _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=[battles[0]],
            suffix=f"{suffix}-prior",
        )
        _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=[battles[1]],
            suffix=f"{suffix}-forged-correction",
            ledger_delta_micros=5,
            supersedes_crosscheck=prior,
        )
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "invalid or nonmatching predecessor arm set" in item
            for item in failure.value.report.violations
        )


def test_authorization_checkpoint_requires_a_valid_hmac_chain() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, _, account, _, _ = _seed_reservations(session, suffix=suffix)
        authorization = session.scalar(
            select(ProviderAccountAuthorization).where(
                ProviderAccountAuthorization.provider_account_budget_id == account.id
            )
        )
        assert authorization is not None
        session.execute(
            update(ProviderAccountAuthorization)
            .where(ProviderAccountAuthorization.id == authorization.id)
            .values(authorization_hmac_sha256="0" * 64)
        )
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "authorization chain is not independently verifiable" in item
            for item in failure.value.report.violations
        )


def test_v4_authorization_remains_verifiable_after_signing_key_rotation() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        _, _, account, _, _ = _seed_reservations(session, suffix=suffix)
        authorization = session.scalar(
            select(ProviderAccountAuthorization).where(
                ProviderAccountAuthorization.provider_account_budget_id == account.id
            )
        )
        assert authorization is not None
        legacy_secret = "legacy-budget-signing-secret-with-32-bytes"
        current_secret = "current-budget-signing-secret-with-32-bytes"
        envelope = dict(authorization.authorization_envelope_json)
        envelope["schema_version"] = "flavourbench-provider-account-authorization-v4"
        envelope["signing_key_id"] = "legacy-2026-01"
        envelope_sha256 = _canonical_sha(envelope)
        signature = hmac.new(
            legacy_secret.encode(),
            envelope_sha256.encode(),
            hashlib.sha256,
        ).hexdigest()
        session.execute(
            update(ProviderAccountAuthorization)
            .where(ProviderAccountAuthorization.id == authorization.id)
            .values(
                authorization_envelope_json=envelope,
                authorization_envelope_sha256=envelope_sha256,
                authorization_hmac_sha256=signature,
            )
        )
        session.expire_all()
        authorization = session.get(ProviderAccountAuthorization, authorization.id)
        assert account_authorization_chain_valid(
            session,
            account,
            authorization,
            root_envelope_sha256=envelope_sha256,
            signing_secret=current_secret,
            verification_keys={
                "legacy-2026-01": legacy_secret,
                "current-2026-07": current_secret,
            },
        )
        assert not account_authorization_chain_valid(
            session,
            account,
            authorization,
            root_envelope_sha256=envelope_sha256,
            signing_secret=current_secret,
            verification_keys={"current-2026-07": current_secret},
        )


def test_untouched_v3_authorization_survives_rotation_in_full_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, _, _, _, _ = _seed_reservations(session, suffix=suffix)
        legacy_secret = get_settings().budget_authorization_signing_secret
        current_secret = "current-budget-signing-secret-with-32-bytes"
        rotated = SimpleNamespace(
            budget_authorization_signing_secret=current_secret,
            budget_authorization_signing_key_id="current-2026-07",
            budget_authorization_verification_keys={
                "legacy-v3": legacy_secret,
                "current-2026-07": current_secret,
            },
        )
        monkeypatch.setattr(
            "flavourbench.budget_integrity.get_settings",
            lambda: rotated,
        )
        assert assert_budget_integrity(session, season.id).ok


def test_crosscheck_rejects_attempt_outside_credential_epoch() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)
        _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=battles,
            suffix=suffix,
        )
        authorization = session.scalar(
            select(ProviderAccountAuthorization).where(
                ProviderAccountAuthorization.provider_account_budget_id == account.id
            )
        )
        assert authorization is not None
        arm_ids = session.scalars(
            select(ResponseArm.id).where(ResponseArm.battle_id == battles[0].id)
        ).all()
        session.execute(
            update(GenerationAttempt)
            .where(
                GenerationAttempt.arm_id.in_(arm_ids),
                GenerationAttempt.event_type == "request_started",
            )
            .values(created_at=authorization.created_at - timedelta(seconds=1))
        )
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "credential epoch is not independently verifiable" in item
            for item in failure.value.report.violations
        )


def test_crosscheck_rejects_coverage_after_arm_completion() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season, provider, account, battles, run)
        crosscheck, event = _add_valid_bedrock_crosscheck(
            session,
            season=season,
            provider=provider,
            account=account,
            battles=battles,
            suffix=suffix,
        )
        coverage_start = datetime.now(UTC) + timedelta(days=1)
        coverage_end = coverage_start + timedelta(hours=1)
        evidence = dict(crosscheck.evidence_json)
        evidence["coverage_start"] = coverage_start.isoformat()
        evidence["coverage_end"] = coverage_end.isoformat()
        evidence_sha256 = _canonical_sha(evidence)
        event_accounting = dict(event.accounting_json)
        event_accounting["evidence_sha256"] = evidence_sha256
        session.execute(
            update(BedrockBillingCrosscheck)
            .where(BedrockBillingCrosscheck.id == crosscheck.id)
            .values(
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                evidence_json=evidence,
                evidence_sha256=evidence_sha256,
            )
        )
        session.execute(
            update(CostEvent)
            .where(CostEvent.id == event.id)
            .values(accounting_json=event_accounting)
        )
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        assert any(
            "cannot be reconstructed from relational cost evidence" in item
            for item in failure.value.report.violations
        )


def test_manual_settlement_receipt_requires_exact_authorization_and_prior_exposure() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season, provider, account, battles, run = _seed_reservations(
            session,
            suffix=suffix,
            amounts=(100,),
            controlled=True,
        )
        assert run is not None
        _settle(session, season, provider, account, battles, run)
        arms = list(
            session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battles[0].id)
            ).all()
        )
        for arm in arms:
            arm.cost_accounting_basis = "manual_authorized_settlement"
            arm.billing_reconciliation_status = "manual_authorized_settlement"
            receipt = session.scalar(
                select(CostEvent).where(
                    CostEvent.arm_id == arm.id,
                    CostEvent.kind == "actual",
                )
            )
            assert receipt is not None
            session.execute(
                update(CostEvent)
                .where(CostEvent.id == receipt.id)
                .values(
                    kind="actual_settlement",
                    accounting_json={
                        "settlement": "manual_authorized",
                        "authorization_reference_sha256": "f" * 64,
                        "supersedes_cost_event_id": None,
                        "prior_cost_state": "unresolved_attempt_journal",
                    },
                )
            )
        session.expire_all()
        assert session.scalar(
            select(RunEvent.id).where(
                RunEvent.entity_id == battles[0].id,
                RunEvent.event_type == "generation_cost_exposure_settled",
            )
        ) is None
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season.id)
        message = " | ".join(failure.value.report.violations)
        assert "authorization event" in message
        assert "prior unresolved provider-cost evidence" in message


def test_shared_account_audit_accepts_valid_crosschecks_from_other_seasons() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season_a, provider_a, account, battles_a, run_a = _seed_reservations(
            session,
            suffix=f"{suffix}-a",
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season_a, provider_a, account, battles_a, run_a)
        _add_valid_bedrock_crosscheck(
            session,
            season=season_a,
            provider=provider_a,
            account=account,
            battles=battles_a,
            suffix=f"{suffix}-a",
        )
        model_id = session.scalar(
            select(ResponseArm.model_id).where(
                ResponseArm.battle_id == battles_a[0].id
            )
        )
        assert model_id is not None
        account_root_sha256 = provider_a.account_authorization_envelope_sha256
        prior_used = account.budget_used_micros
        season_b, provider_b, battles_b = _seed_shared_account_season(
            session,
            account=account,
            account_root_sha256=account_root_sha256,
            model_id=model_id,
            suffix=f"{suffix}-b",
        )
        _settle(
            session,
            season_b,
            provider_b,
            account,
            battles_b,
            None,
            account_used_offset=prior_used,
        )
        _add_valid_bedrock_crosscheck(
            session,
            season=season_b,
            provider=provider_b,
            account=account,
            battles=battles_b,
            suffix=f"{suffix}-b",
        )
        assert assert_budget_integrity(session, season_a.id).ok
        assert assert_budget_integrity(session, season_b.id).ok


def test_shared_account_audit_deeply_validates_foreign_season_crosschecks() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    with session_scope() as session:
        season_a, provider_a, account, battles_a, run_a = _seed_reservations(
            session,
            suffix=f"{suffix}-a",
            amounts=(100,),
            backend="bedrock",
        )
        _settle(session, season_a, provider_a, account, battles_a, run_a)
        _add_valid_bedrock_crosscheck(
            session,
            season=season_a,
            provider=provider_a,
            account=account,
            battles=battles_a,
            suffix=f"{suffix}-a",
        )
        model_id = session.scalar(
            select(ResponseArm.model_id).where(
                ResponseArm.battle_id == battles_a[0].id
            )
        )
        assert model_id is not None
        prior_used = account.budget_used_micros
        season_b, provider_b, battles_b = _seed_shared_account_season(
            session,
            account=account,
            account_root_sha256=provider_a.account_authorization_envelope_sha256,
            model_id=model_id,
            suffix=f"{suffix}-b",
        )
        _settle(
            session,
            season_b,
            provider_b,
            account,
            battles_b,
            None,
            account_used_offset=prior_used,
        )
        crosscheck_b, event_b = _add_valid_bedrock_crosscheck(
            session,
            season=season_b,
            provider=provider_b,
            account=account,
            battles=battles_b,
            suffix=f"{suffix}-b",
        )
        forged_evidence = dict(crosscheck_b.evidence_json)
        forged_evidence["rate_card_estimated_micros"] = (
            crosscheck_b.rate_card_estimated_micros + 1
        )
        forged_sha256 = _canonical_sha(forged_evidence)
        forged_accounting = dict(event_b.accounting_json)
        forged_accounting["evidence_sha256"] = forged_sha256
        session.execute(
            update(BedrockBillingCrosscheck)
            .where(BedrockBillingCrosscheck.id == crosscheck_b.id)
            .values(
                rate_card_estimated_micros=(
                    crosscheck_b.rate_card_estimated_micros + 1
                ),
                evidence_json=forged_evidence,
                evidence_sha256=forged_sha256,
            )
        )
        session.execute(
            update(CostEvent)
            .where(CostEvent.id == event_b.id)
            .values(accounting_json=forged_accounting)
        )
        session.expire_all()
        with pytest.raises(BudgetIntegrityError) as failure:
            assert_budget_integrity(session, season_a.id)
        assert any(
            "cannot be reconstructed from relational cost evidence" in item
            for item in failure.value.report.violations
        )
