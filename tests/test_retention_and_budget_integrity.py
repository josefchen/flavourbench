from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from flavourbench.database import init_database, session_scope
from flavourbench.engine import redact_expired
from flavourbench.models import (
    TOOL_CALL_REDACTION_JSON,
    TOOL_CALL_REDACTION_SENTINEL,
    Battle,
    ControlledRun,
    Incident,
    Job,
    ProviderAccountBudget,
    RunEvent,
    Season,
    SeasonProviderBudget,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _battle(
    season_id: str,
    suffix: str,
    *,
    consent: bool,
    retention_until: datetime,
) -> Battle:
    prompt = f"Retention prompt {suffix}"
    return Battle(
        season_id=season_id,
        run_class="exploratory",
        rank_eligible=False,
        data_stratum="public_freeform",
        manifest_sha256="unfrozen",
        protocol_bundle_sha256="unfrozen",
        scheduler_version="retention-integrity-test-v1",
        assignment_seed=_digest(f"seed:{suffix}"),
        track_assignment_probability="1/1",
        model_assignment_probability="1/1",
        side_assignment_probability="1/2",
        track="model_arena",
        category="composition",
        prompt=prompt,
        prompt_sha256=_digest(prompt),
        client_nonce_sha256=_digest(f"nonce:{suffix}"),
        research_consent=consent,
        release_review_status="pending" if consent else "not_requested",
        requester_pseudonym=_digest(f"rater:{suffix}"),
        status="queued",
        reserved_cost_micros=0,
        retention_until=retention_until,
    )


def test_retention_requires_expiry_and_non_consent_and_scopes_events() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    now = datetime.now(UTC)
    with session_scope() as session:
        season = Season(
            slug=f"retention-{suffix}",
            name="Retention authorization test",
            epicure_release_id="retention-test-release",
        )
        session.add(season)
        session.flush()
        expired = _battle(
            season.id,
            f"expired-{suffix}",
            consent=False,
            retention_until=now - timedelta(seconds=1),
        )
        future = _battle(
            season.id,
            f"future-{suffix}",
            consent=False,
            retention_until=now + timedelta(days=30),
        )
        consented = _battle(
            season.id,
            f"consented-{suffix}",
            consent=True,
            retention_until=now - timedelta(seconds=1),
        )
        session.add_all([expired, future, consented])
        session.flush()
        expired_id, future_id, consented_id = expired.id, future.id, consented.id
        job = Job(
            battle_id=expired_id,
            kind="generate_battle",
            payload_json={},
            status="queued",
            last_error="sensitive job error",
        )
        session.add_all(
            [
                RunEvent(
                    entity_type="battle",
                    entity_id=expired_id,
                    event_type="sensitive_event",
                    payload_json={"prompt_fragment": "sensitive"},
                ),
                RunEvent(
                    entity_type="catalog",
                    entity_id=expired_id,
                    event_type="collision_control",
                    payload_json={"keep": True},
                ),
                Incident(
                    severity="info",
                    code="RetentionTest",
                    detail="sensitive incident detail",
                    battle_id=expired_id,
                ),
                job,
            ]
        )
        session.flush()
        job.status = "failed"
        job.completed_at = datetime.now(UTC)

    for battle_id in (future_id, consented_id):
        with pytest.raises(ValueError, match="one-way retention redaction"):
            with session_scope() as session:
                battle = session.get(Battle, battle_id)
                assert battle is not None
                battle.prompt = None
                battle.prompt_redacted = True

    with session_scope() as session:
        assert redact_expired(session) == 1
        session.flush()
        expired = session.get(Battle, expired_id)
        future = session.get(Battle, future_id)
        consented = session.get(Battle, consented_id)
        assert expired is not None and expired.prompt is None and expired.prompt_redacted
        assert future is not None and future.prompt is not None and not future.prompt_redacted
        assert consented is not None and consented.prompt is not None
        battle_event = session.query(RunEvent).filter_by(
            entity_type="battle",
            entity_id=expired_id,
            event_type="sensitive_event",
        ).one()
        collision = session.query(RunEvent).filter_by(
            entity_type="catalog",
            entity_id=expired_id,
        ).one()
        incident = session.query(Incident).filter_by(battle_id=expired_id).one()
        job = session.query(Job).filter_by(battle_id=expired_id).one()
        assert battle_event.payload_json == TOOL_CALL_REDACTION_JSON
        assert collision.payload_json == {"keep": True}
        assert incident.detail == TOOL_CALL_REDACTION_SENTINEL
        assert job.last_error == TOOL_CALL_REDACTION_SENTINEL


def test_governed_used_spend_is_monotonic_in_the_orm() -> None:
    init_database()
    suffix = uuid.uuid4().hex
    valid_until = datetime.now(UTC) + timedelta(days=30)
    with session_scope() as session:
        season = Season(
            slug=f"monotonic-{suffix}",
            name="Monotonic budget test",
            epicure_release_id="budget-test-release",
            budget_cap_micros=1_000,
            budget_used_micros=20,
        )
        session.add(season)
        session.flush()
        provider = SeasonProviderBudget(
            season_id=season.id,
            execution_backend="bedrock",
            currency="USD",
            budget_cap_micros=1_000,
            budget_used_micros=20,
            account_scope_sha256=_digest(f"scope:{suffix}"),
            authorization_reference_sha256=_digest(f"authorization:{suffix}"),
            account_authorization_envelope_sha256=_digest(f"root:{suffix}"),
            authorization_envelope_json={},
            authorization_envelope_sha256=_digest(f"envelope:{suffix}"),
            valid_until=valid_until,
        )
        account = ProviderAccountBudget(
            execution_backend="bedrock",
            currency="USD",
            status="active",
            budget_cap_micros=1_000,
            budget_used_micros=20,
            budget_reserved_micros=0,
            opening_used_micros=0,
            opening_reserved_micros=0,
            account_scope_sha256=_digest(f"account-scope:{suffix}"),
            authorization_reference_sha256=_digest(f"account-auth:{suffix}"),
            opening_balance_json={},
            opening_balance_sha256=_digest(f"opening:{suffix}"),
            credential_binding_json={},
            credential_binding_sha256=_digest(f"binding:{suffix}"),
            authorization_envelope_json={},
            authorization_envelope_sha256=_digest(f"account-envelope:{suffix}"),
            authorization_hmac_sha256=_digest(f"hmac:{suffix}"),
            valid_until=valid_until,
        )
        run = ControlledRun(
            season_id=season.id,
            organization_reference_sha256=_digest(f"organization:{suffix}"),
            access_token_sha256=_digest(f"access:{suffix}"),
            status="active",
            protocol_version="budget-test-v1",
            rater_plan_sha256=_digest(f"raters:{suffix}"),
            analysis_plan_sha256=_digest(f"analysis:{suffix}"),
            budget_cap_micros=1_000,
            budget_used_micros=20,
            run_card_json={},
            run_card_sha256=_digest(f"run-card:{suffix}"),
            run_card_signature=_digest(f"signature:{suffix}"),
        )
        session.add_all([provider, account, run])
        session.flush()
        identities = (
            (Season, season.id),
            (SeasonProviderBudget, provider.id),
            (ProviderAccountBudget, account.id),
            (ControlledRun, run.id),
        )

    for model, row_id in identities:
        with pytest.raises(ValueError, match="cannot move backward"):
            with session_scope() as session:
                row = session.get(model, row_id)
                assert row is not None
                row.budget_used_micros = 19
